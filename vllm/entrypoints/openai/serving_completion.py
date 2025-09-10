# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import time
from abc import ABC
from collections.abc import AsyncGenerator, AsyncIterator
from collections.abc import Sequence as GenericSequence
from typing import Optional, Union, cast
from http import HTTPStatus

import jinja2
from fastapi import Request
from typing_extensions import assert_never

from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.logger import RequestLogger
# yapf conflicts with isort for this block
# yapf: disable
from vllm.entrypoints.openai.protocol import (CompletionLogProbs,
                                              CompletionRequest,
                                              CompletionResponse,
                                              CompletionResponseChoice,
                                              CompletionResponseStreamChoice,
                                              CompletionStreamResponse,
                                              ErrorResponse,
                                              PromptTokenUsageInfo,
                                              RequestResponseMetadata,
                                              UsageInfo,
                                              VerifiedCompletionRequest,
                                              VerifiedCompletionResponse,
                                              VerifiedCompletionResponseChoice,
                                              VerifiedTokenDetail,
                                              VerifyDecodingRequest,
                                              VerifyDecodingResponse,
                                              TokenVerificationDetail)
# yapf: enable
from vllm.entrypoints.openai.serving_engine import (OpenAIServing,
                                                    clamp_prompt_logprobs)
from vllm.entrypoints.openai.verification_mixin import VerificationMixin
from vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.entrypoints.utils import get_max_tokens
from vllm.inputs.data import (EmbedsPrompt, TokensPrompt, is_embeds_prompt,
                              is_tokens_prompt)
from vllm.logger import init_logger
from vllm.logprobs import Logprob
from vllm.outputs import RequestOutput
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.utils import as_list, merge_async_iterators

logger = init_logger(__name__)


class OpenAIServingCompletion(OpenAIServing, VerificationMixin):

    def __init__(
        self,
        engine_client: EngineClient,
        model_config: ModelConfig,
        models: OpenAIServingModels,
        *,
        request_logger: Optional[RequestLogger],
        return_tokens_as_token_ids: bool = False,
        enable_prompt_tokens_details: bool = False,
        enable_force_include_usage: bool = False,
        log_error_stack: bool = False,
    ):
        super().__init__(
            engine_client=engine_client,
            model_config=model_config,
            models=models,
            request_logger=request_logger,
            return_tokens_as_token_ids=return_tokens_as_token_ids,
            enable_force_include_usage=enable_force_include_usage,
            log_error_stack=log_error_stack,
        )
        self.enable_prompt_tokens_details = enable_prompt_tokens_details
        self.default_sampling_params = (
            self.model_config.get_diff_sampling_param())
        if self.default_sampling_params:
            source = self.model_config.generation_config
            source = "model" if source == "auto" else source
            logger.info(
                "Using default completion sampling params from %s: %s",
                source,
                self.default_sampling_params,
            )

    async def create_completion(
        self,
        request: CompletionRequest,
        raw_request: Optional[Request] = None,
    ) -> Union[AsyncGenerator[str, None], CompletionResponse, ErrorResponse]:
        """Completion API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/completions/create
        for the API specification. This API mimics the OpenAI Completion API.

        NOTE: Currently we do not support the following feature:
            - suffix (the language models we currently support do not support
            suffix)
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        # If the engine is dead, raise the engine's DEAD_ERROR.
        # This is required for the streaming case, where we return a
        # success status before we actually start generating text :).
        if self.engine_client.errored:
            raise self.engine_client.dead_error

        # Return error for unsupported features.
        if request.suffix is not None:
            return self.create_error_response(
                "suffix is not currently supported")

        if request.echo and request.prompt_embeds is not None:
            return self.create_error_response(
                "Echo is unsupported with prompt embeds.")

        request_id = (
            f"cmpl-"
            f"{self._base_request_id(raw_request, request.request_id)}")
        created_time = int(time.time())

        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        try:
            lora_request = self._maybe_get_adapters(request)

            if self.model_config.skip_tokenizer_init:
                tokenizer = None
            else:
                tokenizer = await self.engine_client.get_tokenizer(lora_request
                                                                   )
            renderer = self._get_renderer(tokenizer)
            max_input_tokens_len = self.max_model_len - (request.max_tokens
                                                         or 0)

            engine_prompts = await renderer.render_prompt_and_embeds(
                prompt_or_prompts=request.prompt,
                prompt_embeds=request.prompt_embeds,
                max_length=max_input_tokens_len,
                truncate_prompt_tokens=request.truncate_prompt_tokens,
                add_special_tokens=request.add_special_tokens,
                cache_salt=request.cache_salt,
                needs_detokenization=bool(request.echo
                                          and not request.return_token_ids),
            )
        except ValueError as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(str(e))
        except TypeError as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(str(e))
        except RuntimeError as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(str(e))
        except jinja2.TemplateError as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(str(e))

        # Schedule the request and get the result generator.
        generators: list[AsyncGenerator[RequestOutput, None]] = []
        try:
            for i, engine_prompt in enumerate(engine_prompts):
                sampling_params: Union[SamplingParams, BeamSearchParams]
                # Mypy does not infer that engine_prompt will have only one of
                # "prompt_token_ids" or "prompt_embeds" defined, and both of
                # these as Union[object, the expected type], where it infers
                # object if engine_prompt is a subclass of one of the
                # typeddicts that defines both keys. Worse, because of
                # https://github.com/python/mypy/issues/8586, mypy does not
                # infer the type of engine_prompt correctly because of the
                # enumerate. So we need an unnecessary cast here.
                engine_prompt = cast(Union[EmbedsPrompt, TokensPrompt],
                                     engine_prompt)
                if is_embeds_prompt(engine_prompt):
                    input_length = len(engine_prompt["prompt_embeds"])
                elif is_tokens_prompt(engine_prompt):
                    input_length = len(engine_prompt["prompt_token_ids"])
                else:
                    assert_never(engine_prompt)

                if self.default_sampling_params is None:
                    self.default_sampling_params = {}

                max_tokens = get_max_tokens(
                    max_model_len=self.max_model_len,
                    request=request,
                    input_length=input_length,
                    default_sampling_params=self.default_sampling_params,
                )

                if request.use_beam_search:
                    sampling_params = request.to_beam_search_params(
                        max_tokens, self.default_sampling_params)
                else:
                    sampling_params = request.to_sampling_params(
                        max_tokens,
                        self.model_config.logits_processor_pattern,
                        self.default_sampling_params,
                    )

                request_id_item = f"{request_id}-{i}"

                self._log_inputs(
                    request_id_item,
                    engine_prompt,
                    params=sampling_params,
                    lora_request=lora_request,
                )

                trace_headers = (None if raw_request is None else await
                                 self._get_trace_headers(raw_request.headers))

                # Mypy inconsistently requires this second cast in different
                # environments. It shouldn't be necessary (redundant from above)
                # but pre-commit in CI fails without it.
                engine_prompt = cast(Union[EmbedsPrompt, TokensPrompt],
                                     engine_prompt)
                if isinstance(sampling_params, BeamSearchParams):
                    generator = self.engine_client.beam_search(
                        prompt=engine_prompt,
                        request_id=request_id,
                        params=sampling_params,
                        lora_request=lora_request,
                    )
                else:
                    generator = self.engine_client.generate(
                        engine_prompt,
                        sampling_params,
                        request_id_item,
                        lora_request=lora_request,
                        trace_headers=trace_headers,
                        priority=request.priority,
                    )

                generators.append(generator)
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        result_generator = merge_async_iterators(*generators)

        model_name = self._get_model_name(request.model, lora_request)
        num_prompts = len(engine_prompts)

        # Similar to the OpenAI API, when n != best_of, we do not stream the
        # results. Noting that best_of is only supported in V0. In addition,
        # we do not stream the results when use beam search.
        stream = (request.stream
                  and (request.best_of is None or request.n == request.best_of)
                  and not request.use_beam_search)

        # Streaming response
        if stream:
            return self.completion_stream_generator(
                request,
                engine_prompts,
                result_generator,
                request_id,
                created_time,
                model_name,
                num_prompts=num_prompts,
                tokenizer=tokenizer,
                request_metadata=request_metadata,
                enable_force_include_usage=self.enable_force_include_usage,
            )

        # Non-streaming response
        final_res_batch: list[Optional[RequestOutput]] = [None] * num_prompts
        try:
            async for i, res in result_generator:
                final_res_batch[i] = res

            for i, final_res in enumerate(final_res_batch):
                assert final_res is not None

                # The output should contain the input text
                # We did not pass it into vLLM engine to avoid being redundant
                # with the inputs token IDs
                if final_res.prompt is None:
                    engine_prompt = engine_prompts[i]
                    final_res.prompt = None if is_embeds_prompt(
                        engine_prompt) else engine_prompt.get("prompt")

            final_res_batch_checked = cast(list[RequestOutput],
                                           final_res_batch)

            response = self.request_output_to_completion_response(
                final_res_batch_checked,
                request,
                request_id,
                created_time,
                model_name,
                tokenizer,
                request_metadata,
            )
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        # When user requests streaming but we don't stream, we still need to
        # return a streaming response with a single event.
        if request.stream:
            response_json = response.model_dump_json()

            async def fake_stream_generator() -> AsyncGenerator[str, None]:
                yield f"data: {response_json}\n\n"
                yield "data: [DONE]\n\n"

            return fake_stream_generator()

        return response

    async def completion_stream_generator(
        self,
        request: CompletionRequest,
        engine_prompts: list[Union[TokensPrompt, EmbedsPrompt]],
        result_generator: AsyncIterator[tuple[int, RequestOutput]],
        request_id: str,
        created_time: int,
        model_name: str,
        num_prompts: int,
        tokenizer: AnyTokenizer,
        request_metadata: RequestResponseMetadata,
        enable_force_include_usage: bool,
    ) -> AsyncGenerator[str, None]:
        num_choices = 1 if request.n is None else request.n
        previous_text_lens = [0] * num_choices * num_prompts
        previous_num_tokens = [0] * num_choices * num_prompts
        has_echoed = [False] * num_choices * num_prompts
        num_prompt_tokens = [0] * num_prompts
        num_cached_tokens = None
        first_iteration = True

        stream_options = request.stream_options
        if stream_options:
            include_usage = (stream_options.include_usage
                             or enable_force_include_usage)
            include_continuous_usage = (include_usage and
                                        stream_options.continuous_usage_stats)
        else:
            include_usage, include_continuous_usage = False, False

        try:
            async for prompt_idx, res in result_generator:
                prompt_token_ids = res.prompt_token_ids
                prompt_logprobs = res.prompt_logprobs

                if first_iteration:
                    num_cached_tokens = res.num_cached_tokens
                    first_iteration = False

                prompt_text = res.prompt
                if prompt_text is None:
                    engine_prompt = engine_prompts[prompt_idx]
                    prompt_text = None if is_embeds_prompt(
                        engine_prompt) else engine_prompt.get("prompt")

                # Prompt details are excluded from later streamed outputs
                if prompt_token_ids is not None:
                    num_prompt_tokens[prompt_idx] = len(prompt_token_ids)

                delta_token_ids: GenericSequence[int]
                out_logprobs: Optional[GenericSequence[Optional[dict[
                    int, Logprob]]]]

                for output in res.outputs:
                    i = output.index + prompt_idx * num_choices

                    # Useful when request.return_token_ids is True
                    # Returning prompt token IDs shares the same logic
                    # with the echo implementation.
                    prompt_token_ids_to_return: Optional[list[int]] = None

                    assert request.max_tokens is not None
                    if request.echo and not has_echoed[i]:
                        assert prompt_token_ids is not None
                        if request.return_token_ids:
                            prompt_text = ""
                        assert prompt_text is not None
                        if request.max_tokens == 0:
                            # only return the prompt
                            delta_text = prompt_text
                            delta_token_ids = prompt_token_ids
                            out_logprobs = prompt_logprobs
                        else:
                            # echo the prompt and first token
                            delta_text = prompt_text + output.text
                            delta_token_ids = [
                                *prompt_token_ids,
                                *output.token_ids,
                            ]
                            out_logprobs = [
                                *(prompt_logprobs or []),
                                *(output.logprobs or []),
                            ]
                        prompt_token_ids_to_return = prompt_token_ids
                        has_echoed[i] = True
                    else:
                        # return just the delta
                        delta_text = output.text
                        delta_token_ids = output.token_ids
                        out_logprobs = output.logprobs

                        # has_echoed[i] is reused here to indicate whether
                        # we have already returned the prompt token IDs.
                        if not has_echoed[i]:
                            prompt_token_ids_to_return = prompt_token_ids
                            has_echoed[i] = True

                        if (not delta_text and not delta_token_ids
                                and not previous_num_tokens[i]):
                            # Chunked prefill case, don't return empty chunks
                            continue

                    if request.logprobs is not None:
                        assert out_logprobs is not None, (
                            "Did not output logprobs")
                        logprobs = self._create_completion_logprobs(
                            token_ids=delta_token_ids,
                            top_logprobs=out_logprobs,
                            num_output_top_logprobs=request.logprobs,
                            tokenizer=tokenizer,
                            initial_text_offset=previous_text_lens[i],
                            return_as_token_id=request.
                            return_tokens_as_token_ids,
                        )
                    else:
                        logprobs = None

                    previous_text_lens[i] += len(output.text)
                    previous_num_tokens[i] += len(output.token_ids)
                    finish_reason = output.finish_reason
                    stop_reason = output.stop_reason

                    chunk = CompletionStreamResponse(
                        id=request_id,
                        created=created_time,
                        model=model_name,
                        choices=[
                            CompletionResponseStreamChoice(
                                index=i,
                                text=delta_text,
                                logprobs=logprobs,
                                finish_reason=finish_reason,
                                stop_reason=stop_reason,
                                prompt_token_ids=prompt_token_ids_to_return,
                                token_ids=(as_list(output.token_ids) if
                                           request.return_token_ids else None),
                            )
                        ],
                    )
                    if include_continuous_usage:
                        prompt_tokens = num_prompt_tokens[prompt_idx]
                        completion_tokens = previous_num_tokens[i]
                        chunk.usage = UsageInfo(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=prompt_tokens + completion_tokens,
                        )

                    response_json = chunk.model_dump_json(exclude_unset=False)
                    yield f"data: {response_json}\n\n"

            total_prompt_tokens = sum(num_prompt_tokens)
            total_completion_tokens = sum(previous_num_tokens)
            final_usage_info = UsageInfo(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=total_prompt_tokens + total_completion_tokens,
            )

            if self.enable_prompt_tokens_details and num_cached_tokens:
                final_usage_info.prompt_tokens_details = PromptTokenUsageInfo(
                    cached_tokens=num_cached_tokens)

            if include_usage:
                final_usage_chunk = CompletionStreamResponse(
                    id=request_id,
                    created=created_time,
                    model=model_name,
                    choices=[],
                    usage=final_usage_info,
                )
                final_usage_data = final_usage_chunk.model_dump_json(
                    exclude_unset=False, exclude_none=True)
                yield f"data: {final_usage_data}\n\n"

            # report to FastAPI middleware aggregate usage across all choices
            request_metadata.final_usage_info = final_usage_info

        except Exception as e:
            # TODO: Use a vllm-specific Validation Error
            data = self.create_streaming_error_response(str(e))
            yield f"data: {data}\n\n"
        yield "data: [DONE]\n\n"

    def request_output_to_completion_response(
        self,
        final_res_batch: list[RequestOutput],
        request: CompletionRequest,
        request_id: str,
        created_time: int,
        model_name: str,
        tokenizer: AnyTokenizer,
        request_metadata: RequestResponseMetadata,
    ) -> CompletionResponse:
        choices: list[CompletionResponseChoice] = []
        num_prompt_tokens = 0
        num_generated_tokens = 0
        kv_transfer_params = None
        last_final_res = None
        for final_res in final_res_batch:
            last_final_res = final_res
            prompt_token_ids = final_res.prompt_token_ids
            assert prompt_token_ids is not None
            prompt_logprobs = clamp_prompt_logprobs(final_res.prompt_logprobs)
            prompt_text = final_res.prompt

            token_ids: GenericSequence[int]
            out_logprobs: Optional[GenericSequence[Optional[dict[int,
                                                                 Logprob]]]]

            for output in final_res.outputs:
                assert request.max_tokens is not None
                if request.echo:
                    if request.return_token_ids:
                        prompt_text = ""
                    assert prompt_text is not None
                    if request.max_tokens == 0:
                        token_ids = prompt_token_ids
                        out_logprobs = prompt_logprobs
                        output_text = prompt_text
                    else:
                        token_ids = [*prompt_token_ids, *output.token_ids]

                        if request.logprobs is None:
                            out_logprobs = None
                        else:
                            assert prompt_logprobs is not None
                            assert output.logprobs is not None
                            out_logprobs = [
                                *prompt_logprobs,
                                *output.logprobs,
                            ]

                        output_text = prompt_text + output.text
                else:
                    token_ids = output.token_ids
                    out_logprobs = output.logprobs
                    output_text = output.text

                if request.logprobs is not None:
                    assert out_logprobs is not None, "Did not output logprobs"
                    logprobs = self._create_completion_logprobs(
                        token_ids=token_ids,
                        top_logprobs=out_logprobs,
                        tokenizer=tokenizer,
                        num_output_top_logprobs=request.logprobs,
                        return_as_token_id=request.return_tokens_as_token_ids,
                    )
                else:
                    logprobs = None

                choice_data = CompletionResponseChoice(
                    index=len(choices),
                    text=output_text,
                    logprobs=logprobs,
                    finish_reason=output.finish_reason,
                    stop_reason=output.stop_reason,
                    prompt_logprobs=final_res.prompt_logprobs,
                    prompt_token_ids=(prompt_token_ids
                                      if request.return_token_ids else None),
                    token_ids=(as_list(output.token_ids)
                               if request.return_token_ids else None),
                )
                choices.append(choice_data)

                num_generated_tokens += len(output.token_ids)

            num_prompt_tokens += len(prompt_token_ids)

        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )

        if (self.enable_prompt_tokens_details and last_final_res
                and last_final_res.num_cached_tokens):
            usage.prompt_tokens_details = PromptTokenUsageInfo(
                cached_tokens=last_final_res.num_cached_tokens)

        request_metadata.final_usage_info = usage
        if final_res_batch:
            kv_transfer_params = final_res_batch[0].kv_transfer_params
        return CompletionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage,
            kv_transfer_params=kv_transfer_params,
        )

    def _create_completion_logprobs(
        self,
        token_ids: GenericSequence[int],
        top_logprobs: GenericSequence[Optional[dict[int, Logprob]]],
        num_output_top_logprobs: int,
        tokenizer: AnyTokenizer,
        initial_text_offset: int = 0,
        return_as_token_id: Optional[bool] = None,
    ) -> CompletionLogProbs:
        """Create logprobs for OpenAI Completion API."""
        out_text_offset: list[int] = []
        out_token_logprobs: list[Optional[float]] = []
        out_tokens: list[str] = []
        out_top_logprobs: list[Optional[dict[str, float]]] = []

        last_token_len = 0

        should_return_as_token_id = (return_as_token_id
                                     if return_as_token_id is not None else
                                     self.return_tokens_as_token_ids)
        for i, token_id in enumerate(token_ids):
            step_top_logprobs = top_logprobs[i]
            if step_top_logprobs is None:
                token = tokenizer.decode(token_id)
                if should_return_as_token_id:
                    token = f"token_id:{token_id}"

                out_tokens.append(token)
                out_token_logprobs.append(None)
                out_top_logprobs.append(None)
            else:
                step_token = step_top_logprobs[token_id]

                token = self._get_decoded_token(
                    step_token,
                    token_id,
                    tokenizer,
                    return_as_token_id=should_return_as_token_id,
                )
                token_logprob = max(step_token.logprob, -9999.0)

                out_tokens.append(token)
                out_token_logprobs.append(token_logprob)

                # makes sure to add the top num_output_top_logprobs + 1
                # logprobs, as defined in the openai API
                # (cf. https://github.com/openai/openai-openapi/blob/
                # 893ba52242dbd5387a97b96444ee1c742cfce9bd/openapi.yaml#L7153)
                out_top_logprobs.append({
                    # Convert float("-inf") to the
                    # JSON-serializable float that OpenAI uses
                    self._get_decoded_token(
                        top_lp[1],
                        top_lp[0],
                        tokenizer,
                        return_as_token_id=should_return_as_token_id,
                    ):
                    max(top_lp[1].logprob, -9999.0)
                    for i, top_lp in enumerate(step_top_logprobs.items())
                    if num_output_top_logprobs >= i
                })

            if len(out_text_offset) == 0:
                out_text_offset.append(initial_text_offset)
            else:
                out_text_offset.append(out_text_offset[-1] + last_token_len)
            last_token_len = len(token)

        return CompletionLogProbs(
            text_offset=out_text_offset,
            token_logprobs=out_token_logprobs,
            tokens=out_tokens,
            top_logprobs=out_top_logprobs,
        )

    def _get_decoded_token(
        self,
        logprob: Logprob,
        token_id: int,
        tokenizer: AnyTokenizer,
        return_as_token_id: Optional[bool] = None,
    ) -> str:
        should_return_as_token_id = return_as_token_id if \
            return_as_token_id is not None else self.return_tokens_as_token_ids
        if should_return_as_token_id:
            return f"token_id:{token_id}"
        # Note: This may not be the true token if the tokenizer is not faithful.
        # But it's the best we can do without returning bytes.
        return logprob.decoded_token

    # === New methods for Verified Completion ===

    # Token detail builder moved to VerificationMixin

    async def create_verified_completion(
        self,
        request: VerifiedCompletionRequest,
        raw_request: Optional[Request] = None,
    ) -> Union[VerifiedCompletionResponse, ErrorResponse]:
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        # Resolve adapters and tokenizer (respect adapters)
        lora_request, prompt_adapter_request = self._maybe_get_adapters(request_copy)
        tokenizer = await self.engine_client.get_tokenizer(lora_request)

        request_id = f"vcmpl-{self._base_request_id(raw_request)}"
        created_time = int(time.time())

        request_copy = request.model_copy(deep=True)
        request_copy.temperature = 0.0
        request_copy.n = 1
        if request_copy.logprobs is None or request_copy.logprobs == 0:
            request_copy.logprobs = 1 # Ensure we get at least the sampled token's logprob
        # prompt_logprobs are used if set by the user in the request_copy

        # First, tokenize the prompt to get the number of tokens for max_tokens calculation
        prompt_input_for_max_tokens = self._tokenize_prompt_input(request_copy, tokenizer, request_copy.prompt)
        num_prompt_tokens_for_max_tokens = len(prompt_input_for_max_tokens["prompt_token_ids"])

        if request_copy.max_tokens is None:
            request_copy.max_tokens = self.max_model_len - num_prompt_tokens_for_max_tokens

        # Ensure max_tokens is not negative if prompt is too long
        if request_copy.max_tokens < 0:
            request_copy.max_tokens = 0
            
        try:
            # For verified completion, the prompt_input for _process_model_inputs should be based on request_copy.prompt
            # The prompt_input_for_max_tokens was specifically for length calculation.
            # We re-tokenize here to ensure the prompt_input for the engine is fresh and complete.
            prompt_input = self._tokenize_prompt_input(request_copy, tokenizer, request_copy.prompt)

            # Correctly prepare engine_inputs for OpenAIServingCompletion
            # Mimic parts of _preprocess_completion or standard completion setup
            engine_inputs: TokensPrompt = {"prompt_token_ids": prompt_input["prompt_token_ids"]}
            # If your request_copy or prompt_input could contain multi-modal data:
            # if "multi_modal_data" in prompt_input:
            #    engine_inputs["multi_modal_data"] = prompt_input["multi_modal_data"]
            
            actual_prompt_token_ids = list(prompt_input["prompt_token_ids"]) # Direct assignment
            mm_kwargs = {} # Typically no mm_kwargs for simple completion

            # (engine_inputs, _, _, actual_prompt_token_ids_from_engine, _, mm_kwargs) = \\
            #     await self._process_model_inputs(request_id, prompt_input, request_copy) # This line is removed
            
            # actual_prompt_token_ids = actual_prompt_token_ids_from_engine if actual_prompt_token_ids_from_engine is not None else []

            sampling_params = request_copy.to_sampling_params(
                default_max_tokens=self.max_model_len,
                logits_processor_pattern=self.logits_processor_pattern,
                default_sampling_params=self.default_sampling_params,
            )
            sampling_params.temperature = 0.0
            sampling_params.n = 1
            if sampling_params.logprobs is None or sampling_params.logprobs == 0:
                sampling_params.logprobs = 1 
            # Honor user-provided prompt_logprobs only

            trace_headers = None
            if raw_request is not None:
                trace_headers = await self._get_trace_headers(raw_request.headers)

            result_generator = self.engine_client.generate(
                prompt=engine_inputs,
                sampling_params=sampling_params,
                request_id=request_id,
                lora_request=lora_request,
                prompt_adapter_request=prompt_adapter_request,
                trace_headers=trace_headers,
                priority=request.priority,
                **mm_kwargs
            )
            final_res: Optional[RequestOutput] = None
            try:
                async for res_output in result_generator: # Changed from 'async for _, res_batch_output in result_generator:'
                    if isinstance(res_output, Exception):
                        logger.error(f"Error during generation for verified completion: {res_output}", exc_info=True)
                        return self.create_error_response(f"Engine error: {str(res_output)}", status_code=HTTPStatus.INTERNAL_SERVER_ERROR)
                    final_res = res_output # In non-streaming, the last (and only) output is the one we want
            except Exception as e: # Catch errors during async iteration of result_generator
                logger.error(f"Error consuming result_generator for verified completion: {e}", exc_info=True)
                return self.create_error_response(f"Error during engine generation: {e}", status_code=HTTPStatus.INTERNAL_SERVER_ERROR)

            if final_res is None or not final_res.outputs:
                logger.error("No output received from generation engine for verified completion.")
                return self.create_error_response("No output from engine.", status_code=HTTPStatus.INTERNAL_SERVER_ERROR)

            completion_output = final_res.outputs[0]
            actual_completion_token_ids = list(completion_output.token_ids)

            completion_details = self._build_verified_token_details(
                actual_completion_token_ids,
                completion_output.logprobs, 
                tokenizer
            )

            prompt_details = None
            # final_res.prompt_token_ids should be the source of truth for prompt tokens from engine perspective
            # actual_prompt_token_ids was from _process_model_inputs, should be consistent
            engine_prompt_token_ids = final_res.prompt_token_ids if final_res.prompt_token_ids is not None else []
            if final_res.prompt_logprobs and engine_prompt_token_ids:
                prompt_details = self._build_verified_token_details(
                    engine_prompt_token_ids,
                    clamp_prompt_logprobs(final_res.prompt_logprobs),
                    tokenizer,
                    is_prompt_tokens=True
                )
            elif engine_prompt_token_ids: # If prompt_logprobs were not requested/returned but we have tokens
                 prompt_details = self._build_verified_token_details(
                    engine_prompt_token_ids,
                    None, # Pass None for sample_logprobs
                    tokenizer,
                    is_prompt_tokens=True
                )


            choice = VerifiedCompletionResponseChoice(
                index=0,
                text=completion_output.text,
                prompt_token_ids=engine_prompt_token_ids,
                completion_token_ids=actual_completion_token_ids,
                completion_token_details=completion_details or [],
                prompt_token_details=prompt_details,
                finish_reason=completion_output.finish_reason,
                stop_reason=completion_output.stop_reason
            )

            num_prompt_tokens = len(engine_prompt_token_ids)
            num_completion_tokens = len(actual_completion_token_ids)
            usage = UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=num_completion_tokens,
                total_tokens=num_prompt_tokens + num_completion_tokens,
                prompt_tokens_details=(
                    None if final_res.num_cached_tokens is None else
                    PromptTokenUsageInfo(cached_tokens=final_res.num_cached_tokens)
                ),
            )
            
            response_metadata = RequestResponseMetadata(request_id=request_id, final_usage_info=usage)
            if raw_request and hasattr(raw_request.state, "request_metadata"):
                raw_request.state.request_metadata = response_metadata

            return VerifiedCompletionResponse(
                id=request_id,
                object="text_completion.verified",
                created=created_time,
                model=self._get_model_name(request.model, lora_request=lora_request),
                choices=[choice],
                usage=usage
            )
        except Exception as e:
            logger.error(f"Error processing verified completion request: {e}")
            return self.create_error_response(str(e), status_code=HTTPStatus.INTERNAL_SERVER_ERROR)

    # === Method for Verify Decoding (moved from OpenAIServing) ===
    async def verify_decoding(
        self,
        request: VerifyDecodingRequest,
        raw_request: Optional[Request] = None,
    ) -> Union[VerifyDecodingResponse, ErrorResponse]:
        logger.info("Starting verify_decoding")
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        lora_request = None # Explicitly None, assuming verify_decoding does not use LoRA for now
        # If LoRA support is needed, self._maybe_get_adapters(request) can be used.

        # Use tokenizer respecting adapters
        tokenizer = await self.engine_client.get_tokenizer(lora_request)

        request_id = f"verify-{self._base_request_id(raw_request)}" 
        created_time = int(time.time())
        
        prompt_token_ids: list[int]
        completion_token_ids: list[int]

        try:
            # Standardize tokenization using server logic, do not add special tokens
            if isinstance(request.prompt, str):
                prompt_token_ids = tokenizer.encode(request.prompt)
            elif isinstance(request.prompt, list) and all(isinstance(x, int) for x in request.prompt):
                prompt_token_ids = list(request.prompt)
            else:
                raise ValueError("Invalid prompt format. Must be string or list of token IDs.")

            if isinstance(request.completion, str):
                completion_token_ids = tokenizer.encode(request.completion)
            elif isinstance(request.completion, list) and all(isinstance(x, int) for x in request.completion):
                completion_token_ids = list(request.completion)
            else:
                raise ValueError("Invalid completion format. Must be string or list of token IDs.")
        except ValueError as e: # Catch specific ValueError for format issues
            logger.error(f"Invalid input format for verification: {e}", exc_info=True)
            return self.create_error_response(f"Invalid input format: {str(e)}", status_code=HTTPStatus.BAD_REQUEST)
        except Exception as e: # Catch other tokenization errors
            logger.error(f"Error tokenizing inputs for verification: {e}", exc_info=True)
            return self.create_error_response(f"Error tokenizing inputs: {str(e)}", status_code=HTTPStatus.INTERNAL_SERVER_ERROR)

        if not completion_token_ids:
            return VerifyDecodingResponse(
                id=request_id,
                created=created_time,
                object="text.verification",
                model=self._get_model_name(request.model, lora_request=lora_request),
                is_verified_greedy=True,
                prompt_token_ids=prompt_token_ids,
                completion_token_ids=completion_token_ids,
                verification_details=[],
                usage=UsageInfo(prompt_tokens=len(prompt_token_ids), completion_tokens=0, total_tokens=len(prompt_token_ids))
            )

        full_sequence_token_ids = prompt_token_ids + completion_token_ids

        # Early length check
        if len(full_sequence_token_ids) > self.max_model_len:
            return self.create_error_response(
                f"Combined prompt+completion length {len(full_sequence_token_ids)} exceeds model max length {self.max_model_len}",
                status_code=HTTPStatus.BAD_REQUEST,
            )

        temp_engine_req_obj = CompletionRequest(model=request.model, prompt=full_sequence_token_ids) 

        try:
            prompt_input = self._tokenize_prompt_input(
                request=temp_engine_req_obj, # Pass the temp request to reuse validation logic if any
                tokenizer=tokenizer,
                prompt_input=full_sequence_token_ids,
                add_special_tokens=False 
            )
            engine_llm_inputs = TokensPrompt(prompt_token_ids=prompt_input["prompt_token_ids"])
            mm_kwargs = {} 

            num_prompt_logprobs = request.prompt_logprobs if request.prompt_logprobs is not None else (self.model_config.max_logprobs or 5)
            # Ensure at least 1 if verification is to make sense, to get ranks
            if num_prompt_logprobs == 0 and request.check_greedy:
                 num_prompt_logprobs = 1 
            elif num_prompt_logprobs is None: # Should not happen if above default is applied
                 num_prompt_logprobs = 0 # Default to 0 if not checking greedy and not specified

            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=1, 
                prompt_logprobs=num_prompt_logprobs,
                logprobs=None, 
                skip_special_tokens=False,
                spaces_between_special_tokens=True 
            )

            result_generator = self.engine_client.generate(
                prompt=engine_llm_inputs, 
                sampling_params=sampling_params, 
                request_id=request_id,
                lora_request=lora_request, # Pass lora_request if supported/needed by engine
                **mm_kwargs
            )
            final_res: Optional[RequestOutput] = None
            async for res_output_item in result_generator:
                if isinstance(res_output_item, Exception):
                    logger.error(f"Engine error during verification: {res_output_item}", exc_info=True)
                    return self.create_error_response(f"Engine error: {str(res_output_item)}", status_code=HTTPStatus.INTERNAL_SERVER_ERROR)
                final_res = res_output_item
        except Exception as e:
            logger.error(f"Error during engine processing for verification: {e}", exc_info=True)
            return self.create_error_response(f"Error during engine processing: {str(e)}", status_code=HTTPStatus.INTERNAL_SERVER_ERROR)

        if final_res is None or final_res.prompt_logprobs is None:
            logger.error("Verification failed: No prompt_logprobs received from engine.")
            return self.create_error_response("Failed to get prompt logprobs from engine.", HTTPStatus.INTERNAL_SERVER_ERROR)

        engine_prompt_logprobs = clamp_prompt_logprobs(final_res.prompt_logprobs)

        # The engine returns logprobs for T_0 to T_N-1 based on prefix T_0...T_i-1.
        # For our full_sequence_token_ids[j], the relevant logprobs are in engine_prompt_logprobs[j].
        # The first token engine_prompt_logprobs[0] will be None or empty dict usually.
        if len(engine_prompt_logprobs) != len(full_sequence_token_ids):
            logger.warning(
                f"Logprobs length mismatch: expected {len(full_sequence_token_ids)}, got {len(engine_prompt_logprobs)}. "
                f"This might be due to special tokens or BOS token handling. Details: {engine_prompt_logprobs}"
            )
            # Attempt to align if the difference is 1 (often due to BOS not having prior logprobs)
            # This is a heuristic. Proper alignment depends on how engine returns prompt_logprobs for the very first token.
            if len(engine_prompt_logprobs) == len(full_sequence_token_ids) -1 and full_sequence_token_ids:
                # Prepend None for the first token's logprobs, assuming it had no preceding context for logprobs
                engine_prompt_logprobs = [None] + engine_prompt_logprobs
                logger.info("Adjusted logprobs length by prepending None for the first token.")
            else:
                return self.create_error_response(
                    f"Logprobs length mismatch from engine. Expected {len(full_sequence_token_ids)}, got {len(engine_prompt_logprobs)}.", 
                    HTTPStatus.INTERNAL_SERVER_ERROR
                )

        verification_details: list[TokenVerificationDetail] = []
        overall_is_greedy = True if request.check_greedy else None

        for i, target_token_id in enumerate(completion_token_ids):
            idx_in_full_seq = len(prompt_token_ids) + i
            
            # Ensure we are within bounds for engine_prompt_logprobs
            if idx_in_full_seq >= len(engine_prompt_logprobs):
                logger.error(f"Index {idx_in_full_seq} out of bounds for engine_prompt_logprobs (len {len(engine_prompt_logprobs)}). Cannot verify token.")
                # Or append a detail indicating missing logprobs
                verification_details.append(TokenVerificationDetail(
                    token_id=target_token_id,
                    text=tokenizer.decode(target_token_id),
                    logprob=0.0,
                    is_greedy_choice=False if request.check_greedy else None,
                    top_logprob_at_step=0.0,
                    top_token_id_at_step=None,
                    error_message="Logprobs not available for this step"
                ))
                if request.check_greedy: overall_is_greedy = False
                continue

            logprobs_for_step: Optional[dict[int, Logprob]] = engine_prompt_logprobs[idx_in_full_seq]
            
            token_text = tokenizer.decode(target_token_id)
            actual_logprob_of_target: Optional[float] = None
            rank_of_target: Optional[int] = None
            top_logprob_at_this_step: Optional[float] = None
            greedy_token_id_at_this_step: Optional[int] = None
            is_choice_greedy_for_this_token: Optional[bool] = None
            error_msg: Optional[str] = None

            if logprobs_for_step is not None and isinstance(logprobs_for_step, dict):
                target_token_logprob_obj = logprobs_for_step.get(target_token_id)
                if target_token_logprob_obj:
                    actual_logprob_of_target = target_token_logprob_obj.logprob
                    rank_of_target = target_token_logprob_obj.rank
                    token_text = target_token_logprob_obj.decoded_token 
                else:
                    error_msg = "Target token not found in logprobs for this step."
                    if request.check_greedy: overall_is_greedy = False # Target not even in top-k cannot be greedy

                # Find the greedy token (rank 1 or highest logprob)
                # Check if prompt_logprobs was 0, in which case rank might not be populated
                if num_prompt_logprobs > 0: 
                    for t_id, lp_obj in logprobs_for_step.items():
                        if lp_obj.rank == 1:
                            greedy_token_id_at_this_step = t_id
                            top_logprob_at_this_step = lp_obj.logprob
                            break
                
                if greedy_token_id_at_this_step is None: # Fallback if rank 1 not found or num_prompt_logprobs was 0
                    sorted_logprobs = sorted(logprobs_for_step.items(), key=lambda item: item[1].logprob, reverse=True)
                    if sorted_logprobs:
                        greedy_token_id_at_this_step = sorted_logprobs[0][0]
                        top_logprob_at_this_step = sorted_logprobs[0][1].logprob
                    elif not error_msg: # If sorted_logprobs is empty but we had no prior error
                         error_msg = "Logprobs dictionary for this step was empty."
                         if request.check_greedy: overall_is_greedy = False
            
            else: # logprobs_for_step is None or not a dict
                error_msg = "Logprobs not available or in unexpected format for this step."
                if request.check_greedy: overall_is_greedy = False

            if request.check_greedy and overall_is_greedy is not False: # if already False, no need to check further for this token
                if error_msg: # If there was an error fetching logprobs, it's not greedy
                    is_choice_greedy_for_this_token = False
                elif rank_of_target == 1 and rank_of_target is not None:
                    is_choice_greedy_for_this_token = True
                # Check if target is the greedy token if rank is not 1 (e.g. prompt_logprobs was 0)
                elif greedy_token_id_at_this_step == target_token_id and greedy_token_id_at_this_step is not None:
                    is_choice_greedy_for_this_token = True
                # Allow for threshold only if ranks are not definitive or not available
                elif actual_logprob_of_target is not None and top_logprob_at_this_step is not None and \
                     actual_logprob_of_target >= top_logprob_at_this_step - request.greedy_logprob_threshold and \
                     (rank_of_target is None or rank_of_target !=1) : # consider greedy if within threshold and not explicitly non-greedy rank
                     is_choice_greedy_for_this_token = True 
                else:
                    is_choice_greedy_for_this_token = False
                
                if not is_choice_greedy_for_this_token:
                    overall_is_greedy = False
            elif not request.check_greedy:
                is_choice_greedy_for_this_token = None # Not applicable
            
            verification_details.append(TokenVerificationDetail(
                token_id=target_token_id,
                text=token_text,
                logprob=actual_logprob_of_target if actual_logprob_of_target is not None else 0.0,
                is_greedy_choice=is_choice_greedy_for_this_token,
                top_logprob_at_step=top_logprob_at_this_step if top_logprob_at_this_step is not None else 0.0,
                top_token_id_at_step=greedy_token_id_at_this_step,
                error_message=error_msg,
                rank=rank_of_target
            ))

        usage = UsageInfo(
            prompt_tokens=len(prompt_token_ids), # Only prompt part for original request
            completion_tokens=len(completion_token_ids),
            total_tokens=len(prompt_token_ids) + len(completion_token_ids),
            # internal_processing_tokens=len(full_sequence_token_ids) # Could add this to show what engine processed
        )

        return VerifyDecodingResponse(
            id=request_id,
            object="text.verification",
            created=created_time,
            model=self._get_model_name(request.model, lora_request=lora_request),
            is_verified_greedy=overall_is_greedy,
            prompt_token_ids=prompt_token_ids,
            completion_token_ids=completion_token_ids,
            verification_details=verification_details,
            usage=usage
        )
