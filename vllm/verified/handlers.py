"""Handler adapters for verified endpoints.

These wrap existing OpenAI-serving classes to provide verified endpoints
without modifying their core logic.
"""
import time
from http import HTTPStatus
from typing import Optional, Union

from fastapi import Request

from vllm.entrypoints.openai.protocol import ErrorResponse, PromptTokenUsageInfo, RequestResponseMetadata, UsageInfo, ChatMessage
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_engine import clamp_prompt_logprobs
from vllm.outputs import RequestOutput
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.entrypoints.utils import get_max_tokens
from vllm.logger import init_logger

from .mixin import VerificationMixin
from .protocol import (
    VerifiedChatCompletionRequest,
    VerifiedChatCompletionResponse,
    VerifiedChatCompletionResponseChoice,
    VerifiedCompletionRequest,
    VerifiedCompletionResponse,
    VerifiedCompletionResponseChoice,
    VerifyDecodingRequest,
    VerifyDecodingResponse,
)

logger = init_logger(__name__)


class VerifiedChatHandler(VerificationMixin):
    async def create(self, svc: OpenAIServingChat, request: VerifiedChatCompletionRequest, raw_request: Request) -> Union[VerifiedChatCompletionResponse, ErrorResponse]:
        error = await svc._check_model(request)
        if error is not None:
            return error

        try:
            request_copy = request.model_copy(deep=True)
            self.enforce_greedy_params(request_copy)
            request_copy.stream = False

            lora_request = svc._maybe_get_adapters(request_copy, supports_default_mm_loras=True)
            tokenizer = await svc.engine_client.get_tokenizer(lora_request)

            conversation, request_prompts, engine_prompts = await svc._preprocess_chat(
                request_copy,
                tokenizer,
                request_copy.messages,
                chat_template=request_copy.chat_template or svc.chat_template,
                chat_template_content_format=svc.chat_template_content_format,
                add_generation_prompt=request_copy.add_generation_prompt,
                continue_final_message=request_copy.continue_final_message,
                tool_dicts=None if request_copy.tools is None else [tool.model_dump() for tool in request_copy.tools],
                documents=request_copy.documents,
                chat_template_kwargs=request_copy.chat_template_kwargs,
                tool_parser=svc.tool_parser,
                add_special_tokens=request_copy.add_special_tokens,
            )

            if not engine_prompts:
                return svc.create_error_response("No engine prompts generated")

            input_length = len(engine_prompts[0]["prompt_token_ids"])
            if svc.default_sampling_params is None:
                svc.default_sampling_params = {}
            max_tokens = get_max_tokens(
                max_model_len=svc.max_model_len,
                request=request_copy,
                input_length=input_length,
                default_sampling_params=svc.default_sampling_params,
            )

            if request_copy.use_beam_search:
                sampling_params = request_copy.to_beam_search_params(max_tokens, svc.default_sampling_params)
            else:
                sampling_params = request_copy.to_sampling_params(max_tokens, svc.model_config.logits_processor_pattern, svc.default_sampling_params)

            if isinstance(sampling_params, SamplingParams):
                sampling_params.temperature = 0.0
                sampling_params.n = 1
                if sampling_params.logprobs is None or sampling_params.logprobs == 0:
                    sampling_params.logprobs = 1
                if getattr(sampling_params, "prompt_logprobs", None) in (None, 0):
                    sampling_params.prompt_logprobs = 1  # type: ignore[attr-defined]

            trace_headers = None
            if raw_request is not None:
                trace_headers = await svc._get_trace_headers(raw_request.headers)

            request_id = f"vchatcmpl-{svc._base_request_id(raw_request)}"
            model_name = svc._get_model_name(request.model, lora_request)

            engine_prompt = engine_prompts[0]
            generator = svc.engine_client.generate(
                engine_prompt,
                sampling_params,
                request_id,
                lora_request=lora_request,
                trace_headers=trace_headers,
                priority=request.priority,
            )

            final_res: Optional[RequestOutput] = None
            async for res in generator:
                if isinstance(res, Exception):
                    return svc.create_error_response(f"Engine error: {res}", status_code=HTTPStatus.INTERNAL_SERVER_ERROR)
                final_res = res

            if final_res is None or not final_res.outputs:
                return svc.create_error_response("No output from engine.", status_code=HTTPStatus.INTERNAL_SERVER_ERROR)

            output = final_res.outputs[0]
            completion_token_ids = list(output.token_ids)
            completion_details = self._build_verified_token_details(completion_token_ids, output.logprobs, tokenizer)

            engine_prompt_token_ids = final_res.prompt_token_ids or []
            prompt_details = None
            if final_res.prompt_logprobs and engine_prompt_token_ids:
                prompt_details = self._build_verified_token_details(
                    engine_prompt_token_ids,
                    clamp_prompt_logprobs(final_res.prompt_logprobs),
                    tokenizer,
                    is_prompt_tokens=True,
                )
            elif engine_prompt_token_ids:
                prompt_details = self._build_verified_token_details(
                    engine_prompt_token_ids,
                    None,
                    tokenizer,
                    is_prompt_tokens=True,
                )

            role = svc.get_chat_request_role(request_copy)
            message = ChatMessage(role=role, content=output.text)
            choice = VerifiedChatCompletionResponseChoice(
                index=0,
                message=message,
                prompt_token_ids=engine_prompt_token_ids,
                completion_token_ids=completion_token_ids,
                completion_token_details=completion_details or [],
                prompt_token_details=prompt_details,
                finish_reason=output.finish_reason,
                stop_reason=output.stop_reason,
            )

            num_prompt_tokens = len(engine_prompt_token_ids)
            if final_res.encoder_prompt_token_ids is not None:
                num_prompt_tokens += len(final_res.encoder_prompt_token_ids)
            num_completion_tokens = len(completion_token_ids)
            usage = UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=num_completion_tokens,
                total_tokens=num_prompt_tokens + num_completion_tokens,
                prompt_tokens_details=(None if final_res.num_cached_tokens is None else PromptTokenUsageInfo(cached_tokens=final_res.num_cached_tokens)),
            )

            response_metadata = RequestResponseMetadata(request_id=request_id, final_usage_info=usage)
            if raw_request and hasattr(raw_request.state, "request_metadata"):
                raw_request.state.request_metadata = response_metadata

            return VerifiedChatCompletionResponse(
                id=request_id,
                created=int(time.time()),
                model=model_name,
                choices=[choice],
                usage=usage,
            )
        except Exception as e:
            logger.exception("Error processing verified chat completion request")
            return svc.create_error_response(str(e), status_code=HTTPStatus.INTERNAL_SERVER_ERROR)


class VerifiedCompletionHandler(VerificationMixin):
    async def create(self, svc: OpenAIServingCompletion, request: VerifiedCompletionRequest, raw_request: Optional[Request] = None) -> Union[VerifiedCompletionResponse, ErrorResponse]:
        error = await svc._check_model(request)
        if error is not None:
            return error

        lora_request = svc._maybe_get_adapters(request)
        tokenizer = await svc.engine_client.get_tokenizer(lora_request)

        request_id = f"vcmpl-{svc._base_request_id(raw_request)}"
        created_time = int(time.time())

        request_copy = request.model_copy(deep=True)
        request_copy.temperature = 0.0
        request_copy.n = 1
        if request_copy.logprobs is None or request_copy.logprobs == 0:
            request_copy.logprobs = 1

        prompt_input_for_max_tokens = await svc._tokenize_prompt_input_async(
            request_copy,
            tokenizer,
            request_copy.prompt,
            add_special_tokens=request_copy.add_special_tokens,
        )
        num_prompt_tokens_for_max_tokens = len(prompt_input_for_max_tokens["prompt_token_ids"])
        if request_copy.max_tokens is None:
            request_copy.max_tokens = svc.max_model_len - num_prompt_tokens_for_max_tokens
        if request_copy.max_tokens < 0:
            request_copy.max_tokens = 0

        try:
            prompt_input = await svc._tokenize_prompt_input_async(
                request_copy, tokenizer, request_copy.prompt, add_special_tokens=request_copy.add_special_tokens
            )
            engine_inputs = {"prompt_token_ids": prompt_input["prompt_token_ids"]}
            assert request_copy.max_tokens is not None
            sampling_params = request_copy.to_sampling_params(
                request_copy.max_tokens, svc.model_config.logits_processor_pattern, svc.default_sampling_params
            )
            sampling_params.temperature = 0.0
            sampling_params.n = 1
            if sampling_params.logprobs is None or sampling_params.logprobs == 0:
                sampling_params.logprobs = 1

            trace_headers = None
            if raw_request is not None:
                trace_headers = await svc._get_trace_headers(raw_request.headers)

            result_generator = svc.engine_client.generate(
                prompt=engine_inputs,
                sampling_params=sampling_params,
                request_id=request_id,
                lora_request=lora_request,
                trace_headers=trace_headers,
                priority=request.priority,
            )
            final_res: Optional[RequestOutput] = None
            async for res_output in result_generator:
                if isinstance(res_output, Exception):
                    return svc.create_error_response(f"Engine error: {str(res_output)}", status_code=HTTPStatus.INTERNAL_SERVER_ERROR)
                final_res = res_output

            if final_res is None or not final_res.outputs:
                return svc.create_error_response("No output from engine.", status_code=HTTPStatus.INTERNAL_SERVER_ERROR)

            completion_output = final_res.outputs[0]
            actual_completion_token_ids = list(completion_output.token_ids)
            completion_details = self._build_verified_token_details(
                actual_completion_token_ids, completion_output.logprobs, tokenizer
            )

            engine_prompt_token_ids = final_res.prompt_token_ids or []
            if final_res.prompt_logprobs and engine_prompt_token_ids:
                prompt_details = self._build_verified_token_details(
                    engine_prompt_token_ids,
                    clamp_prompt_logprobs(final_res.prompt_logprobs),
                    tokenizer,
                    is_prompt_tokens=True,
                )
            else:
                prompt_details = self._build_verified_token_details(
                    engine_prompt_token_ids, None, tokenizer, is_prompt_tokens=True
                )

            choice = VerifiedCompletionResponseChoice(
                index=0,
                text=completion_output.text,
                prompt_token_ids=engine_prompt_token_ids,
                completion_token_ids=actual_completion_token_ids,
                completion_token_details=completion_details or [],
                prompt_token_details=prompt_details,
                finish_reason=completion_output.finish_reason,
                stop_reason=completion_output.stop_reason,
            )

            num_prompt_tokens = len(engine_prompt_token_ids)
            num_completion_tokens = len(actual_completion_token_ids)
            usage = UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=num_completion_tokens,
                total_tokens=num_prompt_tokens + num_completion_tokens,
                prompt_tokens_details=(
                    None if final_res.num_cached_tokens is None else PromptTokenUsageInfo(cached_tokens=final_res.num_cached_tokens)
                ),
            )

            response_metadata = RequestResponseMetadata(request_id=request_id, final_usage_info=usage)
            if raw_request and hasattr(raw_request.state, "request_metadata"):
                raw_request.state.request_metadata = response_metadata

            return VerifiedCompletionResponse(
                id=request_id,
                object="text_completion.verified",
                created=created_time,
                model=svc._get_model_name(request.model, lora_request=lora_request),
                choices=[choice],
                usage=usage,
            )
        except Exception as e:
            logger.exception("Error processing verified completion request")
            return svc.create_error_response(str(e), status_code=HTTPStatus.INTERNAL_SERVER_ERROR)


class VerifyDecodingHandler(VerificationMixin):
    async def verify(self, svc: OpenAIServingCompletion, request: VerifyDecodingRequest, raw_request: Optional[Request] = None) -> Union[VerifyDecodingResponse, ErrorResponse]:
        error = await svc._check_model(request)
        if error is not None:
            return error

        lora_request = None
        tokenizer = await svc.engine_client.get_tokenizer(lora_request)

        request_id = f"verify-{svc._base_request_id(raw_request)}"
        created_time = int(time.time())

        if isinstance(request.prompt, str):
            prompt_token_ids = tokenizer.encode(request.prompt)
        else:
            prompt_token_ids = list(request.prompt)

        if isinstance(request.completion, str):
            completion_token_ids = tokenizer.encode(request.completion)
        else:
            completion_token_ids = list(request.completion)

        if not completion_token_ids:
            return VerifyDecodingResponse(
                id=request_id,
                created=created_time,
                object="text.verification",
                model=svc._get_model_name(request.model, lora_request=lora_request),
                is_verified_greedy=True,
                prompt_token_ids=prompt_token_ids,
                completion_token_ids=completion_token_ids,
                verification_details=[],
                usage=UsageInfo(prompt_tokens=len(prompt_token_ids), completion_tokens=0, total_tokens=len(prompt_token_ids)),
            )

        full_seq = prompt_token_ids + completion_token_ids
        if len(full_seq) > svc.max_model_len:
            return svc.create_error_response(
                f"Combined prompt+completion length {len(full_seq)} exceeds model max length {svc.max_model_len}",
                status_code=HTTPStatus.BAD_REQUEST,
            )

        from vllm.entrypoints.openai.protocol import CompletionRequest as CoreCompletionRequest

        temp_req = CoreCompletionRequest(model=request.model, prompt=full_seq)
        prompt_input = await svc._tokenize_prompt_input_async(
            request=temp_req, tokenizer=tokenizer, prompt_input=full_seq, add_special_tokens=False
        )
        engine_llm_inputs = {"prompt_token_ids": prompt_input["prompt_token_ids"]}

        num_prompt_logprobs = (
            request.prompt_logprobs if request.prompt_logprobs is not None else (svc.model_config.max_logprobs or 5)
        )
        if num_prompt_logprobs == 0 and request.check_greedy:
            num_prompt_logprobs = 1
        elif num_prompt_logprobs is None:
            num_prompt_logprobs = 0

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            prompt_logprobs=num_prompt_logprobs,
            logprobs=None,
            skip_special_tokens=False,
            spaces_between_special_tokens=True,
        )

        result_generator = svc.engine_client.generate(
            prompt=engine_llm_inputs,
            sampling_params=sampling_params,
            request_id=request_id,
            lora_request=lora_request,
        )
        final_res: Optional[RequestOutput] = None
        async for res_output_item in result_generator:
            if isinstance(res_output_item, Exception):
                logger.exception("Engine error during verification: %s", res_output_item)
                return svc.create_error_response(f"Engine error: {str(res_output_item)}", status_code=HTTPStatus.INTERNAL_SERVER_ERROR)
            final_res = res_output_item

        if final_res is None or final_res.prompt_logprobs is None:
            logger.exception("Verification failed: No prompt_logprobs received from engine.")
            return svc.create_error_response("Failed to get prompt logprobs from engine.", HTTPStatus.INTERNAL_SERVER_ERROR)

        engine_prompt_logprobs = clamp_prompt_logprobs(final_res.prompt_logprobs)
        if len(engine_prompt_logprobs) != len(full_seq):
            if len(engine_prompt_logprobs) == len(full_seq) - 1 and full_seq:
                engine_prompt_logprobs = [None] + engine_prompt_logprobs  # type: ignore[list-item]
            else:
                return svc.create_error_response(
                    f"Logprobs length mismatch from engine. Expected {len(full_seq)}, got {len(engine_prompt_logprobs)}.",
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )

        details, overall_is_greedy = self.create_token_verification_details(
            completion_token_ids, engine_prompt_logprobs, len(prompt_token_ids), tokenizer, request.check_greedy, request.greedy_logprob_threshold
        )

        usage = UsageInfo(
            prompt_tokens=len(prompt_token_ids),
            completion_tokens=len(completion_token_ids),
            total_tokens=len(prompt_token_ids) + len(completion_token_ids),
        )

        return VerifyDecodingResponse(
            id=request_id,
            object="text.verification",
            created=created_time,
            model=svc._get_model_name(request.model, lora_request=lora_request),
            is_verified_greedy=overall_is_greedy,
            prompt_token_ids=prompt_token_ids,
            completion_token_ids=completion_token_ids,
            verification_details=details,
            usage=usage,
        )


