"""Shared functionality for verified endpoints.

This mixin is imported by serving handlers to keep verification logic
out of the core serving modules.
"""
from typing import Dict, List, Optional, Union

from vllm.entrypoints.openai.protocol import ChatCompletionRequest, CompletionRequest
from vllm.logprobs import Logprob
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.verified.protocol import TokenVerificationDetail, VerifiedTokenDetail
from vllm.logger import init_logger

logger = init_logger(__name__)


class VerificationMixin:
    def enforce_greedy_params(
        self,
        request: Union[CompletionRequest, ChatCompletionRequest],
    ) -> None:
        request.temperature = 0.0
        request.n = 1
        if request.logprobs is None:
            request.logprobs = True
        if hasattr(request, "top_logprobs") and (
            request.top_logprobs is None or request.top_logprobs == 0
        ):
            request.top_logprobs = 1

    def create_verified_token_details(
        self,
        token_ids: Optional[List[int]],
        logprobs: Optional[List[Optional[Dict[int, Logprob]]]],
        tokenizer: AnyTokenizer,
        is_prompt_tokens: bool = False,
    ) -> Optional[List[VerifiedTokenDetail]]:
        if not token_ids:
            return None
        if is_prompt_tokens and (logprobs is None or not logprobs):
            return [
                VerifiedTokenDetail(
                    token_id=tid,
                    text=tokenizer.decode([tid], skip_special_tokens=False),
                    logprob=0.0,
                    rank=None,
                )
                for tid in token_ids
            ]

        if logprobs is None:
            return [
                VerifiedTokenDetail(
                    token_id=tid,
                    text=tokenizer.decode([tid], skip_special_tokens=False),
                    logprob=0.0,
                    rank=None,
                )
                for tid in token_ids
            ]

        details: List[VerifiedTokenDetail] = []
        for i, token_id in enumerate(token_ids):
            step_logprobs_dict = logprobs[i] if i < len(logprobs) else None
            detail = VerifiedTokenDetail(
                token_id=token_id,
                text=tokenizer.decode([token_id], skip_special_tokens=False),
                logprob=0.0,
                rank=None,
            )
            if step_logprobs_dict is not None and token_id in step_logprobs_dict:
                lp = step_logprobs_dict[token_id]
                detail.text = lp.decoded_token
                detail.logprob = lp.logprob
                detail.rank = lp.rank
            elif is_prompt_tokens and i == 0 and step_logprobs_dict is None:
                pass
            else:
                logger.debug(
                    "Missing logprob for token_id %d at index %d. is_prompt=%s",
                    token_id,
                    i,
                    is_prompt_tokens,
                )
            details.append(detail)
        return details

    def create_token_verification_details(
        self,
        completion_token_ids: List[int],
        engine_prompt_logprobs: List[Optional[Dict[int, Logprob]]],
        prompt_length: int,
        tokenizer: AnyTokenizer,
        check_greedy: bool = True,
        greedy_logprob_threshold: float = 0.001,
    ) -> tuple[List[TokenVerificationDetail], Optional[bool]]:
        verification_details: List[TokenVerificationDetail] = []
        overall_is_greedy = True if check_greedy else None

        for i, target_token_id in enumerate(completion_token_ids):
            idx = prompt_length + i
            detail = TokenVerificationDetail(
                token_id=target_token_id,
                text=tokenizer.decode([target_token_id], skip_special_tokens=False),
                logprob=0.0,
                is_greedy_choice=None,
                top_logprob_at_step=0.0,
                top_token_id_at_step=None,
                error_message=None,
                rank=None,
            )

            if idx >= len(engine_prompt_logprobs):
                detail.error_message = "Logprobs not available for this step"
                if check_greedy:
                    detail.is_greedy_choice = False
                    overall_is_greedy = False
                verification_details.append(detail)
                continue

            step = engine_prompt_logprobs[idx]
            if step is None or not isinstance(step, dict):
                detail.error_message = (
                    "Logprobs not available or in unexpected format for this step"
                )
                if check_greedy:
                    detail.is_greedy_choice = False
                    overall_is_greedy = False
                verification_details.append(detail)
                continue

            target = step.get(target_token_id)
            if target:
                detail.text = target.decoded_token
                detail.logprob = target.logprob
                detail.rank = target.rank
            else:
                detail.error_message = "Target token not found in logprobs for this step"
                if check_greedy:
                    detail.is_greedy_choice = False
                    overall_is_greedy = False

            greedy_token_id = None
            top_logprob = None
            for t_id, lp in step.items():
                if lp.rank == 1:
                    greedy_token_id = t_id
                    top_logprob = lp.logprob
                    break
            if greedy_token_id is None and step:
                # fallback to highest logprob
                greedy_token_id, top = max(step.items(), key=lambda kv: kv[1].logprob)
                top_logprob = top.logprob

            detail.top_token_id_at_step = greedy_token_id
            detail.top_logprob_at_step = top_logprob

            if check_greedy and overall_is_greedy is not False:
                if detail.error_message:
                    detail.is_greedy_choice = False
                elif detail.rank == 1:
                    detail.is_greedy_choice = True
                elif greedy_token_id == target_token_id:
                    detail.is_greedy_choice = True
                elif (
                    detail.logprob is not None
                    and top_logprob is not None
                    and detail.logprob >= top_logprob - greedy_logprob_threshold
                ):
                    detail.is_greedy_choice = True
                else:
                    detail.is_greedy_choice = False
                if not detail.is_greedy_choice:
                    overall_is_greedy = False

            verification_details.append(detail)

        return verification_details, overall_is_greedy

    # Compatibility shim for existing serving_* modules
    def _build_verified_token_details(
        self,
        token_ids: Optional[list[int]],
        sample_logprobs: Optional[list[Optional[dict[int, Logprob]]]],
        tokenizer: AnyTokenizer,
        *,
        is_prompt_tokens: bool = False,
    ) -> Optional[list[VerifiedTokenDetail]]:
        return self.create_verified_token_details(
            token_ids, sample_logprobs, tokenizer, is_prompt_tokens=is_prompt_tokens
        )


