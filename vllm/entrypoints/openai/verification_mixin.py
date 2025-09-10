"""Shared functionality for verification endpoints."""
import time
from typing import Dict, List, Optional, Union

from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest, CompletionRequest, VerifiedTokenDetail,
    TokenVerificationDetail
)
from vllm.logger import init_logger
from vllm.sequence import Logprob
from vllm.transformers_utils.tokenizer import AnyTokenizer

logger = init_logger(__name__)


class VerificationMixin:
    """Shared functionality for verification endpoints."""
    
    def enforce_greedy_params(
        self, 
        request: Union[CompletionRequest, ChatCompletionRequest]
    ) -> None:
        """Enforce parameters for deterministic generation."""
        request.temperature = 0.0
        request.n = 1
        if request.logprobs is None:
            request.logprobs = True
        if hasattr(request, 'top_logprobs') and (request.top_logprobs is None or request.top_logprobs == 0):
            request.top_logprobs = 1
    
    def create_verified_token_details(
        self,
        token_ids: Optional[List[int]],
        logprobs: Optional[List[Optional[Dict[int, Logprob]]]],
        tokenizer: AnyTokenizer,
        is_prompt_tokens: bool = False,
        include_alternatives: bool = False
    ) -> Optional[List[VerifiedTokenDetail]]:
        """Create unified token detail objects with optional alternatives."""
        if not token_ids:
            return None
            
        # Handle case where logprobs are not available for prompt tokens
        if is_prompt_tokens and (logprobs is None or not logprobs):
            # Still create details but without logprob info
            return [
                VerifiedTokenDetail(
                    token_id=tid, 
                    text=tokenizer.decode([tid], skip_special_tokens=False), 
                    logprob=0.0, 
                    rank=None
                ) 
                for tid in token_ids
            ]
        
        if logprobs is None:
            if not is_prompt_tokens:
                logger.warning("logprobs is None for completion tokens, cannot create verified details.")
            return [
                VerifiedTokenDetail(
                    token_id=tid, 
                    text=tokenizer.decode([tid], skip_special_tokens=False), 
                    logprob=0.0, 
                    rank=None
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
                rank=None
            )

            if step_logprobs_dict is not None and token_id in step_logprobs_dict:
                logprob_obj = step_logprobs_dict[token_id]
                detail.text = logprob_obj.decoded_token
                detail.logprob = logprob_obj.logprob
                detail.rank = logprob_obj.rank
                
                if include_alternatives and len(step_logprobs_dict) > 1:
                    # Get top alternatives (excluding the selected token)
                    sorted_alternatives = sorted(
                        step_logprobs_dict.items(),
                        key=lambda x: x[1].logprob,
                        reverse=True
                    )[:5]  # Top 5 alternatives
                    
                    detail.alternatives = [
                        {
                            "token_id": alt_tid,
                            "text": alt_lp.decoded_token,
                            "logprob": alt_lp.logprob,
                            "rank": alt_lp.rank
                        }
                        for alt_tid, alt_lp in sorted_alternatives
                        if alt_tid != token_id  # Exclude the selected token
                    ]
            elif is_prompt_tokens and i == 0 and step_logprobs_dict is None:
                # First prompt token often has no logprob
                pass
            else:
                logger.debug(
                    f"Missing logprob for token_id {token_id} at index {i}. "
                    f"is_prompt={is_prompt_tokens}. Logprobs for this step: {step_logprobs_dict}"
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
        greedy_logprob_threshold: float = 0.001
    ) -> tuple[List[TokenVerificationDetail], Optional[bool]]:
        """Create token verification details for verify_decoding endpoint."""
        verification_details: List[TokenVerificationDetail] = []
        overall_is_greedy = True if check_greedy else None
        
        for i, target_token_id in enumerate(completion_token_ids):
            idx_in_full_seq = prompt_length + i
            
            # Initialize detail with defaults
            detail = TokenVerificationDetail(
                token_id=target_token_id,
                text=tokenizer.decode([target_token_id], skip_special_tokens=False),
                logprob=0.0,
                is_greedy_choice=None,
                top_logprob_at_step=0.0,
                top_token_id_at_step=None,
                error_message=None,
                rank=None
            )
            
            # Check bounds
            if idx_in_full_seq >= len(engine_prompt_logprobs):
                detail.error_message = "Logprobs not available for this step"
                if check_greedy:
                    detail.is_greedy_choice = False
                    overall_is_greedy = False
                verification_details.append(detail)
                continue
                
            logprobs_for_step = engine_prompt_logprobs[idx_in_full_seq]
            
            if logprobs_for_step is None or not isinstance(logprobs_for_step, dict):
                detail.error_message = "Logprobs not available or in unexpected format for this step"
                if check_greedy:
                    detail.is_greedy_choice = False
                    overall_is_greedy = False
                verification_details.append(detail)
                continue
                
            # Get target token info
            target_logprob_obj = logprobs_for_step.get(target_token_id)
            if target_logprob_obj:
                detail.text = target_logprob_obj.decoded_token
                detail.logprob = target_logprob_obj.logprob
                detail.rank = target_logprob_obj.rank
            else:
                detail.error_message = "Target token not found in logprobs for this step"
                if check_greedy:
                    detail.is_greedy_choice = False
                    overall_is_greedy = False
                    
            # Find greedy token (rank 1 or highest logprob)
            greedy_token_id = None
            top_logprob = None
            
            # First try to find rank 1 token
            for t_id, lp_obj in logprobs_for_step.items():
                if lp_obj.rank == 1:
                    greedy_token_id = t_id
                    top_logprob = lp_obj.logprob
                    break
                    
            # Fallback to highest logprob if rank 1 not found
            if greedy_token_id is None:
                sorted_logprobs = sorted(
                    logprobs_for_step.items(),
                    key=lambda item: item[1].logprob,
                    reverse=True
                )
                if sorted_logprobs:
                    greedy_token_id = sorted_logprobs[0][0]
                    top_logprob = sorted_logprobs[0][1].logprob
                    
            detail.top_token_id_at_step = greedy_token_id
            detail.top_logprob_at_step = top_logprob
            
            # Check if token is greedy
            if check_greedy and overall_is_greedy is not False:
                if detail.error_message:
                    detail.is_greedy_choice = False
                elif detail.rank == 1:
                    detail.is_greedy_choice = True
                elif greedy_token_id == target_token_id:
                    detail.is_greedy_choice = True
                elif (detail.logprob is not None and top_logprob is not None and
                      detail.logprob >= top_logprob - greedy_logprob_threshold):
                    detail.is_greedy_choice = True
                else:
                    detail.is_greedy_choice = False
                    
                if not detail.is_greedy_choice:
                    overall_is_greedy = False
                    
            verification_details.append(detail)
            
        return verification_details, overall_is_greedy

    def _build_verified_token_details(
        self,
        token_ids: Optional[GenericSequence[int]],
        sample_logprobs: Optional[GenericSequence[Optional[dict[int, Logprob]]]],
        tokenizer: AnyTokenizer,
        *,
        is_prompt_tokens: bool = False,
    ) -> Optional[list[VerifiedTokenDetail]]:
        if not token_ids:
            return None

        # If logprobs were not requested for prompt, still return ids/text.
        if is_prompt_tokens and (sample_logprobs is None or not sample_logprobs):
            return [
                VerifiedTokenDetail(
                    token_id=tid,
                    text=tokenizer.decode(tid),
                    logprob=None,
                    rank=None,
                ) for tid in token_ids
            ]

        # If no logprobs at all, return decoded tokens with unknown logprob.
        if sample_logprobs is None:
            return [
                VerifiedTokenDetail(
                    token_id=tid,
                    text=tokenizer.decode(tid),
                    logprob=None,
                    rank=None,
                ) for tid in token_ids
            ]

        details: list[VerifiedTokenDetail] = []
        for i, token_id in enumerate(token_ids):
            step_logprobs = sample_logprobs[i] if i < len(sample_logprobs) else None

            if step_logprobs is not None and token_id in step_logprobs:
                lp_obj = step_logprobs[token_id]
                details.append(
                    VerifiedTokenDetail(
                        token_id=token_id,
                        text=lp_obj.decoded_token,
                        logprob=lp_obj.logprob,
                        rank=lp_obj.rank,
                    ))
            elif is_prompt_tokens and i == 0 and step_logprobs is None:
                details.append(
                    VerifiedTokenDetail(
                        token_id=token_id,
                        text=tokenizer.decode(token_id),
                        logprob=None,
                        rank=None,
                    ))
            else:
                details.append(
                    VerifiedTokenDetail(
                        token_id=token_id,
                        text=tokenizer.decode(token_id),
                        logprob=None,
                        rank=None,
                    ))
        return details