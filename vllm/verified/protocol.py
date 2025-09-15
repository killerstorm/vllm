"""Protocol extensions for verified endpoints.

These Pydantic models extend the existing OpenAI-compatible protocol with
verified variants and verification response schemas.
"""
import time
from typing import Literal, Optional, Union

from pydantic import Field

from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatMessage,
    CompletionRequest,
    OpenAIBaseModel,
    UsageInfo,
)
from vllm.utils import random_uuid


class VerifiedTokenDetail(OpenAIBaseModel):
    token_id: int
    text: Optional[str] = None
    logprob: Optional[float] = None
    rank: Optional[int] = None


class VerifiedCompletionRequest(CompletionRequest):
    pass


class VerifiedCompletionResponseChoice(OpenAIBaseModel):
    index: int
    text: str
    prompt_token_ids: list[int]
    completion_token_ids: list[int]
    completion_token_details: list[VerifiedTokenDetail]
    prompt_token_details: Optional[list[VerifiedTokenDetail]] = None
    finish_reason: Optional[str] = None
    stop_reason: Union[int, str, None] = None


class VerifiedCompletionResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-verified-{random_uuid()}")
    object: str = "text_completion.verified"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[VerifiedCompletionResponseChoice]
    usage: UsageInfo


class VerifiedChatCompletionRequest(ChatCompletionRequest):
    pass


class VerifiedChatCompletionResponseChoice(OpenAIBaseModel):
    index: int
    message: ChatMessage
    prompt_token_ids: list[int]
    completion_token_ids: list[int]
    completion_token_details: list[VerifiedTokenDetail]
    prompt_token_details: Optional[list[VerifiedTokenDetail]] = None
    finish_reason: Optional[str] = "stop"
    stop_reason: Union[int, str, None] = None


class VerifiedChatCompletionResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-verified-{random_uuid()}")
    object: Literal["chat.completion.verified"] = "chat.completion.verified"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[VerifiedChatCompletionResponseChoice]
    usage: UsageInfo


class VerifyDecodingRequest(OpenAIBaseModel):
    model: Optional[str] = None
    prompt: Union[str, list[int]]
    completion: Union[str, list[int]]
    prompt_logprobs: Optional[int] = Field(default=None)
    check_greedy: bool = Field(default=True)
    greedy_logprob_threshold: float = Field(default=0.001)


class TokenVerificationDetail(OpenAIBaseModel):
    token_id: int
    text: Optional[str] = None
    logprob: float
    is_greedy_choice: Optional[bool] = None
    top_logprob_at_step: Optional[float] = None
    top_token_id_at_step: Optional[int] = None
    error_message: Optional[str] = None
    rank: Optional[int] = None


class VerifyDecodingResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"verdec-{random_uuid()}")
    object: str = "text.verification"
    model: str
    is_verified_greedy: Optional[bool] = None
    prompt_token_ids: Optional[list[int]] = None
    completion_token_ids: Optional[list[int]] = None
    verification_details: list[TokenVerificationDetail]
    usage: UsageInfo


