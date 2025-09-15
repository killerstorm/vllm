"""Verified generation and verification APIs for vLLM.

This package contains implementation for verified endpoints and helpers,
while keeping modifications to the core vLLM code minimal.
"""

# Re-export protocol types for convenience
from .protocol import (  # noqa: F401
    VerifiedTokenDetail,
    VerifiedCompletionRequest,
    VerifiedCompletionResponse,
    VerifiedCompletionResponseChoice,
    VerifiedChatCompletionRequest,
    VerifiedChatCompletionResponse,
    VerifiedChatCompletionResponseChoice,
    VerifyDecodingRequest,
    VerifyDecodingResponse,
    TokenVerificationDetail,
)


