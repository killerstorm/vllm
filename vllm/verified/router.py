"""Route registration for verified endpoints.

This module exposes a helper to add the verified endpoints to an existing
FastAPI router with minimal changes to the main API server module.
"""
from fastapi import APIRouter, Depends, HTTPException, Request
from http import HTTPStatus
from fastapi.responses import JSONResponse
from vllm.logger import init_logger

from vllm.entrypoints.utils import load_aware_call, with_cancellation
from vllm.entrypoints.openai.protocol import ErrorResponse

from .protocol import (
    VerifiedChatCompletionRequest,
    VerifiedCompletionRequest,
    VerifyDecodingRequest,
)
from .handlers import VerifiedChatHandler, VerifiedCompletionHandler, VerifyDecodingHandler


def register_verified_routes(
    router: APIRouter,
    get_chat,
    get_completion,
    validate_json_request,
) -> None:
    logger = init_logger(__name__)
    chat_handler = VerifiedChatHandler()
    completion_handler = VerifiedCompletionHandler()
    verify_handler = VerifyDecodingHandler()

    @router.post("/v1/completions/verified", dependencies=[Depends(validate_json_request)])
    @with_cancellation
    @load_aware_call
    async def create_verified_completion(
        request: VerifiedCompletionRequest,
        raw_request: Request,
    ):
        svc = get_completion(raw_request)
        if svc is None:
            raise HTTPException(HTTPStatus.NOT_FOUND, "Completion model not found.")
        try:
            res = await completion_handler.create(svc, request, raw_request)
        except ValueError as e:
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST.value, detail=str(e)) from e
        except Exception as e:
            logger.exception("Verified completion failed")
            raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=str(e)) from e
        if isinstance(res, ErrorResponse):
            return JSONResponse(content=res.model_dump(), status_code=res.error.code)
        return JSONResponse(content=res.model_dump())

    @router.post("/v1/chat/completions/verified", dependencies=[Depends(validate_json_request)])
    @with_cancellation
    @load_aware_call
    async def create_verified_chat_completion(
        request: VerifiedChatCompletionRequest,
        raw_request: Request,
    ):
        svc = get_chat(raw_request)
        if svc is None:
            raise HTTPException(HTTPStatus.NOT_FOUND, "Chat model not found.")
        try:
            res = await chat_handler.create(svc, request, raw_request)
        except ValueError as e:
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST.value, detail=str(e)) from e
        except Exception as e:
            logger.exception("Verified chat completion failed")
            raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=str(e)) from e
        if isinstance(res, ErrorResponse):
            return JSONResponse(content=res.model_dump(), status_code=res.error.code)
        return JSONResponse(content=res.model_dump())

    @router.post("/v1/verify_decoding", dependencies=[Depends(validate_json_request)])
    @with_cancellation
    async def verify_text_decoding(
        request: VerifyDecodingRequest,
        raw_request: Request,
    ):
        svc = get_completion(raw_request)
        if svc is None:
            raise HTTPException(
                HTTPStatus.NOT_FOUND,
                "Completion model not found, cannot verify decoding.",
            )
        try:
            res = await verify_handler.verify(svc, request, raw_request)
        except ValueError as e:
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST.value, detail=str(e)) from e
        except Exception as e:
            logger.exception("Verify decoding failed")
            raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=str(e)) from e
        if isinstance(res, ErrorResponse):
            return JSONResponse(content=res.model_dump(), status_code=res.error.code)
        return JSONResponse(content=res.model_dump())


