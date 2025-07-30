from fastapi import Request
from fastapi.responses import JSONResponse
from jwt import ExpiredSignatureError, InvalidTokenError

from core import logger
from exceptions import (
    BaseCustomException,
    NotFoundException,
    UnauthorizedException,
)


def log(request: Request, exc: Exception):
    logger.error(
        f"Error occurred: {exc}. Endpoint: {request.url}. Method: {request.method}. "
        f"Client: {request.client}. Headers: {request.headers}"
    )


# Exception Handlers
async def base_custom_exception_handler(request: Request, exc: BaseCustomException) -> JSONResponse:
    log(request, exc)
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "type": "BaseCustomException"},
    )


async def not_found_exception_handler(request: Request, exc: NotFoundException) -> JSONResponse:
    log(request, exc)
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "type": "NotFoundException"},
    )


async def unauthorized_exception_handler(request: Request, exc: UnauthorizedException) -> JSONResponse:
    log(request, exc)
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "type": "UnauthorizedException"},
    )


async def expired_signature_error_exception_handler(request: Request, exc: ExpiredSignatureError) -> JSONResponse:
    log(request, exc)
    return JSONResponse(
        status_code=401,
        content={"detail": "Token has expired", "type": "ExpiredSignatureError"},
    )


async def invalid_token_error_exception_handler(request: Request, exc: InvalidTokenError) -> JSONResponse:
    log(request, exc)
    return JSONResponse(
        status_code=401,
        content={"detail": "Invalid authentication token", "type": "InvalidTokenError"},
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    log(request, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred", "type": "GeneralException"},
    )
