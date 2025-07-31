__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

from typing import Optional

from fastapi import APIRouter
from fastapi.params import Query
from sse_starlette.sse import EventSourceResponse

from schemas import ChatResponse
from services.chat_service import chat_service

prefix = "chat"
router = APIRouter(
    prefix=f"/{prefix}",
    responses={401: {"description": "raw data file upload"}}
)


@router.get(
    "/new-chat",
    response_model=ChatResponse,
    summary="새 채팅 시작",
    description="새로운 채팅 세션을 시작합니다."
)
async def new_chat(
    query: str = Query(..., description="Prompt Query"),
    use_langchain: Optional[bool] = Query(True, description="Trigger for Langchain")
):
    return await chat_service.execute_chat(query, use_langchain)


@router.get(
    "/load-chat",
    response_model=ChatResponse,
    summary="기존 채팅 로드",
    description="기존 채팅 세션을 로드하여 대화를 이어갑니다."
)
async def load_chat(
    query: str = Query(..., description="Prompt Query"),
    thread_id: str = Query(..., description="Thread ID", example="f09d8c6e-fcb5-4275-bf3d-90a87ede2cb8"),
    use_langchain: Optional[bool] = Query(True, description="Trigger for Langchain")
):
    return await chat_service.execute_chat(query, use_langchain, thread_id)


@router.get(
    "/stream/new-chat",
    summary="새 채팅 스트리밍",
    description="새로운 채팅 세션을 스트리밍으로 시작합니다."
)
async def stream_new_chat(
    query: str = Query(..., description="Prompt Query"),
    use_langchain: Optional[bool] = Query(True, description="Trigger for Langchain")
):
    return EventSourceResponse(
        chat_service.execute_stream_chat(query, use_langchain)
    )


@router.get(
    "/stream/load-chat",
    summary="기존 채팅 스트리밍 로드",
    description="기존 채팅 세션을 스트리밍으로 로드합니다."
)
async def stream_load_chat(
    query: str = Query(..., description="Prompt Query"),
    thread_id: str = Query(..., description="Thread ID", example="f09d8c6e-fcb5-4275-bf3d-90a87ede2cb8"),
    use_langchain: Optional[bool] = Query(True, description="Trigger for Langchain")
):
    return EventSourceResponse(
        chat_service.execute_stream_chat(query, use_langchain, thread_id)
    )
