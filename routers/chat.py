__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
from typing import Optional

# Third-party imports
from fastapi import APIRouter
from fastapi.params import Query
from sse_starlette.sse import EventSourceResponse

# Custom imports
from schemas import ChatResponse, SSEStreamDocumentation
from mcp_agent.schemas import ChatHistory
from services.business.chat_service import chat_service

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
    use_langchain: Optional[bool] = Query(True, description="Trigger for Langchain"),
    system: Optional[str] = Query(None, description="User custom system prompt"),
):
    return await chat_service.execute_chat(query, system=system, use_langchain=use_langchain)


@router.get(
    "/load-chat",
    response_model=ChatResponse,
    summary="기존 채팅 로드",
    description="기존 채팅 세션을 로드하여 대화를 이어갑니다."
)
async def load_chat(
    query: str = Query(..., description="Prompt Query"),
    thread_id: str = Query(..., description="Thread ID", example="f09d8c6e-fcb5-4275-bf3d-90a87ede2cb8"),
    use_langchain: Optional[bool] = Query(True, description="Trigger for Langchain"),
    system: Optional[str] = Query(None, description="User custom system prompt"),
):
    return await chat_service.execute_chat(query, system, thread_id, use_langchain=use_langchain)


@router.get(
    "/stream/new-chat",
    summary="새 채팅 스트리밍",
    description="새로운 채팅 세션을 스트리밍으로 시작합니다.",
    responses=SSEStreamDocumentation.get_chat_stream_responses()
)
async def stream_new_chat(
    query: str = Query(..., description="Prompt Query"),
    use_langchain: Optional[bool] = Query(True, description="Trigger for Langchain"),
    system: Optional[str] = Query(None, description="User custom system prompt"),
):
    """
    새로운 채팅을 SSE 스트림으로 시작합니다.

    **이벤트 타입:**
    - `ai`: AI 응답 내용
    - `tool`: MCP 서버 도구 사용 결과
    - `complete`: 작업 완료 (thread_id 반환)
    - `error`: 에러 발생
    """
    return EventSourceResponse(
        chat_service.execute_stream_chat(query, system, use_langchain=use_langchain)
    )


@router.get(
    "/stream/load-chat",
    summary="기존 채팅 스트리밍 로드",
    description="기존 채팅 세션을 스트리밍으로 로드합니다.",
    responses=SSEStreamDocumentation.get_chat_stream_responses()
)
async def stream_load_chat(
        query: str = Query(..., description="Prompt Query"),
        thread_id: str = Query(..., description="Thread ID", example="f09d8c6e-fcb5-4275-bf3d-90a87ede2cb8"),
        system: Optional[str] = Query(None, description="User custom system prompt"),
        use_langchain: Optional[bool] = Query(True, description="Trigger for Langchain"),
):
    """
    기존 채팅 세션을 SSE 스트림으로 로드합니다.

    **이벤트 타입:**
    - `ai`: AI 응답 내용
    - `tool`: MCP 서버 도구 사용 결과
    - `complete`: 작업 완료 (thread_id 반환)
    - `error`: 에러 발생
    """
    return EventSourceResponse(
        chat_service.execute_stream_chat(query, system, thread_id, use_langchain=use_langchain)
    )


@router.get(
    "/history",
    summary="채팅 기록 조회",
    description="특정 thread_id의 채팅 기록을 JSON 형태로 반환합니다.",
    response_model=ChatHistory
)
async def get_chat_history(
    thread_id: str = Query(..., description="Thread ID", example="f09d8c6e-fcb5-4275-bf3d-90a87ede2cb8")
):
    return chat_service.get_chat_history(thread_id)


@router.get(
    "/summary",
    summary="특정 채팅의 내화내용 요약 내용",
    description="특정 thread_id의 채팅 기록의 summary가 있다면 이를 반환합니다.",
)
async def get_chat_summary(
    thread_id: str = Query(..., description="Thread ID", example="f09d8c6e-fcb5-4275-bf3d-90a87ede2cb8")
):
    return await chat_service.get_chat_summary(thread_id)
