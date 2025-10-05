__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
from typing import Optional

# Third-party imports
from fastapi import APIRouter
from fastapi.params import Query
from sse_starlette.sse import EventSourceResponse

# Custom imports
from schemas import SSEStreamDocumentation
from services.business.workflow_service import workflow_service

prefix = "workflow"
router = APIRouter(
    prefix=f"/{prefix}",
    responses={401: {"description": "raw data file upload"}}
)


@router.get(
    "/stream/topic",
    summary="워크플로우: 주제 설정",
    responses=SSEStreamDocumentation.get_workflow_stream_responses()
)
async def stream_set_topic(
        query: str = Query(..., description="Prompt Query"),
):
    """
    워크플로우 1단계: 주제를 설정합니다.

    **이벤트 타입:**
    - `ai`: AI가 주제를 분석하고 설정하는 과정
    - `tool`: MCP 서버 도구 사용 (필요시)
    - `menu`: 다음 단계 선택 메뉴
    - `complete`: 주제 설정 완료 (thread_id 반환)
    - `error`: 에러 발생
    """
    return EventSourceResponse(
        workflow_service.execute_topic_selection(query)
    )


@router.get(
    "/stream/run",
    summary="워크플로우: 데이터 분석 및 코드 생성 실행",
    responses=SSEStreamDocumentation.get_workflow_stream_responses()
)
async def stream_run_workflow(
        thread_id: str = Query(..., description="Thread ID", example="f09d8c6e-fcb5-4275-bf3d-90a87ede2cb8"),
        code: str = Query(None, description="Use `/contents/upload` api to upload file and get code",
                                    example="f09d8c6e"),
):
    return EventSourceResponse(
        workflow_service.execute_run_workflow(thread_id, code)
    )
