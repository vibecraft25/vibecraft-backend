__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

from typing import Optional

from fastapi import APIRouter
from fastapi.params import Query
from sse_starlette.sse import EventSourceResponse

from services.workflow_service import workflow_service

prefix = "workflow"
router = APIRouter(
    prefix=f"/{prefix}", 
    responses={401: {"description": "raw data file upload"}}
)


@router.get(
    "/stream/set-topic",
    summary="워크플로우 1단계: 주제 설정",
    description="워크플로우의 첫 번째 단계로 주제를 설정합니다."
)
async def stream_set_topic(
    query: str = Query(..., description="Prompt Query"),
):
    return EventSourceResponse(
        workflow_service.execute_topic_selection(query)
    )


@router.get(
    "/stream/set-data",
    summary="워크플로우 2-1단계: 데이터 로드",
    description="지정된 스레드에서 데이터를 로드하거나 생성합니다."
)
async def set_data(
    thread_id: str = Query(..., description="Thread ID", example="f09d8c6e-fcb5-4275-bf3d-90a87ede2cb8"),
    code: Optional[str] = Query(None, description="upload file code", example="f09d8c6e"),
):
    return EventSourceResponse(
        workflow_service.execute_set_data(thread_id, code)
    )


@router.post(
    "/stream/process-data-selection",
    summary="워크플로우 2-2단계: 데이터 선택 처리",
    description="사용자가 선택한 데이터를 처리합니다. (loop)"
)
async def process_data_selection(
    thread_id: str = Query(..., description="Thread ID", example="f09d8c6e-fcb5-4275-bf3d-90a87ede2cb8"),
    query: str = Query(..., description="데이터 컬럼명)"),
):
    return EventSourceResponse(
        workflow_service.execute_data_selection_processing(thread_id, query)
    )


# TODO: WIP
@router.get(
    "/code-generator",
    summary="워크플로우 3단계: 코드 생성",
    description="워크플로우의 마지막 단계로 코드를 생성합니다. (WIP)"
)
async def generate_code(
    query: str = Query(..., description="Prompt Query"),
    thread_id: str = Query(..., description="Thread ID", example="f09d8c6e-fcb5-4275-bf3d-90a87ede2cb8"),
):
    # WIP: 현재는 클라이언트 설정만 수행
    client = workflow_service.setup_code_generation(query, thread_id)
    # TODO: 실제 코드 생성 로직 구현 필요
    return {"message": "Code generation setup completed", "thread_id": client.get_thread_id()}
