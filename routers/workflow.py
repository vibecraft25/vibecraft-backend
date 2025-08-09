__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
from typing import Optional

# Third-party imports
from fastapi import APIRouter
from fastapi.params import Query
from sse_starlette.sse import EventSourceResponse

# Custom imports
from schemas import SSEStreamDocumentation
from mcp_agent.schemas import (
    VisualizationRecommendationResponse,
    VisualizationType
)
from services.workflow_service import workflow_service

prefix = "workflow"
router = APIRouter(
    prefix=f"/{prefix}",
    responses={401: {"description": "raw data file upload"}}
)


@router.get(
    "/stream/set-topic",
    summary="워크플로우 1단계: 주제 설정",
    description=(
            "**Menu call api logic:**\n\n"
            "- `1`: call `/workflow/stream/set-data`\n"
            "- `2`: call `/chat/stream/load-chat`\n"
            "- `3`: call `/workflow/stream/set-topic`\n"
    ),
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
    "/stream/set-data",
    summary="워크플로우 2-1단계: 데이터 로드 or 생성",
    description=(
            "**Menu call api logic:**\n\n"
            "- `1`: call `/workflow/stream/process-data-selection`\n"
            "- `2`: call `/workflow/stream/process-data-selection`\n"
            "- `3`: call `/workflow/visualization-type or /workflow/code-generator`\n"
    ),
    responses=SSEStreamDocumentation.get_workflow_stream_responses()
)
async def set_data(
        thread_id: str = Query(..., description="Thread ID", example="f09d8c6e-fcb5-4275-bf3d-90a87ede2cb8"),
        code: Optional[str] = Query(None, description="Use `/contents/upload` api to upload file and get code",
                                    example="f09d8c6e"),
):
    """
    워크플로우 2-1단계: 데이터를 로드하거나 생성합니다.

    **이벤트 타입:**
    - `ai`: 데이터 생성 과정 (샘플 데이터 생성 시)
    - `data`: 데이터 정보 (컬럼 목록, 미리보기 등)
    - `menu`: 데이터 처리 옵션 메뉴
    - `complete`: 데이터 설정 완료 (thread_id 반환)
    - `error`: 에러 발생 (파일 없음 등)
    """
    return EventSourceResponse(
        workflow_service.execute_set_data(thread_id, code)
    )


@router.get(
    "/stream/process-data-selection",
    summary="워크플로우 2-2단계: 데이터 선택 처리",
    description=(
            "**Menu call api logic:**\n\n"
            "- `1`: call `/workflow/stream/process-data-selection`\n"
            "- `2`: call `/workflow/visualization-type or /workflow/code-generator`\n"
    ),
    responses=SSEStreamDocumentation.get_workflow_stream_responses()
)
async def process_data_selection(
        thread_id: str = Query(..., description="Thread ID", example="f09d8c6e-fcb5-4275-bf3d-90a87ede2cb8"),
        query: str = Query(..., description="데이터 컬럼명"),
):
    """
    워크플로우 2-2단계: 사용자가 선택한 데이터 컬럼을 처리합니다.

    **이벤트 타입:**
    - `ai`: 컬럼명 매칭 및 처리 과정
    - `data`: 처리된 데이터 정보
    - `menu`: 추가 처리 옵션 메뉴
    - `complete`: 데이터 처리 완료 (thread_id 반환)
    - `error`: 에러 발생 (데이터 파일 없음, 컬럼 매칭 실패 등)
    """
    return EventSourceResponse(
        workflow_service.execute_data_selection_processing(thread_id, query)
    )


@router.get(
    "/visualization-type",
    summary="[Optional] 워크플로우 2-3단계: 시각화 방식 추천",
    description="",
    response_model=VisualizationRecommendationResponse
)
async def get_visualization_type(
        thread_id: str = Query(..., description="Thread ID", example="f09d8c6e-fcb5-4275-bf3d-90a87ede2cb8"),
):
    return await workflow_service.execute_recommend_visualization_type(thread_id)


@router.get(
    "/stream/code-generator",
    summary="워크플로우 3단계: 코드 생성",
    description="워크플로우의 마지막 단계로 코드를 생성합니다. (WIP)",
    responses=SSEStreamDocumentation.get_workflow_stream_responses()
)
async def generate_code(
        thread_id: str = Query(..., description="Thread ID", example="f09d8c6e-fcb5-4275-bf3d-90a87ede2cb8"),
        visualization_type: VisualizationType = Query(..., description="Visualization Type"),
        project_name: Optional[str] = Query(None, description="Project name for the generated app"),
        model: str = Query("flash", description="Gemini model to use: flash (default) or pro"),
):
    """
    워크플로우 3단계: 웹앱 코드를 생성합니다.
    """
    return EventSourceResponse(
        workflow_service.execute_code_generator(thread_id, visualization_type, project_name, model)
    )
