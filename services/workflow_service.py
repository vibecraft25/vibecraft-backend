__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
import json
from typing import AsyncGenerator, Optional

# Third-party imports
from sse_starlette.sse import ServerSentEvent

# Custom imports
from schemas import SSEEventBuilder
from mcp_agent.schemas import (
    VisualizationType,
    VisualizationRecommendationResponse
)
from services import BaseStreamService
from utils import PathUtils, ContentUtils
from exceptions import NotFoundException


class WorkflowService(BaseStreamService):
    """워크플로우 관련 비즈니스 로직을 처리하는 서비스 클래스"""

    async def execute_topic_selection(self, query: str) -> AsyncGenerator[ServerSentEvent, None]:
        """워크플로우 1단계: 주제 설정"""
        client = self._create_client()

        async def generator():
            async for msg in client.stream_topic_selection(query):
                yield msg

        async for event in self._create_workflow_stream_generator(
                generator,
                lambda: client.get_thread_id()
        ):
            yield event

    async def execute_set_data(
            self,
            thread_id: str,
            code: Optional[str] = None
    ) -> AsyncGenerator[ServerSentEvent, None]:
        """워크플로우 2단계: 데이터 로드"""
        client = self._create_client()
        client.merge_chat_history(thread_id)

        file_path = None
        if code is not None:
            for ext in ContentUtils.ALLOWED_EXTENSIONS:
                path = PathUtils.get_path(thread_id, f"{code}{ext}")
                if path:
                    file_path = path
                    break
        await client.set_data(file_path)

        async def generator():
            async for msg in client.stream_data_processing():
                yield msg

        async for event in self._create_workflow_stream_generator(
                generator,
                lambda: client.get_thread_id()
        ):
            yield event

    async def execute_data_selection_processing(
            self,
            thread_id: str,
            query: str
    ) -> AsyncGenerator[ServerSentEvent, None]:
        """데이터 선택 처리"""
        client = self._create_client()
        client.load_chat_history(thread_id)

        if PathUtils.is_exist(thread_id, f"{thread_id}.sqlite"):
            sqlite_path = PathUtils.get_path(thread_id, f"{thread_id}.sqlite")
            await client.set_data(sqlite_path[0])

            meta = None
            if PathUtils.is_exist(thread_id, f"{thread_id}_meta.json"):
                sqlite_meta_path = PathUtils.get_path(thread_id, f"{thread_id}_meta.json")
                with open(sqlite_meta_path[0], "r", encoding="utf-8") as f:
                    meta = json.load(f)

            async for msg in client.stream_data_handler(query=query, meta=meta):
                yield msg
        else:
            async def error_generator():
                yield SSEEventBuilder.create_error_event(
                    "데이터 파일을 찾을 수 없습니다. 워크플로우 2단계(데이터 로드)를 먼저 완료해주세요."
                )

            async for event in self._create_workflow_stream_generator(
                    error_generator,
                    lambda: thread_id
            ):
                yield event

    async def execute_recommend_visualization_type(
            self,
            thread_id: str,
            file_format: Optional[str] = "csv"
    ) -> VisualizationRecommendationResponse:
        """데이터 선택 처리"""
        client = self._create_client()
        client.load_chat_history(thread_id)

        if PathUtils.is_exist(thread_id, f"{thread_id}.{file_format}"):
            csv_path = PathUtils.get_path(thread_id, f"{thread_id}.{file_format}")
            client.upload_data(csv_path[0])
            return await client.recommend_visualization_type()
        raise NotFoundException(detail=f"Resource Not Found: {thread_id}.{file_format}")

    async def execute_code_generator(
            self, thread_id: str, visualization_type: VisualizationType,
            project_name: str = None, model: str = "flash"
    ) -> AsyncGenerator[ServerSentEvent, None]:
        """워크플로우 3단계: 코드 생성 실행"""
        client = self._create_client()
        client.load_chat_history(thread_id)

        async def generator():
            async for msg in client.stream_run_code_generator(
                thread_id, visualization_type, project_name, model
            ):
                yield msg

        async for event in self._create_workflow_stream_generator(
                generator,
                lambda: client.get_thread_id()
        ):
            yield event


# 싱글톤 인스턴스
workflow_service = WorkflowService()
