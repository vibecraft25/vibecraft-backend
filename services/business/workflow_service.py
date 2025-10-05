__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
from typing import AsyncGenerator, Optional

# Third-party imports
from sse_starlette.sse import ServerSentEvent

# Custom imports
from services import BaseStreamService
from utils import PathUtils, ContentUtils


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

    async def execute_run_workflow(
            self,
            thread_id: str,
            code: str
    ) -> AsyncGenerator[ServerSentEvent, None]:
        """워크플로우 실행"""
        client = self._create_client()
        client.merge_chat_history(thread_id)

        file_path = None
        if code is not None:
            for ext in ContentUtils.ALLOWED_EXTENSIONS:
                path = PathUtils.get_path(thread_id, f"{code}{ext}")
                if path:
                    file_path = path[0]
                    break

        # stream_set_data는 generator이므로 async for로 순회
        async def generator():
            async for msg in client.stream_set_data(file_path):
                yield msg
            async for msg in client.stream_run_workflow():
                yield msg

        async for event in self._create_workflow_stream_generator(
                generator,
                lambda: client.get_thread_id()
        ):
            yield event


# 싱글톤 인스턴스
workflow_service = WorkflowService()
