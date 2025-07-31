__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
import json
import asyncio
from typing import Optional, AsyncGenerator

# Third-party imports
from sse_starlette.sse import ServerSentEvent

# Custom imports
from client.vibe_craft_client import VibeCraftClient
from utils import PathUtils


class WorkflowService:
    """워크플로우 관련 비즈니스 로직을 처리하는 서비스 클래스"""

    def __init__(self, engine: str = "gemini"):
        self.engine = engine

    def _create_client(self) -> VibeCraftClient:
        """클라이언트 생성"""
        return VibeCraftClient(self.engine)

    async def _create_error_stream_generator(
        self,
        generator_func,
        thread_id_getter=None
    ) -> AsyncGenerator[ServerSentEvent, None]:
        """공통 스트림 생성기 (에러 처리 포함)"""
        try:
            async for msg in generator_func():
                if msg:
                    yield msg
                    await asyncio.sleep(0.1)

            if thread_id_getter:
                yield ServerSentEvent(
                    event="complete",
                    data=thread_id_getter()
                )

        except Exception as e:
            yield ServerSentEvent(
                event="error",
                data=f"data: ❗ 오류 발생: {str(e)}"
            )

    async def _create_chat_stream_generator(
        self,
        client: VibeCraftClient,
        query: str
    ) -> AsyncGenerator[ServerSentEvent, None]:
        """채팅 스트림 생성기 (에러 처리 포함)"""
        try:
            async for event, chunk in client.execute_stream_step(query):
                if chunk:
                    yield ServerSentEvent(
                        event=event or "progress",
                        data=f"{chunk}"
                    )
                    await asyncio.sleep(0.1)

            yield ServerSentEvent(
                event="complete",
                data=client.get_thread_id()
            )

        except Exception as e:
            yield ServerSentEvent(
                event="error",
                data=f"data: ❗ 오류 발생: {str(e)}"
            )

    async def execute_topic_selection(self, query: str) -> AsyncGenerator[ServerSentEvent, None]:
        """워크플로우 1단계: 주제 설정"""
        client = self._create_client()

        async def generator():
            async for msg in client.stream_topic_selection(query):
                yield msg

        async for event in self._create_error_stream_generator(
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
            file_path = PathUtils.get_path(thread_id, code)
        await client.set_data(file_path)

        async def generator():
            async for msg in client.stream_data_processing():
                yield msg

        async for event in self._create_error_stream_generator(
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
                yield ServerSentEvent(
                    event="error",
                    data="❗ 데이터 파일을 찾을 수 없습니다. 워크플로우 2단계(데이터 로드)를 먼저 완료해주세요."
                )

            async for event in self._create_error_stream_generator(
                    error_generator,
                    lambda: thread_id
            ):
                yield event

    # TODO: WIP
    def setup_code_generation(self, query: str, thread_id: str) -> VibeCraftClient:
        """워크플로우 3단계: 코드 생성 설정 (WIP)"""
        client = self._create_client()
        client.merge_chat_history(thread_id)
        return client


# 싱글톤 인스턴스
workflow_service = WorkflowService()
