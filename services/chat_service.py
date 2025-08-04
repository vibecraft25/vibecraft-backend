__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
from typing import Optional, AsyncGenerator

# Third-party imports
from sse_starlette.sse import ServerSentEvent
from starlette.responses import JSONResponse

# Custom imports
from mcp_agent.engine import BaseEngine
from mcp_agent.client import VibeCraftClient
from schemas import ChatResponse
from services import BaseStreamService


class ChatService(BaseStreamService):
    """채팅 관련 비즈니스 로직을 처리하는 서비스 클래스"""

    async def _create_client(self, thread_id: Optional[str] = None) -> VibeCraftClient:
        """클라이언트 생성 및 초기화 (채팅용 - 도구 로드 포함)"""
        client = super()._create_client()
        await client.load_tools()

        if thread_id:
            client.load_chat_history(thread_id)

        return client

    async def execute_chat(
            self,
            query: str,
            use_langchain: bool = True,
            thread_id: Optional[str] = None
    ) -> JSONResponse:
        """일반 채팅 실행"""
        client = await self._create_client(thread_id)
        response = await client.execute_step(query, use_langchain=use_langchain)

        return JSONResponse(
            content=ChatResponse(
                data=response,
                thread_id=client.get_thread_id(),
            ).model_dump(),
            status_code=200
        )

    async def execute_stream_chat(
            self,
            query: str,
            use_langchain: bool = True,
            thread_id: Optional[str] = None
    ) -> AsyncGenerator[ServerSentEvent, None]:
        """스트리밍 채팅 실행"""
        client = await self._create_client(thread_id)

        # 부모 클래스의 공통 스트림 생성기 사용
        async for event in self._create_chat_stream_generator(client, query, use_langchain):
            yield event

    @staticmethod
    def get_chat_history(thread_id: str) -> JSONResponse:
        """특정 thread_id의 채팅 기록을 JSONResponse로 반환"""
        chat_history = BaseEngine.load_chat_history_file(thread_id)

        if chat_history:
            return JSONResponse(content=chat_history.model_dump(), status_code=200)
        else:
            return JSONResponse(
                content={"error": f"채팅 기록을 찾을 수 없습니다. Thread ID: {thread_id}"},
                status_code=404
            )

    async def get_chat_summary(self, thread_id: str) -> JSONResponse:
        """특정 thread_id의 채팅 기록을 JSONResponse로 반환"""
        client = await self._create_client(thread_id)
        chat_summary = client.get_summary()

        if chat_summary:
            return JSONResponse(content=chat_summary, status_code=200)
        else:
            return JSONResponse(
                content={"error": f"채팅 기록을 찾을 수 없습니다. Thread ID: {thread_id}"},
                status_code=404
            )


# 싱글톤 인스턴스
chat_service = ChatService()
