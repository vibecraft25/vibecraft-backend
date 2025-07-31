__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
import asyncio
from typing import Optional, AsyncGenerator

# Third-party imports
from sse_starlette.sse import ServerSentEvent
from starlette.responses import JSONResponse

# Custom imports
from client.vibe_craft_client import VibeCraftClient
from schemas import ChatResponse


class ChatService:
    """채팅 관련 비즈니스 로직을 처리하는 서비스 클래스"""

    def __init__(self, engine: str = "gemini"):
        self.engine = engine

    async def _create_client(self, thread_id: Optional[str] = None) -> VibeCraftClient:
        """클라이언트 생성 및 초기화"""
        client = VibeCraftClient(self.engine)
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

        try:
            async for event, chunk in client.execute_stream_step(query, use_langchain=use_langchain):
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


# 싱글톤 인스턴스
chat_service = ChatService()
