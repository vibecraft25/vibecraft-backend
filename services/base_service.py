__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
import asyncio
from typing import AsyncGenerator, Callable, Optional

# Third-party imports
from sse_starlette.sse import ServerSentEvent

# Custom imports
from schemas import SSEEventType, SSEEventBuilder


class BaseStreamService:
    """스트림 처리 관련 공통 로직을 제공하는 기본 서비스 클래스"""

    def __init__(self, engine: str = "gemini"):
        self.engine = engine

    def _create_client(self):
        """클라이언트 생성 (하위 클래스에서 구체적으로 구현)"""
        from mcp_agent.client import VibeCraftClient
        return VibeCraftClient(self.engine)

    async def _create_workflow_stream_generator(
            self,
            generator_func: Callable[[], AsyncGenerator[ServerSentEvent, None]],
            thread_id_getter: Optional[Callable[[], str]] = None
    ) -> AsyncGenerator[ServerSentEvent, None]:
        """공통 스트림 생성기 (에러 처리 포함)"""
        try:
            async for msg in generator_func():
                if msg:
                    yield msg
                    await asyncio.sleep(0.1)

            if thread_id_getter:
                yield SSEEventBuilder.create_complete_event(thread_id_getter())

        except Exception as e:
            yield SSEEventBuilder.create_error_event(str(e))

    async def _create_chat_stream_generator(
            self,
            client,
            query: str,
            use_langchain: bool = True
    ) -> AsyncGenerator[ServerSentEvent, None]:
        """채팅 스트림 생성기 (에러 처리 포함)"""
        try:
            async for event, chunk in client.execute_stream_step(query, use_langchain=use_langchain):
                # ServerSentEvent가 직접 들어오는 경우 바로 yield
                if isinstance(chunk, ServerSentEvent):
                    yield chunk
                else:
                    yield self._create_event_by_type(event, chunk)
            yield SSEEventBuilder.create_complete_event(client.get_thread_id())

        except Exception as e:
            yield SSEEventBuilder.create_error_event(str(e))

    def _create_event_by_type(self, event: str, chunk) -> ServerSentEvent:
        """이벤트 타입에 따른 SSE 이벤트 생성"""
        # 이미 ServerSentEvent 객체인 경우 바로 반환
        if isinstance(chunk, ServerSentEvent):
            return chunk

        # 문자열인 경우 이벤트 타입에 따라 적절한 SSE 이벤트 생성
        if event == SSEEventType.TOOL.value:
            return SSEEventBuilder.create_tool_event(chunk)
        elif event == SSEEventType.AI_MESSAGE_CHUNK.value:
            return SSEEventBuilder.create_ai_message_chunk(chunk)
        elif event == SSEEventType.MENU.value:
            return SSEEventBuilder.create_menu_event(chunk)
        elif event == SSEEventType.DATA.value:
            return SSEEventBuilder.create_data_event(chunk)
        else:
            return SSEEventBuilder.create_undefined_event(chunk)
