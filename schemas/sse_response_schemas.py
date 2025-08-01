__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
from enum import Enum
from typing import Any, Union, Literal

# Third-party imports
from pydantic import BaseModel, Field
from sse_starlette.sse import ServerSentEvent


class SSEEventType(Enum):
    """SSE 이벤트 타입 정의"""
    # 도구 관련
    TOOL = "tool"
    # 메시지 관련
    AI_MESSAGE_CHUNK = "ai"
    # 메뉴 관련
    MENU = "menu"
    # 완료 관련
    COMPLETE = "complete"
    # 데이터 관련
    DATA = "data"
    # 에러 관련
    ERROR = "error"
    # Others
    UNDEFINED = "undefined"


class SSEEventBuilder:
    """SSE 이벤트 생성 헬퍼 클래스"""

    @staticmethod
    def create_tool_event(message: str) -> ServerSentEvent:
        """도구 사용 이벤트"""
        return ServerSentEvent(
            event=SSEEventType.TOOL.value,
            data=message
        )

    @staticmethod
    def create_ai_message_chunk(content: str) -> ServerSentEvent:
        """AI 메시지 청크 이벤트"""
        return ServerSentEvent(
            event=SSEEventType.AI_MESSAGE_CHUNK.value,
            data=content
        )

    @staticmethod
    def create_menu_event(menu_options: str) -> ServerSentEvent:
        """메뉴 표시 이벤트"""
        return ServerSentEvent(
            event=SSEEventType.MENU.value,
            data=menu_options
        )

    @staticmethod
    def create_complete_event(thread_id: str) -> ServerSentEvent:
        """완료 이벤트"""
        return ServerSentEvent(
            event=SSEEventType.COMPLETE.value,
            data=thread_id
        )

    @staticmethod
    def create_data_event(content: Any = None) -> ServerSentEvent:
        """데이터 이벤트"""
        return ServerSentEvent(
            event=SSEEventType.DATA.value,
            data=str(content)
        )

    @staticmethod
    def create_error_event(error_message: str) -> ServerSentEvent:
        """에러 이벤트"""
        return ServerSentEvent(
            event=SSEEventType.ERROR.value,
            data=f"❗ 오류 발생: {error_message}"
        )

    @staticmethod
    def create_undefined_event(content: Any = None) -> ServerSentEvent:
        """정의되지 않은 이벤트"""
        return ServerSentEvent(
            event=SSEEventType.UNDEFINED.value,
            data=str(content)
        )


# Swagger 문서화를 위한 SSE 응답 모델들
class SSEEventModel(BaseModel):
    """SSE 이벤트 기본 모델"""
    event: str = Field(..., description="이벤트 타입")
    data: str = Field(..., description="이벤트 데이터")

    class Config:
        json_schema_extra = {
            "example": {
                "event": "ai",
                "data": "안녕하세요! 어떤 도움이 필요하신가요?"
            }
        }


class ToolEventModel(SSEEventModel):
    """도구 사용 이벤트 모델"""
    event: Literal["tool"] = Field("tool", description="도구 이벤트")
    data: str = Field(..., description="MCP 서버 사용 결과", example="파일을 성공적으로 업로드했습니다.")


class AIMessageEventModel(SSEEventModel):
    """AI 메시지 이벤트 모델"""
    event: Literal["ai"] = Field("ai", description="AI 응답 이벤트")
    data: str = Field(..., description="AI 응답 내용", example="안녕하세요! 어떤 도움이 필요하신가요?")


class MenuEventModel(SSEEventModel):
    """메뉴 이벤트 모델"""
    event: Literal["menu"] = Field("menu", description="메뉴 선택 이벤트")
    data: str = Field(..., description="선택 가능한 메뉴 옵션", example="1. 데이터 로드\n2. 추가 수정\n3. 새 주제")


class DataEventModel(SSEEventModel):
    """데이터 이벤트 모델"""
    event: Literal["data"] = Field("data", description="데이터 정보 이벤트")
    data: str = Field(..., description="데이터 내용", example="컬럼 목록: name, age, city")


class CompleteEventModel(SSEEventModel):
    """완료 이벤트 모델"""
    event: Literal["complete"] = Field("complete", description="작업 완료 이벤트")
    data: str = Field(..., description="워크플로우 Thread ID", example="f09d8c6e-fcb5-4275-bf3d-90a87ede2cb8")


class ErrorEventModel(SSEEventModel):
    """에러 이벤트 모델"""
    event: Literal["error"] = Field("error", description="에러 이벤트")
    data: str = Field(..., description="에러 메시지", example="❗ 오류 발생: 파일을 찾을 수 없습니다.")


class UndefinedEventModel(SSEEventModel):
    """정의되지 않은 이벤트 모델"""
    event: Literal["undefined"] = Field("undefined", description="정의되지 않은 이벤트")
    data: str = Field(..., description="기타 데이터")


# SSE 스트림 응답을 위한 통합 모델
SSEEventResponse = Union[
    ToolEventModel,
    AIMessageEventModel,
    MenuEventModel,
    DataEventModel,
    CompleteEventModel,
    ErrorEventModel,
    UndefinedEventModel
]


class SSEStreamDocumentation:
    """SSE 스트림 응답 문서화를 위한 클래스"""

    @staticmethod
    def get_chat_stream_responses():
        """채팅 스트림 응답 예시"""
        return {
            200: {
                "description": "SSE 스트림 응답",
                "content": {
                    "text/event-stream": {
                        "schema": {
                            "type": "string",
                            "format": "binary"
                        },
                        "examples": {
                            "ai_response": {
                                "summary": "AI 응답 이벤트",
                                "value": "event: ai\ndata: 안녕하세요! 무엇을 도와드릴까요?\n\n"
                            },
                            "tool_usage": {
                                "summary": "도구 사용 이벤트",
                                "value": "event: tool\ndata: 파일 업로드를 시작합니다.\n\n"
                            },
                            "completion": {
                                "summary": "완료 이벤트",
                                "value": "event: complete\ndata: f09d8c6e-fcb5-4275-bf3d-90a87ede2cb8\n\n"
                            }
                        }
                    }
                }
            }
        }

    @staticmethod
    def get_workflow_stream_responses():
        """워크플로우 스트림 응답 예시"""
        return {
            200: {
                "description": "워크플로우 SSE 스트림 응답",
                "content": {
                    "text/event-stream": {
                        "schema": {
                            "type": "string",
                            "format": "binary"
                        },
                        "examples": {
                            "topic_selection": {
                                "summary": "주제 설정 과정",
                                "value": "event: ai\ndata: 주제를 분석하고 있습니다.\n\nevent: menu\ndata: 1. 데이터 로드\\n2. 추가 수정\\n3. 새 주제\n\nevent: complete\ndata: thread-id-123\n\n"
                            },
                            "data_processing": {
                                "summary": "데이터 처리 과정",
                                "value": "event: data\ndata: 컬럼 목록: name, age, city\n\nevent: menu\ndata: 1. 추천 컬럼 삭제\\n2. 직접 선택\n\nevent: complete\ndata: thread-id-123\n\n"
                            },
                            "error_case": {
                                "summary": "에러 발생",
                                "value": "event: error\ndata: ❗ 오류 발생: 데이터 파일을 찾을 수 없습니다.\n\n"
                            }
                        }
                    }
                }
            }
        }
