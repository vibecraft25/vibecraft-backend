__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
from typing import List, Optional
from enum import Enum
from dataclasses import dataclass, field


class Transport(Enum):
    stdio = "stdio"
    sse = "sse"


@dataclass
class MCPServerConfig:
    name: str                         # MCP 서버 식별자
    command: str                      # 실행 명령 (예: "npx")
    args: Optional[List[str]] = None  # 명령 인자 리스트
    _transport: Optional[Transport] = field(default=Transport.stdio, repr=False)

    @property
    def transport(self) -> str:
        return self._transport.value if self._transport else None

    @transport.setter
    def transport(self, value):
        if isinstance(value, Transport):
            self._transport = value
        elif isinstance(value, str):
            self._transport = Transport(value)
        else:
            raise ValueError("transport must be a str or Transport enum")


@dataclass
class TopicStepResult:
    topic_prompt: str
    result: str
