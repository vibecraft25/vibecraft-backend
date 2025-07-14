__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
from typing import List
from dataclasses import dataclass


@dataclass
class MCPServerConfig:
    name: str               # MCP 서버 식별자
    command: str            # 실행 명령 (예: "npx")
    args: List[str] = None  # 명령 인자 리스트


@dataclass
class TopicStepResult:
    topic_prompt: str
    result: str
