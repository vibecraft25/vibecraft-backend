__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
from dataclasses import dataclass


@dataclass
class MCPServerConfig:
    path: str            # MCP server path
    command: str         # e.g.) "npx", "python", ...


@dataclass
class TopicStepResult:
    topic_prompt: str
    result: str
