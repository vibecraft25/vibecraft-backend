__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
from typing import List, Optional

# Third-party imports
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import BaseTool

# Custom imports
from .base import BaseEngine


class ClaudeEngine(BaseEngine):
    def __init__(self, tools: Optional[List[BaseTool]] = None):
        super().__init__(
            model_cls=ChatAnthropic,
            model_name="claude-3-5-sonnet-20241022",
            model_kwargs={"max_tokens": 1000, "temperature": 0},
            tools=tools
        )
