__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
from typing import List, Optional

# Third-party imports
from langchain_community.chat_models import ChatOpenAI
from langchain_core.tools import BaseTool

# Custom imports
from .base import BaseEngine


class OpenAIEngine(BaseEngine):
    def __init__(self, tools: Optional[List[BaseTool]] = None):
        super().__init__(
            model_cls=ChatOpenAI,
            model_name="gpt-4.1",
            model_kwargs={"temperature": 0},
            tools=tools
        )
