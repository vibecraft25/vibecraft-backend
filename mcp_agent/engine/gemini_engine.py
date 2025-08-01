__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
from typing import List, Optional

# Third-party imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import BaseTool

# Custom imports
from .base import BaseEngine


class GeminiEngine(BaseEngine):
    def __init__(self, tools: Optional[List[BaseTool]] = None):
        super().__init__(
            model_cls=ChatGoogleGenerativeAI,
            model_name="gemini-2.5-flash",
            model_kwargs={"temperature": 0},
            tools=tools
        )
