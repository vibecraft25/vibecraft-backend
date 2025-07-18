__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Third-party imports
from langchain_google_genai import ChatGoogleGenerativeAI

# Custom imports
from .base import BaseEngine


class GeminiEngine(BaseEngine):
    def __init__(self):
        super().__init__(
            model_cls=ChatGoogleGenerativeAI,
            model_name="gemini-2.5-flash",
            model_kwargs={"temperature": 0}
        )
