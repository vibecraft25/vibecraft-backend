__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Third-party imports
from langchain.chat_models import ChatOpenAI

# Custom imports
from .base import BaseEngine


class OpenAIEngine(BaseEngine):
    def __init__(self):
        super().__init__(
            model_cls=ChatOpenAI,
            model_name="gpt-4.1",
            model_kwargs={"temperature": 0}
        )
