__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Third-party imports
from langchain_anthropic import ChatAnthropic


# Custom imports
from .base import BaseEngine


class ClaudeEngine(BaseEngine):
    def __init__(self):
        super().__init__(
            model_cls=ChatAnthropic,
            model_name="claude-3-5-sonnet-20241022",
            model_kwargs={"max_tokens": 1000, "temperature": 0}
        )
