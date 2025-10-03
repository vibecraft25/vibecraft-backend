__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Third-party imports
from pydantic import BaseModel


class ChatResponse(BaseModel):
    data: str
    thread_id: str
