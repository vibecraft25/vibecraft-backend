from pydantic import BaseModel
from typing import Optional


class TopicInput(BaseModel):
    topic: str
    description: Optional[str] = None
