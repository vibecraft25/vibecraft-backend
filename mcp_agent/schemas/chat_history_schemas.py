__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
from typing import Dict, Any

# Third-party imports
from pydantic import BaseModel


class ChatHistory(BaseModel):
    thread_id: str
    """Thread ID"""
    values: dict[str, Any] | Any
    """Current values of channels."""
    next: tuple[str, ...]
    """The name of the node to execute in each task for this step."""
    config: Dict
    """Config used to fetch this snapshot."""
    metadata: Dict | None
    """Metadata associated with this snapshot."""
    created_at: str | None
    """Timestamp of snapshot creation."""
    parent_config: Dict | None
    """Config used to fetch the parent snapshot, if any."""
