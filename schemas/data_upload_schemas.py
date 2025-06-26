from pydantic import BaseModel
from typing import List, Dict, Any, Literal


class UploadData(BaseModel):
    filename: str  # without extension
    records: List[Dict[str, Any]]  # list of records in JSON format


class FileUpload(BaseModel):
    filename: str  # including extension (e.g., data.csv)
    filetype: Literal["csv", "sqlite"]
    # filetype: Literal["csv", "sqlite", "pdf", "docx"] # WIP
    content_base64: str  # base64-encoded file content
