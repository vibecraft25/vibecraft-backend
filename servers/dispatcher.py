# TODO: WIP
"""
Dispatcher MCP: Coordinates topic and data upload MCP servers.

Features:
- Set topic via topic_server
- Upload file via data_upload_server
- Automatically call data tool after topic setup

This dispatcher assumes topic_server runs on http://localhost:8080/mcp
and data_upload_server runs on http://localhost:8081/mcp
"""

__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
import requests

# Third-party imports
from fastmcp import FastMCP
from pydantic import BaseModel
from typing import Optional

# FastMCP dispatcher instance
mcp = FastMCP("name=dispatcher_mcp")

# Input schemas
class DispatcherTopicInput(BaseModel):
    topic: str
    description: Optional[str] = None

class DispatcherUploadInput(BaseModel):
    filename: str
    content_base64: str


@mcp.tool(
    name="set_topic_and_suggest",
    description="Set topic using topic_server and get suggestion.",
    tags={"dispatcher", "topic", "suggest"}
)
def set_topic_and_suggest(data: DispatcherTopicInput):
    topic_api = "http://localhost:8080/mcp/tool/set_topic"
    suggest_api = "http://localhost:8080/mcp/tool/get_topic_task_hint"

    # Step 1: Set topic
    res1 = requests.post(topic_api, json=data.dict())
    if res1.status_code != 200:
        return {"status": "error", "message": "Failed to set topic.", "detail": res1.text}

    # Step 2: Get task suggestion
    res2 = requests.post(suggest_api)
    if res2.status_code != 200:
        return {"status": "partial", "message": "Topic set, but no task suggestion.", "detail": res2.text}

    return {"status": "success", "topic_response": res1.json(), "suggestion": res2.json()}


@mcp.tool(
    name="upload_file_to_data_server",
    description="Upload base64 file to data_upload_server.",
    tags={"dispatcher", "file", "upload"}
)
def upload_file_to_data_server(data: DispatcherUploadInput):
    upload_api = "http://localhost:8081/mcp/tool/upload_file"
    res = requests.post(upload_api, json=data.dict())
    if res.status_code != 200:
        return {"status": "error", "message": "Upload failed", "detail": res.text}
    return res.json()


@mcp.tool(
    name="list_uploaded_files",
    description="List all uploaded files from data_upload_server.",
    tags={"dispatcher", "file", "list"}
)
def list_uploaded_files():
    list_api = "http://localhost:8081/mcp/resource/files://uploaded"
    res = requests.get(list_api)
    if res.status_code != 200:
        return {"status": "error", "message": "Could not fetch uploaded files", "detail": res.text}
    return res.json()


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8089, path="/mcp")
