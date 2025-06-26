"""
MCP Topic Management Server

- Features:
  1. Set topic (set_topic)
  2. Get current topic (get_topic)
  3. Reset topic (reset_topic)
  4. Suggest task based on topic (get_topic_task_hint)
  5. Retrieve MCP server info (get_mcp_info)

- Extensibility:
  - Language-agnostic API design with clearly defined input/output schema
  - Can be integrated into web or CLI-based tools
"""

__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
import os
import sys
from typing import Optional, Dict, List

# Ensure project root on path for importing schemas
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Third-party imports
from fastmcp import FastMCP

# Custom imports
from schemas.topic_schemas import TopicInput

# Instantiate MCP
mcp = FastMCP("name=vibecraft_topic")

# Global in-memory storage and topic change history
user_topic_store: Dict[str, Optional[str]] = {
    "topic": None,
    "description": None
}
topic_history: List[Dict[str, Optional[str]]] = []

# Topic validation function
def validate_topic(topic: str) -> bool:
    return 2 < len(topic.strip()) < 100

# 1. Set Topic
@mcp.tool(
    name="set_topic",
    description="Set the current topic and optional description. Topic must be between 3 and 100 characters.",
    tags={"topic", "주제", "set"}
)
def set_topic(data: TopicInput):
    """Stores the user-defined topic in the system."""
    if not validate_topic(data.topic):
        return {
            "status": "error",
            "message": "Topic must be between 3 and 100 characters."
        }

    user_topic_store["topic"] = data.topic
    user_topic_store["description"] = data.description
    topic_history.append(user_topic_store.copy())

    return {
        "status": "success",
        "message": "Topic has been successfully set.",
        "stored_topic": user_topic_store
    }

# 2. Get Topic
@mcp.tool(
    name="get_topic",
    description="Retrieve the currently stored topic and its description.",
    tags={"topic", "주제", "get"}
)
def get_topic():
    """Returns the currently stored topic."""
    if not user_topic_store["topic"]:
        return {
            "status": "empty",
            "message": "No topic has been set yet."
        }

    return {
        "status": "success",
        "topic": user_topic_store["topic"],
        "description": user_topic_store["description"]
    }

# 3. Reset Topic
@mcp.tool(
    name="reset_topic",
    description="Reset the stored topic and description to None.",
    tags={"topic", "주제", "reset"}
)
def reset_topic():
    """Clears the stored topic information."""
    user_topic_store["topic"] = None
    user_topic_store["description"] = None
    return {
        "status": "success",
        "message": "Topic has been reset."
    }

# 4. Suggest Task Based on Topic
@mcp.tool(
    name="get_topic_task_hint",
    description="Suggest a task category based on the current topic content.",
    tags={"topic", "주제", "task", "suggest"}
)
def get_topic_task_hint():
    """Suggests a task category based on the topic content."""
    topic = user_topic_store.get("topic")
    if not topic:
        return {"status": "error", "message": "Please set a topic first."}

    if "리팩토링" in topic or "코드" in topic:
        task = "Code analysis and refactoring"
    elif "웹페이지" in topic or "시각화" in topic:
        task = "Data-driven web content generation"
    else:
        task = "Data collection or modeling based on topic"

    return {
        "status": "success",
        "suggested_task": task
    }

# 5. Get MCP Server Info
@mcp.tool(
    name="get_mcp_info",
    description="Retrieve server metadata including tool list and current topic.",
    tags={"mcp", "info"}
)
async def get_mcp_info():
    """Returns metadata about the MCP server instance."""
    tools = await mcp.get_tools()
    return {
        "mcp_name": mcp.name,
        "available_tools": list(tools.keys()),
        "current_topic": user_topic_store.get("topic")
    }

# Run the MCP server
if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8080, path="/mcp")
