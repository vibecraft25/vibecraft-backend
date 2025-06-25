"""
-Methods-
1. Set topic
2. Get current topic
"""
__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Third-party imports
from mcp.server.fastmcp import FastMCP

# Custom imports
from schemas.topic_schemas import TopicInput

# MCP 서버 인스턴스
mcp = FastMCP()

# 사용자 주제 저장용 전역 변수
user_topic_store = {"topic": None, "description": None}


# MCP Tool: 사용자 주제 설정
@mcp.tool()
def set_topic(data: TopicInput):
    """사용자가 주제를 입력하면 시스템에 저장합니다."""
    user_topic_store["topic"] = data.topic
    user_topic_store["description"] = data.description
    return {
        "status": "success",
        "message": "주제가 성공적으로 설정되었습니다.",
        "stored_topic": user_topic_store
    }

@mcp.tool()
def get_topic():
    """저장된 주제를 반환합니다."""
    if not user_topic_store["topic"]:
        return {
            "status": "empty",
            "message": "아직 주제가 설정되지 않았습니다."
        }
    return {
        "status": "success",
        "topic": user_topic_store["topic"],
        "description": user_topic_store["description"]
    }


if __name__ == "__main__":
    mcp.run()
