"""
-Methods-
1. Search data from web
2. Load custom data from user
3. Show data status
4. Edit data columns
"""
__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
import os
import uuid

# Third-party imports
from mcp.server.fastmcp import FastMCP
import pandas as pd
import requests
from bs4 import BeautifulSoup

from clients.topic import user_topic_store

# MCP 인스턴스 생성
mcp = FastMCP()


# TODO: WIP
@mcp.tool()
def search_data_from_web():
    topic = user_topic_store.get("topic")

    if input.upload_path:
        try:
            df = pd.read_csv(input.upload_path)
            source = "user_upload"
        except Exception as e:
            return {
                "status": "error",
                "message": f"[CSV 오류] {str(e)}"
            }

    elif topic:
        try:
            df = auto_collect_data_from_topic(topic)
            source = "auto_collected"
        except Exception as e:
            return {
                "status": "error",
                "message": f"[자동 수집 오류] {str(e)}"
            }

    else:
        return {
            "status": "error",
            "message": "주제가 설정되지 않았고, 업로드된 파일도 없습니다."
        }

    # CSV 저장
    csv_path = os.path.join(OUTPUT_DIR, f"{uuid.uuid4().hex}_data.csv")
    df.to_csv(csv_path, index=False)

    return {
        "status": "success",
        "source": source,
        "topic": topic if topic else "N/A",
        "csv_path": csv_path,
        "columns": df.columns.tolist(),
        "rows": len(df),
        "preview": df.head().to_dict(orient="records")
    }


# TODO: WIP
@mcp.tool()
def parse_custom_data(file_path: str):
    topic = user_topic_store.get("topic")

    print()


# TODO: WIP
@mcp.tool()
def get_data():
    topic = user_topic_store.get("topic")
    print()


# TODO: WIP
@mcp.tool()
def edit_data_columns():
    topic = user_topic_store.get("topic")
    print()


if __name__ == "__main__":
    mcp.run()
