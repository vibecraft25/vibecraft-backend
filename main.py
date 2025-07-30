__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Third-party imports
import uvicorn
from fastapi import FastAPI
from dotenv import load_dotenv

# Custom imports
from config import settings
from routers import (
    chat,
    workflow,
    content,
)

load_dotenv()   # TODO: 추후 사용자 토큰 받을 수 있게 대체 필요

app = FastAPI(
    version="1.0.0",
    title="VibeCraft SSE API",
    swagger_ui_parameters={"syntaxHighlight": True},
    docs_url="/docs",
)
app.include_router(chat, tags=["chat"])
app.include_router(workflow, tags=["workflow"])
app.include_router(content, tags=["content"])


if __name__ == "__main__":
    from platform import platform
    # 운영 체제에 맞는 방식으로 서버 실행
    if "Windows" in platform():
        uvicorn.run("main:app", host=settings.host, port=settings.port, reload=True)
    else:
        uvicorn.run(app, host=settings.host, port=settings.port)
