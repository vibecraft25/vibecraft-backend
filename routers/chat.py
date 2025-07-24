__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
import asyncio
from typing import Optional

# Third-party imports
from fastapi import APIRouter
from fastapi.params import Query
from sse_starlette.sse import EventSourceResponse, ServerSentEvent
from starlette.responses import JSONResponse

# Custom imports
from client.vibe_craft_client import VibeCraftClient
from schemas import ChatResponse

prefix = "chat"
router = APIRouter(prefix=f"/{prefix}", responses={401: {"description": "raw data file upload"}})


@router.get(
    "/new-chat",
    response_model=ChatResponse
)
async def new_chat(
    query: str = Query(..., description="Prompt Query"),
    use_langchain: Optional[bool] = Query(True, description="Trigger for Langchain")
):
    engine = "gemini"
    client = VibeCraftClient(engine)

    await client.load_tools()
    response = await client.execute_step(query, use_langchain=use_langchain)

    return JSONResponse(
        content=ChatResponse(
            data=response,
            thread_id=client.get_thread_id(),
    ).model_dump(), status_code=200)


@router.get(
    "/load-chat",
    response_model=ChatResponse
)
async def load_chat(
    query: str = Query(..., description="Prompt Query"),
    thread_id: str = Query(..., description="Thread ID", example="f09d8c6e-fcb5-4275-bf3d-90a87ede2cb8"),
    use_langchain: Optional[bool] = Query(True, description="Trigger for Langchain")
):
    engine = "gemini"
    client = VibeCraftClient(engine)

    await client.load_tools()
    client.load_chat_history(thread_id)
    response = await client.execute_step(query, use_langchain=use_langchain)

    return JSONResponse(
        content=ChatResponse(
            data=response,
            thread_id=client.get_thread_id(),
    ).model_dump(), status_code=200)


@router.get("/stream/new-chat")
async def stream_new_chat(
    query: str = Query(..., description="Prompt Query"),
    use_langchain: Optional[bool] = Query(True, description="Trigger for Langchain")
):
    engine = "gemini"
    client = VibeCraftClient(engine)

    await client.load_tools()

    async def event_generator():
        try:
            # 스트리밍 결과 전송
            async for chunk in client.execute_stream_step(
                query, use_langchain=use_langchain
            ):
                if chunk:
                    yield ServerSentEvent(
                        event="progress",
                        data=f"data: {chunk}"
                    )
                    await asyncio.sleep(0.1)

            yield ServerSentEvent(
                event="complete",
                data=client.get_thread_id()
            )

        except Exception as e:
            yield ServerSentEvent(
                event="error",
                data=f"data: ❗ 오류 발생: {str(e)}"
            )

    return EventSourceResponse(event_generator())


@router.get("/stream/load-chat")
async def stream_load_chat(
    query: str = Query(..., description="Prompt Query"),
    thread_id: str = Query(..., description="Thread ID", example="f09d8c6e-fcb5-4275-bf3d-90a87ede2cb8"),
    use_langchain: Optional[bool] = Query(True, description="Trigger for Langchain")
):
    engine = "gemini"
    client = VibeCraftClient(engine)

    await client.load_tools()
    client.load_chat_history(thread_id)

    async def event_generator():
        try:
            # 스트리밍 결과 전송
            async for chunk in client.execute_stream_step(
                query, use_langchain=use_langchain
            ):
                if chunk:
                    yield ServerSentEvent(
                        event="progress",
                        data=f"data: {chunk}"
                    )
                    await asyncio.sleep(0.1)

            yield ServerSentEvent(
                event="complete",
                data=client.get_thread_id()
            )

        except Exception as e:
            yield ServerSentEvent(
                event="error",
                data=f"data: ❗ 오류 발생: {str(e)}"
            )

    return EventSourceResponse(event_generator())

# TODO: implement go_back, return menu, options, etc...