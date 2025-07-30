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

prefix = "workflow"
router = APIRouter(prefix=f"/{prefix}", responses={401: {"description": "raw data file upload"}})


@router.get("/stream/set-topic")
async def stream_set_topic(
    query: str = Query(..., description="Prompt Query"),
):
    engine = "gemini"
    client = VibeCraftClient(engine)

    async def event_generator():
        try:
            async for msg in client.stream_topic_selection(query):
                if msg:
                    yield msg
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


# TODO: WIP
@router.get("/data-generator")
async def generate_data(
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
            async for event, chunk in client.execute_stream_step(
                query, use_langchain=use_langchain
            ):
                if chunk:
                    yield ServerSentEvent(
                        event=event or "progress",
                        data=f"{chunk}"
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


# TODO: WIP
@router.get("/code-generator")
# @router.get("/stream/code-generator")
async def generate_code(
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
            async for event, chunk in client.execute_stream_step(
                query, use_langchain=use_langchain
            ):
                if chunk:
                    yield ServerSentEvent(
                        event=event or "progress",
                        data=f"{chunk}"
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


# TODO: WIP
@router.get("/go-to-step")
async def go_to_step(
    thread_id: str = Query(..., description="Thread ID", example="f09d8c6e-fcb5-4275-bf3d-90a87ede2cb8"),
):
    engine = "gemini"
    client = VibeCraftClient(engine)

    await client.load_tools()
    client.load_chat_history(thread_id)


# TODO: WIP
@router.get("/set-menu")
async def set_menu(
    thread_id: str = Query(..., description="Thread ID", example="f09d8c6e-fcb5-4275-bf3d-90a87ede2cb8"),
    step: str = Query(..., description="Workflow Step", example="1"),
    option: str = Query(..., description="Selected option number", example="1"),
    query: Optional[str] = Query(None, description="Prompt Query"),
):
    engine = "gemini"
    client = VibeCraftClient(engine)
    client.load_chat_history(thread_id)

    async def event_generator():
        try:
            if step == "1":
                async for msg in client.stream_topic_selection_menu_handler(
                    selected_option=option, query=query
                ):
                    if msg:
                        yield msg
                        await asyncio.sleep(0.1)
                yield ServerSentEvent(
                    event="complete",
                    data=client.get_thread_id()
                )
            elif step == "2":
                print("WIP")
        except Exception as e:
            yield ServerSentEvent(
                event="error",
                data=f"data: ❗ 오류 발생: {str(e)}"
            )

    return EventSourceResponse(event_generator())
