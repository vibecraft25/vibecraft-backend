__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
from contextlib import AsyncExitStack
from typing import Optional

# Third-party imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Custom imports
from utils.tools import extract_tool_specs


class VibeCraftClient:
    def __init__(self, engine):
        self.engine = engine
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

    async def connect_to_server(self, server_path: str):
        await self.exit_stack.aclose()
        self.exit_stack = AsyncExitStack()
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(
            StdioServerParameters(command="npx", args=[server_path])
        ))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        await self.session.initialize()
        print(f"🔌 Connected to {server_path}")

    async def run_pipeline(self, topic_prompt: str):
        stage = 1
        prompts = {
            1: topic_prompt,
            2: "이 주제에 대한 데이터를 업로드하거나 웹에서 수집해 주세요.",
            3: "수집된 데이터를 시각화하는 웹페이지 코드를 생성해 주세요."
        }
        server_paths = {
            1: "@aakarsh-sasi/memory-bank-mcp",
            2: "vibecraft_data_upload",     # TODO
            3: "vibecraft_code_generator"   # TODO
        }

        while stage <= 3:
            try:
                print(f"\n🚦 Step {stage} 시작")
                await self.connect_to_server(server_paths[stage])
                tools = await self.session.list_tools()
                tool_specs = extract_tool_specs(tools)

                result, redo, go_back = await self.engine.generate_with_tools(
                    prompt=prompts[stage],
                    tools=tool_specs,
                    session=self.session
                )
                print(f"\n📌 응답:\n{result}")

                if go_back:
                    stage = max(1, stage - 1)
                    print("🔙 이전 단계로 돌아갑니다.")
                elif redo:
                    print("🔁 현재 단계를 다시 시도합니다.")
                    continue
                else:
                    stage += 1

            except Exception as e:
                print(f"❌ 오류 발생: {e}")
                break

    async def cleanup(self):
        await self.exit_stack.aclose()
