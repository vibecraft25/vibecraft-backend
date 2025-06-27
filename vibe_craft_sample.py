""" VibeCraft Claude MCP Pipeline Client (Dynamic Control Version) """
__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

import asyncio
from typing import Optional
from contextlib import AsyncExitStack
from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from anthropic import Anthropic
from openai import OpenAI
from google import genai

load_dotenv()


class VibeCraftPipelineClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()
        self.openai = OpenAI()
        self.google = genai.Client()

    async def connect_to_server(self, server_path: str):
        await self.exit_stack.aclose()  # 연결 초기화
        self.exit_stack = AsyncExitStack()

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(
            StdioServerParameters(command="npx", args=[server_path])
        ))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        await self.session.initialize()
        tools = await self.session.list_tools()
        print(f"\n🔌 Connected to {server_path} with tools: {[t.name for t in tools.tools]}")

    async def call_claude_and_tools(self, user_prompt: str) -> (str, bool, bool):
        messages = [{"role": "user", "content": user_prompt}]
        tools = await self.session.list_tools()
        available_tools = [{
            "name": t.name,
            "description": t.description,
            "input_schema": t.inputSchema
        } for t in tools.tools]

        response = await self.google.aio.models.generate_content(
            model="gemini-2.0-flash",
            contents="Roll 3 dice!",
            config=genai.types.GenerateContentConfig(
                temperature=0,
                tools=[self.session],  # Pass the FastMCP client session
            ),
        )
        print(response.text)

        response = self.anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=messages,
            tools=available_tools
        )

        final_text = []
        go_back = False
        redo = False

        for item in response.content:
            if item.type == "text":
                text = item.text.lower()
                final_text.append(item.text)
                if any(k in text for k in ["다시", "변경", "취소", "redo", "re-do"]):
                    redo = True
                elif any(k in text for k in ["이전", "go back", "되돌리기", "undo"]):
                    go_back = True
            elif item.type == "tool_use":
                tool_result = await self.session.call_tool(item.name, item.input)
                messages.append({"role": "assistant", "content": [item]})
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": item.id,
                        "content": tool_result.content
                    }]
                })
                # Recursive follow-up
                follow_up = self.anthropic.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    messages=messages,
                    tools=available_tools
                )
                for follow_item in follow_up.content:
                    if follow_item.type == "text":
                        text = follow_item.text.lower()
                        final_text.append(follow_item.text)
                        if any(k in text for k in ["다시", "변경", "취소"]):
                            redo = True
                        elif any(k in text for k in ["이전", "되돌리기"]):
                            go_back = True

        return "\n".join(final_text), redo, go_back

    async def run_pipeline(self, topic_prompt: str):
        stage = 1  # 1: topic, 2: data, 3: code
        history = {
            1: topic_prompt,
            2: "데이터를 업로드하거나 웹에서 수집해 주세요.",
            3: "수집된 데이터를 시각화하는 웹페이지 코드를 생성해 주세요."
        }

        while stage <= 3:
            try:
                if stage == 1:
                    print(f"\n[Step 1] Topic 설정 단계")
                    await self.connect_to_server("@aakarsh-sasi/memory-bank-mcp")
                elif stage == 2:
                    print(f"\n[Step 2] 데이터 수집 또는 업로드 단계")
                    await self.connect_to_server("vibecraft_data_upload")
                elif stage == 3:
                    print(f"\n[Step 3] 코드 생성 단계")
                    await self.connect_to_server("vibecraft_code_generator")

                result, redo, go_back = await self.call_claude_and_tools(history[stage])
                print(f"\n📌 Claude 응답:\n{result}")

                if go_back:
                    print("🔄 이전 단계로 되돌아갑니다.")
                    stage = max(1, stage - 1)
                elif redo:
                    print("♻️ 현재 단계를 다시 수행합니다.")
                    continue
                else:
                    stage += 1

            except Exception as e:
                print(f"\n❌ 오류 발생: {e}")
                break

    async def cleanup(self):
        await self.exit_stack.aclose()


async def main():
    client = VibeCraftPipelineClient()
    try:
        topic_prompt = input("🎤 주제를 입력하세요: ").strip()
        await client.run_pipeline(topic_prompt)
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
