__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
from contextlib import AsyncExitStack
from typing import Optional

# Third-party imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Custom imports
from engine.base import BaseEngine
from schemas.pipeline_schemas import TopicStepResult
from utils.tools import extract_tool_specs


class VibeCraftClient:
    def __init__(self, engine: BaseEngine):
        self.engine = engine
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

        self.memory_bank_server: Optional[str] = "@aakarsh-sasi/memory-bank-mcp"
        self.topic_server: Optional[str] = None
        self.data_upload_server: Optional[str] = None
        self.web_search_server: Optional[str] = None
        self.code_generator_server: Optional[str] = None

    async def connect_to_server(self, server_path: Optional[str]):
        if not server_path:
            return
        await self.exit_stack.aclose()
        self.exit_stack = AsyncExitStack()
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(
            StdioServerParameters(command="npx", args=[server_path])
        ))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        await self.session.initialize()
        print(f"\n🔌 Connected to {server_path}")

    async def execute_step(self, prompt: str, server_path: Optional[str]) -> str:
        if server_path:
            await self.connect_to_server(server_path)
            tools = await self.session.list_tools()
            tool_specs = extract_tool_specs(tools)
            return await self.engine.generate_with_tools(
                prompt=prompt,
                tools=tool_specs,
                session=self.session
            )
        return await self.engine.generate(prompt=prompt)

    async def step_topic_selection(self, topic_prompt: str) -> TopicStepResult:
        print("\n🚦 Step 1: 주제 설정")
        prompt = (f"{topic_prompt}"
                  f"\n---\n"
                  f"코드 구현은 여기선 제외하고, 어떤 데이터가 필요한지 설명을 함께 해줘.")
        result = await self.execute_step(prompt, self.topic_server)
        print(f"\n📌 주제 설정 결과:\n{result}")

        while True:
            print("\n[선택지]")
            print("1. 위 결과로 계속 진행")
            print("2. 위 결과를 수정 요청")
            print("3. 주제 재설정")
            user_choice = input("👉 번호를 입력하세요 (1/2/3): ").strip()

            if user_choice == "1":
                return TopicStepResult(
                    topic_prompt=topic_prompt,
                    result=result
                )
            elif user_choice == "2":
                additional_prompt = input("✏️ 추가 수정 요청을 입력해주세요: ")
                additional_query = (f"다음 요청을 반영해 주제 설정 결과를 수정해주세요:"
                                     f"\n{topic_prompt}\n---\n{result}\n---\n"
                                     f"사용자 요청: {additional_prompt}")
                result = await self.execute_step(additional_query , self.topic_server)
                print(f"\n🛠 수정된 주제 결과:\n{result}")
            elif user_choice == "3":
                await self.reset_via_memory_bank("주제를 다시 설정하고 싶습니다.")
                new_prompt = input("🎤 새로운 주제를 입력하세요: ")
                return await self.step_topic_selection(new_prompt)
            else:
                print("⚠️ 유효한 선택지를 입력해주세요 (1, 2, 3)")

    # TODO: upload. wer search 로직, prompt WIP
    async def step_data_upload_or_collection(self, topic_result: TopicStepResult) -> bool:
        print("\n🚦 Step 2: 데이터 업로드 또는 수집")
        prompt = (
            f"{topic_result.topic_prompt}\n\n"
            f"{topic_result.result}\n\n"
            f"해당 주제에 적절한 데이터를 업로드하거나 웹에서 수집한 뒤, SQLite로 저장하고 예시를 보여주세요."
        )

        try:
            await self.connect_to_server(self.data_upload_server)
        except Exception as e:
            print(e)
            print("⚠️ 데이터 업로드 실패 → 웹에서 수집 시도")
            await self.connect_to_server(self.web_search_server)

        tools = await self.session.list_tools()
        tool_specs = extract_tool_specs(tools)
        result = await self.engine.generate_with_tools(prompt=prompt, tools=tool_specs, session=self.session)
        print(f"\n📊 데이터 수집/저장 결과:\n{result}")

        print("\n[선택지]")
        print("1. 계속 진행")
        print("2. 데이터 수집 재시도 또는 주제 변경")
        user_choice = input("👉 번호를 입력하세요 (1/2): ").strip()

        if user_choice == "2":
            await self.reset_via_memory_bank("데이터를 다시 수집하거나 주제를 변경하고 싶습니다.")
            return False

        return True

    # TODO: WIP
    async def step_code_generation(self):
        print("\n🚦 Step 3: 웹앱 코드 생성")
        result, _, _ = await self.execute_step(
            prompt="앞서 설정한 주제와 SQLite 데이터를 기반으로 시각화 기능을 갖춘 웹앱 코드를 생성해주세요.",
            server_path=self.code_generator_server
        )
        print(f"\n💻 웹앱 코드 생성 결과:\n{result}")

    async def reset_via_memory_bank(self, reset_message: str):
        if not self.memory_bank_server:
            print("⚠️ memory_bank_server가 설정되지 않아 초기화 생략")
            return
        print("🔁 Memory Bank 초기화 중...")
        await self.execute_step(reset_message, self.memory_bank_server)

    async def run_pipeline(self, topic_prompt: str):
        topic_prompt = await self.step_topic_selection(topic_prompt)
        data_success = await self.step_data_upload_or_collection(topic_prompt)
        if not data_success:
            return await self.run_pipeline(input("🎤 새롭게 설정할 주제를 입력하세요: "))
        await self.step_code_generation()

    async def cleanup(self):
        await self.exit_stack.aclose()
