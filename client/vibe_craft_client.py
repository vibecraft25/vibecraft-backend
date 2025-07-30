__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
import os
from typing import Optional

# Third-party imports
from langchain_mcp_adapters.client import MultiServerMCPClient
import pandas as pd
from sse_starlette import ServerSentEvent

# Custom imports
from engine import ClaudeEngine, OpenAIEngine, GeminiEngine
from schemas.mcp_schemas import MCPServerConfig
from utils import PathUtils
from utils.menus import *
from utils.prompts import *
from utils.data_loader_utils import (
    load_files,
    load_local_files,
    markdown_table_to_df,
    normalize_column_name,
    parse_first_row_dict_from_text,
    save_metadata,
    save_sqlite
)


class VibeCraftClient:
    def __init__(self, engine: str):
        if engine == "claude":
            self.engine = ClaudeEngine()
        elif engine == "gemini":
            self.engine = GeminiEngine()
        elif engine == "gpt":
            self.engine = OpenAIEngine()
        else:
            raise ValueError("Not Supported Engine")
        self.client: Optional[MultiServerMCPClient] = None

        self.mcp_tools: Optional[List[MCPServerConfig]] = None  # common MCP tools
        self.topic_mcp_server: Optional[List[MCPServerConfig]] = None
        self.web_search_mcp_server: Optional[List[MCPServerConfig]] = None  # TODO: WIP
        self.db_mcp_server: Optional[List[MCPServerConfig]] = None  # TODO: WIP
        self.deploy_mcp_server: Optional[List[MCPServerConfig]] = None  # TODO: WIP

        self.tools: Optional[List] = None

        self.data: Optional[pd.DataFrame] = None

    """Engine Methods"""
    def get_thread_id(self) -> str:
        return str(self.engine.thread_id)

    def merge_chat_history(self, thread_id: str):
        self.engine.merge_chat_history(thread_id=thread_id)

    def load_chat_history(self, thread_id: str):
        self.engine.load_chat_history(thread_id=thread_id)

    async def load_tools(self, mcp_servers: Optional[List[MCPServerConfig]] = None):
        """
        Connect Multiple MCP servers with ClientSessionGroup, and integrate tools, prompts, resources.
        Save self.session
        """

        mcp_servers = mcp_servers or self.mcp_tools
        if mcp_servers:
            try:
                self.client = MultiServerMCPClient(
                    {
                        tool.name: {
                            "command": tool.command,
                            "args": tool.args,
                            "transport": tool.transport
                        }
                        for tool in mcp_servers
                    }
                )
                self.tools = await self.client.get_tools()
                self.engine.update_tools(self.tools)
                print(f"\n🔌 Connected to {', '.join([t.name for t in mcp_servers])}")
                print("Connected to server with tools:", [tool.name for tool in self.tools])
            except Exception as e:
                print(f"⚠️ 서버 연결 실패: {', '.join([t.name for t in mcp_servers])} - {e}")

    async def execute_step(
        self, prompt: str,
        use_langchain: Optional[bool] = True,
    ) -> str:
        if use_langchain:
            return await self.engine.generate_langchain(prompt=prompt)
        return await self.engine.generate(prompt=prompt)

    async def execute_stream_step(
        self, prompt: str,
        use_langchain: Optional[bool] = True,
    ):
        if use_langchain:
            async for chunk in self.engine.stream_generate_langchain(prompt=prompt):
                yield chunk
        else:
            async for chunk in self.engine.stream_generate(prompt=prompt):
                yield chunk

    """Topic Selection Methods"""
    async def topic_selection(self, topic_prompt: str):
        await self.load_tools(self.topic_mcp_server)

        print("\n🚦 Step 1: 주제 설정")
        prompt = set_topic_prompt(topic_prompt)
        result = await self.execute_step(prompt)
        print(result)

    async def topic_selection_menu_handler(self):
        selected_option = input(topic_selection_menu()).strip()

        if selected_option == "1":
            await self.load_data(cli=True)
        elif selected_option == "2":
            additional_query = input("✏️ 추가 수정 요청을 입력해주세요: ")
            result = await self.execute_step(additional_query)
            print(result)
        elif selected_option == "3":
            self.engine.clear_memory()
            new_prompt = input("🎤 새로운 주제를 입력하세요: ")
            result = self.topic_selection(new_prompt)
            print(result)
        else:
            print("⚠️ 유효한 선택지를 입력해주세요 (1, 2, 3)")

    async def stream_topic_selection(self, topic_prompt: str):
        await self.load_tools(self.topic_mcp_server)

        prompt = set_topic_prompt(topic_prompt)
        async for event, chunk in self.execute_stream_step(prompt):
            yield ServerSentEvent(
                event=event,
                data=f"{chunk}"
            )
        yield ServerSentEvent(
            event="menu",
            data=topic_selection_menu()
        )

    async def stream_topic_selection_menu_handler(
        self,
        selected_option: str,
        query: Optional[str] = None,
    ):
        if selected_option == "1":
            await self.load_data(cli=False)
            yield ServerSentEvent(
                event="data",
                # data=f"{chunk}"
                data="[data_path]"
            )
        elif selected_option == "2":
            if query:
                async for event, chunk in self.execute_stream_step(query):
                    yield ServerSentEvent(
                        event=event,
                        data=f"{chunk}"
                    )
        elif selected_option == "3":
            self.engine.clear_memory()
            async for msg in self.stream_topic_selection(query):
                yield msg
        else:
            yield ServerSentEvent(
                event="error",
                data="⚠️ 유효한 선택지를 입력해주세요 (1, 2, 3)"
            )

    """Data Generator and Analysis Methods"""
    # TODO: api와 cli 로직 재설계 필요
    async def load_data(
        self, file_path: Optional[str] = None, cli: bool = False
    ):
        selected_option = None
        if cli:
            file_path = None
            selected_option = select_data_loader_menu()

        if selected_option == "1" or file_path:
            self.data = self.upload_data(file_path)
        self.data = await self.generate_data()

    async def generate_data(self) -> pd.DataFrame:
        await self.load_tools(self.web_search_mcp_server)

        print("\n🚦 Step 2: 주제 기반 샘플 데이터를 생성")
        prompt = generate_sample_prompt()
        sample_data = await self.execute_step(prompt)
        df = markdown_table_to_df(sample_data)

        return df

    def upload_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        print("\n🚦 Step 2: 데이터 업로드")

        if file_path and PathUtils.is_exist(self.engine.thread_id, file_path):
            return pd.read_csv(file_path)
        else:
            return load_files()

    # TODO: refactoring
    async def data_handler(self, df: pd.DataFrame, cli: Optional[bool] = False):
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df.columns = [normalize_column_name(col) for col in df.columns]

        if df is not None:
            # 1. Check data
            print(f"\n📊 최종 데이터프레임 요약:\n{df.head(3).to_string(index=False)}")

            # 2. Check columns
            removal_prompt = recommend_removal_column_prompt(df)
            print("\n🧹 컬럼 삭제 추천 요청 중...")
            suggestion = await self.execute_step(removal_prompt)
            print(f"\n🤖 추천된 컬럼 목록:\n{suggestion}")


            choice = input(select_edit_col_menu()).strip()


            if choice == "1":
                columns_line = suggestion.splitlines()[0]
                to_drop = [col.strip() for col in columns_line.split(",")]
            elif choice == "2":
                print(f"\n🧹 현재 컬럼 목록:\n{', '.join(df.columns)}")
                drop_input = input("삭제할 컬럼명을 쉼표(,)로 입력 (Enter 입력 시 건너뜀): ").strip()
                to_drop = [col.strip() for col in drop_input.split(",")] if drop_input else []
            else:
                print("컬럼 삭제를 건너뜁니다.")
                to_drop = []

            print("\n💾 SQLite 테이블화 요청 중...")
            prompt = df_to_sqlite_with_col_filter_prompt(df, to_drop)
            result = await self.execute_step(prompt)
            print(f"Mapped Column dictionary: {result}")

            new_col = parse_first_row_dict_from_text(result)
            filtered_new_col = {k: v for k, v in new_col.items() if v is not None}

            mapped_df = df.rename(columns=new_col)[list(filtered_new_col.values())]
            print(f"\n🧱 Mapped Result:\n{mapped_df.head(3).to_string(index=False)}")

            save_path = "./data_store"
            os.makedirs(save_path, exist_ok=True)
            df.to_csv(os.path.join(save_path, "data.csv"), encoding="cp949", index=False)
            file_path = save_sqlite(mapped_df, save_path)
            save_metadata(filtered_new_col, save_path, file_path)

            return file_path
        else:
            return await self.load_data(cli)

    """Code Generator Methods"""
    # TODO: WIP
    async def step_code_generation(self):
        # TODO: langchain chat history summary 이후 cli run
        print("\n🚦 Step 3: 웹앱 코드 생성")

    # TODO: WIP
    async def step_deploy(self):
        await self.load_tools(self.deploy_mcp_server)

        print("\n🚦 Step 4: Deploy")
        result = await self.execute_step("WIP")
        print(f"\n💻 배포중...")

    async def run_pipeline(self, topic_prompt: str):
        await self.topic_selection(topic_prompt)
        while self.data is None:
            await self.topic_selection_menu_handler()
        await self.data_handler(self.data)
        breakpoint()
        # await self.step_code_generation()
        # await self.step_deploy()

    async def test(self):
        print("🔥 Run Test...")
        prompt = "주제를 자동으로 설정해줘"

        # Run without Langchain
        result0 = await self.execute_step(prompt, use_langchain=False)
        print(f"\n🤖 Run without tool and Langchain:\n{result0}\n")

        # Run Langchain
        result1 = await self.execute_step(prompt)
        print(f"\n🤖 Langchain without tool:\n{result1}\n")

        while True:
            query = input("\n사용자: ").strip()
            result = await self.execute_step(query)
            print(result)

            self.engine.save_chat_history()
            self.merge_chat_history(thread_id="0d11b676-9cc5-4eb2-a90e-59277ca590fa")
            self.load_chat_history(thread_id="0d11b676-9cc5-4eb2-a90e-59277ca590fa")

    async def cleanup(self):
        self.client = None
