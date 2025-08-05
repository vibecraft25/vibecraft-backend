__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
import os
from typing import Dict, Any

# Third-party imports
from langchain_mcp_adapters.client import MultiServerMCPClient
from sse_starlette import ServerSentEvent

# Custom imports
from mcp_agent.client import VibeCraftAgentRunner
from mcp_agent.engine import (
    ClaudeEngine,
    OpenAIEngine,
    GeminiEngine
)
from mcp_agent.schemas.prompt_parser_schemas import VisualizationType
from schemas import SSEEventBuilder
from mcp_agent.schemas import (
    MCPServerConfig,
    VisualizationRecommendationResponse
)
from utils import FileUtils, PathUtils
from utils.menus import *
from utils.prompts import *


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
        self.set_data_mcp_server: Optional[List[MCPServerConfig]] = None  # TODO: WIP
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
        self, prompt: str, system: Optional[str] = None,
        use_langchain: Optional[bool] = True,
    ) -> str:
        if use_langchain:
            return await self.engine.generate_langchain(prompt=prompt, system=system)
        return await self.engine.generate(prompt=prompt)

    async def execute_stream_step(
        self, prompt: str, system: Optional[str] = None,
        use_langchain: Optional[bool] = True,
    ):
        if use_langchain:
            async for chunk in self.engine.stream_generate_langchain(
                    prompt=prompt, system=system
            ):
                yield chunk
        else:
            async for chunk in self.engine.stream_generate(prompt=prompt):
                yield chunk

    def get_summary(self) -> str:
        stats = self.engine.get_conversation_stats()
        if stats['has_summary']:
            return stats["summary"]
        else:
            self.engine.trigger_summarize()
            stats = self.engine.get_conversation_stats()
            return stats["summary"]

    """Topic Selection Methods"""
    async def topic_selection(self, topic_prompt: str):
        await self.load_tools(self.topic_mcp_server)

        print("\n🚦 Step 1: 주제 설정")
        system, human = set_topic_prompt(topic_prompt)
        result = await self.execute_step(human, system)
        print(result)

    async def topic_selection_menu_handler(self):
        selected_option = input(f"\n{topic_selection_menu()}\n").strip()

        if selected_option == "1":
            await self.set_data(cli=True)
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

        system, human = set_topic_prompt(topic_prompt)
        async for event, chunk in self.execute_stream_step(human, system):
            yield ServerSentEvent(event=event, data=chunk)
        yield SSEEventBuilder.create_menu_event(topic_selection_menu())

    """Data loading and generation Methods"""
    async def set_data(
        self, file_path: Optional[str] = None, cli: bool = False
    ):
        await self.load_tools(self.set_data_mcp_server)

        selected_option = None
        if cli:
            file_path = None
            selected_option = select_data_loader_menu()

        if selected_option == "1" or file_path:
            self.upload_data(file_path)
        else:
            self.data = await self.generate_data()

        await self.data_save(self.data, [])

    async def generate_data(self) -> pd.DataFrame:
        print("\n🚦 Step 2-1: 주제 기반 샘플 데이터를 생성")
        system, human = generate_sample_prompt()
        sample_data = await self.execute_step(human, system)
        df = FileUtils.markdown_table_to_df(sample_data)

        return df

    def upload_data(self, file_path: Optional[str] = None):
        print("\n🚦 Step 2-1: 데이터 업로드")

        if file_path:
            self.data = FileUtils.load_local_files([file_path])
        else:
            self.data = FileUtils.load_files()

    """Data processing Methods"""
    async def data_processing(self, df: Optional[pd.DataFrame] = None):
        """데이터 전처리 및 컬럼 추천"""
        if df is None:
            df = self.data

        # 1. 데이터 전처리
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df.columns = [FileUtils.normalize_column_name(col) for col in df.columns]
        print(f"\n📊 최종 데이터프레임 요약:\n{df.head(3).to_string(index=False)}")

        # 2. 컬럼 삭제 추천
        system, human = recommend_removal_column_prompt(df)
        print("\n🧹 컬럼 삭제 추천 요청 중...")
        suggestion = await self.execute_step(human, system)
        print(f"\n🤖 추천된 컬럼 목록:\n{suggestion}")

        return df, suggestion

    async def data_save(self, df: pd.DataFrame, to_drop: List[str]):
        """데이터 저장 처리"""
        print("\n💾 SQLite 테이블화 요청 중...")
        system, human = df_to_sqlite_with_col_filter_prompt(df, to_drop)
        result = await self.execute_step(human, system)
        print(f"Mapped Column dictionary: {result}")

        new_col = FileUtils.parse_dict_flexible(result)
        filtered_new_col = {k: v for k, v in new_col.items() if v is not None}

        mapped_df = df.rename(columns=new_col)[list(filtered_new_col.values())]
        print(f"\n🧱 Mapped Result:\n{mapped_df.head(3).to_string(index=False)}")

        # 파일 저장
        path = PathUtils.generate_path(self.get_thread_id())
        mapped_df.to_csv(os.path.join(path, f"{self.get_thread_id()}.csv"), encoding="cp949", index=False)
        file_path = FileUtils.save_sqlite(mapped_df, path, self.get_thread_id())
        FileUtils.save_metadata(filtered_new_col, path, file_path)

    async def data_handler(self, df: Optional[pd.DataFrame] = None) -> bool:
        """데이터 처리 메뉴 핸들러"""

        print("\n🚦 Step 2-2: 데이터 수정")

        is_running = True

        if df is None:
            df = self.data
        df, suggestion = await self.data_processing(df)

        selected_option = input(f"\n{select_edit_col_menu()}\n").strip()

        if selected_option == "1":
            columns_line = suggestion.splitlines()[0]
            to_drop = [col.strip() for col in columns_line.split(",")]
        elif selected_option == "2":
            print(f"\n🧹 현재 컬럼 목록:\n{', '.join(df.columns)}")
            drop_input = input("삭제할 컬럼명을 쉼표(,)로 입력 (Enter 입력 시 건너뜀): ").strip()
            to_drop = [col.strip() for col in drop_input.split(",")] if drop_input else []
        else:
            print("컬럼 삭제를 건너뜁니다.")
            to_drop = []
            is_running = False

        await self.data_save(df, to_drop)

        return is_running

    async def stream_data_processing(self, df: Optional[pd.DataFrame] = None):
        """스트림 방식 데이터 처리"""
        if df is None:
            df = self.data

        if df is None:
            yield SSEEventBuilder.create_error_event("데이터가 없습니다.")
            return

        # 데이터 전처리
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df.columns = [FileUtils.normalize_column_name(col) for col in df.columns]

        yield SSEEventBuilder.create_data_event(
            f"📊 최종 데이터프레임 요약:\n{df.head(3).to_string(index=False)}"
        )

        # 컬럼 삭제 추천 스트리밍
        system, human = recommend_removal_column_prompt(df)
        async for event, chunk in self.execute_stream_step(human, system):
            yield ServerSentEvent(event=event, data=chunk)
        yield SSEEventBuilder.create_data_event(', '.join(df.columns))
        yield SSEEventBuilder.create_menu_event(select_edit_col_menu())

    async def stream_data_handler(
        self, query: str,
        df: Optional[pd.DataFrame] = None, meta: Optional[dict] = None,
    ):
        """데이터 처리 메뉴 핸들러"""

        print("\n🚦 Step 2-2: 데이터 수정")

        if df is None:
            df = self.data

        system, human = parse_removal_column_prompt(df, query, meta)
        suggestion = await self.execute_step(human, system)
        columns_line = suggestion.splitlines()[0]
        to_drop = [col.strip() for col in columns_line.split(",")]

        await self.data_save(df, to_drop)
        yield SSEEventBuilder.create_menu_event(additional_select_edit_col_menu())

    async def recommend_visualization_type(self) -> VisualizationRecommendationResponse:
        print("\n🚦 Step 2-3: 주제와 데이터 기반 시각화 방식 설정")

        stats = self.engine.get_conversation_stats()
        if stats['has_summary']:
            user_context = stats["summary"]
        else:
            self.engine.trigger_summarize()
            stats = self.engine.get_conversation_stats()
            user_context = stats["summary"]

        system, human = recommend_visualization_template_prompt(self.data, user_context)
        result = await self.execute_step(human, system)

        recommendations = FileUtils.parse_visualization_recommendation(result)
        return VisualizationRecommendationResponse(
            user_context=user_context,
            recommendations=recommendations
        )

    """Code Generator Methods"""
    def run_code_generator(
            self, thread_id: str, visualization_type: VisualizationType
    ) -> Dict[str, Any]:
        """동기 방식 코드 생성"""
        print("\n🚦 Step 3: 웹앱 코드 생성")

        runner = VibeCraftAgentRunner()
        file_name = f"{thread_id}.sqlite"

        if not runner.is_available() or not PathUtils.is_exist(thread_id, file_name):
            return {"success": False, "message": "전제 조건 확인 실패"}

        file_path = PathUtils.get_path(thread_id, file_name)[0]
        output_dir = f"./output/{thread_id}"

        try:
            result = runner.run_agent(
                sqlite_path=file_path,
                visualization_type=visualization_type,
                user_prompt=self.get_summary(),
                output_dir=output_dir
            )

            if result["success"]:
                print(f"✅ 코드 생성 완료: {result['output_dir']}")

            return result
        except Exception as e:
            return {"success": False, "message": str(e)}

    async def stream_run_code_generator(
            self, thread_id: str, visualization_type: VisualizationType
    ):
        """비동기 스트림 방식 코드 생성 (SSE용)"""

        yield SSEEventBuilder.create_info_event("🚦 Step 3: 웹앱 코드 생성 시작")

        runner = VibeCraftAgentRunner()
        file_name = f"{thread_id}.sqlite"

        # 전제 조건 확인
        if not runner.is_available():
            yield SSEEventBuilder.create_error_event("vibecraft-agent를 사용할 수 없습니다.")
            return

        if not PathUtils.is_exist(thread_id, file_name):
            yield SSEEventBuilder.create_error_event(f"SQLite 파일을 찾을 수 없습니다: {file_name}")
            return

        yield SSEEventBuilder.create_info_event("✅ 사전 검증 완료")

        file_path = PathUtils.get_path(thread_id, file_name)[0]
        output_dir = f"./output/{thread_id}"

        try:
            async for event in runner.run_agent_async(
                    sqlite_path=file_path,
                    visualization_type=visualization_type,
                    user_prompt=self.get_summary(),
                    output_dir=output_dir
            ):
                # 이벤트 타입별 SSE 변환
                event_type = event.get("type", "info")
                message = event.get("message", "")

                if event_type == "error":
                    yield SSEEventBuilder.create_error_event(message)
                elif event_type == "stdout":
                    yield SSEEventBuilder.create_ai_message_chunk(message)
                elif event.get("step") == "execution_complete":
                    yield SSEEventBuilder.create_info_event("🎉 웹앱 코드 생성 완료!")
                    yield SSEEventBuilder.create_complete_event(thread_id)
                    return
                else:
                    yield SSEEventBuilder.create_ai_message_chunk(message)

        except Exception as e:
            yield SSEEventBuilder.create_error_event(f"코드 생성 중 오류: {str(e)}")

    """Deploy Methods"""
    # TODO: WIP
    async def step_deploy(self):
        await self.load_tools(self.deploy_mcp_server)

        print("\n🚦 Step 4: Deploy")
        result = await self.execute_step("WIP")
        print(f"\n💻 배포중...")

    async def run_pipeline(self, topic_prompt: str):
        # Step: 1
        await self.topic_selection(topic_prompt)
        self.engine.trigger_summarize()
        stats = self.engine.get_conversation_stats()
        if stats['has_summary']:
            print(f"Summary Preview: {stats['summary_preview']}")
        # Step: 2-1
        while self.data is None:
            await self.topic_selection_menu_handler()
        # Step: 2-2
        while await self.data_handler():
            pass
        # Step: 2-3
        v_type = (await self.recommend_visualization_type()).get_top_recommendation()
        # Step: 3
        result = self.run_code_generator(self.get_thread_id(), v_type.visualization_type)
        breakpoint()
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
