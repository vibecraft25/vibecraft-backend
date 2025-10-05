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
from schemas import SSEEventBuilder, SSEEventType
from mcp_agent.schemas import (
    MCPServerConfig,
    VisualizationRecommendationResponse
)
from utils import FileUtils, PathUtils
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
    async def topic_selection(self, topic_prompt: str) -> str:
        """Step 1: 주제 설정"""
        await self.load_tools(self.topic_mcp_server)

        print("\n🚦 Step 1: 주제 설정")
        system, human = set_topic_prompt(topic_prompt)
        result = await self.execute_step(human, system)
        print(result)
        return result

    async def stream_topic_selection(self, topic_prompt: str):
        """Step 1: 주제 설정 (스트리밍)"""
        await self.load_tools(self.topic_mcp_server)

        system, human = set_topic_prompt(topic_prompt)
        async for event, chunk in self.execute_stream_step(human, system):
            yield ServerSentEvent(event=event, data=chunk)

    """Data loading and generation Methods"""
    def upload_data(self, file_path: str):
        print("\n🚦 Step 2-1: 데이터 업로드")

        if file_path:
            self.data = FileUtils.load_local_files([file_path])
        else:
            self.data = FileUtils.load_files()

    async def set_data(self, file_path: str) -> pd.DataFrame:
        """Step 2: 데이터 업로드 또는 생성"""
        await self.load_tools(self.set_data_mcp_server)
        self.upload_data(file_path)

        # 데이터 자동 전처리 및 저장
        await self.auto_process_and_save_data()

        return self.data

    async def stream_set_data(self, file_path: str = None):
        """Step 2: 데이터 업로드 또는 생성"""
        await self.load_tools(self.set_data_mcp_server)
        self.upload_data(file_path)

        # 데이터 자동 전처리 및 저장
        async for event in self.stream_auto_process_and_save_data():
            yield event

    """Data processing Methods"""
    async def auto_process_and_save_data(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Step 3: 데이터 자동 전처리 및 저장 (단일 프롬프트)"""
        if df is None:
            df = self.data

        print("\n🚦 Step 3: 데이터 자동 전처리 및 저장")

        # 1. 데이터 전처리
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df.columns = [FileUtils.normalize_column_name(col) for col in df.columns]
        print(f"\n📊 데이터프레임 정제 완료:\n{df.head(3).to_string(index=False)}")

        # 2. 단일 프롬프트로 컬럼 삭제 + 영문 변환 한번에 처리
        print("\n🧹 불필요한 컬럼 제거 및 영문 변환 중...")
        system, human = auto_process_data_prompt(df)
        result = await self.execute_step(human, system)
        print(f"\n🤖 Agent 처리 결과:\n{result}")

        # 3. 결과 파싱 및 적용
        new_col = FileUtils.parse_dict_flexible(result)
        filtered_new_col = {k: v for k, v in new_col.items() if v is not None}

        # 컬럼 매핑 적용 (dictionary에 없는 컬럼은 자동 제거됨)
        mapped_df = df.rename(columns=new_col)[list(filtered_new_col.values())]
        print(f"\n🧱 최종 데이터:\n{mapped_df.head(3).to_string(index=False)}")

        # 4. 파일 저장
        path = PathUtils.generate_path(self.get_thread_id())
        mapped_df.to_csv(os.path.join(path, f"{self.get_thread_id()}.csv"), encoding="cp949", index=False)
        file_path = FileUtils.save_sqlite(mapped_df, path, self.get_thread_id())
        FileUtils.save_metadata(filtered_new_col, path, file_path)
        self.data = mapped_df

        return mapped_df

    async def stream_auto_process_and_save_data(self, df: Optional[pd.DataFrame] = None):
        """Step 3: 데이터 자동 전처리 및 저장 (스트리밍, 단일 프롬프트)"""
        if df is None:
            df = self.data

        if df is None:
            yield SSEEventBuilder.create_error_event("데이터가 없습니다.")
            return

        yield SSEEventBuilder.create_info_event("🚦 Step 3: 데이터 자동 전처리 및 저장")

        # 1. 데이터 전처리
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df.columns = [FileUtils.normalize_column_name(col) for col in df.columns]
        yield SSEEventBuilder.create_data_event(
            f"📊 데이터프레임 정제 완료:\n{df.head(3).to_string(index=False)}"
        )

        # 2. 단일 프롬프트로 컬럼 삭제 + 영문 변환 한번에 처리
        yield SSEEventBuilder.create_info_event("🧹 불필요한 컬럼 제거 및 영문 변환 중...")
        system, human = auto_process_data_prompt(df)
        result_parts = []
        async for event, chunk in self.execute_stream_step(human, system):
            result_parts.append(chunk)
            yield ServerSentEvent(event=event, data=chunk)

        result = ''.join(result_parts)

        # 3. 결과 파싱 및 적용
        new_col = FileUtils.parse_dict_flexible(result)
        filtered_new_col = {k: v for k, v in new_col.items() if v is not None}

        # 컬럼 매핑 적용 (dictionary에 없는 컬럼은 자동 제거됨)
        mapped_df = df.rename(columns=new_col)[list(filtered_new_col.values())]
        yield SSEEventBuilder.create_data_event(
            f"🧱 최종 데이터:\n{mapped_df.head(3).to_string(index=False)}"
        )

        # 4. 파일 저장
        path = PathUtils.generate_path(self.get_thread_id())
        mapped_df.to_csv(os.path.join(path, f"{self.get_thread_id()}.csv"), encoding="cp949", index=False)
        file_path = FileUtils.save_sqlite(mapped_df, path, self.get_thread_id())
        FileUtils.save_metadata(filtered_new_col, path, file_path)
        self.data = mapped_df

        yield SSEEventBuilder.create_info_event("✅ 데이터 전처리 완료")

    """Code Generator Methods"""
    async def auto_recommend_visualization_type(self) -> VisualizationType:
        """Step 4: 시각화 타입 자동 결정"""
        print("\n🚦 Step 4: 시각화 타입 자동 결정")

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
        response = VisualizationRecommendationResponse(
            user_context=user_context,
            recommendations=recommendations
        )

        # 가장 신뢰도 높은 시각화 타입 자동 선택
        top_recommendation = response.get_top_recommendation()
        print(f"💡 자동 선택된 시각화 타입: {top_recommendation.visualization_type} (신뢰도: {top_recommendation.confidence}%)")

        return top_recommendation.visualization_type

    def run_code_generator(
            self, thread_id: str, visualization_type: VisualizationType,
            project_name: str = None, model: str = "pro"
    ) -> Dict[str, Any]:
        """동기 방식 코드 생성"""
        print("\n🚦 Step 5: 웹앱 코드 생성")

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
                output_dir=output_dir,
                project_name=project_name or f"vibecraft-{thread_id}",
                model=model
            )

            if result["success"]:
                print(f"✅ 코드 생성 완료: {result['output_dir']}")

            return result
        except Exception as e:
            return {"success": False, "message": str(e)}

    async def stream_run_code_generator(
            self, thread_id: str, visualization_type: VisualizationType,
            project_name: str = None, model: str = "pro"
    ):
        """비동기 스트림 방식 코드 생성 (SSE용)"""

        yield SSEEventBuilder.create_info_event("🚦 Step 6: 웹앱 코드 생성 시작")

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
                    output_dir=output_dir,
                    project_name=project_name or f"vibecraft-{thread_id}",
                    model=model
            ):
                # 이벤트 타입별 SSE 변환
                event_type = event.event
                message = event.data

                if event_type == SSEEventType.ERROR.value:
                    yield SSEEventBuilder.create_error_event(message)
                elif event_type == SSEEventType.COMPLETE.value:
                    yield SSEEventBuilder.create_info_event("🎉 웹앱 코드 생성 완료!")
                    yield SSEEventBuilder.create_complete_event(thread_id)
                    return
                else:
                    yield SSEEventBuilder.create_ai_message_chunk(message)

        except Exception as e:
            yield SSEEventBuilder.create_error_event(f"코드 생성 중 오류: {str(e)}")

    """Pipeline Methods"""
    async def run_workflow(self):
        # Step 4: 인과관계 분석 (BaseEngine에서 자동으로 RAG 활용)
        print("\n🚦 Step 4: 데이터 인과관계 분석")
        analysis_query = f"다음 데이터의 인과관계를 분석해주세요:\n{self.data.head(10).to_string()}"
        analysis_result = await self.execute_step(analysis_query)
        print(f"\n📊 인과관계 분석 결과:\n{analysis_result}")

        # Step 5: 시각화 타입 자동 결정
        v_type = await self.auto_recommend_visualization_type()

        # Step 6: 코드 자동 생성
        print(f"\n💻 시각화 타입 '{v_type}'으로 코드 생성을 진행합니다...")
        result = self.run_code_generator(self.get_thread_id(), v_type)

        if result["success"]:
            print(f"\n✅ 파이프라인 완료! 생성된 코드: {result['output_dir']}")
            return result
        else:
            print(f"\n❌ 코드 생성 실패: {result['message']}")
            return result

    async def stream_run_workflow(self):
        # Step 4: 인과관계 분석
        yield SSEEventBuilder.create_info_event("🚦 Step 4: 데이터 인과관계 분석")
        analysis_query = f"다음 데이터의 인과관계를 분석해주세요:\n{self.data.head(10).to_string()}"
        async for event, chunk in self.execute_stream_step(analysis_query):
            yield ServerSentEvent(event=event, data=chunk)

        # Step 5: 시각화 타입 자동 결정
        yield SSEEventBuilder.create_info_event("🚦 Step 5: 시각화 타입 자동 결정")
        v_type = await self.auto_recommend_visualization_type()
        yield SSEEventBuilder.create_data_event(f"💡 선택된 시각화 타입: {v_type}")

        # Step 6: 코드 자동 생성
        async for event in self.stream_run_code_generator(self.get_thread_id(), v_type):
            yield event

    """Pipeline Test"""
    async def run_pipeline(self, topic_prompt: str, file_path: str):
        """
        간소화된 자동 파이프라인

        Args:
            topic_prompt: 분석 주제
            file_path: 데이터 파일 경로
        """
        # Step 1: 주제 설정
        await self.topic_selection(topic_prompt)

        # Step 2: 데이터 업로드 또는 생성
        await self.set_data(file_path)

        # 이후 자동화 프로세스
        # Step 4: 인과관계 분석 (BaseEngine에서 자동으로 RAG 활용)
        print("\n🚦 Step 4: 데이터 인과관계 분석")
        analysis_query = f"다음 데이터의 인과관계를 분석해주세요:\n{self.data.head(10).to_string()}"
        analysis_result = await self.execute_step(analysis_query)
        print(f"\n📊 인과관계 분석 결과:\n{analysis_result}")

        # Step 5: 시각화 타입 자동 결정
        v_type = await self.auto_recommend_visualization_type()

        # Step 6: 코드 자동 생성
        print(f"\n💻 시각화 타입 '{v_type}'으로 코드 생성을 진행합니다...")
        result = self.run_code_generator(self.get_thread_id(), v_type)

        if result["success"]:
            print(f"\n✅ 파이프라인 완료! 생성된 코드: {result['output_dir']}")
            return result
        else:
            print(f"\n❌ 코드 생성 실패: {result['message']}")
            return result

    async def stream_run_pipeline(self, topic_prompt: str, file_path: Optional[str] = None):
        """
        간소화된 자동 파이프라인 (스트리밍)

        Args:
            topic_prompt: 분석 주제
            file_path: 데이터 파일 경로 (None이면 자동 생성)
        """
        # Step 1: 주제 설정
        yield SSEEventBuilder.create_info_event("🚦 Step 1: 주제 설정")
        async for event in self.stream_topic_selection(topic_prompt):
            yield event

        # Step 2: 데이터 업로드 또는 생성
        yield SSEEventBuilder.create_info_event("🚦 Step 2: 데이터 로드")
        # self.stream_set_data(file_path)
        self.stream_set_data(r"/samples/sample.csv")
        if self.data is not None:
            yield SSEEventBuilder.create_data_event(f"✅ 데이터 로드 완료 ({len(self.data)} rows)")

        # Step 3: 데이터 자동 전처리 및 저장
        async for event in self.stream_auto_process_and_save_data():
            yield event

        # Step 4: 인과관계 분석
        yield SSEEventBuilder.create_info_event("🚦 Step 4: 데이터 인과관계 분석")
        analysis_query = f"다음 데이터의 인과관계를 분석해주세요:\n{self.data.head(10).to_string()}"
        async for event, chunk in self.execute_stream_step(analysis_query):
            yield ServerSentEvent(event=event, data=chunk)

        # Step 5: 시각화 타입 자동 결정
        yield SSEEventBuilder.create_info_event("🚦 Step 5: 시각화 타입 자동 결정")
        v_type = await self.auto_recommend_visualization_type()
        yield SSEEventBuilder.create_data_event(f"💡 선택된 시각화 타입: {v_type}")

        # Step 6: 코드 자동 생성
        async for event in self.stream_run_code_generator(self.get_thread_id(), v_type):
            yield event

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
