__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
import os
import base64
from contextlib import AsyncExitStack
from typing import Optional, List

# Third-party imports
import pandas as pd
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Custom imports
from engine.base import BaseEngine
from schemas.pipeline_schemas import TopicStepResult
from utils.tools import extract_tool_specs
from utils.menus import *
from utils.prompts import *
from utils.data_loader_utils import (
    load_files,
    markdown_table_to_df
)


class VibeCraftClient:
    def __init__(self, engine: BaseEngine):
        self.engine = engine
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

        self.memory_bank_server: Optional[str] = "@aakarsh-sasi/memory-bank-mcp"
        self.topic_server: Optional[str] = None
        self.web_search_server: Optional[str] = None
        self.data_parser_server: Optional[str] = None
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

    async def execute_step(self, prompt: str, server_path: Optional[str] = None) -> str:
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
        prompt = set_topic_prompt(topic_prompt)
        result = await self.execute_step(prompt, self.topic_server)
        print(f"\n📌 주제 설정 결과:\n{result}")

        while True:
            user_choice = topic_selection_menu()

            if user_choice == "1":
                return TopicStepResult(topic_prompt=topic_prompt, result=result)
            elif user_choice == "2":
                additional_query = additional_query_prompt(topic_prompt, result)
                result = await self.execute_step(additional_query, self.topic_server)
                print(f"\n🛠 수정된 주제 결과:\n{result}")
            elif user_choice == "3":
                await self.reset_via_memory_bank("주제를 다시 설정하고 싶습니다.")
                new_prompt = input("🎤 새로운 주제를 입력하세요: ")
                return await self.step_topic_selection(new_prompt)
            else:
                print("⚠️ 유효한 선택지를 입력해주세요 (1, 2, 3)")

    async def step_data_upload_or_collection(self, topic_result: TopicStepResult) -> Optional[pd.DataFrame]:
        print("\n🚦 Step 2: 데이터 업로드 또는 수집")

        # TODO: 다중 파일 load 테스트 확인 필요
        user_choice = select_data_loader_menu()
        if user_choice == "1":
            df = load_files()
        elif user_choice == "2":
            print("\n🧠 주제 기반 샘플 데이터를 생성 중입니다...")
            prompt = generate_sample_prompt(topic_result.topic_prompt, topic_result.result)
            response = await self.execute_step(prompt)
            df = markdown_table_to_df(response)
        else:
            # TODO: WIP
            print("\n🌐 관련 데이터 다운로드 링크를 추천합니다...")
            prompt = generate_download_link_prompt(topic_result.topic_prompt, topic_result.result)
            try:
                await self.connect_to_server(self.web_search_server)
            except Exception as e:
                print(f"⚠️ 웹 검색 MCP 연결 실패: {e}")
                return None

            tools = await self.session.list_tools()
            tool_specs = extract_tool_specs(tools)
            result = await self.engine.generate_with_tools(prompt, tool_specs, self.session)
            print(f"\n🔗 추천된 다운로드 링크:\n{result}")
            df = load_files()

        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        if df is not None:
            # 1. Check data
            print(f"\n📊 최종 데이터프레임 요약:\n{df.to_string(index=False)}")

            # 2. Check columns
            removal_prompt = recommend_removal_column_prompt(df)
            print("\n🧹 컬럼 삭제 추천 요청 중...")
            suggestion = await self.execute_step(removal_prompt, self.data_parser_server)
            print(f"\n🤖 추천된 컬럼 목록:\n{suggestion}")

            choice = select_edit_col_menu()
            if choice == "1":
                columns_line = suggestion.splitlines()[0]
                to_drop = [col.strip() for col in columns_line.split(",")]
            elif choice == "2":
                print(f"\n🧹 현재 컬럼 목록:\n{', '.join(df.columns)}")
                drop_input = input("삭제할 컬럼명을 쉼표(,)로 입력 (Enter 입력 시 건너뜀): ").strip()
                to_drop = [col.strip() for col in drop_input.split(",")] if drop_input else []
            else:
                print("⚠️ 잘못된 입력입니다. 컬럼 삭제를 건너뜁니다.")
                to_drop = []

            # TODO: WIP
            prompt = df_to_sqlite_with_col_filter_prompt(df, to_drop)
            print("\n💾 SQLite 테이블화 요청 중...")
            save_path = "./data_store"
            os.makedirs(save_path, exist_ok=True)

            response = await self.execute_step(prompt, self.data_parser_server)
            print(f"\n🧱 SQLite 저장 결과:\n{response}")

            # save_sqlite()

            return df
        else:
            return await self.step_data_upload_or_collection(topic_result)

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
        # topic_prompt_result = await self.step_topic_selection(topic_prompt)
        # TODO: TEST WIP
        topic_prompt_result = (
            TopicStepResult(
                topic_prompt=topic_prompt,
                result='''
        피자 일매출 시각화 페이지 제작을 위해 필요한 데이터는 크게 **필수 데이터**와 **추가 분석을 위한 데이터**로 나눌 수 있습니다. 어떤 정보를 얼마나 자세히 보여주고 싶은지에 따라 필요한 데이터의 범위가 달라집니다.
        ---

        ### 1. 필수 데이터 (Core Data)

        일매출 추이를 시각화하는 데 가장 기본적인 정보들입니다.

        1.  **날짜 (Date)**
            *   **데이터 유형:** `날짜 형식 (YYYY-MM-DD)`
            *   **설명:** 각 매출 데이터가 어느 날짜에 발생했는지를 나타냅니다. 시간 흐름에 따른 매출의 변화 추이를 파악하는 데 필수적입니다.
            *   **예시:** `2023-10-26`

        2.  **일매출액 (Daily Sales Revenue)**
            *   **데이터 유형:** `숫자 (정수 또는 소수)`
            *   **설명:** 해당 날짜에 발생한 총 매출 금액입니다. 시각화의 핵심 지표가 됩니다.
            *   **예시:** `1,250,000` (원)

        3.  **판매 수량 (Number of Units Sold)**
            *   **데이터 유형:** `숫자 (정수)`
            *   **설명:** 해당 날짜에 판매된 총 피자(또는 메뉴)의 개수입니다. 매출액과 함께 판매량의 트렌드를 파악하여, 매출액이 올랐을 때 단가가 올랐는지, 판매량이 늘었는지 등을 분석할 수 있습니다.
            *   **예시:** `150` (개)

        ---

        ### 2. 추가 분석을 위한 데이터 (Optional Data for Deeper Insights)

        매출 변동의 원인을 파악하거나, 마케팅 전략 수립에 도움을 줄 수 있는 데이터들입니다. 시각화 페이지에서 더 다양한 필터링, 비교, 분석 기능을 제공하고 싶을 때 유용합니다.

        1.  **요일 (Day of the Week)**
            *   **데이터 유형:** `문자열 (예: 월, 화, 수) 또는 숫자 (예: 1=월, 7=일)`
            *   **설명:** 요일별 매출 패턴을 파악하여 주말과 주중 매출의 차이나 특정 요일의 강세/약세 등을 분석할 수 있습니다. (예: 금요일 저녁 매출이 특히 높다)
            *   **예시:** `금요일`

        2.  **피자 종류별 매출/판매량 (Sales/Quantity by Pizza Type)**
            *   **데이터 유형:** `문자열 (피자 종류명), 숫자 (해당 종류의 매출/수량)`
            *   **설명:** 어떤 피자가 가장 잘 팔리는지, 계절별/기간별 인기 메뉴의 변화를 파악할 수 있습니다. 특정 피자만 따로 떼어내어 매출을 시각화할 수도 있습니다.
            *   **예시:** `페퍼로니 피자`, `고구마 피자`, `불고기 피자` 각각의 매출/수량

        3.  **주문 채널별 매출/판매량 (Sales/Quantity by Order Channel)**
            *   **데이터 유형:** `문자열 (채널명), 숫자 (해당 채널의 매출/수량)`
            *   **설명:** 배달 앱, 전화 주문, 방문 포장, 매장 식사 등 어떤 채널을 통해 매출이 발생하는지 파악하여 채널별 비중과 효율성을 분석할 수 있습니다.
            *   **예시:** `배달의민족`, `요기요`, `자체 앱`, `전화 주문`, `방문 포장`, `매장 식사` 각각의 매출/수량

        4.  **프로모션/할인 여부 (Promotion/Discount Status)**
            *   **데이터 유형:** `불리언 (True/False) 또는 문자열 (프로모션명)`
            *   **설명:** 특정 날짜에 진행된 프로모션이나 할인 행사가 매출에 어떤 영향을 미쳤는지 분석하는 데 활용됩니다.
            *   **예시:** `True` (할인 진행), `False` (할인 없음), 또는 `생일 이벤트`

        5.  **날씨 정보 (Weather Information)**
            *   **데이터 유형:** `문자열 (날씨 상태: 맑음, 비, 눈), 숫자 (기온)`
            *   **설명:** 날씨가 매출에 미치는 영향을 분석할 수 있습니다. 예를 들어, 비 오는 날에는 배달 매출이 늘고 매장 매출은 줄어드는 경향 등을 파악할 수 있습니다.
            *   **예시:** `비`, `15.2` (기온)

        6.  **특별 행사/공휴일 여부 (Special Events/Holidays)**
            *   **데이터 유형:** `불리언 (True/False) 또는 문자열 (행사명)`
            *   **설명:** 크리스마스, 설날, 추석 등 공휴일이나 지역 축제 등이 매출에 미치는 영향을 파악합니다.
            *   **예시:** `True` (공휴일), `False` (평일), 또는 `어린이날`, `지역 축제`

        7.  **주문 건수 (Number of Orders)**
            *   **데이터 유형:** `숫자 (정수)`
            *   **설명:** 해당 날짜에 발생한 총 주문 건수입니다. 이를 통해 '객단가 (Average Order Value = 일매출액 / 주문 건수)'를 계산하여 고객 1명(또는 주문 1건)당 평균 구매액을 파악할 수 있습니다.

        ---

        ### 데이터 수집 시 고려사항

        *   **정확성:** 매출 데이터는 오차 없이 정확하게 기록되어야 합니다.
        *   **일관성:** 데이터 형식(날짜, 통화 단위 등)은 항상 일관되게 유지되어야 합니다.
        *   **세분화:** 처음에는 필요한 데이터만 수집하더라도, 나중에 더 깊은 분석을 원할 경우를 대비하여 최대한 세분화된 데이터를 기록해두는 것이 좋습니다. (예: 단순히 "매출"이 아니라, "배달 매출", "포장 매출" 등)

        이러한 데이터를 바탕으로 다양한 차트(선 그래프, 막대 그래프, 파이 차트 등)를 활용하여 피자 일매출을 효과적으로 시각화할 수 있습니다.
        '''
            )
        )

        data_success = await self.step_data_upload_or_collection(topic_prompt_result)
        if not data_success:
            return await self.run_pipeline(input("🎤 새롭게 설정할 주제를 입력하세요: "))
        await self.step_code_generation()

    async def cleanup(self):
        await self.exit_stack.aclose()
