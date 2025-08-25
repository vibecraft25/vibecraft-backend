__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
from typing import List, Optional, Tuple
import json

# Third-party imports
import pandas as pd


##################################
# Topic selection system prompts #
##################################
def set_topic_prompt(topic_prompt: str) -> Tuple[str, str]:
    """주제 설정 프롬프트를 시스템/사용자 메시지로 분리"""
    system_message = (
        "당신은 데이터 분석 프로젝트의 주제를 설정하는 전문가입니다. "
        "사용자의 요청을 분석하여 명확한 분석 목표를 설정하고, "
        "필요한 데이터의 종류와 특성을 설명해주세요. "
        "코드 구현은 하지 말고 분석 계획과 데이터 요구사항에 집중해주세요."
    )

    human_message = (
        f"{topic_prompt}\n\n"
        f"위 요청에 대해 다음 사항들을 포함하여 분석 계획을 세워주세요:\n"
        f"1. 분석 목표와 기대 결과\n"
        f"2. 필요한 데이터의 종류와 특성\n"
        f"3. 주요 분석 관점\n"
        f"4. 예상되는 시각화 방향"
    )

    return system_message, human_message


##############################
# Data loader system prompts #
##############################
def generate_sample_prompt() -> Tuple[str, str]:
    """샘플 데이터 생성 프롬프트를 시스템/사용자 메시지로 분리"""
    system_message = (
        "당신은 마크다운 테이블 형식의 샘플 데이터를 생성하는 전문가입니다. "
        "지금까지의 대화 맥락을 바탕으로 관련된 현실적인 샘플 데이터를 생성해주세요.\n\n"

        "**출력 규칙:**\n"
        "1. 마크다운 테이블 형식으로만 출력\n"
        "2. 첫 번째 줄: 컬럼명 (파이프 | 로 구분)\n"
        "3. 두 번째 줄: 구분선 (---|---|--- 형태)\n"
        "4. 세 번째 줄부터: 데이터 행 (100개 이상)\n"
        "5. 각 셀은 파이프 | 로 구분\n"
        "6. 테이블 앞뒤에 다른 텍스트나 설명 금지\n\n"

        "**출력 예시:**\n"
        "| 컬럼1 | 컬럼2 | 컬럼3 |\n"
        "|-------|-------|-------|\n"
        "| 데이터1 | 데이터2 | 데이터3 |\n"
        "| 데이터4 | 데이터5 | 데이터6 |"
    )

    human_message = (
        "지금까지의 내용을 기반으로 관련된 샘플 데이터를 마크다운 테이블 형식으로 생성해주세요. "
        "컬럼 이름과 100개 이상의 예시 데이터를 포함하여 위 형식으로만 출력해주세요."
    )

    return system_message, human_message


def recommend_removal_column_prompt(df: pd.DataFrame) -> Tuple[str, str]:
    """컬럼 삭제 추천 프롬프트를 시스템/사용자 메시지로 분리"""
    system_message = (
        "당신은 데이터 전처리 전문가입니다. "
        "제공된 데이터프레임을 분석하여 시각화나 분석 목적에 불필요한 컬럼을 식별해주세요.\n\n"

        "**응답 형식:**\n"
        "1. **첫 번째 줄**: 삭제 추천 컬럼의 정확한 컬럼명만 쉼표로 구분하여 출력 (삭제할 컬럼이 없으면 'NONE')\n"
        "2. **두 번째 줄부터**: 각 컬럼에 대한 삭제 이유를 항목별로 설명\n\n"

        "**삭제 추천 기준:**\n"
        "- 중복 정보: 다른 컬럼과 동일하거나 유사한 정보\n"
        "- 높은 결측률: 70% 이상 결측값이 있는 컬럼\n"
        "- 단일값: 모든 행이 동일한 값 (상수 컬럼)\n"
        "- 식별자: ID, 인덱스 등 분석에 불필요한 식별 컬럼\n"
        "- 분석 무관: 시각화나 분석 목적과 무관한 메타데이터"
    )

    preview = df.head(3).to_string(index=False)
    column_list = ", ".join(df.columns)

    human_message = (
        f"다음 데이터를 분석하여 불필요한 컬럼을 추천해주세요.\n\n"
        f"[컬럼 목록]\n{column_list}\n\n"
        f"[미리보기 (상위 3개 row)]\n```\n{preview}\n```\n\n"
        f"위 정보를 바탕으로 삭제 추천 컬럼을 첫 번째 줄에 쉼표로 구분하여 출력하고, "
        f"각 컬럼별 삭제 이유를 설명해주세요."
    )

    return system_message, human_message


def parse_removal_column_prompt(
        df: pd.DataFrame, query: str, meta: Optional[dict] = None
) -> Tuple[str, str]:
    """사용자 쿼리 기반 컬럼 파싱 프롬프트를 시스템/사용자 메시지로 분리"""
    system_message = (
        "당신은 사용자의 자연어 요청을 데이터프레임의 정확한 컬럼명으로 매핑하는 전문가입니다. "
        "사용자가 언급한 컬럼을 정확히 식별하고, 철자 오류나 부분 언급도 의미를 파악하여 처리해주세요.\n\n"

        "**출력 형식:**\n"
        "1. 첫 번째 줄: 찾은 컬럼의 **정확한 컬럼명만 쉼표로 구분하여 출력**\n"
        "2. 두 번째 줄부터: 각 컬럼을 선택한 이유와 사용자 쿼리와의 연관성 설명\n\n"

        "**참고 사항:**\n"
        "- 컬럼 의미 매핑 정보를 활용하여 축약된 컬럼명의 원래 의미를 파악하세요\n"
        "- 사용자가 언급한 단어와 유사한 의미나 철자를 가진 컬럼을 우선적으로 고려하세요\n"
        "- 컬럼의 실제 데이터 내용도 참고하여 사용자의 의도와 일치하는지 확인하세요\n"
        "- 완전히 일치하는 컬럼이 없다면 가장 근사한 컬럼을 제안하고 이유를 설명하세요\n"
        "- 해당하는 컬럼이 전혀 없다면 '해당 컬럼 없음'이라고 출력하세요"
    )

    preview = df.head(3).to_string(index=False)
    column_list = ", ".join(df.columns)

    # 메타데이터에서 컬럼 매핑 정보 추출
    column_mapping_info = ""
    if meta:
        try:
            column_mapping = meta.get("column_mapping", {})
            if column_mapping:
                mapping_lines = []
                for original_name, current_name in column_mapping.items():
                    mapping_lines.append(f"  - {current_name} ← {original_name}")
                column_mapping_info = (
                        "**컬럼 의미 매핑 정보:**\n"
                        + "\n".join(mapping_lines) + "\n\n"
                )
        except (json.JSONDecodeError, KeyError):
            pass

    human_message = (
        f"다음 사용자 쿼리에서 언급된 컬럼을 찾아주세요.\n\n"
        f"**사용자 쿼리:** {query}\n\n"
        f"**사용 가능한 컬럼 목록:**\n{column_list}\n\n"
        f"{column_mapping_info}"
        f"**데이터 미리보기 (상위 3개 행):**\n```\n{preview}\n```"
    )

    return system_message, human_message


def df_to_sqlite_with_col_filter_prompt(
        df: pd.DataFrame, to_drop: List[str]
) -> Tuple[str, str]:
    """DataFrame SQLite 변환 프롬프트를 시스템/사용자 메시지로 분리"""
    system_message = (
        "당신은 데이터베이스 설계 전문가입니다. "
        "주어진 DataFrame의 모든 컬럼명을 반드시 영문으로 변환하고 "
        "불필요한 컬럼을 제거해주세요.\n\n"
        
        "**🚨 필수 요구사항 (절대 준수):**\n"
            "1. 모든 컬럼명은 반드시 영문으로만 변환\n"
            "2. 한글명 사용 절대 금지 - 어떤 경우에도 한글 포함 불가\n"
            "3. 결과 dictionary의 value는 100% 영문명만 허용\n"
            "4. 한글이 포함된 결과는 무효한 응답으로 간주\n\n"
        
            "**데이터베이스 컬럼명 변환 규칙 (우선순위 순):**\n"
            "1. 반드시 영문 축약어 또는 영문 단어 사용 (ISO/ANSI 표준 기준)\n"
            "2. 소문자 + 언더스코어 형식 (snake_case)\n"
            "3. 최대 30자 이내 권장 (Oracle 호환성)\n"
            "4. 숫자로 시작 금지\n"
            "5. 특수문자는 언더스코어(_)만 허용\n"
            "6. 표준 축약어가 없거나 불분명한 경우 의미가 명확한 영문 단어 사용\n"
            "7. 업계별 관례가 있는 경우 해당 관례 우선 고려\n\n"
        
            "**표준 축약어 가이드 (필수 영문 변환):**\n"
            "- 식별자: id, seq, no, code, key, uuid, ref\n"
            "- 날짜/시간: dt (date), tm (time), ts (timestamp), yr (year), mon (month), dy (day), created_at, updated_at\n"
            "- 이름: nm (name), title, desc (description), label, full_name, display_name\n"
            "- 주소: addr (address), st (street), city, zip, postal_cd, country, region\n"
            "- 연락처: tel (telephone), email, fax, mobile, phone, contact\n"
            "- 상태: stat (status), flag, ind (indicator), active_yn, is_active, enabled\n"
            "- 수량: qty (quantity), cnt (count), amt (amount), val (value), rate, pct (percent), total\n"
            "- 분류: cat (category), type, cls (class), grp (group), div (division), tag\n"
            "- 위치: pos (position), loc (location), coord, lat, lng, x_coord, y_coord\n"
            "- 크기: sz (size), len (length), wt (weight), vol (volume), width, height\n"
            "- 사용자: usr (user), admin, mgr (manager), emp (employee), owner, creator\n"
            "- 조직: org (organization), dept (department), div (division), team, company\n"
            "- 재정: price, cost, fee, tax, discount, total, revenue, profit\n"
            "- 제품: prod (product), item, sku, model, brand, variant\n"
            "- 고객: cust (customer), client, vendor, supplier, buyer\n"
            "- 주문: ord (order), req (request), ship (shipment), delivery, invoice\n"
            "- 웹/앱: url, uri, token, session, cookie, api_key\n"
            "- 미디어: img (image), file, doc (document), video, audio, thumbnail\n"
            "- 어업: catch (어획), fish (어종), vessel (어선), fishing (어업), tonnage (톤수)\n"
            "- 지리: area (지역), region (구역), latitude (위도), longitude (경도), coord (좌표)\n\n"
        
            "**DB 예약어 금지 목록:**\n"
            "SELECT, INSERT, UPDATE, DELETE, CREATE, DROP, ALTER, INDEX, TABLE, VIEW, "
            "DATABASE, SCHEMA, PRIMARY, FOREIGN, KEY, CONSTRAINT, NULL, NOT, AND, OR, "
            "WHERE, ORDER, GROUP, HAVING, JOIN, INNER, OUTER, LEFT, RIGHT, UNION, "
            "DISTINCT, COUNT, SUM, AVG, MAX, MIN, DATE, TIME, TIMESTAMP, INT, VARCHAR, "
            "TEXT, BLOB, BOOLEAN, FLOAT, DOUBLE, DECIMAL, CHAR, BINARY\n\n"
        
            "**컬럼 제거 기준 (최대 5개까지만 제거 추천):**\n"
            "1. 중복 정보 (다른 컬럼과 동일한 의미)\n"
            "2. 높은 결측률 (70% 이상 NULL)\n"
            "3. 단일값 컬럼 (모든 행이 동일한 값)\n"
            "4. 임시 식별자 (임시 ID, 인덱스 번호)\n"
            "5. 메타데이터 (파일명, 생성일시 등 분석 무관 정보)\n"
            "6. 개인정보 (개인식별 가능한 민감 정보)\n"
            "** ✅ 삭제 컬럼은 가장 불필요한 것부터 우선순위를 정하여 최대 5개까지만 추천 **\n"
            "** 삭제 기준에 해당하더라도 데이터 분석에 중요한 컬럼은 보존 **\n\n"
        
            "**영문 변환 예시 (반드시 이렇게 변환):**\n"
            "- '어획 연도' → 'catch_yr' 또는 'fishing_year'\n"
            "- '어획량' → 'catch_tons' 또는 'catch_qty'\n"
            "- 'FAO 지역 코드' → 'fao_area_cd'\n"
            "- 'FAO 지역명' → 'fao_area_nm'\n"
            "- '어획 국가' → 'catch_country'\n"
            "- '위도' → 'latitude' 또는 'lat'\n"
            "- '경도' → 'longitude' 또는 'lng'\n"
            "- '주요 어종' → 'major_species'\n"
            "- '고객이름' → 'cust_nm'\n"
            "- '주문날짜' → 'ord_dt'\n"
            "- '제품가격' → 'prod_price'\n\n"
        
            "**❌ 절대 금지되는 잘못된 출력 예시:**\n"
            "{'어획 연도': '어획 연도', '어획량': '어획량'} ← 이런 식의 한글명 절대 금지!\n\n"
        
            "**✅ 올바른 출력 예시:**\n"
            "{'어획 연도': 'catch_yr', '어획량': 'catch_tons', 'FAO 지역명': 'fao_area_nm'}\n\n"
        
            "**출력 형식 (엄격히 준수):**\n"
            "- 첫 번째 줄에만 Python dictionary 형태로 반환\n"
            "- dictionary의 key는 한글(원본), value는 반드시 영문만 사용\n"
            "- 삭제 대상 컬럼은 매핑에서 제외 (최대 5개까지만)\n"
            "- 줄바꿈, 설명, 추가 텍스트 절대 금지\n"
            "- value에 한글이 포함되면 무효한 응답\n\n"
        
            "**주의사항:**\n"
            "- 한국어 의미를 정확히 파악하여 가장 적절한 영문명으로 변환\n"
            "- 표준 축약어 우선 사용, 하지만 명확성 확보\n"
            "- 동일한 의미의 컬럼은 일관된 명명 방식 사용\n"
            "- 개발팀이 이해하기 쉬운 직관적 영문명 선호\n"
            "- 삭제 추천은 신중하게 판단하여 최대 5개까지만 제한"
    )

    available_columns = list(df.columns)
    to_drop_str = ", ".join(to_drop)

    human_message = (
        f"다음 정보를 바탕으로 모든 컬럼명을 반드시 영문으로 변환하고 불필요한 컬럼을 제거해주세요.\n\n"
        f"**삭제 대상 키워드:** {to_drop_str}\n"
        f"**현재 컬럼 목록:** {available_columns}\n\n"
        f"⚠️ 중요: 결과 dictionary의 value는 100% 영문명만 사용하세요.\n"
        f"한글명이 포함된 결과는 무효합니다.\n\n"
        f"첫 번째 줄에 이전 이름(한글)과 변경된 이름(영문)의 매핑을 dictionary 형식으로 한 줄로 반환해주세요."
    )

    return system_message, human_message


def recommend_visualization_template_prompt(
        df: pd.DataFrame, user_context: Optional[str] = None
) -> Tuple[str, str]:
    """시각화 템플릿 추천 프롬프트를 시스템/사용자 메시지로 분리"""
    system_message = (
        "당신은 데이터 시각화 전문가입니다. "
        "제공된 데이터의 특성과 사용자 요구사항을 분석하여 최적의 시각화 템플릿을 추천해주세요.\n\n"

        "**사용 가능한 템플릿:**\n"
        "- time-series: 시계열 분석 (날짜 범위 선택, 트렌드 분석, 줌 기능) [구현됨]\n"
        "- kpi-dashboard: KPI 대시보드 (메트릭 카드, 게이지 차트, 목표 대비 분석) [구현됨]\n"
        "- comparison: 비교 분석 (그룹별 비교, 차이점 하이라이팅, 통계 요약) [구현됨]\n"
        "- geo-spatial: 지도 시각화 (히트맵, 마커 클러스터링, 지역별 통계) [개발 예정]\n"
        "- gantt-chart: 프로젝트 일정 (작업 타임라인, 진행률, 의존성 표시) [개발 예정]\n"
        "- heatmap: 히트맵 (밀도 분석, 패턴 인식, 다차원 데이터) [개발 예정]\n"
        "- network-graph: 네트워크 그래프 (관계 시각화, 노드 분석, 중심성 계산) [개발 예정]\n"
        "- custom: 사용자 정의 (자유로운 커스터마이징) [개발 예정]\n\n"

        "**응답 형식:** 반드시 아래 JSON 형식으로 응답해주세요.\n"
        "```json\n"
        "[\n"
        "  {\n"
        '    "visualization_type": "시각화_템플릿_종류",\n'
        '    "confidence": 95,\n'
        '    "reason": "추천 근거 (데이터 특성과 요구사항 기반)",\n'
        '    "data_requirements": ["필요한 데이터 컬럼들"],\n'
        '    "benefits": ["이 템플릿의 주요 장점들"]\n'
        "  }\n"
        "]\n"
        "```\n\n"

        "**분석 기준:**\n"
        "1. 데이터 유형 (시계열, KPI, 비교, 지리적, 일정, 상관관계, 네트워크)\n"
        "2. 비즈니스 목적 (모니터링, 분석, 보고, 의사결정)\n"
        "3. 사용자 요구사항 (상호작용 수준, 세부 정도, 업데이트 빈도)\n"
        "4. 구현된 템플릿 우선 권장 (time-series, kpi-dashboard, comparison)\n"
        "5. 최대 3개까지 추천 (신뢰도 순으로 정렬)"
    )

    # 데이터 기본 정보 수집
    preview = df.head(3).to_string(index=False)
    column_info = []

    for col in df.columns:
        dtype = str(df[col].dtype)
        null_count = df[col].isnull().sum()
        unique_count = df[col].nunique()
        sample_values = df[col].dropna().head(3).tolist()
        column_info.append(f"- {col}: {dtype} (고유값: {unique_count}, 결측값: {null_count}) 예시: {sample_values}")

    column_analysis = "\n".join(column_info)

    # 사용자 컨텍스트 처리
    context_section = ""
    if user_context:
        context_section = f"\n[사용자 요구사항 및 분석 목적]\n{user_context}\n"

    human_message = (
        f"다음 데이터를 분석하여 최적의 시각화 템플릿을 추천해주세요.\n\n"
        f"[데이터 정보]\n"
        f"- 총 행 수: {len(df):,}\n"
        f"- 총 컬럼 수: {len(df.columns)}\n\n"
        f"[컬럼 상세 정보]\n{column_analysis}\n"
        f"{context_section}"
        f"[데이터 미리보기 (상위 3개 행)]\n```\n{preview}\n```\n\n"
        f"위 정보를 종합하여 가장 적합한 시각화 템플릿을 JSON 형식으로 추천해주세요."
    )

    return system_message, human_message
