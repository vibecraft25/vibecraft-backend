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
        "주어진 DataFrame의 컬럼명을 데이터베이스 표준에 맞게 변환하고, "
        "불필요한 컬럼을 제거하는 작업을 수행해주세요.\n\n"

        "**작업 규칙:**\n"
        "1. 컬럼명을 데이터베이스 명명 규칙에 맞게 영문 축약어로 변경\n"
        "2. 모든 DB 예약어는 컬럼명으로 사용 금지\n"
        "3. 삭제 대상 컬럼은 의미를 유추하여 판단하고 제거\n"
        "4. 변경된 컬럼명 매핑을 dictionary 형태로 반환\n\n"

        "**출력 형식:**\n"
        "첫 번째 줄에 이전 이름과 변경된 이름의 매핑을 dictionary 형식으로 한 줄로 반환하세요. "
        "줄바꿈 없이 반환하세요."
    )

    available_columns = list(df.columns)
    to_drop_str = ", ".join(to_drop)

    human_message = (
        f"다음 정보를 바탕으로 컬럼명을 변환하고 불필요한 컬럼을 제거해주세요.\n\n"
        f"**삭제 대상 키워드:** {to_drop_str}\n"
        f"**현재 컬럼 목록:** {available_columns}\n\n"
        f"가장 첫 줄에 이전 이름과 변경된 이름의 매핑을 dictionary 형식으로 한 줄로 반환해주세요."
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
