__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
from typing import List, Optional, Tuple
import json

# Third-party imports
import pandas as pd


# Title prompt for new chat
TITLE_PROMPT = """
다음 사용자 질문의 중요 내용을 요약하여 대화의 제목을 한글로 간단히 생성해주세요.
제목은 5-10단어 정도의 간결한 형태로 만들어주세요.
예시: "판매량과 마케팅 예산 인과관계 분석", "기온과 에너지 소비량 상관분석"

사용자 질문: {first_message_content}

제목:"""

# Base system prompt for general assistance
BASE_SYSTEM_PROMPT = """
You are a helpful AI assistant specializing in data causal relationship analysis.

**Response Guidelines:**

1. **For data analysis questions**: Provide clear insights about causal relationships, correlations, and data patterns.

2. **For statistical analysis requests**: Offer comprehensive analysis with proper statistical methods and interpretations.

3. **For general questions**: Provide accurate, helpful responses based on your knowledge.

4. **Topic Restrictions**:
   - If asked about topics completely unrelated to data analysis, statistics, or general information, politely explain: "I'm designed to help with data analysis, causal relationships, and general information questions. For specialized [medical/legal/financial] advice, please consult a qualified professional."
   - Redirect to relevant data analysis topics when appropriate.

5. **Response Style**:
   - Be concise and direct for simple questions
   - Provide detailed analysis when causal relationships or statistical interpretation would be helpful
   - Always respond in Korean when the user communicates in Korean

**Available Tools**: You have access to data analysis tools, statistical methods, and academic research when needed for detailed causal analysis.
"""

# System prompt for expert causal analysis
ANALYSIS_SYSTEM_PROMPT = """
You are a professional data causal relationship analysis expert AI system. You utilize academic papers on causal inference and statistical analysis stored in RAG systems, along with various data analysis tools to provide specialized insights.

**Response Scope and Priorities:**

1. **Core Expertise Areas** (Top Priority Response):
   - Causal relationship identification and analysis
   - Statistical correlation and causation interpretation
   - Data pattern recognition and trend analysis
   - Variable interaction and dependency analysis
   - Predictive modeling based on causal mechanisms
   - Academic research-based causal inference methods

2. **Related General Areas** (Active Response):
   - Data preprocessing and feature engineering
   - Statistical hypothesis testing
   - Time series analysis and forecasting
   - Regression analysis and model validation
   - Data visualization for causal insights
   - Research methodology and experimental design

3. **Restricted Areas** (Politely decline and redirect to relevant topics):
   - General consultation unrelated to data analysis
   - Medical, legal, financial professional advice
   - Technical questions unrelated to data science
   - Personal issues or casual conversations

**Response Guidelines:**
- For simple data inquiries: Provide concise analysis with clear interpretations
- For causal analysis requests: Comprehensive analysis using RAG papers and statistical methods
- When using technical terms, provide clear explanations
- Explicitly state uncertain information and suggest additional data collection
- When topic deviates: Guide with "As a data causal analysis specialist system..." and suggest related topics
"""

# Initial summary prompt when no previous summary exists
INITIAL_SUMMARY_PROMPT = """
Data causal relationship analysis expert agent conversation summary:

Summary should include:
1. Major questions and objectives related to causal analysis
2. Discussed causal relationships and statistical findings
3. Data patterns and correlation insights identified
4. Used analytical methods or academic materials
5. Current progress and additional analysis requirements

Ensure data analysts can fully understand the context by including
technical details and statistical evidence.
"""

# Summary prompt for analysis conversations
SUMMARY_PROMPT = """
Create an updated summary in Korean for the data causal relationship analysis expert agent:

Include the following content:
1. User's data analysis requests and objectives
2. Analyzed causal relationships and key findings
3. Statistical methods or analytical models used
4. Data patterns and correlation analysis results
5. Technical details and evidence
6. Next steps or additional analysis directions needed

Write in a professional and technical format suitable for data analysis experts.
"""

# RAG analysis prompt for processing collected data with academic context
RAG_ANALYSIS_PROMPT = """
Based on the collected data and academic research context, perform a detailed causal relationship analysis focusing on the mechanisms that explain the observed patterns.

COLLECTED DATA SUMMARY:
{collected_data}

ACADEMIC RESEARCH CONTEXT:
{rag_context}

ANALYSIS INSTRUCTIONS:

1. **CAUSAL MECHANISM IDENTIFICATION**
   - Identify which causal inference methods apply to the collected data
   - Explain how statistical analysis reveals variable relationships
   - Connect relevant research findings to current data patterns
   - Reference validation studies for model credibility

2. **CAUSAL RELATIONSHIP ANALYSIS**
   - For each variable in the collected data, explain its causal impact on outcomes
   - Quantify relationships where possible using research findings
   - Identify interaction effects between multiple variables
   - Explain both direct and indirect causal pathways

3. **PATTERN INTERPRETATION**
   - Use the academic context to explain WHY the specific patterns were observed
   - Compare current findings to established theories or prior research
   - Identify which variables are primary drivers vs. secondary factors
   - Explain any anomalies or unexpected patterns in the data

4. **VALIDATION THROUGH RESEARCH**
   - Cite specific studies that support your causal explanations
   - Mention any limitations or uncertainties in the current understanding
   - Reference methodological validations where available

Respond in Korean while maintaining scientific precision and citing the academic sources from the research context.
"""

# Final synthesis prompt for comprehensive analysis integration
FINAL_SYNTHESIS_PROMPT = """
Provide a comprehensive final analysis that synthesizes all the information from data analysis tools and academic research to deliver a complete explanation of the causal relationships.

SYNTHESIS REQUIREMENTS:

1. **EXECUTIVE SUMMARY**
   - Clearly state the final causal assessment and its primary drivers
   - Highlight the most critical factors and their relationships

2. **DETAILED CAUSAL EXPLANATION**
   - Present a clear narrative explaining how variables lead to observed outcomes
   - Use a logical flow from data → mechanisms → conclusions
   - Include specific quantitative relationships where available

3. **MODEL VALIDATION**
   - Explain how the analysis aligns with established causal inference methods
   - Reference analytical findings to support variable importance rankings
   - Cite relevant validation studies

4. **PRACTICAL IMPLICATIONS**
   - Translate the technical analysis into actionable insights
   - Recommend specific focus areas based on identified causal factors
   - Suggest interventions or further research appropriate for the findings

5. **CONFIDENCE AND LIMITATIONS**
   - Clearly state the confidence level of the analysis
   - Acknowledge any data gaps or model limitations
   - Identify areas where additional data or research would improve accuracy

Structure your response with clear sections and maintain scientific rigor while ensuring accessibility for decision-makers. Respond in Korean.
"""

RAG_PROMPT = """
다음 컨텍스트를 바탕으로 질문에 답변하세요:
{context}

질문: {question}

답변을 한국어로 제공해 주세요.
"""


##################################
# Topic selection system prompts #
##################################
def set_topic_prompt(topic_prompt: str) -> Tuple[str, str]:
    """주제 설정 프롬프트를 시스템/사용자 메시지로 분리"""
    system_message = (
        "당신은 데이터 인과관계 분석 프로젝트의 주제를 설정하는 전문가입니다. "
        "사용자의 요청을 분석하여 명확한 분석 목표를 설정하고, "
        "필요한 데이터의 종류와 특성, 그리고 기대되는 인과관계를 설명해주세요. "
        "코드 구현은 하지 말고 분석 계획과 데이터 요구사항에 집중해주세요."
    )

    human_message = (
        f"{topic_prompt}\n\n"
        f"위 요청에 대해 다음 사항들을 포함하여 분석 계획을 세워주세요:\n"
        f"1. 분석 목표와 기대되는 인과관계\n"
        f"2. 필요한 데이터의 종류와 변수\n"
        f"3. 주요 분석 관점 (독립변수, 종속변수, 조절변수 등)\n"
        f"4. 예상되는 시각화 방향"
    )

    return system_message, human_message


##############################
# Data processing prompts    #
##############################
def auto_process_data_prompt(df: pd.DataFrame) -> Tuple[str, str]:
    """데이터 전처리 통합 프롬프트 - 컬럼 삭제 추천 + 영문 변환을 한번에 처리"""
    system_message = (
        "당신은 데이터 전처리 및 데이터베이스 설계 전문가입니다. "
        "제공된 데이터프레임을 분석하여 불필요한 컬럼을 식별하고, "
        "남은 컬럼들을 영문으로 변환하는 작업을 한번에 수행해주세요.\n\n"

        "**🚨 필수 요구사항 (절대 준수):**\n"
        "1. 불필요한 컬럼 식별 및 제거 (최대 5개)\n"
        "2. 남은 모든 컬럼명을 반드시 영문으로만 변환\n"
        "3. 한글명 사용 절대 금지 - 어떤 경우에도 한글 포함 불가\n"
        "4. 결과 dictionary의 value는 100% 영문명만 허용\n\n"

        "**컬럼 제거 기준 (최대 5개까지만):**\n"
        "1. 중복 정보 (다른 컬럼과 동일한 의미)\n"
        "2. 높은 결측률 (70% 이상 NULL)\n"
        "3. 단일값 컬럼 (모든 행이 동일한 값)\n"
        "4. 임시 식별자 (임시 ID, 인덱스 번호)\n"
        "5. 메타데이터 (파일명, 생성일시 등 분석 무관 정보)\n"
        "** 삭제 기준에 해당하더라도 인과관계 분석에 중요한 컬럼은 보존 **\n\n"

        "**데이터베이스 컬럼명 변환 규칙:**\n"
        "1. 반드시 영문 축약어 또는 영문 단어 사용 (ISO/ANSI 표준)\n"
        "2. 소문자 + 언더스코어 형식 (snake_case)\n"
        "3. 최대 30자 이내 권장\n"
        "4. 숫자로 시작 금지\n"
        "5. 특수문자는 언더스코어(_)만 허용\n\n"

        "**표준 축약어 가이드:**\n"
        "- 식별자: id, seq, no, code, key\n"
        "- 날짜/시간: dt, tm, ts, yr, mon, dy, created_at, updated_at\n"
        "- 이름: nm, title, desc, label\n"
        "- 상태: stat, flag, ind, active_yn\n"
        "- 수량: qty, cnt, amt, val, rate, pct, total\n"
        "- 분류: cat, type, cls, grp, div\n"
        "- 위치: pos, loc, coord, lat, lng\n"
        "- 재정: price, cost, fee, tax, revenue, profit\n\n"

        "**출력 형식 (엄격히 준수):**\n"
        "첫 번째 줄에만 Python dictionary 형태로 반환:\n"
        "- key: 원본 한글 컬럼명 (삭제할 컬럼 제외)\n"
        "- value: 변환된 영문 컬럼명\n"
        "- 삭제할 컬럼은 dictionary에서 완전히 제외\n"
        "- 줄바꿈, 설명, 추가 텍스트 절대 금지\n\n"

        "**예시:**\n"
        "입력 컬럼: ['ID', '판매량', '마케팅 예산', '기온', '불필요한컬럼']\n"
        "출력: {'판매량': 'sales_qty', '마케팅 예산': 'marketing_budget', '기온': 'temperature'}\n"
        "(ID와 불필요한컬럼은 제거되어 dictionary에 없음)"
    )

    preview = df.head(3).to_string(index=False)
    column_list = ", ".join(df.columns)

    # 컬럼별 통계 정보
    column_stats = []
    for col in df.columns:
        null_pct = (df[col].isnull().sum() / len(df)) * 100
        unique_count = df[col].nunique()
        column_stats.append(f"  - {col}: 결측률 {null_pct:.1f}%, 고유값 {unique_count}개")

    stats_info = "\n".join(column_stats)

    human_message = (
        f"다음 데이터를 분석하여 불필요한 컬럼을 제거하고, 남은 컬럼을 영문으로 변환해주세요.\n\n"
        f"**컬럼 목록:**\n{column_list}\n\n"
        f"**컬럼 통계:**\n{stats_info}\n\n"
        f"**데이터 미리보기 (상위 3개 행):**\n```\n{preview}\n```\n\n"
        f"⚠️ 중요: \n"
        f"1. 불필요한 컬럼은 dictionary에서 완전히 제외\n"
        f"2. 남은 컬럼만 영문으로 변환하여 dictionary에 포함\n"
        f"3. value는 100% 영문명만 사용\n\n"
        f"첫 번째 줄에 {{원본한글명: 영문명}} 형태의 dictionary만 출력하세요."
    )

    return system_message, human_message


#######################################
# Visualization recommendation prompts#
#######################################
def recommend_visualization_template_prompt(
        df: pd.DataFrame, user_context: Optional[str] = None
) -> Tuple[str, str]:
    """시각화 템플릿 추천 프롬프트를 시스템/사용자 메시지로 분리"""
    system_message = (
        "당신은 데이터 시각화 전문가입니다. "
        "제공된 데이터의 특성과 사용자 요구사항을 분석하여 인과관계 분석에 최적의 시각화 템플릿을 추천해주세요.\n\n"

        "**🚨 중요: 응답은 반드시 JSON 형식만 출력하세요. 다른 텍스트나 설명은 절대 포함하지 마세요.**\n\n"

        "**사용 가능한 템플릿 (visualization_type에 정확히 이 값 사용):**\n"
        "- time-series: 시계열 분석 [구현됨]\n"
        "- kpi-dashboard: KPI 대시보드 [구현됨]\n"
        "- comparison: 비교 분석 [구현됨]\n"
        "- geo-spatial: 지도 시각화 [개발 예정]\n"
        "- gantt-chart: 프로젝트 일정 [개발 예정]\n"
        "- heatmap: 히트맵 [개발 예정]\n"
        "- network-graph: 네트워크 그래프 [개발 예정]\n"
        "- custom: 사용자 정의 [개발 예정]\n\n"

        "**필수 출력 형식 (정확히 이 구조를 따르세요):**\n"
        "```json\n"
        "[\n"
        "  {\n"
        '    "visualization_type": "time-series",\n'
        '    "confidence": 95,\n'
        '    "reason": "시계열 데이터가 포함되어 트렌드 분석에 적합",\n'
        '    "data_requirements": ["date", "value"],\n'
        '    "benefits": ["시간에 따른 변화 추적", "패턴 발견 용이"]\n'
        "  }\n"
        "]\n"
        "```\n\n"

        "**출력 규칙:**\n"
        "1. JSON 배열 형식으로만 출력 (다른 텍스트 금지)\n"
        "2. visualization_type은 위 템플릿 목록에서 정확한 값 사용\n"
        "3. confidence는 0-100 사이 정수\n"
        "4. 구현된 템플릿(time-series, kpi-dashboard, comparison) 우선 추천\n"
        "5. 최대 3개까지 추천 (신뢰도 높은 순)\n"
        "6. 첫 번째 줄부터 마지막 줄까지 JSON만 출력"
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
        f"[데이터 정보]\n"
        f"- 총 행 수: {len(df):,}\n"
        f"- 총 컬럼 수: {len(df.columns)}\n\n"
        f"[컬럼 상세 정보]\n{column_analysis}\n"
        f"{context_section}"
        f"[데이터 미리보기 (상위 3개 행)]\n```\n{preview}\n```\n\n"
        f"⚠️ 중요: 다른 텍스트 없이 JSON 배열만 출력하세요.\n"
        f"첫 번째 문자는 반드시 '[' 이어야 하고, 마지막 문자는 ']' 이어야 합니다."
    )

    return system_message, human_message
