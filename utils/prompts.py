__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
from io import StringIO
from typing import List

# Third-party imports
import pandas as pd


###########################
# Topic selection prompts #
###########################
def set_topic_prompt(topic_prompt: str) -> str:
    return (
        f"{topic_prompt}\n---\n"
        f"코드 구현은 여기선 제외하고, 어떤 데이터가 필요한지 설명을 함께 해줘."
    )


def additional_query_prompt(topic_prompt: str, result: str) -> str:
    additional_prompt = input("✏️ 추가 수정 요청을 입력해주세요: ")
    return (
        f"다음 요청을 반영해 주제 설정 결과를 수정해주세요:"
        f"\n{topic_prompt}\n---\n{result}\n---\n"
        f"사용자 요청: {additional_prompt}"
    )


#######################
# Data loader prompts #
#######################
def generate_sample_prompt(topic_prompt: str, topic_result: str) -> str:
    return (
        f"{topic_prompt}\n\n"
        f"{topic_result}\n\n"
        f"위 주제를 기반으로, 관련된 샘플 데이터를 만들어주세요. "
        f"컬럼 이름과 100개 이상의 예시 row를 포함하여 표 형태로 출력해주세요."
        # TODO: TEST WIP
        # f"컬럼 이름과 10개의 예시 row를 포함하여 표 형태로 출력해주세요."
    )


def generate_download_link_prompt(topic_prompt: str) -> str:
    return (
        f"{topic_prompt}\n\n"
        f"이 주제와 관련된 CSV 또는 SQLite 형식의 오픈 데이터를 다운로드할 수 있는 URL 목록을 알려주세요."
    )


def recommend_removal_column_prompt(df: pd.DataFrame) -> str:
    """
    LLM에게 불필요해 보이는 컬럼을 추천해달라고 요청하는 프롬프트
    """
    preview = df.head(3).to_string(index=False)
    column_list = ", ".join(df.columns)

    prompt = (
        "아래는 현재 사용자가 제공한 데이터의 일부입니다.\n"
        "이 중에서 분석 또는 시각화 목적에 비추어 불필요하거나 중복되어 보이는 컬럼이 있다면 추천해주세요.\n"
        "1. 첫 번째 줄에는 삭제 추천 컬럼의 **정확한 컬럼명만 쉼표로 구분하여 출력**\n"
        "2. 두 번째 줄부터는 각 컬럼에 대한 삭제 이유를 항목별로 설명\n\n"
        f"[컬럼 목록]\n{column_list}\n\n"
        f"[미리보기 (상위 3개 row)]\n```\n{preview}\n```"
    )
    return prompt


def df_to_sqlite_with_col_filter_prompt(save_path: str, df: pd.DataFrame, to_drop: List[str]) -> str:
    """
    DataFrame을 CSV 형식으로 직렬화하고, SQLite 테이블화 요청 프롬프트로 변환
    """
    available_columns = list(df.columns)
    to_drop = ", ".join(to_drop)

    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_text = csv_buffer.getvalue()

    prompt = (
        f"다음은 사용자가 제공한 CSV 형식의 데이터입니다.\n"
        f"```\n{csv_text}\n```\n\n"
        f"요청 사항:\n"
        f"1. 테이블 이름은 내용에 어울리게 적절히 설정해 주세요.\n"
        f"2. 모든 컬럼은 자동으로 데이터 타입을 추론하여 생성하세요.\n"
        f"3. 단, 다음 사용자가 삭제하고자 입력한 컬럼명을 기준으로 불필요한 컬럼을 삭제하고 DB를 생성하세요.\n"
        f"   - 입력한 컬럼명이 일부만 입력되었거나 축약되었더라도, 실제 컬럼 중 의미상 가장 가까운 컬럼을 삭제 대상으로 간주하세요.\n\n"
        f"[삭제 희망 컬럼 입력]\n"
        f"{', '.join(to_drop)}\n\n"
        f"[현재 실제 컬럼 목록]\n"
        f"{available_columns}\n\n"
        f"이후 남은 컬럼들만으로 SQLite 데이터베이스를 생성해 주세요.\n"
        f"저장 경로는 사용자가 지정한 위치를 사용하고, 파일명은 자동으로 결정해도 좋습니다.\n"
        f"저장 경로{save_path}"
    )
    return prompt
