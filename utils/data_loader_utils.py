__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
import os
from io import StringIO
from typing import List, Optional
import sqlite3

# Third-party imports
import pandas as pd
import chardet


def load_files() -> pd.DataFrame:
    print("\n📁 CSV 또는 SQLite 파일 경로를 입력하세요. 쉼표(,)로 여러 개 입력 가능합니다.")
    file_input = input("파일 경로들: ").strip()
    paths = [path.strip() for path in file_input.split(",")]
    return load_local_files(paths)


def detect_file_encoding(path: str, num_bytes: int = 10000) -> str:
    with open(path, 'rb') as f:
        raw_data = f.read(num_bytes)
    result = chardet.detect(raw_data)
    return result['encoding'] or 'utf-8'


def load_local_files(file_paths: List[str]) -> Optional[pd.DataFrame]:
    dataframes = []
    for path in file_paths:
        if not os.path.exists(path):
            print(f"❌ 파일 없음: {path}")
            continue
        try:
            if path.endswith(".csv"):
                encoding = detect_file_encoding(path)
                df_part = pd.read_csv(path, encoding=encoding)
            elif path.endswith(".sqlite") or path.endswith(".db"):
                with sqlite3.connect(path) as conn:
                    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
                    table_name = tables['name'].iloc[0]
                    df_part = pd.read_sql(f"SELECT * FROM {table_name}", conn)
            else:
                print(f"⚠️ 지원되지 않는 파일 형식: {path}")
                continue
            dataframes.append(df_part)
        except Exception as e:
            print(f"⚠️ 오류 발생: {e} (파일: {path})")
    if dataframes:
        return pd.concat(dataframes, ignore_index=True)
    return None


def markdown_table_to_df(text: str) -> Optional[pd.DataFrame]:
    try:
        if "|" in text:
            lines = [line[1:-1] for line in text.splitlines() if "|" in line and "---" not in line]
            parsed_data = "\n".join(lines)
        else:
            parsed_data = text
        return pd.read_csv(StringIO(parsed_data), sep="|")
    except Exception as e:
        print(f"⚠️ 샘플 데이터 파싱 오류: {e}")
        return None
