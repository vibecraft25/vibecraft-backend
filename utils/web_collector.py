__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
import os

# Third-party imports
import pandas as pd
from serpapi import GoogleSearch

# TODO: WIP
def collect_web_data(topic: str, max_results: int = 10, output_csv: str = "collected_data.csv") -> dict:
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        return {"status": "error", "message": "환경변수 SERPAPI_API_KEY가 설정되지 않았습니다."}

    search = GoogleSearch({
        "q": topic,
        "location": "South Korea",
        "hl": "ko",
        "gl": "kr",
        "api_key": api_key,
        "num": max_results
    })

    results = search.get_dict()
    if "organic_results" not in results:
        return {"status": "error", "message": "검색 결과가 없습니다."}

    articles = [{
        "title": item.get("title"),
        "link": item.get("link"),
        "snippet": item.get("snippet"),
        "source": item.get("source")
    } for item in results["organic_results"]]

    df = pd.DataFrame(articles)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    return {
        "status": "success",
        "message": f"{len(df)}개의 데이터를 저장했습니다.",
        "csv_path": output_csv
    }
