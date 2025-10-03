__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class DocumentSearchResult:
    """검색 결과 데이터 클래스"""
    file_path: str
    content: str
    score: float
    metadata: Dict[str, Any]
