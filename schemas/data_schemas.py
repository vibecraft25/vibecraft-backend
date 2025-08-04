__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
from datetime import datetime
from typing import Dict

# Third-party imports
from pydantic import BaseModel, Field


class DatasetMetadata(BaseModel):
    """데이터셋 메타데이터 모델"""
    created_at: datetime = Field(..., description="메타데이터 생성 시간")
    column_mapping: Dict[str, str] = Field(..., description="컬럼 매핑 정보")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        json_schema_extra = {
            "example": {
                "created_at": "2025-07-31T16:39:34.548399",
                "column_mapping": {
                    "날짜": "order_dt",
                    "주문 번호": "order_id",
                    "주문 시간": "order_tm",
                    "피자 종류": "pizza_nm",
                    "피자 사이즈": "pizza_sz",
                    "개별 피자 판매 수량": "item_qty",
                    "개별 피자 매출액": "item_rev",
                    "할인 금액": "disc_amt",
                    "주문 채널": "order_chnl"
                }
            }
        }
