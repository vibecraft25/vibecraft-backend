__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class VisualizationRecommendation(BaseModel):
    """단일 시각화 템플릿 추천 정보"""
    template_id: str = Field(..., description="템플릿 ID")
    confidence: int = Field(..., ge=0, le=100, description="신뢰도 점수 (0-100)")
    reason: str = Field(..., description="추천 근거")
    data_requirements: List[str] = Field(..., description="필요한 데이터 컬럼들")
    benefits: List[str] = Field(..., description="템플릿의 주요 장점들")

    @field_validator('template_id')
    def validate_template_id(cls, v):
        valid_templates = {
            'time-series', 'kpi-dashboard', 'comparison', 'geo-spatial',
            'gantt-chart', 'heatmap', 'network-graph', 'custom'
        }
        if v not in valid_templates:
            raise ValueError(f"Invalid template_id: {v}. Must be one of {valid_templates}")
        return v

    @field_validator('confidence')
    def validate_confidence(cls, v):
        if not 0 <= v <= 100:
            raise ValueError("Confidence must be between 0 and 100")
        return v


class VisualizationRecommendationResponse(BaseModel):
    """시각화 템플릿 추천 목록"""
    user_context: Optional[str] = Field(None, description="주제 설정 대화 맥락 요약", examples=["XX시 피자 일매출 시각화를 위한 FE 제작을 수행하며.."])
    recommendations: List[VisualizationRecommendation] = Field(..., description="추천 목록")

    @field_validator('recommendations')
    def validate_recommendations_count(cls, v):
        """3개를 초과하는 경우 신뢰도 기준으로 상위 3개만 선택"""
        if len(v) > 3:
            # 신뢰도 기준으로 내림차순 정렬하여 상위 3개만 선택
            sorted_recommendations = sorted(v, key=lambda x: x.confidence, reverse=True)
            return sorted_recommendations[:3]
        return v

    def get_top_recommendation(self) -> Optional[VisualizationRecommendation]:
        """가장 신뢰도가 높은 추천을 반환"""
        if not self.recommendations:
            return None
        return max(self.recommendations, key=lambda x: x.confidence)

    def get_implemented_recommendations(self) -> List[VisualizationRecommendation]:
        """구현된 템플릿만 필터링하여 반환"""
        implemented_templates = {'time-series', 'kpi-dashboard', 'comparison'}
        return [rec for rec in self.recommendations if rec.template_id in implemented_templates]
