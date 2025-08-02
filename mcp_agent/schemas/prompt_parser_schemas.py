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


class VisualizationRecommendationList(BaseModel):
    """시각화 템플릿 추천 목록"""
    recommendations: List[VisualizationRecommendation] = Field(..., description="추천 목록")

    @field_validator('recommendations')
    def validate_recommendations_count(cls, v):
        if len(v) > 3:
            raise ValueError("최대 3개까지만 추천 가능합니다")
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
