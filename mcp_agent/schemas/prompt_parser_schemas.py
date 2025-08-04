__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"

# Standard imports
from typing import List, Dict, Optional
from enum import Enum

# Third-party imports
from pydantic import BaseModel, Field, field_validator


class VisualizationStatus(Enum):
    """시각화 구현 상태"""
    IMPLEMENTED = "구현됨"
    PLANNED = "개발 예정"


class VisualizationType(Enum):
    """시각화 타입 정의"""
    TIME_SERIES = "time-series"
    KPI_DASHBOARD = "kpi-dashboard"
    COMPARISON = "comparison"
    GEO_SPATIAL = "geo-spatial"
    GANTT_CHART = "gantt-chart"
    HEATMAP = "heatmap"
    NETWORK_GRAPH = "network-graph"
    CUSTOM = "custom"

    @property
    def description(self) -> str:
        """시각화 타입 설명"""
        descriptions = {
            self.TIME_SERIES: "시계열 분석 (날짜 범위 선택, 트렌드 분석, 줌 기능)",
            self.KPI_DASHBOARD: "KPI 대시보드 (메트릭 카드, 게이지 차트, 목표 대비 분석)",
            self.COMPARISON: "비교 분석 (그룹별 비교, 차이점 하이라이팅, 통계 요약)",
            self.GEO_SPATIAL: "지도 시각화 (히트맵, 마커 클러스터링, 지역별 통계)",
            self.GANTT_CHART: "프로젝트 일정 (작업 타임라인, 진행률, 의존성 표시)",
            self.HEATMAP: "히트맵 (밀도 분석, 패턴 인식, 다차원 데이터)",
            self.NETWORK_GRAPH: "네트워크 그래프 (관계 시각화, 노드 분석, 중심성 계산)",
            self.CUSTOM: "사용자 정의 (자유로운 커스터마이징)"
        }
        return descriptions.get(self, "설명 없음")

    @property
    def status(self) -> VisualizationStatus:
        """구현 상태"""
        implemented_types = {
            self.TIME_SERIES,
            self.KPI_DASHBOARD,
            self.COMPARISON
        }

        if self in implemented_types:
            return VisualizationStatus.IMPLEMENTED
        else:
            return VisualizationStatus.PLANNED

    @property
    def is_implemented(self) -> bool:
        """구현 완료 여부"""
        return self.status == VisualizationStatus.IMPLEMENTED

    @classmethod
    def get_implemented_types(cls) -> List['VisualizationType']:
        """구현된 시각화 타입 목록"""
        return [vt for vt in cls if vt.is_implemented]

    @classmethod
    def get_planned_types(cls) -> List['VisualizationType']:
        """개발 예정인 시각화 타입 목록"""
        return [vt for vt in cls if not vt.is_implemented]

    @classmethod
    def get_all_values(cls) -> List[str]:
        """모든 시각화 타입 값 목록"""
        return [vt.value for vt in cls]

    @classmethod
    def get_implemented_values(cls) -> List[str]:
        """구현된 시각화 타입 값 목록"""
        return [vt.value for vt in cls.get_implemented_types()]

    @classmethod
    def get_all_info(cls) -> Dict[str, Dict[str, str]]:
        """모든 시각화 타입 정보"""
        return {
            vt.value: {
                "name": vt.value,
                "description": vt.description,
                "status": vt.status.value,
                "is_implemented": vt.is_implemented
            }
            for vt in cls
        }

    @classmethod
    def from_string(cls, value: str) -> 'VisualizationType':
        """문자열로부터 VisualizationType 생성"""
        for vt in cls:
            if vt.value == value:
                return vt
        raise ValueError(f"Unknown visualization type: {value}")

    @classmethod
    def is_valid_template_id(cls, template_id: str) -> bool:
        """유효한 템플릿 ID인지 확인"""
        try:
            cls.from_string(template_id)
            return True
        except ValueError:
            return False

    @classmethod
    def is_implemented_template_id(cls, template_id: str) -> bool:
        """구현된 템플릿 ID인지 확인"""
        try:
            vt = cls.from_string(template_id)
            return vt.is_implemented
        except ValueError:
            return False

    def __str__(self) -> str:
        return f"{self.value}: {self.description} [{self.status.value}]"


class VisualizationRecommendation(BaseModel):
    """단일 시각화 템플릿 추천 정보"""
    visualization_type: VisualizationType = Field(..., description="시각화 템플릿 종류"),
    confidence: int = Field(..., ge=0, le=100, description="신뢰도 점수 (0-100)")
    reason: str = Field(..., description="추천 근거")
    data_requirements: List[str] = Field(..., description="필요한 데이터 컬럼들")
    benefits: List[str] = Field(..., description="템플릿의 주요 장점들")

    @field_validator('visualization_type', mode='before')
    @classmethod
    def validate_visualization_type(cls, v):
        """문자열을 VisualizationType Enum으로 자동 변환"""
        if isinstance(v, str):
            try:
                return VisualizationType.from_string(v)
            except ValueError:
                valid_types = VisualizationType.get_all_values()
                raise ValueError(f"Invalid visualization_type: {v}. Must be one of {valid_types}")
        elif isinstance(v, VisualizationType):
            return v
        else:
            raise ValueError(f"visualization_type must be string or VisualizationType, got {type(v)}")

    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v):
        if not 0 <= v <= 100:
            raise ValueError("Confidence must be between 0 and 100")
        return v

    @property
    def template_id(self) -> str:
        """하위 호환성을 위한 template_id 속성"""
        return self.visualization_type.value

    @property
    def is_implemented(self) -> bool:
        """이 추천의 템플릿이 구현되었는지 확인"""
        return self.visualization_type.is_implemented

    @property
    def template_description(self) -> str:
        """템플릿 설명"""
        return self.visualization_type.description

    @property
    def template_status(self) -> VisualizationStatus:
        """템플릿 구현 상태"""
        return self.visualization_type.status


class VisualizationRecommendationResponse(BaseModel):
    """시각화 템플릿 추천 목록"""
    user_context: Optional[str] = Field(
        None,
        description="주제 설정 대화 맥락 요약",
        examples=["XX시 피자 일매출 시각화를 위한 FE 제작을 수행하며.."]
    )
    recommendations: List[VisualizationRecommendation] = Field(..., description="추천 목록")

    @field_validator('recommendations')
    @classmethod
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
        return [rec for rec in self.recommendations if rec.is_implemented]

    def get_planned_recommendations(self) -> List[VisualizationRecommendation]:
        """개발 예정인 템플릿만 필터링하여 반환"""
        return [rec for rec in self.recommendations if not rec.is_implemented]

    def get_recommendations_by_status(self, status: VisualizationStatus) -> List[VisualizationRecommendation]:
        """상태별로 추천 필터링"""
        return [rec for rec in self.recommendations if rec.template_status == status]

    def get_recommendations_by_type(self, visualization_type: VisualizationType) -> List[VisualizationRecommendation]:
        """특정 시각화 타입의 추천만 필터링"""
        return [rec for rec in self.recommendations if rec.visualization_type == visualization_type]

    @property
    def has_implemented_recommendations(self) -> bool:
        """구현된 추천이 있는지 확인"""
        return len(self.get_implemented_recommendations()) > 0

    @property
    def implementation_summary(self) -> Dict[str, int]:
        """구현 상태별 추천 개수 요약"""
        implemented = self.get_implemented_recommendations()
        planned = self.get_planned_recommendations()

        return {
            "total": len(self.recommendations),
            "implemented": len(implemented),
            "planned": len(planned)
        }
