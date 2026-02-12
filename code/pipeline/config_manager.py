"""
실험 설정 관리자

YAML 기반 실험 설정 로드 및 검증
"""

import yaml
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
from pathlib import Path


@dataclass
class ExperimentConfig:
    """실험 설정 클래스"""

    # 모델 설정
    model_type: str  # "zone", "lstm", "gbm"
    model_params: Dict[str, Any] = field(default_factory=dict)

    # 데이터 설정
    use_cache: bool = True
    sample_rate: float = 1.0  # 빠른 실험: 0.1 (10% 샘플)

    # CV 설정
    n_folds: int = 5
    fold_indices: Optional[List[int]] = None  # None = 전체, [1,3] = Fold 1,3만
    random_state: int = 42

    # 검증 설정
    check_leakage: bool = True
    sweet_spot_range: tuple = (16.27, 16.34)

    # 제출 설정
    auto_submit: bool = False
    submit_threshold: float = 0.5  # Gap 추정 < 0.5면 제출 추천

    # 메타데이터
    experiment_name: Optional[str] = None
    description: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """
        YAML 파일에서 설정 로드

        Args:
            path: YAML 파일 경로

        Returns:
            ExperimentConfig 인스턴스

        Raises:
            FileNotFoundError: YAML 파일이 없는 경우
            ValueError: 필수 필드 누락 또는 유효하지 않은 값
        """
        yaml_path = Path(path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {path}")

        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        # 필수 필드 체크
        if 'model_type' not in config_dict:
            raise ValueError("model_type 필드가 필수입니다")

        # sweet_spot_range 튜플 변환 (YAML은 리스트로 로드)
        if 'sweet_spot_range' in config_dict and isinstance(config_dict['sweet_spot_range'], list):
            config_dict['sweet_spot_range'] = tuple(config_dict['sweet_spot_range'])

        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return asdict(self)

    def to_yaml(self, path: str):
        """
        YAML 파일로 저장

        Args:
            path: 저장할 YAML 파일 경로
        """
        config_dict = self.to_dict()

        # 튜플을 리스트로 변환 (YAML 호환)
        if isinstance(config_dict.get('sweet_spot_range'), tuple):
            config_dict['sweet_spot_range'] = list(config_dict['sweet_spot_range'])

        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)

    def validate(self) -> bool:
        """
        설정 유효성 검증

        Returns:
            유효하면 True

        Raises:
            ValueError: 유효하지 않은 설정
        """
        # 모델 타입 체크
        valid_models = ['zone', 'lstm', 'gbm', 'ensemble']
        if self.model_type not in valid_models:
            raise ValueError(f"지원하지 않는 모델: {self.model_type}. 사용 가능: {valid_models}")

        # 샘플링 비율 체크
        if not 0 < self.sample_rate <= 1.0:
            raise ValueError(f"sample_rate는 0과 1 사이여야 합니다: {self.sample_rate}")

        # Fold 수 체크
        if self.n_folds < 2:
            raise ValueError(f"n_folds는 2 이상이어야 합니다: {self.n_folds}")

        # Fold indices 체크
        if self.fold_indices:
            if any(idx < 1 or idx > self.n_folds for idx in self.fold_indices):
                raise ValueError(f"fold_indices는 1~{self.n_folds} 범위여야 합니다: {self.fold_indices}")

        # Sweet spot 범위 체크
        if len(self.sweet_spot_range) != 2:
            raise ValueError(f"sweet_spot_range는 (min, max) 형태여야 합니다: {self.sweet_spot_range}")

        sweet_min, sweet_max = self.sweet_spot_range
        if sweet_min >= sweet_max:
            raise ValueError(f"sweet_spot_range의 min < max 여야 합니다: {self.sweet_spot_range}")

        return True

    def __str__(self) -> str:
        """사람이 읽기 쉬운 문자열 표현"""
        lines = [
            "ExperimentConfig:",
            f"  Model: {self.model_type}",
            f"  Params: {self.model_params}",
            f"  Sample: {self.sample_rate*100:.1f}%",
            f"  Folds: {self.fold_indices or 'All'}",
            f"  Sweet Spot: {self.sweet_spot_range}",
        ]
        if self.experiment_name:
            lines.insert(1, f"  Name: {self.experiment_name}")
        return "\n".join(lines)


def create_quick_config(base_config: ExperimentConfig) -> ExperimentConfig:
    """
    빠른 실험용 설정 생성

    Args:
        base_config: 기본 설정

    Returns:
        빠른 실험용 설정 (10% 샘플, Fold 1,3만)
    """
    quick_config = ExperimentConfig(
        model_type=base_config.model_type,
        model_params=base_config.model_params.copy(),
        use_cache=True,
        sample_rate=0.1,  # 10% 샘플
        n_folds=base_config.n_folds,
        fold_indices=[1, 3],  # Fold 1, 3만
        random_state=base_config.random_state,
        check_leakage=base_config.check_leakage,
        sweet_spot_range=base_config.sweet_spot_range,
        auto_submit=False,  # 빠른 실험은 제출 안함
        submit_threshold=base_config.submit_threshold,
        experiment_name=f"{base_config.experiment_name}_quick" if base_config.experiment_name else "quick",
        description=f"Quick test of {base_config.experiment_name or base_config.model_type}"
    )
    return quick_config
