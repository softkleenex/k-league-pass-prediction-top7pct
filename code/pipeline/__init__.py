"""
K리그 패스 예측 자동화 파이프라인

실험 → 검증 → 제출 전 과정 자동화
"""

__version__ = "1.0.0"
__author__ = "Backend Developer (Claude Code)"

from .config_manager import ExperimentConfig
from .data_pipeline import DataPipeline
from .model_trainer import ModelTrainer
from .auto_validator import AutoValidator
from .submission_manager import SubmissionManager
from .orchestrator import PipelineOrchestrator

__all__ = [
    "ExperimentConfig",
    "DataPipeline",
    "ModelTrainer",
    "AutoValidator",
    "SubmissionManager",
    "PipelineOrchestrator",
]
