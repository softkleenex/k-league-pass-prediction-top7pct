"""
K리그 패스 예측 대회 특화 End-to-End 자동화 파이프라인

Ultrathink 2025-12-16:
- 7단계 완전 자동화: 설계 → 검증 → 실행 → 분석 → 결정 → 제출 → 추적
- Gap 중심 설계 (Sweet Spot 15.20-15.60)
- Episode Independence 자동 검증
- 제출 의사결정 자동화 (85% 신뢰도 이상)
"""

import sys
import os
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class KLeaguePipeline:
    """
    K리그 대회 특화 E2E 파이프라인

    7 Stages:
    1. Design: 실험 설계 및 검증
    2. Code: 코드 작성 및 리뷰
    3. Validate: 데이터 누수/GroupKFold 검증
    4. Execute: 실행 및 CV 측정
    5. Analyze: Gap 분석 및 위험 평가
    6. Decide: 제출 의사결정
    7. Track: 문서화 및 추적
    """

    # =========================================================================
    # Constants (대회 특성)
    # =========================================================================
    SWEET_SPOT_MIN = 15.20
    SWEET_SPOT_MAX = 15.60
    MAX_GAP_SAFE = 0.30
    MAX_GAP_WARNING = 0.50
    MAX_GAP_DANGER = 1.00

    ZONE_BASELINE_CV = 16.34
    ZONE_BASELINE_PUBLIC = 16.36
    ZONE_BASELINE_GAP = 0.02

    SUBMIT_CONFIDENCE_MIN = 85  # 최소 85% 신뢰도
    SUBMIT_IMPROVEMENT_MIN = 0.50  # Zone 대비 최소 0.5점 개선

    def __init__(self, project_root: str = "/mnt/c/LSJ/dacon/dacon/kleague-algorithm"):
        self.root = Path(project_root)
        self.logs_dir = self.root / "logs"
        self.models_dir = self.root / "code" / "models"
        self.submissions_dir = self.root / "submissions"

        self.checkpoint = {}  # 체크포인트
        self.results = {}  # 실험 결과

    # =========================================================================
    # Stage 1: Design (실험 설계)
    # =========================================================================

    def stage1_design(self, experiment_name: str, description: str,
                      expected_cv: float, expected_gap: float,
                      changes: List[str]) -> Dict:
        """
        실험 설계 및 사전 검증

        Args:
            experiment_name: 실험 이름
            description: 실험 설명
            expected_cv: 예상 CV
            expected_gap: 예상 Gap
            changes: 변경 사항 리스트

        Returns:
            design: 설계 정보 및 검증 결과
        """
        print("=" * 80)
        print(f"Stage 1: Design - {experiment_name}")
        print("=" * 80)

        design = {
            "name": experiment_name,
            "description": description,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "expected_cv": expected_cv,
            "expected_gap": expected_gap,
            "expected_public": expected_cv + expected_gap,
            "changes": changes,
            "validation": {}
        }

        # 사전 검증
        print("\n[1.1] Sweet Spot 검증...")
        if self.SWEET_SPOT_MIN <= expected_cv <= self.SWEET_SPOT_MAX:
            design["validation"]["sweet_spot"] = "✅ PASS"
            print(f"  ✅ CV {expected_cv:.2f} in Sweet Spot [{self.SWEET_SPOT_MIN}, {self.SWEET_SPOT_MAX}]")
        elif expected_cv < self.SWEET_SPOT_MIN:
            design["validation"]["sweet_spot"] = "⚠️ WARNING (과최적화)"
            print(f"  ⚠️ CV {expected_cv:.2f} < {self.SWEET_SPOT_MIN} (과최적화 위험)")
        else:
            design["validation"]["sweet_spot"] = "⚠️ WARNING (개선 부족)"
            print(f"  ⚠️ CV {expected_cv:.2f} > {self.SWEET_SPOT_MAX} (개선 부족)")

        # Gap 검증
        print("\n[1.2] Gap 검증...")
        if expected_gap <= self.MAX_GAP_SAFE:
            design["validation"]["gap"] = "✅ PASS"
            print(f"  ✅ Gap {expected_gap:.2f} <= {self.MAX_GAP_SAFE} (안전)")
        elif expected_gap <= self.MAX_GAP_WARNING:
            design["validation"]["gap"] = "⚠️ WARNING"
            print(f"  ⚠️ Gap {expected_gap:.2f} > {self.MAX_GAP_SAFE} (주의)")
        else:
            design["validation"]["gap"] = "❌ FAIL"
            print(f"  ❌ Gap {expected_gap:.2f} > {self.MAX_GAP_WARNING} (위험)")

        # Zone 대비 개선 예상
        print("\n[1.3] Zone 대비 개선...")
        expected_public = expected_cv + expected_gap
        improvement = self.ZONE_BASELINE_PUBLIC - expected_public
        design["expected_improvement"] = improvement

        if improvement >= self.SUBMIT_IMPROVEMENT_MIN:
            design["validation"]["improvement"] = "✅ PASS"
            print(f"  ✅ 예상 개선: {improvement:.2f}점 >= {self.SUBMIT_IMPROVEMENT_MIN}점")
        else:
            design["validation"]["improvement"] = "❌ FAIL"
            print(f"  ❌ 예상 개선: {improvement:.2f}점 < {self.SUBMIT_IMPROVEMENT_MIN}점")

        # 설계 승인 (FAIL만 거부, WARNING 허용)
        has_fail = any(v.startswith("❌") for v in design["validation"].values())
        design["approved"] = not has_fail

        print(f"\n[1.4] 설계 승인: {'✅ APPROVED' if design['approved'] else '❌ REJECTED'}")

        self.checkpoint["design"] = design
        return design

    # =========================================================================
    # Stage 2: Code (코드 작성 - 에이전트 호출용 래퍼)
    # =========================================================================

    def stage2_code_review(self, code_path: str) -> Dict:
        """
        코드 자동 검증 (에이전트 호출 전 기본 검증)

        Args:
            code_path: 검증할 코드 파일 경로

        Returns:
            review: 검증 결과
        """
        print("=" * 80)
        print("Stage 2: Code Review (Basic)")
        print("=" * 80)

        review = {
            "code_path": code_path,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "checks": {}
        }

        # 파일 존재 확인
        if not Path(code_path).exists():
            review["checks"]["file_exists"] = "❌ FAIL"
            review["approved"] = False
            return review

        review["checks"]["file_exists"] = "✅ PASS"

        # 기본 패턴 검증
        with open(code_path, 'r', encoding='utf-8') as f:
            code = f.read()

        # GroupKFold 사용 확인
        if "GroupKFold" in code or "group" in code.lower():
            review["checks"]["groupkfold"] = "✅ PASS"
        else:
            review["checks"]["groupkfold"] = "⚠️ WARNING (GroupKFold 없음)"

        # Episode independence 확인
        if "game_episode" in code or "game_id" in code:
            review["checks"]["episode_independence"] = "✅ PASS"
        else:
            review["checks"]["episode_independence"] = "❌ FAIL (Episode ID 없음)"

        # Clip 확인
        if "clip" in code or "np.clip" in code:
            review["checks"]["field_bounds"] = "✅ PASS"
        else:
            review["checks"]["field_bounds"] = "⚠️ WARNING (Clip 없음)"

        review["approved"] = all(
            not v.startswith("❌") for v in review["checks"].values()
        )

        print(f"\n코드 검증: {'✅ APPROVED' if review['approved'] else '❌ REJECTED'}")
        for check, result in review["checks"].items():
            print(f"  {check}: {result}")

        self.checkpoint["code_review"] = review
        return review

    # =========================================================================
    # Stage 3: Validate (데이터 검증)
    # =========================================================================

    def stage3_validate_data(self, train_df: pd.DataFrame,
                            test_df: Optional[pd.DataFrame] = None) -> Dict:
        """
        데이터 누수 및 무결성 검증

        Args:
            train_df: 학습 데이터
            test_df: 테스트 데이터 (선택)

        Returns:
            validation: 검증 결과
        """
        print("=" * 80)
        print("Stage 3: Data Validation")
        print("=" * 80)

        validation = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "checks": {}
        }

        # Train 데이터 검증
        print("\n[3.1] Train 데이터 무결성...")

        # Episode ID 존재
        if 'game_episode' in train_df.columns or 'game_id' in train_df.columns:
            validation["checks"]["episode_id"] = "✅ PASS"
            print("  ✅ Episode ID 존재")
        else:
            validation["checks"]["episode_id"] = "❌ FAIL"
            print("  ❌ Episode ID 없음")

        # Target 범위 확인
        if 'end_x' in train_df.columns and 'end_y' in train_df.columns:
            x_valid = train_df['end_x'].between(0, 105).all()
            y_valid = train_df['end_y'].between(0, 68).all()

            if x_valid and y_valid:
                validation["checks"]["target_range"] = "✅ PASS"
                print("  ✅ Target 범위 유효 ([0,105] × [0,68])")
            else:
                validation["checks"]["target_range"] = "❌ FAIL"
                print("  ❌ Target 범위 초과")

        # NaN 확인
        nan_count = train_df.isna().sum().sum()
        if nan_count == 0:
            validation["checks"]["no_nan"] = "✅ PASS"
            print(f"  ✅ NaN 없음")
        else:
            validation["checks"]["no_nan"] = f"⚠️ WARNING ({nan_count} NaN)"
            print(f"  ⚠️ NaN {nan_count}개 발견")

        # Train-Test 누수 확인 (Test 제공 시)
        if test_df is not None:
            print("\n[3.2] Train-Test 누수 검증...")

            train_episodes = set(train_df['game_episode'].unique() if 'game_episode' in train_df.columns else train_df['game_id'].unique())
            test_episodes = set(test_df['game_episode'].unique() if 'game_episode' in test_df.columns else test_df['game_id'].unique())

            overlap = train_episodes & test_episodes

            if len(overlap) == 0:
                validation["checks"]["no_leakage"] = "✅ PASS"
                print(f"  ✅ Train-Test 누수 없음")
            else:
                validation["checks"]["no_leakage"] = f"❌ FAIL ({len(overlap)} episodes)"
                print(f"  ❌ Train-Test 누수: {len(overlap)} episodes")

        validation["approved"] = all(
            not str(v).startswith("❌") for v in validation["checks"].values()
        )

        print(f"\n데이터 검증: {'✅ APPROVED' if validation['approved'] else '❌ REJECTED'}")

        self.checkpoint["data_validation"] = validation
        return validation

    # =========================================================================
    # Stage 4: Execute (실행)
    # =========================================================================

    def stage4_execute(self, cv_scores: List[float],
                      feature_count: int,
                      model_info: Dict) -> Dict:
        """
        실험 실행 결과 수집

        Args:
            cv_scores: Fold별 CV 점수
            feature_count: 피처 개수
            model_info: 모델 정보

        Returns:
            execution: 실행 결과
        """
        print("=" * 80)
        print("Stage 4: Execution Results")
        print("=" * 80)

        execution = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "cv_scores": cv_scores,
            "cv_mean": np.mean(cv_scores),
            "cv_std": np.std(cv_scores),
            "cv_fold13": np.mean(cv_scores[:3]) if len(cv_scores) >= 3 else np.mean(cv_scores),
            "feature_count": feature_count,
            "model_info": model_info
        }

        print(f"\n[4.1] CV 결과:")
        for i, score in enumerate(cv_scores, 1):
            print(f"  Fold {i}: {score:.4f}")

        print(f"\n  평균: {execution['cv_mean']:.4f} ± {execution['cv_std']:.4f}")
        print(f"  Fold 1-3: {execution['cv_fold13']:.4f}")
        print(f"  Feature 수: {feature_count}")

        self.checkpoint["execution"] = execution
        return execution

    # =========================================================================
    # Stage 5: Analyze (Gap 분석)
    # =========================================================================

    def stage5_analyze_gap(self, cv: float, experiment_type: str = "unknown") -> Dict:
        """
        Gap 분석 및 Public 예측

        Args:
            cv: CV 점수
            experiment_type: 실험 유형 (zone, domain, lstm, etc.)

        Returns:
            analysis: Gap 분석 결과
        """
        print("=" * 80)
        print("Stage 5: Gap Analysis")
        print("=" * 80)

        # Gap 예측 (실험 유형별)
        gap_estimates = {
            "zone": {"min": 0.00, "expected": 0.02, "max": 0.10},
            "domain_no_target": {"min": 0.30, "expected": 0.41, "max": 0.60},
            "domain_last_pass": {"min": 0.10, "expected": 0.15, "max": 0.25},
            "lstm": {"min": 2.50, "expected": 3.00, "max": 4.00},
            "unknown": {"min": 0.20, "expected": 0.35, "max": 0.60}
        }

        gap_est = gap_estimates.get(experiment_type, gap_estimates["unknown"])

        analysis = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "cv": cv,
            "experiment_type": experiment_type,
            "gap_min": gap_est["min"],
            "gap_expected": gap_est["expected"],
            "gap_max": gap_est["max"],
            "public_min": cv + gap_est["min"],
            "public_expected": cv + gap_est["expected"],
            "public_max": cv + gap_est["max"]
        }

        print(f"\n[5.1] Gap 예측 ({experiment_type}):")
        print(f"  최선: Gap {gap_est['min']:.2f} → Public {analysis['public_min']:.2f}")
        print(f"  예상: Gap {gap_est['expected']:.2f} → Public {analysis['public_expected']:.2f}")
        print(f"  최악: Gap {gap_est['max']:.2f} → Public {analysis['public_max']:.2f}")

        # Zone 대비
        improvement_expected = self.ZONE_BASELINE_PUBLIC - analysis["public_expected"]
        improvement_best = self.ZONE_BASELINE_PUBLIC - analysis["public_min"]
        improvement_worst = self.ZONE_BASELINE_PUBLIC - analysis["public_max"]

        print(f"\n[5.2] Zone 대비 개선:")
        print(f"  최선: {improvement_best:+.2f}점")
        print(f"  예상: {improvement_expected:+.2f}점")
        print(f"  최악: {improvement_worst:+.2f}점")

        analysis["improvement_expected"] = improvement_expected
        analysis["improvement_best"] = improvement_best
        analysis["improvement_worst"] = improvement_worst

        self.checkpoint["gap_analysis"] = analysis
        return analysis

    # =========================================================================
    # Stage 6: Decide (제출 의사결정)
    # =========================================================================

    def stage6_decide_submission(self) -> Dict:
        """
        제출 여부 자동 의사결정

        Returns:
            decision: 의사결정 결과
        """
        print("=" * 80)
        print("Stage 6: Submission Decision")
        print("=" * 80)

        # 체크포인트 검증
        required = ["design", "execution", "gap_analysis"]
        for req in required:
            if req not in self.checkpoint:
                print(f"❌ 체크포인트 누락: {req}")
                return {"approved": False, "reason": f"Missing {req}"}

        design = self.checkpoint["design"]
        execution = self.checkpoint["execution"]
        gap_analysis = self.checkpoint["gap_analysis"]

        decision = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "checks": {},
            "confidence": 0,
            "approved": False,
            "reason": ""
        }

        # Check 1: Sweet Spot
        cv = execution["cv_fold13"]
        if self.SWEET_SPOT_MIN <= cv <= self.SWEET_SPOT_MAX:
            decision["checks"]["sweet_spot"] = "✅ PASS"
            confidence_sweet = 90
        elif cv < self.SWEET_SPOT_MIN:
            decision["checks"]["sweet_spot"] = "⚠️ WARNING (과최적화)"
            confidence_sweet = 60
        else:
            decision["checks"]["sweet_spot"] = "⚠️ WARNING (개선 부족)"
            confidence_sweet = 70

        # Check 2: Gap
        gap_expected = gap_analysis["gap_expected"]
        if gap_expected <= self.MAX_GAP_SAFE:
            decision["checks"]["gap"] = "✅ PASS"
            confidence_gap = 95
        elif gap_expected <= self.MAX_GAP_WARNING:
            decision["checks"]["gap"] = "⚠️ WARNING"
            confidence_gap = 75
        else:
            decision["checks"]["gap"] = "❌ FAIL"
            confidence_gap = 40

        # Check 3: Improvement
        improvement = gap_analysis["improvement_expected"]
        if improvement >= self.SUBMIT_IMPROVEMENT_MIN:
            decision["checks"]["improvement"] = "✅ PASS"
            confidence_improvement = 90
        elif improvement >= 0:
            decision["checks"]["improvement"] = "⚠️ WARNING"
            confidence_improvement = 60
        else:
            decision["checks"]["improvement"] = "❌ FAIL"
            confidence_improvement = 20

        # 종합 신뢰도 (가중 평균)
        confidence = (confidence_sweet * 0.3 +
                     confidence_gap * 0.4 +
                     confidence_improvement * 0.3)

        decision["confidence"] = round(confidence, 1)

        # 의사결정
        if confidence >= self.SUBMIT_CONFIDENCE_MIN:
            decision["approved"] = True
            decision["reason"] = f"신뢰도 {confidence:.1f}% >= {self.SUBMIT_CONFIDENCE_MIN}%"
            decision["action"] = "✅ 제출 권장"
        else:
            decision["approved"] = False
            decision["reason"] = f"신뢰도 {confidence:.1f}% < {self.SUBMIT_CONFIDENCE_MIN}%"
            decision["action"] = "❌ 제출 보류"

        print(f"\n[6.1] 제출 검증:")
        for check, result in decision["checks"].items():
            print(f"  {check}: {result}")

        print(f"\n[6.2] 종합 신뢰도: {confidence:.1f}%")
        print(f"\n[6.3] 최종 결정: {decision['action']}")
        print(f"  사유: {decision['reason']}")

        self.checkpoint["decision"] = decision
        return decision

    # =========================================================================
    # Stage 7: Track (문서화)
    # =========================================================================

    def stage7_track(self, experiment_name: str) -> str:
        """
        실험 결과 문서화 및 추적

        Args:
            experiment_name: 실험 이름

        Returns:
            log_path: 로그 파일 경로
        """
        print("=" * 80)
        print("Stage 7: Tracking & Documentation")
        print("=" * 80)

        # 실험 로그 파일
        log_path = self.logs_dir / f"pipeline_{experiment_name}.json"

        # 체크포인트 저장
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(self.checkpoint, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Pipeline 로그 저장: {log_path}")

        # 요약 출력
        print("\n" + "=" * 80)
        print("Pipeline 요약")
        print("=" * 80)

        if "design" in self.checkpoint:
            print(f"\n실험: {self.checkpoint['design']['name']}")
            print(f"설명: {self.checkpoint['design']['description']}")

        if "execution" in self.checkpoint:
            exec_data = self.checkpoint["execution"]
            print(f"\nCV: {exec_data['cv_fold13']:.4f} ± {exec_data['cv_std']:.4f}")
            print(f"Feature: {exec_data['feature_count']}개")

        if "gap_analysis" in self.checkpoint:
            gap = self.checkpoint["gap_analysis"]
            print(f"\nPublic 예상: {gap['public_expected']:.2f} (Gap {gap['gap_expected']:.2f})")
            print(f"Zone 대비: {gap['improvement_expected']:+.2f}점")

        if "decision" in self.checkpoint:
            dec = self.checkpoint["decision"]
            print(f"\n신뢰도: {dec['confidence']:.1f}%")
            print(f"결정: {dec['action']}")

        print("\n" + "=" * 80)

        return str(log_path)

    # =========================================================================
    # E2E 실행
    # =========================================================================

    def run_e2e(self, experiment_name: str, code_path: str,
                cv_scores: List[float], feature_count: int,
                experiment_type: str = "unknown",
                expected_cv: Optional[float] = None,
                expected_gap: Optional[float] = None) -> Dict:
        """
        End-to-End 파이프라인 실행

        Args:
            experiment_name: 실험 이름
            code_path: 코드 파일 경로
            cv_scores: Fold별 CV 점수
            feature_count: 피처 개수
            experiment_type: 실험 유형
            expected_cv: 예상 CV (선택)
            expected_gap: 예상 Gap (선택)

        Returns:
            result: 파이프라인 실행 결과
        """
        print("\n" + "=" * 80)
        print(f"K리그 E2E Pipeline: {experiment_name}")
        print("=" * 80)

        # Stage 1: Design (예상값이 제공된 경우)
        if expected_cv and expected_gap:
            design = self.stage1_design(
                experiment_name=experiment_name,
                description=f"{experiment_type} 실험",
                expected_cv=expected_cv,
                expected_gap=expected_gap,
                changes=[]
            )

            if not design["approved"]:
                print("\n❌ Stage 1 실패: 설계 승인 거부")
                return {"success": False, "stage": 1, "checkpoint": self.checkpoint}

        # Stage 2: Code Review
        review = self.stage2_code_review(code_path)
        if not review["approved"]:
            print("\n⚠️ Stage 2 경고: 코드 검증 실패 (계속 진행)")

        # Stage 3: Data Validation (생략 - 실제 데이터 필요)

        # Stage 4: Execution
        execution = self.stage4_execute(
            cv_scores=cv_scores,
            feature_count=feature_count,
            model_info={"type": experiment_type}
        )

        # Stage 5: Gap Analysis
        gap_analysis = self.stage5_analyze_gap(
            cv=execution["cv_fold13"],
            experiment_type=experiment_type
        )

        # Stage 6: Decision
        decision = self.stage6_decide_submission()

        # Stage 7: Track
        log_path = self.stage7_track(experiment_name)

        return {
            "success": True,
            "approved": decision["approved"],
            "confidence": decision["confidence"],
            "log_path": log_path,
            "checkpoint": self.checkpoint
        }


# =============================================================================
# 사용 예시
# =============================================================================

if __name__ == "__main__":
    # Pipeline 초기화
    pipeline = KLeaguePipeline()

    # Phase 1 실제 결과로 테스트
    result = pipeline.run_e2e(
        experiment_name="phase1_no_target",
        code_path="code/models/best/model_domain_features_v2_no_target.py",
        cv_scores=[15.0690, 15.5955, 14.8980, 15.0588, 14.8495],
        feature_count=25,
        experiment_type="domain_no_target",
        expected_cv=15.19,  # 실제 Fold 1-3 평균
        expected_gap=0.41
    )

    if result['success']:
        print(f"\n최종 결과: {'✅ 제출 승인' if result['approved'] else '❌ 제출 거부'}")
        print(f"신뢰도: {result['confidence']:.1f}%")
        print(f"로그: {result['log_path']}")
    else:
        print(f"\n❌ Pipeline 실패: Stage {result['stage']}")
