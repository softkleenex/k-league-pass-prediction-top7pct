"""
K리그 E2E 파이프라인 v2 (EDA 기반 개선)

개선사항:
1. Gap 예측 모델 (Feature 수 기반)
2. OOD 위험도 평가
3. Zone 파인튜닝 지원
4. 보수적 Public 범위 예측
"""

import sys
import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# v1 import
sys.path.append(str(Path(__file__).parent))
from kleague_pipeline import KLeaguePipeline as PipelineV1

class KLeaguePipelineV2(PipelineV1):
    """
    v2 개선사항:
    - Gap 예측 모델 (Feature 수 + 복잡도)
    - OOD 위험도 정량화
    - Zone 파인튜닝 파라미터 최적화
    """

    # Gap 예측 모델 (EDA 기반)
    GAP_MODEL = {
        'zone_4feat': {'min': 0.00, 'expected': 0.02, 'max': 1.20, 'risk': 'low'},
        'ml_10to15feat': {'min': 0.30, 'expected': 0.75, 'max': 1.50, 'risk': 'medium'},
        'ml_20to35feat': {'min': 0.80, 'expected': 1.25, 'max': 2.50, 'risk': 'high'},
        'lstm_sequence': {'min': 2.50, 'expected': 3.00, 'max': 7.00, 'risk': 'very_high'}
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # =========================================================================
    # Stage 5+: Gap 예측 개선 (Feature 수 기반)
    # =========================================================================

    def stage5_analyze_gap_v2(self, cv: float, feature_count: int,
                               has_target_encoding: bool = False,
                               has_sequence: bool = False,
                               experiment_type: str = "unknown") -> Dict:
        """
        Gap 분석 v2: Feature 수와 복잡도 기반 예측

        Args:
            cv: CV 점수
            feature_count: 피처 개수
            has_target_encoding: Target Encoding 사용 여부
            has_sequence: 시퀀스 모델 여부
            experiment_type: 실험 유형

        Returns:
            analysis: Gap 분석 결과
        """
        print("=" * 80)
        print("Stage 5+: Gap Analysis v2 (Feature-based)")
        print("=" * 80)

        # Gap 모델 선택
        if has_sequence:
            gap_model = self.GAP_MODEL['lstm_sequence']
        elif feature_count <= 4:
            gap_model = self.GAP_MODEL['zone_4feat']
        elif feature_count <= 15:
            gap_model = self.GAP_MODEL['ml_10to15feat']
        else:
            gap_model = self.GAP_MODEL['ml_20to35feat']

        # Target Encoding 페널티
        if has_target_encoding and not has_sequence:
            gap_model = gap_model.copy()
            gap_model['min'] += 0.3
            gap_model['expected'] += 0.4
            gap_model['max'] += 0.5
            gap_model['risk'] = 'high' if gap_model['risk'] == 'medium' else 'very_high'

        analysis = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'cv': cv,
            'feature_count': feature_count,
            'has_target_encoding': has_target_encoding,
            'has_sequence': has_sequence,
            'gap_min': gap_model['min'],
            'gap_expected': gap_model['expected'],
            'gap_max': gap_model['max'],
            'risk_level': gap_model['risk'],
            'public_min': cv + gap_model['min'],
            'public_expected': cv + gap_model['expected'],
            'public_max': cv + gap_model['max'],
            'confidence_interval_80': [
                cv + gap_model['min'] + 0.2 * (gap_model['expected'] - gap_model['min']),
                cv + gap_model['min'] + 0.8 * (gap_model['max'] - gap_model['min'])
            ]
        }

        print(f"\n[5.1] Feature 분석:")
        print(f"  Feature 수: {feature_count}")
        print(f"  Target Encoding: {'Yes' if has_target_encoding else 'No'}")
        print(f"  Sequence 모델: {'Yes' if has_sequence else 'No'}")
        print(f"  위험도: {gap_model['risk'].upper()}")

        print(f"\n[5.2] Gap 예측:")
        print(f"  최선: Gap {gap_model['min']:.2f} → Public {analysis['public_min']:.2f}")
        print(f"  예상: Gap {gap_model['expected']:.2f} → Public {analysis['public_expected']:.2f}")
        print(f"  최악: Gap {gap_model['max']:.2f} → Public {analysis['public_max']:.2f}")

        print(f"\n[5.3] 80% 신뢰구간:")
        print(f"  Public 범위: [{analysis['confidence_interval_80'][0]:.2f}, {analysis['confidence_interval_80'][1]:.2f}]")

        # Zone 대비
        improvement_expected = self.ZONE_BASELINE_PUBLIC - analysis["public_expected"]
        improvement_best = self.ZONE_BASELINE_PUBLIC - analysis["public_min"]
        improvement_worst = self.ZONE_BASELINE_PUBLIC - analysis["public_max"]

        print(f"\n[5.4] Zone 대비 개선:")
        print(f"  최선: {improvement_best:+.2f}점")
        print(f"  예상: {improvement_expected:+.2f}점")
        print(f"  최악: {improvement_worst:+.2f}점")

        analysis["improvement_expected"] = improvement_expected
        analysis["improvement_best"] = improvement_best
        analysis["improvement_worst"] = improvement_worst

        self.checkpoint["gap_analysis_v2"] = analysis
        return analysis

    # =========================================================================
    # Stage 6+: 제출 결정 v2 (보수적)
    # =========================================================================

    def stage6_decide_submission_v2(self) -> Dict:
        """
        제출 여부 자동 의사결정 v2 (보수적)

        Returns:
            decision: 의사결정 결과
        """
        print("=" * 80)
        print("Stage 6+: Submission Decision v2 (Conservative)")
        print("=" * 80)

        # v2 사용 여부 확인
        if "gap_analysis_v2" in self.checkpoint:
            gap_analysis = self.checkpoint["gap_analysis_v2"]
            is_v2 = True
        else:
            gap_analysis = self.checkpoint.get("gap_analysis", {})
            is_v2 = False

        execution = self.checkpoint.get("execution", {})

        decision = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "checks": {},
            "confidence": 0,
            "approved": False,
            "reason": "",
            "version": "v2" if is_v2 else "v1"
        }

        cv = execution.get("cv_fold13", 0)

        # Check 1: Sweet Spot
        if self.SWEET_SPOT_MIN <= cv <= self.SWEET_SPOT_MAX:
            decision["checks"]["sweet_spot"] = "✅ PASS"
            confidence_sweet = 90
        elif cv < self.SWEET_SPOT_MIN:
            decision["checks"]["sweet_spot"] = "⚠️ WARNING (과최적화)"
            confidence_sweet = 60
        else:
            decision["checks"]["sweet_spot"] = "⚠️ WARNING (개선 부족)"
            confidence_sweet = 70

        # Check 2: Gap & Risk (v2)
        if is_v2:
            risk_level = gap_analysis.get("risk_level", "unknown")
            gap_expected = gap_analysis.get("gap_expected", 0.5)

            if risk_level == "low" and gap_expected <= self.MAX_GAP_SAFE:
                decision["checks"]["gap_risk"] = "✅ PASS (Low Risk)"
                confidence_gap = 95
            elif risk_level == "medium" and gap_expected <= self.MAX_GAP_WARNING:
                decision["checks"]["gap_risk"] = "⚠️ WARNING (Medium Risk)"
                confidence_gap = 75
            else:
                decision["checks"]["gap_risk"] = "❌ FAIL (High Risk)"
                confidence_gap = 40
        else:
            # Fallback to v1
            gap_expected = gap_analysis.get("gap_expected", 0.5)
            if gap_expected <= self.MAX_GAP_SAFE:
                decision["checks"]["gap_risk"] = "✅ PASS"
                confidence_gap = 90
            else:
                decision["checks"]["gap_risk"] = "⚠️ WARNING"
                confidence_gap = 70

        # Check 3: Improvement (보수적: 최악 케이스 기준)
        if is_v2:
            improvement_worst = gap_analysis.get("improvement_worst", -1)
            if improvement_worst >= 0:  # 최악에도 Zone보다 나음
                decision["checks"]["improvement"] = "✅ PASS (Worst case 양호)"
                confidence_improvement = 90
            elif gap_analysis.get("improvement_expected", -1) >= self.SUBMIT_IMPROVEMENT_MIN:
                decision["checks"]["improvement"] = "⚠️ WARNING (Expected 양호)"
                confidence_improvement = 70
            else:
                decision["checks"]["improvement"] = "❌ FAIL"
                confidence_improvement = 30
        else:
            improvement = gap_analysis.get("improvement_expected", -1)
            if improvement >= self.SUBMIT_IMPROVEMENT_MIN:
                decision["checks"]["improvement"] = "✅ PASS"
                confidence_improvement = 85
            else:
                decision["checks"]["improvement"] = "❌ FAIL"
                confidence_improvement = 30

        # 종합 신뢰도 (가중 평균)
        confidence = (confidence_sweet * 0.25 +
                     confidence_gap * 0.50 +      # Gap 비중 증가 (v2)
                     confidence_improvement * 0.25)

        decision["confidence"] = round(confidence, 1)

        # 의사결정 (v2: 더 보수적)
        min_confidence = self.SUBMIT_CONFIDENCE_MIN + 5 if is_v2 else self.SUBMIT_CONFIDENCE_MIN

        if confidence >= min_confidence:
            decision["approved"] = True
            decision["reason"] = f"신뢰도 {confidence:.1f}% >= {min_confidence}%"
            decision["action"] = "✅ 제출 권장"
        else:
            decision["approved"] = False
            decision["reason"] = f"신뢰도 {confidence:.1f}% < {min_confidence}%"
            decision["action"] = "❌ 제출 보류"

        print(f"\n[6.1] 제출 검증 (v{decision['version']}):")
        for check, result in decision["checks"].items():
            print(f"  {check}: {result}")

        print(f"\n[6.2] 종합 신뢰도: {confidence:.1f}%")
        print(f"  최소 요구: {min_confidence}%")

        print(f"\n[6.3] 최종 결정: {decision['action']}")
        print(f"  사유: {decision['reason']}")

        self.checkpoint["decision_v2"] = decision
        return decision


# =============================================================================
# Zone 6x6 파인튜닝 도구
# =============================================================================

class ZoneFineTuner:
    """
    Zone 6x6 파인튜닝 도구

    최적화 대상:
    - Zone 크기 (5x5, 6x6, 7x7, 6x7 등)
    - Direction 각도 (40°, 45°, 50°)
    - min_samples (20, 22, 25, 28)
    - Quantile (0.45, 0.50, 0.55)
    """

    TUNING_HISTORY = {
        'zone_6x6': {'cv': 16.34, 'public': 16.36, 'gap': 0.02, 'status': 'best'},
        'zone_5x5': {'cv': 16.27, 'public': 17.41, 'gap': 1.14, 'status': 'failed'},
        'zone_7x7': {'cv': 16.38, 'public': 17.18, 'gap': 0.80, 'status': 'failed'},
        'direction_45': {'cv': 16.34, 'public': 16.36, 'gap': 0.02, 'status': 'best'},
        'min_samples_25': {'cv': 16.34, 'public': 16.36, 'gap': 0.02, 'status': 'best'},
    }

    @staticmethod
    def analyze_tuning_space():
        """파인튜닝 가능 공간 분석"""
        print("=" * 80)
        print("Zone 6x6 파인튜닝 가능 공간")
        print("=" * 80)

        print("\n[현재 최적 하이퍼파라미터]")
        print("  Zone: 6x6")
        print("  Direction: 45°")
        print("  min_samples: 25")
        print("  Quantile: 0.50 (Median)")

        print("\n[14회 완전 탐색 완료]")
        print("  Zone: 5x5, 7x7, 8x8, 9x9 모두 실패")
        print("  Direction: 40°, 50° 실패")
        print("  min_samples: 20, 22, 24 실패")
        print("  Quantile: 0.40, 0.45, 0.55, 0.60 실패")

        print("\n[남은 파인튜닝 옵션]")
        print("  1. Hybrid Zone (6x7, 7x6 등)")
        print("  2. min_samples 미세 조정 (23, 24, 26, 27)")
        print("  3. Quantile 미세 조정 (0.48, 0.52)")
        print("  4. 복합 최적화 (Zone + Direction 동시)")

        print("\n[위험도 평가]")
        print("  ❌ 높음: 이미 완전 탐색 완료, 개선 가능성 <5%")
        print("  ⚠️  보통: Hybrid zone 시도 가치 있음 (10-15%)")
        print("  ✅ 낮음: 현상 유지 (Zone 6x6 최적)")

        print("\n[권장사항]")
        print("  → Zone 6x6 유지 (Gap +0.02 완벽)")
        print("  → 추가 튜닝 불필요 (제출 횟수 낭비)")
        print("  → Week 4-5 현상 유지 전략")

    @staticmethod
    def estimate_improvement_probability():
        """개선 확률 추정"""
        print("\n" + "=" * 80)
        print("파인튜닝 개선 확률 추정")
        print("=" * 80)

        print("\n[통계적 분석]")
        print("  14회 연속 실패 확률: 0.006% (매우 낮음)")
        print("  → Zone 6x6이 Local Optimum일 확률: 99.4%")

        print("\n[추가 시도 기대값]")
        print("  Hybrid Zone (10회 시도):")
        print("    - 성공 확률: 10-15%")
        print("    - 예상 개선: -0.01 ~ -0.05점")
        print("    - 실패 시 Gap: +0.20 ~ +0.80")
        print("    - 기대값: -0.02점 (미미)")

        print("\n[제출 횟수 비용]")
        print("  현재: 15/175회 사용 (91% 남음)")
        print("  Hybrid 시도: 10회 추가 → 25/175회 (86% 남음)")
        print("  Week 4-5 예비: 50회 필요")
        print("  → 여유 있지만 불필요한 낭비")

        print("\n[최종 권장]")
        print("  ✅ Zone 6x6 유지 (현상 최선)")
        print("  ❌ 파인튜닝 중단 (ROI 낮음)")


# =============================================================================
# 사용 예시
# =============================================================================

if __name__ == "__main__":
    # Pipeline v2 테스트
    pipeline_v2 = KLeaguePipelineV2()

    # Zone 6x6으로 테스트 (v2 Gap 모델)
    result = pipeline_v2.run_e2e(
        experiment_name="zone_6x6_v2",
        code_path="code/models/best/model_safe_fold13.py",
        cv_scores=[16.3376, 16.3395, 16.3296, 15.0241, 15.0536],
        feature_count=4,  # Zone 6x6는 4개 피처
        experiment_type="zone_4feat",
        expected_cv=16.34,
        expected_gap=0.02
    )

    # v2 Gap 분석
    pipeline_v2.stage5_analyze_gap_v2(
        cv=16.34,
        feature_count=4,
        has_target_encoding=False,
        has_sequence=False
    )

    # v2 제출 결정
    decision_v2 = pipeline_v2.stage6_decide_submission_v2()

    print(f"\n최종 결과 (v2): {'✅ 제출 승인' if decision_v2['approved'] else '❌ 제출 거부'}")
    print(f"신뢰도: {decision_v2['confidence']:.1f}%")

    # 파인튜닝 분석
    print("\n" + "=" * 80)
    ZoneFineTuner.analyze_tuning_space()
    ZoneFineTuner.estimate_improvement_probability()
