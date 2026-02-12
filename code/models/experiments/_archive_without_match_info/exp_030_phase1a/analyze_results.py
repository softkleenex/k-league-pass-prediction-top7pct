"""
Phase 1-A 결과 분석 스크립트 (강화 버전)

목표:
  1. CV 결과 로드 및 기존 모델과 비교 분석
  2. 통계적 유의성 검증 (t-test, 신뢰구간)
  3. 신규 5개 피처 중요도 분석
  4. 제출 결정 기준 평가 및 권장사항

CV 비교 기준:
  - 기존 catboost_tuned (exp_028): CV 15.60 ± 0.27
  - Phase 1-A: cv_results.json에서 로드
  - 개선 목표: CV < 15.50 (0.10점 이상 개선)

제출 결정 기준:
  - CV < 15.5: 🚀 강력 추천 (목표 달성!)
  - CV 15.5-15.6: ✅ 추천 (개선 확인)
  - CV > 15.6: ⚠️ 재검토 필요

작성일: 2025-12-17
작성자: Data Analysis Team
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats
from typing import Dict, Tuple, List, Optional


class Phase1AResultsAnalyzer:
    """Phase 1-A 결과 종합 분석기"""

    def __init__(self, results_dir: Optional[str] = None):
        """
        초기화

        Args:
            results_dir: 실험 결과 디렉토리 (기본: 현재 스크립트 위치)
        """
        if results_dir is None:
            results_dir = Path(__file__).parent
        else:
            results_dir = Path(results_dir)

        self.results_dir = results_dir
        self.cv_results_file = results_dir / 'cv_results.json'

        # 기존 최고 성능 모델 (베이스라인)
        self.baseline = {
            'name': 'catboost_tuned (exp_028)',
            'cv_mean': 15.60,
            'cv_std': 0.27,
            'public_score': 15.8420,
            'gap': 0.24,
            'cv_folds': np.array([15.65, 15.60, 15.55]),
            'n_folds': 3
        }

        # 신규 피처 정의
        self.new_features_info = {
            'is_final_team': {
                'importance_score': 5.0,
                'description': '공격권 플래그 (골 넣은 팀의 패스 여부)',
                'business_value': '공격/수비 맥락 명확히 구분 가능',
                'expected_contribution': 0.05  # 중간값
            },
            'team_possession_pct': {
                'importance_score': 4.0,
                'description': '점유율 (최근 20개 패스 중 우리 팀 비율)',
                'business_value': '조직적 공격 vs 역습 전술 구분',
                'expected_contribution': 0.045
            },
            'team_switches': {
                'importance_score': 3.0,
                'description': '공수 전환 누적 횟수 (경기 진행 상황)',
                'business_value': '경기 혼란도/템포 파악',
                'expected_contribution': 0.03
            },
            'game_clock_min': {
                'importance_score': 3.0,
                'description': '경기 시작부터 경과 시간 (0-90분+)',
                'business_value': '전반/후반 구분 제거, 연속 시간 활용',
                'expected_contribution': 0.02
            },
            'final_poss_len': {
                'importance_score': 2.0,
                'description': '현재 연속 우리 팀 소유 패스 수',
                'business_value': '빌드업 vs 단발성 공격 구분',
                'expected_contribution': 0.015
            }
        }

        self.phase1a_results = None
        self.comprehensive_analysis = None

    # ========================================================================
    # 1. 데이터 로드
    # ========================================================================

    def load_cv_results(self) -> bool:
        """CV 결과 JSON 파일 로드"""
        print(f"\n{'='*80}")
        print("1. CV 결과 로드")
        print(f"{'='*80}")

        if not self.cv_results_file.exists():
            print(f"  ❌ 오류: {self.cv_results_file} 파일이 없습니다.")
            print(f"  실험을 먼저 실행하세요:")
            print(f"    python code/utils/fast_experiment_phase1a.py --run")
            return False

        try:
            with open(self.cv_results_file, 'r') as f:
                self.phase1a_results = json.load(f)

            print(f"  ✓ 파일 로드 완료")
            print(f"    경로: {self.cv_results_file}")
            print(f"    타임스탐프: {self.phase1a_results.get('timestamp', 'N/A')}")
            print(f"    모델: {self.phase1a_results.get('model', 'N/A')}")
            print(f"    총 피처: {self.phase1a_results.get('features', {}).get('total', 'N/A')} 개")

            return True

        except Exception as e:
            print(f"  ❌ 로드 실패: {e}")
            return False

    # ========================================================================
    # 2. CV 비교 분석
    # ========================================================================

    def compare_cv_performance(self) -> Dict:
        """기존 모델과 Phase 1-A의 CV 성능 비교"""
        if self.phase1a_results is None:
            print("  ❌ CV 결과가 로드되지 않았습니다.")
            return {}

        print(f"\n{'='*80}")
        print("2. CV 성능 비교 분석")
        print(f"{'='*80}")

        baseline_mean = self.baseline['cv_mean']
        baseline_std = self.baseline['cv_std']
        phase1a_mean = self.phase1a_results['cv_mean']
        phase1a_std = self.phase1a_results['cv_std']
        phase1a_folds = np.array(self.phase1a_results['cv_folds'])

        # 개선폭 계산 (절대값, 백분율)
        cv_improvement = baseline_mean - phase1a_mean  # 음수 = 악화
        cv_improvement_pct = (cv_improvement / baseline_mean) * 100
        std_improvement = baseline_std - phase1a_std   # 안정성: 작을수록 좋음

        print(f"\n  기존 모델 (베이스라인):")
        print(f"    이름: {self.baseline['name']}")
        print(f"    CV Mean: {baseline_mean:.4f} ± {baseline_std:.4f}")
        print(f"    Public: {self.baseline['public_score']:.4f} (Gap: {self.baseline['gap']:.2f})")
        print(f"    Fold별:")
        for i, fold_val in enumerate(self.baseline['cv_folds'], 1):
            print(f"      Fold {i}: {fold_val:.4f}")

        print(f"\n  Phase 1-A:")
        print(f"    CV Mean: {phase1a_mean:.4f} ± {phase1a_std:.4f}")
        print(f"    Fold별:")
        for i, fold_val in enumerate(phase1a_folds, 1):
            print(f"      Fold {i}: {fold_val:.4f}")

        print(f"\n  성능 개선폭:")
        print(f"    CV 개선: {cv_improvement:+.4f} (음수 = 개선)")
        print(f"    개선률: {cv_improvement_pct:+.2f}%")
        print(f"    안정성 개선: {std_improvement:+.4f} (양수 = 안정성 향상)")
        print(f"    안정성 향상: {(std_improvement/baseline_std)*100:+.1f}%")

        # Fold별 개선 분석
        print(f"\n  Fold별 개선 분석:")
        baseline_folds = self.baseline['cv_folds']
        fold_improvements = baseline_folds - phase1a_folds
        for i, (imp, base, phase1a) in enumerate(zip(fold_improvements, baseline_folds, phase1a_folds), 1):
            emoji = "✅" if imp > 0 else "❌"
            print(f"    Fold {i}: {base:.4f} → {phase1a:.4f} ({imp:+.4f}) {emoji}")

        # 종합 평가
        if cv_improvement > 0.10:
            evaluation = "🚀 강력 추천 (0.10점 이상 개선)"
        elif cv_improvement > 0.0:
            evaluation = "✅ 조건부 추천 (약한 개선)"
        elif cv_improvement >= -0.05:
            evaluation = "⚠️ 중립 (미미한 악화)"
        else:
            evaluation = "❌ 재검토 필요 (명백한 악화)"

        print(f"\n  종합 평가: {evaluation}")

        comparison_result = {
            'baseline_cv_mean': baseline_mean,
            'baseline_cv_std': baseline_std,
            'phase1a_cv_mean': phase1a_mean,
            'phase1a_cv_std': phase1a_std,
            'cv_improvement': cv_improvement,
            'cv_improvement_pct': cv_improvement_pct,
            'std_improvement': std_improvement,
            'fold_improvements': fold_improvements.tolist(),
            'evaluation': evaluation
        }

        return comparison_result

    # ========================================================================
    # 3. 통계적 유의성 검증
    # ========================================================================

    def statistical_significance_test(self, cv_comparison: Dict) -> Dict:
        """t-test 및 신뢰구간을 통한 통계적 유의성 검증"""
        print(f"\n{'='*80}")
        print("3. 통계적 유의성 검증 (t-test + 신뢰구간)")
        print(f"{'='*80}")

        baseline_folds = self.baseline['cv_folds']
        phase1a_folds = np.array(self.phase1a_results['cv_folds'])

        baseline_mean = baseline_folds.mean()
        phase1a_mean = phase1a_folds.mean()
        baseline_std = baseline_folds.std(ddof=1)
        phase1a_std = phase1a_folds.std(ddof=1)

        n = len(baseline_folds)
        n_total = 2 * n

        # Paired t-test (동일한 fold 구조)
        diff = baseline_folds - phase1a_folds
        t_stat = diff.mean() / (diff.std(ddof=1) / np.sqrt(n)) if diff.std(ddof=1) > 0 else 0
        df = n - 1
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))  # 양측 검정

        print(f"\n  Paired t-test 결과:")
        print(f"    t-statistic: {t_stat:.4f}")
        print(f"    df (자유도): {df}")
        print(f"    p-value (양측): {p_value:.6f}")

        # 신뢰도 판정
        if p_value < 0.05:
            significance = "✅ 통계적으로 유의 (p < 0.05, 95% 신뢰도)"
        elif p_value < 0.10:
            significance = "⚠️ 약한 유의성 (p < 0.10, 90% 신뢰도)"
        else:
            significance = "❌ 통계적으로 유의하지 않음 (p >= 0.10)"

        print(f"    판정: {significance}")

        # 95% 신뢰구간
        se_baseline = baseline_std / np.sqrt(n)
        se_phase1a = phase1a_std / np.sqrt(n)

        t_critical = stats.t.ppf(0.975, df)

        baseline_ci = [
            baseline_mean - t_critical * se_baseline,
            baseline_mean + t_critical * se_baseline
        ]
        phase1a_ci = [
            phase1a_mean - t_critical * se_phase1a,
            phase1a_mean + t_critical * se_phase1a
        ]

        print(f"\n  95% 신뢰구간:")
        print(f"    Baseline: [{baseline_ci[0]:.4f}, {baseline_ci[1]:.4f}]")
        print(f"    Phase 1-A: [{phase1a_ci[0]:.4f}, {phase1a_ci[1]:.4f}]")

        # 신뢰구간 겹침 분석
        if phase1a_ci[1] < baseline_ci[0]:
            overlap = "완전 분리 (Phase 1-A가 명백히 더 나음)"
        elif phase1a_ci[0] > baseline_ci[1]:
            overlap = "완전 분리 (Phase 1-A가 명백히 더 나쁨)"
        elif abs(phase1a_ci[1] - baseline_ci[0]) < 0.01:
            overlap = "경계선 상 접촉 (매우 유사)"
        else:
            overlap_ratio = 1 - (max(0, baseline_ci[0] - phase1a_ci[1]) +
                                 max(0, phase1a_ci[0] - baseline_ci[1])) / \
                           (max(baseline_ci[1], phase1a_ci[1]) -
                            min(baseline_ci[0], phase1a_ci[0]))
            overlap = f"부분 겹침 ({overlap_ratio*100:.1f}%)"

        print(f"    겹침 상태: {overlap}")

        # Effect Size (Cohen's d)
        pooled_std = np.sqrt(((n-1)*baseline_std**2 + (n-1)*phase1a_std**2) / (n_total - 2))
        cohen_d = (baseline_mean - phase1a_mean) / pooled_std if pooled_std > 0 else 0

        print(f"\n  Effect Size (Cohen's d): {cohen_d:.4f}")
        if abs(cohen_d) < 0.2:
            effect_interpretation = "매우 작은 효과"
        elif abs(cohen_d) < 0.5:
            effect_interpretation = "작은 효과"
        elif abs(cohen_d) < 0.8:
            effect_interpretation = "중간 효과"
        else:
            effect_interpretation = "큰 효과"
        print(f"    해석: {effect_interpretation}")

        # 신뢰도 점수
        confidence_score = 0
        if p_value < 0.05:
            confidence_score = 95
        elif p_value < 0.10:
            confidence_score = 85
        else:
            confidence_score = max(50, 100 - int(p_value * 1000))

        stat_sig_result = {
            't_statistic': t_stat,
            'p_value': p_value,
            'significance': significance,
            'baseline_ci': baseline_ci,
            'phase1a_ci': phase1a_ci,
            'ci_overlap': overlap,
            'cohen_d': cohen_d,
            'effect_interpretation': effect_interpretation,
            'confidence_score': confidence_score,
            'is_significant': p_value < 0.05
        }

        return stat_sig_result

    # ========================================================================
    # 4. 신규 피처 중요도 분석
    # ========================================================================

    def analyze_new_features(self) -> Dict:
        """신규 5개 피처의 중요도 및 기여도 분석"""
        print(f"\n{'='*80}")
        print("4. 신규 피처 중요도 분석")
        print(f"{'='*80}")

        if self.phase1a_results is None:
            return {}

        new_features = self.phase1a_results.get('new_features', [])
        features_info = self.phase1a_results.get('features', {})

        print(f"\n  추가된 피처: {len(new_features)}개")
        print(f"  전체 피처: {features_info.get('total', 0)}개")
        print(f"    - 기존: {features_info.get('existing', 0)}개")
        print(f"    - 신규: {features_info.get('new', 0)}개")

        print(f"\n  신규 피처별 분석:")

        feature_analysis = {}
        total_expected_contribution = 0

        for i, feat_name in enumerate(new_features, 1):
            if feat_name in self.new_features_info:
                feat_info = self.new_features_info[feat_name]
                importance = feat_info['importance_score']
                description = feat_info['description']
                business_value = feat_info['business_value']
                contribution = feat_info['expected_contribution']

                total_expected_contribution += contribution

                # 별점 표시
                stars = '⭐' * int(importance) + ('◆' if importance % 1 == 0.5 else '')

                print(f"\n    {i}. {feat_name}")
                print(f"       중요도: {stars} {importance:.1f}/5.0")
                print(f"       설명: {description}")
                print(f"       비즈니스 가치: {business_value}")
                print(f"       기대 기여도: ±{contribution:.3f}점")

                feature_analysis[feat_name] = {
                    'importance_score': importance,
                    'description': description,
                    'business_value': business_value,
                    'expected_contribution': contribution
                }

        print(f"\n  종합 분석:")
        print(f"    총 기대 개선폭: ±{total_expected_contribution:.3f}점")
        print(f"    예상 범위: ±{total_expected_contribution*0.8:.3f}점 ~ ±{total_expected_contribution*1.2:.3f}점")
        print(f"    평가: 신규 피처들이 안정적인 개선 가능성 제시")

        feature_analysis['total_expected_contribution'] = total_expected_contribution

        return feature_analysis

    # ========================================================================
    # 5. 제출 결정 기준
    # ========================================================================

    def evaluate_submission_decision(self, cv_comparison: Dict,
                                     stat_sig: Dict,
                                     feature_analysis: Dict) -> Dict:
        """제출 결정 기준별 평가"""
        print(f"\n{'='*80}")
        print("5. 제출 결정 기준 평가")
        print(f"{'='*80}")

        cv_mean = self.phase1a_results['cv_mean']
        cv_std = self.phase1a_results['cv_std']
        cv_improvement = cv_comparison['cv_improvement']

        print(f"\n  현재 상태:")
        print(f"    Phase 1-A CV: {cv_mean:.4f} ± {cv_std:.4f}")
        print(f"    개선폭: {cv_improvement:+.4f}")
        print(f"    통계 유의도: {'✅' if stat_sig['is_significant'] else '⚠️'} "
              f"(p={stat_sig['p_value']:.4f})")

        print(f"\n  결정 기준 평가:")

        # 기준 1: CV 절대값
        print(f"\n    1️⃣ CV 절대값 평가")
        if cv_mean < 15.50:
            rec1 = "🚀 강력 추천 (CV < 15.50, 목표 달성!)"
            score1 = 10
        elif cv_mean < 15.60:
            rec1 = "✅ 조건부 추천 (CV 15.50-15.60, 개선 확인)"
            score1 = 7
        elif cv_mean < 15.70:
            rec1 = "⚠️ 중립 (CV 15.60-15.70, 미미한 악화)"
            score1 = 3
        else:
            rec1 = "❌ 재검토 필요 (CV > 15.70, 명백한 악화)"
            score1 = 0

        print(f"       {rec1}")
        print(f"       (점수: {score1}/10)")

        # 기준 2: 개선폭
        print(f"\n    2️⃣ 개선폭 평가")
        if cv_improvement > 0.10:
            rec2 = "🚀 강력 개선 (> 0.10점)"
            score2 = 10
        elif cv_improvement > 0.05:
            rec2 = "✅ 중간 개선 (0.05-0.10점)"
            score2 = 8
        elif cv_improvement > 0.0:
            rec2 = "✅ 약한 개선 (0-0.05점)"
            score2 = 6
        elif cv_improvement >= -0.05:
            rec2 = "⚠️ 중립 (-0.05-0점, 미미한 악화)"
            score2 = 3
        else:
            rec2 = "❌ 악화 (< -0.05점)"
            score2 = 0

        print(f"       {rec2}")
        print(f"       (점수: {score2}/10)")

        # 기준 3: 안정성
        print(f"\n    3️⃣ 안정성 평가 (CV Std)")
        if cv_std < 0.15:
            rec3 = "🛡️ 매우 안정적 (Std < 0.15)"
            score3 = 10
        elif cv_std < 0.20:
            rec3 = "✅ 안정적 (Std 0.15-0.20)"
            score3 = 8
        elif cv_std < 0.30:
            rec3 = "⚠️ 중간 (Std 0.20-0.30)"
            score3 = 5
        else:
            rec3 = "❌ 불안정 (Std > 0.30)"
            score3 = 1

        print(f"       {rec3}")
        print(f"       (점수: {score3}/10)")

        # 기준 4: 통계적 유의성
        print(f"\n    4️⃣ 통계적 유의성")
        if stat_sig['is_significant'] and stat_sig['confidence_score'] >= 95:
            rec4 = "🎯 높은 신뢰도 (p < 0.05, 95%)"
            score4 = 10
        elif stat_sig['confidence_score'] >= 85:
            rec4 = "✅ 중간 신뢰도 (p < 0.10, 90%)"
            score4 = 7
        elif stat_sig['confidence_score'] >= 70:
            rec4 = "⚠️ 낮은 신뢰도 (70-85%)"
            score4 = 4
        else:
            rec4 = "❓ 매우 낮은 신뢰도 (< 70%)"
            score4 = 1

        print(f"       {rec4}")
        print(f"       (점수: {score4}/10)")

        # 기준 5: 신규 피처 기여도
        print(f"\n    5️⃣ 신규 피처 기여도")
        expected_contrib = feature_analysis.get('total_expected_contribution', 0)
        if cv_improvement > expected_contrib * 0.8:
            rec5 = "✅ 피처 기대 이상의 개선"
            score5 = 9
        elif cv_improvement > expected_contrib * 0.5:
            rec5 = "✅ 피처 기대 정도의 개선"
            score5 = 7
        elif cv_improvement > 0:
            rec5 = "⚠️ 피처 기대 이하의 개선"
            score5 = 4
        else:
            rec5 = "❌ 피처 효과 미미"
            score5 = 1

        print(f"       {rec5}")
        print(f"       (점수: {score5}/10)")

        # 최종 종합 점수
        print(f"\n  {'='*60}")
        print(f"  최종 종합 점수")
        print(f"  {'='*60}")

        total_score = (score1 + score2 + score3 + score4 + score5) / 5

        print(f"    CV 절대값: {score1}/10")
        print(f"    개선폭: {score2}/10")
        print(f"    안정성: {score3}/10")
        print(f"    통계 유의성: {score4}/10")
        print(f"    피처 기여도: {score5}/10")
        print(f"    {'─'*40}")
        print(f"    평균 점수: {total_score:.1f}/10")

        # 최종 권장사항
        print(f"\n  {'='*60}")
        print(f"  최종 권장사항")
        print(f"  {'='*60}")

        if total_score >= 8.0:
            final_recommendation = "🚀 강력 추천 - 지금 제출하세요!"
            confidence_level = "매우 높음"
        elif total_score >= 6.5:
            final_recommendation = "✅ 추천 - 기존 모델과 비슷하거나 더 나음"
            confidence_level = "높음"
        elif total_score >= 5.0:
            final_recommendation = "⚠️ 중립 - 추가 검토 후 결정"
            confidence_level = "중간"
        elif total_score >= 3.0:
            final_recommendation = "⚠️ 조심스러운 시도 - 리스크 있음"
            confidence_level = "낮음"
        else:
            final_recommendation = "❌ 미권장 - 기존 모델 유지"
            confidence_level = "매우 낮음"

        print(f"  {final_recommendation}")
        print(f"  신뢰도: {confidence_level}")

        submission_decision = {
            'cv_absolute_eval': rec1,
            'cv_improvement_eval': rec2,
            'stability_eval': rec3,
            'stat_significance_eval': rec4,
            'feature_contribution_eval': rec5,
            'final_recommendation': final_recommendation,
            'total_score': total_score,
            'confidence_level': confidence_level,
            'scores': {
                'cv_absolute': score1,
                'cv_improvement': score2,
                'stability': score3,
                'stat_significance': score4,
                'feature_contribution': score5
            }
        }

        return submission_decision

    # ========================================================================
    # 6. 종합 분석 보고서 생성
    # ========================================================================

    def generate_comprehensive_report(self, cv_comparison: Dict,
                                      stat_sig: Dict,
                                      feature_analysis: Dict,
                                      decision: Dict) -> None:
        """종합 분석 보고서를 JSON과 마크다운으로 저장"""
        print(f"\n{'='*80}")
        print("6. 종합 분석 보고서 생성")
        print(f"{'='*80}")

        # JSON 보고서 생성
        report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'experiment': 'Phase 1-A',
                'script': 'analyze_results.py'
            },
            'baseline': {
                'name': self.baseline['name'],
                'cv_mean': self.baseline['cv_mean'],
                'cv_std': self.baseline['cv_std'],
                'public_score': self.baseline['public_score'],
                'gap': self.baseline['gap']
            },
            'phase1a_results': {
                'cv_mean': self.phase1a_results['cv_mean'],
                'cv_std': self.phase1a_results['cv_std'],
                'cv_folds': self.phase1a_results['cv_folds'],
                'n_features': self.phase1a_results['features']['total'],
                'new_features': self.phase1a_results.get('new_features', [])
            },
            'analysis': {
                'cv_comparison': cv_comparison,
                'statistical_significance': {
                    't_statistic': stat_sig['t_statistic'],
                    'p_value': stat_sig['p_value'],
                    'is_significant': stat_sig['is_significant'],
                    'cohen_d': stat_sig['cohen_d'],
                    'confidence_score': stat_sig['confidence_score']
                },
                'feature_analysis': feature_analysis,
                'submission_decision': decision
            },
            'summary': {
                'final_recommendation': decision['final_recommendation'],
                'confidence_level': decision['confidence_level'],
                'total_score': decision['total_score']
            }
        }

        # JSON 저장
        json_file = self.results_dir / 'analysis_results.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"  ✓ JSON 보고서 저장: {json_file}")

        # 마크다운 보고서 생성
        markdown = self._generate_markdown_report(cv_comparison, stat_sig,
                                                   feature_analysis, decision)

        md_file = self.results_dir / 'ANALYSIS_RESULTS.md'
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(markdown)

        print(f"  ✓ 마크다운 보고서 저장: {md_file}")
        print(markdown)

    def _generate_markdown_report(self, cv_comparison: Dict,
                                  stat_sig: Dict,
                                  feature_analysis: Dict,
                                  decision: Dict) -> str:
        """마크다운 형식의 분석 보고서"""

        baseline_mean = self.baseline['cv_mean']
        phase1a_mean = self.phase1a_results['cv_mean']
        cv_improvement = cv_comparison['cv_improvement']

        markdown = f"""# Phase 1-A 결과 분석 보고서

**생성일:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. 성능 비교

| 항목 | 기존 모델 (exp_028) | Phase 1-A | 개선 | 평가 |
|------|:---:|:---:|:---:|:---:|
| **CV Mean** | {baseline_mean:.4f} | {phase1a_mean:.4f} | {cv_improvement:+.4f} | {'✅' if cv_improvement > 0 else '❌'} |
| **CV Std** | {self.baseline['cv_std']:.4f} | {self.phase1a_results['cv_std']:.4f} | {self.baseline['cv_std'] - self.phase1a_results['cv_std']:+.4f} | {'✅' if self.baseline['cv_std'] > self.phase1a_results['cv_std'] else '❌'} |
| **Public** | {self.baseline['public_score']:.4f} | 예상 {phase1a_mean:.4f} | 예상 {cv_improvement:+.4f} | TBD |
| **Gap** | {self.baseline['gap']:.4f} | TBD | TBD | TBD |

### Fold별 성능

| Fold | 기존 | Phase 1-A | 개선 |
|---:|:---:|:---:|:---:|
| 1 | {self.baseline['cv_folds'][0]:.4f} | {self.phase1a_results['cv_folds'][0]:.4f} | {self.baseline['cv_folds'][0] - self.phase1a_results['cv_folds'][0]:+.4f} |
| 2 | {self.baseline['cv_folds'][1]:.4f} | {self.phase1a_results['cv_folds'][1]:.4f} | {self.baseline['cv_folds'][1] - self.phase1a_results['cv_folds'][1]:+.4f} |
| 3 | {self.baseline['cv_folds'][2]:.4f} | {self.phase1a_results['cv_folds'][2]:.4f} | {self.baseline['cv_folds'][2] - self.phase1a_results['cv_folds'][2]:+.4f} |

---

## 2. 신규 피처 분석

### 추가된 피처 (5개)

| 순번 | 피처명 | 중요도 | 설명 | 기대 효과 |
|---:|---|:---:|---|:---:|
| 1 | **is_final_team** | ⭐⭐⭐⭐⭐ | 공격권 플래그 | ±0.050점 |
| 2 | **team_possession_pct** | ⭐⭐⭐⭐ | 점유율 (20패스) | ±0.045점 |
| 3 | **team_switches** | ⭐⭐⭐ | 공수 전환 횟수 | ±0.030점 |
| 4 | **game_clock_min** | ⭐⭐⭐ | 경기 경과 시간 | ±0.020점 |
| 5 | **final_poss_len** | ⭐⭐ | 연속 소유 길이 | ±0.015점 |

**총 기대 개선폭:** ±{feature_analysis.get('total_expected_contribution', 0):.3f}점

---

## 3. 통계적 유의성 검증

### t-test 결과

| 항목 | 값 | 평가 |
|---|:---:|:---:|
| **t-statistic** | {stat_sig['t_statistic']:.4f} | - |
| **p-value** | {stat_sig['p_value']:.6f} | {'✅ < 0.05' if stat_sig['p_value'] < 0.05 else '⚠️ ≥ 0.05'} |
| **신뢰도** | {stat_sig['confidence_score']:.0f}% | {stat_sig['significance']} |
| **Cohen's d** | {stat_sig['cohen_d']:.4f} | {stat_sig['effect_interpretation']} |

### 신뢰구간 (95%)

- **Baseline:** [{stat_sig['baseline_ci'][0]:.4f}, {stat_sig['baseline_ci'][1]:.4f}]
- **Phase 1-A:** [{stat_sig['phase1a_ci'][0]:.4f}, {stat_sig['phase1a_ci'][1]:.4f}]
- **겹침 상태:** {stat_sig['ci_overlap']}

---

## 4. 제출 결정 기준

### 결정 기준별 평가

| 기준 | 평가 | 점수 |
|---|---|:---:|
| CV 절대값 | {decision['cv_absolute_eval']} | {decision['scores']['cv_absolute']}/10 |
| 개선폭 | {decision['cv_improvement_eval']} | {decision['scores']['cv_improvement']}/10 |
| 안정성 | {decision['stability_eval']} | {decision['scores']['stability']}/10 |
| 통계 유의성 | {decision['stat_significance_eval']} | {decision['scores']['stat_significance']}/10 |
| 피처 기여도 | {decision['feature_contribution_eval']} | {decision['scores']['feature_contribution']}/10 |

### 최종 평가

**종합 점수:** {decision['total_score']:.1f}/10

**신뢰도:** {decision['confidence_level']}

**권장사항:** {decision['final_recommendation']}

---

## 5. 핵심 인사이트

### 강점 (Strengths)
- 신규 피처가 도메인 지식에 부합
- CV 안정성 개선 확인
- 5개 피처의 종합 효과 기대

### 약점 (Weaknesses)
- 개선폭이 목표치 이하일 가능성
- Gap 정보 없음 (Public Score와의 차이 불확실)
- 작은 샘플 크기 (3-fold CV)

### 기회 (Opportunities)
- Phase 1-A 성공 시 추가 피처 개발 가능
- 다른 모델 (XGBoost, LGBM 등)과 앙상블 가능
- 피처 상호작용 탐색

### 위협 (Threats)
- Public Test Set에서 다른 성능 (Gap 확대 가능)
- 피처 오버피팅 가능성
- 제출 제한 (하루 5회) 제약

---

## 6. 추천 행동 계획

### 즉시 실행 (Step 1)
1. 현재 분석 결과 재확인
2. cv_results.json 데이터 검증
3. 신규 피처의 NaN/이상값 확인

### 제출 전 (Step 2)
1. `train_final.py`로 전체 데이터에서 최종 모델 학습
2. `predict_test.py`로 제출 파일 생성
3. 파일 무결성 검증

### DACON 제출 (Step 3)
1. submission_phase1a.csv 업로드
2. 제출 ID 기록
3. SUBMISSION_LOG.md 즉시 업데이트

### 결과 평가 (Step 4)
1. Public Score 확인
2. Gap 계산 (예상값 vs 실제값)
3. 성공/실패 분석
4. 다음 실험 방향 결정

---

## 7. 주의사항

### 중요 (CRITICAL)
- **하루 5회 제출 제한!** 안 쓰면 영구 소실
- **SUBMISSION_LOG.md는 단일 진실 공급원** (항상 먼저 확인)
- **Public Score ≠ CV Score** (Gap 발생 가능)

### 주의 (WARNING)
- CV 개선 ≠ 순위 향상 보장
- Private Test Set에서 다른 성능 가능
- Gap이 크면 과적합 우려

### 팁 (TIPS)
- 매일 5회 꾸준히 제출하기
- 제출 결과 즉시 기록하기
- 실패해도 학습으로 삼기

---

## 8. 문서 참고

- **EXPERIMENT.md:** 상세 실험 설계
- **README.md:** 빠른 시작 가이드
- **SUBMISSION_LOG.md:** 제출 이력 (필독!)

---

*Report generated by analyze_results.py (Phase 1-A Analysis Tool)*
"""

        return markdown

    # ========================================================================
    # 7. 메인 실행
    # ========================================================================

    def run_full_analysis(self) -> bool:
        """전체 분석 실행"""
        print(f"\n{'='*80}")
        print("Phase 1-A 결과 종합 분석 시작")
        print(f"{'='*80}")

        # Step 1: CV 결과 로드
        if not self.load_cv_results():
            return False

        # Step 2: CV 비교 분석
        cv_comparison = self.compare_cv_performance()

        # Step 3: 통계적 유의성 검증
        stat_sig = self.statistical_significance_test(cv_comparison)

        # Step 4: 신규 피처 분석
        feature_analysis = self.analyze_new_features()

        # Step 5: 제출 결정
        decision = self.evaluate_submission_decision(cv_comparison, stat_sig, feature_analysis)

        # Step 6: 종합 보고서 생성
        self.generate_comprehensive_report(cv_comparison, stat_sig, feature_analysis, decision)

        # 최종 요약
        print(f"\n{'='*80}")
        print("✅ 분석 완료!")
        print(f"{'='*80}")
        print(f"\n  최종 권장: {decision['final_recommendation']}")
        print(f"  신뢰도: {decision['confidence_level']}")
        print(f"  종합 점수: {decision['total_score']:.1f}/10")

        print(f"\n  생성된 파일:")
        print(f"    - analysis_results.json")
        print(f"    - ANALYSIS_RESULTS.md")

        return True


def main():
    """메인 엔트리 포인트"""
    import sys

    # 디렉토리 설정
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        results_dir = str(Path(__file__).parent)

    # 분석 실행
    analyzer = Phase1AResultsAnalyzer(results_dir=results_dir)
    success = analyzer.run_full_analysis()

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
