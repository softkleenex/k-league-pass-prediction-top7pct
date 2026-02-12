"""
CatBoost Batch 1 Features

Ultrathink 2025-12-16:
- Baseline: CV 16.04 ± 0.27 (13 features)
- Batch 1: +3 features (is_home, type_name, result_name)
- 목표: CV 15.7-15.9 (△0.1-0.3 개선)

실험:
1. 10% 샘플로 빠른 검증
2. Full data로 최종 확인
"""

import sys
sys.path.append('../../utils')

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
import catboost as cb
import time
import warnings
warnings.filterwarnings('ignore')

from fast_experiment_v2 import FastExperimentV2

print("=" * 80)
print("CatBoost Batch 1 Features")
print("=" * 80)
print("\n기존 Baseline: CV 16.04 ± 0.27 (13 features)")
print("Batch 1: +3 features (is_home, type, result)")
print("목표: CV 15.7-15.9 (△0.1-0.3)")
print("\n예상 시간: 10% 샘플 1초, Full data 3초")

# =============================================================================
# Experiment 1: 10% 샘플로 빠른 검증
# =============================================================================
print("\n" + "=" * 80)
print("Experiment 1: 10% 샘플 (빠른 검증)")
print("=" * 80)

exp_10 = FastExperimentV2(sample_frac=0.1, n_folds=3, random_state=42)

# Load & Features
train_df = exp_10.load_data(train_path='../../../train.csv', sample=True)
train_df = exp_10.create_features(train_df)
X, y, groups, feature_cols = exp_10.prepare_data(train_df)

# CatBoost
cat_params = {
    'iterations': 100,
    'depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'random_state': 42,
    'verbose': 0
}

start = time.time()

model_x = cb.CatBoostRegressor(**cat_params)
model_y = cb.CatBoostRegressor(**cat_params)

cv_mean, cv_std, fold_scores = exp_10.run_cv(
    model_x, model_y, X, y, groups,
    model_name='CatBoost Batch1 (10%)'
)

runtime_10 = time.time() - start

# Log
exp_10.log_experiment(
    name='catboost_batch1_10pct',
    cv=(cv_mean, cv_std, fold_scores),
    params=cat_params,
    features=feature_cols,
    runtime=runtime_10,
    notes='Batch 1: is_home, type, result (10% sample)'
)

# =============================================================================
# Experiment 2: Full data
# =============================================================================
print("\n" + "=" * 80)
print("Experiment 2: Full Data (100%)")
print("=" * 80)

# 10% 결과 체크
improvement_10 = 16.77 - cv_mean  # 16.77 = 이전 10% baseline

if improvement_10 > 0:
    print(f"\n✅ 10% 샘플에서 {improvement_10:.2f}점 개선!")
    print("→ Full data 진행!")
else:
    print(f"\n⚠️ 10% 샘플에서 {-improvement_10:.2f}점 악화...")
    print("→ Full data로 확인 필요")

exp_100 = FastExperimentV2(sample_frac=1.0, n_folds=5, random_state=42)

# Load & Features
train_df = exp_100.load_data(train_path='../../../train.csv', sample=False)
train_df = exp_100.create_features(train_df)
X, y, groups, feature_cols = exp_100.prepare_data(train_df)

start = time.time()

# CatBoost
model_x_full = cb.CatBoostRegressor(**cat_params)
model_y_full = cb.CatBoostRegressor(**cat_params)

cv_mean_full, cv_std_full, fold_scores_full = exp_100.run_cv(
    model_x_full, model_y_full, X, y, groups,
    model_name='CatBoost Batch1 (100%)'
)

runtime_100 = time.time() - start

# Log
exp_100.log_experiment(
    name='catboost_batch1_100pct',
    cv=(cv_mean_full, cv_std_full, fold_scores_full),
    params=cat_params,
    features=feature_cols,
    runtime=runtime_100,
    notes='Batch 1: is_home, type, result (full data)'
)

# =============================================================================
# 결과 비교
# =============================================================================
print("\n" + "=" * 80)
print("결과 비교")
print("=" * 80)

baseline_cv = 16.04
improvement = baseline_cv - cv_mean_full

results = [
    ('Baseline (13 features)', baseline_cv, 0.27, 2.5, '-'),
    ('Batch 1 10% (16 features)', cv_mean, cv_std, runtime_10, f'{cv_mean - 16.77:.2f}'),
    ('Batch 1 100% (16 features)', cv_mean_full, cv_std_full, runtime_100, '?')
]

print(f"\n{'Model':<30} {'CV':<18} {'Runtime':<10} {'vs Prev':<10}")
print("-" * 70)
for name, cv, std, runtime, vs_prev in results:
    print(f"{name:<30} {cv:.4f}±{std:.4f}    {runtime:<10.1f}s {vs_prev:<10}")

# Improvement
print(f"\n개선 분석:")
print(f"  Baseline → Batch 1: {improvement:+.4f}점")

if improvement > 0:
    print(f"\n✅ SUCCESS! Batch 1이 {improvement:.2f}점 개선!")
    print(f"   목표: 0.1-0.3점 개선")
    print(f"   실제: {improvement:.2f}점")

    if improvement >= 0.3:
        print(f"   → 🔥 목표 초과 달성!")
    elif improvement >= 0.1:
        print(f"   → ✅ 목표 달성!")
    else:
        print(f"   → ⚠️ 목표 미달 (하지만 개선은 개선)")

    print(f"\n예상 Public: {cv_mean_full + 1.0:.2f} (Gap ~1.0 가정)")
    print(f"\n다음 단계:")
    print("  1. ✅ Batch 1 효과 확인")
    print("  2. Batch 2 시도 (team encoding)")
    print("  3. Batch 3 시도 (episode context)")

else:
    print(f"\n❌ Batch 1이 {-improvement:.2f}점 악화...")
    print(f"\n가능한 원인:")
    print("  1. 새 피처가 noise만 추가")
    print("  2. CatBoost 과적합 (iterations 줄이기?)")
    print("  3. 피처 encoding 문제")
    print(f"\n다음 단계:")
    print("  1. 피처별 중요도 확인")
    print("  2. Hyperparameter tuning")
    print("  3. 다른 피처 조합 시도")

print("\n" + "=" * 80)
print("✅ Batch 1 실험 완료!")
print("=" * 80)
