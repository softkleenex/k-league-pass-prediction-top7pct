"""
CatBoost Batch 2 Features (Team Encoding)

Ultrathink 2025-12-16:
- Batch 1: CV 15.79 ± 0.27 (16 features)
- Batch 2: +3 features (team_end_x_mean, team_end_y_mean, team_dx_mean)
- 목표: CV 15.5-15.7 (△0.1-0.3 개선)
"""

import sys
sys.path.append('../../utils')

import pandas as pd
import numpy as np
import catboost as cb
import time
import warnings
warnings.filterwarnings('ignore')

from fast_experiment_v3 import FastExperimentV3

print("=" * 80)
print("CatBoost Batch 2 Features (Team Encoding)")
print("=" * 80)
print("\nBatch 1: CV 15.79 ± 0.27 (16 features)")
print("Batch 2: +3 features (team encoding)")
print("목표: CV 15.5-15.7 (△0.1-0.3)")

# =============================================================================
# Experiment: Full Data
# =============================================================================
print("\n" + "=" * 80)
print("Full Data (100%)")
print("=" * 80)

exp = FastExperimentV3(sample_frac=1.0, n_folds=5, random_state=42)

# Load & Features
train_df = exp.load_data(train_path='../../../train.csv', sample=False)
train_df = exp.create_features(train_df)
X, y, groups, feature_cols = exp.prepare_data(train_df)

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

cv_mean, cv_std, fold_scores = exp.run_cv(
    model_x, model_y, X, y, groups,
    model_name='CatBoost Batch2 (100%)'
)

runtime = time.time() - start

# Log
exp.log_experiment(
    name='catboost_batch2_100pct',
    cv=(cv_mean, cv_std, fold_scores),
    params=cat_params,
    features=feature_cols,
    runtime=runtime,
    notes='Batch 2: team encoding (full data)'
)

# =============================================================================
# 결과 비교
# =============================================================================
print("\n" + "=" * 80)
print("결과 비교")
print("=" * 80)

baseline_cv = 16.04
batch1_cv = 15.79
improvement = batch1_cv - cv_mean

results = [
    ('Baseline (13 features)', baseline_cv, 0.27, 2.5),
    ('Batch 1 (16 features)', batch1_cv, 0.27, 2.6),
    ('Batch 2 (19 features)', cv_mean, cv_std, runtime)
]

print(f"\n{'Model':<25} {'CV':<18} {'Runtime':<10} {'vs Batch1':<12}")
print("-" * 68)
for i, (name, cv, std, rt) in enumerate(results):
    if i < 2:
        vs_prev = '-'
    else:
        diff = cv - batch1_cv
        vs_prev = f'{diff:+.4f}'

    print(f"{name:<25} {cv:.4f}±{std:.4f}    {rt:<10.1f}s {vs_prev:<12}")

# Analysis
print(f"\n개선 분석:")
print(f"  Batch 1 → Batch 2: {improvement:+.4f}점")

if improvement > 0:
    print(f"\n✅ SUCCESS! Batch 2가 {improvement:.2f}점 개선!")
    print(f"   목표: 0.1-0.3점 개선")
    print(f"   실제: {improvement:.2f}점")

    if improvement >= 0.3:
        print(f"   → 🔥 목표 초과 달성!")
    elif improvement >= 0.1:
        print(f"   → ✅ 목표 달성!")
    else:
        print(f"   → ⚠️ 목표 미달 (하지만 개선)")

    cumulative = baseline_cv - cv_mean
    print(f"\n누적 개선:")
    print(f"  Baseline → Batch 2: {cumulative:.2f}점")
    print(f"  CV: 16.04 → {cv_mean:.2f}")

    print(f"\n예상 Public: {cv_mean + 1.0:.2f} (Gap ~1.0 가정)")

    if cv_mean < 15.6:
        print(f"\n🎯 Phase 2 목표 달성! (CV < 15.6)")
        print(f"\n다음 단계:")
        print("  1. ✅ Phase 2 완료")
        print("  2. Test 예측 및 제출")
        print("  3. Gap 확인")
        print("  4. Phase 3 Ensemble (선택)")
    else:
        print(f"\n다음 단계:")
        print("  1. Batch 3 시도 (episode context)")
        print("  2. Hyperparameter tuning")
        print("  3. Feature importance 확인")

else:
    print(f"\n❌ Batch 2가 {-improvement:.2f}점 악화...")
    print(f"\n가능한 원인:")
    print("  1. Team 피처가 noise만 추가")
    print("  2. 과적합")
    print("  3. Batch 1이 이미 충분")

    print(f"\n다음 단계:")
    print("  1. Batch 1로 롤백")
    print("  2. 다른 피처 시도")
    print("  3. Hyperparameter tuning")

print("\n" + "=" * 80)
print("✅ Batch 2 실험 완료!")
print("=" * 80)
