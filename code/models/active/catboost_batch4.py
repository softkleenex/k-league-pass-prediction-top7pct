"""
CatBoost Batch 4 - Period 상호작용 (로컬 실험)

Ultrathink 2025-12-16:
- Batch 1: CV 15.79 ± 0.27 (Public 16.14)
- Batch 4: +4 features (Period 상호작용)
- 목표: CV 15.74 (△0.05)

⚠️ 로컬 실험만! 제출 안 함!
"""

import sys
sys.path.append('../../utils')

import pandas as pd
import numpy as np
import catboost as cb
import time
import warnings
warnings.filterwarnings('ignore')

from fast_experiment_v4 import FastExperimentV4

print("=" * 80)
print("CatBoost Batch 4 - Period 상호작용 (로컬 실험)")
print("=" * 80)
print("\nBatch 1: CV 15.79 ± 0.27 (Public 16.14)")
print("Batch 4: +4 features (Period 상호작용)")
print("목표: CV 15.74 (△0.05)")
print("\n⚠️ 로컬 실험만! 제출 안 함!")

# =============================================================================
# Full Data
# =============================================================================
print("\n" + "=" * 80)
print("Full Data (100%)")
print("=" * 80)

exp = FastExperimentV4(sample_frac=1.0, n_folds=5, random_state=42)

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
    model_name='CatBoost Batch4 (100%)'
)

runtime = time.time() - start

# Log
exp.log_experiment(
    name='catboost_batch4_100pct_local',
    cv=(cv_mean, cv_std, fold_scores),
    params=cat_params,
    features=feature_cols,
    runtime=runtime,
    notes='Batch 4: Period interaction (LOCAL ONLY)'
)

# =============================================================================
# 결과 비교
# =============================================================================
print("\n" + "=" * 80)
print("결과 비교")
print("=" * 80)

batch1_cv = 15.79
improvement = batch1_cv - cv_mean

results = [
    ('Batch 1 (16 features)', batch1_cv, 0.27, 2.6),
    ('Batch 4 (20 features)', cv_mean, cv_std, runtime)
]

print(f"\n{'Model':<25} {'CV':<18} {'Runtime':<10} {'vs Batch1':<12}")
print("-" * 68)
for i, (name, cv, std, rt) in enumerate(results):
    if i == 0:
        vs_prev = '-'
    else:
        diff = cv - batch1_cv
        vs_prev = f'{diff:+.4f}'

    print(f"{name:<25} {cv:.4f}±{std:.4f}    {rt:<10.1f}s {vs_prev:<12}")

# Analysis
print(f"\n개선 분석:")
print(f"  Batch 1 → Batch 4: {improvement:+.4f}점")

if improvement > 0:
    print(f"\n✅ Batch 4가 {improvement:.2f}점 개선!")

    if improvement >= 0.05:
        print(f"   → ✅ 목표 달성! (△0.05 이상)")
        print(f"\n다음 단계:")
        print(f"  1. ✅ Batch 4 효과 확인")
        print(f"  2. Hyperparameter Tuning 시도")
        print(f"  3. 누적 개선 추적")
    else:
        print(f"   → ⚠️ 작은 개선 (△0.05 미만)")
        print(f"\n다음 단계:")
        print(f"  1. Hyperparameter Tuning으로 추가 개선")
        print(f"  2. 누적 개선 △0.3 달성 시 제출")

else:
    print(f"\n❌ Batch 4가 {-improvement:.2f}점 악화...")
    print(f"\n다음 단계:")
    print(f"  1. Batch 1로 롤백")
    print(f"  2. 다른 피처 시도")

# 누적 개선 추적
print("\n" + "=" * 80)
print("누적 개선 추적 (Baseline 대비)")
print("=" * 80)

baseline_cv = 16.04
cumulative = baseline_cv - cv_mean

print(f"\nBaseline: {baseline_cv:.2f}")
print(f"Batch 1:  {batch1_cv:.2f} (△{baseline_cv - batch1_cv:+.2f})")
print(f"Batch 4:  {cv_mean:.2f} (△{baseline_cv - cv_mean:+.2f})")
print(f"\n누적 개선: {cumulative:.2f}점")

if cumulative >= 0.3:
    print(f"\n🎯 제출 기준 달성! (△0.3 이상)")
    print(f"   Public 예상: {cv_mean + 0.35:.2f} (Gap 0.35 가정)")
    print(f"   vs 현재 Best (16.14): {16.14 - (cv_mean + 0.35):.2f}점 개선")
else:
    print(f"\n⏳ 제출 기준 미달 (△0.3 필요, 현재 △{cumulative:.2f})")
    print(f"   추가 필요: {0.3 - cumulative:.2f}점")
    print(f"   다음: Hyperparameter Tuning")

print("\n" + "=" * 80)
print("✅ Batch 4 로컬 실험 완료!")
print("=" * 80)
