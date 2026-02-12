"""
CatBoost Full Data (100%)

Ultrathink 2025-12-15:
- 10% 샘플: CV 16.77 (Zone 6x6보다 +0.41 나쁨)
- Full data로 재학습 → 개선 기대!
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

from fast_experiment import FastExperiment

print("=" * 80)
print("CatBoost Full Data (100%)")
print("=" * 80)
print("\n10% 샘플 결과: CV 16.77 ± 0.25")
print("Zone 6x6 Baseline: 16.36")
print("목표: Full data로 16.36 이하")
print("\n예상 시간: 1-3분")

# =============================================================================
# Setup
# =============================================================================
exp = FastExperiment(sample_frac=1.0, n_folds=5, random_state=42)  # Full data, 5-fold

# Load data (NO sampling)
train_df = exp.load_data(train_path='../../../train.csv', sample=False)

# Create features
train_df = exp.create_features(train_df)

# Prepare
X, y, groups, feature_cols = exp.prepare_data(train_df)

# =============================================================================
# CatBoost Full Data
# =============================================================================
print("\n" + "=" * 80)
print("CatBoost Training (Full Data, 5-fold)")
print("=" * 80)

start = time.time()

cat_params = {
    'iterations': 100,
    'depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'random_state': 42,
    'verbose': 0
}

gkf = GroupKFold(n_splits=5)
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
    fold_start = time.time()

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Separate models for x and y
    model_x = cb.CatBoostRegressor(**cat_params)
    model_y = cb.CatBoostRegressor(**cat_params)

    model_x.fit(X_train, y_train[:, 0])
    model_y.fit(X_train, y_train[:, 1])

    pred_x = np.clip(model_x.predict(X_val), 0, 105)
    pred_y = np.clip(model_y.predict(X_val), 0, 68)

    dist = np.sqrt((pred_x - y_val[:, 0])**2 + (pred_y - y_val[:, 1])**2)
    cv = dist.mean()
    fold_scores.append(cv)

    fold_time = time.time() - fold_start
    print(f"  Fold {fold+1}: {cv:.4f}  ({fold_time:.1f}s)")

cv_mean = np.mean(fold_scores)
cv_std = np.std(fold_scores)
total_time = time.time() - start

print(f"\n  CatBoost Full CV: {cv_mean:.4f} ± {cv_std:.4f}")
print(f"  Total Runtime: {total_time:.1f}s")

# Log
exp.log_experiment(
    name='catboost_full_100pct',
    cv=(cv_mean, cv_std, fold_scores),
    params=cat_params,
    features=feature_cols,
    runtime=total_time,
    notes='Full data (100%), 5-fold CV'
)

# =============================================================================
# Comparison
# =============================================================================
print("\n" + "=" * 80)
print("결과 비교")
print("=" * 80)

baseline_zone66 = 16.36
sample10_cv = 16.77

results = [
    ('Zone 6x6 (Baseline)', baseline_zone66, 0.006, 0, '-'),
    ('CatBoost 10%', sample10_cv, 0.25, 1.1, '❌ +0.41'),
    ('CatBoost 100%', cv_mean, cv_std, total_time, '?')
]

print(f"\n{'Model':<25} {'CV':<15} {'Runtime':<10} {'vs Zone':<12}")
print("-" * 65)
for name, cv, std, runtime, vs_zone in results:
    if runtime == 0:
        rt_str = '-'
    else:
        rt_str = f"{runtime:.1f}s"

    if name == 'CatBoost 100%':
        diff = cv - baseline_zone66
        if diff < 0:
            symbol = f'✅ {diff:.2f}'
        else:
            symbol = f'❌ +{diff:.2f}'
        vs_zone = symbol

    print(f"{name:<25} {cv:.4f}±{std:.4f}  {rt_str:<10} {vs_zone:<12}")

# Analysis
print("\n" + "=" * 80)
print("분석")
print("=" * 80)

diff = cv_mean - baseline_zone66

if diff < 0:
    print(f"✅ SUCCESS! CatBoost가 Zone 6x6보다 {-diff:.2f}점 개선!")
    print(f"\n   10% 샘플: 16.77 (+0.41)")
    print(f"   100% Full: {cv_mean:.2f} ({diff:.2f})")
    print(f"   개선: {16.77 - cv_mean:.2f}점")
    print(f"\n   예상 Public: {cv_mean + 1.0:.2f} (Gap ~1.0 가정)")
    print(f"   예상 순위: ~200위 (상위 20%)")
    print("\n다음 단계:")
    print("1. ✅ Full data 개선 확인")
    print("2. Feature engineering 추가")
    print("3. Hyperparameter tuning")
    print("4. Test 예측 및 제출")
else:
    print(f"⚠️ CatBoost가 여전히 Zone 6x6보다 {diff:.2f}점 나쁨")
    print(f"\n   10% → 100% 개선: {16.77 - cv_mean:.2f}점")
    if 16.77 - cv_mean > 0.2:
        print(f"   → Full data로 개선되었지만 아직 부족")
    else:
        print(f"   → Full data로도 큰 개선 없음")
    print("\n다음 단계:")
    print("1. Feature engineering 강화 (우선!)")
    print("2. Hyperparameter tuning")
    print("3. 다른 접근 고려 (Zone 10x10, Quantile)")

print("\n" + "=" * 80)
print("✅ CatBoost Full Data 완료!")
print("=" * 80)

# Show all experiments
exp.compare_experiments()
