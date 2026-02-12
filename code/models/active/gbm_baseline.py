"""
GBM Baseline Comparison

3개 라이브러리 비교:
- XGBoost
- LightGBM
- CatBoost

Ultrathink 2025-12-15:
- 10% 샘플로 빠른 비교
- Episode 독립성 유지
- Separate models for x and y
"""

import sys
sys.path.append('../../utils')

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import time
import warnings
warnings.filterwarnings('ignore')

from fast_experiment import FastExperiment

print("=" * 80)
print("GBM Baseline Comparison (10% Sample)")
print("=" * 80)
print("\n목표: XGBoost, LightGBM, CatBoost 비교")
print("예상 CV: 15.5-16.5 (Zone 6x6: 16.36)")
print("예상 시간: 10-60초 (10% 샘플)")

# =============================================================================
# Setup
# =============================================================================
exp = FastExperiment(sample_frac=0.1, n_folds=3, random_state=42)

# Load data
train_df = exp.load_data(train_path='../../../train.csv', sample=True)

# Create features
train_df = exp.create_features(train_df)

# Prepare
X, y, groups, feature_cols = exp.prepare_data(train_df)

# =============================================================================
# 1. XGBoost
# =============================================================================
print("\n" + "=" * 80)
print("[1] XGBoost")
print("=" * 80)

start = time.time()

# Separate models for x and y
xgb_params = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0
}

# Cross-validation
gkf = GroupKFold(n_splits=3)
fold_scores_xgb = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Separate models
    model_x = xgb.XGBRegressor(**xgb_params)
    model_y = xgb.XGBRegressor(**xgb_params)

    # Fit
    model_x.fit(X_train, y_train[:, 0])
    model_y.fit(X_train, y_train[:, 1])

    # Predict
    pred_x = model_x.predict(X_val)
    pred_y = model_y.predict(X_val)

    # Clip
    pred_x = np.clip(pred_x, 0, 105)
    pred_y = np.clip(pred_y, 0, 68)

    # Score
    dist = np.sqrt((pred_x - y_val[:, 0])**2 + (pred_y - y_val[:, 1])**2)
    cv = dist.mean()
    fold_scores_xgb.append(cv)

    print(f"  Fold {fold+1}: {cv:.4f}")

xgb_cv = np.mean(fold_scores_xgb)
xgb_std = np.std(fold_scores_xgb)
xgb_time = time.time() - start

print(f"\n  XGBoost CV: {xgb_cv:.4f} ± {xgb_std:.4f}")
print(f"  Runtime: {xgb_time:.1f}s")

# Log
exp.log_experiment(
    name='xgb_baseline_10pct',
    cv=(xgb_cv, xgb_std, fold_scores_xgb),
    params=xgb_params,
    features=feature_cols,
    runtime=xgb_time,
    notes='10% sample, separate models for x and y'
)

# =============================================================================
# 2. LightGBM
# =============================================================================
print("\n" + "=" * 80)
print("[2] LightGBM")
print("=" * 80)

start = time.time()

lgb_params = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

fold_scores_lgb = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model_x = lgb.LGBMRegressor(**lgb_params)
    model_y = lgb.LGBMRegressor(**lgb_params)

    model_x.fit(X_train, y_train[:, 0])
    model_y.fit(X_train, y_train[:, 1])

    pred_x = np.clip(model_x.predict(X_val), 0, 105)
    pred_y = np.clip(model_y.predict(X_val), 0, 68)

    dist = np.sqrt((pred_x - y_val[:, 0])**2 + (pred_y - y_val[:, 1])**2)
    cv = dist.mean()
    fold_scores_lgb.append(cv)

    print(f"  Fold {fold+1}: {cv:.4f}")

lgb_cv = np.mean(fold_scores_lgb)
lgb_std = np.std(fold_scores_lgb)
lgb_time = time.time() - start

print(f"\n  LightGBM CV: {lgb_cv:.4f} ± {lgb_std:.4f}")
print(f"  Runtime: {lgb_time:.1f}s")

exp.log_experiment(
    name='lgb_baseline_10pct',
    cv=(lgb_cv, lgb_std, fold_scores_lgb),
    params=lgb_params,
    features=feature_cols,
    runtime=lgb_time,
    notes='10% sample, separate models for x and y'
)

# =============================================================================
# 3. CatBoost
# =============================================================================
print("\n" + "=" * 80)
print("[3] CatBoost")
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

fold_scores_cat = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model_x = cb.CatBoostRegressor(**cat_params)
    model_y = cb.CatBoostRegressor(**cat_params)

    model_x.fit(X_train, y_train[:, 0])
    model_y.fit(X_train, y_train[:, 1])

    pred_x = np.clip(model_x.predict(X_val), 0, 105)
    pred_y = np.clip(model_y.predict(X_val), 0, 68)

    dist = np.sqrt((pred_x - y_val[:, 0])**2 + (pred_y - y_val[:, 1])**2)
    cv = dist.mean()
    fold_scores_cat.append(cv)

    print(f"  Fold {fold+1}: {cv:.4f}")

cat_cv = np.mean(fold_scores_cat)
cat_std = np.std(fold_scores_cat)
cat_time = time.time() - start

print(f"\n  CatBoost CV: {cat_cv:.4f} ± {cat_std:.4f}")
print(f"  Runtime: {cat_time:.1f}s")

exp.log_experiment(
    name='cat_baseline_10pct',
    cv=(cat_cv, cat_std, fold_scores_cat),
    params=cat_params,
    features=feature_cols,
    runtime=cat_time,
    notes='10% sample, separate models for x and y'
)

# =============================================================================
# Comparison
# =============================================================================
print("\n" + "=" * 80)
print("최종 비교")
print("=" * 80)

results = [
    ('XGBoost', xgb_cv, xgb_std, xgb_time),
    ('LightGBM', lgb_cv, lgb_std, lgb_time),
    ('CatBoost', cat_cv, cat_std, cat_time)
]

results = sorted(results, key=lambda x: x[1])

baseline_zone66 = 16.36

print(f"\n{'Rank':<5} {'Model':<12} {'CV':<15} {'Runtime':<10} {'vs Zone':<12}")
print("-" * 65)
for i, (name, cv, std, runtime) in enumerate(results):
    diff = cv - baseline_zone66
    symbol = '✅' if diff < 0 else '❌'
    print(f"{i+1:<5} {name:<12} {cv:.4f}±{std:.4f}  {runtime:.1f}s      {diff:+.2f} {symbol}")

print(f"\n{'Baseline Zone 6x6:':<20} {baseline_zone66:.4f}")

best_name, best_cv, best_std, best_time = results[0]
print(f"\n✅ Best: {best_name}")
print(f"   CV: {best_cv:.4f} ± {best_std:.4f}")
print(f"   Runtime: {best_time:.1f}s")
print(f"   vs Zone 6x6: {best_cv - baseline_zone66:+.2f}")

# Analysis
print("\n" + "=" * 80)
print("분석")
print("=" * 80)

if best_cv < baseline_zone66:
    improvement = baseline_zone66 - best_cv
    print(f"✅ GBM이 Zone 6x6보다 {improvement:.2f}점 개선!")
    print(f"   예상 Public: {baseline_zone66 - improvement:.2f} (Gap ~1.0 가정)")
    print(f"   예상 순위: ~220위 (상위 22%)")
    print("\n다음 단계:")
    print("1. Full data (100%)로 재학습")
    print("2. Feature engineering 추가")
    print("3. Hyperparameter tuning")
else:
    gap = best_cv - baseline_zone66
    print(f"⚠️ GBM이 Zone 6x6보다 {gap:.2f}점 나쁨")
    print(f"   원인 가능성:")
    print(f"   - 10% 샘플이 너무 적음 (Full data 필요)")
    print(f"   - 기본 파라미터가 적합하지 않음 (Tuning 필요)")
    print(f"   - Feature가 부족함 (Engineering 필요)")
    print("\n다음 단계:")
    print("1. Full data (100%)로 재학습 (우선)")
    print("2. Feature engineering")
    print("3. Hyperparameter tuning")

print("\n" + "=" * 80)
print("✅ GBM Baseline 완료!")
print("=" * 80)

# Show all experiments
exp.compare_experiments()
