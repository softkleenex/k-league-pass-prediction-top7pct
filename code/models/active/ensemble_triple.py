"""
Triple Ensemble: CatBoost + XGBoost + LightGBM (로컬 실험)

Ultrathink 2025-12-16:
- 현재 Best: Tuned CatBoost CV 15.60 ± 0.27
- Ensemble: CatBoost + XGBoost + LightGBM
- 목표: CV 15.4-15.5 (△0.1-0.2)

⚠️ 로컬 실험만! 제출 안 함!
"""

import sys
sys.path.append('../../utils')

import pandas as pd
import numpy as np
import catboost as cb
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
import time
import json
import warnings
warnings.filterwarnings('ignore')

from fast_experiment_v2 import FastExperimentV2
from pathlib import Path

print("=" * 80)
print("Triple Ensemble: CatBoost + XGBoost + LightGBM (로컬 실험)")
print("=" * 80)
print("\n현재 Best: Tuned CatBoost CV 15.60")
print("목표: CV 15.4-15.5 (△0.1-0.2)")
print("\n⚠️ 로컬 실험만! 제출 안 함!")

# =============================================================================
# Setup
# =============================================================================
exp = FastExperimentV2(sample_frac=1.0, n_folds=5, random_state=42)

# Load & Features
print("\n" + "=" * 80)
print("데이터 로드 및 피처 생성")
print("=" * 80)

train_df = exp.load_data(train_path='../../../train.csv', sample=False)
train_df = exp.create_features(train_df)
X, y, groups, feature_cols = exp.prepare_data(train_df)

# Load best CatBoost params
with open('../../../logs/best_params.json', 'r') as f:
    best_catboost = json.load(f)

# =============================================================================
# Model Parameters
# =============================================================================
print("\n" + "=" * 80)
print("모델 파라미터")
print("=" * 80)

# CatBoost (tuned)
cat_params = best_catboost['params']
print(f"\n1. CatBoost (Tuned):")
for k, v in cat_params.items():
    print(f"   {k}: {v}")

# XGBoost (similar to CatBoost)
xgb_params = {
    'n_estimators': 300,
    'max_depth': 8,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'tree_method': 'hist',
    'verbosity': 0
}
print(f"\n2. XGBoost:")
for k, v in xgb_params.items():
    print(f"   {k}: {v}")

# LightGBM (similar to CatBoost)
lgb_params = {
    'n_estimators': 300,
    'max_depth': 8,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'verbosity': -1
}
print(f"\n3. LightGBM:")
for k, v in lgb_params.items():
    print(f"   {k}: {v}")

# =============================================================================
# Cross-Validation with Ensemble
# =============================================================================
print("\n" + "=" * 80)
print("Ensemble Cross-Validation")
print("=" * 80)

gkf = GroupKFold(n_splits=5)
fold_scores_cat = []
fold_scores_xgb = []
fold_scores_lgb = []
fold_scores_ensemble = []

start_all = time.time()

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
    print(f"\n[Fold {fold+1}/5]")
    fold_start = time.time()

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # ========== CatBoost ==========
    print("  Training CatBoost...", end=" ")
    cat_x = cb.CatBoostRegressor(**cat_params)
    cat_y = cb.CatBoostRegressor(**cat_params)

    cat_x.fit(X_train, y_train[:, 0])
    cat_y.fit(X_train, y_train[:, 1])

    pred_cat_x = np.clip(cat_x.predict(X_val), 0, 105)
    pred_cat_y = np.clip(cat_y.predict(X_val), 0, 68)

    dist_cat = np.sqrt((pred_cat_x - y_val[:, 0])**2 + (pred_cat_y - y_val[:, 1])**2)
    cv_cat = dist_cat.mean()
    fold_scores_cat.append(cv_cat)
    print(f"CV {cv_cat:.4f}")

    # ========== XGBoost ==========
    print("  Training XGBoost...", end=" ")
    xgb_x = xgb.XGBRegressor(**xgb_params)
    xgb_y = xgb.XGBRegressor(**xgb_params)

    xgb_x.fit(X_train, y_train[:, 0])
    xgb_y.fit(X_train, y_train[:, 1])

    pred_xgb_x = np.clip(xgb_x.predict(X_val), 0, 105)
    pred_xgb_y = np.clip(xgb_y.predict(X_val), 0, 68)

    dist_xgb = np.sqrt((pred_xgb_x - y_val[:, 0])**2 + (pred_xgb_y - y_val[:, 1])**2)
    cv_xgb = dist_xgb.mean()
    fold_scores_xgb.append(cv_xgb)
    print(f"CV {cv_xgb:.4f}")

    # ========== LightGBM ==========
    print("  Training LightGBM...", end=" ")
    lgb_x = lgb.LGBMRegressor(**lgb_params)
    lgb_y = lgb.LGBMRegressor(**lgb_params)

    lgb_x.fit(X_train, y_train[:, 0])
    lgb_y.fit(X_train, y_train[:, 1])

    pred_lgb_x = np.clip(lgb_x.predict(X_val), 0, 105)
    pred_lgb_y = np.clip(lgb_y.predict(X_val), 0, 68)

    dist_lgb = np.sqrt((pred_lgb_x - y_val[:, 0])**2 + (pred_lgb_y - y_val[:, 1])**2)
    cv_lgb = dist_lgb.mean()
    fold_scores_lgb.append(cv_lgb)
    print(f"CV {cv_lgb:.4f}")

    # ========== Ensemble (Simple Average) ==========
    pred_ensemble_x = (pred_cat_x + pred_xgb_x + pred_lgb_x) / 3
    pred_ensemble_y = (pred_cat_y + pred_xgb_y + pred_lgb_y) / 3

    dist_ensemble = np.sqrt((pred_ensemble_x - y_val[:, 0])**2 +
                            (pred_ensemble_y - y_val[:, 1])**2)
    cv_ensemble = dist_ensemble.mean()
    fold_scores_ensemble.append(cv_ensemble)

    fold_time = time.time() - fold_start
    print(f"  Ensemble: {cv_ensemble:.4f} ({fold_time:.1f}s)")

total_time = time.time() - start_all

# =============================================================================
# 결과 요약
# =============================================================================
print("\n" + "=" * 80)
print("결과 요약")
print("=" * 80)

cv_cat_mean = np.mean(fold_scores_cat)
cv_cat_std = np.std(fold_scores_cat)

cv_xgb_mean = np.mean(fold_scores_xgb)
cv_xgb_std = np.std(fold_scores_xgb)

cv_lgb_mean = np.mean(fold_scores_lgb)
cv_lgb_std = np.std(fold_scores_lgb)

cv_ensemble_mean = np.mean(fold_scores_ensemble)
cv_ensemble_std = np.std(fold_scores_ensemble)

print(f"\n{'Model':<20} {'CV':<18} {'vs Best':<12}")
print("-" * 52)
print(f"{'CatBoost (Tuned)':<20} {cv_cat_mean:.4f}±{cv_cat_std:.4f}  {cv_cat_mean - 15.60:+.4f}")
print(f"{'XGBoost':<20} {cv_xgb_mean:.4f}±{cv_xgb_std:.4f}  {cv_xgb_mean - 15.60:+.4f}")
print(f"{'LightGBM':<20} {cv_lgb_mean:.4f}±{cv_lgb_std:.4f}  {cv_lgb_mean - 15.60:+.4f}")
print(f"{'Ensemble (Avg)':<20} {cv_ensemble_mean:.4f}±{cv_ensemble_std:.4f}  {cv_ensemble_mean - 15.60:+.4f} ⭐")

print(f"\n총 실행 시간: {total_time:.1f}s")

# 개선 분석
baseline_best = 15.60
improvement = baseline_best - cv_ensemble_mean

print("\n" + "=" * 80)
print("개선 분석")
print("=" * 80)

print(f"\nTuned CatBoost: {baseline_best:.4f}")
print(f"Ensemble:       {cv_ensemble_mean:.4f}")
print(f"개선:           {improvement:+.4f}점")

if improvement > 0:
    print(f"\n✅ Ensemble 성공! {improvement:.2f}점 개선")

    # 누적 개선
    baseline_zone = 16.04
    cumulative = baseline_zone - cv_ensemble_mean

    print(f"\n누적 개선 (Zone 6x6 대비):")
    print(f"  Zone 6x6:        16.04")
    print(f"  → Batch 1:       15.79 (△0.25)")
    print(f"  → Tuning:        15.60 (△0.19)")
    print(f"  → Ensemble:      {cv_ensemble_mean:.2f} (△{improvement:.2f})")
    print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  총 누적:         {cumulative:.2f}점")

    if cumulative >= 0.5:
        print(f"\n🎯🎯 제출 기준 초과 달성! (△0.5 이상)")
        print(f"   Public 예상: {cv_ensemble_mean + 0.35:.2f} (Gap 0.35 가정)")
        print(f"   vs 현재 Best (16.14): {16.14 - (cv_ensemble_mean + 0.35):.2f}점 개선")
        print(f"\n다음 단계:")
        print(f"  1. Test 예측")
        print(f"  2. 제출!")
    else:
        print(f"\n⏳ 더 개선 가능? (현재 △{cumulative:.2f})")
        print(f"   추가 필요: {0.5 - cumulative:.2f}점")

else:
    print(f"\n❌ Ensemble 효과 없음 ({improvement:.2f}점)")
    print(f"\n다음 단계:")
    print(f"  1. Tuned CatBoost로 제출")
    print(f"  2. 또는 Stacking 시도")

# Save results
results = {
    'catboost': {'cv_mean': float(cv_cat_mean), 'cv_std': float(cv_cat_std)},
    'xgboost': {'cv_mean': float(cv_xgb_mean), 'cv_std': float(cv_xgb_std)},
    'lightgbm': {'cv_mean': float(cv_lgb_mean), 'cv_std': float(cv_lgb_std)},
    'ensemble': {'cv_mean': float(cv_ensemble_mean), 'cv_std': float(cv_ensemble_std)},
    'improvement': float(improvement),
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
}

output_file = Path('../../../logs/ensemble_results.json')
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Ensemble 결과 저장: {output_file}")

print("\n" + "=" * 80)
print("✅ Ensemble 로컬 실험 완료!")
print("=" * 80)
