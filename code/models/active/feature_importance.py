"""
Feature Importance 분석

Ultrathink 2025-12-16:
- 현재 Best: Tuned CatBoost CV 15.60 (16 features)
- Feature Importance로 불필요한 피처 제거
- 목표: CV 15.5-15.55 (단순화로 과적합 방지)
"""

import sys
sys.path.append('../../utils')

import pandas as pd
import numpy as np
import catboost as cb
import json
import warnings
warnings.filterwarnings('ignore')

from fast_experiment_v2 import FastExperimentV2
from pathlib import Path

print("=" * 80)
print("Feature Importance 분석")
print("=" * 80)

# =============================================================================
# Setup
# =============================================================================
exp = FastExperimentV2(sample_frac=1.0, n_folds=5, random_state=42)

# Load & Features
train_df = exp.load_data(train_path='../../../train.csv', sample=False)
train_df = exp.create_features(train_df)
X, y, groups, feature_cols = exp.prepare_data(train_df)

# Load best params
with open('../../../logs/best_params.json', 'r') as f:
    best_result = json.load(f)

best_params = best_result['params']

print(f"\n총 피처 수: {len(feature_cols)}개")
print(f"피처: {feature_cols}")

# =============================================================================
# Train on full data
# =============================================================================
print("\n" + "=" * 80)
print("Feature Importance 계산 (Full data)")
print("=" * 80)

model_x = cb.CatBoostRegressor(**best_params)
model_y = cb.CatBoostRegressor(**best_params)

print("\n  모델 학습...")
model_x.fit(X, y[:, 0])
model_y.fit(X, y[:, 1])

# Feature importance
feat_imp_x = model_x.get_feature_importance()
feat_imp_y = model_y.get_feature_importance()
feat_imp_avg = (feat_imp_x + feat_imp_y) / 2

# Create DataFrame
feat_df = pd.DataFrame({
    'feature': feature_cols,
    'importance_x': feat_imp_x,
    'importance_y': feat_imp_y,
    'importance_avg': feat_imp_avg
}).sort_values('importance_avg', ascending=False)

print("\n" + "=" * 80)
print("Feature Importance Ranking")
print("=" * 80)

print(f"\n{'Rank':<5} {'Feature':<25} {'Avg Imp':<12} {'X Imp':<12} {'Y Imp':<12}")
print("-" * 70)

for i, row in feat_df.iterrows():
    print(f"{feat_df.index.get_loc(i)+1:<5} {row['feature']:<25} "
          f"{row['importance_avg']:<12.2f} {row['importance_x']:<12.2f} "
          f"{row['importance_y']:<12.2f}")

# =============================================================================
# 분석
# =============================================================================
print("\n" + "=" * 80)
print("분석")
print("=" * 80)

# Cumulative importance
feat_df['cumulative'] = feat_df['importance_avg'].cumsum() / feat_df['importance_avg'].sum()

print(f"\n누적 중요도:")
for idx, row in feat_df.iterrows():
    cum = row['cumulative']
    marker = "✅" if cum <= 0.9 else "❌"
    print(f"  Top {feat_df.index.get_loc(idx)+1:<2}: {cum*100:5.1f}% {marker}")

# Find threshold
n_features_90 = (feat_df['cumulative'] <= 0.9).sum()
n_features_95 = (feat_df['cumulative'] <= 0.95).sum()

print(f"\n중요도 90%: Top {n_features_90}개 피처")
print(f"중요도 95%: Top {n_features_95}개 피처")

# Low importance features
low_imp = feat_df[feat_df['importance_avg'] < 1.0]['feature'].tolist()
print(f"\n낮은 중요도 (< 1.0): {low_imp}")

# =============================================================================
# Feature Selection 실험
# =============================================================================
print("\n" + "=" * 80)
print("Feature Selection 실험")
print("=" * 80)

# Try different feature sets
feature_sets = {
    'All (16)': feature_cols,
    f'Top {n_features_90} (90%)': feat_df.head(n_features_90)['feature'].tolist(),
    f'Top {n_features_95} (95%)': feat_df.head(n_features_95)['feature'].tolist(),
    'Top 12': feat_df.head(12)['feature'].tolist(),
    'Top 10': feat_df.head(10)['feature'].tolist(),
}

from sklearn.model_selection import GroupKFold

results = []

for name, feats in feature_sets.items():
    print(f"\n[{name}] ({len(feats)} features)")

    # Get feature indices
    feat_indices = [feature_cols.index(f) for f in feats]
    X_subset = X[:, feat_indices]

    # Quick CV (3-fold)
    gkf = GroupKFold(n_splits=3)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_subset, y, groups=groups)):
        X_train, X_val = X_subset[train_idx], X_subset[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model_x = cb.CatBoostRegressor(**best_params)
        model_y = cb.CatBoostRegressor(**best_params)

        model_x.fit(X_train, y_train[:, 0])
        model_y.fit(X_train, y_train[:, 1])

        pred_x = np.clip(model_x.predict(X_val), 0, 105)
        pred_y = np.clip(model_y.predict(X_val), 0, 68)

        dist = np.sqrt((pred_x - y_val[:, 0])**2 + (pred_y - y_val[:, 1])**2)
        cv = dist.mean()
        fold_scores.append(cv)

    cv_mean = np.mean(fold_scores)
    cv_std = np.std(fold_scores)

    results.append({
        'name': name,
        'n_features': len(feats),
        'cv_mean': cv_mean,
        'cv_std': cv_std
    })

    print(f"  CV: {cv_mean:.4f} ± {cv_std:.4f}")

# =============================================================================
# 결과 요약
# =============================================================================
print("\n" + "=" * 80)
print("결과 요약")
print("=" * 80)

results_df = pd.DataFrame(results).sort_values('cv_mean')

print(f"\n{'Rank':<5} {'Feature Set':<20} {'N Features':<12} {'CV':<18} {'vs All':<10}")
print("-" * 70)

baseline_cv = results_df[results_df['name'] == 'All (16)']['cv_mean'].values[0]

for i, row in results_df.iterrows():
    diff = row['cv_mean'] - baseline_cv
    marker = "⭐" if i == results_df.index[0] else ""
    print(f"{results_df.index.get_loc(i)+1:<5} {row['name']:<20} {row['n_features']:<12} "
          f"{row['cv_mean']:.4f}±{row['cv_std']:.4f}  {diff:+.4f}    {marker}")

# Best
best_result = results_df.iloc[0]
print(f"\n최적 피처 세트: {best_result['name']}")
print(f"  피처 수: {best_result['n_features']}")
print(f"  CV: {best_result['cv_mean']:.4f}")
print(f"  vs All: {best_result['cv_mean'] - baseline_cv:+.4f}")

if best_result['cv_mean'] < baseline_cv:
    print(f"\n✅ Feature Selection 성공!")
else:
    print(f"\n❌ Feature Selection 효과 없음")

# Save feature importance
feat_df.to_csv('../../../logs/feature_importance.csv', index=False)
print(f"\n✅ Feature Importance 저장: ../../../logs/feature_importance.csv")

print("\n" + "=" * 80)
