"""
CatBoost Tuned - Test 예측 및 제출

Ultrathink 2025-12-16:
- Batch 1 + Tuning: CV 15.60 ± 0.27
- 누적 개선: △0.44점 (Zone 6x6 대비)
- 예상 Public: 15.95 (Gap 0.35 가정)

🎯 제출 준비!
"""

import sys
sys.path.append('../../utils')

import pandas as pd
import numpy as np
import catboost as cb
import time
import json
import warnings
warnings.filterwarnings('ignore')

from fast_experiment_v2 import FastExperimentV2
from pathlib import Path

print("=" * 80)
print("CatBoost Tuned - Test Prediction")
print("=" * 80)
print("\nCV: 15.60 ± 0.27")
print("예상 Public: 15.95 (Gap 0.35 가정)")
print("vs 현재 Best (16.14): -0.19점 예상")

# =============================================================================
# Load Best Params
# =============================================================================
best_params_file = Path('../../../logs/best_params.json')
with open(best_params_file, 'r') as f:
    best_result = json.load(f)

best_params = best_result['params']
cv_mean = best_result['cv_mean']

print(f"\n최적 파라미터:")
for k, v in best_params.items():
    print(f"  {k}: {v}")

# =============================================================================
# Full Train
# =============================================================================
print("\n" + "=" * 80)
print("1. Full Train 재학습")
print("=" * 80)

exp = FastExperimentV2(sample_frac=1.0, n_folds=5, random_state=42)

# Load & Features
train_df = exp.load_data(train_path='../../../train.csv', sample=False)
train_df = exp.create_features(train_df)
X, y, groups, feature_cols = exp.prepare_data(train_df)

print(f"\n  Train Episodes: {len(X):,}개")
print(f"  Features: {len(feature_cols)}개")

# Train on full data
print("\n  모델 학습...")
start = time.time()

model_x_final = cb.CatBoostRegressor(**best_params)
model_y_final = cb.CatBoostRegressor(**best_params)

model_x_final.fit(X, y[:, 0])
model_y_final.fit(X, y[:, 1])

train_time = time.time() - start
print(f"  학습 완료: {train_time:.1f}s")

# =============================================================================
# Test 데이터 로드 및 예측
# =============================================================================
print("\n" + "=" * 80)
print("2. Test 데이터 로드")
print("=" * 80)

# Test metadata
test_meta = pd.read_csv('../../../test.csv')
sample_sub = pd.read_csv('../../../sample_submission.csv')

print(f"  Test episodes: {len(test_meta):,}개")

# Load all test episodes
print("\n  Test episodes 로드...")
test_episodes = []
for idx, row in test_meta.iterrows():
    ep_df = pd.read_csv('../../../' + row['path'])
    ep_df['game_episode'] = row['game_episode']
    test_episodes.append(ep_df)

    if (idx + 1) % 500 == 0:
        print(f"    {idx + 1}/{len(test_meta)} episodes...")

test_df = pd.concat(test_episodes, ignore_index=True)
print(f"  Test 패스: {len(test_df):,}개")

# Create features
print("\n  피처 생성...")
test_df = exp.create_features(test_df)

# Last pass per episode
test_last = test_df.groupby('game_episode').last().reset_index()
X_test = test_last[feature_cols].values

print(f"  Test Episodes (last): {len(X_test):,}개")

# =============================================================================
# Predict
# =============================================================================
print("\n" + "=" * 80)
print("3. Test 예측")
print("=" * 80)

pred_x = np.clip(model_x_final.predict(X_test), 0, 105)
pred_y = np.clip(model_y_final.predict(X_test), 0, 68)

print(f"  예측 완료!")
print(f"  pred_x 범위: {pred_x.min():.2f} ~ {pred_x.max():.2f}")
print(f"  pred_y 범위: {pred_y.min():.2f} ~ {pred_y.max():.2f}")

# =============================================================================
# Submission
# =============================================================================
print("\n" + "=" * 80)
print("4. Submission 생성")
print("=" * 80)

submission = pd.DataFrame({
    'game_episode': test_last['game_episode'],
    'end_x': pred_x,
    'end_y': pred_y
})

# Merge with sample_submission
submission = sample_sub[['game_episode']].merge(
    submission,
    on='game_episode',
    how='left'
)

# Check
print(f"\n  Submission shape: {submission.shape}")
print(f"  NaN count: {submission.isna().sum().sum()}")

if submission.isna().sum().sum() > 0:
    print("  ⚠️ NaN 발견! 확인 필요")
else:
    print("  ✅ NaN 없음")

# Save
output_path = '../../../submissions/submission_catboost_tuned_cv15.60.csv'
submission.to_csv(output_path, index=False)

print(f"\n  저장 완료: {output_path}")

# Sample check
print("\n  샘플 확인:")
print(submission.head(5).to_string(index=False))

# =============================================================================
# 요약
# =============================================================================
print("\n" + "=" * 80)
print("✅ Test 예측 완료!")
print("=" * 80)

print(f"\n📊 최종 정보:")
print(f"  CV: 15.60 ± 0.27")
print(f"  Features: {len(feature_cols)}개")
print(f"  Test episodes: {len(test_last):,}개")
print(f"  Submission: {output_path}")

print(f"\n🎯 예상 결과:")
print(f"  예상 Public: 15.95 (Gap 0.35 가정)")
print(f"  vs 현재 Best (16.14): -0.19점")
print(f"  예상 순위: 150-180등 (상위 15-18%)")

print(f"\n📝 개선 요약:")
print(f"  Zone 6x6:        16.04")
print(f"  → Batch 1:       15.79 (△0.25)")
print(f"  → Tuning:        15.60 (△0.19)")
print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"  총 누적 개선:    0.44점 ✅")

print(f"\n🚀 다음 단계:")
print(f"  1. DACON 제출")
print(f"  2. Public score 확인")
print(f"  3. Gap 분석 (예상 0.35)")
print(f"  4. 결과에 따라:")
print(f"     - Public < 16.0: 🎉 성공!")
print(f"     - Public 16.0-16.1: ✅ 양호")
print(f"     - Public > 16.1: ⚠️ Gap 증가 (Ensemble 고려)")

print("\n" + "=" * 80)
