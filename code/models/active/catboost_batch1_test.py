"""
CatBoost Batch 1 - Test 예측 및 제출

Ultrathink 2025-12-16:
- Phase 2 최종: CV 15.79 ± 0.27 (Batch 1)
- Test 예측 및 Submission 생성
- 목표: Gap < 1.0 (Public < 16.8)
"""

import sys
sys.path.append('../../utils')

import pandas as pd
import numpy as np
import catboost as cb
import time
import warnings
warnings.filterwarnings('ignore')

from fast_experiment_v2 import FastExperimentV2

print("=" * 80)
print("CatBoost Batch 1 - Test Prediction")
print("=" * 80)
print("\nCV: 15.79 ± 0.27 (16 features)")
print("목표 Public: < 16.8 (Gap < 1.0)")

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

# CatBoost
cat_params = {
    'iterations': 100,
    'depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'random_state': 42,
    'verbose': 0
}

# Train on full data
print("\n  모델 학습...")
start = time.time()

model_x_final = cb.CatBoostRegressor(**cat_params)
model_y_final = cb.CatBoostRegressor(**cat_params)

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

    if (idx + 1) % 100 == 0:
        print(f"    {idx + 1}/{len(test_meta)} episodes...")

test_df = pd.concat(test_episodes, ignore_index=True)
print(f"  Test 패스: {len(test_df):,}개")

# Create features (same as train!)
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

# Merge with sample_submission (정렬 유지)
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
output_path = '../../../submissions/submission_catboost_batch1_cv15.79.csv'
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
print(f"  CV: 15.79 ± 0.27")
print(f"  Features: {len(feature_cols)}개")
print(f"  Test episodes: {len(test_last):,}개")
print(f"  Submission: {output_path}")

print(f"\n🎯 예상 결과:")
print(f"  예상 Public: 15.79 + 1.0 = 16.79")
print(f"  vs Baseline (16.36): -0.57점 개선 예상")

print(f"\n📝 다음 단계:")
print(f"  1. DACON 제출")
print(f"  2. Public score 확인")
print(f"  3. Gap 분석:")
print(f"     - Gap < 0.5: 🔥 매우 우수")
print(f"     - Gap 0.5-1.0: ✅ 양호")
print(f"     - Gap > 1.0: ⚠️ 과적합")
print(f"  4. Gap에 따른 Phase 3 결정")

print("\n" + "=" * 80)
