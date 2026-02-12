"""
Phase 1-A Test Prediction - Simple Version

학습된 .pkl 모델로 test 데이터 예측 및 submission 생성
"""

import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import time

# Add utils to path
utils_path = Path(__file__).resolve().parent.parent.parent.parent / 'utils'
sys.path.insert(0, str(utils_path))

from fast_experiment_phase1a import FastExperimentPhase1A

print(f"\n{'='*80}")
print("Phase 1-A Test Prediction")
print(f"{'='*80}")

# 1. 모델 로드
print(f"\n{'='*80}")
print("1. 모델 로드")
print(f"{'='*80}")

model_x_path = Path(__file__).parent / 'model_x_catboost.pkl'
model_y_path = Path(__file__).parent / 'model_y_catboost.pkl'

print(f"  Loading model_x...", end='', flush=True)
with open(model_x_path, 'rb') as f:
    model_x = pickle.load(f)
print(" ✓")

print(f"  Loading model_y...", end='', flush=True)
with open(model_y_path, 'rb') as f:
    model_y = pickle.load(f)
print(" ✓")

# 2. Test 메타데이터 로드
print(f"\n{'='*80}")
print("2. Test 데이터 로드")
print(f"{'='*80}")

data_dir = Path(__file__).resolve().parent.parent.parent.parent.parent / 'data'
test_meta = pd.read_csv(data_dir / 'test.csv')
sample_sub = pd.read_csv(data_dir / 'sample_submission.csv')

print(f"  Test episodes: {len(test_meta):,}개")

# 3. Episode별 데이터 로드
print(f"\n{'='*80}")
print("3. Episode별 데이터 로드")
print(f"{'='*80}")

test_episodes = []
for idx, row in test_meta.iterrows():
    # path는 data/ 디렉토리 기준 상대 경로
    ep_path = data_dir / row['path']
    ep_df = pd.read_csv(ep_path)
    ep_df['game_episode'] = row['game_episode']
    test_episodes.append(ep_df)

    if (idx + 1) % 500 == 0:
        print(f"    {idx + 1}/{len(test_meta)} episodes...")

test_df = pd.concat(test_episodes, ignore_index=True)
print(f"  ✓ 전체 패스: {len(test_df):,}개")

# 4. 피처 생성
print(f"\n{'='*80}")
print("4. 피처 생성 (Phase 1-A)")
print(f"{'='*80}")

exp = FastExperimentPhase1A(sample_frac=1.0, n_folds=1)

# 피처 생성 (FastExperimentPhase1A.create_features 사용)
test_df = exp.create_features(test_df)

# 5. 데이터 준비
print(f"\n{'='*80}")
print("5. 데이터 준비")
print(f"{'='*80}")

X_test, _, _, feature_cols = exp.prepare_data(test_df)
test_last = test_df.groupby('game_episode').last().reset_index()

print(f"  Test episodes (last): {len(X_test):,}개")
print(f"  Features: {len(feature_cols)}개")

# 6. 예측
print(f"\n{'='*80}")
print("6. 예측")
print(f"{'='*80}")

print(f"  예측 중...", end='', flush=True)
pred_x = np.clip(model_x.predict(X_test), 0, 105)
pred_y = np.clip(model_y.predict(X_test), 0, 68)
print(" ✓")

print(f"  pred_x 범위: [{pred_x.min():.2f}, {pred_x.max():.2f}]")
print(f"  pred_y 범위: [{pred_y.min():.2f}, {pred_y.max():.2f}]")

# 7. Submission 생성
print(f"\n{'='*80}")
print("7. Submission 생성")
print(f"{'='*80}")

submission = pd.DataFrame({
    'game_episode': test_last['game_episode'],
    'end_x': pred_x,
    'end_y': pred_y
})

# sample_submission과 merge (순서 맞추기)
submission = sample_sub[['game_episode']].merge(
    submission,
    on='game_episode',
    how='left'
)

# NaN 체크
print(f"  Submission shape: {submission.shape}")
print(f"  NaN count: {submission.isna().sum().sum()}")

if submission.isna().sum().sum() > 0:
    print("  ⚠️ NaN 발견! 확인 필요")
else:
    print("  ✅ NaN 없음")

# 저장
submissions_dir = Path(__file__).resolve().parent.parent.parent.parent.parent / 'submissions'
submissions_dir.mkdir(exist_ok=True)

output_path = submissions_dir / 'submission_phase1a_cv15.45.csv'
submission.to_csv(output_path, index=False)

print(f"\n  ✓ 저장 완료: {output_path}")

# 샘플 확인
print(f"\n  샘플 확인:")
print(submission.head(10).to_string(index=False))

# 8. 요약
print(f"\n{'='*80}")
print("✅ Test 예측 완료!")
print(f"{'='*80}")

print(f"\n📊 최종 정보:")
print(f"  CV: 15.45 ± 0.18")
print(f"  Features: 21개 (기존 16 + 신규 5)")
print(f"  Test episodes: {len(test_last):,}개")
print(f"  Submission: {output_path}")

print(f"\n🎯 예상 결과:")
print(f"  예상 Public: 15.65-15.70 (Gap 0.20-0.25 가정)")
print(f"  vs 현재 Best (15.84): -0.14 ~ -0.19점 개선 예상")

print(f"\n🚀 다음 단계:")
print(f"  1. DACON 제출")
print(f"  2. Public score 확인")
print(f"  3. SUBMISSION_LOG.md 업데이트")
print(f"  4. Gap 분석")

print(f"\n{'='*80}")
