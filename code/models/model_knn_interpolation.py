"""
K리그 패스 좌표 예측 - KNN Distance-weighted Interpolation

완전히 새로운 접근법:
- Zone/Direction 대신 k-Nearest Neighbors 사용
- (start_x, start_y, prev_dx, prev_dy) 4D 공간에서 유사 패스 탐색
- 거리 역수 가중 평균으로 부드러운 예측

Grid Search:
- k = [3, 5, 7]
- epsilon = [0.1, 1.0, 10.0]

장점:
- Zone 경계 문제 해결
- Direction 분할 문제 해결
- 부드러운 예측
- min_samples 제약 없음

목표: Fold 1-3 CV < 16.32 (Sweet Spot 진입)

2025-12-09 Phase 1
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(".")

print("=" * 80)
print("K리그 패스 좌표 예측 - KNN Distance-weighted Interpolation")
print("=" * 80)

# =============================================================================
# 1. 데이터 로드
# =============================================================================
print("\n[1] 데이터 로드...")
train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")
sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")

test_episodes = []
for _, row in test_df.iterrows():
    ep_df = pd.read_csv(DATA_DIR / row['path'])
    test_episodes.append(ep_df)
test_all = pd.concat(test_episodes, ignore_index=True)

print(f"  Train episodes: {train_df['game_episode'].nunique():,}")
print(f"  Test episodes: {test_all['game_episode'].nunique():,}")

# =============================================================================
# 2. 피처 준비
# =============================================================================
print("\n[2] 피처 준비...")

def prepare_features(df):
    df = df.copy()
    df['dx'] = df['end_x'] - df['start_x']
    df['dy'] = df['end_y'] - df['start_y']
    df['prev_dx'] = df.groupby('game_episode')['dx'].shift(1).fillna(0)
    df['prev_dy'] = df.groupby('game_episode')['dy'].shift(1).fillna(0)
    return df

train_df = prepare_features(train_df)
test_all = prepare_features(test_all)

train_last = train_df.groupby('game_episode').last().reset_index()
train_last = train_last.dropna(subset=['end_x', 'end_y'])
train_last['delta_x'] = train_last['end_x'] - train_last['start_x']
train_last['delta_y'] = train_last['end_y'] - train_last['start_y']

test_last = test_all.groupby('game_episode').last().reset_index()

print(f"  Train samples: {len(train_last):,}")
print(f"  Test samples: {len(test_last):,}")

# =============================================================================
# 3. KNN Interpolation 모델
# =============================================================================
print("\n[3] KNN Interpolation 모델 정의...")

class KNNInterpolation:
    """k-Nearest Neighbors Distance-weighted Interpolation"""

    def __init__(self, k=5, epsilon=1.0):
        """
        Parameters:
        - k: number of neighbors
        - epsilon: smoothing factor for distance weighting
        """
        self.k = k
        self.epsilon = epsilon
        self.knn = None
        self.train_features = None
        self.train_deltas = None

    def fit(self, df):
        """Train on episode data"""
        # 4D feature space: (start_x, start_y, prev_dx, prev_dy)
        features = df[['start_x', 'start_y', 'prev_dx', 'prev_dy']].values
        deltas = df[['delta_x', 'delta_y']].values

        # Normalize features for better distance calculation
        self.feature_mean = features.mean(axis=0)
        self.feature_std = features.std(axis=0) + 1e-8
        features_norm = (features - self.feature_mean) / self.feature_std

        # Build KNN index
        self.knn = NearestNeighbors(n_neighbors=min(self.k, len(df)),
                                    algorithm='ball_tree',
                                    metric='euclidean')
        self.knn.fit(features_norm)

        self.train_features = features_norm
        self.train_deltas = deltas

        # Global fallback
        self.global_delta = deltas.mean(axis=0)

    def predict(self, row):
        """Predict for a single row"""
        # Extract features
        feature = np.array([[row['start_x'], row['start_y'],
                           row['prev_dx'], row['prev_dy']]])
        feature_norm = (feature - self.feature_mean) / self.feature_std

        # Find k nearest neighbors
        distances, indices = self.knn.kneighbors(feature_norm)
        distances = distances[0]
        indices = indices[0]

        # Distance-weighted average
        if len(distances) > 0 and distances[0] < 100:  # Reasonable distance
            # Inverse distance weighting
            weights = 1.0 / (distances + self.epsilon)
            weights = weights / weights.sum()

            # Weighted average of deltas
            neighbor_deltas = self.train_deltas[indices]
            pred_delta = (weights[:, np.newaxis] * neighbor_deltas).sum(axis=0)
        else:
            # Fallback to global average
            pred_delta = self.global_delta

        # Apply delta to start position
        pred_x = np.clip(row['start_x'] + pred_delta[0], 0, 105)
        pred_y = np.clip(row['start_y'] + pred_delta[1], 0, 68)

        return pred_x, pred_y

# =============================================================================
# 4. Grid Search - 5-Fold CV
# =============================================================================
print("\n[4] Grid Search (k × epsilon)...")

gkf = GroupKFold(n_splits=5)
game_ids = train_last['game_id'].values

# Grid parameters
k_values = [3, 5, 7]
epsilon_values = [0.1, 1.0, 10.0]

best_fold13_cv = float('inf')
best_params = None
all_results = []

print(f"\n총 {len(k_values) * len(epsilon_values)}개 조합 테스트...")
print(f"{'k':>3} {'eps':>6} {'Fold1-3 CV':>12} {'Fold4-5 CV':>12} {'차이':>8}")
print("-" * 50)

for k in k_values:
    for epsilon in epsilon_values:
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(gkf.split(train_last, groups=game_ids)):
            train_fold = train_last.iloc[train_idx]
            val_fold = train_last.iloc[val_idx]

            # Build model
            model = KNNInterpolation(k=k, epsilon=epsilon)
            model.fit(train_fold)

            # Predict
            predictions = val_fold.apply(lambda r: model.predict(r), axis=1)
            pred_x = predictions.apply(lambda x: x[0])
            pred_y = predictions.apply(lambda x: x[1])

            # Evaluate
            dist = np.sqrt((pred_x - val_fold['end_x'])**2 +
                          (pred_y - val_fold['end_y'])**2)
            cv = dist.mean()
            fold_scores.append(cv)

        # Calculate metrics
        fold13_cv = np.mean(fold_scores[:3])
        fold13_std = np.std(fold_scores[:3])
        fold45_cv = np.mean(fold_scores[3:])
        diff = fold45_cv - fold13_cv

        all_results.append({
            'k': k,
            'epsilon': epsilon,
            'fold13_cv': fold13_cv,
            'fold13_std': fold13_std,
            'fold45_cv': fold45_cv,
            'diff': diff,
            'fold_scores': fold_scores
        })

        print(f"{k:3d} {epsilon:6.1f} {fold13_cv:12.4f} {fold45_cv:12.4f} {diff:+8.4f}")

        # Track best
        if fold13_cv < best_fold13_cv:
            best_fold13_cv = fold13_cv
            best_params = {'k': k, 'epsilon': epsilon}

# =============================================================================
# 5. 최적 파라미터로 최종 모델
# =============================================================================
print("\n" + "=" * 80)
print("Grid Search 결과")
print("=" * 80)

print(f"\n최적 파라미터:")
print(f"  k = {best_params['k']}")
print(f"  epsilon = {best_params['epsilon']}")

# Find best result
best_result = [r for r in all_results
               if r['k'] == best_params['k'] and r['epsilon'] == best_params['epsilon']][0]

print(f"\n최적 성능:")
print(f"  Fold 1-3 CV:   {best_result['fold13_cv']:.4f} ± {best_result['fold13_std']:.4f}")
print(f"  Fold 4-5 CV:   {best_result['fold45_cv']:.4f}")
print(f"  차이:          {best_result['diff']:+.4f}")

print(f"\nFold별 상세:")
for i, score in enumerate(best_result['fold_scores']):
    print(f"  Fold {i+1}: {score:.4f}")

# =============================================================================
# 6. Test 예측
# =============================================================================
print("\n[6] Test 예측...")

final_model = KNNInterpolation(k=best_params['k'], epsilon=best_params['epsilon'])
final_model.fit(train_last)

predictions = test_last.apply(lambda r: final_model.predict(r), axis=1)
pred_x = predictions.apply(lambda x: x[0]).values
pred_y = predictions.apply(lambda x: x[1]).values

# =============================================================================
# 7. 제출 파일 생성
# =============================================================================
print("\n[7] 제출 파일 생성...")

submission = pd.DataFrame({
    'game_episode': test_last['game_episode'],
    'end_x': pred_x,
    'end_y': pred_y
})
submission = sample_sub[['game_episode']].merge(submission, on='game_episode', how='left')
submission.to_csv('/mnt/c/LSJ/dacon/dacon/kleague-algorithm/submission_knn_interpolation.csv',
                  index=False)

print("  submission_knn_interpolation.csv 저장 완료")

# =============================================================================
# 8. 최종 요약 및 제출 판단
# =============================================================================
print("\n" + "=" * 80)
print("최종 요약 및 제출 판단")
print("=" * 80)

print(f"\n[모델 구성]")
print(f"  접근법: k-Nearest Neighbors Distance-weighted Interpolation")
print(f"  특징 공간: (start_x, start_y, prev_dx, prev_dy) 4D")
print(f"  k = {best_params['k']}")
print(f"  epsilon = {best_params['epsilon']}")

print(f"\n[성능]")
print(f"  Fold 1-3 CV:   {best_result['fold13_cv']:.4f} ± {best_result['fold13_std']:.4f}")
print(f"  Fold 4-5 CV:   {best_result['fold45_cv']:.4f}")

# CV Sweet Spot 체크
fold13_cv = best_result['fold13_cv']
fold13_std = best_result['fold13_std']

if fold13_cv < 16.27:
    print(f"\n⚠️ 경고: CV < 16.27 (과최적화 위험!)")
    gap_estimate = 0.13
    verdict = "REJECT"
elif fold13_cv <= 16.34:
    print(f"\n✅ CV Sweet Spot 범위 (16.27-16.34)")
    gap_estimate = 0.03 + (fold13_cv - 16.27) * 0.10 / 0.07
    verdict = "ACCEPT"
else:
    print(f"\n⚠️ CV가 Sweet Spot 상한 초과")
    gap_estimate = 0.08
    verdict = "REVIEW"

public_estimate = fold13_cv + gap_estimate

print(f"\n[예상]")
print(f"  예상 Gap:      +{gap_estimate:.3f}")
print(f"  예상 Public:   {public_estimate:.4f}")

print(f"\n[비교]")
print(f"  현재 Best:           16.3639 (safe_fold13)")
print(f"  KNN Interpolation:   {public_estimate:.4f} (예상)")
print(f"  개선:                {16.3639 - public_estimate:+.4f}")

print(f"\n[최종 판정]")
if verdict == "ACCEPT" and fold13_std < 0.02 and fold13_cv < 16.32:
    print(f"  ✅✅✅ 즉시 제출 강력 권장! ✅✅✅")
    print(f"  - CV Sweet Spot 범위")
    print(f"  - Fold 분산 안정 (< 0.02)")
    print(f"  - CV < 16.32 (새로운 영역!)")
    print(f"  - 완전히 새로운 접근법")
elif verdict == "ACCEPT" and fold13_cv <= 16.34:
    print(f"  ✅ 제출 권장")
    print(f"  - CV Sweet Spot 범위")
    print(f"  - 새로운 접근법으로 개선")
elif verdict == "ACCEPT":
    print(f"  ⚠️ 제출 보류 (현재 Best와 유사)")
    print(f"  - CV Sweet Spot 범위이나 개선 미미")
elif verdict == "REVIEW":
    print(f"  ⚠️ 제출 보류")
    print(f"  - CV Sweet Spot 상한 초과")
else:
    print(f"  ❌ 제출 불가")
    print(f"  - 과최적화 위험")

print(f"\n[Grid Search 전체 결과]")
print(f"\n상위 3개 조합:")
sorted_results = sorted(all_results, key=lambda x: x['fold13_cv'])
for i, r in enumerate(sorted_results[:3]):
    print(f"  {i+1}. k={r['k']}, eps={r['epsilon']:.1f}: CV {r['fold13_cv']:.4f}")

print("\n" + "=" * 80)
print("완료!")
print("=" * 80)
