"""
K리그 패스 좌표 예측 - 방향 조건부 Zone + 앙상블 모델
가장 좋은 방향 조건부 모델(CV 16.35)을 기반으로 다양한 Zone 크기와 앙상블

전략:
1. 5x5, 6x6, 7x7 Zone에 각각 방향 조건부 적용
2. 가중치 앙상블로 최적 조합 탐색
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(".")

print("=" * 70)
print("K리그 패스 좌표 예측 - 방향 조건부 Zone + 앙상블 모델")
print("=" * 70)

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

print(f"Train episodes: {train_df['game_episode'].nunique():,}")
print(f"Test episodes: {test_all['game_episode'].nunique():,}")

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

# =============================================================================
# 3. Zone 및 방향 분류 함수
# =============================================================================
print("\n[3] Zone 및 방향 분류 함수 정의...")

def get_zone(x, y, n_x, n_y):
    """NxN Zone 분류"""
    x_zone = min(n_x - 1, int(x / (105 / n_x)))
    y_zone = min(n_y - 1, int(y / (68 / n_y)))
    return x_zone * n_y + y_zone

def get_direction(prev_dx, prev_dy):
    """직전 패스 방향을 5가지로 분류"""
    if abs(prev_dx) < 1 and abs(prev_dy) < 1:
        return 'none'

    angle = np.arctan2(prev_dy, prev_dx)

    if angle > np.pi * 2/3:
        return 'back_up'
    elif angle > np.pi / 3:
        return 'up'
    elif angle > -np.pi / 3:
        if prev_dx > 0:
            return 'forward'
        else:
            return 'backward'
    elif angle > -np.pi * 2/3:
        return 'down'
    else:
        return 'back_down'

# =============================================================================
# 4. 다양한 Zone 크기로 예측
# =============================================================================
print("\n[4] 다양한 Zone 크기로 방향 조건부 예측...")

MIN_SAMPLES = 20
ZONE_CONFIGS = [(5, 5), (6, 6), (7, 7)]

def build_direction_model(df, n_x, n_y):
    """방향 조건부 Zone 통계 구축"""
    df = df.copy()
    df['zone'] = df.apply(lambda r: get_zone(r['start_x'], r['start_y'], n_x, n_y), axis=1)
    df['direction'] = df.apply(lambda r: get_direction(r['prev_dx'], r['prev_dy']), axis=1)
    df['zone_dir'] = df['zone'].astype(str) + '_' + df['direction']

    # 기본 Zone 통계
    zone_stats = df.groupby('zone').agg({
        'delta_x': 'median',
        'delta_y': 'median'
    }).to_dict()

    # 방향 조건부 통계
    zone_dir_stats = df.groupby('zone_dir').agg({
        'delta_x': 'median',
        'delta_y': 'median',
        'game_episode': 'count'
    }).rename(columns={'game_episode': 'count'})

    return {
        'zone_stats': zone_stats,
        'zone_dir_x': zone_dir_stats['delta_x'].to_dict(),
        'zone_dir_y': zone_dir_stats['delta_y'].to_dict(),
        'zone_dir_count': zone_dir_stats['count'].to_dict(),
        'n_x': n_x,
        'n_y': n_y
    }

def predict_direction_zone(row, model, min_samples=MIN_SAMPLES):
    """방향 조건부 예측"""
    zone = get_zone(row['start_x'], row['start_y'], model['n_x'], model['n_y'])
    direction = get_direction(row['prev_dx'], row['prev_dy'])
    key = f"{zone}_{direction}"

    if key in model['zone_dir_x'] and model['zone_dir_count'].get(key, 0) >= min_samples:
        dx = model['zone_dir_x'][key]
        dy = model['zone_dir_y'][key]
    else:
        dx = model['zone_stats']['delta_x'].get(zone, 0)
        dy = model['zone_stats']['delta_y'].get(zone, 0)

    pred_x = np.clip(row['start_x'] + dx, 0, 105)
    pred_y = np.clip(row['start_y'] + dy, 0, 68)

    return pred_x, pred_y

# =============================================================================
# 5. GroupKFold 교차 검증
# =============================================================================
print("\n[5] GroupKFold 교차 검증...")

n_splits = 5
gkf = GroupKFold(n_splits=n_splits)
game_ids = train_last['game_id'].values

# 각 Zone 크기별 CV 점수 저장
cv_scores = {config: [] for config in ZONE_CONFIGS}

for fold, (train_idx, val_idx) in enumerate(gkf.split(train_last, groups=game_ids)):
    train_fold = train_last.iloc[train_idx]
    val_fold = train_last.iloc[val_idx].copy()

    # 각 Zone 크기별 모델 구축 및 예측
    for n_x, n_y in ZONE_CONFIGS:
        model = build_direction_model(train_fold, n_x, n_y)

        predictions = val_fold.apply(lambda r: predict_direction_zone(r, model), axis=1)
        val_fold[f'pred_x_{n_x}x{n_y}'] = predictions.apply(lambda x: x[0])
        val_fold[f'pred_y_{n_x}x{n_y}'] = predictions.apply(lambda x: x[1])

        dist = np.sqrt(
            (val_fold[f'pred_x_{n_x}x{n_y}'] - val_fold['end_x'])**2 +
            (val_fold[f'pred_y_{n_x}x{n_y}'] - val_fold['end_y'])**2
        )
        cv_scores[(n_x, n_y)].append(dist.mean())

print("\n개별 모델 CV 점수:")
for config in ZONE_CONFIGS:
    print(f"  {config[0]}x{config[1]} 방향조건부: {np.mean(cv_scores[config]):.4f} (std: {np.std(cv_scores[config]):.4f})")

# =============================================================================
# 6. 앙상블 가중치 최적화
# =============================================================================
print("\n[6] 앙상블 가중치 최적화...")

# 전체 데이터로 각 모델 예측 생성
models = {}
for n_x, n_y in ZONE_CONFIGS:
    models[(n_x, n_y)] = build_direction_model(train_last, n_x, n_y)

for n_x, n_y in ZONE_CONFIGS:
    predictions = train_last.apply(lambda r: predict_direction_zone(r, models[(n_x, n_y)]), axis=1)
    train_last[f'pred_x_{n_x}x{n_y}'] = predictions.apply(lambda x: x[0])
    train_last[f'pred_y_{n_x}x{n_y}'] = predictions.apply(lambda x: x[1])

# 다양한 가중치 조합 테스트
print("\n가중치 조합 탐색:")
best_score = float('inf')
best_weights = None

for w1 in np.arange(0.2, 0.6, 0.1):
    for w2 in np.arange(0.2, 0.6, 0.1):
        w3 = 1 - w1 - w2
        if w3 < 0.1 or w3 > 0.6:
            continue

        # 앙상블 예측
        pred_x = (w1 * train_last['pred_x_5x5'] +
                  w2 * train_last['pred_x_6x6'] +
                  w3 * train_last['pred_x_7x7'])
        pred_y = (w1 * train_last['pred_y_5x5'] +
                  w2 * train_last['pred_y_6x6'] +
                  w3 * train_last['pred_y_7x7'])

        dist = np.sqrt((pred_x - train_last['end_x'])**2 + (pred_y - train_last['end_y'])**2)
        score = dist.mean()

        if score < best_score:
            best_score = score
            best_weights = (w1, w2, w3)
            print(f"  w5x5={w1:.1f}, w6x6={w2:.1f}, w7x7={w3:.1f}: CV = {score:.4f} *")

print(f"\n최적 가중치: 5x5={best_weights[0]:.1f}, 6x6={best_weights[1]:.1f}, 7x7={best_weights[2]:.1f}")
print(f"최적 CV Score: {best_score:.4f}")

# =============================================================================
# 7. CV로 앙상블 검증
# =============================================================================
print("\n[7] CV로 앙상블 검증...")

ensemble_scores = []
w1, w2, w3 = best_weights

for fold, (train_idx, val_idx) in enumerate(gkf.split(train_last, groups=game_ids)):
    train_fold = train_last.iloc[train_idx]
    val_fold = train_last.iloc[val_idx].copy()

    # 각 Zone 크기별 모델 구축 및 예측
    fold_models = {}
    for n_x, n_y in ZONE_CONFIGS:
        fold_models[(n_x, n_y)] = build_direction_model(train_fold, n_x, n_y)

    for n_x, n_y in ZONE_CONFIGS:
        predictions = val_fold.apply(lambda r: predict_direction_zone(r, fold_models[(n_x, n_y)]), axis=1)
        val_fold[f'pred_x_{n_x}x{n_y}'] = predictions.apply(lambda x: x[0])
        val_fold[f'pred_y_{n_x}x{n_y}'] = predictions.apply(lambda x: x[1])

    # 앙상블 예측
    pred_x = (w1 * val_fold['pred_x_5x5'] +
              w2 * val_fold['pred_x_6x6'] +
              w3 * val_fold['pred_x_7x7'])
    pred_y = (w1 * val_fold['pred_y_5x5'] +
              w2 * val_fold['pred_y_6x6'] +
              w3 * val_fold['pred_y_7x7'])

    dist = np.sqrt((pred_x - val_fold['end_x'])**2 + (pred_y - val_fold['end_y'])**2)
    ensemble_scores.append(dist.mean())

    print(f"  Fold {fold+1}: {dist.mean():.4f}")

print(f"\n앙상블 평균 CV Score: {np.mean(ensemble_scores):.4f} (std: {np.std(ensemble_scores):.4f})")

# =============================================================================
# 8. Test 예측 및 제출 파일 생성
# =============================================================================
print("\n[8] Test 예측 및 제출 파일 생성...")

# Test 데이터에 대해 각 모델 예측
for n_x, n_y in ZONE_CONFIGS:
    predictions = test_last.apply(lambda r: predict_direction_zone(r, models[(n_x, n_y)]), axis=1)
    test_last[f'pred_x_{n_x}x{n_y}'] = predictions.apply(lambda x: x[0])
    test_last[f'pred_y_{n_x}x{n_y}'] = predictions.apply(lambda x: x[1])

# 앙상블
test_last['pred_x'] = (w1 * test_last['pred_x_5x5'] +
                       w2 * test_last['pred_x_6x6'] +
                       w3 * test_last['pred_x_7x7'])
test_last['pred_y'] = (w1 * test_last['pred_y_5x5'] +
                       w2 * test_last['pred_y_6x6'] +
                       w3 * test_last['pred_y_7x7'])

# 제출 파일 생성
submission = pd.DataFrame({
    'game_episode': test_last['game_episode'],
    'end_x': test_last['pred_x'],
    'end_y': test_last['pred_y']
})
submission = sample_sub[['game_episode']].merge(submission, on='game_episode', how='left')
submission.to_csv('submission_direction_ensemble.csv', index=False)

print(f"  submission_direction_ensemble.csv 저장 완료")
print(f"  CV Score: {np.mean(ensemble_scores):.4f}")

# 개별 모델 제출 파일도 생성
for n_x, n_y in ZONE_CONFIGS:
    sub = pd.DataFrame({
        'game_episode': test_last['game_episode'],
        'end_x': test_last[f'pred_x_{n_x}x{n_y}'],
        'end_y': test_last[f'pred_y_{n_x}x{n_y}']
    })
    sub = sample_sub[['game_episode']].merge(sub, on='game_episode', how='left')
    sub.to_csv(f'submission_direction_{n_x}x{n_y}.csv', index=False)
    print(f"  submission_direction_{n_x}x{n_y}.csv 저장 완료 (CV: {np.mean(cv_scores[(n_x, n_y)]):.4f})")

# =============================================================================
# 9. 최종 요약
# =============================================================================
print("\n" + "=" * 70)
print("최종 요약")
print("=" * 70)

print(f"\n[개별 모델 CV 점수]")
for config in ZONE_CONFIGS:
    print(f"  {config[0]}x{config[1]} 방향조건부: {np.mean(cv_scores[config]):.4f}")

print(f"\n[앙상블 모델]")
print(f"  가중치: 5x5={best_weights[0]:.1f}, 6x6={best_weights[1]:.1f}, 7x7={best_weights[2]:.1f}")
print(f"  CV Score: {np.mean(ensemble_scores):.4f}")

print(f"\n[제출 파일]")
print(f"  1. submission_direction_ensemble.csv (앙상블)")
print(f"  2. submission_direction_5x5.csv")
print(f"  3. submission_direction_6x6.csv")
print(f"  4. submission_direction_7x7.csv")

print("\n" + "=" * 70)
print("완료!")
print("=" * 70)
