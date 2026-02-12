"""
K리그 패스 좌표 예측 - 6x6 min_samples=22

Week 2 실험: min_samples 미세 조정
- 6x6 Zone + 8-way Direction
- min_samples: 22 (기존 25)
- Fold 1-3 CV 목표: 16.28-16.32

2025-12-08 야간 실험
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(".")

print("=" * 80)
print("K리그 패스 좌표 예측 - 6x6 min_samples=22")
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

# =============================================================================
# 3. Zone 및 Direction 함수
# =============================================================================
print("\n[3] Zone 및 Direction 함수 정의...")

def get_zone_6x6(x, y):
    """6x6 Zone 분류"""
    x_zone = min(5, int(x / (105 / 6)))
    y_zone = min(5, int(y / (68 / 6)))
    return x_zone * 6 + y_zone

def get_direction_8way(prev_dx, prev_dy):
    """8방향 분류"""
    if abs(prev_dx) < 1 and abs(prev_dy) < 1:
        return 'none'

    angle = np.arctan2(prev_dy, prev_dx)
    angle_deg = np.degrees(angle)

    if -22.5 <= angle_deg < 22.5:
        return 'forward'
    elif 22.5 <= angle_deg < 67.5:
        return 'forward_up'
    elif 67.5 <= angle_deg < 112.5:
        return 'up'
    elif 112.5 <= angle_deg < 157.5:
        return 'back_up'
    elif angle_deg >= 157.5 or angle_deg < -157.5:
        return 'backward'
    elif -157.5 <= angle_deg < -112.5:
        return 'back_down'
    elif -112.5 <= angle_deg < -67.5:
        return 'down'
    else:
        return 'forward_down'

# =============================================================================
# 4. 모델 구축 및 예측 함수
# =============================================================================
print("\n[4] 모델 함수 정의...")

def build_model_6x6(df, min_samples):
    """6x6 Zone + 8-way Direction 모델"""
    df = df.copy()
    df['zone'] = df.apply(lambda r: get_zone_6x6(r['start_x'], r['start_y']), axis=1)
    df['direction'] = df.apply(lambda r: get_direction_8way(r['prev_dx'], r['prev_dy']), axis=1)
    df['key'] = df['zone'].astype(str) + '_' + df['direction']

    stats = df.groupby('key').agg({
        'delta_x': 'median',
        'delta_y': 'median',
        'game_episode': 'count'
    }).rename(columns={'game_episode': 'count'})

    zone_fallback = df.groupby('zone').agg({
        'delta_x': 'median',
        'delta_y': 'median'
    }).to_dict()

    global_dx = df['delta_x'].median()
    global_dy = df['delta_y'].median()

    return {
        'stats': stats,
        'zone_fallback': zone_fallback,
        'global': (global_dx, global_dy),
        'min_samples': min_samples
    }

def predict_6x6(row, model):
    """예측"""
    zone = get_zone_6x6(row['start_x'], row['start_y'])
    direction = get_direction_8way(row['prev_dx'], row['prev_dy'])
    key = f"{zone}_{direction}"

    if key in model['stats'].index and model['stats'].loc[key, 'count'] >= model['min_samples']:
        dx = model['stats'].loc[key, 'delta_x']
        dy = model['stats'].loc[key, 'delta_y']
    elif zone in model['zone_fallback']['delta_x']:
        dx = model['zone_fallback']['delta_x'][zone]
        dy = model['zone_fallback']['delta_y'][zone]
    else:
        dx, dy = model['global']

    pred_x = np.clip(row['start_x'] + dx, 0, 105)
    pred_y = np.clip(row['start_y'] + dy, 0, 68)
    return pred_x, pred_y

# =============================================================================
# 5. 5-Fold 교차 검증
# =============================================================================
print("\n[5] 5-Fold 교차 검증...")

gkf = GroupKFold(n_splits=5)
game_ids = train_last['game_id'].values

min_samples = 22

print(f"  min_samples: {min_samples}")

fold_scores = []
for fold, (train_idx, val_idx) in enumerate(gkf.split(train_last, groups=game_ids)):
    train_fold = train_last.iloc[train_idx]
    val_fold = train_last.iloc[val_idx]

    model = build_model_6x6(train_fold, min_samples)
    predictions = val_fold.apply(lambda r: predict_6x6(r, model), axis=1)
    pred_x = predictions.apply(lambda x: x[0])
    pred_y = predictions.apply(lambda x: x[1])

    dist = np.sqrt((pred_x - val_fold['end_x'])**2 + (pred_y - val_fold['end_y'])**2)
    cv = dist.mean()
    fold_scores.append(cv)

    print(f"  Fold {fold+1}: {cv:.4f}")

fold13_cv = np.mean(fold_scores[:3])
fold13_std = np.std(fold_scores[:3])
fold45_cv = np.mean(fold_scores[3:])

print(f"\n성능 요약:")
print(f"  Fold 1-3 CV:   {fold13_cv:.4f} ± {fold13_std:.4f}")
print(f"  Fold 4-5 CV:   {fold45_cv:.4f}")
print(f"  차이:          {fold45_cv - fold13_cv:+.4f}")

# =============================================================================
# 6. Test 예측
# =============================================================================
print("\n[6] Test 예측...")

final_model = build_model_6x6(train_last, min_samples)
predictions = test_last.apply(lambda r: predict_6x6(r, final_model), axis=1)
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
submission.to_csv('/mnt/c/LSJ/dacon/dacon/kleague-algorithm/submission_6x6_min22.csv', index=False)

print("  submission_6x6_min22.csv 저장 완료")

# =============================================================================
# 8. 최종 요약
# =============================================================================
print("\n" + "=" * 80)
print("최종 요약")
print("=" * 80)

print(f"\n[성능]")
print(f"  Fold 1-3 CV:   {fold13_cv:.4f} ± {fold13_std:.4f}")

if fold13_cv < 16.27:
    verdict = "REJECT"
    gap_estimate = 0.13
elif fold13_cv <= 16.34:
    verdict = "ACCEPT"
    gap_estimate = 0.03 + (fold13_cv - 16.27) * 0.10 / 0.07
else:
    verdict = "REVIEW"
    gap_estimate = 0.08

public_estimate = fold13_cv + gap_estimate

print(f"\n[예상]")
print(f"  예상 Public:   {public_estimate:.4f}")
print(f"  판정:          {verdict}")

print("\n" + "=" * 80)
print("완료!")
print("=" * 80)
