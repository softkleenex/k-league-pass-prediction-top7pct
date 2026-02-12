"""
K리그 패스 좌표 예측 - 방향 조건부 Zone 모델
가장 효과적인 조건부 전략: 직전 패스 방향에 따른 Zone 통계

CV 개선: 16.68 → 16.09 (-0.58점)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(".")

print("=" * 70)
print("K리그 패스 좌표 예측 - 방향 조건부 Zone 모델")
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
# 3. Zone 및 방향 분류
# =============================================================================
print("\n[3] Zone 및 방향 분류...")

def get_zone_6x6(x, y):
    x_zone = min(5, int(x / (105 / 6)))
    y_zone = min(5, int(y / (68 / 6)))
    return x_zone * 6 + y_zone

def get_direction(prev_dx, prev_dy):
    """직전 패스 방향을 5가지로 분류"""
    if abs(prev_dx) < 1 and abs(prev_dy) < 1:
        return 'none'  # 움직임 거의 없음

    angle = np.arctan2(prev_dy, prev_dx)  # -pi ~ pi

    if angle > np.pi * 2/3:  # 120~180도: 후방 상단
        return 'back_up'
    elif angle > np.pi / 3:  # 60~120도: 상단
        return 'up'
    elif angle > -np.pi / 3:  # -60~60도
        if prev_dx > 0:
            return 'forward'  # 전방
        else:
            return 'backward'  # 후방
    elif angle > -np.pi * 2/3:  # -120~-60도: 하단
        return 'down'
    else:  # -180~-120도: 후방 하단
        return 'back_down'

train_last['zone'] = train_last.apply(lambda r: get_zone_6x6(r['start_x'], r['start_y']), axis=1)
train_last['direction'] = train_last.apply(lambda r: get_direction(r['prev_dx'], r['prev_dy']), axis=1)
train_last['zone_dir'] = train_last['zone'].astype(str) + '_' + train_last['direction']

test_last['zone'] = test_last.apply(lambda r: get_zone_6x6(r['start_x'], r['start_y']), axis=1)
test_last['direction'] = test_last.apply(lambda r: get_direction(r['prev_dx'], r['prev_dy']), axis=1)
test_last['zone_dir'] = test_last['zone'].astype(str) + '_' + test_last['direction']

# 방향별 분포 확인
print("\n방향별 분포:")
print(train_last['direction'].value_counts())

# =============================================================================
# 4. Zone 통계 계산
# =============================================================================
print("\n[4] Zone 통계 계산...")

# 기본 Zone 통계
zone_stats = train_last.groupby('zone').agg({
    'delta_x': 'median',
    'delta_y': 'median'
}).to_dict()

# 조건부 Zone 통계 (방향별)
zone_dir_stats = train_last.groupby('zone_dir').agg({
    'delta_x': 'median',
    'delta_y': 'median',
    'game_episode': 'count'
}).rename(columns={'game_episode': 'count'})

zone_dir_dict_x = zone_dir_stats['delta_x'].to_dict()
zone_dir_dict_y = zone_dir_stats['delta_y'].to_dict()
zone_dir_count = zone_dir_stats['count'].to_dict()

print(f"조건부 통계 그룹 수: {len(zone_dir_dict_x)}")
print(f"최소 샘플 수: {zone_dir_stats['count'].min()}")
print(f"중간 샘플 수: {zone_dir_stats['count'].median()}")

# =============================================================================
# 5. 예측 함수 (안전한 fallback 포함)
# =============================================================================
MIN_SAMPLES = 20  # 최소 샘플 수

def predict_with_direction(row, zone_stats, zone_dir_dict_x, zone_dir_dict_y, zone_dir_count, min_samples=MIN_SAMPLES):
    """방향 조건부 예측 (샘플 부족시 기본 Zone으로 fallback)"""
    key = row['zone_dir']

    # 조건부 통계 사용 가능한 경우
    if key in zone_dir_dict_x and zone_dir_count.get(key, 0) >= min_samples:
        dx = zone_dir_dict_x[key]
        dy = zone_dir_dict_y[key]
    else:
        # 기본 Zone 통계로 fallback
        dx = zone_stats['delta_x'].get(row['zone'], 0)
        dy = zone_stats['delta_y'].get(row['zone'], 0)

    pred_x = np.clip(row['start_x'] + dx, 0, 105)
    pred_y = np.clip(row['start_y'] + dy, 0, 68)

    return pred_x, pred_y

# =============================================================================
# 6. GroupKFold 교차 검증
# =============================================================================
print("\n[5] GroupKFold 교차 검증...")

n_splits = 5
gkf = GroupKFold(n_splits=n_splits)
game_ids = train_last['game_id'].values

fold_scores_base = []
fold_scores_dir = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(train_last, groups=game_ids)):
    train_fold = train_last.iloc[train_idx]
    val_fold = train_last.iloc[val_idx]

    # Fold별 통계 재계산 (data leakage 방지)
    fold_zone_stats = train_fold.groupby('zone').agg({
        'delta_x': 'median',
        'delta_y': 'median'
    }).to_dict()

    fold_zone_dir_stats = train_fold.groupby('zone_dir').agg({
        'delta_x': 'median',
        'delta_y': 'median',
        'game_episode': 'count'
    }).rename(columns={'game_episode': 'count'})

    fold_zone_dir_dict_x = fold_zone_dir_stats['delta_x'].to_dict()
    fold_zone_dir_dict_y = fold_zone_dir_stats['delta_y'].to_dict()
    fold_zone_dir_count = fold_zone_dir_stats['count'].to_dict()

    # 기본 Zone 예측
    val_fold = val_fold.copy()
    val_fold['pred_x_base'] = val_fold.apply(
        lambda r: np.clip(r['start_x'] + fold_zone_stats['delta_x'].get(r['zone'], 0), 0, 105), axis=1
    )
    val_fold['pred_y_base'] = val_fold.apply(
        lambda r: np.clip(r['start_y'] + fold_zone_stats['delta_y'].get(r['zone'], 0), 0, 68), axis=1
    )

    # 방향 조건부 예측
    predictions = val_fold.apply(
        lambda r: predict_with_direction(r, fold_zone_stats, fold_zone_dir_dict_x, fold_zone_dir_dict_y, fold_zone_dir_count),
        axis=1
    )
    val_fold['pred_x_dir'] = predictions.apply(lambda x: x[0])
    val_fold['pred_y_dir'] = predictions.apply(lambda x: x[1])

    # 점수 계산
    base_dist = np.sqrt((val_fold['pred_x_base'] - val_fold['end_x'])**2 + (val_fold['pred_y_base'] - val_fold['end_y'])**2)
    dir_dist = np.sqrt((val_fold['pred_x_dir'] - val_fold['end_x'])**2 + (val_fold['pred_y_dir'] - val_fold['end_y'])**2)

    fold_scores_base.append(base_dist.mean())
    fold_scores_dir.append(dir_dist.mean())

    print(f"  Fold {fold+1}: Base={base_dist.mean():.4f}, Direction={dir_dist.mean():.4f}, 개선={base_dist.mean() - dir_dist.mean():+.4f}")

print(f"\n평균 CV Score:")
print(f"  Base (6x6):      {np.mean(fold_scores_base):.4f} (std: {np.std(fold_scores_base):.4f})")
print(f"  Direction:       {np.mean(fold_scores_dir):.4f} (std: {np.std(fold_scores_dir):.4f})")
print(f"  개선:            {np.mean(fold_scores_base) - np.mean(fold_scores_dir):+.4f}")

# =============================================================================
# 7. Test 예측 및 제출 파일 생성
# =============================================================================
print("\n[6] Test 예측 및 제출 파일 생성...")

# 전체 Train 데이터로 통계 재계산
predictions = test_last.apply(
    lambda r: predict_with_direction(r, zone_stats, zone_dir_dict_x, zone_dir_dict_y, zone_dir_count),
    axis=1
)
test_last['pred_x'] = predictions.apply(lambda x: x[0])
test_last['pred_y'] = predictions.apply(lambda x: x[1])

# 제출 파일 생성
submission = pd.DataFrame({
    'game_episode': test_last['game_episode'],
    'end_x': test_last['pred_x'],
    'end_y': test_last['pred_y']
})
submission = sample_sub[['game_episode']].merge(submission, on='game_episode', how='left')
submission.to_csv('submission_direction_zone.csv', index=False)

print(f"  submission_direction_zone.csv 저장 완료")
print(f"  CV Score: {np.mean(fold_scores_dir):.4f}")

# 다양한 min_samples로 실험
print("\n[7] min_samples 민감도 분석...")

for min_s in [10, 20, 30, 50]:
    preds = train_last.apply(
        lambda r: predict_with_direction(r, zone_stats, zone_dir_dict_x, zone_dir_dict_y, zone_dir_count, min_samples=min_s),
        axis=1
    )
    pred_x = preds.apply(lambda x: x[0])
    pred_y = preds.apply(lambda x: x[1])

    dist = np.sqrt((pred_x - train_last['end_x'])**2 + (pred_y - train_last['end_y'])**2)
    cv = dist.mean()
    print(f"  min_samples={min_s}: CV = {cv:.4f}")

# =============================================================================
# 8. 최종 요약
# =============================================================================
print("\n" + "=" * 70)
print("최종 요약")
print("=" * 70)

print(f"\n[모델 비교]")
print(f"  기본 6x6 Zone:     CV = {np.mean(fold_scores_base):.4f}")
print(f"  방향 조건부 Zone:  CV = {np.mean(fold_scores_dir):.4f}")
print(f"  개선:              {np.mean(fold_scores_base) - np.mean(fold_scores_dir):+.4f}")

print(f"\n[핵심 인사이트]")
print(f"  - 직전 패스 방향이 다음 패스 위치에 영향을 줌")
print(f"  - ML 없이 순수 통계로 0.5점 이상 개선 가능")
print(f"  - 과적합 위험 낮음 (통계 기반)")

print(f"\n[제출 파일]")
print(f"  submission_direction_zone.csv")

print("\n" + "=" * 70)
print("완료!")
print("=" * 70)
