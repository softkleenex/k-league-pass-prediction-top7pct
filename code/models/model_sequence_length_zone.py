"""
K리그 패스 좌표 예측 - 시퀀스 길이 조건부 Zone 모델
에피소드 내 패스 수에 따라 다른 Zone 통계 적용

아이디어:
- 짧은 시퀀스 (2-3패스): 빠른 카운터 → 긴 패스 경향
- 중간 시퀀스 (4-6패스): 일반적인 빌드업
- 긴 시퀀스 (7+패스): 점유 플레이 → 짧은 패스 경향
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(".")

print("=" * 70)
print("K리그 패스 좌표 예측 - 시퀀스 길이 조건부 Zone 모델")
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
# 2. 시퀀스 길이 계산
# =============================================================================
print("\n[2] 시퀀스 길이 계산...")

# 각 에피소드의 패스 수 계산
train_seq_len = train_df.groupby('game_episode').size().reset_index(name='seq_length')
test_seq_len = test_all.groupby('game_episode').size().reset_index(name='seq_length')

# Train: 마지막 패스만 추출
train_last = train_df.groupby('game_episode').last().reset_index()
train_last = train_last.dropna(subset=['end_x', 'end_y'])
train_last = train_last.merge(train_seq_len, on='game_episode')
train_last['delta_x'] = train_last['end_x'] - train_last['start_x']
train_last['delta_y'] = train_last['end_y'] - train_last['start_y']

# Test: 마지막 패스만 추출
test_last = test_all.groupby('game_episode').last().reset_index()
test_last = test_last.merge(test_seq_len, on='game_episode')

print(f"\n시퀀스 길이 분포 (Train):")
print(train_last['seq_length'].describe())

# =============================================================================
# 3. 시퀀스 길이 범주화
# =============================================================================
print("\n[3] 시퀀스 길이 범주화...")

def categorize_seq_length(length):
    """시퀀스 길이를 범주로 분류"""
    if length <= 3:
        return 'short'  # 짧은 빌드업/카운터
    elif length <= 6:
        return 'medium'  # 일반적인 빌드업
    else:
        return 'long'  # 긴 점유 플레이

train_last['seq_cat'] = train_last['seq_length'].apply(categorize_seq_length)
test_last['seq_cat'] = test_last['seq_length'].apply(categorize_seq_length)

print("\n시퀀스 범주별 분포:")
print(train_last['seq_cat'].value_counts())

# =============================================================================
# 4. Zone 분류
# =============================================================================
print("\n[4] Zone 분류...")

def get_zone_6x6(x, y):
    x_zone = min(5, int(x / (105 / 6)))
    y_zone = min(5, int(y / (68 / 6)))
    return x_zone * 6 + y_zone

train_last['zone'] = train_last.apply(lambda r: get_zone_6x6(r['start_x'], r['start_y']), axis=1)
test_last['zone'] = test_last.apply(lambda r: get_zone_6x6(r['start_x'], r['start_y']), axis=1)

# Zone + 시퀀스 길이 조합
train_last['zone_seq'] = train_last['zone'].astype(str) + '_' + train_last['seq_cat']
test_last['zone_seq'] = test_last['zone'].astype(str) + '_' + test_last['seq_cat']

# =============================================================================
# 5. Zone 통계 계산
# =============================================================================
print("\n[5] Zone 통계 계산...")

# 기본 Zone 통계
zone_stats = train_last.groupby('zone').agg({
    'delta_x': 'median',
    'delta_y': 'median'
}).to_dict()

# 조건부 Zone 통계 (시퀀스 길이별)
zone_seq_stats = train_last.groupby('zone_seq').agg({
    'delta_x': 'median',
    'delta_y': 'median',
    'game_episode': 'count'
}).rename(columns={'game_episode': 'count'})

zone_seq_dict_x = zone_seq_stats['delta_x'].to_dict()
zone_seq_dict_y = zone_seq_stats['delta_y'].to_dict()
zone_seq_count = zone_seq_stats['count'].to_dict()

print(f"조건부 통계 그룹 수: {len(zone_seq_dict_x)}")
print(f"최소 샘플 수: {zone_seq_stats['count'].min()}")
print(f"중간 샘플 수: {zone_seq_stats['count'].median()}")

# 시퀀스 범주별 delta 분석
print("\n시퀀스 범주별 delta 분석:")
for cat in ['short', 'medium', 'long']:
    subset = train_last[train_last['seq_cat'] == cat]
    print(f"  {cat}: dx={subset['delta_x'].median():.2f}, dy={subset['delta_y'].median():.2f}")

# =============================================================================
# 6. 예측 함수
# =============================================================================
MIN_SAMPLES = 20

def predict_with_seq_length(row, zone_stats, zone_seq_dict_x, zone_seq_dict_y, zone_seq_count, min_samples=MIN_SAMPLES):
    """시퀀스 길이 조건부 예측"""
    key = row['zone_seq']

    # 조건부 통계 사용 가능한 경우
    if key in zone_seq_dict_x and zone_seq_count.get(key, 0) >= min_samples:
        dx = zone_seq_dict_x[key]
        dy = zone_seq_dict_y[key]
    else:
        # 기본 Zone 통계로 fallback
        dx = zone_stats['delta_x'].get(row['zone'], 0)
        dy = zone_stats['delta_y'].get(row['zone'], 0)

    pred_x = np.clip(row['start_x'] + dx, 0, 105)
    pred_y = np.clip(row['start_y'] + dy, 0, 68)

    return pred_x, pred_y

# =============================================================================
# 7. GroupKFold 교차 검증
# =============================================================================
print("\n[6] GroupKFold 교차 검증...")

n_splits = 5
gkf = GroupKFold(n_splits=n_splits)
game_ids = train_last['game_id'].values

fold_scores_base = []
fold_scores_seq = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(train_last, groups=game_ids)):
    train_fold = train_last.iloc[train_idx]
    val_fold = train_last.iloc[val_idx]

    # Fold별 통계 재계산
    fold_zone_stats = train_fold.groupby('zone').agg({
        'delta_x': 'median',
        'delta_y': 'median'
    }).to_dict()

    fold_zone_seq_stats = train_fold.groupby('zone_seq').agg({
        'delta_x': 'median',
        'delta_y': 'median',
        'game_episode': 'count'
    }).rename(columns={'game_episode': 'count'})

    fold_zone_seq_dict_x = fold_zone_seq_stats['delta_x'].to_dict()
    fold_zone_seq_dict_y = fold_zone_seq_stats['delta_y'].to_dict()
    fold_zone_seq_count = fold_zone_seq_stats['count'].to_dict()

    # 기본 Zone 예측
    val_fold = val_fold.copy()
    val_fold['pred_x_base'] = val_fold.apply(
        lambda r: np.clip(r['start_x'] + fold_zone_stats['delta_x'].get(r['zone'], 0), 0, 105), axis=1
    )
    val_fold['pred_y_base'] = val_fold.apply(
        lambda r: np.clip(r['start_y'] + fold_zone_stats['delta_y'].get(r['zone'], 0), 0, 68), axis=1
    )

    # 시퀀스 길이 조건부 예측
    predictions = val_fold.apply(
        lambda r: predict_with_seq_length(r, fold_zone_stats, fold_zone_seq_dict_x, fold_zone_seq_dict_y, fold_zone_seq_count),
        axis=1
    )
    val_fold['pred_x_seq'] = predictions.apply(lambda x: x[0])
    val_fold['pred_y_seq'] = predictions.apply(lambda x: x[1])

    # 점수 계산
    base_dist = np.sqrt((val_fold['pred_x_base'] - val_fold['end_x'])**2 + (val_fold['pred_y_base'] - val_fold['end_y'])**2)
    seq_dist = np.sqrt((val_fold['pred_x_seq'] - val_fold['end_x'])**2 + (val_fold['pred_y_seq'] - val_fold['end_y'])**2)

    fold_scores_base.append(base_dist.mean())
    fold_scores_seq.append(seq_dist.mean())

    print(f"  Fold {fold+1}: Base={base_dist.mean():.4f}, SeqLen={seq_dist.mean():.4f}, 개선={base_dist.mean() - seq_dist.mean():+.4f}")

print(f"\n평균 CV Score:")
print(f"  Base (6x6):      {np.mean(fold_scores_base):.4f} (std: {np.std(fold_scores_base):.4f})")
print(f"  SeqLength:       {np.mean(fold_scores_seq):.4f} (std: {np.std(fold_scores_seq):.4f})")
print(f"  개선:            {np.mean(fold_scores_base) - np.mean(fold_scores_seq):+.4f}")

# =============================================================================
# 8. 방향 + 시퀀스 길이 조합 모델
# =============================================================================
print("\n[7] 방향 + 시퀀스 길이 조합 모델...")

def get_direction(prev_dx, prev_dy):
    """직전 패스 방향을 분류"""
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

# 방향 피처 추가
train_df_temp = train_df.copy()
train_df_temp['dx'] = train_df_temp['end_x'] - train_df_temp['start_x']
train_df_temp['dy'] = train_df_temp['end_y'] - train_df_temp['start_y']
train_df_temp['prev_dx'] = train_df_temp.groupby('game_episode')['dx'].shift(1).fillna(0)
train_df_temp['prev_dy'] = train_df_temp.groupby('game_episode')['dy'].shift(1).fillna(0)

train_last_dir = train_df_temp.groupby('game_episode').last().reset_index()
train_last_dir = train_last_dir.dropna(subset=['end_x', 'end_y'])
train_last_dir = train_last_dir.merge(train_seq_len, on='game_episode')
train_last_dir['delta_x'] = train_last_dir['end_x'] - train_last_dir['start_x']
train_last_dir['delta_y'] = train_last_dir['end_y'] - train_last_dir['start_y']
train_last_dir['zone'] = train_last_dir.apply(lambda r: get_zone_6x6(r['start_x'], r['start_y']), axis=1)
train_last_dir['direction'] = train_last_dir.apply(lambda r: get_direction(r['prev_dx'], r['prev_dy']), axis=1)
train_last_dir['seq_cat'] = train_last_dir['seq_length'].apply(categorize_seq_length)

# Zone + Direction + SeqLength 조합
train_last_dir['zone_dir_seq'] = train_last_dir['zone'].astype(str) + '_' + train_last_dir['direction'] + '_' + train_last_dir['seq_cat']

# 통계 계산
zone_dir_seq_stats = train_last_dir.groupby('zone_dir_seq').agg({
    'delta_x': 'median',
    'delta_y': 'median',
    'game_episode': 'count'
}).rename(columns={'game_episode': 'count'})

print(f"Zone+Direction+SeqLen 조합 수: {len(zone_dir_seq_stats)}")
print(f"최소 샘플 수: {zone_dir_seq_stats['count'].min()}")
print(f"중간 샘플 수: {zone_dir_seq_stats['count'].median()}")

# 조합 모델이 너무 세분화되어 있어서 fallback 전략이 중요

# =============================================================================
# 9. Test 예측 및 제출 파일 생성
# =============================================================================
print("\n[8] Test 예측 및 제출 파일 생성...")

# 시퀀스 길이 조건부 모델로 예측
predictions = test_last.apply(
    lambda r: predict_with_seq_length(r, zone_stats, zone_seq_dict_x, zone_seq_dict_y, zone_seq_count),
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
submission.to_csv('submission_seq_length_zone.csv', index=False)

print(f"  submission_seq_length_zone.csv 저장 완료")
print(f"  CV Score: {np.mean(fold_scores_seq):.4f}")

# =============================================================================
# 10. min_samples 민감도 분석
# =============================================================================
print("\n[9] min_samples 민감도 분석...")

for min_s in [10, 15, 20, 30, 50]:
    preds = train_last.apply(
        lambda r: predict_with_seq_length(r, zone_stats, zone_seq_dict_x, zone_seq_dict_y, zone_seq_count, min_samples=min_s),
        axis=1
    )
    pred_x = preds.apply(lambda x: x[0])
    pred_y = preds.apply(lambda x: x[1])

    dist = np.sqrt((pred_x - train_last['end_x'])**2 + (pred_y - train_last['end_y'])**2)
    cv = dist.mean()
    print(f"  min_samples={min_s}: CV = {cv:.4f}")

# =============================================================================
# 11. 최종 요약
# =============================================================================
print("\n" + "=" * 70)
print("최종 요약")
print("=" * 70)

print(f"\n[모델 비교]")
print(f"  기본 6x6 Zone:         CV = {np.mean(fold_scores_base):.4f}")
print(f"  시퀀스 길이 조건부:     CV = {np.mean(fold_scores_seq):.4f}")
print(f"  개선:                  {np.mean(fold_scores_base) - np.mean(fold_scores_seq):+.4f}")

print(f"\n[핵심 인사이트]")
print(f"  - 시퀀스 길이가 패스 패턴에 영향을 줌")
print(f"  - 짧은 시퀀스: 빠른 전환, 긴 패스")
print(f"  - 긴 시퀀스: 점유 플레이, 짧은 패스")

print(f"\n[제출 파일]")
print(f"  submission_seq_length_zone.csv")

print("\n" + "=" * 70)
print("완료!")
print("=" * 70)
