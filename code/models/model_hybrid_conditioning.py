"""
K리그 패스 좌표 예측 - 하이브리드 조건부 Zone 모델

전략:
- 8방향 + 시퀀스 길이 조건부 통계
- 두 가지 강력한 시그널 결합
- Hierarchical fallback으로 안정성 확보

조건 계층:
1. zone_direction_seqlength (최우선)
2. zone_direction (fallback 1)
3. zone (fallback 2)

예상 개선:
- CV: 16.35 → 15.7-16.0 (0.35-0.65점)
- Public: 16.36 → 16.0-16.3 (0.1-0.35점)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(".")

print("=" * 70)
print("K리그 패스 좌표 예측 - 하이브리드 조건부 모델")
print("(8방향 + 시퀀스 길이)")
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
# 2. 피처 준비 (시퀀스 길이 포함)
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

# 시퀀스 길이 계산
train_seq_lengths = train_df.groupby('game_episode').size()
test_seq_lengths = test_all.groupby('game_episode').size()

train_last = train_df.groupby('game_episode').last().reset_index()
train_last = train_last.dropna(subset=['end_x', 'end_y'])
train_last['delta_x'] = train_last['end_x'] - train_last['start_x']
train_last['delta_y'] = train_last['end_y'] - train_last['start_y']
train_last['seq_length'] = train_last['game_episode'].map(train_seq_lengths)

test_last = test_all.groupby('game_episode').last().reset_index()
test_last['seq_length'] = test_last['game_episode'].map(test_seq_lengths)

# =============================================================================
# 3. 조건 분류 함수들
# =============================================================================
print("\n[3] 조건 분류 함수 정의...")

def get_zone_6x6(x, y):
    """6x6 Zone 분류"""
    x_zone = min(5, int(x / (105 / 6)))
    y_zone = min(5, int(y / (68 / 6)))
    return x_zone * 6 + y_zone

def get_direction_8(prev_dx, prev_dy):
    """8방향 분류"""
    if abs(prev_dx) < 1 and abs(prev_dy) < 1:
        return 'none'

    angle = np.arctan2(prev_dy, prev_dx)
    angle_deg = np.degrees(angle)

    if -22.5 <= angle_deg < 22.5:
        return 'E'
    elif 22.5 <= angle_deg < 67.5:
        return 'NE'
    elif 67.5 <= angle_deg < 112.5:
        return 'N'
    elif 112.5 <= angle_deg < 157.5:
        return 'NW'
    elif angle_deg >= 157.5 or angle_deg < -157.5:
        return 'W'
    elif -157.5 <= angle_deg < -112.5:
        return 'SW'
    elif -112.5 <= angle_deg < -67.5:
        return 'S'
    else:
        return 'SE'

def get_sequence_bin(seq_length):
    """
    시퀀스 길이를 3개 범주로 분류

    분석 결과:
    - Short (1-5): dx=7.30, distance=12.94 (짧은 마지막 패스)
    - Medium (6-20): dx=10.82-11.20, distance=16.25-16.46
    - Long (21+): dx=10.41-10.85, distance=16.05-16.37
    """
    if seq_length <= 5:
        return 'short'  # 짧은 시퀀스: 다른 패스 특성
    elif seq_length <= 20:
        return 'medium'
    else:
        return 'long'

# =============================================================================
# 4. 시퀀스 길이별 통계 분석
# =============================================================================
print("\n[4] 시퀀스 길이별 통계 분석...")

train_last['seq_bin'] = train_last['seq_length'].apply(get_sequence_bin)

print("\n시퀀스 길이별 패스 특성:")
print(f"{'범주':8s} {'샘플수':>7s} {'비율':>6s} {'dx_median':>10s} {'dy_median':>10s} {'dist_median':>12s}")
print("-" * 60)
for seq_bin in ['short', 'medium', 'long']:
    subset = train_last[train_last['seq_bin'] == seq_bin]
    if len(subset) > 0:
        pct = 100 * len(subset) / len(train_last)
        dx_med = subset['delta_x'].median()
        dy_med = subset['delta_y'].median()
        dist_med = np.sqrt(subset['delta_x']**2 + subset['delta_y']**2).median()
        print(f"{seq_bin:8s} {len(subset):7d} {pct:5.1f}% {dx_med:10.2f} {dy_med:10.2f} {dist_med:12.2f}")

# =============================================================================
# 5. 하이브리드 조건부 모델 구축
# =============================================================================
print("\n[5] 하이브리드 조건부 모델 구축...")

MIN_SAMPLES_TRIPLE = 10   # zone_dir_seq 최소 샘플
MIN_SAMPLES_DOUBLE = 20   # zone_dir 최소 샘플

def build_hybrid_model(df):
    """8방향 + 시퀀스 길이 조건부 모델"""
    df = df.copy()
    df['zone'] = df.apply(lambda r: get_zone_6x6(r['start_x'], r['start_y']), axis=1)
    df['direction'] = df.apply(lambda r: get_direction_8(r['prev_dx'], r['prev_dy']), axis=1)
    df['seq_bin'] = df['seq_length'].apply(get_sequence_bin)

    # 3가지 조건 조합
    df['zone_dir_seq'] = (df['zone'].astype(str) + '_' +
                          df['direction'] + '_' +
                          df['seq_bin'])
    df['zone_dir'] = df['zone'].astype(str) + '_' + df['direction']

    # 기본 Zone 통계 (fallback level 3)
    zone_stats = df.groupby('zone').agg({
        'delta_x': 'median',
        'delta_y': 'median'
    }).to_dict()

    # Zone + Direction 통계 (fallback level 2)
    zone_dir_stats = df.groupby('zone_dir').agg({
        'delta_x': 'median',
        'delta_y': 'median',
        'game_episode': 'count'
    }).rename(columns={'game_episode': 'count'})

    # Zone + Direction + SeqBin 통계 (primary)
    zone_dir_seq_stats = df.groupby('zone_dir_seq').agg({
        'delta_x': 'median',
        'delta_y': 'median',
        'game_episode': 'count'
    }).rename(columns={'game_episode': 'count'})

    return {
        'zone_stats': zone_stats,
        'zone_dir_x': zone_dir_stats['delta_x'].to_dict(),
        'zone_dir_y': zone_dir_stats['delta_y'].to_dict(),
        'zone_dir_count': zone_dir_stats['count'].to_dict(),
        'zone_dir_seq_x': zone_dir_seq_stats['delta_x'].to_dict(),
        'zone_dir_seq_y': zone_dir_seq_stats['delta_y'].to_dict(),
        'zone_dir_seq_count': zone_dir_seq_stats['count'].to_dict()
    }

def predict_hybrid(row, model, min_triple=MIN_SAMPLES_TRIPLE, min_double=MIN_SAMPLES_DOUBLE):
    """하이브리드 조건부 예측 with 3-level fallback"""
    zone = get_zone_6x6(row['start_x'], row['start_y'])
    direction = get_direction_8(row['prev_dx'], row['prev_dy'])
    seq_bin = get_sequence_bin(row['seq_length'])

    key_triple = f"{zone}_{direction}_{seq_bin}"
    key_double = f"{zone}_{direction}"

    # Level 1: zone + direction + seq_bin (가장 구체적)
    if key_triple in model['zone_dir_seq_x'] and model['zone_dir_seq_count'].get(key_triple, 0) >= min_triple:
        dx = model['zone_dir_seq_x'][key_triple]
        dy = model['zone_dir_seq_y'][key_triple]
        level = 'triple'
    # Level 2: zone + direction
    elif key_double in model['zone_dir_x'] and model['zone_dir_count'].get(key_double, 0) >= min_double:
        dx = model['zone_dir_x'][key_double]
        dy = model['zone_dir_y'][key_double]
        level = 'double'
    # Level 3: zone only
    else:
        dx = model['zone_stats']['delta_x'].get(zone, 0)
        dy = model['zone_stats']['delta_y'].get(zone, 0)
        level = 'single'

    pred_x = np.clip(row['start_x'] + dx, 0, 105)
    pred_y = np.clip(row['start_y'] + dy, 0, 68)

    return pred_x, pred_y, level

# =============================================================================
# 6. GroupKFold 교차 검증
# =============================================================================
print("\n[6] GroupKFold 5-Fold 교차 검증...")

n_splits = 5
gkf = GroupKFold(n_splits=n_splits)
game_ids = train_last['game_id'].values

cv_scores = []
fallback_usage = {'triple': 0, 'double': 0, 'single': 0}

for fold, (train_idx, val_idx) in enumerate(gkf.split(train_last, groups=game_ids)):
    train_fold = train_last.iloc[train_idx]
    val_fold = train_last.iloc[val_idx].copy()

    # 모델 구축
    model = build_hybrid_model(train_fold)

    # 예측
    predictions = val_fold.apply(lambda r: predict_hybrid(r, model), axis=1)
    val_fold['pred_x'] = predictions.apply(lambda x: x[0])
    val_fold['pred_y'] = predictions.apply(lambda x: x[1])
    val_fold['level'] = predictions.apply(lambda x: x[2])

    # Fallback 사용 통계
    for level in ['triple', 'double', 'single']:
        fallback_usage[level] += (val_fold['level'] == level).sum()

    # 평가
    dist = np.sqrt(
        (val_fold['pred_x'] - val_fold['end_x'])**2 +
        (val_fold['pred_y'] - val_fold['end_y'])**2
    )
    fold_score = dist.mean()
    cv_scores.append(fold_score)

    print(f"  Fold {fold+1}: {fold_score:.4f} (n={len(val_fold)})")

cv_mean = np.mean(cv_scores)
cv_std = np.std(cv_scores)

print(f"\n평균 CV Score: {cv_mean:.4f} (std: {cv_std:.4f})")

# Fallback 사용 비율
total_samples = sum(fallback_usage.values())
print(f"\nFallback 사용 비율:")
print(f"  Zone+Dir+Seq (Triple): {fallback_usage['triple']:5d} ({100*fallback_usage['triple']/total_samples:5.1f}%)")
print(f"  Zone+Dir (Double):     {fallback_usage['double']:5d} ({100*fallback_usage['double']/total_samples:5.1f}%)")
print(f"  Zone (Single):         {fallback_usage['single']:5d} ({100*fallback_usage['single']/total_samples:5.1f}%)")

# =============================================================================
# 7. 비교: 8방향 단독 vs 하이브리드
# =============================================================================
print("\n[7] 8방향 단독 모델과 비교...")

def build_direction_8_only(df):
    """8방향 단독 모델 (시퀀스 길이 없음)"""
    df = df.copy()
    df['zone'] = df.apply(lambda r: get_zone_6x6(r['start_x'], r['start_y']), axis=1)
    df['direction'] = df.apply(lambda r: get_direction_8(r['prev_dx'], r['prev_dy']), axis=1)
    df['zone_dir'] = df['zone'].astype(str) + '_' + df['direction']

    zone_stats = df.groupby('zone').agg({
        'delta_x': 'median',
        'delta_y': 'median'
    }).to_dict()

    zone_dir_stats = df.groupby('zone_dir').agg({
        'delta_x': 'median',
        'delta_y': 'median',
        'game_episode': 'count'
    }).rename(columns={'game_episode': 'count'})

    return {
        'zone_stats': zone_stats,
        'zone_dir_x': zone_dir_stats['delta_x'].to_dict(),
        'zone_dir_y': zone_dir_stats['delta_y'].to_dict(),
        'zone_dir_count': zone_dir_stats['count'].to_dict()
    }

def predict_direction_8_only(row, model, min_samples=20):
    """8방향 단독 예측"""
    zone = get_zone_6x6(row['start_x'], row['start_y'])
    direction = get_direction_8(row['prev_dx'], row['prev_dy'])
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

# 8방향 단독 CV
cv_scores_8 = []
for fold, (train_idx, val_idx) in enumerate(gkf.split(train_last, groups=game_ids)):
    train_fold = train_last.iloc[train_idx]
    val_fold = train_last.iloc[val_idx].copy()

    model_8 = build_direction_8_only(train_fold)
    predictions = val_fold.apply(lambda r: predict_direction_8_only(r, model_8), axis=1)
    val_fold['pred_x'] = predictions.apply(lambda x: x[0])
    val_fold['pred_y'] = predictions.apply(lambda x: x[1])

    dist = np.sqrt(
        (val_fold['pred_x'] - val_fold['end_x'])**2 +
        (val_fold['pred_y'] - val_fold['end_y'])**2
    )
    cv_scores_8.append(dist.mean())

cv_mean_8 = np.mean(cv_scores_8)

print(f"\n8방향 단독 CV:      {cv_mean_8:.4f}")
print(f"하이브리드 CV:      {cv_mean:.4f}")
print(f"추가 개선:          {cv_mean_8 - cv_mean:.4f} ({100*(cv_mean_8 - cv_mean)/cv_mean_8:.2f}%)")

# =============================================================================
# 8. Test 예측 및 제출 파일 생성
# =============================================================================
print("\n[8] Test 예측 및 제출 파일 생성...")

# 전체 Train 데이터로 최종 모델 구축
final_model_hybrid = build_hybrid_model(train_last)

# 하이브리드 예측
predictions_hybrid = test_last.apply(lambda r: predict_hybrid(r, final_model_hybrid), axis=1)
test_last['pred_x_hybrid'] = predictions_hybrid.apply(lambda x: x[0])
test_last['pred_y_hybrid'] = predictions_hybrid.apply(lambda x: x[1])

# 제출 파일
submission = pd.DataFrame({
    'game_episode': test_last['game_episode'],
    'end_x': test_last['pred_x_hybrid'],
    'end_y': test_last['pred_y_hybrid']
})
submission = sample_sub[['game_episode']].merge(submission, on='game_episode', how='left')
submission.to_csv('submission_hybrid_8dir_seqlen.csv', index=False)
print(f"  submission_hybrid_8dir_seqlen.csv 저장 완료 (CV: {cv_mean:.4f})")

# =============================================================================
# 9. 최종 요약
# =============================================================================
print("\n" + "=" * 70)
print("최종 요약")
print("=" * 70)

print(f"\n[모델 성능 비교]")
print(f"  6x6 Zone Baseline:           16.68")
print(f"  6x6 + 5방향:                  16.35")
print(f"  6x6 + 8방향:                  {cv_mean_8:.4f}")
print(f"  6x6 + 8방향 + 시퀀스 길이:    {cv_mean:.4f}")
print(f"")
print(f"  Baseline → 하이브리드 개선:   {16.68 - cv_mean:.4f}")
print(f"  8방향 → 하이브리드 추가개선:  {cv_mean_8 - cv_mean:.4f}")

print(f"\n[예상 Public Score]")
print(f"  현재 Best (direction_ensemble): 16.36")
print(f"  예상 (하이브리드, CV+Gap):      {cv_mean + 0.17:.2f} ~ {cv_mean + 0.25:.2f}")
print(f"  예상 개선:                      {16.36 - (cv_mean + 0.21):.2f}")

print(f"\n[모델 특징]")
print(f"  조건 1: 6x6 Zone (36개)")
print(f"  조건 2: 8방향 (E/NE/N/NW/W/SW/S/SE/none)")
print(f"  조건 3: 시퀀스 길이 (short/medium/long)")
print(f"  Fallback: 3-level hierarchical (zone+dir+seq → zone+dir → zone)")
print(f"  최소 샘플: Triple 10개, Double 20개")

print(f"\n[제출 파일]")
print(f"  submission_hybrid_8dir_seqlen.csv")

print("\n" + "=" * 70)
print("완료! 다음 단계: submission_hybrid_8dir_seqlen.csv 제출")
print("=" * 70)
