"""
K리그 패스 좌표 예측 - 8방향 조건부 Zone 모델

전략:
- 5방향 대신 8방향(E/NE/N/NW/W/SW/S/SE) 사용
- 대각선 패스(NE/SE/SW/NW)의 고유한 패턴 포착
- 6x6 Zone과 결합하여 최적 예측

예상 개선:
- CV: 16.35 → 16.0-16.2 (0.15-0.35점)
- Public: 16.36 → 16.1-16.3 (0.1-0.25점)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(".")

print("=" * 70)
print("K리그 패스 좌표 예측 - 8방향 조건부 Zone 모델")
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
# 3. Zone 및 8방향 분류 함수
# =============================================================================
print("\n[3] Zone 및 8방향 분류 함수 정의...")

def get_zone_6x6(x, y):
    """6x6 Zone 분류 (검증된 최적 크기)"""
    x_zone = min(5, int(x / (105 / 6)))
    y_zone = min(5, int(y / (68 / 6)))
    return x_zone * 6 + y_zone

def get_direction_8(prev_dx, prev_dy):
    """
    직전 패스 방향을 8가지로 분류 (+ none)

    8방향:
    - E (East): 전진 (0°)
    - NE (Northeast): 전진+좌측 (45°)
    - N (North): 좌측 (90°)
    - NW (Northwest): 후퇴+좌측 (135°)
    - W (West): 후퇴 (180°)
    - SW (Southwest): 후퇴+우측 (-135°)
    - S (South): 우측 (-90°)
    - SE (Southeast): 전진+우측 (-45°)
    - none: 첫 패스 (이전 없음)
    """
    if abs(prev_dx) < 1 and abs(prev_dy) < 1:
        return 'none'

    angle = np.arctan2(prev_dy, prev_dx)
    angle_deg = np.degrees(angle)

    # 8 directions with 45° bins
    if -22.5 <= angle_deg < 22.5:
        return 'E'   # East: 전진
    elif 22.5 <= angle_deg < 67.5:
        return 'NE'  # Northeast: 전진+좌측 대각선
    elif 67.5 <= angle_deg < 112.5:
        return 'N'   # North: 좌측
    elif 112.5 <= angle_deg < 157.5:
        return 'NW'  # Northwest: 후퇴+좌측 대각선
    elif angle_deg >= 157.5 or angle_deg < -157.5:
        return 'W'   # West: 후퇴
    elif -157.5 <= angle_deg < -112.5:
        return 'SW'  # Southwest: 후퇴+우측 대각선
    elif -112.5 <= angle_deg < -67.5:
        return 'S'   # South: 우측
    else:  # -67.5 to -22.5
        return 'SE'  # Southeast: 전진+우측 대각선

# =============================================================================
# 4. 방향별 통계 분석
# =============================================================================
print("\n[4] 방향별 통계 분석...")

train_last['direction_8'] = train_last.apply(
    lambda r: get_direction_8(r['prev_dx'], r['prev_dy']), axis=1
)

print("\n방향별 샘플 수 및 패스 특성:")
print(f"{'방향':6s} {'샘플수':>7s} {'비율':>6s} {'dx_median':>10s} {'dy_median':>10s}")
print("-" * 50)
for direction in ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE', 'none']:
    subset = train_last[train_last['direction_8'] == direction]
    if len(subset) > 0:
        pct = 100 * len(subset) / len(train_last)
        dx_med = subset['delta_x'].median()
        dy_med = subset['delta_y'].median()
        print(f"{direction:6s} {len(subset):7d} {pct:5.1f}% {dx_med:10.2f} {dy_med:10.2f}")

# =============================================================================
# 5. 8방향 조건부 모델 구축
# =============================================================================
print("\n[5] 8방향 조건부 모델 구축...")

MIN_SAMPLES = 20  # 조건부 통계 최소 샘플 수

def build_direction_8_model(df):
    """8방향 조건부 Zone 통계 구축"""
    df = df.copy()
    df['zone'] = df.apply(lambda r: get_zone_6x6(r['start_x'], r['start_y']), axis=1)
    df['direction'] = df.apply(lambda r: get_direction_8(r['prev_dx'], r['prev_dy']), axis=1)
    df['zone_dir'] = df['zone'].astype(str) + '_' + df['direction']

    # 기본 Zone 통계 (fallback)
    zone_stats = df.groupby('zone').agg({
        'delta_x': 'median',
        'delta_y': 'median'
    }).to_dict()

    # 8방향 조건부 통계
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

def predict_direction_8(row, model, min_samples=MIN_SAMPLES):
    """8방향 조건부 예측 with fallback"""
    zone = get_zone_6x6(row['start_x'], row['start_y'])
    direction = get_direction_8(row['prev_dx'], row['prev_dy'])
    key = f"{zone}_{direction}"

    # Hierarchical fallback: zone_dir → zone
    if key in model['zone_dir_x'] and model['zone_dir_count'].get(key, 0) >= min_samples:
        dx = model['zone_dir_x'][key]
        dy = model['zone_dir_y'][key]
    else:
        # Fallback to zone-only statistics
        dx = model['zone_stats']['delta_x'].get(zone, 0)
        dy = model['zone_stats']['delta_y'].get(zone, 0)

    pred_x = np.clip(row['start_x'] + dx, 0, 105)
    pred_y = np.clip(row['start_y'] + dy, 0, 68)

    return pred_x, pred_y

# =============================================================================
# 6. GroupKFold 교차 검증
# =============================================================================
print("\n[6] GroupKFold 5-Fold 교차 검증...")

n_splits = 5
gkf = GroupKFold(n_splits=n_splits)
game_ids = train_last['game_id'].values

cv_scores = []
fold_details = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(train_last, groups=game_ids)):
    train_fold = train_last.iloc[train_idx]
    val_fold = train_last.iloc[val_idx].copy()

    # 모델 구축
    model = build_direction_8_model(train_fold)

    # 예측
    predictions = val_fold.apply(lambda r: predict_direction_8(r, model), axis=1)
    val_fold['pred_x'] = predictions.apply(lambda x: x[0])
    val_fold['pred_y'] = predictions.apply(lambda x: x[1])

    # 평가
    dist = np.sqrt(
        (val_fold['pred_x'] - val_fold['end_x'])**2 +
        (val_fold['pred_y'] - val_fold['end_y'])**2
    )
    fold_score = dist.mean()
    cv_scores.append(fold_score)
    fold_details.append({
        'fold': fold + 1,
        'score': fold_score,
        'n_samples': len(val_fold)
    })

    print(f"  Fold {fold+1}: {fold_score:.4f} (n={len(val_fold)})")

cv_mean = np.mean(cv_scores)
cv_std = np.std(cv_scores)

print(f"\n평균 CV Score: {cv_mean:.4f} (std: {cv_std:.4f})")

# =============================================================================
# 7. 5방향 모델과 비교
# =============================================================================
print("\n[7] 5방향 모델과 비교...")

def get_direction_5(prev_dx, prev_dy):
    """기존 5방향 분류"""
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

def build_direction_5_model(df):
    """5방향 조건부 모델 (비교용)"""
    df = df.copy()
    df['zone'] = df.apply(lambda r: get_zone_6x6(r['start_x'], r['start_y']), axis=1)
    df['direction'] = df.apply(lambda r: get_direction_5(r['prev_dx'], r['prev_dy']), axis=1)
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

def predict_direction_5(row, model, min_samples=MIN_SAMPLES):
    """5방향 조건부 예측"""
    zone = get_zone_6x6(row['start_x'], row['start_y'])
    direction = get_direction_5(row['prev_dx'], row['prev_dy'])
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

# 5방향 CV
cv_scores_5 = []
for fold, (train_idx, val_idx) in enumerate(gkf.split(train_last, groups=game_ids)):
    train_fold = train_last.iloc[train_idx]
    val_fold = train_last.iloc[val_idx].copy()

    model_5 = build_direction_5_model(train_fold)
    predictions = val_fold.apply(lambda r: predict_direction_5(r, model_5), axis=1)
    val_fold['pred_x'] = predictions.apply(lambda x: x[0])
    val_fold['pred_y'] = predictions.apply(lambda x: x[1])

    dist = np.sqrt(
        (val_fold['pred_x'] - val_fold['end_x'])**2 +
        (val_fold['pred_y'] - val_fold['end_y'])**2
    )
    cv_scores_5.append(dist.mean())

cv_mean_5 = np.mean(cv_scores_5)

print(f"\n5방향 모델 CV: {cv_mean_5:.4f}")
print(f"8방향 모델 CV: {cv_mean:.4f}")
print(f"개선: {cv_mean_5 - cv_mean:.4f} ({100*(cv_mean_5 - cv_mean)/cv_mean_5:.2f}%)")

# =============================================================================
# 8. Test 예측 및 제출 파일 생성
# =============================================================================
print("\n[8] Test 예측 및 제출 파일 생성...")

# 전체 Train 데이터로 최종 모델 구축
final_model_8 = build_direction_8_model(train_last)
final_model_5 = build_direction_5_model(train_last)

# 8방향 예측
predictions_8 = test_last.apply(lambda r: predict_direction_8(r, final_model_8), axis=1)
test_last['pred_x_8'] = predictions_8.apply(lambda x: x[0])
test_last['pred_y_8'] = predictions_8.apply(lambda x: x[1])

# 5방향 예측 (비교용)
predictions_5 = test_last.apply(lambda r: predict_direction_5(r, final_model_5), axis=1)
test_last['pred_x_5'] = predictions_5.apply(lambda x: x[0])
test_last['pred_y_5'] = predictions_5.apply(lambda x: x[1])

# 8방향 제출 파일
submission_8 = pd.DataFrame({
    'game_episode': test_last['game_episode'],
    'end_x': test_last['pred_x_8'],
    'end_y': test_last['pred_y_8']
})
submission_8 = sample_sub[['game_episode']].merge(submission_8, on='game_episode', how='left')
submission_8.to_csv('submission_direction_8.csv', index=False)
print(f"  submission_direction_8.csv 저장 완료 (CV: {cv_mean:.4f})")

# 5방향 제출 파일 (비교용)
submission_5 = pd.DataFrame({
    'game_episode': test_last['game_episode'],
    'end_x': test_last['pred_x_5'],
    'end_y': test_last['pred_y_5']
})
submission_5 = sample_sub[['game_episode']].merge(submission_5, on='game_episode', how='left')
submission_5.to_csv('submission_direction_5_comparison.csv', index=False)
print(f"  submission_direction_5_comparison.csv 저장 완료 (CV: {cv_mean_5:.4f})")

# =============================================================================
# 9. 최종 요약
# =============================================================================
print("\n" + "=" * 70)
print("최종 요약")
print("=" * 70)

print(f"\n[모델 성능 비교]")
print(f"  6x6 Zone Baseline:       16.68")
print(f"  6x6 + 5방향 조건부:       {cv_mean_5:.4f} (개선: {16.68 - cv_mean_5:.4f})")
print(f"  6x6 + 8방향 조건부:       {cv_mean:.4f} (개선: {16.68 - cv_mean:.4f})")
print(f"  5방향 → 8방향 추가 개선:  {cv_mean_5 - cv_mean:.4f}")

print(f"\n[예상 Public Score]")
print(f"  현재 Best (direction_ensemble): 16.36")
print(f"  예상 (8방향, CV+Gap):           {cv_mean + 0.17:.2f} ~ {cv_mean + 0.25:.2f}")
print(f"  예상 개선:                      {16.36 - (cv_mean + 0.21):.2f}")

print(f"\n[제출 파일]")
print(f"  1. submission_direction_8.csv (권장)")
print(f"  2. submission_direction_5_comparison.csv (비교용)")

print(f"\n[방향 분류 상세]")
print(f"  8방향: E(전진), NE(전진+좌), N(좌), NW(후+좌), W(후), SW(후+우), S(우), SE(전진+우), none")
print(f"  특징: 대각선 패스(NE/SE/SW/NW)의 고유한 패턴 포착")
print(f"  최소 샘플: 20개 (불충분시 zone-only 통계로 fallback)")

print("\n" + "=" * 70)
print("완료! 다음 단계: submission_direction_8.csv 제출")
print("=" * 70)
