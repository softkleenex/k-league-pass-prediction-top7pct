"""
K리그 패스 좌표 예측 - 패스 거리 조건부 Zone 모델 (보수적 버전)

핵심 변경:
1. min_samples를 더 높게 유지 (30-50)
2. 앙상블 대신 단일 최적 Zone 크기 사용
3. Fallback 전략 강화

목표:
- CV >= 16.2 엄격 유지
- Gap ~0.17-0.20 예상
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(".")

print("=" * 80)
print("K리그 패스 좌표 예측 - 패스 거리 조건부 Zone 모델 (보수적)")
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
    df['prev_distance'] = np.sqrt(df['prev_dx']**2 + df['prev_dy']**2)
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
# 3. 분류 함수 (보수적)
# =============================================================================
print("\n[3] 분류 함수 정의...")

def get_zone(x, y, n_x, n_y):
    """NxN Zone 분류"""
    x_zone = min(n_x - 1, int(x / (105 / n_x)))
    y_zone = min(n_y - 1, int(y / (68 / n_y)))
    return x_zone * n_y + y_zone

def get_direction_5way(prev_dx, prev_dy):
    """
    5방향 분류 (간단한 버전 - 안정성 우선)
    """
    if abs(prev_dx) < 1 and abs(prev_dy) < 1:
        return 'none'

    angle = np.arctan2(prev_dy, prev_dx)
    angle_deg = np.degrees(angle)

    if -45 <= angle_deg < 45:
        return 'forward'
    elif 45 <= angle_deg < 135:
        return 'up'
    elif angle_deg >= 135 or angle_deg < -135:
        return 'backward'
    else:
        return 'down'

def get_pass_distance_category(distance):
    """패스 거리 분류 (2단계 - 간단화)"""
    if distance < 15:
        return 'short'
    else:
        return 'long'

# =============================================================================
# 4. 모델 구축 및 예측 (보수적)
# =============================================================================
print("\n[4] 모델 구축 및 예측 함수...")

def build_conservative_model(df, n_x, n_y):
    """보수적 모델 (높은 threshold)"""
    df = df.copy()
    df['zone'] = df.apply(lambda r: get_zone(r['start_x'], r['start_y'], n_x, n_y), axis=1)
    df['direction'] = df.apply(lambda r: get_direction_5way(r['prev_dx'], r['prev_dy']), axis=1)
    df['distance_cat'] = df['prev_distance'].apply(get_pass_distance_category)

    # Level 1: Zone only
    zone_stats = df.groupby('zone').agg({
        'delta_x': 'median',
        'delta_y': 'median'
    }).to_dict()

    # Level 2: Zone + Direction (보수적 threshold)
    zone_dir_stats = df.groupby(['zone', 'direction']).agg({
        'delta_x': 'median',
        'delta_y': 'median',
        'game_episode': 'count'
    }).rename(columns={'game_episode': 'count'})

    # Level 3: Zone + Direction + Distance (매우 보수적 threshold)
    zone_dir_dist_stats = df.groupby(['zone', 'direction', 'distance_cat']).agg({
        'delta_x': 'median',
        'delta_y': 'median',
        'game_episode': 'count'
    }).rename(columns={'game_episode': 'count'})

    # 딕셔너리로 변환
    zone_dir_dict_x = {}
    zone_dir_dict_y = {}
    zone_dir_dict_count = {}
    for (zone, direction), row in zone_dir_stats.iterrows():
        key = f"{zone}_{direction}"
        zone_dir_dict_x[key] = row['delta_x']
        zone_dir_dict_y[key] = row['delta_y']
        zone_dir_dict_count[key] = row['count']

    zone_dir_dist_dict_x = {}
    zone_dir_dist_dict_y = {}
    zone_dir_dist_dict_count = {}
    for (zone, direction, distance_cat), row in zone_dir_dist_stats.iterrows():
        key = f"{zone}_{direction}_{distance_cat}"
        zone_dir_dist_dict_x[key] = row['delta_x']
        zone_dir_dist_dict_y[key] = row['delta_y']
        zone_dir_dist_dict_count[key] = row['count']

    return {
        'zone_stats': zone_stats,
        'zone_dir_x': zone_dir_dict_x,
        'zone_dir_y': zone_dir_dict_y,
        'zone_dir_count': zone_dir_dict_count,
        'zone_dir_dist_x': zone_dir_dist_dict_x,
        'zone_dir_dist_y': zone_dir_dist_dict_y,
        'zone_dir_dist_count': zone_dir_dist_dict_count,
        'n_x': n_x,
        'n_y': n_y
    }

def predict_conservative(row, model):
    """보수적 예측 (높은 threshold로 과적합 방지)"""
    zone = get_zone(row['start_x'], row['start_y'], model['n_x'], model['n_y'])
    direction = get_direction_5way(row['prev_dx'], row['prev_dy'])
    distance_cat = get_pass_distance_category(row['prev_distance'])

    # Level 3 시도: Zone + Direction + Distance (threshold = 40)
    key_full = f"{zone}_{direction}_{distance_cat}"
    full_count = model['zone_dir_dist_count'].get(key_full, 0)

    if key_full in model['zone_dir_dist_x'] and full_count >= 40:
        dx = model['zone_dir_dist_x'][key_full]
        dy = model['zone_dir_dist_y'][key_full]
    else:
        # Level 2 Fallback: Zone + Direction (threshold = 30)
        key_dir = f"{zone}_{direction}"
        dir_count = model['zone_dir_count'].get(key_dir, 0)

        if key_dir in model['zone_dir_x'] and dir_count >= 30:
            dx = model['zone_dir_x'][key_dir]
            dy = model['zone_dir_y'][key_dir]
        else:
            # Level 1 Fallback: Zone only
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

ZONE_CONFIGS = [(5, 5), (6, 6), (7, 7)]

# 각 Zone 크기별 CV 점수 저장
cv_scores = {config: [] for config in ZONE_CONFIGS}

print("\n  5-Fold Cross Validation:")
for fold, (train_idx, val_idx) in enumerate(gkf.split(train_last, groups=game_ids)):
    train_fold = train_last.iloc[train_idx]
    val_fold = train_last.iloc[val_idx].copy()

    print(f"\n  Fold {fold+1}:")

    # 각 Zone 크기별 모델 구축 및 예측
    for n_x, n_y in ZONE_CONFIGS:
        model = build_conservative_model(train_fold, n_x, n_y)

        predictions = val_fold.apply(
            lambda r: predict_conservative(r, model),
            axis=1
        )
        val_fold[f'pred_x_{n_x}x{n_y}'] = predictions.apply(lambda x: x[0])
        val_fold[f'pred_y_{n_x}x{n_y}'] = predictions.apply(lambda x: x[1])

        dist = np.sqrt(
            (val_fold[f'pred_x_{n_x}x{n_y}'] - val_fold['end_x'])**2 +
            (val_fold[f'pred_y_{n_x}x{n_y}'] - val_fold['end_y'])**2
        )
        score = dist.mean()
        cv_scores[(n_x, n_y)].append(score)
        print(f"    {n_x}x{n_y}: {score:.4f}")

# =============================================================================
# 6. CV 결과 요약
# =============================================================================
print("\n" + "=" * 80)
print("[6] CV 결과 요약")
print("=" * 80)

print("\n개별 모델 CV 점수:")
for config in ZONE_CONFIGS:
    mean_score = np.mean(cv_scores[config])
    std_score = np.std(cv_scores[config])
    print(f"  {config[0]}x{config[1]} Conservative:  {mean_score:.4f} ± {std_score:.4f}")

# 최적 모델 선택
best_config = min(ZONE_CONFIGS, key=lambda c: np.mean(cv_scores[c]))
best_cv = np.mean(cv_scores[best_config])

print(f"\n최적 모델: {best_config[0]}x{best_config[1]} (CV: {best_cv:.4f})")

# 과적합 위험 평가
print(f"\n과적합 위험 평가:")
if best_cv >= 16.2:
    print(f"  ✅ SAFE: CV {best_cv:.4f} >= 16.2 (안전 구간)")
    print(f"  예상 Gap: +0.15-0.25")
    risk = "낮음"
elif best_cv >= 16.0:
    print(f"  ⚠️  WARNING: CV {best_cv:.4f} in 16.0-16.2 (경계 구간)")
    print(f"  예상 Gap: +0.25-0.35")
    risk = "중간"
else:
    print(f"  ❌ DANGER: CV {best_cv:.4f} < 16.0 (위험 구간)")
    print(f"  예상 Gap: +0.40+")
    risk = "높음"

# =============================================================================
# 7. 전체 데이터로 모델 학습 및 예측
# =============================================================================
print("\n[7] 전체 데이터로 모델 학습...")

# 최적 모델만 사용
n_x, n_y = best_config
model = build_conservative_model(train_last, n_x, n_y)

# Test 예측
predictions = test_last.apply(
    lambda r: predict_conservative(r, model),
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

# 제출 파일 저장
submission_path = Path("submission_pass_distance_conservative.csv")
submission.to_csv(submission_path, index=False)

print(f"  {submission_path} 저장 완료")
print(f"  CV Score: {best_cv:.4f}")

# =============================================================================
# 8. 최종 요약
# =============================================================================
print("\n" + "=" * 80)
print("최종 요약 및 제출 권장사항")
print("=" * 80)

print(f"\n[모델 성능]")
print(f"  {best_config[0]}x{best_config[1]} CV:  {best_cv:.4f} ± {np.std(cv_scores[best_config]):.4f}")
print(f"  과적합 위험:   {risk}")

print(f"\n[기준 모델과 비교]")
print(f"  8방향 모델:        CV 16.03 → Public 16.36 (Gap +0.33)")
print(f"  Distance 보수적:   CV {best_cv:.4f} → Public ??? (예상 Gap +0.17-0.25)")
print(f"  CV 차이:           {best_cv - 16.03:+.4f}")

print(f"\n[예상 Public Score]")
if best_cv >= 16.2:
    lower = best_cv + 0.15
    upper = best_cv + 0.25
    print(f"  예상 범위:     {lower:.2f} - {upper:.2f}")
    print(f"  목표 달성:     {'✅ Yes' if lower <= 16.36 else '⚠️  Marginal'}")
elif best_cv >= 16.0:
    lower = best_cv + 0.25
    upper = best_cv + 0.35
    print(f"  예상 범위:     {lower:.2f} - {upper:.2f}")
    print(f"  목표 달성:     ⚠️  Uncertain")
else:
    print(f"  예상 Public:   > 16.50 (위험)")
    print(f"  목표 달성:     ❌ No")

print(f"\n[제출 권장]")
if best_cv >= 16.2:
    print(f"  ✅ 즉시 제출 권장")
    print(f"  이유: CV 안전 구간 ({best_cv:.4f} >= 16.2)")
elif best_cv >= 16.0:
    print(f"  ⚠️  신중한 제출 권장")
    print(f"  이유: CV 경계 구간 ({best_cv:.4f} in 16.0-16.2)")
else:
    print(f"  ❌ 제출 보류 권장")
    print(f"  이유: 과적합 위험 ({best_cv:.4f} < 16.0)")

print(f"\n[제출 파일]")
print(f"  {submission_path}")
print(f"  ({best_config[0]}x{best_config[1]} Zone, 5방향 + 2단계 거리)")

print("\n" + "=" * 80)
print("완료!")
print("=" * 80)
