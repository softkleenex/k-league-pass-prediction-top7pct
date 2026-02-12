"""
K리그 패스 좌표 예측 - 필드 위치 피처 모델

Week 2 첫 실험: 필드 위치 기반 조건화
- 6x6 Zone + 8-way Direction + Field Region
- field_region: defense/midfield/attack (수비/중앙/공격)
- 가설: 필드 위치에 따라 패스 전략 변화
- Fold 1-3 CV 목표: 16.28-16.32

2025-12-08 13:00
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(".")

print("=" * 80)
print("K리그 패스 좌표 예측 - 필드 위치 피처 모델")
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

# 필드 위치 분포 확인
print(f"\n필드 위치 (start_x) 분포:")
print(f"  평균: {train_last['start_x'].mean():.1f}m")
print(f"  중앙값: {train_last['start_x'].median():.1f}m")
print(f"  0-35m (수비): {(train_last['start_x'] < 35).sum()} ({(train_last['start_x'] < 35).sum()/len(train_last)*100:.1f}%)")
print(f"  35-70m (중앙): {((train_last['start_x'] >= 35) & (train_last['start_x'] < 70)).sum()} ({((train_last['start_x'] >= 35) & (train_last['start_x'] < 70)).sum()/len(train_last)*100:.1f}%)")
print(f"  70-105m (공격): {(train_last['start_x'] >= 70).sum()} ({(train_last['start_x'] >= 70).sum()/len(train_last)*100:.1f}%)")

# =============================================================================
# 3. Zone, Direction, Field Region 함수
# =============================================================================
print("\n[3] Zone, Direction, Field Region 함수 정의...")

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

def get_field_region(x):
    """필드 위치 분류"""
    if x < 35:
        return 'defense'  # 수비진영
    elif x < 70:
        return 'midfield'  # 중앙
    else:
        return 'attack'  # 공격진영

# =============================================================================
# 4. 모델 구축 및 예측 함수
# =============================================================================
print("\n[4] 모델 함수 정의...")

def build_model_with_field(df, min_samples):
    """6x6 Zone + 8-way Direction + Field Region 모델"""
    df = df.copy()
    df['zone'] = df.apply(lambda r: get_zone_6x6(r['start_x'], r['start_y']), axis=1)
    df['direction'] = df.apply(lambda r: get_direction_8way(r['prev_dx'], r['prev_dy']), axis=1)
    df['field_region'] = df['start_x'].apply(get_field_region)

    # 3가지 레벨의 통계
    # Level 1: zone + direction + field_region
    df['key_full'] = df['zone'].astype(str) + '_' + df['direction'] + '_' + df['field_region']
    stats_full = df.groupby('key_full').agg({
        'delta_x': 'median',
        'delta_y': 'median',
        'game_episode': 'count'
    }).rename(columns={'game_episode': 'count'})

    # Level 2: zone + direction
    df['key_zd'] = df['zone'].astype(str) + '_' + df['direction']
    stats_zd = df.groupby('key_zd').agg({
        'delta_x': 'median',
        'delta_y': 'median',
        'game_episode': 'count'
    }).rename(columns={'game_episode': 'count'})

    # Level 3: global
    global_dx = df['delta_x'].median()
    global_dy = df['delta_y'].median()

    return {
        'stats_full': stats_full,
        'stats_zd': stats_zd,
        'global': (global_dx, global_dy),
        'min_samples': min_samples
    }

def predict_with_field(row, model):
    """3단계 Fallback 예측"""
    zone = get_zone_6x6(row['start_x'], row['start_y'])
    direction = get_direction_8way(row['prev_dx'], row['prev_dy'])
    field_region = get_field_region(row['start_x'])

    key_full = f"{zone}_{direction}_{field_region}"
    key_zd = f"{zone}_{direction}"

    # Level 1: zone + direction + field_region
    if key_full in model['stats_full'].index and model['stats_full'].loc[key_full, 'count'] >= model['min_samples']:
        dx = model['stats_full'].loc[key_full, 'delta_x']
        dy = model['stats_full'].loc[key_full, 'delta_y']
    # Level 2: zone + direction
    elif key_zd in model['stats_zd'].index and model['stats_zd'].loc[key_zd, 'count'] >= model['min_samples']:
        dx = model['stats_zd'].loc[key_zd, 'delta_x']
        dy = model['stats_zd'].loc[key_zd, 'delta_y']
    # Level 3: global
    else:
        dx, dy = model['global']

    pred_x = np.clip(row['start_x'] + dx, 0, 105)
    pred_y = np.clip(row['start_y'] + dy, 0, 68)
    return pred_x, pred_y

# =============================================================================
# 5. Fold 1-3 교차 검증
# =============================================================================
print("\n[5] Fold 1-3 교차 검증...")

gkf = GroupKFold(n_splits=5)
game_ids = train_last['game_id'].values

min_samples = 25  # 안전한 값

print(f"  min_samples: {min_samples}")

fold_scores = []
for fold, (train_idx, val_idx) in enumerate(gkf.split(train_last, groups=game_ids)):
    if fold >= 3:  # Fold 1-3만
        continue

    train_fold = train_last.iloc[train_idx]
    val_fold = train_last.iloc[val_idx]

    # 모델 구축
    model = build_model_with_field(train_fold, min_samples)

    # 예측
    predictions = val_fold.apply(lambda r: predict_with_field(r, model), axis=1)
    pred_x = predictions.apply(lambda x: x[0])
    pred_y = predictions.apply(lambda x: x[1])

    # CV 계산
    dist = np.sqrt((pred_x - val_fold['end_x'])**2 + (pred_y - val_fold['end_y'])**2)
    cv = dist.mean()
    fold_scores.append(cv)

    print(f"    Fold {fold+1}: {cv:.4f}")

fold13_cv = np.mean(fold_scores)
fold13_std = np.std(fold_scores)

print(f"\n  Fold 1-3 CV: {fold13_cv:.4f} ± {fold13_std:.4f}")

# =============================================================================
# 6. 전체 Fold 검증 (참고용)
# =============================================================================
print("\n[6] 전체 Fold 검증 (참고용)...")

all_fold_scores = []
for fold, (train_idx, val_idx) in enumerate(gkf.split(train_last, groups=game_ids)):
    train_fold = train_last.iloc[train_idx]
    val_fold = train_last.iloc[val_idx]

    model = build_model_with_field(train_fold, min_samples)
    predictions = val_fold.apply(lambda r: predict_with_field(r, model), axis=1)
    pred_x = predictions.apply(lambda x: x[0])
    pred_y = predictions.apply(lambda x: x[1])

    dist = np.sqrt((pred_x - val_fold['end_x'])**2 + (pred_y - val_fold['end_y'])**2)
    cv = dist.mean()
    all_fold_scores.append(cv)

    print(f"  Fold {fold+1}: {cv:.4f}")

fold45_cv = np.mean(all_fold_scores[3:])

print(f"\n  Fold 4-5 CV: {fold45_cv:.4f}")
print(f"  차이: {fold45_cv - fold13_cv:+.4f}")

# =============================================================================
# 7. Test 예측
# =============================================================================
print("\n[7] Test 예측...")

final_model = build_model_with_field(train_last, min_samples)

predictions = test_last.apply(lambda r: predict_with_field(r, final_model), axis=1)
pred_x = predictions.apply(lambda x: x[0]).values
pred_y = predictions.apply(lambda x: x[1]).values

# =============================================================================
# 8. 제출 파일 생성
# =============================================================================
print("\n[8] 제출 파일 생성...")

submission = pd.DataFrame({
    'game_episode': test_last['game_episode'],
    'end_x': pred_x,
    'end_y': pred_y
})
submission = sample_sub[['game_episode']].merge(submission, on='game_episode', how='left')
submission.to_csv('/mnt/c/LSJ/dacon/dacon/kleague-algorithm/submission_field_position.csv', index=False)

print("  submission_field_position.csv 저장 완료")

# =============================================================================
# 9. 최종 요약 및 제출 판단
# =============================================================================
print("\n" + "=" * 80)
print("최종 요약 및 제출 판단")
print("=" * 80)

print(f"\n[모델 구성]")
print(f"  Zone: 6x6 (36 zones)")
print(f"  Direction: 8-way")
print(f"  Field Region: defense/midfield/attack")
print(f"  min_samples: {min_samples}")
print(f"  Fallback: 3단계")

print(f"\n[성능]")
print(f"  Fold 1-3 CV:   {fold13_cv:.4f} ± {fold13_std:.4f}")
print(f"  Fold 4-5 CV:   {fold45_cv:.4f}")

# CV Sweet Spot 체크
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
print(f"  필드 위치 모델:      {public_estimate:.4f} (예상)")
print(f"  개선:                {16.3639 - public_estimate:+.4f}")

print(f"\n[최종 판정]")
if verdict == "ACCEPT" and fold13_std < 0.01 and public_estimate < 16.36:
    print(f"  ✅✅✅ 즉시 제출 권장! ✅✅✅")
    print(f"  - CV Sweet Spot 범위")
    print(f"  - Fold 분산 안정")
    print(f"  - 예상 Public < 16.36")
elif verdict == "ACCEPT":
    print(f"  ✅ 제출 권장")
    print(f"  - CV Sweet Spot 범위")
elif verdict == "REVIEW":
    print(f"  ⚠️ 제출 보류")
    print(f"  - CV Sweet Spot 상한 초과")
    print(f"  - Week 2 다른 접근 시도")
else:
    print(f"  ❌ 제출 불가")

print("\n" + "=" * 80)
print("완료!")
print("=" * 80)
