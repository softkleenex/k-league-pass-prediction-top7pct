"""
K리그 패스 좌표 예측 - 시퀀스 길이 피처 모델

Week 1 → Week 2 앞당김: 새 피처 탐색
- 6x6 Zone + 8-way Direction + Sequence Position
- sequence_position: early (1-5), mid (6-10), late (11+)
- 가설: 경기 진행에 따라 패스 패턴 변화
- Fold 1-3 CV 목표: 16.28-16.32

2025-12-08 11:30
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(".")

print("=" * 80)
print("K리그 패스 좌표 예측 - 시퀀스 길이 피처 모델")
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
# 2. 피처 준비 (시퀀스 길이 추가!)
# =============================================================================
print("\n[2] 피처 준비 (시퀀스 길이 추가)...")

def prepare_features(df):
    df = df.copy()
    df['dx'] = df['end_x'] - df['start_x']
    df['dy'] = df['end_y'] - df['start_y']
    df['prev_dx'] = df.groupby('game_episode')['dx'].shift(1).fillna(0)
    df['prev_dy'] = df.groupby('game_episode')['dy'].shift(1).fillna(0)

    # 시퀀스 길이 피처 추가
    df['pass_order'] = df.groupby('game_episode').cumcount() + 1

    return df

train_df = prepare_features(train_df)
test_all = prepare_features(test_all)

train_last = train_df.groupby('game_episode').last().reset_index()
train_last = train_last.dropna(subset=['end_x', 'end_y'])
train_last['delta_x'] = train_last['end_x'] - train_last['start_x']
train_last['delta_y'] = train_last['end_y'] - train_last['start_y']

test_last = test_all.groupby('game_episode').last().reset_index()

# 시퀀스 분포 확인
print(f"\n시퀀스 길이 분포:")
print(f"  평균 패스 수: {train_last['pass_order'].mean():.1f}")
print(f"  중앙값: {train_last['pass_order'].median():.0f}")
print(f"  최소: {train_last['pass_order'].min()}")
print(f"  최대: {train_last['pass_order'].max()}")

# =============================================================================
# 3. Zone, Direction, Sequence Position 함수
# =============================================================================
print("\n[3] Zone, Direction, Sequence Position 함수 정의...")

def get_zone_6x6(x, y):
    """6x6 Zone 분류 (36 zones)"""
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

def get_sequence_position(pass_order):
    """시퀀스 위치 분류"""
    if pass_order <= 5:
        return 'early'
    elif pass_order <= 10:
        return 'mid'
    else:
        return 'late'

# =============================================================================
# 4. 모델 구축 및 예측 함수
# =============================================================================
print("\n[4] 모델 함수 정의...")

def build_model_with_sequence(df, min_samples):
    """6x6 Zone + 8-way Direction + Sequence Position 모델"""
    df = df.copy()
    df['zone'] = df.apply(lambda r: get_zone_6x6(r['start_x'], r['start_y']), axis=1)
    df['direction'] = df.apply(lambda r: get_direction_8way(r['prev_dx'], r['prev_dy']), axis=1)
    df['seq_pos'] = df['pass_order'].apply(get_sequence_position)

    # 3가지 레벨의 통계
    # Level 1: zone + direction + sequence position
    df['key_full'] = df['zone'].astype(str) + '_' + df['direction'] + '_' + df['seq_pos']
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

    # Level 3: zone only
    zone_fallback = df.groupby('zone').agg({
        'delta_x': 'median',
        'delta_y': 'median'
    }).to_dict()

    # Level 4: global
    global_dx = df['delta_x'].median()
    global_dy = df['delta_y'].median()

    return {
        'stats_full': stats_full,
        'stats_zd': stats_zd,
        'zone_fallback': zone_fallback,
        'global': (global_dx, global_dy),
        'min_samples': min_samples
    }

def predict_with_sequence(row, model):
    """4단계 Fallback 예측"""
    zone = get_zone_6x6(row['start_x'], row['start_y'])
    direction = get_direction_8way(row['prev_dx'], row['prev_dy'])
    seq_pos = get_sequence_position(row['pass_order'])

    key_full = f"{zone}_{direction}_{seq_pos}"
    key_zd = f"{zone}_{direction}"

    # Level 1: zone + direction + sequence
    if key_full in model['stats_full'].index and model['stats_full'].loc[key_full, 'count'] >= model['min_samples']:
        dx = model['stats_full'].loc[key_full, 'delta_x']
        dy = model['stats_full'].loc[key_full, 'delta_y']
    # Level 2: zone + direction
    elif key_zd in model['stats_zd'].index and model['stats_zd'].loc[key_zd, 'count'] >= model['min_samples']:
        dx = model['stats_zd'].loc[key_zd, 'delta_x']
        dy = model['stats_zd'].loc[key_zd, 'delta_y']
    # Level 3: zone only
    elif zone in model['zone_fallback']['delta_x']:
        dx = model['zone_fallback']['delta_x'][zone]
        dy = model['zone_fallback']['delta_y'][zone]
    # Level 4: global
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
    model = build_model_with_sequence(train_fold, min_samples)

    # 예측
    predictions = val_fold.apply(lambda r: predict_with_sequence(r, model), axis=1)
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

    model = build_model_with_sequence(train_fold, min_samples)
    predictions = val_fold.apply(lambda r: predict_with_sequence(r, model), axis=1)
    pred_x = predictions.apply(lambda x: x[0])
    pred_y = predictions.apply(lambda x: x[1])

    dist = np.sqrt((pred_x - val_fold['end_x'])**2 + (pred_y - val_fold['end_y'])**2)
    cv = dist.mean()
    all_fold_scores.append(cv)

    print(f"  Fold {fold+1}: {cv:.4f}")

fold45_cv = np.mean(all_fold_scores[3:])
print(f"\n  Fold 4-5 CV: {fold45_cv:.4f}")
print(f"  차이 (Fold 4-5 - Fold 1-3): {fold45_cv - fold13_cv:+.4f}")

# =============================================================================
# 7. Test 예측
# =============================================================================
print("\n[7] Test 예측...")

final_model = build_model_with_sequence(train_last, min_samples)

predictions = test_last.apply(lambda r: predict_with_sequence(r, final_model), axis=1)
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
submission.to_csv('/mnt/c/LSJ/dacon/dacon/kleague-algorithm/submission_sequence_length.csv', index=False)

print("  submission_sequence_length.csv 저장 완료")

# =============================================================================
# 9. 최종 요약 및 제출 판단
# =============================================================================
print("\n" + "=" * 80)
print("최종 요약 및 제출 판단")
print("=" * 80)

print(f"\n[모델 구성]")
print(f"  Zone: 6x6 (36 zones)")
print(f"  Direction: 8-way")
print(f"  Sequence Position: early/mid/late")
print(f"  min_samples: {min_samples}")
print(f"  Fallback: 4단계 (full → zd → zone → global)")

print(f"\n[성능]")
print(f"  Fold 1-3 CV:   {fold13_cv:.4f} ± {fold13_std:.4f}")
print(f"  Fold 4-5 CV:   {fold45_cv:.4f}")
print(f"  차이:          {fold45_cv - fold13_cv:+.4f}")

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

print(f"\n[비교 - 현재 Best]")
print(f"  현재 Best:     16.3639 (safe_fold13)")
print(f"  시퀀스 모델:   {public_estimate:.4f} (예상)")
print(f"  개선:          {16.3639 - public_estimate:+.4f}")

print(f"\n[기존 6x6 모델과 비교]")
print(f"  기존 6x6_8dir: CV ~16.45")
print(f"  시퀀스 모델:   CV {fold13_cv:.4f}")

print(f"\n[최종 판정]")
if verdict == "ACCEPT" and fold13_std < 0.01 and public_estimate < 16.3639:
    print(f"  ✅ 제출 권장!")
    print(f"  - Fold 1-3 CV Sweet Spot 범위")
    print(f"  - Fold 분산 안정 (< 0.01)")
    print(f"  - 예상 Public 개선")
    print(f"  - 내일 아침 최종 검토 후 제출")
elif verdict == "ACCEPT" and fold13_std < 0.01:
    print(f"  ⚠️ 조건부 권장")
    print(f"  - CV는 Sweet Spot 범위")
    print(f"  - 하지만 예상 Public이 현재 Best와 비슷")
    print(f"  - 추가 검토 필요")
elif verdict == "REVIEW":
    print(f"  ⚠️ 추가 검토 필요")
    print(f"  - CV가 Sweet Spot 상한 초과")
    print(f"  - min_samples 조정 고려")
else:
    print(f"  ❌ 제출 불가")
    print(f"  - CV Sweet Spot 이탈 또는 Fold 분산 높음")
    print(f"  - 다른 접근 필요")

print("\n" + "=" * 80)
print("완료!")
print("=" * 80)
