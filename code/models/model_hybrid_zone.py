"""
K리그 패스 좌표 예측 - Hybrid Zone 적응적 해상도

핵심 아이디어:
- 필드 위치에 따라 Zone 해상도 다르게 적용
- 수비 (0-35m):     5x5 Zone (샘플 많음 → 세밀)
- 미드필드 (35-70m): 6x6 Zone (균형)
- 공격 (70-105m):    7x7 Zone (샘플 적음 → 안정)

근거:
- 데이터 분석 결과 기반
- Zone 경계 문제 완화
- 샘플 밀도 고려한 적응적 해상도

목표: Fold 1-3 CV < 16.32 (현재 16.34 대비 -0.02)

2025-12-09 Hybrid Zone 구현
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(".")

print("=" * 80)
print("K리그 패스 좌표 예측 - Hybrid Zone 적응적 해상도")
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
# 3. Hybrid Zone 함수
# =============================================================================
print("\n[3] Hybrid Zone 함수 정의...")

# 하이퍼파라미터 (데이터 분석 결과)
BOUNDARIES = (35, 70)  # 필드 분할 경계
ZONE_CONFIG = {
    'defense': (5, 5),    # 0-35m: 5x5
    'midfield': (6, 6),   # 35-70m: 6x6
    'attack': (7, 7)      # 70-105m: 7x7
}

def get_field_region(x):
    """필드 영역 판별"""
    if x < BOUNDARIES[0]:
        return 'defense'
    elif x < BOUNDARIES[1]:
        return 'midfield'
    else:
        return 'attack'

def get_zone_hybrid(x, y):
    """적응적 Zone 분류"""
    region = get_field_region(x)
    n_x, n_y = ZONE_CONFIG[region]

    # 영역별 x 좌표 범위
    if region == 'defense':
        x_min, x_max = 0, BOUNDARIES[0]
    elif region == 'midfield':
        x_min, x_max = BOUNDARIES[0], BOUNDARIES[1]
    else:
        x_min, x_max = BOUNDARIES[1], 105

    # 영역 내에서 Zone 계산
    x_local = x - x_min
    zone_width_x = (x_max - x_min) / n_x
    zone_width_y = 68 / n_y

    x_zone = min(n_x - 1, int(x_local / zone_width_x))
    y_zone = min(n_y - 1, int(y / zone_width_y))

    # 전역 zone_id (region별로 고유)
    zone_id = f"{region}_{x_zone}_{y_zone}"

    return zone_id, region

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
# 4. Hybrid Zone 모델
# =============================================================================
print("\n[4] Hybrid Zone 모델 함수...")

def build_model_hybrid(df, min_samples):
    """Hybrid Zone + 8-way Direction 모델"""
    df = df.copy()

    # Zone 및 Direction 계산
    df['zone_id'], df['region'] = zip(*df.apply(
        lambda r: get_zone_hybrid(r['start_x'], r['start_y']), axis=1))
    df['direction'] = df.apply(
        lambda r: get_direction_8way(r['prev_dx'], r['prev_dy']), axis=1)

    # 통계 Level 1: Zone + Direction
    df['key_full'] = df['zone_id'] + '_' + df['direction']
    stats_full = df.groupby('key_full').agg({
        'delta_x': 'median',
        'delta_y': 'median',
        'game_episode': 'count'
    }).rename(columns={'game_episode': 'count'})

    # 통계 Level 2: Zone only
    stats_zone = df.groupby('zone_id').agg({
        'delta_x': 'median',
        'delta_y': 'median'
    }).to_dict()

    # 통계 Level 3: Region only
    stats_region = df.groupby('region').agg({
        'delta_x': 'median',
        'delta_y': 'median'
    }).to_dict()

    # Global fallback
    global_dx = df['delta_x'].median()
    global_dy = df['delta_y'].median()

    return {
        'stats_full': stats_full,
        'stats_zone': stats_zone,
        'stats_region': stats_region,
        'global': (global_dx, global_dy),
        'min_samples': min_samples
    }

def predict_hybrid(row, model):
    """계층적 Fallback 예측"""
    zone_id, region = get_zone_hybrid(row['start_x'], row['start_y'])
    direction = get_direction_8way(row['prev_dx'], row['prev_dy'])

    # Level 1: Zone + Direction
    key_full = f"{zone_id}_{direction}"
    if (key_full in model['stats_full'].index and
        model['stats_full'].loc[key_full, 'count'] >= model['min_samples']):
        dx = model['stats_full'].loc[key_full, 'delta_x']
        dy = model['stats_full'].loc[key_full, 'delta_y']
    # Level 2: Zone only
    elif zone_id in model['stats_zone']['delta_x']:
        dx = model['stats_zone']['delta_x'][zone_id]
        dy = model['stats_zone']['delta_y'][zone_id]
    # Level 3: Region only
    elif region in model['stats_region']['delta_x']:
        dx = model['stats_region']['delta_x'][region]
        dy = model['stats_region']['delta_y'][region]
    # Level 4: Global
    else:
        dx, dy = model['global']

    pred_x = np.clip(row['start_x'] + dx, 0, 105)
    pred_y = np.clip(row['start_y'] + dy, 0, 68)
    return pred_x, pred_y

# =============================================================================
# 5. Grid Search (min_samples)
# =============================================================================
print("\n[5] min_samples Grid Search...")

gkf = GroupKFold(n_splits=5)
game_ids = train_last['game_id'].values

min_samples_range = [20, 22, 25]
best_fold13_cv = float('inf')
best_min_samples = None
all_results = []

print(f"\n{'min_samples':>12} {'Fold1-3 CV':>12} {'Std':>8} {'Fold4-5 CV':>12}")
print("-" * 50)

for min_samples in min_samples_range:
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(train_last, groups=game_ids)):
        train_fold = train_last.iloc[train_idx]
        val_fold = train_last.iloc[val_idx]

        # 모델 구축
        model = build_model_hybrid(train_fold, min_samples)

        # 예측
        predictions = val_fold.apply(lambda r: predict_hybrid(r, model), axis=1)
        pred_x = predictions.apply(lambda x: x[0])
        pred_y = predictions.apply(lambda x: x[1])

        # CV 계산
        dist = np.sqrt((pred_x - val_fold['end_x'])**2 +
                      (pred_y - val_fold['end_y'])**2)
        cv = dist.mean()
        fold_scores.append(cv)

    # 결과 집계
    fold13_cv = np.mean(fold_scores[:3])
    fold13_std = np.std(fold_scores[:3])
    fold45_cv = np.mean(fold_scores[3:])

    all_results.append({
        'min_samples': min_samples,
        'fold13_cv': fold13_cv,
        'fold13_std': fold13_std,
        'fold45_cv': fold45_cv,
        'fold_scores': fold_scores
    })

    print(f"{min_samples:12d} {fold13_cv:12.4f} {fold13_std:8.4f} {fold45_cv:12.4f}")

    if fold13_cv < best_fold13_cv:
        best_fold13_cv = fold13_cv
        best_min_samples = min_samples

# =============================================================================
# 6. 최적 모델로 최종 평가
# =============================================================================
print("\n" + "=" * 80)
print("Grid Search 결과")
print("=" * 80)

print(f"\n최적 min_samples: {best_min_samples}")

best_result = [r for r in all_results if r['min_samples'] == best_min_samples][0]

print(f"\n최적 성능:")
print(f"  Fold 1-3 CV:   {best_result['fold13_cv']:.4f} ± {best_result['fold13_std']:.4f}")
print(f"  Fold 4-5 CV:   {best_result['fold45_cv']:.4f}")
print(f"  차이:          {best_result['fold45_cv'] - best_result['fold13_cv']:+.4f}")

print(f"\nFold별 상세:")
for i, score in enumerate(best_result['fold_scores']):
    print(f"  Fold {i+1}: {score:.4f}")

# =============================================================================
# 7. Test 예측
# =============================================================================
print("\n[7] Test 예측...")

final_model = build_model_hybrid(train_last, best_min_samples)

predictions = test_last.apply(lambda r: predict_hybrid(r, final_model), axis=1)
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
submission.to_csv('/mnt/c/LSJ/dacon/dacon/kleague-algorithm/submission_hybrid_zone.csv',
                  index=False)

print("  submission_hybrid_zone.csv 저장 완료")

# =============================================================================
# 9. 최종 요약 및 제출 판단
# =============================================================================
print("\n" + "=" * 80)
print("최종 요약 및 제출 판단")
print("=" * 80)

fold13_cv = best_result['fold13_cv']
fold13_std = best_result['fold13_std']

print(f"\n[모델 구성]")
print(f"  접근법: Hybrid Zone (적응적 해상도)")
print(f"  경계: {BOUNDARIES}")
print(f"  Zone 설정:")
print(f"    수비 (0-{BOUNDARIES[0]}m):     {ZONE_CONFIG['defense'][0]}x{ZONE_CONFIG['defense'][1]}")
print(f"    미드 ({BOUNDARIES[0]}-{BOUNDARIES[1]}m): {ZONE_CONFIG['midfield'][0]}x{ZONE_CONFIG['midfield'][1]}")
print(f"    공격 ({BOUNDARIES[1]}-105m):    {ZONE_CONFIG['attack'][0]}x{ZONE_CONFIG['attack'][1]}")
print(f"  min_samples: {best_min_samples}")

print(f"\n[성능]")
print(f"  Fold 1-3 CV:   {fold13_cv:.4f} ± {fold13_std:.4f}")

# CV Sweet Spot 체크
if fold13_cv < 16.27:
    verdict = "REJECT"
    gap_estimate = 0.13
    risk = "HIGH"
elif fold13_cv <= 16.34:
    verdict = "ACCEPT"
    gap_estimate = 0.03 + (fold13_cv - 16.27) * 0.05 / 0.07
    risk = "LOW"
else:
    verdict = "REVIEW"
    gap_estimate = 0.08
    risk = "MEDIUM"

public_estimate = fold13_cv + gap_estimate

print(f"\n[예상]")
print(f"  예상 Gap:      +{gap_estimate:.3f}")
print(f"  예상 Public:   {public_estimate:.4f}")
print(f"  리스크:        {risk}")

print(f"\n[비교]")
print(f"  현재 Best (safe_fold13):")
print(f"    Fold 1-3 CV: 16.3356")
print(f"    Public:      16.3639")
print(f"    Gap:         +0.028")
print(f"\n  Hybrid Zone:")
print(f"    Fold 1-3 CV: {fold13_cv:.4f} ± {fold13_std:.4f}")
print(f"    예상 Public: {public_estimate:.4f}")
print(f"    예상 Gap:    +{gap_estimate:.3f}")
print(f"\n  개선:")
print(f"    CV:         {16.3356 - fold13_cv:+.4f}")
print(f"    Public:     {16.3639 - public_estimate:+.4f}")

print(f"\n[최종 판정]")
if verdict == "ACCEPT" and fold13_cv < 16.30 and fold13_std < 0.01:
    print(f"  ✅✅✅ 즉시 제출 강력 권장! ✅✅✅")
    print(f"  - CV Sweet Spot 범위")
    print(f"  - CV < 16.30 (대성공!)")
    print(f"  - Fold 분산 < 0.01 (안정)")
    print(f"  - 완전히 새로운 접근법")
elif verdict == "ACCEPT" and fold13_cv <= 16.32 and fold13_std < 0.01:
    print(f"  ✅✅ 제출 권장! ✅✅")
    print(f"  - CV Sweet Spot 범위")
    print(f"  - CV ≤ 16.32 (성공!)")
    print(f"  - Fold 분산 안정")
elif verdict == "ACCEPT" and fold13_cv <= 16.34:
    print(f"  ✅ 제출 고려")
    print(f"  - CV Sweet Spot 범위")
    print(f"  - 소폭 개선")
elif verdict == "REVIEW":
    print(f"  ⚠️ 제출 보류")
    print(f"  - CV Sweet Spot 상한 초과")
else:
    print(f"  ❌ 제출 불가")
    print(f"  - 과최적화 위험")

print(f"\n[전체 min_samples 결과]")
for r in all_results:
    marker = " ⭐" if r['min_samples'] == best_min_samples else ""
    print(f"  min={r['min_samples']:2d}: CV {r['fold13_cv']:.4f} ± {r['fold13_std']:.4f}{marker}")

print("\n" + "=" * 80)
print("완료!")
print("=" * 80)
