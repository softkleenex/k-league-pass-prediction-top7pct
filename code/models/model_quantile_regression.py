"""
K리그 패스 좌표 예측 - Quantile Regression

간단한 아이디어:
- Median (50th percentile) 대신 다른 Quantile 시도
- 40th, 45th, 50th, 55th, 60th percentile Grid Search
- 6x6 Zone + 8-way Direction 유지

가설:
- Median이 최적이 아닐 수 있음
- 약간 보수적/공격적 quantile이 더 나을 수도

목표: Fold 1-3 CV < 16.32

2025-12-09 Phase 2 대체 (빠른 시도)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(".")

print("=" * 80)
print("K리그 패스 좌표 예측 - Quantile Regression")
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
# 3. Zone 및 방향 분류 함수
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
# 4. Quantile 모델 함수
# =============================================================================
print("\n[4] Quantile 모델 함수 정의...")

def build_model_quantile(df, min_samples, quantile):
    """6x6 Zone + 8-way Direction + Quantile Regression"""
    df = df.copy()
    df['zone'] = df.apply(lambda r: get_zone_6x6(r['start_x'], r['start_y']), axis=1)
    df['direction'] = df.apply(lambda r: get_direction_8way(r['prev_dx'], r['prev_dy']), axis=1)
    df['key'] = df['zone'].astype(str) + '_' + df['direction']

    # Quantile 통계 계산
    stats = df.groupby('key').agg({
        'delta_x': lambda x: x.quantile(quantile),
        'delta_y': lambda x: x.quantile(quantile),
        'game_episode': 'count'
    }).rename(columns={'game_episode': 'count'})

    # Zone fallback (Quantile)
    zone_fallback = df.groupby('zone').agg({
        'delta_x': lambda x: x.quantile(quantile),
        'delta_y': lambda x: x.quantile(quantile)
    }).to_dict()

    # Global fallback (Quantile)
    global_dx = df['delta_x'].quantile(quantile)
    global_dy = df['delta_y'].quantile(quantile)

    return {
        'stats': stats,
        'zone_fallback': zone_fallback,
        'global': (global_dx, global_dy),
        'min_samples': min_samples,
        'quantile': quantile
    }

def predict_quantile(row, model):
    """예측"""
    zone = get_zone_6x6(row['start_x'], row['start_y'])
    direction = get_direction_8way(row['prev_dx'], row['prev_dy'])
    key = f"{zone}_{direction}"

    # 계층적 Fallback
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
# 5. Quantile Grid Search
# =============================================================================
print("\n[5] Quantile Grid Search...")

gkf = GroupKFold(n_splits=5)
game_ids = train_last['game_id'].values

quantiles = [0.40, 0.45, 0.50, 0.55, 0.60]
min_samples = 25  # safe_fold13의 검증된 값

best_fold13_cv = float('inf')
best_quantile = None
all_results = []

print(f"\n{'Quantile':>8} {'Fold1-3 CV':>12} {'Fold4-5 CV':>12} {'차이':>8}")
print("-" * 45)

for q in quantiles:
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(train_last, groups=game_ids)):
        train_fold = train_last.iloc[train_idx]
        val_fold = train_last.iloc[val_idx]

        # 모델 구축
        model = build_model_quantile(train_fold, min_samples, q)

        # 예측
        predictions = val_fold.apply(lambda r: predict_quantile(r, model), axis=1)
        pred_x = predictions.apply(lambda x: x[0])
        pred_y = predictions.apply(lambda x: x[1])

        # CV 계산
        dist = np.sqrt((pred_x - val_fold['end_x'])**2 + (pred_y - val_fold['end_y'])**2)
        cv = dist.mean()
        fold_scores.append(cv)

    # 결과 집계
    fold13_cv = np.mean(fold_scores[:3])
    fold13_std = np.std(fold_scores[:3])
    fold45_cv = np.mean(fold_scores[3:])
    diff = fold45_cv - fold13_cv

    all_results.append({
        'quantile': q,
        'fold13_cv': fold13_cv,
        'fold13_std': fold13_std,
        'fold45_cv': fold45_cv,
        'diff': diff,
        'fold_scores': fold_scores
    })

    print(f"{q:8.2f} {fold13_cv:12.4f} {fold45_cv:12.4f} {diff:+8.4f}")

    if fold13_cv < best_fold13_cv:
        best_fold13_cv = fold13_cv
        best_quantile = q

# =============================================================================
# 6. 최적 Quantile로 최종 모델
# =============================================================================
print("\n" + "=" * 80)
print("Grid Search 결과")
print("=" * 80)

print(f"\n최적 Quantile: {best_quantile:.2f}")

best_result = [r for r in all_results if r['quantile'] == best_quantile][0]

print(f"\n최적 성능:")
print(f"  Fold 1-3 CV:   {best_result['fold13_cv']:.4f} ± {best_result['fold13_std']:.4f}")
print(f"  Fold 4-5 CV:   {best_result['fold45_cv']:.4f}")
print(f"  차이:          {best_result['diff']:+.4f}")

print(f"\nFold별 상세:")
for i, score in enumerate(best_result['fold_scores']):
    print(f"  Fold {i+1}: {score:.4f}")

# =============================================================================
# 7. Test 예측
# =============================================================================
print("\n[7] Test 예측...")

final_model = build_model_quantile(train_last, min_samples, best_quantile)

predictions = test_last.apply(lambda r: predict_quantile(r, final_model), axis=1)
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
submission.to_csv('/mnt/c/LSJ/dacon/dacon/kleague-algorithm/submission_quantile_regression.csv',
                  index=False)

print("  submission_quantile_regression.csv 저장 완료")

# =============================================================================
# 9. 최종 요약
# =============================================================================
print("\n" + "=" * 80)
print("최종 요약 및 제출 판단")
print("=" * 80)

fold13_cv = best_result['fold13_cv']
fold13_std = best_result['fold13_std']

print(f"\n[모델 구성]")
print(f"  접근법: Quantile Regression")
print(f"  Zone: 6x6 (36 zones)")
print(f"  Direction: 8-way (45도 간격)")
print(f"  min_samples: {min_samples}")
print(f"  Quantile: {best_quantile:.2f}")

print(f"\n[성능]")
print(f"  Fold 1-3 CV:   {fold13_cv:.4f} ± {fold13_std:.4f}")

# CV Sweet Spot 체크
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
print(f"  예상 Gap:      +{gap_estimate:.3f}")
print(f"  예상 Public:   {public_estimate:.4f}")

print(f"\n[비교]")
print(f"  현재 Best (Median 0.50):  16.3639 (safe_fold13)")
print(f"  Quantile {best_quantile:.2f}:         {public_estimate:.4f} (예상)")
print(f"  개선:                     {16.3639 - public_estimate:+.4f}")

print(f"\n[최종 판정]")
if verdict == "ACCEPT" and fold13_cv < 16.32 and fold13_std < 0.01:
    print(f"  ✅✅✅ 즉시 제출 강력 권장! ✅✅✅")
    print(f"  - CV Sweet Spot 범위")
    print(f"  - CV < 16.32 (개선!)")
    print(f"  - Fold 분산 안정")
elif verdict == "ACCEPT" and fold13_cv <= 16.34:
    print(f"  ✅ 제출 권장")
    print(f"  - CV Sweet Spot 범위")
elif verdict == "ACCEPT":
    print(f"  ⚠️ 제출 보류")
    print(f"  - CV Sweet Spot이나 개선 미미")
elif verdict == "REVIEW":
    print(f"  ⚠️ 제출 보류")
    print(f"  - CV Sweet Spot 상한 초과")
else:
    print(f"  ❌ 제출 불가")

print(f"\n[전체 Quantile 결과]")
for r in all_results:
    marker = " ⭐" if r['quantile'] == best_quantile else ""
    print(f"  q={r['quantile']:.2f}: CV {r['fold13_cv']:.4f}{marker}")

print("\n" + "=" * 80)
print("완료!")
print("=" * 80)
