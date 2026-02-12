"""
K리그 패스 좌표 예측 - 5x5 Zone 최적화

전략:
- 5x5 grid (25 zones) → 샘플 많음
- 8-way direction
- min_samples Grid Search [23, 25, 27, 29]
- Fold 1-3 CV 목표: 16.28-16.32

2025-12-08 12:00
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(".")

print("=" * 80)
print("K리그 패스 좌표 예측 - 5x5 Zone 최적화")
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

def get_zone_5x5(x, y):
    """5x5 Zone 분류 (25 zones)"""
    x_zone = min(4, int(x / (105 / 5)))
    y_zone = min(4, int(y / (68 / 5)))
    return x_zone * 5 + y_zone

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

def build_model_5x5(df, min_samples):
    """5x5 Zone + 8-way Direction 모델"""
    df = df.copy()
    df['zone'] = df.apply(lambda r: get_zone_5x5(r['start_x'], r['start_y']), axis=1)
    df['direction'] = df.apply(lambda r: get_direction_8way(r['prev_dx'], r['prev_dy']), axis=1)
    df['key'] = df['zone'].astype(str) + '_' + df['direction']

    # 통계 계산
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

def predict_5x5(row, model):
    """예측"""
    zone = get_zone_5x5(row['start_x'], row['start_y'])
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
# 5. min_samples Grid Search
# =============================================================================
print("\n[5] min_samples Grid Search...")

gkf = GroupKFold(n_splits=5)
game_ids = train_last['game_id'].values

min_samples_candidates = [23, 25, 27, 29]
results = []

for min_s in min_samples_candidates:
    print(f"\n  min_samples = {min_s}")

    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(gkf.split(train_last, groups=game_ids)):
        train_fold = train_last.iloc[train_idx]
        val_fold = train_last.iloc[val_idx]

        # 모델 구축
        model = build_model_5x5(train_fold, min_s)

        # 예측
        predictions = val_fold.apply(lambda r: predict_5x5(r, model), axis=1)
        pred_x = predictions.apply(lambda x: x[0])
        pred_y = predictions.apply(lambda x: x[1])

        # CV 계산
        dist = np.sqrt((pred_x - val_fold['end_x'])**2 + (pred_y - val_fold['end_y'])**2)
        cv = dist.mean()
        fold_scores.append(cv)

        if fold < 3:  # Fold 1-3만 출력
            print(f"    Fold {fold+1}: {cv:.4f}")

    fold13_cv = np.mean(fold_scores[:3])
    fold13_std = np.std(fold_scores[:3])
    fold45_cv = np.mean(fold_scores[3:])

    print(f"    Fold 1-3 CV: {fold13_cv:.4f} ± {fold13_std:.4f}")
    print(f"    Fold 4-5 CV: {fold45_cv:.4f}")
    print(f"    차이: {fold45_cv - fold13_cv:+.4f}")

    results.append({
        'min_samples': min_s,
        'fold13_cv': fold13_cv,
        'fold13_std': fold13_std,
        'fold45_cv': fold45_cv,
        'all_cv': np.mean(fold_scores)
    })

# =============================================================================
# 6. 최적 min_samples 선정
# =============================================================================
print("\n[6] 최적 min_samples 선정...")

results_df = pd.DataFrame(results)
print("\n결과 요약:")
print(results_df.to_string(index=False))

best_result = results_df.loc[results_df['fold13_cv'].idxmin()]
best_min_samples = int(best_result['min_samples'])

print(f"\n최적 min_samples: {best_min_samples}")
print(f"  Fold 1-3 CV: {best_result['fold13_cv']:.4f} ± {best_result['fold13_std']:.4f}")

# =============================================================================
# 7. 최종 모델로 Test 예측
# =============================================================================
print("\n[7] Test 예측...")

final_model = build_model_5x5(train_last, best_min_samples)

predictions = test_last.apply(lambda r: predict_5x5(r, final_model), axis=1)
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
submission.to_csv('/mnt/c/LSJ/dacon/dacon/kleague-algorithm/submission_5x5_optimized.csv', index=False)

print("  submission_5x5_optimized.csv 저장 완료")

# =============================================================================
# 9. 최종 요약 및 제출 판단
# =============================================================================
print("\n" + "=" * 80)
print("최종 요약 및 제출 판단")
print("=" * 80)

print(f"\n[모델 구성]")
print(f"  Zone: 5x5 (25 zones)")
print(f"  Direction: 8-way")
print(f"  min_samples: {best_min_samples}")

print(f"\n[성능]")
print(f"  Fold 1-3 CV:   {best_result['fold13_cv']:.4f} ± {best_result['fold13_std']:.4f}")
print(f"  Fold 4-5 CV:   {best_result['fold45_cv']:.4f}")
print(f"  차이:          {best_result['fold45_cv'] - best_result['fold13_cv']:+.4f}")

# CV Sweet Spot 체크
if best_result['fold13_cv'] < 16.27:
    print(f"\n⚠️ 경고: CV < 16.27 (과최적화 위험!)")
    gap_estimate = 0.13
    verdict = "REJECT"
elif best_result['fold13_cv'] <= 16.34:
    print(f"\n✅ CV Sweet Spot 범위 (16.27-16.34)")
    gap_estimate = 0.03 + (best_result['fold13_cv'] - 16.27) * 0.10 / 0.07
    verdict = "ACCEPT"
else:
    print(f"\n⚠️ CV가 Sweet Spot 상한 초과")
    gap_estimate = 0.08
    verdict = "REVIEW"

public_estimate = best_result['fold13_cv'] + gap_estimate

print(f"\n[예상]")
print(f"  예상 Gap:      +{gap_estimate:.3f}")
print(f"  예상 Public:   {public_estimate:.4f}")

print(f"\n[비교 - 현재 Best]")
print(f"  현재 Best:     16.3639 (safe_fold13)")
print(f"  5x5 최적화:    {public_estimate:.4f} (예상)")
print(f"  개선:          {16.3639 - public_estimate:+.4f}")

print(f"\n[기존 5x5와 비교]")
print(f"  기존 5x5 (min_samples=25): CV ~16.50")
print(f"  최적화 5x5: CV {best_result['fold13_cv']:.4f}")

print(f"\n[최종 판정]")
if verdict == "ACCEPT" and best_result['fold13_std'] < 0.01 and public_estimate < 16.3639:
    print(f"  ✅✅✅ 즉시 제출 권장! ✅✅✅")
    print(f"  - Fold 1-3 CV Sweet Spot 범위")
    print(f"  - Fold 분산 안정 (< 0.01)")
    print(f"  - 예상 Public이 현재 Best보다 개선")
    print(f"  - 바로 제출하세요!")
elif verdict == "ACCEPT" and best_result['fold13_std'] < 0.01:
    print(f"  ✅ 제출 권장 (개선폭 작음)")
    print(f"  - CV는 Sweet Spot 범위")
    print(f"  - Fold 분산 안정")
    print(f"  - 예상 Public이 현재 Best와 비슷")
elif verdict == "REVIEW":
    print(f"  ⚠️ 추가 검토 필요")
    print(f"  - CV가 Sweet Spot 상한 초과")
else:
    print(f"  ❌ 제출 불가")
    print(f"  - CV Sweet Spot 이탈")

print("\n" + "=" * 80)
print("완료!")
print("=" * 80)
