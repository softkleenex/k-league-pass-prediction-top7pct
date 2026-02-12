"""
K리그 패스 좌표 예측 - Phase 1 최적화 모델

전략:
1. min_samples Grid Search (15-35)
2. Ensemble 가중치 최적화 (scipy.optimize)
3. Fold 1-3 기준 평가
4. 안전하고 점진적 개선

목표:
- Fold 1-3 CV: 16.28-16.32
- Public: 16.33-16.37
- Gap: < 0.05
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(".")

print("=" * 80)
print("K리그 패스 좌표 예측 - Phase 1 최적화 모델")
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
# 3. 유틸리티 함수
# =============================================================================
print("\n[3] 유틸리티 함수 정의...")

def get_zone(x, y, n_x, n_y):
    """NxN Zone 분류"""
    x_zone = min(n_x - 1, int(x / (105 / n_x)))
    y_zone = min(n_y - 1, int(y / (68 / n_y)))
    return x_zone * n_y + y_zone

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

def build_model(df, n_x, n_y, use_direction, min_samples):
    """모델 구축"""
    df = df.copy()
    df['zone'] = df.apply(lambda r: get_zone(r['start_x'], r['start_y'], n_x, n_y), axis=1)

    if use_direction:
        df['direction'] = df.apply(lambda r: get_direction_8way(r['prev_dx'], r['prev_dy']), axis=1)
        df['key'] = df['zone'].astype(str) + '_' + df['direction']
    else:
        df['key'] = df['zone'].astype(str)

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
        'n_x': n_x,
        'n_y': n_y,
        'use_direction': use_direction,
        'min_samples': min_samples
    }

def predict(row, model):
    """예측"""
    zone = get_zone(row['start_x'], row['start_y'], model['n_x'], model['n_y'])

    if model['use_direction']:
        direction = get_direction_8way(row['prev_dx'], row['prev_dy'])
        key = f"{zone}_{direction}"
    else:
        key = str(zone)

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
# 4. min_samples Grid Search
# =============================================================================
print("\n[4] min_samples Grid Search...")

base_configs = [
    {'name': '5x5_8dir', 'zone': (5, 5), 'direction': True},
    {'name': '6x6_8dir', 'zone': (6, 6), 'direction': True},
    {'name': '7x7_8dir', 'zone': (7, 7), 'direction': True},
    {'name': '6x6_simple', 'zone': (6, 6), 'direction': False},
]

gkf = GroupKFold(n_splits=5)
game_ids = train_last['game_id'].values

# Grid Search
min_samples_range = range(15, 36, 3)  # 15, 18, 21, 24, 27, 30, 33
best_configs = []

for config in base_configs:
    print(f"\n  {config['name']} 최적화...")
    best_min_s = None
    best_cv = float('inf')
    best_fold13_cv = float('inf')

    for min_s in min_samples_range:
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(gkf.split(train_last, groups=game_ids)):
            train_fold = train_last.iloc[train_idx]
            val_fold = train_last.iloc[val_idx]

            # 모델 구축
            model = build_model(train_fold, config['zone'][0], config['zone'][1],
                              config['direction'], min_s)

            # 예측
            predictions = val_fold.apply(lambda r: predict(r, model), axis=1)
            pred_x = predictions.apply(lambda x: x[0])
            pred_y = predictions.apply(lambda x: x[1])

            # CV 계산
            dist = np.sqrt((pred_x - val_fold['end_x'])**2 + (pred_y - val_fold['end_y'])**2)
            fold_scores.append(dist.mean())

        # Fold 1-3 평균
        fold13_cv = np.mean(fold_scores[:3])

        if fold13_cv < best_fold13_cv:
            best_fold13_cv = fold13_cv
            best_min_s = min_s
            best_cv = np.mean(fold_scores)

    print(f"    최적 min_samples: {best_min_s}")
    print(f"    Fold 1-3 CV: {best_fold13_cv:.4f}")
    print(f"    전체 CV: {best_cv:.4f}")

    best_configs.append({
        'name': config['name'],
        'zone': config['zone'],
        'direction': config['direction'],
        'min_samples': best_min_s
    })

# =============================================================================
# 5. 최적 가중치 탐색
# =============================================================================
print("\n[5] 최적 가중치 탐색...")

# 각 모델의 Fold 1-3 예측 수집
model_predictions_fold13 = []

for config in best_configs:
    fold13_preds = []

    for fold in range(3):  # Fold 1-3만
        train_idx, val_idx = list(gkf.split(train_last, groups=game_ids))[fold]
        train_fold = train_last.iloc[train_idx]
        val_fold = train_last.iloc[val_idx]

        model = build_model(train_fold, config['zone'][0], config['zone'][1],
                          config['direction'], config['min_samples'])

        predictions = val_fold.apply(lambda r: predict(r, model), axis=1)
        pred_x = predictions.apply(lambda x: x[0]).values
        pred_y = predictions.apply(lambda x: x[1]).values

        fold13_preds.append(np.column_stack([pred_x, pred_y]))

    model_predictions_fold13.append(fold13_preds)

# 실제 값 수집
true_values_fold13 = []
for fold in range(3):
    _, val_idx = list(gkf.split(train_last, groups=game_ids))[fold]
    val_fold = train_last.iloc[val_idx]
    true_values_fold13.append(val_fold[['end_x', 'end_y']].values)

# 가중치 최적화 함수
def objective(weights):
    """가중치 조합의 CV 계산"""
    weights = weights / weights.sum()  # Normalize
    total_error = 0

    for fold in range(3):
        # 앙상블 예측
        ensemble_pred = sum(w * model_predictions_fold13[i][fold]
                          for i, w in enumerate(weights))

        # 에러 계산
        errors = np.sqrt(((ensemble_pred - true_values_fold13[fold])**2).sum(axis=1))
        total_error += errors.mean()

    return total_error / 3  # Fold 1-3 평균

# 초기 가중치 (Inverse Variance)
init_weights = np.array([0.25, 0.25, 0.25, 0.25])

# 최적화
result = minimize(objective, init_weights, method='L-BFGS-B',
                 bounds=[(0.01, 1.0)] * len(best_configs))

optimal_weights = result.x / result.x.sum()
optimal_cv = result.fun

print(f"\n최적 가중치:")
for config, weight in zip(best_configs, optimal_weights):
    print(f"  {config['name']:15s}: {weight:.4f}")
print(f"\nFold 1-3 CV: {optimal_cv:.4f}")

# =============================================================================
# 6. 전체 Fold 검증
# =============================================================================
print("\n[6] 전체 Fold 검증...")

all_fold_scores = []
for fold, (train_idx, val_idx) in enumerate(gkf.split(train_last, groups=game_ids)):
    train_fold = train_last.iloc[train_idx]
    val_fold = train_last.iloc[val_idx]

    # 각 모델 예측
    ensemble_pred_x = np.zeros(len(val_fold))
    ensemble_pred_y = np.zeros(len(val_fold))

    for config, weight in zip(best_configs, optimal_weights):
        model = build_model(train_fold, config['zone'][0], config['zone'][1],
                          config['direction'], config['min_samples'])

        predictions = val_fold.apply(lambda r: predict(r, model), axis=1)
        pred_x = predictions.apply(lambda x: x[0]).values
        pred_y = predictions.apply(lambda x: x[1]).values

        ensemble_pred_x += weight * pred_x
        ensemble_pred_y += weight * pred_y

    # CV 계산
    dist = np.sqrt((ensemble_pred_x - val_fold['end_x'].values)**2 +
                  (ensemble_pred_y - val_fold['end_y'].values)**2)
    cv = dist.mean()
    all_fold_scores.append(cv)

    print(f"  Fold {fold+1}: {cv:.4f}")

fold13_cv_final = np.mean(all_fold_scores[:3])
fold45_cv_final = np.mean(all_fold_scores[3:])
fold13_std = np.std(all_fold_scores[:3])

print(f"\n최종 성능:")
print(f"  Fold 1-3 CV: {fold13_cv_final:.4f} ± {fold13_std:.4f}")
print(f"  Fold 4-5 CV: {fold45_cv_final:.4f}")
print(f"  차이: {fold45_cv_final - fold13_cv_final:+.4f}")

# =============================================================================
# 7. Test 예측
# =============================================================================
print("\n[7] Test 예측...")

ensemble_pred_x = np.zeros(len(test_last))
ensemble_pred_y = np.zeros(len(test_last))

for config, weight in zip(best_configs, optimal_weights):
    model = build_model(train_last, config['zone'][0], config['zone'][1],
                      config['direction'], config['min_samples'])

    predictions = test_last.apply(lambda r: predict(r, model), axis=1)
    pred_x = predictions.apply(lambda x: x[0]).values
    pred_y = predictions.apply(lambda x: x[1]).values

    ensemble_pred_x += weight * pred_x
    ensemble_pred_y += weight * pred_y

# =============================================================================
# 8. 제출 파일 생성
# =============================================================================
print("\n[8] 제출 파일 생성...")

submission = pd.DataFrame({
    'game_episode': test_last['game_episode'],
    'end_x': ensemble_pred_x,
    'end_y': ensemble_pred_y
})
submission = sample_sub[['game_episode']].merge(submission, on='game_episode', how='left')
submission.to_csv('/mnt/c/LSJ/dacon/dacon/kleague-algorithm/submission_tuned_v1.csv', index=False)

print("  submission_tuned_v1.csv 저장 완료")

# =============================================================================
# 9. 최종 요약
# =============================================================================
print("\n" + "=" * 80)
print("최종 요약")
print("=" * 80)

print(f"\n[모델 구성]")
for config, weight in zip(best_configs, optimal_weights):
    print(f"  {config['name']:15s}: min_samples={config['min_samples']:2d}, weight={weight:.4f}")

print(f"\n[성능]")
print(f"  Fold 1-3 CV:   {fold13_cv_final:.4f} ± {fold13_std:.4f}")
print(f"  Fold 4-5 CV:   {fold45_cv_final:.4f}")

# 예상 Public 계산
if fold13_cv_final < 16.30:
    gap_estimate = 0.03
elif fold13_cv_final < 16.33:
    gap_estimate = 0.05
else:
    gap_estimate = 0.08

public_estimate = fold13_cv_final + gap_estimate

print(f"\n[예상]")
print(f"  예상 Gap:      +{gap_estimate:.2f}")
print(f"  예상 Public:   {public_estimate:.2f}")

print(f"\n[비교 - 현재 Best]")
print(f"  현재 Best:     16.3639 (safe_fold13)")
print(f"  튜닝 모델:     {public_estimate:.4f} (예상)")
print(f"  개선:          {16.3639 - public_estimate:+.4f}")

if public_estimate < 16.3639:
    print(f"\n✅ 개선 예상! 제출 권장")
else:
    print(f"\n⚠️ 개선 미미. 추가 조정 고려")

print("\n" + "=" * 80)
print("완료!")
print("=" * 80)
