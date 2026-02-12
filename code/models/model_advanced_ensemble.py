"""
K리그 패스 좌표 예측 - Advanced Ensemble Model
목표: CV < 16.0, Gap 관리

전략:
1. 가중치 Grid Search (최적 조합)
2. 더 많은 Zone + Direction 조합
3. min_samples 최적화
4. Inverse Variance Weighting 개선
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from itertools import product
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("/mnt/c/LSJ/dacon/dacon/kleague-algorithm")

print("=" * 80)
print("K리그 패스 좌표 예측 - Advanced Ensemble Model")
print("목표: CV < 16.0, Gap 관리")
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

    # 패스 거리
    df['pass_distance'] = np.sqrt(df['dx']**2 + df['dy']**2)
    df['prev_pass_distance'] = df.groupby('game_episode')['pass_distance'].shift(1).fillna(0)

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
# 3. Zone 및 방향 분류 함수
# =============================================================================
print("\n[3] Zone 및 방향 분류 함수 정의...")

def get_zone(x, y, n_x, n_y):
    x_zone = min(n_x - 1, int(x / (105 / n_x)))
    y_zone = min(n_y - 1, int(y / (68 / n_y)))
    return x_zone * n_y + y_zone

def get_direction_5way(prev_dx, prev_dy):
    if abs(prev_dx) < 1 and abs(prev_dy) < 1:
        return 'none'
    angle = np.arctan2(prev_dy, prev_dx)
    if angle > np.pi / 3:
        return 'up'
    elif angle < -np.pi / 3:
        return 'down'
    elif prev_dx > 0:
        return 'forward'
    else:
        return 'backward'

def get_direction_8way(prev_dx, prev_dy):
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
# 4. 모델 구축 함수
# =============================================================================
print("\n[4] 모델 구축 함수...")

def build_model(df, n_x, n_y, direction_type='8way', min_samples=20):
    """방향 조건부 Zone 통계 구축"""
    df = df.copy()
    df['zone'] = df.apply(lambda r: get_zone(r['start_x'], r['start_y'], n_x, n_y), axis=1)

    if direction_type == '5way':
        df['direction'] = df.apply(lambda r: get_direction_5way(r['prev_dx'], r['prev_dy']), axis=1)
    else:
        df['direction'] = df.apply(lambda r: get_direction_8way(r['prev_dx'], r['prev_dy']), axis=1)

    df['zone_dir'] = df['zone'].astype(str) + '_' + df['direction']

    # Zone 통계
    zone_stats = df.groupby('zone').agg({
        'delta_x': 'median',
        'delta_y': 'median'
    }).to_dict()

    # Zone+Direction 통계
    zone_dir_stats = {}
    for zd, group in df.groupby('zone_dir'):
        if len(group) >= min_samples:
            zone_dir_stats[zd] = {
                'delta_x': group['delta_x'].median(),
                'delta_y': group['delta_y'].median()
            }

    # 전역 통계
    global_dx = df['delta_x'].median()
    global_dy = df['delta_y'].median()

    return {
        'zone_stats': zone_stats,
        'zone_dir_stats': zone_dir_stats,
        'global': {'delta_x': global_dx, 'delta_y': global_dy},
        'n_x': n_x, 'n_y': n_y,
        'direction_type': direction_type,
        'min_samples': min_samples
    }

def predict_with_model(df, model):
    df = df.copy()
    n_x, n_y = model['n_x'], model['n_y']

    df['zone'] = df.apply(lambda r: get_zone(r['start_x'], r['start_y'], n_x, n_y), axis=1)

    if model['direction_type'] == '5way':
        df['direction'] = df.apply(lambda r: get_direction_5way(r['prev_dx'], r['prev_dy']), axis=1)
    else:
        df['direction'] = df.apply(lambda r: get_direction_8way(r['prev_dx'], r['prev_dy']), axis=1)

    df['zone_dir'] = df['zone'].astype(str) + '_' + df['direction']

    pred_dx, pred_dy = [], []

    for _, row in df.iterrows():
        zd = row['zone_dir']
        z = row['zone']

        if zd in model['zone_dir_stats']:
            pred_dx.append(model['zone_dir_stats'][zd]['delta_x'])
            pred_dy.append(model['zone_dir_stats'][zd]['delta_y'])
        elif z in model['zone_stats']['delta_x']:
            pred_dx.append(model['zone_stats']['delta_x'][z])
            pred_dy.append(model['zone_stats']['delta_y'][z])
        else:
            pred_dx.append(model['global']['delta_x'])
            pred_dy.append(model['global']['delta_y'])

    df['pred_end_x'] = df['start_x'] + np.array(pred_dx)
    df['pred_end_y'] = df['start_y'] + np.array(pred_dy)

    # Clipping
    df['pred_end_x'] = df['pred_end_x'].clip(0, 105)
    df['pred_end_y'] = df['pred_end_y'].clip(0, 68)

    return df[['pred_end_x', 'pred_end_y']]

# =============================================================================
# 5. 다양한 모델 설정
# =============================================================================
print("\n[5] 다양한 모델 설정...")

MODEL_CONFIGS = [
    # (name, n_x, n_y, direction_type, min_samples)
    ('5x5_8way_m20', 5, 5, '8way', 20),
    ('5x5_8way_m25', 5, 5, '8way', 25),
    ('6x6_8way_m20', 6, 6, '8way', 20),
    ('6x6_8way_m25', 6, 6, '8way', 25),
    ('6x6_5way_m20', 6, 6, '5way', 20),
    ('7x7_8way_m20', 7, 7, '8way', 20),
    ('7x7_5way_m25', 7, 7, '5way', 25),
    ('4x4_8way_m15', 4, 4, '8way', 15),
]

print(f"  총 {len(MODEL_CONFIGS)}개 모델 설정")

# =============================================================================
# 6. Cross-Validation with Grid Search
# =============================================================================
print("\n[6] Cross-Validation with Ensemble Optimization...")

n_splits = 5
gkf = GroupKFold(n_splits=n_splits)

# 각 모델의 OOF 예측 저장
model_oof_preds = {name: {'end_x': np.zeros(len(train_last)), 'end_y': np.zeros(len(train_last))}
                   for name, _, _, _, _ in MODEL_CONFIGS}
model_cv_scores = {name: [] for name, _, _, _, _ in MODEL_CONFIGS}

train_last_reset = train_last.reset_index(drop=True)

for fold, (train_idx, val_idx) in enumerate(gkf.split(train_last_reset, groups=train_last_reset['game_id']), 1):
    print(f"\n  Fold {fold}/{n_splits}...")

    fold_train = train_last_reset.iloc[train_idx]
    fold_val = train_last_reset.iloc[val_idx]

    for name, n_x, n_y, dir_type, min_samp in MODEL_CONFIGS:
        # 모델 학습
        model = build_model(fold_train, n_x, n_y, dir_type, min_samp)

        # 예측
        preds = predict_with_model(fold_val.copy(), model)

        # OOF 저장
        model_oof_preds[name]['end_x'][val_idx] = preds['pred_end_x'].values
        model_oof_preds[name]['end_y'][val_idx] = preds['pred_end_y'].values

        # CV 점수
        dist = np.sqrt((preds['pred_end_x'].values - fold_val['end_x'].values)**2 +
                       (preds['pred_end_y'].values - fold_val['end_y'].values)**2)
        model_cv_scores[name].append(dist.mean())

# 개별 모델 CV 점수
print("\n[7] 개별 모델 CV 점수:")
model_mean_cv = {}
for name in model_cv_scores:
    mean_cv = np.mean(model_cv_scores[name])
    std_cv = np.std(model_cv_scores[name])
    model_mean_cv[name] = mean_cv
    print(f"  {name}: {mean_cv:.4f} ± {std_cv:.4f}")

# =============================================================================
# 7. Ensemble Weight Optimization (Grid Search)
# =============================================================================
print("\n[8] 앙상블 가중치 최적화 (Grid Search)...")

# 상위 6개 모델 선택
sorted_models = sorted(model_mean_cv.items(), key=lambda x: x[1])[:6]
top_models = [name for name, _ in sorted_models]
print(f"  상위 모델: {top_models}")

# Grid Search for weights
best_cv = float('inf')
best_weights = None
best_ensemble_name = None

# 여러 가중치 조합 시도
weight_options = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

# 빠른 탐색을 위해 상위 4개만 사용
top_4_models = top_models[:4]

print(f"  Grid Search 진행 중 (상위 4개 모델)...")

count = 0
for w1, w2, w3, w4 in product(weight_options, repeat=4):
    if abs(w1 + w2 + w3 + w4 - 1.0) > 0.01:
        continue
    if w1 == 0 and w2 == 0 and w3 == 0 and w4 == 0:
        continue

    count += 1

    # 앙상블 예측
    ens_x = (w1 * model_oof_preds[top_4_models[0]]['end_x'] +
             w2 * model_oof_preds[top_4_models[1]]['end_x'] +
             w3 * model_oof_preds[top_4_models[2]]['end_x'] +
             w4 * model_oof_preds[top_4_models[3]]['end_x'])

    ens_y = (w1 * model_oof_preds[top_4_models[0]]['end_y'] +
             w2 * model_oof_preds[top_4_models[1]]['end_y'] +
             w3 * model_oof_preds[top_4_models[2]]['end_y'] +
             w4 * model_oof_preds[top_4_models[3]]['end_y'])

    # CV 점수
    dist = np.sqrt((ens_x - train_last_reset['end_x'].values)**2 +
                   (ens_y - train_last_reset['end_y'].values)**2)
    cv = dist.mean()

    if cv < best_cv:
        best_cv = cv
        best_weights = (w1, w2, w3, w4)

print(f"  탐색 조합 수: {count}")
print(f"  최적 가중치: {dict(zip(top_4_models, best_weights))}")
print(f"  최적 CV: {best_cv:.4f}")

# =============================================================================
# 8. 추가 앙상블 시도 (Inverse Variance + Top Models)
# =============================================================================
print("\n[9] Inverse Variance Weighting 앙상블...")

# 상위 5개 모델로 Inverse Variance
top_5_models = top_models[:5]
variances = [np.var(model_cv_scores[name]) for name in top_5_models]
inv_var_weights = [1/v if v > 0 else 1 for v in variances]
inv_var_weights = [w/sum(inv_var_weights) for w in inv_var_weights]

print(f"  모델: {top_5_models}")
print(f"  가중치: {[f'{w:.3f}' for w in inv_var_weights]}")

# Inverse Variance 앙상블 CV
ens_x_iv = sum(w * model_oof_preds[name]['end_x'] for w, name in zip(inv_var_weights, top_5_models))
ens_y_iv = sum(w * model_oof_preds[name]['end_y'] for w, name in zip(inv_var_weights, top_5_models))

dist_iv = np.sqrt((ens_x_iv - train_last_reset['end_x'].values)**2 +
                  (ens_y_iv - train_last_reset['end_y'].values)**2)
cv_iv = dist_iv.mean()
print(f"  Inverse Variance CV: {cv_iv:.4f}")

# =============================================================================
# 9. 최적 앙상블 선택
# =============================================================================
print("\n[10] 최적 앙상블 선택...")

if best_cv < cv_iv:
    print(f"  Grid Search 앙상블 선택: CV {best_cv:.4f}")
    final_weights = dict(zip(top_4_models, best_weights))
    final_cv = best_cv
    ensemble_type = 'grid_search'
else:
    print(f"  Inverse Variance 앙상블 선택: CV {cv_iv:.4f}")
    final_weights = dict(zip(top_5_models, inv_var_weights))
    final_cv = cv_iv
    ensemble_type = 'inverse_variance'

print(f"  최종 CV: {final_cv:.4f}")
print(f"  가중치: {final_weights}")

# =============================================================================
# 10. Test 예측 및 제출 파일 생성
# =============================================================================
print("\n[11] Test 예측...")

# 전체 train으로 모델 재학습
final_models = {}
for name, n_x, n_y, dir_type, min_samp in MODEL_CONFIGS:
    if name in final_weights:
        final_models[name] = build_model(train_last, n_x, n_y, dir_type, min_samp)

# Test 예측
test_preds = {}
for name, model in final_models.items():
    preds = predict_with_model(test_last.copy(), model)
    test_preds[name] = preds

# 앙상블
final_pred_x = sum(w * test_preds[name]['pred_end_x'].values for name, w in final_weights.items())
final_pred_y = sum(w * test_preds[name]['pred_end_y'].values for name, w in final_weights.items())

# Clipping
final_pred_x = np.clip(final_pred_x, 0, 105)
final_pred_y = np.clip(final_pred_y, 0, 68)

# 제출 파일 생성
submission = sample_sub.copy()
submission['end_x'] = final_pred_x
submission['end_y'] = final_pred_y

output_path = DATA_DIR / "submission_advanced_ensemble.csv"
submission.to_csv(output_path, index=False)
print(f"\n제출 파일 저장: {output_path}")

# =============================================================================
# 11. 결과 요약
# =============================================================================
print("\n" + "=" * 80)
print("최종 요약")
print("=" * 80)
print(f"  앙상블 타입: {ensemble_type}")
print(f"  최종 CV: {final_cv:.4f}")
print(f"  모델 수: {len(final_weights)}")
print(f"  가중치: {final_weights}")

# Gap 예측
if final_cv >= 16.2:
    expected_gap = 0.17
elif final_cv >= 16.0:
    expected_gap = 0.30
else:
    expected_gap = 0.35

expected_public = final_cv + expected_gap
print(f"\n  예상 Gap: +{expected_gap:.2f}")
print(f"  예상 Public: {expected_public:.2f}")

if final_cv < 16.0:
    print(f"\n  [SUCCESS] CV < 16.0 달성!")
    print(f"  제출 권장: submission_advanced_ensemble.csv")
else:
    print(f"\n  [INFO] CV >= 16.0")
    print(f"  현재 Best (16.3502)와 비교하여 제출 결정 필요")

print("\n" + "=" * 80)
print("완료!")
print("=" * 80)
