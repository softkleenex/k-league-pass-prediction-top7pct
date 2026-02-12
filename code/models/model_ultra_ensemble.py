"""
K리그 패스 좌표 예측 - Ultra Ensemble Model
목표: CV < 15.95, Public < 16.30

전략:
1. 20+ 통계 모델: Zone × Direction × Pass Distance × min_samples
2. 보수적 XGBoost (15% 가중치 이하)
3. Bayesian Optimization
4. 안전장치: CV < 16.0 달성 시만 제출 권장
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("/mnt/c/LSJ/dacon/dacon/kleague-algorithm")

print("=" * 80)
print("K리그 패스 좌표 예측 - Ultra Ensemble Model")
print("목표: CV < 15.95, Public < 16.30")
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
# 2. 피처 엔지니어링
# =============================================================================
print("\n[2] 피처 엔지니어링...")

def engineer_features(df):
    """통계 및 ML 모델을 위한 피처 생성"""
    df = df.copy()

    # 기본 델타
    df['dx'] = df['end_x'] - df['start_x']
    df['dy'] = df['end_y'] - df['start_y']
    df['pass_distance'] = np.sqrt(df['dx']**2 + df['dy']**2)

    # 이전 패스 정보
    df['prev_dx'] = df.groupby('game_episode')['dx'].shift(1).fillna(0)
    df['prev_dy'] = df.groupby('game_episode')['dy'].shift(1).fillna(0)
    df['prev_distance'] = df.groupby('game_episode')['pass_distance'].shift(1).fillna(0)

    # 시퀀스 위치
    df['seq_pos'] = df.groupby('game_episode').cumcount()
    df['seq_pos_norm'] = df.groupby('game_episode')['seq_pos'].transform(lambda x: x / max(x.max(), 1))

    # 필드 영역
    df['field_zone_x'] = pd.cut(df['start_x'], bins=[0, 35, 70, 105], labels=['defensive', 'middle', 'offensive'])
    df['field_zone_y'] = pd.cut(df['start_y'], bins=[0, 22.67, 45.33, 68], labels=['left', 'center', 'right'])

    # 패스 거리 구간
    df['pass_dist_bin'] = pd.cut(df['prev_distance'], bins=[0, 15, 30, 100], labels=['short', 'medium', 'long'])

    return df

train_df = engineer_features(train_df)
test_all = engineer_features(test_all)

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
    """Zone 분류"""
    x_zone = min(n_x - 1, int(x / (105 / n_x)))
    y_zone = min(n_y - 1, int(y / (68 / n_y)))
    return x_zone * n_y + y_zone

def get_direction_5way(prev_dx, prev_dy):
    """5방향 분류"""
    if abs(prev_dx) < 1 and abs(prev_dy) < 1:
        return 'none'
    angle = np.arctan2(prev_dy, prev_dx)
    if angle > np.pi * 2/3:
        return 'back_up'
    elif angle > np.pi / 3:
        return 'up'
    elif angle > -np.pi / 3:
        return 'forward' if prev_dx > 0 else 'backward'
    elif angle > -np.pi * 2/3:
        return 'down'
    else:
        return 'back_down'

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
# 4. 통계 모델 구축 함수
# =============================================================================
print("\n[4] 통계 모델 구축 함수...")

def build_stat_model(df, n_x, n_y, direction_type='8way', pass_dist_cond=False, min_samples=20):
    """다차원 조건부 통계 모델"""
    df = df.copy()
    df['zone'] = df.apply(lambda r: get_zone(r['start_x'], r['start_y'], n_x, n_y), axis=1)

    # 방향
    if direction_type == '5way':
        df['direction'] = df.apply(lambda r: get_direction_5way(r['prev_dx'], r['prev_dy']), axis=1)
    else:
        df['direction'] = df.apply(lambda r: get_direction_8way(r['prev_dx'], r['prev_dy']), axis=1)

    # 조건 키 생성
    if pass_dist_cond:
        df['condition_key'] = (df['zone'].astype(str) + '_' +
                               df['direction'] + '_' +
                               df['pass_dist_bin'].astype(str))
    else:
        df['condition_key'] = df['zone'].astype(str) + '_' + df['direction']

    # 계층적 통계 (3단계 fallback)
    # Level 1: 전체 조건
    cond_stats = {}
    for key, group in df.groupby('condition_key'):
        if len(group) >= min_samples:
            cond_stats[key] = {
                'delta_x': group['delta_x'].median(),
                'delta_y': group['delta_y'].median()
            }

    # Level 2: Zone + Direction만
    zone_dir_stats = {}
    df['zone_dir'] = df['zone'].astype(str) + '_' + df['direction']
    for key, group in df.groupby('zone_dir'):
        if len(group) >= min_samples:
            zone_dir_stats[key] = {
                'delta_x': group['delta_x'].median(),
                'delta_y': group['delta_y'].median()
            }

    # Level 3: Zone만
    zone_stats = df.groupby('zone').agg({
        'delta_x': 'median',
        'delta_y': 'median'
    }).to_dict()

    # Level 4: Global
    global_dx = df['delta_x'].median()
    global_dy = df['delta_y'].median()

    return {
        'cond_stats': cond_stats,
        'zone_dir_stats': zone_dir_stats,
        'zone_stats': zone_stats,
        'global': {'delta_x': global_dx, 'delta_y': global_dy},
        'n_x': n_x, 'n_y': n_y,
        'direction_type': direction_type,
        'pass_dist_cond': pass_dist_cond,
        'min_samples': min_samples
    }

def predict_with_stat_model(row, model):
    """계층적 예측 (Fallback)"""
    zone = get_zone(row['start_x'], row['start_y'], model['n_x'], model['n_y'])

    if model['direction_type'] == '5way':
        direction = get_direction_5way(row['prev_dx'], row['prev_dy'])
    else:
        direction = get_direction_8way(row['prev_dx'], row['prev_dy'])

    # 조건 키 생성
    if model['pass_dist_cond']:
        cond_key = f"{zone}_{direction}_{row['pass_dist_bin']}"
    else:
        cond_key = f"{zone}_{direction}"

    zone_dir_key = f"{zone}_{direction}"

    # 계층적 Fallback
    if cond_key in model['cond_stats']:
        dx = model['cond_stats'][cond_key]['delta_x']
        dy = model['cond_stats'][cond_key]['delta_y']
    elif zone_dir_key in model['zone_dir_stats']:
        dx = model['zone_dir_stats'][zone_dir_key]['delta_x']
        dy = model['zone_dir_stats'][zone_dir_key]['delta_y']
    elif zone in model['zone_stats']['delta_x']:
        dx = model['zone_stats']['delta_x'][zone]
        dy = model['zone_stats']['delta_y'][zone]
    else:
        dx = model['global']['delta_x']
        dy = model['global']['delta_y']

    pred_x = np.clip(row['start_x'] + dx, 0, 105)
    pred_y = np.clip(row['start_y'] + dy, 0, 68)

    return pred_x, pred_y

# =============================================================================
# 5. 다양한 통계 모델 설정 (20+ 모델)
# =============================================================================
print("\n[5] 다양한 통계 모델 설정 (20+ 모델)...")

STAT_MODEL_CONFIGS = []

# Zone × Direction × min_samples (16 모델)
for zone_size in [(4, 4), (5, 5), (6, 6), (7, 7)]:
    for dir_type in ['5way', '8way']:
        for min_samp in [20, 25]:
            name = f"{zone_size[0]}x{zone_size[1]}_{dir_type}_m{min_samp}"
            STAT_MODEL_CONFIGS.append((name, zone_size[0], zone_size[1], dir_type, False, min_samp))

# Zone × Direction × PassDistance (8 모델 - 가장 세밀)
for zone_size in [(5, 5), (6, 6)]:
    for dir_type in ['5way', '8way']:
        for min_samp in [15, 20]:
            name = f"{zone_size[0]}x{zone_size[1]}_{dir_type}_pd_m{min_samp}"
            STAT_MODEL_CONFIGS.append((name, zone_size[0], zone_size[1], dir_type, True, min_samp))

# 보수적 모델 (높은 min_samples) - 4 모델
for zone_size in [(6, 6), (7, 7)]:
    for dir_type in ['5way', '8way']:
        name = f"{zone_size[0]}x{zone_size[1]}_{dir_type}_conservative"
        STAT_MODEL_CONFIGS.append((name, zone_size[0], zone_size[1], dir_type, False, 30))

print(f"  통계 모델 총 {len(STAT_MODEL_CONFIGS)}개")

# =============================================================================
# 6. XGBoost 모델 (보수적)
# =============================================================================
print("\n[6] 보수적 XGBoost 모델 준비...")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    print("  XGBoost 사용 가능")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("  XGBoost 미설치 - 통계 모델만 사용")

def build_xgboost_model(df, target_col='delta_x'):
    """극도로 보수적인 XGBoost"""
    if not XGBOOST_AVAILABLE:
        return None

    # 핵심 피처만 선택 (10개 이하)
    feature_cols = [
        'start_x', 'start_y',
        'prev_dx', 'prev_dy', 'prev_distance',
        'seq_pos_norm'
    ]

    X = df[feature_cols].copy()
    y = df[target_col].values

    # 극도 정규화 파라미터
    params = {
        'max_depth': 3,
        'learning_rate': 0.01,
        'n_estimators': 100,
        'min_child_weight': 100,
        'subsample': 0.5,
        'colsample_bytree': 0.5,
        'reg_alpha': 10,
        'reg_lambda': 10,
        'objective': 'reg:squarederror',
        'random_state': 42,
        'n_jobs': -1
    }

    model = xgb.XGBRegressor(**params)
    model.fit(X, y, verbose=False)

    return {
        'model': model,
        'features': feature_cols,
        'scaler': None
    }

def predict_with_xgboost(df, model_x, model_y):
    """XGBoost 예측"""
    if model_x is None or model_y is None:
        return None, None

    X = df[model_x['features']].copy()

    pred_dx = model_x['model'].predict(X)
    pred_dy = model_y['model'].predict(X)

    pred_x = np.clip(df['start_x'].values + pred_dx, 0, 105)
    pred_y = np.clip(df['start_y'].values + pred_dy, 0, 68)

    return pred_x, pred_y

# =============================================================================
# 7. Cross-Validation with OOF Predictions
# =============================================================================
print("\n[7] Cross-Validation 시작...")

n_splits = 5
gkf = GroupKFold(n_splits=n_splits)

# OOF 저장
stat_model_oof = {name: {'x': np.zeros(len(train_last)), 'y': np.zeros(len(train_last))}
                  for name, _, _, _, _, _ in STAT_MODEL_CONFIGS}
xgb_oof = {'x': np.zeros(len(train_last)), 'y': np.zeros(len(train_last))}

stat_model_cv_scores = {name: [] for name, _, _, _, _, _ in STAT_MODEL_CONFIGS}
xgb_cv_scores = []

train_last_reset = train_last.reset_index(drop=True)

for fold, (train_idx, val_idx) in enumerate(gkf.split(train_last_reset, groups=train_last_reset['game_id']), 1):
    print(f"\n  Fold {fold}/{n_splits}...")

    fold_train = train_last_reset.iloc[train_idx]
    fold_val = train_last_reset.iloc[val_idx].copy()

    # 통계 모델들
    for name, n_x, n_y, dir_type, pass_dist_cond, min_samp in STAT_MODEL_CONFIGS:
        model = build_stat_model(fold_train, n_x, n_y, dir_type, pass_dist_cond, min_samp)

        predictions = fold_val.apply(lambda r: predict_with_stat_model(r, model), axis=1)
        pred_x = predictions.apply(lambda x: x[0])
        pred_y = predictions.apply(lambda x: x[1])

        stat_model_oof[name]['x'][val_idx] = pred_x.values
        stat_model_oof[name]['y'][val_idx] = pred_y.values

        dist = np.sqrt((pred_x - fold_val['end_x'])**2 + (pred_y - fold_val['end_y'])**2)
        stat_model_cv_scores[name].append(dist.mean())

    # XGBoost 모델
    if XGBOOST_AVAILABLE:
        xgb_model_x = build_xgboost_model(fold_train, 'delta_x')
        xgb_model_y = build_xgboost_model(fold_train, 'delta_y')

        pred_x, pred_y = predict_with_xgboost(fold_val, xgb_model_x, xgb_model_y)

        if pred_x is not None:
            xgb_oof['x'][val_idx] = pred_x
            xgb_oof['y'][val_idx] = pred_y

            dist = np.sqrt((pred_x - fold_val['end_x'].values)**2 + (pred_y - fold_val['end_y'].values)**2)
            xgb_cv_scores.append(dist.mean())

    if fold == 1:
        print(f"    통계 모델 예측 완료: {len(STAT_MODEL_CONFIGS)}개")
        if XGBOOST_AVAILABLE and len(xgb_cv_scores) > 0:
            print(f"    XGBoost 예측 완료")

# =============================================================================
# 8. 개별 모델 성능 분석
# =============================================================================
print("\n[8] 개별 모델 성능 분석...")

stat_model_mean_cv = {}
for name in stat_model_cv_scores:
    mean_cv = np.mean(stat_model_cv_scores[name])
    std_cv = np.std(stat_model_cv_scores[name])
    stat_model_mean_cv[name] = mean_cv

# 상위 10개 모델 출력
sorted_stat_models = sorted(stat_model_mean_cv.items(), key=lambda x: x[1])
print("\n  Top 10 통계 모델:")
for i, (name, cv) in enumerate(sorted_stat_models[:10], 1):
    print(f"    {i}. {name}: {cv:.4f}")

if XGBOOST_AVAILABLE and len(xgb_cv_scores) > 0:
    xgb_mean_cv = np.mean(xgb_cv_scores)
    xgb_std_cv = np.std(xgb_cv_scores)
    print(f"\n  XGBoost: {xgb_mean_cv:.4f} ± {xgb_std_cv:.4f}")

# =============================================================================
# 9. Ensemble Weight Optimization (Greedy + Grid Search)
# =============================================================================
print("\n[9] 앙상블 가중치 최적화...")

# 상위 12개 통계 모델 선택
top_stat_models = [name for name, _ in sorted_stat_models[:12]]

# Greedy Forward Selection
print("\n  Greedy Forward Selection...")
selected_models = []
selected_weights = []
current_best_cv = float('inf')

# 1단계: 최고 성능 모델로 시작
best_single_model = top_stat_models[0]
selected_models.append(best_single_model)
selected_weights.append(1.0)
current_best_cv = stat_model_mean_cv[best_single_model]
print(f"    Start: {best_single_model} (CV: {current_best_cv:.4f})")

# 2단계: 하나씩 추가하며 최적화
for step in range(1, min(8, len(top_stat_models))):
    best_addition = None
    best_addition_weights = None
    best_addition_cv = current_best_cv

    for candidate in top_stat_models:
        if candidate in selected_models:
            continue

        # 가중치 탐색 (0.05 ~ 0.35)
        for new_weight in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]:
            test_models = selected_models + [candidate]
            test_weights = [w * (1 - new_weight) for w in selected_weights] + [new_weight]

            # 앙상블 예측
            ens_x = sum(w * stat_model_oof[m]['x'] for w, m in zip(test_weights, test_models))
            ens_y = sum(w * stat_model_oof[m]['y'] for w, m in zip(test_weights, test_models))

            dist = np.sqrt((ens_x - train_last_reset['end_x'].values)**2 +
                          (ens_y - train_last_reset['end_y'].values)**2)
            cv = dist.mean()

            if cv < best_addition_cv:
                best_addition_cv = cv
                best_addition = candidate
                best_addition_weights = test_weights

    if best_addition is not None and best_addition_cv < current_best_cv:
        selected_models.append(best_addition)
        selected_weights = best_addition_weights
        current_best_cv = best_addition_cv
        print(f"    Step {step}: +{best_addition} (CV: {current_best_cv:.4f})")
    else:
        print(f"    Step {step}: 개선 없음, 중단")
        break

print(f"\n  Greedy Selection 결과:")
print(f"    선택된 모델: {len(selected_models)}개")
print(f"    최종 CV: {current_best_cv:.4f}")

final_ensemble_cv = current_best_cv
final_ensemble_models = selected_models
final_ensemble_weights = selected_weights

# XGBoost 추가 시도 (가중치 15% 이하)
if XGBOOST_AVAILABLE and len(xgb_cv_scores) > 0 and xgb_mean_cv < 17.0:
    print("\n  XGBoost 추가 시도...")

    best_xgb_weight = 0
    best_cv_with_xgb = final_ensemble_cv

    for xgb_weight in [0.05, 0.10, 0.15]:
        adjusted_weights = [w * (1 - xgb_weight) for w in final_ensemble_weights]

        ens_x = (sum(w * stat_model_oof[m]['x'] for w, m in zip(adjusted_weights, final_ensemble_models)) +
                 xgb_weight * xgb_oof['x'])
        ens_y = (sum(w * stat_model_oof[m]['y'] for w, m in zip(adjusted_weights, final_ensemble_models)) +
                 xgb_weight * xgb_oof['y'])

        dist = np.sqrt((ens_x - train_last_reset['end_x'].values)**2 +
                      (ens_y - train_last_reset['end_y'].values)**2)
        cv = dist.mean()

        if cv < best_cv_with_xgb:
            best_cv_with_xgb = cv
            best_xgb_weight = xgb_weight

    if best_xgb_weight > 0 and best_cv_with_xgb < final_ensemble_cv:
        print(f"    XGBoost 추가 성공! (가중치: {best_xgb_weight:.2f}, CV: {best_cv_with_xgb:.4f})")
        final_ensemble_weights = [w * (1 - best_xgb_weight) for w in final_ensemble_weights]
        final_ensemble_models.append('xgboost')
        final_ensemble_weights.append(best_xgb_weight)
        final_ensemble_cv = best_cv_with_xgb
    else:
        print(f"    XGBoost 추가 시 개선 없음")

# =============================================================================
# 10. 최종 CV 검증
# =============================================================================
print("\n[10] 최종 앙상블 CV 검증...")

final_cv_scores = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(train_last_reset, groups=train_last_reset['game_id']), 1):
    fold_train = train_last_reset.iloc[train_idx]
    fold_val = train_last_reset.iloc[val_idx].copy()

    # 통계 모델 예측
    stat_preds = {}
    for model_name in [m for m in final_ensemble_models if m != 'xgboost']:
        # 모델 설정 찾기
        model_config = next((c for c in STAT_MODEL_CONFIGS if c[0] == model_name), None)
        if model_config:
            _, n_x, n_y, dir_type, pass_dist_cond, min_samp = model_config
            model = build_stat_model(fold_train, n_x, n_y, dir_type, pass_dist_cond, min_samp)

            predictions = fold_val.apply(lambda r: predict_with_stat_model(r, model), axis=1)
            stat_preds[model_name] = {
                'x': predictions.apply(lambda x: x[0]).values,
                'y': predictions.apply(lambda x: x[1]).values
            }

    # XGBoost 예측
    xgb_preds = None
    if 'xgboost' in final_ensemble_models and XGBOOST_AVAILABLE:
        xgb_model_x = build_xgboost_model(fold_train, 'delta_x')
        xgb_model_y = build_xgboost_model(fold_train, 'delta_y')
        pred_x, pred_y = predict_with_xgboost(fold_val, xgb_model_x, xgb_model_y)
        if pred_x is not None:
            xgb_preds = {'x': pred_x, 'y': pred_y}

    # 앙상블
    ens_x = np.zeros(len(fold_val))
    ens_y = np.zeros(len(fold_val))

    for model_name, weight in zip(final_ensemble_models, final_ensemble_weights):
        if model_name == 'xgboost' and xgb_preds is not None:
            ens_x += weight * xgb_preds['x']
            ens_y += weight * xgb_preds['y']
        elif model_name in stat_preds:
            ens_x += weight * stat_preds[model_name]['x']
            ens_y += weight * stat_preds[model_name]['y']

    dist = np.sqrt((ens_x - fold_val['end_x'].values)**2 + (ens_y - fold_val['end_y'].values)**2)
    final_cv_scores.append(dist.mean())
    print(f"  Fold {fold}: {dist.mean():.4f}")

final_cv = np.mean(final_cv_scores)
final_std = np.std(final_cv_scores)

print(f"\n최종 앙상블 CV: {final_cv:.4f} ± {final_std:.4f}")

# =============================================================================
# 11. 과적합 위험 평가 및 제출 권장
# =============================================================================
print("\n[11] 과적합 위험 평가...")

if final_cv < 16.0:
    expected_gap = 0.30
    risk_level = "중간 (CV < 16.0 달성!)"
    recommendation = "제출 권장 (목표 달성)"
    color = "GREEN"
elif final_cv < 16.2:
    expected_gap = 0.25
    risk_level = "낮음 (안전 구간)"
    recommendation = "제출 권장"
    color = "GREEN"
else:
    expected_gap = 0.20
    risk_level = "매우 낮음"
    recommendation = "현재 Best와 비교 필요"
    color = "YELLOW"

expected_public = final_cv + expected_gap

print(f"\n  CV Score: {final_cv:.4f}")
print(f"  과적합 위험: {risk_level}")
print(f"  예상 Gap: +{expected_gap:.2f}")
print(f"  예상 Public: {expected_public:.2f}")
print(f"  제출 권장: {recommendation}")

# =============================================================================
# 12. Test 예측 및 제출 파일 생성
# =============================================================================
print("\n[12] Test 예측 및 제출 파일 생성...")

# 전체 train으로 최종 모델 학습
final_stat_models = {}
for model_name in [m for m in final_ensemble_models if m != 'xgboost']:
    model_config = next((c for c in STAT_MODEL_CONFIGS if c[0] == model_name), None)
    if model_config:
        _, n_x, n_y, dir_type, pass_dist_cond, min_samp = model_config
        final_stat_models[model_name] = build_stat_model(train_last, n_x, n_y, dir_type, pass_dist_cond, min_samp)

# Test 예측
test_stat_preds = {}
for model_name, model in final_stat_models.items():
    predictions = test_last.apply(lambda r: predict_with_stat_model(r, model), axis=1)
    test_stat_preds[model_name] = {
        'x': predictions.apply(lambda x: x[0]).values,
        'y': predictions.apply(lambda x: x[1]).values
    }

# XGBoost Test 예측
test_xgb_preds = None
if 'xgboost' in final_ensemble_models and XGBOOST_AVAILABLE:
    xgb_model_x = build_xgboost_model(train_last, 'delta_x')
    xgb_model_y = build_xgboost_model(train_last, 'delta_y')
    pred_x, pred_y = predict_with_xgboost(test_last, xgb_model_x, xgb_model_y)
    if pred_x is not None:
        test_xgb_preds = {'x': pred_x, 'y': pred_y}

# 최종 앙상블
final_pred_x = np.zeros(len(test_last))
final_pred_y = np.zeros(len(test_last))

for model_name, weight in zip(final_ensemble_models, final_ensemble_weights):
    if model_name == 'xgboost' and test_xgb_preds is not None:
        final_pred_x += weight * test_xgb_preds['x']
        final_pred_y += weight * test_xgb_preds['y']
    elif model_name in test_stat_preds:
        final_pred_x += weight * test_stat_preds[model_name]['x']
        final_pred_y += weight * test_stat_preds[model_name]['y']

# 제출 파일 생성
submission = sample_sub.copy()
submission['end_x'] = final_pred_x
submission['end_y'] = final_pred_y

output_path = DATA_DIR / "submission_ultra_ensemble.csv"
submission.to_csv(output_path, index=False)
print(f"\n제출 파일 저장: {output_path}")

# =============================================================================
# 13. 최종 요약 및 제출 권장사항
# =============================================================================
print("\n" + "=" * 80)
print("최종 요약 및 제출 권장사항")
print("=" * 80)

print(f"\n[모델 구성]")
print(f"  통계 모델: {len([m for m in final_ensemble_models if m != 'xgboost'])}개")
if 'xgboost' in final_ensemble_models:
    xgb_idx = final_ensemble_models.index('xgboost')
    print(f"  XGBoost: 포함 (가중치 {final_ensemble_weights[xgb_idx]:.2f})")
else:
    print(f"  XGBoost: 미포함")

print(f"\n[모델 가중치]")
for model_name, weight in zip(final_ensemble_models, final_ensemble_weights):
    print(f"  {model_name}: {weight:.3f}")

print(f"\n[성능 지표]")
print(f"  최종 CV: {final_cv:.4f} ± {final_std:.4f}")
print(f"  예상 Public: {expected_public:.2f}")
print(f"  예상 Gap: +{expected_gap:.2f}")

print(f"\n[기준 모델과 비교]")
print(f"  현재 Best: CV 16.04 → Public 16.3502")
print(f"  신규 모델: CV {final_cv:.4f} → Public {expected_public:.2f} (예상)")
print(f"  CV 개선: {16.04 - final_cv:+.4f}")
print(f"  Public 개선: {16.3502 - expected_public:+.4f} (예상)")

print(f"\n[제출 권장]")
if final_cv < 15.95:
    print(f"  [SUCCESS] CV < 15.95 달성! 목표 달성!")
    print(f"  즉시 제출 강력 권장")
elif final_cv < 16.0:
    print(f"  [SUCCESS] CV < 16.0 달성!")
    print(f"  제출 권장 (Gap 관리 필요)")
elif final_cv < current_best_cv:
    print(f"  현재 Best CV보다 개선")
    print(f"  제출 권장")
else:
    print(f"  개선 미달")
    print(f"  추가 튜닝 필요")

print(f"\n[제출 파일]")
print(f"  {output_path}")

print("\n" + "=" * 80)
print("완료!")
print("=" * 80)
