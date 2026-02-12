"""
K리그 패스 좌표 예측 - Regularized LightGBM + Zone 앙상블
과적합 방지를 위한 극도의 정규화 적용
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(".")

print("=" * 70)
print("K리그 패스 좌표 예측 - Regularized LightGBM + Zone 앙상블")
print("=" * 70)

# =============================================================================
# 1. 데이터 로드
# =============================================================================
print("\n[1] 데이터 로드...")
train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")
sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")

# 테스트 에피소드 로드
test_episodes = []
for _, row in test_df.iterrows():
    ep_df = pd.read_csv(DATA_DIR / row['path'])
    test_episodes.append(ep_df)
test_all = pd.concat(test_episodes, ignore_index=True)

print(f"Train rows: {len(train_df):,}")
print(f"Test rows: {len(test_all):,}")
print(f"Train episodes: {train_df['game_episode'].nunique():,}")
print(f"Test episodes: {test_all['game_episode'].nunique():,}")

# =============================================================================
# 2. 간소화된 피처 엔지니어링 (10개 핵심 피처만)
# =============================================================================
print("\n[2] 간소화된 피처 엔지니어링 (과적합 방지)...")

def create_minimal_features(df):
    """최소한의 핵심 피처만 생성 - 과적합 방지"""
    df = df.copy()

    # 기본 이동량
    df['dx'] = df['end_x'] - df['start_x']
    df['dy'] = df['end_y'] - df['start_y']
    df['distance'] = np.sqrt(df['dx']**2 + df['dy']**2)

    # 골문까지 거리/각도
    df['dist_to_goal'] = np.sqrt((105 - df['start_x'])**2 + (34 - df['start_y'])**2)
    df['angle_to_goal'] = np.arctan2(34 - df['start_y'], 105 - df['start_x'])

    # 직전 패스 정보만 (1개만!)
    for col in ['dx', 'dy', 'distance']:
        df[f'prev_1_{col}'] = df.groupby('game_episode')[col].shift(1)

    # 이동 방향
    df['move_angle'] = np.arctan2(df['dy'], df['dx'])
    df['prev_1_move_angle'] = df.groupby('game_episode')['move_angle'].shift(1)
    df['angle_change'] = df['move_angle'] - df['prev_1_move_angle']

    # 액션 순서 관련
    df['action_order'] = df.groupby('game_episode').cumcount()
    df['total_actions'] = df.groupby('game_episode')['action_id'].transform('count')

    df = df.fillna(0)
    return df

print("  Train 피처 생성 중...")
train_df = create_minimal_features(train_df)

print("  Test 피처 생성 중...")
test_all = create_minimal_features(test_all)

# =============================================================================
# 3. 학습 데이터 준비
# =============================================================================
print("\n[3] 학습 데이터 준비...")

train_last = train_df.groupby('game_episode').last().reset_index()
train_last = train_last.dropna(subset=['end_x', 'end_y'])
test_last = test_all.groupby('game_episode').last().reset_index()

print(f"Train last actions: {len(train_last):,}")
print(f"Test last actions: {len(test_last):,}")

# 핵심 피처만 선택 (10개)
feature_cols = [
    'start_x', 'start_y',           # 위치 (2)
    'dist_to_goal', 'angle_to_goal', # 골문 관련 (2)
    'prev_1_dx', 'prev_1_dy',       # 직전 패스 (2)
    'angle_change',                  # 방향 변화 (1)
    'action_order', 'total_actions', # 시퀀스 (2)
    'period_id',                     # 경기 정보 (1)
]

feature_cols = [c for c in feature_cols if c in train_last.columns]
print(f"사용할 피처 수: {len(feature_cols)}")
print(f"피처 목록: {feature_cols}")

X = train_last[feature_cols].values
y_x = train_last['end_x'].values
y_y = train_last['end_y'].values
game_ids = train_last['game_id'].values

X_test = test_last[feature_cols].values

# =============================================================================
# 4. Zone Baseline 준비
# =============================================================================
print("\n[4] Zone Baseline (6x6 median) 준비...")

def get_zone_6x6(x, y):
    x_zone = min(5, int(x / (105 / 6)))
    y_zone = min(5, int(y / (68 / 6)))
    return x_zone * 6 + y_zone

train_last['zone'] = train_last.apply(lambda r: get_zone_6x6(r['start_x'], r['start_y']), axis=1)
train_last['delta_x'] = train_last['end_x'] - train_last['start_x']
train_last['delta_y'] = train_last['end_y'] - train_last['start_y']

zone_stats = train_last.groupby('zone').agg({
    'delta_x': 'median',
    'delta_y': 'median'
}).to_dict()

# Zone baseline 예측 (Train)
zone_pred_x_train = []
zone_pred_y_train = []
for _, row in train_last.iterrows():
    zone = get_zone_6x6(row['start_x'], row['start_y'])
    dx = zone_stats['delta_x'].get(zone, 0)
    dy = zone_stats['delta_y'].get(zone, 0)
    zone_pred_x_train.append(np.clip(row['start_x'] + dx, 0, 105))
    zone_pred_y_train.append(np.clip(row['start_y'] + dy, 0, 68))

zone_pred_x_train = np.array(zone_pred_x_train)
zone_pred_y_train = np.array(zone_pred_y_train)

zone_dist = np.sqrt((zone_pred_x_train - y_x)**2 + (zone_pred_y_train - y_y)**2)
zone_score = zone_dist.mean()
print(f"Zone Baseline CV: {zone_score:.4f}")

# Zone baseline 예측 (Test)
test_last['zone'] = test_last.apply(lambda r: get_zone_6x6(r['start_x'], r['start_y']), axis=1)
zone_pred_x_test = []
zone_pred_y_test = []
for _, row in test_last.iterrows():
    zone = get_zone_6x6(row['start_x'], row['start_y'])
    dx = zone_stats['delta_x'].get(zone, 0)
    dy = zone_stats['delta_y'].get(zone, 0)
    zone_pred_x_test.append(np.clip(row['start_x'] + dx, 0, 105))
    zone_pred_y_test.append(np.clip(row['start_y'] + dy, 0, 68))

zone_pred_x_test = np.array(zone_pred_x_test)
zone_pred_y_test = np.array(zone_pred_y_test)

# =============================================================================
# 5. Regularized LightGBM
# =============================================================================
print("\n[5] Regularized LightGBM 학습...")

# 극도로 정규화된 파라미터
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 7,           # 극도로 낮음 (기본 31)
    'max_depth': 3,            # 극도로 낮음 (기본 -1)
    'learning_rate': 0.03,     # 낮음 (기본 0.1)
    'feature_fraction': 0.7,   # 피처 샘플링
    'bagging_fraction': 0.7,   # 데이터 샘플링
    'bagging_freq': 5,
    'reg_alpha': 1.0,          # L1 정규화 강화
    'reg_lambda': 1.0,         # L2 정규화 강화
    'min_child_samples': 100,  # 높음 (기본 20)
    'min_child_weight': 10,    # 추가 정규화
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42
}

n_splits = 5
gkf = GroupKFold(n_splits=n_splits)

oof_pred_x = np.zeros(len(X))
oof_pred_y = np.zeros(len(X))
test_pred_x = np.zeros(len(X_test))
test_pred_y = np.zeros(len(X_test))

fold_scores = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, groups=game_ids)):
    print(f"\n  Fold {fold+1}/{n_splits}")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train_x, y_val_x = y_x[train_idx], y_x[val_idx]
    y_train_y, y_val_y = y_y[train_idx], y_y[val_idx]

    # X 좌표 모델
    train_data_x = lgb.Dataset(X_train, label=y_train_x)
    val_data_x = lgb.Dataset(X_val, label=y_val_x, reference=train_data_x)

    model_x = lgb.train(
        params,
        train_data_x,
        num_boost_round=500,
        valid_sets=[val_data_x],
        callbacks=[
            lgb.early_stopping(20),  # 빠른 early stopping
            lgb.log_evaluation(0)
        ]
    )

    # Y 좌표 모델
    train_data_y = lgb.Dataset(X_train, label=y_train_y)
    val_data_y = lgb.Dataset(X_val, label=y_val_y, reference=train_data_y)

    model_y = lgb.train(
        params,
        train_data_y,
        num_boost_round=500,
        valid_sets=[val_data_y],
        callbacks=[
            lgb.early_stopping(20),
            lgb.log_evaluation(0)
        ]
    )

    # OOF 예측
    oof_pred_x[val_idx] = model_x.predict(X_val)
    oof_pred_y[val_idx] = model_y.predict(X_val)

    # 테스트 예측
    test_pred_x += model_x.predict(X_test) / n_splits
    test_pred_y += model_y.predict(X_test) / n_splits

    # Fold 점수
    fold_dist = np.sqrt((oof_pred_x[val_idx] - y_val_x)**2 + (oof_pred_y[val_idx] - y_val_y)**2)
    fold_score = fold_dist.mean()
    fold_scores.append(fold_score)
    print(f"    Fold {fold+1} Score: {fold_score:.4f}, Best iter: X={model_x.best_iteration}, Y={model_y.best_iteration}")

# 전체 OOF 점수
oof_dist = np.sqrt((oof_pred_x - y_x)**2 + (oof_pred_y - y_y)**2)
lgbm_score = oof_dist.mean()

print("\n" + "=" * 70)
print(f"Regularized LightGBM CV Score: {lgbm_score:.4f}")
print(f"Fold Scores: {[f'{s:.4f}' for s in fold_scores]}")
print(f"Std: {np.std(fold_scores):.4f}")
print("=" * 70)

# =============================================================================
# 6. 앙상블 최적화
# =============================================================================
print("\n[6] Zone + LightGBM 앙상블 최적화...")

best_alpha = None
best_ensemble_score = float('inf')
results = []

for alpha in [0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
    # 앙상블 예측 (Train)
    ensemble_x = alpha * zone_pred_x_train + (1 - alpha) * oof_pred_x
    ensemble_y = alpha * zone_pred_y_train + (1 - alpha) * oof_pred_y

    ensemble_dist = np.sqrt((ensemble_x - y_x)**2 + (ensemble_y - y_y)**2)
    ensemble_score = ensemble_dist.mean()

    results.append((alpha, ensemble_score))

    if ensemble_score < best_ensemble_score:
        best_ensemble_score = ensemble_score
        best_alpha = alpha

print("\n앙상블 가중치별 CV Score:")
for alpha, score in results:
    marker = " *** BEST" if alpha == best_alpha else ""
    print(f"  Zone {alpha:.0%} + LightGBM {1-alpha:.0%}: CV = {score:.4f}{marker}")

print(f"\n최적 가중치: Zone {best_alpha:.0%} + LightGBM {1-best_alpha:.0%}")
print(f"앙상블 CV Score: {best_ensemble_score:.4f}")

# =============================================================================
# 7. 제출 파일 생성
# =============================================================================
print("\n[7] 제출 파일 생성...")

# 1. Pure Regularized LightGBM
test_pred_x_clipped = np.clip(test_pred_x, 0, 105)
test_pred_y_clipped = np.clip(test_pred_y, 0, 68)

submission_lgbm = pd.DataFrame({
    'game_episode': test_last['game_episode'],
    'end_x': test_pred_x_clipped,
    'end_y': test_pred_y_clipped
})
submission_lgbm = sample_sub[['game_episode']].merge(submission_lgbm, on='game_episode', how='left')
submission_lgbm.to_csv('submission_regularized_lgbm.csv', index=False)
print(f"  1. submission_regularized_lgbm.csv 저장 (CV: {lgbm_score:.4f})")

# 2. 최적 앙상블
best_ensemble_x = best_alpha * zone_pred_x_test + (1 - best_alpha) * test_pred_x
best_ensemble_y = best_alpha * zone_pred_y_test + (1 - best_alpha) * test_pred_y

best_ensemble_x = np.clip(best_ensemble_x, 0, 105)
best_ensemble_y = np.clip(best_ensemble_y, 0, 68)

submission_ensemble = pd.DataFrame({
    'game_episode': test_last['game_episode'],
    'end_x': best_ensemble_x,
    'end_y': best_ensemble_y
})
submission_ensemble = sample_sub[['game_episode']].merge(submission_ensemble, on='game_episode', how='left')
submission_ensemble.to_csv('submission_ensemble_zone_lgbm_optimal.csv', index=False)
print(f"  2. submission_ensemble_zone_lgbm_optimal.csv 저장 (CV: {best_ensemble_score:.4f})")

# 3. 다양한 가중치 앙상블 생성
weight_configs = [
    (0.6, 0.4),
    (0.65, 0.35),
    (0.7, 0.3),
]

for zone_w, lgbm_w in weight_configs:
    ens_x = zone_w * zone_pred_x_test + lgbm_w * test_pred_x
    ens_y = zone_w * zone_pred_y_test + lgbm_w * test_pred_y
    ens_x = np.clip(ens_x, 0, 105)
    ens_y = np.clip(ens_y, 0, 68)

    sub = pd.DataFrame({
        'game_episode': test_last['game_episode'],
        'end_x': ens_x,
        'end_y': ens_y
    })
    sub = sample_sub[['game_episode']].merge(sub, on='game_episode', how='left')

    # Train CV 계산
    ens_train_x = zone_w * zone_pred_x_train + lgbm_w * oof_pred_x
    ens_train_y = zone_w * zone_pred_y_train + lgbm_w * oof_pred_y
    ens_dist = np.sqrt((ens_train_x - y_x)**2 + (ens_train_y - y_y)**2)
    cv = ens_dist.mean()

    filename = f'submission_ensemble_zone{int(zone_w*100)}_lgbm{int(lgbm_w*100)}.csv'
    sub.to_csv(filename, index=False)
    print(f"  3. {filename} 저장 (CV: {cv:.4f})")

# =============================================================================
# 8. 최종 요약
# =============================================================================
print("\n" + "=" * 70)
print("최종 요약")
print("=" * 70)

print(f"\n[모델 비교]")
print(f"  Zone Baseline (6x6 median):  CV = {zone_score:.4f} → Public = 16.85")
print(f"  Regularized LightGBM:        CV = {lgbm_score:.4f}")
print(f"  최적 앙상블 (Zone {best_alpha:.0%}):     CV = {best_ensemble_score:.4f}")

print(f"\n[과적합 위험도 분석]")
print(f"  이전 LightGBM v2: CV 8.55 → Public 20.57 (Gap +12.02)")
print(f"  현재 Regularized: CV {lgbm_score:.4f}")
if lgbm_score > 12:
    print(f"  → CV가 12 이상이므로 과적합 위험 낮음!")
else:
    print(f"  → CV가 12 미만이므로 과적합 가능성 있음, 주의 필요")

print(f"\n[예상 Public Score]")
gap_estimate = 0.5 if lgbm_score > 14 else (1.0 if lgbm_score > 12 else 3.0)
print(f"  Regularized LightGBM: {lgbm_score + gap_estimate:.2f} ~ {lgbm_score + gap_estimate + 1:.2f}")
print(f"  최적 앙상블:          {best_ensemble_score + 0.3:.2f} ~ {best_ensemble_score + 0.5:.2f}")

print(f"\n[제출 파일 목록]")
print(f"  1. submission_regularized_lgbm.csv")
print(f"  2. submission_ensemble_zone_lgbm_optimal.csv (권장)")
print(f"  3. submission_ensemble_zone60_lgbm40.csv")
print(f"  4. submission_ensemble_zone65_lgbm35.csv")
print(f"  5. submission_ensemble_zone70_lgbm30.csv")

print("\n" + "=" * 70)
print("완료!")
print("=" * 70)
