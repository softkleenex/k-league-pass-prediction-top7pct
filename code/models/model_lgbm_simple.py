"""
K리그 패스 좌표 예측 - LightGBM Simple (과적합 방지)
핵심 피처만 사용 + 강한 정규화
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(".")

print("=" * 60)
print("K리그 패스 좌표 예측 - LightGBM Simple (과적합 방지)")
print("=" * 60)

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

print(f"Train: {len(train_df):,} rows")
print(f"Test: {len(test_all):,} rows")

# =============================================================================
# 2. 단순화된 피처 (핵심만!)
# =============================================================================
print("\n[2] 단순화된 피처 엔지니어링...")

def create_simple_features(df, is_train=True):
    """핵심 피처만 사용"""
    features_list = []

    for game_ep, group in df.groupby('game_episode'):
        group = group.sort_values('action_id')
        last_row = group.iloc[-1]

        feat = {'game_episode': game_ep}

        # ===== 핵심 피처 1: 마지막 패스 시작점 =====
        feat['last_start_x'] = last_row['start_x']
        feat['last_start_y'] = last_row['start_y']

        # ===== 핵심 피처 2: 구역 (단순) =====
        x, y = feat['last_start_x'], feat['last_start_y']
        feat['zone_x'] = 0 if x < 35 else (1 if x < 70 else 2)
        feat['zone_y'] = 0 if y < 22.67 else (1 if y < 45.33 else 2)

        # ===== 핵심 피처 3: 골문 거리 =====
        feat['dist_to_goal'] = np.sqrt((105 - x)**2 + (34 - y)**2)

        # ===== 핵심 피처 4: 시퀀스 길이 =====
        feat['seq_length'] = len(group)

        # ===== 핵심 피처 5: 이전 액션 (최근 2개만) =====
        if len(group) >= 2:
            prev = group.iloc[-2]
            feat['prev_end_x'] = prev['end_x'] if pd.notna(prev['end_x']) else prev['start_x']
            feat['prev_end_y'] = prev['end_y'] if pd.notna(prev['end_y']) else prev['start_y']
        else:
            feat['prev_end_x'] = feat['last_start_x']
            feat['prev_end_y'] = feat['last_start_y']

        # ===== 핵심 피처 6: 평균 이동 방향 =====
        pass_actions = group[group['type_name'].str.contains('Pass', na=False)]
        valid_passes = pass_actions.dropna(subset=['end_x', 'end_y'])
        if len(valid_passes) > 0:
            feat['avg_pass_dx'] = (valid_passes['end_x'] - valid_passes['start_x']).mean()
            feat['avg_pass_dy'] = (valid_passes['end_y'] - valid_passes['start_y']).mean()
        else:
            feat['avg_pass_dx'] = 13.53  # 전체 평균
            feat['avg_pass_dy'] = 0.01

        # ===== 핵심 피처 7: 시간/경기 상태 =====
        feat['period'] = last_row['period_id']
        feat['is_home'] = int(last_row['is_home'])

        # ===== 타겟 =====
        if is_train:
            feat['end_x'] = last_row['end_x']
            feat['end_y'] = last_row['end_y']

        features_list.append(feat)

    return pd.DataFrame(features_list)

train_features = create_simple_features(train_df, is_train=True)
test_features = create_simple_features(test_all, is_train=False)

train_features = train_features.dropna(subset=['end_x', 'end_y'])

print(f"Train features: {train_features.shape}")
print(f"Test features: {test_features.shape}")

# =============================================================================
# 3. 모델 학습
# =============================================================================
print("\n[3] 모델 학습 (강한 정규화)...")

feature_cols = [c for c in train_features.columns if c not in ['game_episode', 'end_x', 'end_y']]
X = train_features[feature_cols].values
y_x = train_features['end_x'].values
y_y = train_features['end_y'].values

X_test = test_features[feature_cols].values

print(f"Feature columns ({len(feature_cols)}): {feature_cols}")

# 강한 정규화 파라미터
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 15,          # 작게
    'max_depth': 4,            # 얕게
    'min_child_samples': 100,  # 크게
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'reg_alpha': 1.0,          # 강한 정규화
    'reg_lambda': 1.0,         # 강한 정규화
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42
}

N_FOLDS = 5
gkf = GroupKFold(n_splits=N_FOLDS)

# game_episode에서 game_id 추출
train_features['game_id'] = train_features['game_episode'].apply(lambda x: int(x.split('_')[0]))
game_ids = train_features['game_id'].values

oof_pred_x = np.zeros(len(X))
oof_pred_y = np.zeros(len(X))
test_pred_x = np.zeros(len(X_test))
test_pred_y = np.zeros(len(X_test))

fold_scores = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, groups=game_ids)):
    print(f"\n  Fold {fold+1}/{N_FOLDS}")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train_x, y_val_x = y_x[train_idx], y_x[val_idx]
    y_train_y, y_val_y = y_y[train_idx], y_y[val_idx]

    # end_x 모델
    model_x = lgb.train(
        params,
        lgb.Dataset(X_train, label=y_train_x),
        num_boost_round=500,
        valid_sets=[lgb.Dataset(X_val, label=y_val_x)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )

    # end_y 모델
    model_y = lgb.train(
        params,
        lgb.Dataset(X_train, label=y_train_y),
        num_boost_round=500,
        valid_sets=[lgb.Dataset(X_val, label=y_val_y)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )

    oof_pred_x[val_idx] = model_x.predict(X_val)
    oof_pred_y[val_idx] = model_y.predict(X_val)

    test_pred_x += model_x.predict(X_test) / N_FOLDS
    test_pred_y += model_y.predict(X_test) / N_FOLDS

    fold_dist = np.sqrt((oof_pred_x[val_idx] - y_val_x)**2 + (oof_pred_y[val_idx] - y_val_y)**2)
    fold_score = fold_dist.mean()
    fold_scores.append(fold_score)
    print(f"    Fold {fold+1} Score: {fold_score:.4f}")

cv_dist = np.sqrt((oof_pred_x - y_x)**2 + (oof_pred_y - y_y)**2)
cv_score = cv_dist.mean()
print(f"\n  CV Score: {cv_score:.4f} (std: {np.std(fold_scores):.4f})")

# =============================================================================
# 4. 제출 파일 생성
# =============================================================================
print("\n[4] 제출 파일 생성...")

test_pred_x = np.clip(test_pred_x, 0, 105)
test_pred_y = np.clip(test_pred_y, 0, 68)

submission = pd.DataFrame({
    'game_episode': test_features['game_episode'],
    'end_x': test_pred_x,
    'end_y': test_pred_y
})

submission = sample_sub[['game_episode']].merge(submission, on='game_episode', how='left')
submission.to_csv('submission_simple.csv', index=False)

print(f"제출 파일: submission_simple.csv")
print(f"CV Score: {cv_score:.4f}")
print(submission.head())

# =============================================================================
# 5. 베이스라인과 앙상블
# =============================================================================
print("\n[5] 베이스라인과 앙상블...")

# 구역별 평균 이동량 (베이스라인)
last_actions = train_df.groupby('game_episode').last().reset_index()
last_actions['delta_x'] = last_actions['end_x'] - last_actions['start_x']
last_actions['delta_y'] = last_actions['end_y'] - last_actions['start_y']

def get_zone(x, y):
    x_zone = 0 if x < 35 else (1 if x < 70 else 2)
    y_zone = 0 if y < 22.67 else (1 if y < 45.33 else 2)
    return x_zone * 3 + y_zone

last_actions['zone'] = last_actions.apply(lambda r: get_zone(r['start_x'], r['start_y']), axis=1)
zone_stats = last_actions.groupby('zone').agg({'delta_x': 'mean', 'delta_y': 'mean'}).to_dict()

mean_dx = last_actions['delta_x'].mean()
mean_dy = last_actions['delta_y'].mean()

# 테스트 데이터에 베이스라인 적용
test_last = test_all.groupby('game_episode').last().reset_index()
baseline_preds = []

for _, row in test_features.iterrows():
    game_ep = row['game_episode']
    test_row = test_last[test_last['game_episode'] == game_ep]

    if len(test_row) > 0:
        sx = test_row['start_x'].values[0]
        sy = test_row['start_y'].values[0]
        zone = get_zone(sx, sy)
        dx = zone_stats['delta_x'].get(zone, mean_dx)
        dy = zone_stats['delta_y'].get(zone, mean_dy)
        baseline_preds.append([np.clip(sx + dx, 0, 105), np.clip(sy + dy, 0, 68)])
    else:
        baseline_preds.append([68.45, 33.62])

baseline_preds = np.array(baseline_preds)

# 앙상블: LightGBM 70% + 베이스라인 30%
ensemble_x = 0.7 * test_pred_x + 0.3 * baseline_preds[:, 0]
ensemble_y = 0.7 * test_pred_y + 0.3 * baseline_preds[:, 1]

submission_ensemble = pd.DataFrame({
    'game_episode': test_features['game_episode'],
    'end_x': ensemble_x,
    'end_y': ensemble_y
})
submission_ensemble = sample_sub[['game_episode']].merge(submission_ensemble, on='game_episode', how='left')
submission_ensemble.to_csv('submission_ensemble.csv', index=False)

print(f"앙상블 제출 파일: submission_ensemble.csv")

print("\n" + "=" * 60)
print("완료!")
print("=" * 60)
