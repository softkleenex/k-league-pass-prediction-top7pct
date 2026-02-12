"""
K리그 패스 좌표 예측 - LightGBM v2 (피처 강화)
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
print("K리그 패스 좌표 예측 - LightGBM v2 (피처 강화)")
print("=" * 60)

# =============================================================================
# 1. 데이터 로드
# =============================================================================
print("\n[1] 데이터 로드...")
train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")
sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
match_info = pd.read_csv(DATA_DIR / "match_info.csv")

# 테스트 에피소드 로드
test_episodes = []
for _, row in test_df.iterrows():
    ep_df = pd.read_csv(DATA_DIR / row['path'])
    test_episodes.append(ep_df)
test_all = pd.concat(test_episodes, ignore_index=True)

print(f"Train: {len(train_df):,} rows")
print(f"Test: {len(test_all):,} rows")

# =============================================================================
# 2. 강화된 피처 엔지니어링
# =============================================================================
print("\n[2] 강화된 피처 엔지니어링...")

def create_features_v2(df, is_train=True):
    """강화된 에피소드별 피처 생성"""
    features_list = []

    for game_ep, group in df.groupby('game_episode'):
        group = group.sort_values('action_id')
        last_row = group.iloc[-1]

        feat = {
            'game_episode': game_ep,
            'game_id': last_row['game_id'],
        }

        # ===== 마지막 액션 피처 =====
        feat['last_start_x'] = last_row['start_x']
        feat['last_start_y'] = last_row['start_y']
        feat['last_time'] = last_row['time_seconds']
        feat['last_period'] = last_row['period_id']
        feat['last_is_home'] = int(last_row['is_home'])
        feat['last_team_id'] = last_row['team_id']
        feat['last_player_id'] = last_row['player_id']
        feat['last_type'] = last_row['type_name']

        # ===== 시퀀스 길이 피처 =====
        feat['seq_length'] = len(group)
        feat['seq_length_log'] = np.log1p(len(group))

        # ===== 액션 유형별 카운트 =====
        type_counts = group['type_name'].value_counts()
        for t in ['Pass', 'Carry', 'Duel', 'Recovery', 'Interception', 'Tackle', 'Cross', 'Shot']:
            feat[f'n_{t.lower()}'] = type_counts.get(t, 0)

        # 비율
        total_actions = len(group)
        feat['pass_ratio'] = feat['n_pass'] / total_actions
        feat['carry_ratio'] = feat['n_carry'] / total_actions

        # ===== 결과 통계 =====
        result_counts = group['result_name'].value_counts()
        feat['n_successful'] = result_counts.get('Successful', 0)
        feat['n_unsuccessful'] = result_counts.get('Unsuccessful', 0)
        feat['success_ratio'] = feat['n_successful'] / max(1, feat['n_successful'] + feat['n_unsuccessful'])

        # ===== 좌표 통계 =====
        feat['mean_x'] = group['start_x'].mean()
        feat['mean_y'] = group['start_y'].mean()
        feat['std_x'] = group['start_x'].std() if len(group) > 1 else 0
        feat['std_y'] = group['start_y'].std() if len(group) > 1 else 0
        feat['min_x'] = group['start_x'].min()
        feat['max_x'] = group['start_x'].max()
        feat['min_y'] = group['start_y'].min()
        feat['max_y'] = group['start_y'].max()
        feat['range_x'] = feat['max_x'] - feat['min_x']
        feat['range_y'] = feat['max_y'] - feat['min_y']

        # ===== 진행 방향 =====
        if len(group) > 1:
            feat['x_progression'] = group['start_x'].iloc[-1] - group['start_x'].iloc[0]
            feat['y_progression'] = group['start_y'].iloc[-1] - group['start_y'].iloc[0]
            feat['total_distance'] = np.sqrt(feat['x_progression']**2 + feat['y_progression']**2)
        else:
            feat['x_progression'] = 0
            feat['y_progression'] = 0
            feat['total_distance'] = 0

        # ===== 시간 피처 =====
        feat['duration'] = group['time_seconds'].max() - group['time_seconds'].min()
        feat['actions_per_second'] = len(group) / max(1, feat['duration'])

        # ===== 이전 액션 피처 (최근 5개) =====
        for offset in range(2, 7):
            if len(group) >= offset:
                prev_row = group.iloc[-offset]
                feat[f'prev{offset}_start_x'] = prev_row['start_x']
                feat[f'prev{offset}_start_y'] = prev_row['start_y']
                feat[f'prev{offset}_end_x'] = prev_row['end_x'] if pd.notna(prev_row['end_x']) else prev_row['start_x']
                feat[f'prev{offset}_end_y'] = prev_row['end_y'] if pd.notna(prev_row['end_y']) else prev_row['start_y']

                # 이동량
                if pd.notna(prev_row['end_x']) and pd.notna(prev_row['end_y']):
                    feat[f'prev{offset}_dx'] = prev_row['end_x'] - prev_row['start_x']
                    feat[f'prev{offset}_dy'] = prev_row['end_y'] - prev_row['start_y']
                else:
                    feat[f'prev{offset}_dx'] = 0
                    feat[f'prev{offset}_dy'] = 0
            else:
                feat[f'prev{offset}_start_x'] = feat['last_start_x']
                feat[f'prev{offset}_start_y'] = feat['last_start_y']
                feat[f'prev{offset}_end_x'] = feat['last_start_x']
                feat[f'prev{offset}_end_y'] = feat['last_start_y']
                feat[f'prev{offset}_dx'] = 0
                feat[f'prev{offset}_dy'] = 0

        # ===== 평균 이동량 (최근 3개 Pass) =====
        pass_actions = group[group['type_name'].str.contains('Pass|Cross', na=False)]
        if len(pass_actions) >= 2:
            recent_passes = pass_actions.tail(3)
            recent_passes = recent_passes.dropna(subset=['end_x', 'end_y'])
            if len(recent_passes) > 0:
                feat['recent_pass_dx'] = (recent_passes['end_x'] - recent_passes['start_x']).mean()
                feat['recent_pass_dy'] = (recent_passes['end_y'] - recent_passes['start_y']).mean()
                feat['recent_pass_dist'] = np.sqrt(
                    (recent_passes['end_x'] - recent_passes['start_x'])**2 +
                    (recent_passes['end_y'] - recent_passes['start_y'])**2
                ).mean()
            else:
                feat['recent_pass_dx'] = 13.53  # 전체 평균
                feat['recent_pass_dy'] = 0.01
                feat['recent_pass_dist'] = 20.37
        else:
            feat['recent_pass_dx'] = 13.53
            feat['recent_pass_dy'] = 0.01
            feat['recent_pass_dist'] = 20.37

        # ===== 구역 피처 =====
        x, y = feat['last_start_x'], feat['last_start_y']

        # 9구역
        feat['start_zone_x'] = 0 if x < 35 else (1 if x < 70 else 2)
        feat['start_zone_y'] = 0 if y < 22.67 else (1 if y < 45.33 else 2)
        feat['start_zone'] = feat['start_zone_x'] * 3 + feat['start_zone_y']

        # 더 세밀한 구역 (16구역)
        feat['zone_x_fine'] = int(x // 26.25)  # 0-3
        feat['zone_y_fine'] = int(y // 17)     # 0-3

        # 공격/수비 지역
        feat['attacking_third'] = 1 if x > 70 else 0
        feat['defensive_third'] = 1 if x < 35 else 0
        feat['middle_third'] = 1 if 35 <= x <= 70 else 0

        # 측면/중앙
        feat['left_side'] = 1 if y < 22.67 else 0
        feat['right_side'] = 1 if y > 45.33 else 0
        feat['center'] = 1 if 22.67 <= y <= 45.33 else 0

        # ===== 거리 피처 =====
        feat['dist_to_goal'] = np.sqrt((105 - x)**2 + (34 - y)**2)
        feat['dist_to_own_goal'] = np.sqrt(x**2 + (34 - y)**2)
        feat['dist_to_center'] = np.sqrt((52.5 - x)**2 + (34 - y)**2)

        # ===== 각도 피처 =====
        feat['angle_to_goal'] = np.arctan2(34 - y, 105 - x)

        # ===== 팀 상태 (최근 액션) =====
        last_team = last_row['team_id']
        team_actions = group[group['team_id'] == last_team]
        feat['team_possession_ratio'] = len(team_actions) / len(group)

        # ===== 타겟 (Train만) =====
        if is_train:
            feat['end_x'] = last_row['end_x']
            feat['end_y'] = last_row['end_y']

        features_list.append(feat)

    return pd.DataFrame(features_list)

# 피처 생성
train_features = create_features_v2(train_df, is_train=True)
test_features = create_features_v2(test_all, is_train=False)

print(f"Train features: {train_features.shape}")
print(f"Test features: {test_features.shape}")

# 결측치 제거
train_features = train_features.dropna(subset=['end_x', 'end_y'])
print(f"Train features (after dropna): {train_features.shape}")

# =============================================================================
# 3. 피처 준비
# =============================================================================
print("\n[3] 피처 준비...")

# 범주형 변수 인코딩
cat_cols = ['last_team_id', 'last_player_id', 'game_id', 'last_type']
for col in cat_cols:
    le = LabelEncoder()
    all_values = pd.concat([train_features[col], test_features[col]]).astype(str)
    le.fit(all_values)
    train_features[col] = le.transform(train_features[col].astype(str))
    test_features[col] = le.transform(test_features[col].astype(str))

# 피처 컬럼
exclude_cols = ['game_episode', 'end_x', 'end_y']
feature_cols = [c for c in train_features.columns if c not in exclude_cols]

X = train_features[feature_cols].values
y_x = train_features['end_x'].values
y_y = train_features['end_y'].values
game_ids = train_features['game_id'].values

X_test = test_features[feature_cols].values

print(f"Feature columns: {len(feature_cols)}")

# =============================================================================
# 4. 모델 학습 (GroupKFold)
# =============================================================================
print("\n[4] 모델 학습 (5-Fold CV)...")

N_FOLDS = 5
gkf = GroupKFold(n_splits=N_FOLDS)

# 최적화된 LightGBM 파라미터
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.03,
    'num_leaves': 63,
    'max_depth': 8,
    'min_child_samples': 30,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'bagging_freq': 5,
    'reg_alpha': 0.5,
    'reg_lambda': 0.5,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42
}

# CV 학습
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
    train_data_x = lgb.Dataset(X_train, label=y_train_x)
    val_data_x = lgb.Dataset(X_val, label=y_val_x, reference=train_data_x)

    model_x = lgb.train(
        params,
        train_data_x,
        num_boost_round=2000,
        valid_sets=[val_data_x],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )

    # end_y 모델
    train_data_y = lgb.Dataset(X_train, label=y_train_y)
    val_data_y = lgb.Dataset(X_val, label=y_val_y, reference=train_data_y)

    model_y = lgb.train(
        params,
        train_data_y,
        num_boost_round=2000,
        valid_sets=[val_data_y],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )

    # OOF 예측
    oof_pred_x[val_idx] = model_x.predict(X_val)
    oof_pred_y[val_idx] = model_y.predict(X_val)

    # 테스트 예측
    test_pred_x += model_x.predict(X_test) / N_FOLDS
    test_pred_y += model_y.predict(X_test) / N_FOLDS

    # Fold 스코어
    fold_dist = np.sqrt((oof_pred_x[val_idx] - y_val_x)**2 + (oof_pred_y[val_idx] - y_val_y)**2)
    fold_score = fold_dist.mean()
    fold_scores.append(fold_score)
    print(f"    Fold {fold+1} Score: {fold_score:.4f}")

# 전체 CV 스코어
cv_dist = np.sqrt((oof_pred_x - y_x)**2 + (oof_pred_y - y_y)**2)
cv_score = cv_dist.mean()
print(f"\n  CV Score (Mean): {cv_score:.4f}")
print(f"  CV Score (Std): {np.std(fold_scores):.4f}")

# =============================================================================
# 5. 피처 중요도
# =============================================================================
print("\n[5] 피처 중요도 (Top 20)...")
importance = model_x.feature_importance(importance_type='gain')
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': importance
}).sort_values('importance', ascending=False)

print(importance_df.head(20).to_string(index=False))

# =============================================================================
# 6. 제출 파일 생성
# =============================================================================
print("\n[6] 제출 파일 생성...")

# 좌표 클리핑
test_pred_x = np.clip(test_pred_x, 0, 105)
test_pred_y = np.clip(test_pred_y, 0, 68)

# 제출 파일
submission = pd.DataFrame({
    'game_episode': test_features['game_episode'],
    'end_x': test_pred_x,
    'end_y': test_pred_y
})

submission = sample_sub[['game_episode']].merge(submission, on='game_episode', how='left')
submission.to_csv('submission_lgbm_v2.csv', index=False)

print(f"제출 파일 저장: submission_lgbm_v2.csv")
print(f"예상 스코어: ~{cv_score:.2f}")
print(submission.head(10))

print("\n" + "=" * 60)
print("완료!")
print("=" * 60)
