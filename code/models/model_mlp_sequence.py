"""
K리그 패스 좌표 예측 - MLP 시퀀스 모델 (sklearn 기반)
PyTorch 환경 문제로 인한 대안
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(".")

print("=" * 70)
print("K리그 패스 좌표 예측 - MLP 시퀀스 모델 (sklearn)")
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

print(f"Train episodes: {train_df['game_episode'].nunique():,}")
print(f"Test episodes: {test_all['game_episode'].nunique():,}")

# =============================================================================
# 2. 시퀀스 피처 집계 (마지막 N개 액션 요약)
# =============================================================================
print("\n[2] 시퀀스 피처 집계...")

def create_sequence_summary_features(df, lookback=5):
    """마지막 N개 액션을 요약하는 피처 생성"""
    df = df.copy()

    # 기본 이동량
    df['dx'] = df['end_x'] - df['start_x']
    df['dy'] = df['end_y'] - df['start_y']
    df['distance'] = np.sqrt(df['dx']**2 + df['dy']**2)

    # 골문 방향
    df['dist_to_goal'] = np.sqrt((105 - df['start_x'])**2 + (34 - df['start_y'])**2)
    df['angle_to_goal'] = np.arctan2(34 - df['start_y'], 105 - df['start_x'])

    # 이동 방향
    df['move_angle'] = np.arctan2(df['dy'], df['dx'])

    # 에피소드별 집계
    episodes = df.groupby('game_episode')

    all_features = []

    for ep_id, ep_df in episodes:
        ep_df = ep_df.sort_values('action_id')
        last_row = ep_df.iloc[-1]

        features = {
            'game_episode': ep_id,
            'game_id': last_row['game_id'],
            # 현재 위치
            'start_x': last_row['start_x'],
            'start_y': last_row['start_y'],
            'dist_to_goal': last_row['dist_to_goal'],
            'angle_to_goal': last_row['angle_to_goal'],
            # 경기 정보
            'period_id': last_row['period_id'],
            'total_actions': len(ep_df),
        }

        # 마지막 N개 액션 통계
        last_n = ep_df.tail(lookback)

        # 이동 통계
        features['last_n_dx_mean'] = last_n['dx'].mean()
        features['last_n_dx_std'] = last_n['dx'].std() if len(last_n) > 1 else 0
        features['last_n_dy_mean'] = last_n['dy'].mean()
        features['last_n_dy_std'] = last_n['dy'].std() if len(last_n) > 1 else 0
        features['last_n_dist_mean'] = last_n['distance'].mean()

        # 위치 변화
        features['last_n_x_range'] = last_n['start_x'].max() - last_n['start_x'].min()
        features['last_n_y_range'] = last_n['start_y'].max() - last_n['start_y'].min()

        # 방향 변화
        angles = last_n['move_angle'].values
        if len(angles) > 1:
            angle_changes = np.diff(angles)
            features['angle_change_mean'] = np.nanmean(angle_changes)
            features['angle_change_std'] = np.nanstd(angle_changes) if len(angle_changes) > 1 else 0
        else:
            features['angle_change_mean'] = 0
            features['angle_change_std'] = 0

        # 직전 패스들
        for i in range(1, min(4, len(ep_df) + 1)):
            idx = -(i + 1) if i < len(ep_df) else 0
            prev_row = ep_df.iloc[idx] if idx >= -len(ep_df) else ep_df.iloc[0]
            features[f'prev_{i}_dx'] = prev_row['dx'] if not np.isnan(prev_row['dx']) else 0
            features[f'prev_{i}_dy'] = prev_row['dy'] if not np.isnan(prev_row['dy']) else 0

        # 타겟 (있는 경우)
        if not np.isnan(last_row['end_x']):
            features['end_x'] = last_row['end_x']
            features['end_y'] = last_row['end_y']
        else:
            features['end_x'] = np.nan
            features['end_y'] = np.nan

        all_features.append(features)

    return pd.DataFrame(all_features)

print("  Train 피처 생성 중...")
train_features = create_sequence_summary_features(train_df, lookback=5)
train_features = train_features.dropna(subset=['end_x', 'end_y'])

print("  Test 피처 생성 중...")
test_features = create_sequence_summary_features(test_all, lookback=5)

print(f"Train samples: {len(train_features):,}")
print(f"Test samples: {len(test_features):,}")

# =============================================================================
# 3. 학습 데이터 준비
# =============================================================================
print("\n[3] 학습 데이터 준비...")

feature_cols = [
    'start_x', 'start_y', 'dist_to_goal', 'angle_to_goal',
    'period_id', 'total_actions',
    'last_n_dx_mean', 'last_n_dx_std', 'last_n_dy_mean', 'last_n_dy_std',
    'last_n_dist_mean', 'last_n_x_range', 'last_n_y_range',
    'angle_change_mean', 'angle_change_std',
    'prev_1_dx', 'prev_1_dy', 'prev_2_dx', 'prev_2_dy', 'prev_3_dx', 'prev_3_dy'
]

feature_cols = [c for c in feature_cols if c in train_features.columns]
print(f"피처 수: {len(feature_cols)}")

X = train_features[feature_cols].values
y_x = train_features['end_x'].values
y_y = train_features['end_y'].values
game_ids = train_features['game_id'].values
ep_ids = train_features['game_episode'].values

X_test = test_features[feature_cols].values
test_ep_ids = test_features['game_episode'].values

# NaN 처리
X = np.nan_to_num(X, nan=0)
X_test = np.nan_to_num(X_test, nan=0)

# =============================================================================
# 4. Zone Baseline 준비
# =============================================================================
print("\n[4] Zone Baseline (6x6 median) 준비...")

def get_zone_6x6(x, y):
    x_zone = min(5, int(x / (105 / 6)))
    y_zone = min(5, int(y / (68 / 6)))
    return x_zone * 6 + y_zone

train_features['zone'] = train_features.apply(lambda r: get_zone_6x6(r['start_x'], r['start_y']), axis=1)
train_features['delta_x'] = train_features['end_x'] - train_features['start_x']
train_features['delta_y'] = train_features['end_y'] - train_features['start_y']

zone_stats = train_features.groupby('zone').agg({
    'delta_x': 'median',
    'delta_y': 'median'
}).to_dict()

# Zone 예측 (Train)
zone_pred_x_train = []
zone_pred_y_train = []
for _, row in train_features.iterrows():
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

# Zone 예측 (Test)
test_features['zone'] = test_features.apply(lambda r: get_zone_6x6(r['start_x'], r['start_y']), axis=1)
zone_pred_x_test = []
zone_pred_y_test = []
for _, row in test_features.iterrows():
    zone = get_zone_6x6(row['start_x'], row['start_y'])
    dx = zone_stats['delta_x'].get(zone, 0)
    dy = zone_stats['delta_y'].get(zone, 0)
    zone_pred_x_test.append(np.clip(row['start_x'] + dx, 0, 105))
    zone_pred_y_test.append(np.clip(row['start_y'] + dy, 0, 68))

zone_pred_x_test = np.array(zone_pred_x_test)
zone_pred_y_test = np.array(zone_pred_y_test)

# =============================================================================
# 5. MLP 학습 (GroupKFold)
# =============================================================================
print("\n[5] MLP 학습 (GroupKFold)...")

n_splits = 5
gkf = GroupKFold(n_splits=n_splits)

oof_pred_x = np.zeros(len(X))
oof_pred_y = np.zeros(len(X))
test_pred_x = np.zeros(len(X_test))
test_pred_y = np.zeros(len(X_test))

fold_scores = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, groups=game_ids)):
    print(f"\n  Fold {fold+1}/{n_splits}")

    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr_x, y_val_x = y_x[train_idx], y_x[val_idx]
    y_tr_y, y_val_y = y_y[train_idx], y_y[val_idx]

    # 스케일링
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # MLP for X
    mlp_x = MLPRegressor(
        hidden_layer_sizes=(32, 16),  # 작은 네트워크
        activation='relu',
        solver='adam',
        alpha=1.0,  # 강한 L2 정규화
        batch_size=64,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
        verbose=False
    )
    mlp_x.fit(X_tr_scaled, y_tr_x)

    # MLP for Y
    mlp_y = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        activation='relu',
        solver='adam',
        alpha=1.0,
        batch_size=64,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
        verbose=False
    )
    mlp_y.fit(X_tr_scaled, y_tr_y)

    # OOF 예측
    oof_pred_x[val_idx] = mlp_x.predict(X_val_scaled)
    oof_pred_y[val_idx] = mlp_y.predict(X_val_scaled)

    # 테스트 예측
    test_pred_x += mlp_x.predict(X_test_scaled) / n_splits
    test_pred_y += mlp_y.predict(X_test_scaled) / n_splits

    # Fold 점수
    fold_dist = np.sqrt((oof_pred_x[val_idx] - y_val_x)**2 + (oof_pred_y[val_idx] - y_val_y)**2)
    fold_score = fold_dist.mean()
    fold_scores.append(fold_score)
    print(f"    Fold {fold+1} Score: {fold_score:.4f}")

# 전체 OOF 점수
oof_dist = np.sqrt((oof_pred_x - y_x)**2 + (oof_pred_y - y_y)**2)
mlp_score = oof_dist.mean()

print("\n" + "=" * 70)
print(f"MLP CV Score: {mlp_score:.4f}")
print(f"Fold Scores: {[f'{s:.4f}' for s in fold_scores]}")
print(f"Std: {np.std(fold_scores):.4f}")
print("=" * 70)

# =============================================================================
# 6. 앙상블 최적화
# =============================================================================
print("\n[6] Zone + MLP 앙상블 최적화...")

best_alpha = None
best_ensemble_score = float('inf')
results = []

for alpha in [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]:
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
    print(f"  Zone {alpha:.0%} + MLP {1-alpha:.0%}: CV = {score:.4f}{marker}")

print(f"\n최적 가중치: Zone {best_alpha:.0%} + MLP {1-best_alpha:.0%}")
print(f"앙상블 CV Score: {best_ensemble_score:.4f}")

# =============================================================================
# 7. 제출 파일 생성
# =============================================================================
print("\n[7] 제출 파일 생성...")

# 1. Pure MLP
test_pred_x_clipped = np.clip(test_pred_x, 0, 105)
test_pred_y_clipped = np.clip(test_pred_y, 0, 68)

# episode_id 매핑
test_ep_to_pred = dict(zip(test_ep_ids, zip(test_pred_x_clipped, test_pred_y_clipped)))

submission_mlp = []
for ep_id in sample_sub['game_episode']:
    pred = test_ep_to_pred.get(ep_id, (52.5, 34))
    submission_mlp.append({'game_episode': ep_id, 'end_x': pred[0], 'end_y': pred[1]})

submission_mlp = pd.DataFrame(submission_mlp)
submission_mlp.to_csv('submission_mlp_sequence.csv', index=False)
print(f"  1. submission_mlp_sequence.csv 저장 (CV: {mlp_score:.4f})")

# 2. Zone + MLP 앙상블
for alpha in [0.65, 0.7, 0.75, 0.8]:
    ens_x = alpha * zone_pred_x_test + (1 - alpha) * test_pred_x_clipped
    ens_y = alpha * zone_pred_y_test + (1 - alpha) * test_pred_y_clipped
    ens_x = np.clip(ens_x, 0, 105)
    ens_y = np.clip(ens_y, 0, 68)

    test_ep_to_ens = dict(zip(test_ep_ids, zip(ens_x, ens_y)))

    submission_ens = []
    for ep_id in sample_sub['game_episode']:
        pred = test_ep_to_ens.get(ep_id, (52.5, 34))
        submission_ens.append({'game_episode': ep_id, 'end_x': pred[0], 'end_y': pred[1]})

    submission_ens = pd.DataFrame(submission_ens)

    # Train CV 계산
    ens_train_x = alpha * zone_pred_x_train + (1 - alpha) * oof_pred_x
    ens_train_y = alpha * zone_pred_y_train + (1 - alpha) * oof_pred_y
    ens_dist = np.sqrt((ens_train_x - y_x)**2 + (ens_train_y - y_y)**2)
    cv = ens_dist.mean()

    filename = f'submission_mlp_zone{int(alpha*100)}.csv'
    submission_ens.to_csv(filename, index=False)
    print(f"  2. {filename} 저장 (CV: {cv:.4f})")

# =============================================================================
# 8. 최종 요약
# =============================================================================
print("\n" + "=" * 70)
print("최종 요약")
print("=" * 70)

print(f"\n[모델 비교]")
print(f"  Zone Baseline (6x6 median): CV = {zone_score:.4f} → Public = 16.85")
print(f"  MLP Sequence:               CV = {mlp_score:.4f}")
print(f"  최적 앙상블 (Zone {best_alpha:.0%}):    CV = {best_ensemble_score:.4f}")

print(f"\n[생성된 파일]")
print(f"  1. submission_mlp_sequence.csv")
print(f"  2. submission_mlp_zone65.csv")
print(f"  3. submission_mlp_zone70.csv")
print(f"  4. submission_mlp_zone75.csv")
print(f"  5. submission_mlp_zone80.csv")

print("\n" + "=" * 70)
print("완료!")
print("=" * 70)
