"""
exp_068: Enhanced Sequence Features
- 기존 MAE 모델 + 풍부한 시퀀스 통계 피처
- zen MCP 권장: "Start with Aggregated Features"
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

BASE = Path("/mnt/c/LSJ/dacon/dacon/kleague-algorithm")
DATA_DIR = BASE / "data"
SUBMISSION_DIR = BASE / "submissions"


def create_base_features(df):
    """기존 피처 (exp_067)"""
    df['zone_x'] = (df['start_x'] / (105/6)).astype(int).clip(0, 5)
    df['zone_y'] = (df['start_y'] / (68/6)).astype(int).clip(0, 5)
    df['goal_distance'] = np.sqrt((105 - df['start_x'])**2 + (34 - df['start_y'])**2)
    df['goal_angle'] = np.degrees(np.arctan2(34 - df['start_y'], 105 - df['start_x']))

    df['dx'] = df['end_x'] - df['start_x']
    df['dy'] = df['end_y'] - df['start_y']
    df['prev_dx'] = df.groupby('game_episode')['dx'].shift(1).fillna(0)
    df['prev_dy'] = df.groupby('game_episode')['dy'].shift(1).fillna(0)

    result_map = {'Successful': 0, 'Unsuccessful': 1}
    df['result_encoded'] = df['result_name'].map(result_map).fillna(2).astype(int)

    ema_span = 2
    df['ema_start_x'] = df.groupby('game_episode')['start_x'].transform(
        lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)
    ).fillna(df['start_x'])
    df['ema_start_y'] = df.groupby('game_episode')['start_y'].transform(
        lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)
    ).fillna(df['start_y'])

    result_map2 = {'Successful': 1, 'Unsuccessful': 0}
    df['is_successful'] = df['result_name'].map(result_map2).fillna(0)
    df['ema_success_rate'] = df.groupby('game_episode')['is_successful'].transform(
        lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)
    ).fillna(0.5)

    df['final_team_id'] = df.groupby('game_episode')['team_id'].transform('last')
    df['is_final_team'] = (df['team_id'] == df['final_team_id']).astype(int)
    df['ema_possession'] = df.groupby('game_episode')['is_final_team'].transform(
        lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)
    ).fillna(0.5)

    df['dist_to_goal_line'] = 105 - df['start_x']
    df['dist_to_center_y'] = np.abs(df['start_y'] - 34)
    df['diff_x'] = df.groupby('game_episode')['start_x'].diff().fillna(0)

    df['prev_start_x'] = df.groupby('game_episode')['start_x'].shift(1).fillna(df['start_x'])
    df['prev_start_y'] = df.groupby('game_episode')['start_y'].shift(1).fillna(df['start_y'])
    df['velocity'] = np.sqrt(
        (df['start_x'] - df['prev_start_x'])**2 +
        (df['start_y'] - df['prev_start_y'])**2
    )

    # 패스 거리
    df['pass_distance'] = np.sqrt(df['dx']**2 + df['dy']**2)

    return df


def compute_sequence_stats(df):
    """시퀀스별 통계 피처 계산"""
    seq_stats = df.groupby('game_episode').agg({
        # 시퀀스 길이
        'start_x': ['count', 'mean', 'std', 'min', 'max'],
        'start_y': ['mean', 'std', 'min', 'max'],

        # 패스 성공률
        'is_successful': 'mean',

        # 패스 거리 통계
        'pass_distance': ['mean', 'std', 'max'],

        # 속도 통계
        'velocity': ['mean', 'std', 'max'],

        # x 진행도
        'dx': ['sum', 'mean'],
        'dy': ['sum', 'mean'],
    })

    # 컬럼명 평탄화
    seq_stats.columns = ['_'.join(col).strip() for col in seq_stats.columns.values]
    seq_stats = seq_stats.rename(columns={
        'start_x_count': 'seq_length',
        'start_x_mean': 'seq_x_mean',
        'start_x_std': 'seq_x_std',
        'start_x_min': 'seq_x_min',
        'start_x_max': 'seq_x_max',
        'start_y_mean': 'seq_y_mean',
        'start_y_std': 'seq_y_std',
        'start_y_min': 'seq_y_min',
        'start_y_max': 'seq_y_max',
        'is_successful_mean': 'seq_success_rate',
        'pass_distance_mean': 'seq_pass_dist_mean',
        'pass_distance_std': 'seq_pass_dist_std',
        'pass_distance_max': 'seq_pass_dist_max',
        'velocity_mean': 'seq_velocity_mean',
        'velocity_std': 'seq_velocity_std',
        'velocity_max': 'seq_velocity_max',
        'dx_sum': 'seq_dx_total',
        'dx_mean': 'seq_dx_mean',
        'dy_sum': 'seq_dy_total',
        'dy_mean': 'seq_dy_mean',
    })

    # 추가 계산
    seq_stats['seq_x_range'] = seq_stats['seq_x_max'] - seq_stats['seq_x_min']
    seq_stats['seq_y_range'] = seq_stats['seq_y_max'] - seq_stats['seq_y_min']
    seq_stats['seq_spatial_coverage'] = seq_stats['seq_x_range'] * seq_stats['seq_y_range']

    return seq_stats.reset_index()


def compute_last_n_stats(df, n=3):
    """마지막 N개 패스 통계"""
    def last_n_agg(group):
        last_n = group.tail(n)
        return pd.Series({
            f'last{n}_x_mean': last_n['start_x'].mean(),
            f'last{n}_y_mean': last_n['start_y'].mean(),
            f'last{n}_dx_mean': last_n['dx'].mean(),
            f'last{n}_dy_mean': last_n['dy'].mean(),
            f'last{n}_dist_mean': last_n['pass_distance'].mean(),
            f'last{n}_success_rate': last_n['is_successful'].mean(),
        })

    return df.groupby('game_episode').apply(last_n_agg).reset_index()


def main():
    print("=" * 70)
    print("exp_068: Enhanced Sequence Features")
    print("=" * 70)

    # 데이터 로드
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    train_df = create_base_features(train_df)

    # 시퀀스 통계 계산
    seq_stats = compute_sequence_stats(train_df)
    last3_stats = compute_last_n_stats(train_df, n=3)

    # 마지막 패스 추출
    last_passes = train_df.groupby('game_episode').last().reset_index()

    # 시퀀스 통계 병합
    last_passes = last_passes.merge(seq_stats, on='game_episode', how='left')
    last_passes = last_passes.merge(last3_stats, on='game_episode', how='left')

    # 기존 TOP_12 피처
    BASE_FEATURES = [
        'goal_angle', 'zone_y', 'goal_distance', 'dist_to_goal_line',
        'dist_to_center_y', 'prev_dx', 'prev_dy', 'ema_start_x',
        'ema_start_y', 'ema_success_rate', 'ema_possession', 'velocity'
    ]

    # 새 시퀀스 피처
    SEQ_FEATURES = [
        'seq_length', 'seq_x_mean', 'seq_x_std', 'seq_y_mean', 'seq_y_std',
        'seq_success_rate', 'seq_pass_dist_mean', 'seq_pass_dist_std',
        'seq_velocity_mean', 'seq_dx_total', 'seq_dy_total',
        'seq_x_range', 'seq_y_range', 'seq_spatial_coverage',
        'last3_x_mean', 'last3_y_mean', 'last3_dx_mean', 'last3_dy_mean',
        'last3_dist_mean', 'last3_success_rate'
    ]

    ALL_FEATURES = BASE_FEATURES + SEQ_FEATURES

    print(f"\n피처 수: {len(ALL_FEATURES)}")
    print(f"  - 기존: {len(BASE_FEATURES)}")
    print(f"  - 시퀀스: {len(SEQ_FEATURES)}")

    X = last_passes[ALL_FEATURES].fillna(0).replace([np.inf, -np.inf], 0).values
    y = last_passes[['end_x', 'end_y']].values
    groups = last_passes['game_id'].values

    params = {
        'iterations': 4000, 'depth': 8, 'learning_rate': 0.01,
        'l2_leaf_reg': 7.0, 'random_state': 42, 'verbose': 0,
        'early_stopping_rounds': 100,
        'loss_function': 'MAE'
    }

    print("\n[1] CV (5-Fold)...")
    gkf = GroupKFold(n_splits=5)
    fold_scores = []
    models = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
        model_x = CatBoostRegressor(**params)
        model_y = CatBoostRegressor(**params)

        model_x.fit(X[train_idx], y[train_idx, 0],
                   eval_set=(X[val_idx], y[val_idx, 0]), use_best_model=True)
        model_y.fit(X[train_idx], y[train_idx, 1],
                   eval_set=(X[val_idx], y[val_idx, 1]), use_best_model=True)

        pred_x = model_x.predict(X[val_idx])
        pred_y = model_y.predict(X[val_idx])
        errors = np.sqrt((pred_x - y[val_idx, 0])**2 + (pred_y - y[val_idx, 1])**2)
        fold_scores.append(errors.mean())
        print(f"  Fold {fold}: {fold_scores[-1]:.4f}")
        models.append((model_x, model_y))

    cv = np.mean(fold_scores)
    print(f"  CV: {cv:.4f}")

    # Feature Importance
    print("\n[2] Feature Importance (Top 15)...")
    imp_x = models[0][0].get_feature_importance()
    imp_y = models[0][1].get_feature_importance()
    imp_avg = (imp_x + imp_y) / 2

    feat_imp = sorted(zip(ALL_FEATURES, imp_avg), key=lambda x: -x[1])[:15]
    for name, imp in feat_imp:
        marker = " *NEW*" if name in SEQ_FEATURES else ""
        print(f"  {name}: {imp:.1f}{marker}")

    # Test 예측
    print("\n[3] Test 예측...")
    test_df = pd.read_csv(DATA_DIR / 'test.csv')
    test_episodes = []
    for _, row in test_df.iterrows():
        path = DATA_DIR / row['path'].replace('./', '')
        ep_df = pd.read_csv(path)
        if 'dx' not in ep_df.columns:
            ep_df['dx'] = ep_df['end_x'] - ep_df['start_x']
            ep_df['dy'] = ep_df['end_y'] - ep_df['start_y']
        ep_df['game_episode'] = row['game_episode']
        ep_df['game_id'] = row['game_id']
        test_episodes.append(ep_df)

    test_all = pd.concat(test_episodes, ignore_index=True)
    test_all = create_base_features(test_all)

    # 시퀀스 통계
    test_seq_stats = compute_sequence_stats(test_all)
    test_last3_stats = compute_last_n_stats(test_all, n=3)

    test_last = test_all.groupby('game_episode').last().reset_index()
    test_last = test_last.merge(test_seq_stats, on='game_episode', how='left')
    test_last = test_last.merge(test_last3_stats, on='game_episode', how='left')

    X_test = test_last[ALL_FEATURES].fillna(0).replace([np.inf, -np.inf], 0).values

    pred_x = np.zeros(len(X_test))
    pred_y = np.zeros(len(X_test))

    for mx, my in models:
        pred_x += mx.predict(X_test) / len(models)
        pred_y += my.predict(X_test) / len(models)

    # Clip to valid range
    pred_y = np.clip(pred_y, 0, 68)

    submission = pd.DataFrame({
        'game_episode': test_last['game_episode'],
        'end_x': pred_x,
        'end_y': pred_y
    })

    submission = submission.set_index('game_episode').loc[test_df['game_episode']].reset_index()

    output_path = SUBMISSION_DIR / f"submission_seqfeat_cv{cv:.2f}.csv"
    submission.to_csv(output_path, index=False)
    print(f"  저장: {output_path}")

    print("\n" + "=" * 70)
    print(f"Sequence Features CV: {cv:.4f}")
    print(f"vs MAE-only (13.66): {'+' if cv > 13.66 else ''}{cv - 13.66:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
