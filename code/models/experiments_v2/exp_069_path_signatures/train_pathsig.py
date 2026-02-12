"""
exp_069: Path Signatures
- 시퀀스를 Path Signature로 인코딩
- 수학적으로 경로의 기하학적 특성 캡처
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from catboost import CatBoostRegressor
import esig.tosig as ts
import warnings
warnings.filterwarnings('ignore')

BASE = Path("/mnt/c/LSJ/dacon/dacon/kleague-algorithm")
DATA_DIR = BASE / "data"
SUBMISSION_DIR = BASE / "submissions"


def compute_path_signature(coords, order=3):
    """
    경로의 signature 계산
    coords: (n, 2) 배열 - [[x1,y1], [x2,y2], ...]
    order: signature truncation order
    """
    if len(coords) < 2:
        # 점이 1개면 signature 계산 불가, 영벡터 반환
        sig_dim = ts.sigdim(2, order)
        return np.zeros(sig_dim)

    # float64로 변환
    coords = np.array(coords, dtype=np.float64)

    try:
        sig = ts.stream2sig(coords, order)
        return sig
    except Exception:
        sig_dim = ts.sigdim(2, order)
        return np.zeros(sig_dim)


def create_base_features(df):
    """기존 피처"""
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

    return df


def compute_episode_signatures(df, order=3):
    """각 에피소드의 path signature 계산"""
    sig_dim = ts.sigdim(2, order)
    print(f"  Signature dimension: {sig_dim}")

    signatures = []
    episodes = []

    for ep, group in df.groupby('game_episode'):
        # start_x, start_y 좌표 추출 (마지막 패스 제외)
        if len(group) > 1:
            coords = group[['start_x', 'start_y']].values[:-1]  # 마지막 제외
        else:
            coords = group[['start_x', 'start_y']].values

        sig = compute_path_signature(coords, order)
        signatures.append(sig)
        episodes.append(ep)

    sig_df = pd.DataFrame(
        signatures,
        columns=[f'sig_{i}' for i in range(sig_dim)]
    )
    sig_df['game_episode'] = episodes

    return sig_df


def main():
    print("=" * 70)
    print("exp_069: Path Signatures")
    print("=" * 70)

    # 데이터 로드
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    train_df = create_base_features(train_df)

    # Path Signatures 계산
    print("\n[0] Path Signatures 계산...")
    order = 3  # signature truncation order
    sig_df = compute_episode_signatures(train_df, order)

    # 마지막 패스 추출
    last_passes = train_df.groupby('game_episode').last().reset_index()
    last_passes = last_passes.merge(sig_df, on='game_episode', how='left')

    # 기존 TOP_12 피처
    BASE_FEATURES = [
        'goal_angle', 'zone_y', 'goal_distance', 'dist_to_goal_line',
        'dist_to_center_y', 'prev_dx', 'prev_dy', 'ema_start_x',
        'ema_start_y', 'ema_success_rate', 'ema_possession', 'velocity'
    ]

    # Signature 피처
    sig_dim = ts.sigdim(2, order)
    SIG_FEATURES = [f'sig_{i}' for i in range(sig_dim)]

    ALL_FEATURES = BASE_FEATURES + SIG_FEATURES

    print(f"\n피처 수: {len(ALL_FEATURES)}")
    print(f"  - 기존: {len(BASE_FEATURES)}")
    print(f"  - Signature: {len(SIG_FEATURES)}")

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
        marker = " *SIG*" if name.startswith('sig_') else ""
        print(f"  {name}: {imp:.1f}{marker}")

    # Signature 피처 총 importance
    sig_total = sum(imp_avg[len(BASE_FEATURES):])
    print(f"\n  Signature 총 importance: {sig_total:.1f}%")

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

    # Test Path Signatures
    test_sig_df = compute_episode_signatures(test_all, order)

    test_last = test_all.groupby('game_episode').last().reset_index()
    test_last = test_last.merge(test_sig_df, on='game_episode', how='left')

    X_test = test_last[ALL_FEATURES].fillna(0).replace([np.inf, -np.inf], 0).values

    pred_x = np.zeros(len(X_test))
    pred_y = np.zeros(len(X_test))

    for mx, my in models:
        pred_x += mx.predict(X_test) / len(models)
        pred_y += my.predict(X_test) / len(models)

    pred_y = np.clip(pred_y, 0, 68)

    submission = pd.DataFrame({
        'game_episode': test_last['game_episode'],
        'end_x': pred_x,
        'end_y': pred_y
    })

    submission = submission.set_index('game_episode').loc[test_df['game_episode']].reset_index()

    output_path = SUBMISSION_DIR / f"submission_pathsig_cv{cv:.2f}.csv"
    submission.to_csv(output_path, index=False)
    print(f"  저장: {output_path}")

    print("\n" + "=" * 70)
    print(f"Path Signatures CV: {cv:.4f}")
    print(f"vs MAE-only (13.66): {'+' if cv > 13.66 else ''}{cv - 13.66:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
