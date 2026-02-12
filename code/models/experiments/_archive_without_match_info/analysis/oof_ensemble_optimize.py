"""
OOF 앙상블 최적화

아이디어:
  - Phase1A와 exp_036의 OOF 예측 수집
  - 최적 가중치를 CV로 찾기
  - 제출 전 가중치 검증

작성일: 2025-12-21
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from catboost import CatBoostRegressor
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


def create_phase1a_features(df):
    """Phase1A 피처 생성"""
    df['zone_x'] = (df['start_x'] / (105/6)).astype(int).clip(0, 5)
    df['zone_y'] = (df['start_y'] / (68/6)).astype(int).clip(0, 5)

    df['dx'] = df['end_x'] - df['start_x']
    df['dy'] = df['end_y'] - df['start_y']
    df['prev_dx'] = df.groupby('game_episode')['dx'].shift(1).fillna(0)
    df['prev_dy'] = df.groupby('game_episode')['dy'].shift(1).fillna(0)

    angle = np.degrees(np.arctan2(df['prev_dy'], df['prev_dx']))
    df['direction'] = ((angle + 22.5) // 45).astype(int) % 8

    df['goal_distance'] = np.sqrt((105 - df['start_x'])**2 + (34 - df['start_y'])**2)
    df['goal_angle'] = np.degrees(np.arctan2(34 - df['start_y'], 105 - df['start_x']))

    df['time_left'] = 5400 - df['time_seconds']
    df['game_clock_min'] = np.where(df['period_id'] == 1, df['time_seconds'] / 60.0, 45.0 + df['time_seconds'] / 60.0)

    df['pass_count'] = df.groupby('game_episode').cumcount() + 1

    df['is_home_encoded'] = df['is_home'].astype(int)
    type_map = {'Pass': 0, 'Carry': 1}
    df['type_encoded'] = df['type_name'].map(type_map).fillna(2).astype(int)
    result_map = {'Successful': 0, 'Unsuccessful': 1}
    df['result_encoded'] = df['result_name'].map(result_map).fillna(2).astype(int)

    df['final_team_id'] = df.groupby('game_episode')['team_id'].transform('last')
    df['is_final_team'] = (df['team_id'] == df['final_team_id']).astype(int)

    df['team_possession_pct'] = df.groupby('game_episode')['is_final_team'].transform(
        lambda x: x.rolling(window=20, min_periods=1).mean()
    )

    df['team_switch_event'] = (df.groupby('game_episode')['is_final_team'].diff() != 0).astype(int)
    df['team_switches'] = df.groupby('game_episode')['team_switch_event'].cumsum()

    def calc_streak(group):
        values = group['is_final_team'].values
        streaks = []
        current_streak = 0
        for val in values:
            if val == 1:
                current_streak += 1
            else:
                current_streak = 0
            streaks.append(current_streak)
        return pd.Series(streaks, index=group.index)

    df['final_poss_len'] = df.groupby('game_episode', group_keys=False).apply(calc_streak)
    df = df.drop(columns=['dx', 'dy', 'team_switch_event', 'final_team_id'], errors='ignore')

    return df


def train_and_get_oof(X, y, groups, params, name="Model"):
    """Train and get OOF predictions"""
    gkf = GroupKFold(n_splits=3)
    oof = np.zeros_like(y)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
        model_x = CatBoostRegressor(**params, verbose=0)
        model_y = CatBoostRegressor(**params, verbose=0)

        model_x.fit(X[train_idx], y[train_idx, 0])
        model_y.fit(X[train_idx], y[train_idx, 1])

        oof[val_idx, 0] = model_x.predict(X[val_idx])
        oof[val_idx, 1] = model_y.predict(X[val_idx])

        errors = np.sqrt((oof[val_idx, 0] - y[val_idx, 0])**2 +
                        (oof[val_idx, 1] - y[val_idx, 1])**2)
        fold_scores.append(errors.mean())

    cv_mean = np.mean(fold_scores)
    print(f"  {name}: CV {cv_mean:.4f}")
    return oof, cv_mean


def ensemble_error(weights, oof_list, y):
    """Calculate ensemble error"""
    weights = np.array(weights)
    weights = weights / weights.sum()  # Normalize

    ensemble = np.zeros_like(y)
    for w, oof in zip(weights, oof_list):
        ensemble += w * oof

    ensemble[:, 0] = np.clip(ensemble[:, 0], 0, 105)
    ensemble[:, 1] = np.clip(ensemble[:, 1], 0, 68)

    errors = np.sqrt((ensemble[:, 0] - y[:, 0])**2 + (ensemble[:, 1] - y[:, 1])**2)
    return errors.mean()


def main():
    print("\n" + "=" * 80)
    print("OOF 앙상블 최적화")
    print("=" * 80)

    # Load data
    print("\n[1] 데이터 로딩...")
    train_df = pd.read_csv('../../../../data/train.csv')

    # All passes version
    print("\n[2] Phase1A (전체 패스) OOF 생성...")
    all_passes_df = create_phase1a_features(train_df.copy())
    all_last = all_passes_df.groupby('game_episode').last().reset_index()

    exclude_cols = [
        'game_id', 'game_episode', 'player_id', 'team_id', 'period_id',
        'time_seconds', 'type_name', 'result_name', 'is_home',
        'start_x', 'start_y', 'end_x', 'end_y'
    ]
    feature_cols = [col for col in all_last.columns if col not in exclude_cols]

    X_all = all_last[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    y = all_last[['end_x', 'end_y']].values
    groups = all_last['game_id'].values

    params_phase1a = {
        'iterations': 1000,
        'depth': 8,
        'learning_rate': 0.05,
        'l2_leaf_reg': 3.0,
        'random_state': 42
    }

    oof_phase1a, cv_phase1a = train_and_get_oof(X_all, y, groups, params_phase1a, "Phase1A")

    # Last pass only version
    print("\n[3] exp_036 (마지막 패스만) OOF 생성...")
    last_pass_df = train_df.groupby('game_episode').last().reset_index()
    last_pass_df = create_phase1a_features(last_pass_df)

    # Same feature columns
    X_last = last_pass_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values

    params_last = {
        'iterations': 1000,
        'depth': 8,
        'learning_rate': 0.05,
        'l2_leaf_reg': 3.0,
        'random_state': 42
    }

    oof_last, cv_last = train_and_get_oof(X_last, y, groups, params_last, "exp_036")

    # Tuned CatBoost (lr=0.03)
    print("\n[4] Tuned CatBoost (lr=0.03) OOF 생성...")
    params_tuned = {
        'iterations': 1000,
        'depth': 8,
        'learning_rate': 0.03,
        'l2_leaf_reg': 3.0,
        'random_state': 42
    }

    oof_tuned, cv_tuned = train_and_get_oof(X_last, y, groups, params_tuned, "Tuned (lr=0.03)")

    # Find optimal weights
    print("\n" + "=" * 80)
    print("[5] 최적 가중치 탐색")
    print("=" * 80)

    oof_list = [oof_phase1a, oof_last, oof_tuned]
    names = ['Phase1A', 'exp_036', 'Tuned']

    # Grid search
    print("\n  [그리드 탐색: 2모델 조합]")
    print("  " + "-" * 50)

    best_2model = (1.0, 'Phase1A only')
    best_2model_score = cv_phase1a

    # Phase1A + exp_036
    for w1 in np.arange(0.4, 0.7, 0.02):
        w2 = 1 - w1
        score = ensemble_error([w1, w2], [oof_phase1a, oof_last], y)
        if score < best_2model_score:
            best_2model_score = score
            best_2model = (w1, f'Phase1A {w1*100:.0f}% + exp_036 {w2*100:.0f}%')

    print(f"  Phase1A + exp_036 최적: {best_2model[1]}")
    print(f"  CV: {best_2model_score:.4f}")

    # 3 model grid search
    print("\n  [그리드 탐색: 3모델 조합]")
    print("  " + "-" * 50)

    best_3model = None
    best_3model_score = float('inf')

    for w1 in np.arange(0.3, 0.7, 0.05):
        for w2 in np.arange(0.2, 0.6, 0.05):
            w3 = 1 - w1 - w2
            if w3 < 0.05:
                continue
            score = ensemble_error([w1, w2, w3], oof_list, y)
            if score < best_3model_score:
                best_3model_score = score
                best_3model = (w1, w2, w3)

    if best_3model:
        w1, w2, w3 = best_3model
        print(f"  3모델 최적: Phase1A {w1*100:.0f}% + exp_036 {w2*100:.0f}% + Tuned {w3*100:.0f}%")
        print(f"  CV: {best_3model_score:.4f}")

    # Scipy optimize for 2 models
    print("\n  [연속 최적화: Phase1A + exp_036]")
    print("  " + "-" * 50)

    def objective_2(w):
        return ensemble_error([w[0], 1-w[0]], [oof_phase1a, oof_last], y)

    result = minimize(objective_2, [0.5], bounds=[(0.3, 0.8)], method='L-BFGS-B')
    opt_w = result.x[0]
    opt_score = result.fun

    print(f"  최적 가중치: Phase1A {opt_w*100:.1f}% + exp_036 {(1-opt_w)*100:.1f}%")
    print(f"  CV: {opt_score:.4f}")

    # Summary
    print("\n" + "=" * 80)
    print("결과 요약")
    print("=" * 80)

    print(f"""
  개별 모델 성능:
    - Phase1A:       CV {cv_phase1a:.4f}
    - exp_036:       CV {cv_last:.4f}
    - Tuned (lr=0.03): CV {cv_tuned:.4f}

  앙상블 성능:
    - 2모델 최적:    CV {opt_score:.4f}
    - 최적 가중치:   Phase1A {opt_w*100:.1f}% + exp_036 {(1-opt_w)*100:.1f}%

  권장 사항:
    - 제출용 가중치: Phase1A {int(opt_w*100)}:{int((1-opt_w)*100)} exp_036
    - 예상 개선:     {cv_phase1a - opt_score:.4f}점
""")

    print("=" * 80)


if __name__ == '__main__':
    main()
