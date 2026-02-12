"""
빠른 실험 시스템 v3

Ultrathink 2025-12-16:
- Batch 1: is_home, type_name, result_name (✅ CV 15.79)
- Batch 2: team encoding (team별 평균 패스 거리)

예상 효과: CV 15.79 → 15.5-15.7
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
import time
import json
from pathlib import Path

class FastExperimentV3:
    """빠른 실험을 위한 유틸리티 v3 (Batch 2 포함)"""

    def __init__(self, sample_frac=0.1, n_folds=3, random_state=42):
        self.sample_frac = sample_frac
        self.n_folds = n_folds
        self.random_state = random_state
        np.random.seed(random_state)

    def load_data(self, train_path='../../../train.csv', sample=True):
        """데이터 로드 (Episode 단위 샘플링)"""
        print(f"\n{'='*80}")
        print("데이터 로드")
        print(f"{'='*80}")

        train_df = pd.read_csv(train_path)
        print(f"  원본: {len(train_df):,}개 패스")

        if sample:
            episodes = train_df['game_episode'].unique()
            print(f"  전체 Episode: {len(episodes):,}개")

            n_sample = int(len(episodes) * self.sample_frac)
            sampled_episodes = np.random.choice(
                episodes, size=n_sample, replace=False
            )

            train_df = train_df[train_df['game_episode'].isin(sampled_episodes)]
            print(f"  샘플링: {len(sampled_episodes):,}개 Episode ({self.sample_frac*100:.0f}%)")
            print(f"  샘플링 후: {len(train_df):,}개 패스")

        return train_df

    def create_features(self, df):
        """
        피처 생성 v3 (Batch 1 + Batch 2)

        CONSTRAINT: Episode 독립성 유지
        """
        print(f"\n{'='*80}")
        print("피처 생성 v3 (Batch 1 + Batch 2)")
        print(f"{'='*80}")

        df = df.copy()

        # ========== 기존 피처 ==========
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
        df['pass_count'] = df.groupby('game_episode').cumcount() + 1

        # ========== Batch 1 피처 ==========
        df['is_home_encoded'] = df['is_home'].astype(int)

        type_map = {'Pass': 0, 'Carry': 1}
        df['type_encoded'] = df['type_name'].map(type_map).fillna(2).astype(int)

        result_map = {'Successful': 0, 'Unsuccessful': 1}
        df['result_encoded'] = df['result_name'].map(result_map).fillna(2).astype(int)

        print("  Batch 1: ✅ is_home, type, result")

        # ========== Batch 2: Team Encoding ==========
        print("  Batch 2 피처:")

        # Team별 평균 패스 위치 (전체 데이터에서 계산)
        team_stats = df.groupby('team_id').agg({
            'end_x': 'mean',
            'end_y': 'mean',
            'start_x': 'mean'
        })
        team_stats.columns = ['team_end_x_mean', 'team_end_y_mean', 'team_start_x_mean']
        team_stats['team_dx_mean'] = team_stats['team_end_x_mean'] - team_stats['team_start_x_mean']

        # Merge
        df = df.merge(team_stats, left_on='team_id', right_index=True, how='left')

        print("    ✅ team_end_x_mean")
        print("    ✅ team_end_y_mean")
        print("    ✅ team_dx_mean")

        print(f"\n  총 피처: 기존 13개 + Batch 1 (3개) + Batch 2 (3개) = 19개")

        return df

    def prepare_data(self, df):
        """마지막 패스만 추출 & Feature/Target 분리"""
        train_last = df.groupby('game_episode').last().reset_index()

        # Feature columns (v3: Batch 2 포함!)
        feature_cols = [
            # 기존 피처
            'start_x', 'start_y',
            'zone_x', 'zone_y',
            'direction',
            'goal_distance', 'goal_angle',
            'period_id', 'time_seconds', 'time_left',
            'pass_count',
            'prev_dx', 'prev_dy',
            # Batch 1 피처
            'is_home_encoded',
            'type_encoded',
            'result_encoded',
            # Batch 2 피처
            'team_end_x_mean',
            'team_end_y_mean',
            'team_dx_mean'
        ]

        X = train_last[feature_cols].values
        y = train_last[['end_x', 'end_y']].values
        groups = train_last['game_episode'].str.split('_').str[0].values

        print(f"\n{'='*80}")
        print("데이터 준비")
        print(f"{'='*80}")
        print(f"  Episodes: {len(train_last):,}개")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  Features: {len(feature_cols)}개")

        return X, y, groups, feature_cols

    def run_cv(self, model_x, model_y, X, y, groups, model_name='Model'):
        """Cross-validation (GroupKFold) - 별도 x/y 모델"""
        print(f"\n{'='*80}")
        print(f"{model_name} Cross-Validation")
        print(f"{'='*80}")

        gkf = GroupKFold(n_splits=self.n_folds)
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model_x.fit(X_train, y_train[:, 0])
            model_y.fit(X_train, y_train[:, 1])

            pred_x = np.clip(model_x.predict(X_val), 0, 105)
            pred_y = np.clip(model_y.predict(X_val), 0, 68)

            dist = np.sqrt((pred_x - y_val[:, 0])**2 + (pred_y - y_val[:, 1])**2)
            cv = dist.mean()
            fold_scores.append(cv)

            print(f"  Fold {fold+1}: {cv:.4f}")

        mean_cv = np.mean(fold_scores)
        std_cv = np.std(fold_scores)

        print(f"\n  Mean CV: {mean_cv:.4f} ± {std_cv:.4f}")

        return mean_cv, std_cv, fold_scores

    def log_experiment(self, name, cv, params, features, runtime, notes=''):
        """실험 로그 저장"""
        log = {
            'name': name,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'cv_mean': float(cv[0]),
            'cv_std': float(cv[1]),
            'cv_folds': [float(x) for x in cv[2]],
            'params': params,
            'features': features,
            'n_features': len(features),
            'runtime': float(runtime),
            'sample_frac': float(self.sample_frac),
            'n_folds': int(self.n_folds),
            'notes': notes
        }

        log_file = Path(__file__).parent.parent.parent / 'logs' / 'experiment_log.json'
        log_file.parent.mkdir(exist_ok=True)

        with open(log_file, 'a') as f:
            f.write(json.dumps(log, ensure_ascii=False) + '\n')

        print(f"\n  ✅ 로그 저장: {log_file}")

        return log
