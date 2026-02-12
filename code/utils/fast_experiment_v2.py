"""
빠른 실험 시스템 v2

Ultrathink 2025-12-16:
- Batch 1 피처 추가:
  * is_home (홈/어웨이)
  * type_name (Pass/Carry/Other)
  * result_name (Successful/Unsuccessful/Other)

예상 효과: CV 16.04 → 15.7-15.9
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
import time
import json
from pathlib import Path

class FastExperimentV2:
    """빠른 실험을 위한 유틸리티 v2 (새 피처 포함)"""

    def __init__(self, sample_frac=0.1, n_folds=3, random_state=42):
        """
        Args:
            sample_frac: Episode 샘플링 비율 (0.1 = 10%)
            n_folds: Cross-validation fold 수
            random_state: Random seed
        """
        self.sample_frac = sample_frac
        self.n_folds = n_folds
        self.random_state = random_state
        np.random.seed(random_state)

    def load_data(self, train_path='../../../train.csv', sample=True):
        """
        데이터 로드 (Episode 단위 샘플링)

        IMPORTANT: Episode 단위로 샘플링해야 함!
        """
        print(f"\n{'='*80}")
        print("데이터 로드")
        print(f"{'='*80}")

        train_df = pd.read_csv(train_path)
        print(f"  원본: {len(train_df):,}개 패스")

        if sample:
            # Episode 단위 샘플링 (중요!)
            episodes = train_df['game_episode'].unique()
            print(f"  전체 Episode: {len(episodes):,}개")

            n_sample = int(len(episodes) * self.sample_frac)
            sampled_episodes = np.random.choice(
                episodes,
                size=n_sample,
                replace=False
            )

            train_df = train_df[train_df['game_episode'].isin(sampled_episodes)]
            print(f"  샘플링: {len(sampled_episodes):,}개 Episode ({self.sample_frac*100:.0f}%)")
            print(f"  샘플링 후: {len(train_df):,}개 패스")

        return train_df

    def create_features(self, df):
        """
        피처 생성 v2 (새 피처 포함!)

        CONSTRAINT: 모든 피처는 Episode 내부에서만 계산
        """
        print(f"\n{'='*80}")
        print("피처 생성 v2 (Batch 1)")
        print(f"{'='*80}")

        df = df.copy()

        # ========== 기존 피처 ==========

        # 1. Zone 6x6
        df['zone_x'] = (df['start_x'] / (105/6)).astype(int).clip(0, 5)
        df['zone_y'] = (df['start_y'] / (68/6)).astype(int).clip(0, 5)
        df['zone'] = df['zone_x'].astype(str) + '_' + df['zone_y'].astype(str)

        # 2. Direction 8-way (이전 패스 방향)
        df['dx'] = df['end_x'] - df['start_x']
        df['dy'] = df['end_y'] - df['start_y']

        # Episode별로 shift (중요!)
        df['prev_dx'] = df.groupby('game_episode')['dx'].shift(1).fillna(0)
        df['prev_dy'] = df.groupby('game_episode')['dy'].shift(1).fillna(0)

        angle = np.degrees(np.arctan2(df['prev_dy'], df['prev_dx']))
        df['direction'] = ((angle + 22.5) // 45).astype(int) % 8

        # 3. Goal distance & angle
        df['goal_distance'] = np.sqrt((105 - df['start_x'])**2 + (34 - df['start_y'])**2)
        df['goal_angle'] = np.degrees(np.arctan2(34 - df['start_y'], 105 - df['start_x']))

        # 4. Time features
        df['time_left'] = 5400 - df['time_seconds']

        # 5. Episode features (Episode별!)
        df['pass_count'] = df.groupby('game_episode').cumcount() + 1

        # ========== Batch 1: 새 피처 ==========

        print("  Batch 1 피처:")

        # 6. is_home (홈/어웨이)
        df['is_home_encoded'] = df['is_home'].astype(int)
        print("    ✅ is_home_encoded")

        # 7. type_name (Pass/Carry/Other)
        type_map = {
            'Pass': 0,
            'Carry': 1
        }
        df['type_encoded'] = df['type_name'].map(type_map).fillna(2).astype(int)
        print("    ✅ type_encoded (0=Pass, 1=Carry, 2=Other)")

        # 8. result_name (Successful/Unsuccessful/Other)
        result_map = {
            'Successful': 0,
            'Unsuccessful': 1
        }
        df['result_encoded'] = df['result_name'].map(result_map).fillna(2).astype(int)
        print("    ✅ result_encoded (0=Success, 1=Fail, 2=Other)")

        print(f"\n  총 피처: 기존 13개 + 새로운 3개 = 16개")

        return df

    def prepare_data(self, df):
        """
        마지막 패스만 추출 & Feature/Target 분리
        """
        # 마지막 패스만 (Episode별!)
        train_last = df.groupby('game_episode').last().reset_index()

        # Feature columns (v2: 새 피처 포함!)
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
            'result_encoded'
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
        print(f"  Features: {feature_cols}")

        return X, y, groups, feature_cols

    def run_cv(self, model_x, model_y, X, y, groups, model_name='Model'):
        """
        Cross-validation (GroupKFold) - 별도 x/y 모델

        CONSTRAINT: GroupKFold로 game-level 분리
        """
        print(f"\n{'='*80}")
        print(f"{model_name} Cross-Validation")
        print(f"{'='*80}")

        gkf = GroupKFold(n_splits=self.n_folds)
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Fit separate models
            model_x.fit(X_train, y_train[:, 0])
            model_y.fit(X_train, y_train[:, 1])

            # Predict
            pred_x = np.clip(model_x.predict(X_val), 0, 105)
            pred_y = np.clip(model_y.predict(X_val), 0, 68)

            # Euclidean distance
            dist = np.sqrt((pred_x - y_val[:, 0])**2 +
                          (pred_y - y_val[:, 1])**2)
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

        # Append to log file
        log_file = Path(__file__).parent.parent.parent / 'logs' / 'experiment_log.json'
        log_file.parent.mkdir(exist_ok=True)

        with open(log_file, 'a') as f:
            f.write(json.dumps(log, ensure_ascii=False) + '\n')

        print(f"\n  ✅ 로그 저장: {log_file}")

        return log


# Test
if __name__ == '__main__':
    print("=" * 80)
    print("FastExperimentV2 테스트")
    print("=" * 80)

    # Initialize
    exp = FastExperimentV2(sample_frac=0.1, n_folds=3)

    # Load data
    train_df = exp.load_data(sample=True)

    # Create features
    train_df = exp.create_features(train_df)

    # Prepare
    X, y, groups, feature_cols = exp.prepare_data(train_df)

    print(f"\n{'='*80}")
    print("✅ FastExperimentV2 준비 완료!")
    print(f"{'='*80}")
