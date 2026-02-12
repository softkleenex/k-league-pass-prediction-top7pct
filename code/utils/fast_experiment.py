"""
빠른 실험 시스템

Carla 조언:
- 10% 샘플로 빠른 테스트
- 메모리 주의
- 자주 저장

Ultrathink 2025-12-15:
- Episode 단위 샘플링 (중요!)
- Episode 독립성 유지
- 자동 CV 계산
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
import time
import json
from pathlib import Path

class FastExperiment:
    """빠른 실험을 위한 유틸리티"""

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

    def load_data(self, train_path='../../train.csv', sample=True):
        """
        데이터 로드 (Episode 단위 샘플링)

        IMPORTANT: Episode 단위로 샘플링해야 함!
        개별 패스 단위 샘플링 시 Episode 정보 손실
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
        피처 생성 (Episode 독립성 유지!)

        CONSTRAINT: 모든 피처는 Episode 내부에서만 계산
        """
        print(f"\n{'='*80}")
        print("피처 생성")
        print(f"{'='*80}")

        df = df.copy()

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

        # Target
        df['delta_x'] = df['end_x'] - df['start_x']
        df['delta_y'] = df['end_y'] - df['start_y']

        print(f"  피처 생성 완료")

        return df

    def prepare_data(self, df):
        """
        마지막 패스만 추출 & Feature/Target 분리
        """
        # 마지막 패스만 (Episode별!)
        train_last = df.groupby('game_episode').last().reset_index()

        # Feature columns
        feature_cols = [
            'start_x', 'start_y',
            'zone_x', 'zone_y',
            'direction',
            'goal_distance', 'goal_angle',
            'period_id', 'time_seconds', 'time_left',
            'pass_count',
            'prev_dx', 'prev_dy'
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

    def run_cv(self, model, X, y, groups, model_name='Model'):
        """
        Cross-validation (GroupKFold)

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

            # Fit
            model.fit(X_train, y_train)

            # Predict
            pred = model.predict(X_val)

            # Clip to field boundaries
            pred[:, 0] = np.clip(pred[:, 0], 0, 105)
            pred[:, 1] = np.clip(pred[:, 1], 0, 68)

            # Euclidean distance
            dist = np.sqrt((pred[:, 0] - y_val[:, 0])**2 +
                          (pred[:, 1] - y_val[:, 1])**2)
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

    def compare_experiments(self, log_file='../../logs/experiment_log.json'):
        """실험 비교 테이블"""
        log_file = Path(log_file)

        if not log_file.exists():
            print(f"  ⚠️ 로그 파일 없음: {log_file}")
            return []

        logs = []
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    logs.append(json.loads(line))
                except:
                    pass

        if not logs:
            print("  ⚠️ 로그 없음")
            return []

        # Sort by CV
        logs = sorted(logs, key=lambda x: x['cv_mean'])

        print(f"\n{'='*80}")
        print("실험 비교")
        print(f"{'='*80}")
        print(f"{'Rank':<5} {'Name':<25} {'CV':<15} {'Runtime':<10} {'Sample':<8}")
        print('-'*80)

        for i, log in enumerate(logs[:20]):  # Top 20
            name = log['name'][:24]
            cv_mean = log['cv_mean']
            cv_std = log.get('cv_std', 0)
            runtime = log['runtime']
            sample = log.get('sample_frac', 1.0) * 100

            print(f"{i+1:<5} {name:<25} {cv_mean:.4f}±{cv_std:.4f}  {runtime:<10.1f}s {sample:<8.0f}%")

        print(f"{'='*80}\n")

        return logs


# Test
if __name__ == '__main__':
    print("=" * 80)
    print("FastExperiment 테스트")
    print("=" * 80)

    # Initialize
    exp = FastExperiment(sample_frac=0.1, n_folds=3)

    # Load data
    train_df = exp.load_data(sample=True)

    # Create features
    train_df = exp.create_features(train_df)

    # Prepare
    X, y, groups, feature_cols = exp.prepare_data(train_df)

    print(f"\n{'='*80}")
    print("✅ FastExperiment 준비 완료!")
    print(f"{'='*80}")
    print(f"\n다음: gbm_baseline.py 실행")
