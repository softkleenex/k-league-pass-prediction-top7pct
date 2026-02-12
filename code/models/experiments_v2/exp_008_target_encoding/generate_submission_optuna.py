"""
exp_008: Optuna Best Params로 Submission 생성
CV: 14.188
"""
import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostRegressor, Pool
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("exp_008: Optuna Best Params Submission")
print("Best CV: 14.188")
print("="*70)

# 경로
BASE = Path("/mnt/c/LSJ/dacon/dacon/kleague-algorithm")
TRAIN_CSV = BASE / "data/train.csv"
TEST_CSV = BASE / "data/test.csv"
SUBMISSION_DIR = BASE / "submissions"

# Best params from Optuna
BEST_PARAMS = {
    'iterations': 804,
    'depth': 8,
    'learning_rate': 0.027036160666620016,
    'l2_leaf_reg': 3.6210622617823773,
    'random_state': 42,
    'loss_function': 'MultiRMSE',
    'verbose': 100
}

def load_train_data():
    """Train 데이터 로드 (각 에피소드별 피처 추출)"""
    df = pd.read_csv(TRAIN_CSV)
    episodes = []

    for game_ep, ep_df in df.groupby('game_episode'):
        game_id = game_ep.split('_')[0]
        ep_df = ep_df.sort_values('time_seconds').reset_index(drop=True)

        last = ep_df.iloc[-1]
        first = ep_df.iloc[0]

        ep_features = {
            'game_episode': game_ep,
            'game_id': game_id,
            'start_x': last['start_x'],
            'start_y': last['start_y'],
            'dx': last['end_x'] - last['start_x'],
            'dy': last['end_y'] - last['start_y'],
            'seq_len': len(ep_df),
        }

        # Speed & direction
        ep_features['speed'] = np.sqrt(ep_features['dx']**2 + ep_features['dy']**2)
        ep_features['direction'] = np.arctan2(ep_features['dy'], ep_features['dx'])

        # Zone features
        zone_x = int(last['start_x'] // (105/6))
        zone_y = int(last['start_y'] // (68/6))
        zone_x = min(max(zone_x, 0), 5)
        zone_y = min(max(zone_y, 0), 5)
        ep_features['zone_id'] = zone_y * 6 + zone_x

        # Progression
        ep_features['progression_x'] = last['start_x'] - first['start_x']
        ep_features['progression_y'] = last['start_y'] - first['start_y']

        # Rolling stats (multi-scale)
        for col in ['start_x', 'start_y']:
            dx_col = ep_df['end_x'] - ep_df['start_x']
            dy_col = ep_df['end_y'] - ep_df['start_y']

            # Window 3
            roll3 = ep_df[col].rolling(3, min_periods=1)
            ep_features[f'{col}_roll3_mean'] = roll3.mean().iloc[-1]
            ep_features[f'{col}_roll3_std'] = roll3.std().iloc[-1] if len(ep_df) > 1 else 0

            # Window 5
            roll5 = ep_df[col].rolling(5, min_periods=1)
            ep_features[f'{col}_roll5_mean'] = roll5.mean().iloc[-1]
            ep_features[f'{col}_roll5_std'] = roll5.std().iloc[-1] if len(ep_df) > 1 else 0

        # dx, dy rolling
        dx_series = ep_df['end_x'] - ep_df['start_x']
        dy_series = ep_df['end_y'] - ep_df['start_y']

        for name, series in [('dx', dx_series), ('dy', dy_series)]:
            roll3 = series.rolling(3, min_periods=1)
            ep_features[f'{name}_roll3_mean'] = roll3.mean().iloc[-1]
            ep_features[f'{name}_roll3_std'] = roll3.std().iloc[-1] if len(ep_df) > 1 else 0

            roll5 = series.rolling(5, min_periods=1)
            ep_features[f'{name}_roll5_mean'] = roll5.mean().iloc[-1]
            ep_features[f'{name}_roll5_std'] = roll5.std().iloc[-1] if len(ep_df) > 1 else 0

        # Target (마지막 패스의 도착점)
        ep_features['end_x'] = last['end_x']
        ep_features['end_y'] = last['end_y']

        episodes.append(ep_features)

    return pd.DataFrame(episodes)

def load_test_data():
    """Test 데이터 로드 (path에서 각 에피소드 로드)"""
    df = pd.read_csv(TEST_CSV)
    episodes = []

    for _, row in df.iterrows():
        path = BASE / "data" / row['path'].replace('./', '')
        ep_df = pd.read_csv(path)
        game_ep = row['game_episode']
        game_id = str(row['game_id'])

        last = ep_df.iloc[-1]
        first = ep_df.iloc[0]

        ep_features = {
            'game_episode': game_ep,
            'game_id': game_id,
            'start_x': last['start_x'],
            'start_y': last['start_y'],
            'dx': last['end_x'] - last['start_x'],
            'dy': last['end_y'] - last['start_y'],
            'seq_len': len(ep_df),
        }

        # Speed & direction
        ep_features['speed'] = np.sqrt(ep_features['dx']**2 + ep_features['dy']**2)
        ep_features['direction'] = np.arctan2(ep_features['dy'], ep_features['dx'])

        # Zone features
        zone_x = int(last['start_x'] // (105/6))
        zone_y = int(last['start_y'] // (68/6))
        zone_x = min(max(zone_x, 0), 5)
        zone_y = min(max(zone_y, 0), 5)
        ep_features['zone_id'] = zone_y * 6 + zone_x

        # Progression
        ep_features['progression_x'] = last['start_x'] - first['start_x']
        ep_features['progression_y'] = last['start_y'] - first['start_y']

        # Rolling stats (multi-scale)
        for col in ['start_x', 'start_y']:
            roll3 = ep_df[col].rolling(3, min_periods=1)
            ep_features[f'{col}_roll3_mean'] = roll3.mean().iloc[-1]
            ep_features[f'{col}_roll3_std'] = roll3.std().iloc[-1] if len(ep_df) > 1 else 0

            roll5 = ep_df[col].rolling(5, min_periods=1)
            ep_features[f'{col}_roll5_mean'] = roll5.mean().iloc[-1]
            ep_features[f'{col}_roll5_std'] = roll5.std().iloc[-1] if len(ep_df) > 1 else 0

        # dx, dy rolling
        dx_series = ep_df['end_x'] - ep_df['start_x']
        dy_series = ep_df['end_y'] - ep_df['start_y']

        for name, series in [('dx', dx_series), ('dy', dy_series)]:
            roll3 = series.rolling(3, min_periods=1)
            ep_features[f'{name}_roll3_mean'] = roll3.mean().iloc[-1]
            ep_features[f'{name}_roll3_std'] = roll3.std().iloc[-1] if len(ep_df) > 1 else 0

            roll5 = series.rolling(5, min_periods=1)
            ep_features[f'{name}_roll5_mean'] = roll5.mean().iloc[-1]
            ep_features[f'{name}_roll5_std'] = roll5.std().iloc[-1] if len(ep_df) > 1 else 0

        episodes.append(ep_features)

    return pd.DataFrame(episodes)

# Features
FEATURES = [
    'start_x', 'start_y', 'dx', 'dy', 'speed', 'direction',
    'seq_len', 'zone_id', 'progression_x', 'progression_y',
    'start_x_roll3_mean', 'start_x_roll3_std',
    'start_y_roll3_mean', 'start_y_roll3_std',
    'dx_roll3_mean', 'dx_roll3_std',
    'dy_roll3_mean', 'dy_roll3_std',
    'start_x_roll5_mean', 'start_x_roll5_std',
    'start_y_roll5_mean', 'start_y_roll5_std',
    'dx_roll5_mean', 'dx_roll5_std',
    'dy_roll5_mean', 'dy_roll5_std',
]
TARGETS = ['end_x', 'end_y']

print("\n[1] Train 데이터 로드...")
train_df = load_train_data()
train_df = train_df.fillna(0)
print(f"  Train: {len(train_df)}")

print("\n[2] Test 데이터 로드...")
test_df = load_test_data()
test_df = test_df.fillna(0)
print(f"  Test: {len(test_df)}")

print("\n[3] 모델 학습 (Full Train)...")
X_train = train_df[FEATURES]
y_train = train_df[TARGETS]

model = CatBoostRegressor(**BEST_PARAMS)
model.fit(X_train, y_train)

print("\n[4] 예측...")
X_test = test_df[FEATURES]
preds = model.predict(X_test)

print("\n[5] 제출 파일 생성...")
submission = pd.DataFrame({
    'game_episode': test_df['game_episode'],
    'end_x': preds[:, 0],
    'end_y': preds[:, 1]
})

output_path = SUBMISSION_DIR / "submission_optuna_cv14.19.csv"
submission.to_csv(output_path, index=False)
print(f"  저장: {output_path}")
print(f"  Shape: {submission.shape}")

print("\n[예측 분포]")
print(f"  end_x: mean={submission['end_x'].mean():.2f}, std={submission['end_x'].std():.2f}")
print(f"  end_y: mean={submission['end_y'].mean():.2f}, std={submission['end_y'].std():.2f}")

print(submission.head())
print("\n" + "="*70)
print("완료!")
print("="*70)
