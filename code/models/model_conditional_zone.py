"""
K리그 패스 좌표 예측 - 조건부 Zone 모델
ML 없이 순수 통계 기반으로 Zone을 정교화

아이디어:
1. 공격 지역 (x > 70)에서만 더 세분화
2. 직전 패스 방향에 따른 조건부 통계
3. 시퀀스 길이에 따른 조건부 통계
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(".")

print("=" * 70)
print("K리그 패스 좌표 예측 - 조건부 Zone 모델")
print("=" * 70)

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

print(f"Train episodes: {train_df['game_episode'].nunique():,}")
print(f"Test episodes: {test_all['game_episode'].nunique():,}")

# =============================================================================
# 2. 에피소드 피처 준비
# =============================================================================
print("\n[2] 에피소드 피처 준비...")

def prepare_episode_features(df):
    """에피소드별 피처 추출"""
    df = df.copy()

    # 이동량
    df['dx'] = df['end_x'] - df['start_x']
    df['dy'] = df['end_y'] - df['start_y']

    # 직전 이동량
    df['prev_dx'] = df.groupby('game_episode')['dx'].shift(1).fillna(0)
    df['prev_dy'] = df.groupby('game_episode')['dy'].shift(1).fillna(0)

    # 시퀀스 길이
    df['seq_length'] = df.groupby('game_episode')['action_id'].transform('count')

    return df

train_df = prepare_episode_features(train_df)
test_all = prepare_episode_features(test_all)

# 마지막 액션 추출
train_last = train_df.groupby('game_episode').last().reset_index()
train_last = train_last.dropna(subset=['end_x', 'end_y'])
train_last['delta_x'] = train_last['end_x'] - train_last['start_x']
train_last['delta_y'] = train_last['end_y'] - train_last['start_y']

test_last = test_all.groupby('game_episode').last().reset_index()

print(f"Train last: {len(train_last):,}")
print(f"Test last: {len(test_last):,}")

# =============================================================================
# 3. 기본 6x6 Zone Baseline
# =============================================================================
print("\n[3] 기본 6x6 Zone Baseline...")

def get_zone_6x6(x, y):
    x_zone = min(5, int(x / (105 / 6)))
    y_zone = min(5, int(y / (68 / 6)))
    return x_zone * 6 + y_zone

train_last['zone'] = train_last.apply(lambda r: get_zone_6x6(r['start_x'], r['start_y']), axis=1)

zone_stats = train_last.groupby('zone').agg({
    'delta_x': 'median',
    'delta_y': 'median'
}).to_dict()

# 기본 예측
train_last['pred_x_base'] = train_last.apply(
    lambda r: np.clip(r['start_x'] + zone_stats['delta_x'].get(r['zone'], 0), 0, 105), axis=1
)
train_last['pred_y_base'] = train_last.apply(
    lambda r: np.clip(r['start_y'] + zone_stats['delta_y'].get(r['zone'], 0), 0, 68), axis=1
)

base_dist = np.sqrt(
    (train_last['pred_x_base'] - train_last['end_x'])**2 +
    (train_last['pred_y_base'] - train_last['end_y'])**2
)
base_cv = base_dist.mean()
print(f"기본 6x6 Zone CV: {base_cv:.4f}")

# =============================================================================
# 4. 조건부 Zone 실험들
# =============================================================================
print("\n[4] 조건부 Zone 실험...")

results = []

# 실험 1: 공격 지역 세분화 (x > 70에서 8x8)
print("\n  [4.1] 공격 지역 세분화...")

def get_adaptive_zone(x, y):
    """공격 지역에서만 더 세분화"""
    if x > 70:  # 공격 3분의1
        # 8x8 (공격 지역 35x68을 8x8로)
        x_zone = min(7, int((x - 70) / (35 / 8)))
        y_zone = min(7, int(y / (68 / 8)))
        return 100 + x_zone * 8 + y_zone  # 100+ for 공격 지역
    else:
        # 6x6 (나머지)
        x_zone = min(5, int(x / (105 / 6)))
        y_zone = min(5, int(y / (68 / 6)))
        return x_zone * 6 + y_zone

train_last['adaptive_zone'] = train_last.apply(
    lambda r: get_adaptive_zone(r['start_x'], r['start_y']), axis=1
)

adaptive_stats = train_last.groupby('adaptive_zone').agg({
    'delta_x': 'median',
    'delta_y': 'median'
}).to_dict()

train_last['pred_x_adaptive'] = train_last.apply(
    lambda r: np.clip(r['start_x'] + adaptive_stats['delta_x'].get(r['adaptive_zone'], 0), 0, 105), axis=1
)
train_last['pred_y_adaptive'] = train_last.apply(
    lambda r: np.clip(r['start_y'] + adaptive_stats['delta_y'].get(r['adaptive_zone'], 0), 0, 68), axis=1
)

adaptive_dist = np.sqrt(
    (train_last['pred_x_adaptive'] - train_last['end_x'])**2 +
    (train_last['pred_y_adaptive'] - train_last['end_y'])**2
)
adaptive_cv = adaptive_dist.mean()
print(f"  적응형 Zone CV: {adaptive_cv:.4f} (vs base {base_cv:.4f}, diff {adaptive_cv - base_cv:+.4f})")
results.append(('적응형 Zone', adaptive_cv))

# 실험 2: 직전 패스 방향 조건부
print("\n  [4.2] 직전 패스 방향 조건부...")

def get_direction_category(prev_dx, prev_dy):
    """직전 패스 방향 분류"""
    if abs(prev_dx) < 0.1 and abs(prev_dy) < 0.1:
        return 'none'
    angle = np.arctan2(prev_dy, prev_dx)
    if angle > np.pi/4:
        return 'up'
    elif angle < -np.pi/4:
        return 'down'
    elif prev_dx > 0:
        return 'forward'
    else:
        return 'backward'

train_last['direction'] = train_last.apply(
    lambda r: get_direction_category(r['prev_dx'], r['prev_dy']), axis=1
)
train_last['zone_dir'] = train_last['zone'].astype(str) + '_' + train_last['direction']

zone_dir_stats = train_last.groupby('zone_dir').agg({
    'delta_x': 'median',
    'delta_y': 'median',
    'game_episode': 'count'
}).rename(columns={'game_episode': 'count'})

# 최소 샘플 수 확인
min_samples = zone_dir_stats['count'].min()
print(f"  최소 샘플 수: {min_samples}")

# 샘플이 적은 경우 기본 zone 통계 사용
zone_dir_dict_x = zone_dir_stats['delta_x'].to_dict()
zone_dir_dict_y = zone_dir_stats['delta_y'].to_dict()
zone_dir_count = zone_dir_stats['count'].to_dict()

def get_conditional_delta(row, zone_stats, zone_dir_dict, min_count=30):
    key = str(row['zone']) + '_' + row['direction']
    if key in zone_dir_dict and zone_dir_count.get(key, 0) >= min_count:
        return zone_dir_dict[key]
    return zone_stats.get(row['zone'], 0)

train_last['pred_x_dir'] = train_last.apply(
    lambda r: np.clip(r['start_x'] + get_conditional_delta(r, zone_stats['delta_x'], zone_dir_dict_x), 0, 105), axis=1
)
train_last['pred_y_dir'] = train_last.apply(
    lambda r: np.clip(r['start_y'] + get_conditional_delta(r, zone_stats['delta_y'], zone_dir_dict_y), 0, 68), axis=1
)

dir_dist = np.sqrt(
    (train_last['pred_x_dir'] - train_last['end_x'])**2 +
    (train_last['pred_y_dir'] - train_last['end_y'])**2
)
dir_cv = dir_dist.mean()
print(f"  방향 조건부 Zone CV: {dir_cv:.4f} (vs base {base_cv:.4f}, diff {dir_cv - base_cv:+.4f})")
results.append(('방향 조건부', dir_cv))

# 실험 3: 시퀀스 길이 조건부
print("\n  [4.3] 시퀀스 길이 조건부...")

train_last['seq_cat'] = pd.cut(train_last['seq_length'], bins=[0, 10, 20, 30, 100], labels=['short', 'medium', 'long', 'vlong'])
train_last['zone_seq'] = train_last['zone'].astype(str) + '_' + train_last['seq_cat'].astype(str)

zone_seq_stats = train_last.groupby('zone_seq').agg({
    'delta_x': 'median',
    'delta_y': 'median',
    'game_episode': 'count'
}).rename(columns={'game_episode': 'count'})

zone_seq_dict_x = zone_seq_stats['delta_x'].to_dict()
zone_seq_dict_y = zone_seq_stats['delta_y'].to_dict()
zone_seq_count = zone_seq_stats['count'].to_dict()

def get_seq_conditional_delta(row, zone_stats, zone_seq_dict, zone_seq_count, min_count=30):
    key = str(row['zone']) + '_' + str(row['seq_cat'])
    if key in zone_seq_dict and zone_seq_count.get(key, 0) >= min_count:
        return zone_seq_dict[key]
    return zone_stats.get(row['zone'], 0)

train_last['pred_x_seq'] = train_last.apply(
    lambda r: np.clip(r['start_x'] + get_seq_conditional_delta(r, zone_stats['delta_x'], zone_seq_dict_x, zone_seq_count), 0, 105), axis=1
)
train_last['pred_y_seq'] = train_last.apply(
    lambda r: np.clip(r['start_y'] + get_seq_conditional_delta(r, zone_stats['delta_y'], zone_seq_dict_y, zone_seq_count), 0, 68), axis=1
)

seq_dist = np.sqrt(
    (train_last['pred_x_seq'] - train_last['end_x'])**2 +
    (train_last['pred_y_seq'] - train_last['end_y'])**2
)
seq_cv = seq_dist.mean()
print(f"  시퀀스 조건부 Zone CV: {seq_cv:.4f} (vs base {base_cv:.4f}, diff {seq_cv - base_cv:+.4f})")
results.append(('시퀀스 조건부', seq_cv))

# 실험 4: 7x7 Zone (위험하지만 테스트)
print("\n  [4.4] 7x7 Zone...")

def get_zone_7x7(x, y):
    x_zone = min(6, int(x / (105 / 7)))
    y_zone = min(6, int(y / (68 / 7)))
    return x_zone * 7 + y_zone

train_last['zone_7x7'] = train_last.apply(lambda r: get_zone_7x7(r['start_x'], r['start_y']), axis=1)

zone_7x7_stats = train_last.groupby('zone_7x7').agg({
    'delta_x': 'median',
    'delta_y': 'median',
    'game_episode': 'count'
}).rename(columns={'game_episode': 'count'})

print(f"  7x7 최소 샘플: {zone_7x7_stats['count'].min()}")

zone_7x7_dict_x = zone_7x7_stats['delta_x'].to_dict()
zone_7x7_dict_y = zone_7x7_stats['delta_y'].to_dict()

train_last['pred_x_7x7'] = train_last.apply(
    lambda r: np.clip(r['start_x'] + zone_7x7_dict_x.get(r['zone_7x7'], 0), 0, 105), axis=1
)
train_last['pred_y_7x7'] = train_last.apply(
    lambda r: np.clip(r['start_y'] + zone_7x7_dict_y.get(r['zone_7x7'], 0), 0, 68), axis=1
)

zone_7x7_dist = np.sqrt(
    (train_last['pred_x_7x7'] - train_last['end_x'])**2 +
    (train_last['pred_y_7x7'] - train_last['end_y'])**2
)
zone_7x7_cv = zone_7x7_dist.mean()
print(f"  7x7 Zone CV: {zone_7x7_cv:.4f} (vs base {base_cv:.4f}, diff {zone_7x7_cv - base_cv:+.4f})")
results.append(('7x7 Zone', zone_7x7_cv))

# 실험 5: 앙상블 (5x5 + 6x6 + 7x7)
print("\n  [4.5] Zone 앙상블 (5x5 + 6x6 + 7x7)...")

def get_zone_5x5(x, y):
    x_zone = min(4, int(x / (105 / 5)))
    y_zone = min(4, int(y / (68 / 5)))
    return x_zone * 5 + y_zone

train_last['zone_5x5'] = train_last.apply(lambda r: get_zone_5x5(r['start_x'], r['start_y']), axis=1)

zone_5x5_stats = train_last.groupby('zone_5x5').agg({
    'delta_x': 'median',
    'delta_y': 'median'
})
zone_5x5_dict_x = zone_5x5_stats['delta_x'].to_dict()
zone_5x5_dict_y = zone_5x5_stats['delta_y'].to_dict()

train_last['pred_x_5x5'] = train_last.apply(
    lambda r: np.clip(r['start_x'] + zone_5x5_dict_x.get(r['zone_5x5'], 0), 0, 105), axis=1
)
train_last['pred_y_5x5'] = train_last.apply(
    lambda r: np.clip(r['start_y'] + zone_5x5_dict_y.get(r['zone_5x5'], 0), 0, 68), axis=1
)

# 가중 앙상블
for w5, w6, w7 in [(0.3, 0.5, 0.2), (0.2, 0.6, 0.2), (0.25, 0.5, 0.25)]:
    ens_x = w5 * train_last['pred_x_5x5'] + w6 * train_last['pred_x_base'] + w7 * train_last['pred_x_7x7']
    ens_y = w5 * train_last['pred_y_5x5'] + w6 * train_last['pred_y_base'] + w7 * train_last['pred_y_7x7']

    ens_dist = np.sqrt((ens_x - train_last['end_x'])**2 + (ens_y - train_last['end_y'])**2)
    ens_cv = ens_dist.mean()
    print(f"  앙상블 ({w5:.0%}/{w6:.0%}/{w7:.0%}): CV = {ens_cv:.4f}")
    results.append((f'앙상블 {w5:.0%}/{w6:.0%}/{w7:.0%}', ens_cv))

# =============================================================================
# 5. 결과 정리
# =============================================================================
print("\n" + "=" * 70)
print("[5] 결과 정리")
print("=" * 70)

print(f"\n기준: 6x6 median CV = {base_cv:.4f}")
print("\n실험 결과:")
for name, cv in sorted(results, key=lambda x: x[1]):
    diff = cv - base_cv
    marker = "개선" if diff < 0 else "악화"
    print(f"  {name:<30} CV = {cv:.4f} ({diff:+.4f}, {marker})")

# =============================================================================
# 6. 최적 모델로 제출 파일 생성
# =============================================================================
print("\n[6] 제출 파일 생성...")

# Test 데이터 준비
test_last['zone'] = test_last.apply(lambda r: get_zone_6x6(r['start_x'], r['start_y']), axis=1)
test_last['zone_5x5'] = test_last.apply(lambda r: get_zone_5x5(r['start_x'], r['start_y']), axis=1)
test_last['zone_7x7'] = test_last.apply(lambda r: get_zone_7x7(r['start_x'], r['start_y']), axis=1)
test_last['direction'] = test_last.apply(lambda r: get_direction_category(r['prev_dx'], r['prev_dy']), axis=1)
test_last['seq_length'] = test_all.groupby('game_episode')['action_id'].transform('count').groupby(test_all['game_episode']).last().reindex(test_last['game_episode']).values
test_last['seq_cat'] = pd.cut(test_last['seq_length'], bins=[0, 10, 20, 30, 100], labels=['short', 'medium', 'long', 'vlong'])

# 6x6 기본
test_last['pred_x_base'] = test_last.apply(
    lambda r: np.clip(r['start_x'] + zone_stats['delta_x'].get(r['zone'], 0), 0, 105), axis=1
)
test_last['pred_y_base'] = test_last.apply(
    lambda r: np.clip(r['start_y'] + zone_stats['delta_y'].get(r['zone'], 0), 0, 68), axis=1
)

# 5x5
test_last['pred_x_5x5'] = test_last.apply(
    lambda r: np.clip(r['start_x'] + zone_5x5_dict_x.get(r['zone_5x5'], 0), 0, 105), axis=1
)
test_last['pred_y_5x5'] = test_last.apply(
    lambda r: np.clip(r['start_y'] + zone_5x5_dict_y.get(r['zone_5x5'], 0), 0, 68), axis=1
)

# 7x7
test_last['pred_x_7x7'] = test_last.apply(
    lambda r: np.clip(r['start_x'] + zone_7x7_dict_x.get(r['zone_7x7'], 0), 0, 105), axis=1
)
test_last['pred_y_7x7'] = test_last.apply(
    lambda r: np.clip(r['start_y'] + zone_7x7_dict_y.get(r['zone_7x7'], 0), 0, 68), axis=1
)

# 앙상블 (최적 가중치)
best_w5, best_w6, best_w7 = 0.2, 0.6, 0.2
test_last['pred_x_ens'] = best_w5 * test_last['pred_x_5x5'] + best_w6 * test_last['pred_x_base'] + best_w7 * test_last['pred_x_7x7']
test_last['pred_y_ens'] = best_w5 * test_last['pred_y_5x5'] + best_w6 * test_last['pred_y_base'] + best_w7 * test_last['pred_y_7x7']

# 제출 파일 생성
submissions = [
    ('submission_zone_ensemble_567.csv', 'pred_x_ens', 'pred_y_ens'),
]

for filename, x_col, y_col in submissions:
    sub = pd.DataFrame({
        'game_episode': test_last['game_episode'],
        'end_x': test_last[x_col],
        'end_y': test_last[y_col]
    })
    sub = sample_sub[['game_episode']].merge(sub, on='game_episode', how='left')
    sub.to_csv(filename, index=False)
    print(f"  {filename} 저장 완료")

print("\n" + "=" * 70)
print("완료!")
print("=" * 70)
