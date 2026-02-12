#!/usr/bin/env python3
"""
게임 수준 EDA: Train vs Test 게임 특성 비교
2025-12-16
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime

print("=" * 80)
print("게임 수준 EDA: Train vs Test 비교")
print("=" * 80)

# 1. 데이터 로드
print("\n[1] 데이터 로드...")
train = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Test 전체 로드
test_episodes = []
for _, row in test_df.iterrows():
    ep_df = pd.read_csv(row['path'])
    test_episodes.append(ep_df)
test_all = pd.concat(test_episodes, ignore_index=True)

print(f"Train: {len(train):,} passes, {train['game_id'].nunique()} games")
print(f"Test:  {len(test_all):,} passes, {test_all['game_id'].nunique()} games")

# 2. 게임별 통계
print("\n[2] 게임별 기본 통계...")

results = {}

# Train 게임 통계
train_game_stats = train.groupby('game_id').agg({
    'episode_id': 'nunique',
    'start_x': 'mean',
    'start_y': 'mean',
    'end_x': 'mean',
    'end_y': 'mean',
    'time_seconds': 'max',
    'game_episode': 'count'
}).rename(columns={'game_episode': 'total_passes'})

train_game_stats['dx'] = train.groupby('game_id').apply(
    lambda x: (x['end_x'] - x['start_x']).mean()
).values
train_game_stats['dy'] = train.groupby('game_id').apply(
    lambda x: (x['end_y'] - x['start_y']).mean()
).values
train_game_stats['distance'] = train.groupby('game_id').apply(
    lambda x: np.sqrt((x['end_x'] - x['start_x'])**2 + (x['end_y'] - x['start_y'])**2).mean()
).values

# Test 게임 통계
test_game_stats = test_all.groupby('game_id').agg({
    'episode_id': 'nunique',
    'start_x': 'mean',
    'start_y': 'mean',
    'end_x': 'mean',
    'end_y': 'mean',
    'time_seconds': 'max',
    'game_episode': 'count'
}).rename(columns={'game_episode': 'total_passes'})

test_game_stats['dx'] = test_all.groupby('game_id').apply(
    lambda x: (x['end_x'] - x['start_x']).mean()
).values
test_game_stats['dy'] = test_all.groupby('game_id').apply(
    lambda x: (x['end_y'] - x['start_y']).mean()
).values
test_game_stats['distance'] = test_all.groupby('game_id').apply(
    lambda x: np.sqrt((x['end_x'] - x['start_x'])**2 + (x['end_y'] - x['start_y'])**2).mean()
).values

# 3. 비교 분석
print("\n[3] Train vs Test 비교...")

comparison = {}

for col in ['episode_id', 'start_x', 'start_y', 'end_x', 'end_y', 'dx', 'dy', 'distance', 'total_passes']:
    train_mean = train_game_stats[col].mean()
    test_mean = test_game_stats[col].mean()
    diff = test_mean - train_mean
    diff_pct = (diff / train_mean * 100) if train_mean != 0 else 0

    comparison[col] = {
        'train_mean': float(train_mean),
        'test_mean': float(test_mean),
        'diff': float(diff),
        'diff_pct': float(diff_pct)
    }

    print(f"\n{col}:")
    print(f"  Train: {train_mean:.2f}")
    print(f"  Test:  {test_mean:.2f}")
    print(f"  Diff:  {diff:+.2f} ({diff_pct:+.1f}%)")

results['comparison'] = comparison

# 4. 게임 ID 범위 분석
print("\n[4] 게임 ID 범위...")

train_games = sorted(train['game_id'].unique())
test_games = sorted(test_all['game_id'].unique())

game_range_info = {
    'train': {
        'min': int(train_games[0]),
        'max': int(train_games[-1]),
        'count': len(train_games),
        'range': int(train_games[-1] - train_games[0])
    },
    'test': {
        'min': int(test_games[0]),
        'max': int(test_games[-1]),
        'count': len(test_games),
        'range': int(test_games[-1] - test_games[0])
    },
    'gap': int(test_games[0] - train_games[-1])
}

print(f"\nTrain games: {game_range_info['train']['min']}-{game_range_info['train']['max']} ({game_range_info['train']['count']}개)")
print(f"Test games:  {game_range_info['test']['min']}-{game_range_info['test']['max']} ({game_range_info['test']['count']}개)")
print(f"Gap: {game_range_info['gap']:,} (Train 마지막 → Test 첫)")

results['game_range'] = game_range_info

# 5. 패스 타입 분포
print("\n[5] 패스 타입 분포...")

if 'type_name' in train.columns:
    train_types = train['type_name'].value_counts(normalize=True)
    test_types = test_all['type_name'].value_counts(normalize=True)

    type_comparison = {}
    for type_name in set(train_types.index) | set(test_types.index):
        train_pct = train_types.get(type_name, 0) * 100
        test_pct = test_types.get(type_name, 0) * 100

        type_comparison[type_name] = {
            'train_pct': float(train_pct),
            'test_pct': float(test_pct),
            'diff_pct': float(test_pct - train_pct)
        }

        print(f"\n{type_name}:")
        print(f"  Train: {train_pct:.1f}%")
        print(f"  Test:  {test_pct:.1f}%")
        print(f"  Diff:  {test_pct - train_pct:+.1f}%p")

    results['type_distribution'] = type_comparison

# 6. 필드 위치 분포
print("\n[6] 필드 위치 분포...")

# Zone 분할 (6x6)
def get_zone(x, y):
    zone_x = min(int(x / (105/6)), 5)
    zone_y = min(int(y / (68/6)), 5)
    return zone_x * 6 + zone_y

train['zone'] = train.apply(lambda row: get_zone(row['start_x'], row['start_y']), axis=1)
test_all['zone'] = test_all.apply(lambda row: get_zone(row['start_x'], row['start_y']), axis=1)

train_zones = train['zone'].value_counts(normalize=True).sort_index()
test_zones = test_all['zone'].value_counts(normalize=True).sort_index()

zone_diff = {}
for zone in range(36):
    train_pct = train_zones.get(zone, 0) * 100
    test_pct = test_zones.get(zone, 0) * 100
    zone_diff[int(zone)] = {
        'train_pct': float(train_pct),
        'test_pct': float(test_pct),
        'diff_pct': float(test_pct - train_pct)
    }

# Top 5 차이 큰 zone
zone_diff_sorted = sorted(zone_diff.items(), key=lambda x: abs(x[1]['diff_pct']), reverse=True)[:5]
print("\nTop 5 차이 큰 Zone:")
for zone, info in zone_diff_sorted:
    print(f"  Zone {zone}: Train {info['train_pct']:.1f}% vs Test {info['test_pct']:.1f}% (Diff: {info['diff_pct']:+.1f}%p)")

results['zone_distribution'] = zone_diff

# 7. 마지막 패스 특성
print("\n[7] 마지막 패스 특성...")

train['pass_number'] = train.groupby('game_episode').cumcount() + 1
train['total_passes'] = train.groupby('game_episode')['pass_number'].transform('max')
train['is_last'] = (train['pass_number'] == train['total_passes']).astype(int)

test_all['pass_number'] = test_all.groupby('game_episode').cumcount() + 1
test_all['total_passes_ep'] = test_all.groupby('game_episode')['pass_number'].transform('max')
test_all['is_last'] = (test_all['pass_number'] == test_all['total_passes_ep']).astype(int)

train_last = train[train['is_last'] == 1]
test_last = test_all[test_all['is_last'] == 1]

last_pass_stats = {}
for name, df_last, df_all in [('train', train_last, train), ('test', test_last, test_all)]:
    df_last_copy = df_last.copy()
    df_all_copy = df_all.copy()

    df_last_copy['dx'] = df_last_copy['end_x'] - df_last_copy['start_x']
    df_last_copy['dy'] = df_last_copy['end_y'] - df_last_copy['start_y']
    df_last_copy['distance'] = np.sqrt(df_last_copy['dx']**2 + df_last_copy['dy']**2)

    df_all_copy['dx'] = df_all_copy['end_x'] - df_all_copy['start_x']
    df_all_copy['dy'] = df_all_copy['end_y'] - df_all_copy['start_y']
    df_all_copy['distance'] = np.sqrt(df_all_copy['dx']**2 + df_all_copy['dy']**2)

    last_pass_stats[name] = {
        'last_dx': float(df_last_copy['dx'].mean()),
        'last_dy': float(df_last_copy['dy'].mean()),
        'last_distance': float(df_last_copy['distance'].mean()),
        'all_dx': float(df_all_copy['dx'].mean()),
        'all_dy': float(df_all_copy['dy'].mean()),
        'all_distance': float(df_all_copy['distance'].mean())
    }

    print(f"\n{name.upper()}:")
    print(f"  마지막 패스 평균 거리: {df_last_copy['distance'].mean():.2f}m")
    print(f"  전체 패스 평균 거리:   {df_all_copy['distance'].mean():.2f}m")
    print(f"  차이: {df_last_copy['distance'].mean() - df_all_copy['distance'].mean():.2f}m ({(df_last_copy['distance'].mean() / df_all_copy['distance'].mean() - 1) * 100:+.1f}%)")

results['last_pass_stats'] = last_pass_stats

# 8. 결과 저장
print("\n[8] 결과 저장...")

results['metadata'] = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'train_games': len(train_games),
    'test_games': len(test_games),
    'train_passes': len(train),
    'test_passes': len(test_all)
}

output_file = 'logs/game_eda_results.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n결과 저장: {output_file}")

print("\n" + "=" * 80)
print("게임 수준 EDA 완료!")
print("=" * 80)
