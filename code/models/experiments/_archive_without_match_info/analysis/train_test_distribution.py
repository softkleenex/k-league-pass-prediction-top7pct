"""
Train vs Test 분포 비교 분석

목적:
  1. Train/Test 주요 피처 분포 비교
  2. 분포 차이 정량화
  3. 인사이트 도출

작성일: 2025-12-21
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def load_test_data():
    """Test 데이터 로드"""
    test_meta = pd.read_csv('../../../../data/test.csv')
    test_dfs = []
    for idx, row in test_meta.iterrows():
        ep_path = Path('../../../../data') / row['path']
        if ep_path.exists():
            ep_df = pd.read_csv(ep_path)
            ep_df['game_episode'] = row['game_episode']
            test_dfs.append(ep_df)
    return pd.concat(test_dfs, ignore_index=True)


def create_basic_features(df):
    """기본 피처 생성"""
    df['zone_x'] = (df['start_x'] / (105/6)).astype(int).clip(0, 5)
    df['zone_y'] = (df['start_y'] / (68/6)).astype(int).clip(0, 5)
    df['goal_distance'] = np.sqrt((105 - df['start_x'])**2 + (34 - df['start_y'])**2)
    df['pass_count'] = df.groupby('game_episode').cumcount() + 1
    df['episode_length'] = df.groupby('game_episode')['pass_count'].transform('max')
    return df


def compare_distributions(train_df, test_df, col, n_bins=10):
    """두 데이터셋의 분포 비교"""
    train_vals = train_df[col].dropna()
    test_vals = test_df[col].dropna()

    # 기본 통계
    stats = {
        'train_mean': train_vals.mean(),
        'test_mean': test_vals.mean(),
        'train_std': train_vals.std(),
        'test_std': test_vals.std(),
        'train_min': train_vals.min(),
        'test_min': test_vals.min(),
        'train_max': train_vals.max(),
        'test_max': test_vals.max(),
    }

    # 분포 차이 (KL divergence 근사)
    all_vals = pd.concat([train_vals, test_vals])
    bins = np.linspace(all_vals.min(), all_vals.max(), n_bins + 1)

    train_hist, _ = np.histogram(train_vals, bins=bins, density=True)
    test_hist, _ = np.histogram(test_vals, bins=bins, density=True)

    # Add small value to avoid log(0)
    train_hist = train_hist + 1e-10
    test_hist = test_hist + 1e-10

    # Normalize
    train_hist = train_hist / train_hist.sum()
    test_hist = test_hist / test_hist.sum()

    # KL divergence
    kl_div = np.sum(train_hist * np.log(train_hist / test_hist))
    stats['kl_divergence'] = kl_div

    return stats


def main():
    print("\n" + "=" * 80)
    print("Train vs Test 분포 비교 분석")
    print("=" * 80)

    # Load data
    print("\n[1] 데이터 로딩...")
    train_df = pd.read_csv('../../../../data/train.csv')
    test_df = load_test_data()

    print(f"  Train: {len(train_df):,} rows, {train_df['game_episode'].nunique():,} episodes")
    print(f"  Test:  {len(test_df):,} rows, {test_df['game_episode'].nunique():,} episodes")

    # Create features
    print("\n[2] 피처 생성...")
    train_df = create_basic_features(train_df)
    test_df = create_basic_features(test_df)

    # Select last pass only (for fair comparison)
    train_last = train_df.groupby('game_episode').last().reset_index()
    test_last = test_df.groupby('game_episode').last().reset_index()

    print(f"  Train last passes: {len(train_last):,}")
    print(f"  Test last passes:  {len(test_last):,}")

    # Compare distributions
    print("\n" + "=" * 80)
    print("[3] 분포 비교 (마지막 pass 기준)")
    print("=" * 80)

    compare_cols = [
        'start_x', 'start_y', 'zone_x', 'zone_y',
        'goal_distance', 'episode_length', 'time_seconds'
    ]

    print("\n  {:<15} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
        "Feature", "Train Mean", "Test Mean", "Diff", "Diff%", "KL Div"))
    print("  " + "-" * 70)

    distribution_diffs = []

    for col in compare_cols:
        if col in train_last.columns and col in test_last.columns:
            stats = compare_distributions(train_last, test_last, col)
            diff = stats['test_mean'] - stats['train_mean']
            diff_pct = (diff / stats['train_mean'] * 100) if stats['train_mean'] != 0 else 0

            marker = "***" if abs(diff_pct) > 10 or stats['kl_divergence'] > 0.1 else ""

            print("  {:<15} {:>10.2f} {:>10.2f} {:>+10.2f} {:>+9.1f}% {:>10.4f} {}".format(
                col, stats['train_mean'], stats['test_mean'],
                diff, diff_pct, stats['kl_divergence'], marker))

            distribution_diffs.append({
                'feature': col,
                'train_mean': stats['train_mean'],
                'test_mean': stats['test_mean'],
                'diff_pct': diff_pct,
                'kl_div': stats['kl_divergence']
            })

    # Zone distribution
    print("\n" + "=" * 80)
    print("[4] Zone 분포 비교")
    print("=" * 80)

    print("\n  [Train Zone 분포]")
    train_zone = train_last.groupby(['zone_x', 'zone_y']).size().reset_index(name='count')
    train_zone['pct'] = train_zone['count'] / len(train_last) * 100
    train_zone = train_zone.sort_values('count', ascending=False)

    print("  Top 5 Zones:")
    for _, row in train_zone.head(5).iterrows():
        print(f"    Zone ({int(row['zone_x'])}, {int(row['zone_y'])}): {row['count']:,} ({row['pct']:.1f}%)")

    print("\n  [Test Zone 분포]")
    test_zone = test_last.groupby(['zone_x', 'zone_y']).size().reset_index(name='count')
    test_zone['pct'] = test_zone['count'] / len(test_last) * 100
    test_zone = test_zone.sort_values('count', ascending=False)

    print("  Top 5 Zones:")
    for _, row in test_zone.head(5).iterrows():
        print(f"    Zone ({int(row['zone_x'])}, {int(row['zone_y'])}): {row['count']:,} ({row['pct']:.1f}%)")

    # Zone difference
    print("\n  [Zone 분포 차이]")
    train_zone_dict = {(int(r['zone_x']), int(r['zone_y'])): r['pct'] for _, r in train_zone.iterrows()}
    test_zone_dict = {(int(r['zone_x']), int(r['zone_y'])): r['pct'] for _, r in test_zone.iterrows()}

    all_zones = set(train_zone_dict.keys()) | set(test_zone_dict.keys())
    zone_diffs = []
    for zone in all_zones:
        train_pct = train_zone_dict.get(zone, 0)
        test_pct = test_zone_dict.get(zone, 0)
        diff = test_pct - train_pct
        zone_diffs.append((zone, train_pct, test_pct, diff))

    zone_diffs.sort(key=lambda x: abs(x[3]), reverse=True)
    print("  Biggest differences:")
    for zone, train_pct, test_pct, diff in zone_diffs[:5]:
        print(f"    Zone {zone}: Train {train_pct:.1f}% → Test {test_pct:.1f}% (diff {diff:+.1f}%)")

    # Episode length distribution
    print("\n" + "=" * 80)
    print("[5] 에피소드 길이 분포")
    print("=" * 80)

    train_len = train_last['episode_length']
    test_len = test_last['episode_length']

    print(f"\n  Train: mean={train_len.mean():.1f}, std={train_len.std():.1f}, "
          f"min={train_len.min()}, max={train_len.max()}")
    print(f"  Test:  mean={test_len.mean():.1f}, std={test_len.std():.1f}, "
          f"min={test_len.min()}, max={test_len.max()}")

    # Binned distribution
    bins = [0, 5, 10, 15, 20, 30, 50, 100, 500]
    train_binned = pd.cut(train_len, bins=bins).value_counts(normalize=True).sort_index()
    test_binned = pd.cut(test_len, bins=bins).value_counts(normalize=True).sort_index()

    print("\n  Length Distribution:")
    print("  {:>15} {:>10} {:>10} {:>10}".format("Range", "Train%", "Test%", "Diff"))
    for idx in train_binned.index:
        t_pct = train_binned[idx] * 100
        te_pct = test_binned.get(idx, 0) * 100
        diff = te_pct - t_pct
        print("  {:>15} {:>9.1f}% {:>9.1f}% {:>+9.1f}%".format(str(idx), t_pct, te_pct, diff))

    # Summary
    print("\n" + "=" * 80)
    print("분석 요약")
    print("=" * 80)

    print("""
  핵심 발견:
  1. Train/Test 분포 유사성 확인
  2. 주요 차이점 식별
  3. Zone 분포 패턴 비교

  다음 단계:
  - 분포 차이가 큰 피처에 집중
  - Zone별 가중치 조정 고려
  - 에피소드 길이별 모델 분리 고려
""")

    # Save results
    results_df = pd.DataFrame(distribution_diffs)
    results_df.to_csv('train_test_distribution_results.csv', index=False)
    print("  결과 저장: train_test_distribution_results.csv")
    print("=" * 80)


if __name__ == '__main__':
    main()
