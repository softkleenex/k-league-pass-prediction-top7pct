"""
Episode 길이 분석 스크립트

목적: max_length 최적값 결정
- 50, 60, 70 중 선택
- 목표: 손실율 < 5%
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 80)
print("Episode 길이 분석")
print("=" * 80)

# 데이터 로드
DATA_DIR = Path(".")
train_df = pd.read_csv(DATA_DIR / "train.csv")

# Episode별 길이 계산
episode_lengths = train_df.groupby('game_episode').size()

print(f"\n[1] 기본 통계:")
print(f"  총 Episodes: {len(episode_lengths):,}개")
print(f"  총 패스: {len(train_df):,}개")
print(f"  평균 길이: {episode_lengths.mean():.1f}개")
print(f"  중앙값: {episode_lengths.median():.1f}개")
print(f"  표준편차: {episode_lengths.std():.1f}개")
print(f"  최소: {episode_lengths.min()}개")
print(f"  최대: {episode_lengths.max()}개")

# Percentile 분석
print(f"\n[2] Percentile 분포:")
percentiles = [50, 75, 80, 85, 90, 95, 99]
for p in percentiles:
    val = np.percentile(episode_lengths, p)
    print(f"  {p:2d}%: {val:5.0f}개")

# max_length 후보 분석
print(f"\n[3] max_length 후보 분석:")
candidates = [40, 50, 60, 70]
for max_len in candidates:
    lost_count = (episode_lengths > max_len).sum()
    lost_pct = lost_count / len(episode_lengths) * 100
    covered_pct = 100 - lost_pct

    # Padding 비율 계산
    avg_len = episode_lengths.mean()
    padding_pct = (1 - avg_len / max_len) * 100

    print(f"\n  max_length = {max_len}:")
    print(f"    커버율: {covered_pct:.1f}% ({len(episode_lengths) - lost_count:,}/{len(episode_lengths):,} episodes)")
    print(f"    손실율: {lost_pct:.1f}% ({lost_count:,} episodes)")
    print(f"    Padding: {padding_pct:.1f}% (평균 {avg_len:.1f} / {max_len})")

# 최적값 결정
print(f"\n[4] 최적값 결정:")

# 95 percentile 기준
p95 = np.percentile(episode_lengths, 95)
print(f"  95 percentile: {p95:.0f}")

if p95 <= 45:
    max_length = 50
    reason = "95%ile <= 45, 50으로 충분"
elif p95 <= 55:
    max_length = 60
    reason = "95%ile 45-55, 60 선택"
else:
    max_length = 70
    reason = "95%ile > 55, 70 필요"

# 최종 검증
lost_count = (episode_lengths > max_length).sum()
lost_pct = lost_count / len(episode_lengths) * 100
padding_pct = (1 - episode_lengths.mean() / max_length) * 100

print(f"\n  선택: max_length = {max_length}")
print(f"  이유: {reason}")
print(f"  손실율: {lost_pct:.1f}%")
print(f"  Padding: {padding_pct:.1f}%")

# 검증
if lost_pct > 5:
    print(f"\n  ⚠️  경고: 손실율 {lost_pct:.1f}% > 5%")
    print(f"  다음 후보 고려 권장")
else:
    print(f"\n  ✅ 손실율 < 5% 만족!")

# 저장
output_file = DATA_DIR / "max_length_config.txt"
with open(output_file, 'w') as f:
    f.write(str(max_length))

print(f"\n[5] 저장 완료:")
print(f"  파일: {output_file}")
print(f"  값: {max_length}")

# 분포 시각화 (텍스트)
print(f"\n[6] 길이 분포 (히스토그램):")
bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 270]
hist, _ = np.histogram(episode_lengths, bins=bins)

for i, count in enumerate(hist):
    bin_start = bins[i]
    bin_end = bins[i+1]
    pct = count / len(episode_lengths) * 100
    bar = '█' * int(pct / 2)
    print(f"  {bin_start:3d}-{bin_end:3d}: {count:5,} ({pct:5.1f}%) {bar}")

print("\n" + "=" * 80)
print(f"분석 완료! max_length = {max_length}")
print("=" * 80)
