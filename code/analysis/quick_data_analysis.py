"""
빠른 데이터 분석 - 시퀀스 모델 필요성 검증

분석 목표:
1. Episode 길이 분포
2. 마지막 패스 vs 이전 패스 특성
3. 골대 접근 패턴
4. 시퀀스 의존성
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

DATA_DIR = Path(".")

print("=" * 80)
print("K리그 패스 데이터 빠른 분석")
print("=" * 80)

# 1. 데이터 로드
print("\n[1] 데이터 로드...")
train_df = pd.read_csv(DATA_DIR / "train.csv")
print(f"  전체 패스: {len(train_df):,}개")

# Episode별로 그룹화
print("\n[2] Episode별 분석...")
episodes = train_df.groupby('game_episode')
print(f"  전체 Episodes: {len(episodes):,}개")

# 샘플 100개 Episode
sample_episode_ids = list(episodes.groups.keys())[:100]
sample_episodes = [episodes.get_group(ep_id) for ep_id in sample_episode_ids]

# Episode 길이
episode_lengths = [len(ep) for ep in sample_episodes]
print(f"\n[Episode 길이 분석]")
print(f"  평균: {np.mean(episode_lengths):.1f}개")
print(f"  최소: {np.min(episode_lengths)}개")
print(f"  최대: {np.max(episode_lengths)}개")
print(f"  중앙값: {np.median(episode_lengths):.1f}개")

# 3. 마지막 패스 vs 이전 패스
print("\n[3] 마지막 패스 vs 이전 패스 비교...")

last_passes = []
prev_passes = []

for ep_df in sample_episodes:
    # 골대 거리 계산
    ep_df['goal_dist'] = np.sqrt((105 - ep_df['end_x'])**2 + (34 - ep_df['end_y'])**2)
    ep_df['dx'] = ep_df['end_x'] - ep_df['start_x']
    ep_df['dy'] = ep_df['end_y'] - ep_df['start_y']
    ep_df['distance'] = np.sqrt(ep_df['dx']**2 + ep_df['dy']**2)

    last_passes.append(ep_df.iloc[-1])
    if len(ep_df) > 1:
        prev_passes.extend([ep_df.iloc[i] for i in range(len(ep_df)-1)])

last_df = pd.DataFrame(last_passes)
prev_df = pd.DataFrame(prev_passes)

print(f"\n[골대 거리]")
print(f"  마지막 패스 평균: {last_df['goal_dist'].mean():.1f}m")
print(f"  이전 패스 평균: {prev_df['goal_dist'].mean():.1f}m")
print(f"  차이: {prev_df['goal_dist'].mean() - last_df['goal_dist'].mean():.1f}m")

print(f"\n[패스 거리]")
print(f"  마지막 패스 평균: {last_df['distance'].mean():.1f}m")
print(f"  이전 패스 평균: {prev_df['distance'].mean():.1f}m")

print(f"\n[X 방향 이동 (전진)]")
print(f"  마지막 패스 평균: {last_df['dx'].mean():.1f}m")
print(f"  이전 패스 평균: {prev_df['dx'].mean():.1f}m")

# 4. 시퀀스 의존성 분석
print("\n[4] 시퀀스 의존성 분석...")

# 마지막 패스가 이전 패스와 얼마나 다른지
correlations = []
for ep_df in sample_episodes:
    if len(ep_df) < 2:
        continue

    # 이전 패스 방향과 마지막 패스 방향 비교
    ep_df['dx'] = ep_df['end_x'] - ep_df['start_x']
    ep_df['dy'] = ep_df['end_y'] - ep_df['start_y']

    prev_dx = ep_df['dx'].iloc[-2]
    prev_dy = ep_df['dy'].iloc[-2]
    last_dx = ep_df['dx'].iloc[-1]
    last_dy = ep_df['dy'].iloc[-1]

    # 방향 유사도 (cosine similarity)
    prev_norm = np.sqrt(prev_dx**2 + prev_dy**2)
    last_norm = np.sqrt(last_dx**2 + last_dy**2)

    if prev_norm > 0 and last_norm > 0:
        cos_sim = (prev_dx * last_dx + prev_dy * last_dy) / (prev_norm * last_norm)
        correlations.append(cos_sim)

print(f"  이전 패스와 마지막 패스 방향 유사도: {np.mean(correlations):.3f}")
print(f"  (1.0 = 완전 동일, 0.0 = 수직, -1.0 = 반대)")

if np.mean(correlations) > 0.3:
    print(f"  → 시퀀스 의존성 있음! LSTM/GRU 유용할 것으로 예상")
else:
    print(f"  → 시퀀스 의존성 약함, 독립적 예측 가능")

# 5. 골대 접근 패턴
print("\n[5] 골대 접근 패턴...")

goal_approach = []
for ep_df in sample_episodes:
    ep_df['goal_dist'] = np.sqrt((105 - ep_df['end_x'])**2 + (34 - ep_df['end_y'])**2)

    # Episode가 골대에 가까워지는지
    dist_change = ep_df['goal_dist'].iloc[-1] - ep_df['goal_dist'].iloc[0]
    goal_approach.append(dist_change)

print(f"  Episode 시작 → 끝 골대 거리 변화: {np.mean(goal_approach):.1f}m")
if np.mean(goal_approach) < -5:
    print(f"  → Episode가 골대로 접근하는 경향 (공격적)")
elif np.mean(goal_approach) > 5:
    print(f"  → Episode가 골대에서 멀어지는 경향 (수비적)")
else:
    print(f"  → 골대 거리 변화 적음 (중립)")

# 6. 결론
print("\n" + "=" * 80)
print("분석 결론")
print("=" * 80)

print("\n[시퀀스 모델 필요성]")
if np.mean(correlations) > 0.3:
    print("  🔥 높음! LSTM/GRU/Transformer 시도 필요")
    print(f"  근거: 방향 유사도 {np.mean(correlations):.3f}")
else:
    print("  ⚠️ 낮음, 전통적 ML도 충분할 수 있음")

print("\n[골대 지향성]")
if abs(np.mean(goal_approach)) > 5:
    print("  ✅ 골대 접근 패턴 명확, 도메인 피처 중요")
else:
    print("  ⚠️ 골대 지향성 약함")

print("\n[Episode 길이]")
print(f"  평균 {np.mean(episode_lengths):.1f}개")
if np.mean(episode_lengths) > 15:
    print("  → LSTM이 긴 시퀀스 학습 가능")
elif np.mean(episode_lengths) > 5:
    print("  → GRU 또는 간단한 RNN 충분")
else:
    print("  → 시퀀스가 짧아 전통적 ML도 가능")

print("\n" + "=" * 80)
print("완료!")
print("=" * 80)
