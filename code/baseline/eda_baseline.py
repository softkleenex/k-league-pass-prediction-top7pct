"""
K리그 패스 좌표 예측 대회 - EDA & 베이스라인
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# 1. 데이터 로드
# =============================================================================
print("=" * 60)
print("1. 데이터 로드")
print("=" * 60)

DATA_DIR = Path(".")
train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")
sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
match_info = pd.read_csv(DATA_DIR / "match_info.csv")

print(f"Train shape: {train_df.shape}")
print(f"Test index shape: {test_df.shape}")
print(f"Sample submission shape: {sample_sub.shape}")
print(f"Match info shape: {match_info.shape}")

# =============================================================================
# 2. 기본 통계
# =============================================================================
print("\n" + "=" * 60)
print("2. 기본 통계")
print("=" * 60)

# 에피소드 수
n_train_episodes = train_df['game_episode'].nunique()
n_test_episodes = test_df['game_episode'].nunique()
print(f"Train 에피소드 수: {n_train_episodes:,}")
print(f"Test 에피소드 수: {n_test_episodes:,}")

# 경기 수
n_train_games = train_df['game_id'].nunique()
n_test_games = test_df['game_id'].nunique()
print(f"Train 경기 수: {n_train_games}")
print(f"Test 경기 수: {n_test_games}")

# 에피소드당 액션 수
episode_lengths = train_df.groupby('game_episode').size()
print(f"\n에피소드당 액션 수:")
print(f"  평균: {episode_lengths.mean():.1f}")
print(f"  중앙값: {episode_lengths.median():.1f}")
print(f"  최소: {episode_lengths.min()}")
print(f"  최대: {episode_lengths.max()}")

# =============================================================================
# 3. 타겟 분석 (end_x, end_y)
# =============================================================================
print("\n" + "=" * 60)
print("3. 타겟 분석 (마지막 패스의 end_x, end_y)")
print("=" * 60)

# 각 에피소드의 마지막 행 추출
last_actions = train_df.groupby('game_episode').last().reset_index()
print(f"마지막 액션 수: {len(last_actions)}")

# 마지막 액션의 type_name 분포
print(f"\n마지막 액션 type_name 분포:")
print(last_actions['type_name'].value_counts().head(10))

# 마지막 액션이 Pass인 경우만 필터
last_pass = last_actions[last_actions['type_name'].str.contains('Pass|Cross', na=False)]
print(f"\n마지막 액션이 Pass/Cross인 에피소드: {len(last_pass)} ({len(last_pass)/len(last_actions)*100:.1f}%)")

# end_x, end_y 통계
print(f"\n타겟 좌표 통계 (end_x):")
print(f"  평균: {last_actions['end_x'].mean():.2f}")
print(f"  표준편차: {last_actions['end_x'].std():.2f}")
print(f"  최소: {last_actions['end_x'].min():.2f}")
print(f"  최대: {last_actions['end_x'].max():.2f}")

print(f"\n타겟 좌표 통계 (end_y):")
print(f"  평균: {last_actions['end_y'].mean():.2f}")
print(f"  표준편차: {last_actions['end_y'].std():.2f}")
print(f"  최소: {last_actions['end_y'].min():.2f}")
print(f"  최대: {last_actions['end_y'].max():.2f}")

# =============================================================================
# 4. 패스 이동 패턴 분석
# =============================================================================
print("\n" + "=" * 60)
print("4. 패스 이동 패턴 분석")
print("=" * 60)

# 마지막 패스의 이동 거리/방향
last_actions['delta_x'] = last_actions['end_x'] - last_actions['start_x']
last_actions['delta_y'] = last_actions['end_y'] - last_actions['start_y']
last_actions['distance'] = np.sqrt(last_actions['delta_x']**2 + last_actions['delta_y']**2)

print(f"패스 이동 거리 통계:")
print(f"  평균: {last_actions['distance'].mean():.2f}")
print(f"  표준편차: {last_actions['distance'].std():.2f}")
print(f"  중앙값: {last_actions['distance'].median():.2f}")

print(f"\n패스 이동 방향 (delta_x) 통계:")
print(f"  평균: {last_actions['delta_x'].mean():.2f}")
print(f"  표준편차: {last_actions['delta_x'].std():.2f}")

print(f"\n패스 이동 방향 (delta_y) 통계:")
print(f"  평균: {last_actions['delta_y'].mean():.2f}")
print(f"  표준편차: {last_actions['delta_y'].std():.2f}")

# =============================================================================
# 5. type_name별 분석
# =============================================================================
print("\n" + "=" * 60)
print("5. type_name별 분석")
print("=" * 60)

type_counts = train_df['type_name'].value_counts()
print("액션 유형 분포:")
print(type_counts.head(15))

# Pass 관련 액션만
pass_actions = train_df[train_df['type_name'].str.contains('Pass|Cross', na=False)]
print(f"\nPass/Cross 액션 수: {len(pass_actions):,} ({len(pass_actions)/len(train_df)*100:.1f}%)")

# =============================================================================
# 6. 시작 위치별 패스 도착 위치 분석
# =============================================================================
print("\n" + "=" * 60)
print("6. 시작 위치별 패스 도착 위치 분석")
print("=" * 60)

# 경기장을 9개 구역으로 분할
def get_zone(x, y):
    """경기장을 9개 구역으로 분할"""
    x_zone = 0 if x < 35 else (1 if x < 70 else 2)  # 수비, 중앙, 공격
    y_zone = 0 if y < 22.67 else (1 if y < 45.33 else 2)  # 좌, 중, 우
    return x_zone * 3 + y_zone

last_actions['start_zone'] = last_actions.apply(
    lambda row: get_zone(row['start_x'], row['start_y']), axis=1
)

zone_names = ['수비-좌', '수비-중', '수비-우',
              '중앙-좌', '중앙-중', '중앙-우',
              '공격-좌', '공격-중', '공격-우']

print("시작 구역별 평균 도착 좌표:")
for zone in range(9):
    zone_data = last_actions[last_actions['start_zone'] == zone]
    if len(zone_data) > 0:
        print(f"  {zone_names[zone]:8s}: end_x={zone_data['end_x'].mean():5.1f}, "
              f"end_y={zone_data['end_y'].mean():5.1f}, n={len(zone_data)}")

# =============================================================================
# 7. 베이스라인 모델
# =============================================================================
print("\n" + "=" * 60)
print("7. 베이스라인 모델 평가 (Train 데이터 기준)")
print("=" * 60)

def euclidean_distance(pred_x, pred_y, true_x, true_y):
    """유클리드 거리 계산"""
    return np.sqrt((pred_x - true_x)**2 + (pred_y - true_y)**2)

# 결측치 제거
valid_last = last_actions.dropna(subset=['end_x', 'end_y', 'start_x', 'start_y'])
print(f"유효 샘플 수: {len(valid_last)}")

# 베이스라인 1: 전체 평균
mean_end_x = valid_last['end_x'].mean()
mean_end_y = valid_last['end_y'].mean()
baseline1_dist = euclidean_distance(mean_end_x, mean_end_y,
                                     valid_last['end_x'], valid_last['end_y'])
print(f"\n[베이스라인 1] 전체 평균 좌표 ({mean_end_x:.2f}, {mean_end_y:.2f})")
print(f"  평균 유클리드 거리: {baseline1_dist.mean():.4f}")

# 베이스라인 2: 시작점 + 평균 이동량
mean_delta_x = valid_last['delta_x'].mean()
mean_delta_y = valid_last['delta_y'].mean()
pred_x_2 = valid_last['start_x'] + mean_delta_x
pred_y_2 = valid_last['start_y'] + mean_delta_y
baseline2_dist = euclidean_distance(pred_x_2, pred_y_2,
                                     valid_last['end_x'], valid_last['end_y'])
print(f"\n[베이스라인 2] 시작점 + 평균 이동량 (dx={mean_delta_x:.2f}, dy={mean_delta_y:.2f})")
print(f"  평균 유클리드 거리: {baseline2_dist.mean():.4f}")

# 베이스라인 3: 시작점 그대로 (이동 없음)
baseline3_dist = euclidean_distance(valid_last['start_x'], valid_last['start_y'],
                                     valid_last['end_x'], valid_last['end_y'])
print(f"\n[베이스라인 3] 시작점 = 도착점 (이동 없음)")
print(f"  평균 유클리드 거리: {baseline3_dist.mean():.4f}")

# 베이스라인 4: 구역별 평균 이동량
valid_last['pred_x_zone'] = valid_last['start_x'].copy()
valid_last['pred_y_zone'] = valid_last['start_y'].copy()

for zone in range(9):
    zone_mask = valid_last['start_zone'] == zone
    if zone_mask.sum() > 0:
        zone_mean_dx = valid_last.loc[zone_mask, 'delta_x'].mean()
        zone_mean_dy = valid_last.loc[zone_mask, 'delta_y'].mean()
        valid_last.loc[zone_mask, 'pred_x_zone'] += zone_mean_dx
        valid_last.loc[zone_mask, 'pred_y_zone'] += zone_mean_dy

baseline4_dist = euclidean_distance(valid_last['pred_x_zone'], valid_last['pred_y_zone'],
                                     valid_last['end_x'], valid_last['end_y'])
print(f"\n[베이스라인 4] 시작점 + 구역별 평균 이동량")
print(f"  평균 유클리드 거리: {baseline4_dist.mean():.4f}")

# =============================================================================
# 8. 테스트 데이터 분석 및 제출 파일 생성
# =============================================================================
print("\n" + "=" * 60)
print("8. 테스트 데이터 분석 및 제출 파일 생성")
print("=" * 60)

# 테스트 에피소드 로드
test_episodes = []
for _, row in test_df.iterrows():
    ep_path = DATA_DIR / row['path']
    ep_df = pd.read_csv(ep_path)
    test_episodes.append(ep_df)

test_all = pd.concat(test_episodes, ignore_index=True)
print(f"Test 전체 액션 수: {len(test_all):,}")

# 각 에피소드의 마지막 행 추출
test_last = test_all.groupby('game_episode').last().reset_index()
print(f"Test 에피소드 수: {len(test_last)}")

# 마지막 액션 type_name 확인
print(f"\nTest 마지막 액션 type_name 분포:")
print(test_last['type_name'].value_counts().head(10))

# 제출 파일 생성 (베이스라인 2 사용: 시작점 + 평균 이동량)
submission = sample_sub.copy()

# test_last와 submission 매칭
test_last_dict = test_last.set_index('game_episode')[['start_x', 'start_y']].to_dict('index')

pred_end_x = []
pred_end_y = []

for game_ep in submission['game_episode']:
    if game_ep in test_last_dict:
        start_x = test_last_dict[game_ep]['start_x']
        start_y = test_last_dict[game_ep]['start_y']

        # 베이스라인 2: 시작점 + 평균 이동량
        pred_x = start_x + mean_delta_x
        pred_y = start_y + mean_delta_y

        # 좌표 범위 클리핑
        pred_x = np.clip(pred_x, 0, 105)
        pred_y = np.clip(pred_y, 0, 68)

        pred_end_x.append(pred_x)
        pred_end_y.append(pred_y)
    else:
        # 매칭 실패 시 전체 평균
        pred_end_x.append(mean_end_x)
        pred_end_y.append(mean_end_y)

submission['end_x'] = pred_end_x
submission['end_y'] = pred_end_y

# 저장
submission.to_csv('submission_baseline.csv', index=False)
print(f"\n제출 파일 저장: submission_baseline.csv")
print(submission.head(10))

# =============================================================================
# 9. 시각화
# =============================================================================
print("\n" + "=" * 60)
print("9. 시각화 저장")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. 패스 도착 좌표 분포
ax1 = axes[0, 0]
ax1.scatter(valid_last['end_x'], valid_last['end_y'], alpha=0.1, s=1)
ax1.set_xlim(0, 105)
ax1.set_ylim(0, 68)
ax1.set_xlabel('end_x')
ax1.set_ylabel('end_y')
ax1.set_title('Pass End Coordinates Distribution')
ax1.axhline(y=34, color='gray', linestyle='--', alpha=0.5)
ax1.axvline(x=52.5, color='gray', linestyle='--', alpha=0.5)

# 2. 패스 이동 벡터
ax2 = axes[0, 1]
ax2.scatter(valid_last['delta_x'], valid_last['delta_y'], alpha=0.1, s=1)
ax2.axhline(y=0, color='red', linestyle='-', alpha=0.5)
ax2.axvline(x=0, color='red', linestyle='-', alpha=0.5)
ax2.set_xlabel('delta_x (end_x - start_x)')
ax2.set_ylabel('delta_y (end_y - start_y)')
ax2.set_title('Pass Movement Vector')

# 3. 에피소드 길이 분포
ax3 = axes[1, 0]
ax3.hist(episode_lengths, bins=50, edgecolor='black', alpha=0.7)
ax3.set_xlabel('Episode Length (number of actions)')
ax3.set_ylabel('Frequency')
ax3.set_title('Episode Length Distribution')
ax3.axvline(x=episode_lengths.mean(), color='red', linestyle='--', label=f'Mean: {episode_lengths.mean():.1f}')
ax3.legend()

# 4. 패스 거리 분포
ax4 = axes[1, 1]
ax4.hist(valid_last['distance'].dropna(), bins=50, edgecolor='black', alpha=0.7)
ax4.set_xlabel('Pass Distance')
ax4.set_ylabel('Frequency')
ax4.set_title('Pass Distance Distribution')
ax4.axvline(x=valid_last['distance'].mean(), color='red', linestyle='--',
            label=f'Mean: {valid_last["distance"].mean():.1f}')
ax4.legend()

plt.tight_layout()
plt.savefig('eda_plots.png', dpi=150, bbox_inches='tight')
print("EDA 시각화 저장: eda_plots.png")

# =============================================================================
# 10. 요약
# =============================================================================
print("\n" + "=" * 60)
print("10. 요약")
print("=" * 60)

print(f"""
[데이터 요약]
- Train: {n_train_episodes:,} 에피소드, {len(train_df):,} 액션
- Test: {n_test_episodes:,} 에피소드
- 에피소드당 평균 액션: {episode_lengths.mean():.1f}개

[타겟 통계]
- end_x: 평균 {valid_last['end_x'].mean():.2f}, 표준편차 {valid_last['end_x'].std():.2f}
- end_y: 평균 {valid_last['end_y'].mean():.2f}, 표준편차 {valid_last['end_y'].std():.2f}
- 패스 거리: 평균 {valid_last['distance'].mean():.2f}

[베이스라인 성능 (Train CV)]
- 전체 평균: {baseline1_dist.mean():.4f}
- 시작점 + 평균 이동량: {baseline2_dist.mean():.4f} ★ Best
- 시작점 그대로: {baseline3_dist.mean():.4f}
- 구역별 평균 이동량: {baseline4_dist.mean():.4f}

[제출 파일]
- submission_baseline.csv 생성 완료
""")

print("완료!")
