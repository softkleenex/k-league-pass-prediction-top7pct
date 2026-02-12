"""
K리그 패스 좌표 예측 - 구역별 베이스라인 (Best)
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(".")

# 데이터 로드
train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")
sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")

# 마지막 액션 추출 (Train)
last_actions = train_df.groupby('game_episode').last().reset_index()
last_actions['delta_x'] = last_actions['end_x'] - last_actions['start_x']
last_actions['delta_y'] = last_actions['end_y'] - last_actions['start_y']

# 구역 정의 (9구역)
def get_zone(x, y):
    x_zone = 0 if x < 35 else (1 if x < 70 else 2)
    y_zone = 0 if y < 22.67 else (1 if y < 45.33 else 2)
    return x_zone * 3 + y_zone

last_actions['start_zone'] = last_actions.apply(
    lambda row: get_zone(row['start_x'], row['start_y']), axis=1
)

# 구역별 평균 이동량 계산
zone_stats = last_actions.groupby('start_zone').agg({
    'delta_x': 'mean',
    'delta_y': 'mean'
}).to_dict()

# 전체 평균 (fallback용)
mean_delta_x = last_actions['delta_x'].mean()
mean_delta_y = last_actions['delta_y'].mean()

print("구역별 평균 이동량:")
zone_names = ['수비-좌', '수비-중', '수비-우', '중앙-좌', '중앙-중', '중앙-우', '공격-좌', '공격-중', '공격-우']
for zone in range(9):
    dx = zone_stats['delta_x'].get(zone, mean_delta_x)
    dy = zone_stats['delta_y'].get(zone, mean_delta_y)
    print(f"  {zone_names[zone]:8s}: dx={dx:+6.2f}, dy={dy:+6.2f}")

# 테스트 데이터 로드
test_episodes = []
for _, row in test_df.iterrows():
    ep_df = pd.read_csv(DATA_DIR / row['path'])
    test_episodes.append(ep_df)

test_all = pd.concat(test_episodes, ignore_index=True)
test_last = test_all.groupby('game_episode').last().reset_index()

# 예측
predictions = []
for _, row in sample_sub.iterrows():
    game_ep = row['game_episode']
    test_row = test_last[test_last['game_episode'] == game_ep]

    if len(test_row) > 0:
        start_x = test_row['start_x'].values[0]
        start_y = test_row['start_y'].values[0]
        zone = get_zone(start_x, start_y)

        # 구역별 이동량 적용
        dx = zone_stats['delta_x'].get(zone, mean_delta_x)
        dy = zone_stats['delta_y'].get(zone, mean_delta_y)

        pred_x = np.clip(start_x + dx, 0, 105)
        pred_y = np.clip(start_y + dy, 0, 68)
    else:
        pred_x = 68.45  # 전체 평균
        pred_y = 33.62

    predictions.append({'game_episode': game_ep, 'end_x': pred_x, 'end_y': pred_y})

submission = pd.DataFrame(predictions)
submission.to_csv('submission_zone_baseline.csv', index=False)
print(f"\n제출 파일 저장: submission_zone_baseline.csv")
print(f"예상 스코어: ~17.57 (Train CV 기준)")
print(submission.head(10))
