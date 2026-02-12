"""
LightGBM - Zone 통계 기반
Conservative 설정으로 과적합 방지
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import GroupKFold
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print(f"[{datetime.now().strftime('%H:%M:%S')}] LightGBM 훈련 시작")

# Zone 계산 함수
def calculate_zone(x, y, n=6):
    zone_x = min(int(x // (105 / n)), n - 1)
    zone_y = min(int(y // (68 / n)), n - 1)
    return zone_x * n + zone_y

# Direction 계산
def calculate_direction(dx, dy):
    angle = np.arctan2(dy, dx) * 180 / np.pi
    angle = (angle + 360) % 360
    return int(angle // 45)

# 데이터 로드
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Zone, Direction 계산
train['zone'] = train.apply(lambda r: calculate_zone(r['start_x'], r['start_y']), axis=1)
train['prev_dx'] = train.groupby('game_episode')['end_x'].shift(1) - train.groupby('game_episode')['start_x'].shift(1)
train['prev_dy'] = train.groupby('game_episode')['end_y'].shift(1) - train.groupby('game_episode')['start_y'].shift(1)
train['prev_dx'] = train['prev_dx'].fillna(0)
train['prev_dy'] = train['prev_dy'].fillna(0)
train['direction'] = train.apply(lambda r: calculate_direction(r['prev_dx'], r['prev_dy']) if r['prev_dx'] != 0 or r['prev_dy'] != 0 else 0, axis=1)

# Target
train['delta_x'] = train['end_x'] - train['start_x']
train['delta_y'] = train['end_y'] - train['start_y']

# 피처
features = ['zone', 'direction', 'start_x', 'start_y']
cat_features = ['zone', 'direction']

# 5-Fold CV
gkf = GroupKFold(n_splits=5)
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(train, groups=train['game_id'])):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Fold {fold+1}/5")
    
    X_train, X_val = train.iloc[train_idx][features], train.iloc[val_idx][features]
    y_train_x, y_val_x = train.iloc[train_idx]['delta_x'], train.iloc[val_idx]['delta_x']
    y_train_y, y_val_y = train.iloc[train_idx]['delta_y'], train.iloc[val_idx]['delta_y']
    
    # X 예측
    model_x = lgb.LGBMRegressor(
        num_leaves=31, max_depth=6, learning_rate=0.05,
        n_estimators=500, min_child_samples=50,
        verbose=-1, force_col_wise=True
    )
    model_x.fit(X_train, y_train_x, categorical_feature=cat_features,
                eval_set=[(X_val, y_val_x)], eval_metric='rmse',
                callbacks=[lgb.early_stopping(50, verbose=False)])
    
    # Y 예측
    model_y = lgb.LGBMRegressor(
        num_leaves=31, max_depth=6, learning_rate=0.05,
        n_estimators=500, min_child_samples=50,
        verbose=-1, force_col_wise=True
    )
    model_y.fit(X_train, y_train_y, categorical_feature=cat_features,
                eval_set=[(X_val, y_val_y)], eval_metric='rmse',
                callbacks=[lgb.early_stopping(50, verbose=False)])
    
    # 예측
    pred_x = model_x.predict(X_val)
    pred_y = model_y.predict(X_val)
    
    end_x_pred = train.iloc[val_idx]['start_x'].values + pred_x
    end_y_pred = train.iloc[val_idx]['start_y'].values + pred_y
    end_x_true = train.iloc[val_idx]['end_x'].values
    end_y_true = train.iloc[val_idx]['end_y'].values
    
    dist = np.sqrt((end_x_pred - end_x_true)**2 + (end_y_pred - end_y_true)**2).mean()
    fold_scores.append(dist)
    
    if fold < 3:
        print(f"  Fold {fold+1} Score: {dist:.4f}")

cv = np.mean(fold_scores[:3])
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] CV (Fold 1-3): {cv:.4f}")

# 결과 저장
result = {'cv_fold_1_3': float(cv), 'fold_scores': [float(s) for s in fold_scores[:3]]}
with open('checkpoints/lightgbm/results.json', 'w') as f:
    json.dump(result, f)

print(f"[{datetime.now().strftime('%H:%M:%S')}] LightGBM 완료!")
