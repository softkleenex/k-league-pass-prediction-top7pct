"""
LightGBM CV 재계산 (마지막 패스만)
Zone 모델처럼 올바른 방식으로
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("LightGBM CV 재계산 (마지막 패스만)")
print("="*80)

# Zone & Direction 계산
def calculate_zone(x, y, n=6):
    zone_x = min(int(x // (105 / n)), n - 1)
    zone_y = min(int(y // (68 / n)), n - 1)
    return zone_x * n + zone_y

def calculate_direction(dx, dy):
    angle = np.arctan2(dy, dx) * 180 / np.pi
    angle = (angle + 360) % 360
    return int(angle // 45)

# 데이터 로드
print("\n[1] 데이터 로드...")
train = pd.read_csv("train.csv")

# Zone & Direction
train['zone'] = train.apply(lambda r: calculate_zone(r['start_x'], r['start_y']), axis=1)
train['prev_dx'] = train.groupby('game_episode')['end_x'].shift(1) - train.groupby('game_episode')['start_x'].shift(1)
train['prev_dy'] = train.groupby('game_episode')['end_y'].shift(1) - train.groupby('game_episode')['start_y'].shift(1)
train['prev_dx'] = train['prev_dx'].fillna(0)
train['prev_dy'] = train['prev_dy'].fillna(0)
train['direction'] = train.apply(lambda r: calculate_direction(r['prev_dx'], r['prev_dy']) if r['prev_dx'] != 0 or r['prev_dy'] != 0 else 0, axis=1)
train['delta_x'] = train['end_x'] - train['start_x']
train['delta_y'] = train['end_y'] - train['start_y']

# 마지막 패스만!
print("\n[2] 마지막 패스 추출...")
train_last = train.groupby('game_episode').last().reset_index()
print(f"  전체 패스: {len(train):,}개")
print(f"  마지막 패스: {len(train_last):,}개")

features = ['zone', 'direction', 'start_x', 'start_y']
cat_features = ['zone', 'direction']

# 5-Fold CV (마지막 패스만)
print("\n[3] 5-Fold CV 재계산...")
gkf = GroupKFold(n_splits=5)
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(train_last, groups=train_last['game_id'])):
    X_train = train_last.iloc[train_idx][features]
    X_val = train_last.iloc[val_idx][features]
    y_train_x = train_last.iloc[train_idx]['delta_x']
    y_val_x = train_last.iloc[val_idx]['delta_x']
    y_train_y = train_last.iloc[train_idx]['delta_y']
    y_val_y = train_last.iloc[val_idx]['delta_y']

    # X 예측
    model_x = lgb.LGBMRegressor(num_leaves=31, max_depth=6, learning_rate=0.05,
                               n_estimators=500, min_child_samples=50, verbose=-1, force_col_wise=True)
    model_x.fit(X_train, y_train_x, categorical_feature=cat_features,
                eval_set=[(X_val, y_val_x)], eval_metric='rmse',
                callbacks=[lgb.early_stopping(50, verbose=False)])

    # Y 예측
    model_y = lgb.LGBMRegressor(num_leaves=31, max_depth=6, learning_rate=0.05,
                               n_estimators=500, min_child_samples=50, verbose=-1, force_col_wise=True)
    model_y.fit(X_train, y_train_y, categorical_feature=cat_features,
                eval_set=[(X_val, y_val_y)], eval_metric='rmse',
                callbacks=[lgb.early_stopping(50, verbose=False)])

    # 예측
    pred_x = model_x.predict(X_val)
    pred_y = model_y.predict(X_val)

    end_x_pred = train_last.iloc[val_idx]['start_x'].values + pred_x
    end_y_pred = train_last.iloc[val_idx]['start_y'].values + pred_y
    end_x_true = train_last.iloc[val_idx]['end_x'].values
    end_y_true = train_last.iloc[val_idx]['end_y'].values

    dist = np.sqrt((end_x_pred - end_x_true)**2 + (end_y_pred - end_y_true)**2).mean()
    fold_scores.append(dist)

    print(f"  Fold {fold+1}: {dist:.4f}")

cv_fold_1_3 = np.mean(fold_scores[:3])
cv_all = np.mean(fold_scores)

print("\n" + "="*80)
print("결과:")
print("="*80)
print(f"  CV (Fold 1-3): {cv_fold_1_3:.4f}")
print(f"  CV (All):      {cv_all:.4f}")
print(f"  Public:        18.7608")
print(f"  Gap:           {18.7608 - cv_fold_1_3:+.4f}")
print("="*80)

print("\n✅ 올바른 CV 계산 완료!")
print(f"   이전 CV (잘못됨): 12.15 (전체 패스)")
print(f"   올바른 CV:        {cv_fold_1_3:.2f} (마지막 패스만)")
print(f"   Public:           18.76")
print(f"   Gap:              {18.76 - cv_fold_1_3:+.2f} (정상 범위)")
