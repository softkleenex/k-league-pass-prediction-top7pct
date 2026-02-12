"""Random Forest - Zone 통계"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print(f"[{datetime.now().strftime('%H:%M:%S')}] Random Forest 시작")

def calculate_zone(x, y, n=6):
    zone_x = min(int(x // (105 / n)), n - 1)
    zone_y = min(int(y // (68 / n)), n - 1)
    return zone_x * n + zone_y

train = pd.read_csv("train.csv")
train['zone'] = train.apply(lambda r: calculate_zone(r['start_x'], r['start_y']), axis=1)
train['delta_x'] = train['end_x'] - train['start_x']
train['delta_y'] = train['end_y'] - train['start_y']

features = ['zone', 'start_x', 'start_y']

gkf = GroupKFold(n_splits=5)
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(train, groups=train['game_id'])):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Fold {fold+1}/5")
    X_train, X_val = train.iloc[train_idx][features], train.iloc[val_idx][features]
    y_train_x, y_val_x = train.iloc[train_idx]['delta_x'], train.iloc[val_idx]['delta_x']
    y_train_y, y_val_y = train.iloc[train_idx]['delta_y'], train.iloc[val_idx]['delta_y']
    
    model_x = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=50, n_jobs=-1, random_state=42)
    model_x.fit(X_train, y_train_x)
    
    model_y = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=50, n_jobs=-1, random_state=42)
    model_y.fit(X_train, y_train_y)
    
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

with open('checkpoints/randomforest/results.json', 'w') as f:
    json.dump({'cv_fold_1_3': float(cv), 'fold_scores': [float(s) for s in fold_scores[:3]]}, f)
print(f"[{datetime.now().strftime('%H:%M:%S')}] Random Forest 완료!")
