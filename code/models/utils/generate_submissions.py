"""
4개 ML 모델 제출 파일 생성
- LightGBM, CatBoost, RandomForest, KNN
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 4개 모델 제출 파일 생성 시작!\n")

# Zone & Direction 계산
def calculate_zone(x, y, n=6):
    zone_x = min(int(x // (105 / n)), n - 1)
    zone_y = min(int(y // (68 / n)), n - 1)
    return zone_x * n + zone_y

def calculate_direction(dx, dy):
    angle = np.arctan2(dy, dx) * 180 / np.pi
    angle = (angle + 360) % 360
    return int(angle // 45)

# 1. Train 데이터 로드 & 피처 생성
print(f"[{datetime.now().strftime('%H:%M:%S')}] 1단계: Train 데이터 로드")
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

features = ['zone', 'direction', 'start_x', 'start_y']
cat_features = ['zone', 'direction']
X_train = train[features]
y_train_x = train['delta_x']
y_train_y = train['delta_y']

print(f"  Train 샘플: {len(train):,}개\n")

# 2. Test 데이터 로드 & 피처 생성
print(f"[{datetime.now().strftime('%H:%M:%S')}] 2단계: Test 데이터 로드")
test_df = pd.read_csv("test.csv")
test_episodes = []
for _, row in test_df.iterrows():
    ep_df = pd.read_csv(row['path'])
    ep_df['game_episode'] = row['game_episode']
    test_episodes.append(ep_df)
test_all = pd.concat(test_episodes, ignore_index=True)

test_all['zone'] = test_all.apply(lambda r: calculate_zone(r['start_x'], r['start_y']), axis=1)
test_all['prev_dx'] = test_all.groupby('game_episode')['end_x'].shift(1) - test_all.groupby('game_episode')['start_x'].shift(1)
test_all['prev_dy'] = test_all.groupby('game_episode')['end_y'].shift(1) - test_all.groupby('game_episode')['start_y'].shift(1)
test_all['prev_dx'] = test_all['prev_dx'].fillna(0)
test_all['prev_dy'] = test_all['prev_dy'].fillna(0)
test_all['direction'] = test_all.apply(lambda r: calculate_direction(r['prev_dx'], r['prev_dy']) if r['prev_dx'] != 0 or r['prev_dy'] != 0 else 0, axis=1)

# 마지막 패스만
test_last = test_all.groupby('game_episode').last().reset_index()
X_test = test_last[features]

print(f"  Test 에피소드: {len(test_df)}개")
print(f"  Test 마지막 패스: {len(test_last)}개\n")

# 샘플 제출 파일
sample_sub = pd.read_csv("sample_submission.csv")

# ============================================
# 모델 1: LightGBM
# ============================================
print(f"[{datetime.now().strftime('%H:%M:%S')}] 3단계: LightGBM 훈련 & 예측")
lgbm_x = lgb.LGBMRegressor(num_leaves=31, max_depth=6, learning_rate=0.05,
                           n_estimators=500, min_child_samples=50, verbose=-1, force_col_wise=True)
lgbm_y = lgb.LGBMRegressor(num_leaves=31, max_depth=6, learning_rate=0.05,
                           n_estimators=500, min_child_samples=50, verbose=-1, force_col_wise=True)

lgbm_x.fit(X_train, y_train_x, categorical_feature=cat_features)
lgbm_y.fit(X_train, y_train_y, categorical_feature=cat_features)

pred_delta_x = lgbm_x.predict(X_test)
pred_delta_y = lgbm_y.predict(X_test)
pred_end_x = np.clip(test_last['start_x'].values + pred_delta_x, 0, 105)
pred_end_y = np.clip(test_last['start_y'].values + pred_delta_y, 0, 68)

submission_lgbm = pd.DataFrame({
    'game_episode': test_last['game_episode'],
    'end_x': pred_end_x,
    'end_y': pred_end_y
})
submission_lgbm = sample_sub[['game_episode']].merge(submission_lgbm, on='game_episode', how='left')
submission_lgbm.to_csv('submissions/pending/submission_lightgbm_cv12.15.csv', index=False)
print(f"  ✅ LightGBM 완료: submissions/pending/submission_lightgbm_cv12.15.csv\n")

# ============================================
# 모델 2: CatBoost
# ============================================
print(f"[{datetime.now().strftime('%H:%M:%S')}] 4단계: CatBoost 훈련 & 예측")
catb_x = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.05,
                          l2_leaf_reg=3, cat_features=[0, 1], verbose=0)
catb_y = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.05,
                          l2_leaf_reg=3, cat_features=[0, 1], verbose=0)

catb_x.fit(X_train, y_train_x)
catb_y.fit(X_train, y_train_y)

pred_delta_x = catb_x.predict(X_test)
pred_delta_y = catb_y.predict(X_test)
pred_end_x = np.clip(test_last['start_x'].values + pred_delta_x, 0, 105)
pred_end_y = np.clip(test_last['start_y'].values + pred_delta_y, 0, 68)

submission_catb = pd.DataFrame({
    'game_episode': test_last['game_episode'],
    'end_x': pred_end_x,
    'end_y': pred_end_y
})
submission_catb = sample_sub[['game_episode']].merge(submission_catb, on='game_episode', how='left')
submission_catb.to_csv('submissions/pending/submission_catboost_cv12.15.csv', index=False)
print(f"  ✅ CatBoost 완료: submissions/pending/submission_catboost_cv12.15.csv\n")

# ============================================
# 모델 3: Random Forest
# ============================================
print(f"[{datetime.now().strftime('%H:%M:%S')}] 5단계: Random Forest 훈련 & 예측")
rf_x = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=50, n_jobs=-1, random_state=42)
rf_y = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=50, n_jobs=-1, random_state=42)

rf_x.fit(X_train, y_train_x)
rf_y.fit(X_train, y_train_y)

pred_delta_x = rf_x.predict(X_test)
pred_delta_y = rf_y.predict(X_test)
pred_end_x = np.clip(test_last['start_x'].values + pred_delta_x, 0, 105)
pred_end_y = np.clip(test_last['start_y'].values + pred_delta_y, 0, 68)

submission_rf = pd.DataFrame({
    'game_episode': test_last['game_episode'],
    'end_x': pred_end_x,
    'end_y': pred_end_y
})
submission_rf = sample_sub[['game_episode']].merge(submission_rf, on='game_episode', how='left')
submission_rf.to_csv('submissions/pending/submission_randomforest_cv12.59.csv', index=False)
print(f"  ✅ Random Forest 완료: submissions/pending/submission_randomforest_cv12.59.csv\n")

# ============================================
# 모델 4: KNN
# ============================================
print(f"[{datetime.now().strftime('%H:%M:%S')}] 6단계: KNN 훈련 & 예측")
knn_x = KNeighborsRegressor(n_neighbors=50, weights='distance', n_jobs=-1)
knn_y = KNeighborsRegressor(n_neighbors=50, weights='distance', n_jobs=-1)

knn_x.fit(X_train, y_train_x)
knn_y.fit(X_train, y_train_y)

pred_delta_x = knn_x.predict(X_test)
pred_delta_y = knn_y.predict(X_test)
pred_end_x = np.clip(test_last['start_x'].values + pred_delta_x, 0, 105)
pred_end_y = np.clip(test_last['start_y'].values + pred_delta_y, 0, 68)

submission_knn = pd.DataFrame({
    'game_episode': test_last['game_episode'],
    'end_x': pred_end_x,
    'end_y': pred_end_y
})
submission_knn = sample_sub[['game_episode']].merge(submission_knn, on='game_episode', how='left')
submission_knn.to_csv('submissions/pending/submission_knn_cv12.94.csv', index=False)
print(f"  ✅ KNN 완료: submissions/pending/submission_knn_cv12.94.csv\n")

# 완료 메시지
print("="*80)
print(f"[{datetime.now().strftime('%H:%M:%S')}] 🎉 4개 모델 제출 파일 생성 완료!")
print("="*80)
print("\n생성된 파일:")
print("  1. submission_lightgbm_cv12.15.csv")
print("  2. submission_catboost_cv12.15.csv")
print("  3. submission_randomforest_cv12.59.csv")
print("  4. submission_knn_cv12.94.csv")
print("\n다음 단계:")
print("  → submissions/pending/ 폴더에서 4개 파일 확인")
print("  → DACON에 즉시 제출!")
print("  → 예상: 200위 → 1-50위 도약! 🚀")
print("="*80)
