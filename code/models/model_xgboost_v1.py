"""
K리그 패스 좌표 예측 - XGBoost + 시퀀스 피처
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(".")

print("=" * 80)
print("K리그 패스 좌표 예측 - XGBoost + 시퀀스 피처")
print("=" * 80)

# =============================================================================
# 1. 데이터 로드
# =============================================================================
print("\n[1] 데이터 로드...")
train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")
sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")

# 테스트 에피소드 로드
test_episodes = []
for _, row in test_df.iterrows():
    ep_df = pd.read_csv(DATA_DIR / row['path'])
    test_episodes.append(ep_df)
test_all = pd.concat(test_episodes, ignore_index=True)

print(f"Train rows: {len(train_df):,}")
print(f"Test rows: {len(test_all):,}")
print(f"Train episodes: {train_df['game_episode'].nunique():,}")
print(f"Test episodes: {test_all['game_episode'].nunique():,}")

# =============================================================================
# 2. 시퀀스 피처 생성
# =============================================================================
print("\n[2] 시퀀스 피처 생성...")

def create_sequence_features(df):
    """시퀀스 기반 피처 생성"""
    df = df.copy()

    # 기본 이동량
    df['dx'] = df['end_x'] - df['start_x']
    df['dy'] = df['end_y'] - df['start_y']
    df['distance'] = np.sqrt(df['dx']**2 + df['dy']**2)

    # 이전 N개 패스 정보
    for shift in [1, 2, 3]:
        df[f'prev_{shift}_dx'] = df.groupby('game_episode')['dx'].shift(shift).fillna(0)
        df[f'prev_{shift}_dy'] = df.groupby('game_episode')['dy'].shift(shift).fillna(0)
        df[f'prev_{shift}_distance'] = df.groupby('game_episode')['distance'].shift(shift).fillna(0)

    # 3개 평균
    df['avg_dx_3'] = df[[f'dx', 'prev_1_dx', 'prev_2_dx']].mean(axis=1)
    df['avg_dy_3'] = df[[f'dy', 'prev_1_dy', 'prev_2_dy']].mean(axis=1)

    # 3개 분산
    df['std_dx_3'] = df[[f'dx', 'prev_1_dx', 'prev_2_dx']].std(axis=1).fillna(0)
    df['std_dy_3'] = df[[f'dy', 'prev_1_dy', 'prev_2_dy']].std(axis=1).fillna(0)

    # 패스 순서 (에피소드 내)
    df['pass_count'] = df.groupby('game_episode').cumcount()

    # 이동 방향 각도
    df['move_angle'] = np.arctan2(df['dy'], df['dx'])
    df['prev_1_move_angle'] = df.groupby('game_episode')['move_angle'].shift(1).fillna(0)

    # NaN 처리
    df = df.fillna(0)

    return df

print("  Train 피처 생성 중...")
train_df = create_sequence_features(train_df)

print("  Test 피처 생성 중...")
test_all = create_sequence_features(test_all)

# =============================================================================
# 3. 학습 데이터 준비
# =============================================================================
print("\n[3] 학습 데이터 준비...")

# 마지막 액션 추출
train_last = train_df.groupby('game_episode').last().reset_index()
train_last = train_last.dropna(subset=['end_x', 'end_y'])

test_last = test_all.groupby('game_episode').last().reset_index()

print(f"Train last actions: {len(train_last):,}")
print(f"Test last actions: {len(test_last):,}")

# 피처 선택
feature_cols = [
    # 현재 위치
    'start_x', 'start_y',
    # 이전 패스 정보
    'prev_1_dx', 'prev_1_dy', 'prev_1_distance',
    'prev_2_dx', 'prev_2_dy', 'prev_2_distance',
    'prev_3_dx', 'prev_3_dy', 'prev_3_distance',
    # 통계
    'avg_dx_3', 'avg_dy_3',
    'std_dx_3', 'std_dy_3',
    # 액션 정보
    'pass_count',
    # 방향
    'move_angle', 'prev_1_move_angle',
]

# 존재하는 피처만 선택
feature_cols = [c for c in feature_cols if c in train_last.columns]
print(f"사용할 피처 수: {len(feature_cols)}")

X = train_last[feature_cols].values.astype(np.float32)
y_x = train_last['end_x'].values.astype(np.float32)
y_y = train_last['end_y'].values.astype(np.float32)
game_ids = train_last['game_id'].values

X_test = test_last[feature_cols].values.astype(np.float32)

print(f"Feature shape: {X.shape}")

# =============================================================================
# 4. GroupKFold 교차 검증
# =============================================================================
print("\n[4] GroupKFold 교차 검증 (Fold 1-3 기반)...")

gkf = GroupKFold(n_splits=5)

oof_pred_x = np.zeros(len(X))
oof_pred_y = np.zeros(len(X))
test_pred_x = np.zeros(len(X_test))
test_pred_y = np.zeros(len(X_test))

fold_scores = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, groups=game_ids)):
    print(f"\n  Fold {fold+1}/5")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train_x, y_val_x = y_x[train_idx], y_x[val_idx]
    y_train_y, y_val_y = y_y[train_idx], y_y[val_idx]

    # XGBoost 파라미터
    params = {
        'max_depth': 6,
        'learning_rate': 0.1,
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0,
    }

    # X 좌표 모델
    dtrain_x = xgb.DMatrix(X_train, label=y_train_x)
    dval_x = xgb.DMatrix(X_val, label=y_val_x)

    evals_x = [(dtrain_x, 'train'), (dval_x, 'val')]

    model_x = xgb.train(
        params,
        dtrain_x,
        num_boost_round=100,
        evals=evals_x,
        early_stopping_rounds=20,
        verbose_eval=False
    )

    # Y 좌표 모델
    dtrain_y = xgb.DMatrix(X_train, label=y_train_y)
    dval_y = xgb.DMatrix(X_val, label=y_val_y)

    evals_y = [(dtrain_y, 'train'), (dval_y, 'val')]

    model_y = xgb.train(
        params,
        dtrain_y,
        num_boost_round=100,
        evals=evals_y,
        early_stopping_rounds=20,
        verbose_eval=False
    )

    # OOF 예측
    dval = xgb.DMatrix(X_val)
    oof_pred_x[val_idx] = model_x.predict(dval)
    oof_pred_y[val_idx] = model_y.predict(dval)

    # 테스트 예측
    dtest = xgb.DMatrix(X_test)
    test_pred_x += model_x.predict(dtest) / 5
    test_pred_y += model_y.predict(dtest) / 5

    # Fold 점수
    fold_dist = np.sqrt((oof_pred_x[val_idx] - y_val_x)**2 + (oof_pred_y[val_idx] - y_val_y)**2)
    fold_score = fold_dist.mean()
    fold_scores.append(fold_score)

    print(f"    Score: {fold_score:.4f}")

# 전체 OOF 점수
oof_dist = np.sqrt((oof_pred_x - y_x)**2 + (oof_pred_y - y_y)**2)
oof_score = oof_dist.mean()

# Fold 1-3 점수
fold13_scores = fold_scores[:3]
fold13_mean = np.mean(fold13_scores)
fold13_std = np.std(fold13_scores)

print("\n" + "=" * 80)
print("교차 검증 결과")
print("=" * 80)
print(f"\nFold 1-3:")
for i, score in enumerate(fold13_scores):
    print(f"  Fold {i+1}: {score:.4f}")
print(f"  평균: {fold13_mean:.4f} ± {fold13_std:.4f}")

print(f"\nFold 4-5:")
fold45_scores = fold_scores[3:]
for i, score in enumerate(fold45_scores, start=4):
    print(f"  Fold {i}: {score:.4f}")
print(f"  평균: {np.mean(fold45_scores):.4f}")

print(f"\nFold Gap (Fold 4-5 - Fold 1-3): {np.mean(fold45_scores) - fold13_mean:+.4f}")
print(f"전체 OOF Score: {oof_score:.4f}")

# =============================================================================
# 5. 제출 파일 생성
# =============================================================================
print("\n[5] 제출 파일 생성...")

# 클리핑
test_pred_x = np.clip(test_pred_x, 0, 105)
test_pred_y = np.clip(test_pred_y, 0, 68)

submission = pd.DataFrame({
    'game_episode': test_last['game_episode'],
    'end_x': test_pred_x,
    'end_y': test_pred_y
})

# sample_submission 순서에 맞추기
submission = sample_sub[['game_episode']].merge(submission, on='game_episode', how='left')
submission.to_csv('submission_xgboost_v1.csv', index=False)

print(f"  submission_xgboost_v1.csv 저장 완료")
print(f"  예측 X 범위: [{test_pred_x.min():.2f}, {test_pred_x.max():.2f}]")
print(f"  예측 Y 범위: [{test_pred_y.min():.2f}, {test_pred_y.max():.2f}]")

# =============================================================================
# 6. Zone Baseline과 비교
# =============================================================================
print("\n[6] Zone Baseline과 비교...")

def get_zone_6x6(x, y):
    x_zone = min(5, int(x / (105 / 6)))
    y_zone = min(5, int(y / (68 / 6)))
    return x_zone * 6 + y_zone

train_last['zone'] = train_last.apply(lambda r: get_zone_6x6(r['start_x'], r['start_y']), axis=1)
train_last['delta_x'] = train_last['end_x'] - train_last['start_x']
train_last['delta_y'] = train_last['end_y'] - train_last['start_y']

zone_stats = train_last.groupby('zone').agg({
    'delta_x': 'median',
    'delta_y': 'median'
}).to_dict()

# Zone Baseline CV 점수 (전체)
zone_pred_x = []
zone_pred_y = []
for _, row in train_last.iterrows():
    zone = get_zone_6x6(row['start_x'], row['start_y'])
    dx = zone_stats['delta_x'].get(zone, 0)
    dy = zone_stats['delta_y'].get(zone, 0)
    zone_pred_x.append(np.clip(row['start_x'] + dx, 0, 105))
    zone_pred_y.append(np.clip(row['start_y'] + dy, 0, 68))

zone_pred_x = np.array(zone_pred_x)
zone_pred_y = np.array(zone_pred_y)

zone_dist = np.sqrt((zone_pred_x - y_x)**2 + (zone_pred_y - y_y)**2)
zone_score = zone_dist.mean()

print(f"\n  Zone Baseline (6x6 median): {zone_score:.4f}")
print(f"  XGBoost + Sequence:         {oof_score:.4f}")
print(f"  차이:                       {oof_score - zone_score:+.4f}")

# =============================================================================
# 7. 최종 요약
# =============================================================================
print("\n" + "=" * 80)
print("최종 요약")
print("=" * 80)

print(f"\n[성능]")
print(f"  Fold 1-3 평균 CV: {fold13_mean:.4f} ± {fold13_std:.4f}")
print(f"  Fold 4-5 평균 CV: {np.mean(fold45_scores):.4f}")
print(f"  전체 OOF Score:   {oof_score:.4f}")

print(f"\n[피처 개수]")
print(f"  사용 피처: {len(feature_cols)}")
print(f"  {feature_cols}")

print(f"\n[제출 파일]")
print(f"  submission_xgboost_v1.csv")

print(f"\n[참고]")
if fold13_mean < 16.35:
    print(f"  CV {fold13_mean:.4f} - Sweet Spot 근처 (16.27-16.34)")
elif fold13_mean < 16.27:
    print(f"  CV {fold13_mean:.4f} - 경고: 과최적화 위험 (Fold Gap 확인)")
else:
    print(f"  CV {fold13_mean:.4f} - 기준선 이상")

print("\n" + "=" * 80)
print("완료!")
print("=" * 80)
