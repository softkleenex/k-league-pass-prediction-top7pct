"""
피처별 과적합 위험도 분석
목표: LightGBM 모델의 31개 피처 중 과적합을 유발하는 피처 식별
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("피처별 과적합 위험도 분석")
print("=" * 80)

# 데이터 로드
train = pd.read_csv('/mnt/c/LSJ/dacon/dacon/kleague-algorithm/train.csv')

# 마지막 패스만
pass_data = train[train['type_name'] == 'Pass'].copy()
last_passes = pass_data.groupby('game_episode').tail(1).copy()

print(f"\n총 {len(last_passes):,}개 마지막 패스")

# 시퀀스 피처 생성 (model_sequence_lgbm.py 기반)
def create_sequence_features(df):
    """시퀀스 피처 생성"""
    # 기본 위치/이동 피처
    df['dx'] = df['end_x'] - df['start_x']
    df['dy'] = df['end_y'] - df['start_y']
    df['distance'] = np.sqrt(df['dx']**2 + df['dy']**2)
    df['move_angle'] = np.arctan2(df['dy'], df['dx'])

    # 골문까지 거리
    df['dist_to_goal'] = np.sqrt((105 - df['start_x'])**2 + (34 - df['start_y'])**2)

    # 시간 정보
    df['time_delta'] = df.groupby('game_episode')['time_seconds'].diff()

    # 직전 패스 정보 (prev_1)
    for col in ['dx', 'dy', 'distance', 'move_angle', 'start_x', 'start_y']:
        df[f'prev_1_{col}'] = df.groupby('game_episode')[col].shift(1)

    # 방향 변화
    df['angle_change'] = df['move_angle'] - df['prev_1_move_angle']

    # 위치 변화율
    df['x_accel'] = df['dx'] - df['prev_1_dx']
    df['y_accel'] = df['dy'] - df['prev_1_dy']

    return df

# 전체 패스에 대해 피처 생성
pass_data = create_sequence_features(pass_data)

# 마지막 패스만 추출
last_passes = pass_data.groupby('game_episode').tail(1).copy()

# 피처 리스트
feature_cols = [
    'start_x', 'start_y',
    'distance', 'move_angle', 'dist_to_goal',
    'time_delta',
    'prev_1_dx', 'prev_1_dy', 'prev_1_distance', 'prev_1_move_angle',
    'prev_1_start_x', 'prev_1_start_y',
    'angle_change', 'x_accel', 'y_accel'
]

# 결측치 처리
for col in feature_cols:
    if col in last_passes.columns:
        last_passes[col].fillna(last_passes[col].median(), inplace=True)

# 타겟
last_passes['target_x'] = last_passes['end_x']
last_passes['target_y'] = last_passes['end_y']

# 유효한 데이터만
valid_data = last_passes.dropna(subset=feature_cols + ['target_x', 'target_y', 'game_id'])

print(f"유효 데이터: {len(valid_data):,}개")

# 피처 카테고리 정의
feature_categories = {
    'Basic Position': ['start_x', 'start_y'],
    'Movement': ['distance', 'move_angle', 'dist_to_goal'],
    'Temporal': ['time_delta'],
    'Previous Pass': ['prev_1_dx', 'prev_1_dy', 'prev_1_distance', 'prev_1_move_angle',
                      'prev_1_start_x', 'prev_1_start_y'],
    'Dynamics': ['angle_change', 'x_accel', 'y_accel']
}

print("\n피처 카테고리:")
for cat, feats in feature_categories.items():
    print(f"  {cat}: {len(feats)}개")

# 1. 개별 피처 중요도 vs 과적합 위험도
print("\n" + "=" * 80)
print("[1] 개별 피처 분석")
print("=" * 80)

def evaluate_feature_subset(X, y, game_ids, features, n_splits=5):
    """피처 서브셋 평가"""
    gkf = GroupKFold(n_splits=n_splits)

    train_scores = []
    val_scores = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=game_ids)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # LightGBM 학습
        model = LGBMRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=31,
            min_child_samples=20,
            random_state=42,
            verbose=-1
        )

        model.fit(X_train[features], y_train)

        # 예측
        train_pred = model.predict(X_train[features])
        val_pred = model.predict(X_val[features])

        # MAE 계산
        train_scores.append(mean_absolute_error(y_train, train_pred))
        val_scores.append(mean_absolute_error(y_val, val_pred))

    train_mean = np.mean(train_scores)
    val_mean = np.mean(val_scores)
    gap = val_mean - train_mean

    return train_mean, val_mean, gap

# X, y 준비
X = valid_data[feature_cols]
y_x = valid_data['target_x']
y_y = valid_data['target_y']
game_ids = valid_data['game_id']

print("\n개별 피처 과적합 위험도 (target_x 기준):")
print(f"{'Feature':<20} {'Train MAE':>10} {'Val MAE':>10} {'Gap':>10} {'위험도':<10}")
print("-" * 70)

feature_risks = []

for feature in feature_cols:
    try:
        train_mae, val_mae, gap = evaluate_feature_subset(X, y_x, game_ids, [feature])
        risk = 'HIGH' if gap > 1.0 else ('MED' if gap > 0.5 else 'LOW')
        feature_risks.append({
            'feature': feature,
            'train_mae': train_mae,
            'val_mae': val_mae,
            'gap': gap,
            'risk': risk
        })
        print(f"{feature:<20} {train_mae:>10.2f} {val_mae:>10.2f} {gap:>10.2f} {risk:<10}")
    except Exception as e:
        print(f"{feature:<20} Error: {e}")

# 2. 피처 카테고리별 분석
print("\n" + "=" * 80)
print("[2] 피처 카테고리별 분석")
print("=" * 80)

print(f"\n{'Category':<20} {'Train MAE':>10} {'Val MAE':>10} {'Gap':>10} {'위험도':<10}")
print("-" * 70)

category_risks = []

for category, features in feature_categories.items():
    # 해당 카테고리 피처만 사용
    available_features = [f for f in features if f in feature_cols]
    if len(available_features) == 0:
        continue

    train_mae, val_mae, gap = evaluate_feature_subset(X, y_x, game_ids, available_features)
    risk = 'HIGH' if gap > 1.0 else ('MED' if gap > 0.5 else 'LOW')

    category_risks.append({
        'category': category,
        'n_features': len(available_features),
        'train_mae': train_mae,
        'val_mae': val_mae,
        'gap': gap,
        'risk': risk
    })

    print(f"{category:<20} {train_mae:>10.2f} {val_mae:>10.2f} {gap:>10.2f} {risk:<10}")

# 3. 전체 피처 vs 선택 피처
print("\n" + "=" * 80)
print("[3] 피처 선택 전략 비교")
print("=" * 80)

# 전체 피처
train_mae_all, val_mae_all, gap_all = evaluate_feature_subset(X, y_x, game_ids, feature_cols)
print(f"\n전체 피처 ({len(feature_cols)}개):")
print(f"  Train MAE: {train_mae_all:.2f}")
print(f"  Val MAE: {val_mae_all:.2f}")
print(f"  Gap: {gap_all:.2f}")

# 위험도 낮은 피처만 (Gap < 0.5)
low_risk_features = [f['feature'] for f in feature_risks if f['gap'] < 0.5]
if len(low_risk_features) > 0:
    train_mae_low, val_mae_low, gap_low = evaluate_feature_subset(X, y_x, game_ids, low_risk_features)
    print(f"\n저위험 피처만 ({len(low_risk_features)}개):")
    print(f"  Features: {low_risk_features}")
    print(f"  Train MAE: {train_mae_low:.2f}")
    print(f"  Val MAE: {val_mae_low:.2f}")
    print(f"  Gap: {gap_low:.2f}")
    print(f"  Improvement: {gap_all - gap_low:+.2f}")

# Basic + Movement만
basic_features = feature_categories['Basic Position'] + feature_categories['Movement']
train_mae_basic, val_mae_basic, gap_basic = evaluate_feature_subset(X, y_x, game_ids, basic_features)
print(f"\nBasic + Movement ({len(basic_features)}개):")
print(f"  Features: {basic_features}")
print(f"  Train MAE: {train_mae_basic:.2f}")
print(f"  Val MAE: {val_mae_basic:.2f}")
print(f"  Gap: {gap_basic:.2f}")

# 4. 정규화 파라미터 실험
print("\n" + "=" * 80)
print("[4] 정규화 파라미터 영향")
print("=" * 80)

regularization_configs = [
    {'name': 'Default', 'min_child_samples': 20, 'max_depth': 5, 'lambda_l1': 0, 'lambda_l2': 0},
    {'name': 'Light Reg', 'min_child_samples': 50, 'max_depth': 4, 'lambda_l1': 0.1, 'lambda_l2': 0.1},
    {'name': 'Medium Reg', 'min_child_samples': 100, 'max_depth': 3, 'lambda_l1': 0.5, 'lambda_l2': 0.5},
    {'name': 'Heavy Reg', 'min_child_samples': 200, 'max_depth': 3, 'lambda_l1': 1.0, 'lambda_l2': 1.0},
]

print(f"\n{'Config':<15} {'Train MAE':>10} {'Val MAE':>10} {'Gap':>10}")
print("-" * 50)

for config in regularization_configs:
    gkf = GroupKFold(n_splits=5)
    train_scores = []
    val_scores = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_x, groups=game_ids)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_x.iloc[train_idx], y_x.iloc[val_idx]

        model = LGBMRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=config['max_depth'],
            num_leaves=min(31, 2**config['max_depth'] - 1),
            min_child_samples=config['min_child_samples'],
            lambda_l1=config['lambda_l1'],
            lambda_l2=config['lambda_l2'],
            random_state=42,
            verbose=-1
        )

        model.fit(X_train[feature_cols], y_train)

        train_pred = model.predict(X_train[feature_cols])
        val_pred = model.predict(X_val[feature_cols])

        train_scores.append(mean_absolute_error(y_train, train_pred))
        val_scores.append(mean_absolute_error(y_val, val_pred))

    train_mean = np.mean(train_scores)
    val_mean = np.mean(val_scores)
    gap = val_mean - train_mean

    print(f"{config['name']:<15} {train_mean:>10.2f} {val_mean:>10.2f} {gap:>10.2f}")

# 5. 종합 권장사항
print("\n" + "=" * 80)
print("[5] 종합 권장사항")
print("=" * 80)

# 위험도 높은 피처
high_risk_features = [f['feature'] for f in feature_risks if f['gap'] > 1.0]
med_risk_features = [f['feature'] for f in feature_risks if 0.5 < f['gap'] <= 1.0]

print("\n과적합 위험 피처:")
if len(high_risk_features) > 0:
    print(f"  HIGH 위험: {high_risk_features}")
if len(med_risk_features) > 0:
    print(f"  MED 위험: {med_risk_features}")

# 안전한 피처
safe_features = [f['feature'] for f in feature_risks if f['gap'] <= 0.5]
print(f"\n안전한 피처 ({len(safe_features)}개):")
print(f"  {safe_features}")

# 위험도 높은 카테고리
high_risk_cats = [c for c in category_risks if c['gap'] > 1.0]
if len(high_risk_cats) > 0:
    print(f"\n위험한 피처 카테고리:")
    for cat in high_risk_cats:
        print(f"  - {cat['category']}: Gap={cat['gap']:.2f}")

print("\n" + "=" * 80)
print("추천 모델 설정:")
print("=" * 80)

print("""
1. 피처 선택 (15-20개):
   - 필수: Basic Position (start_x, start_y)
   - 추가: Movement (distance, move_angle, dist_to_goal)
   - 선택적: Previous Pass (Gap 낮은 것만)
   - 제외: High Risk Features

2. LightGBM 파라미터:
   - max_depth: 3-4 (현재 5)
   - min_child_samples: 100-200 (현재 20)
   - lambda_l1/l2: 0.5-1.0
   - learning_rate: 0.03-0.05
   - n_estimators: 100-200

3. 검증 전략:
   - GroupKFold by game_id (현재 사용 중)
   - 5-fold sufficient
   - Early stopping based on validation

4. 앙상블:
   - Zone Baseline: 60-70%
   - Regularized LightGBM: 30-40%
   - 가중 평균
""")

print("\n분석 완료!")
print("=" * 80)
