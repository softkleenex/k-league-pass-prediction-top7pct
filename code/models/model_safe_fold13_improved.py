"""
K리그 패스 좌표 예측 - Fold 1-3 기반 안전 모델 (Zone Fallback 개선)

개선 사항:
1. Zone fallback에 min_samples 체크 추가
2. 샘플 부족 시 global fallback으로 안전하게 전환
3. 코드 구조 개선 (중복 제거)

예상 효과:
- CV ~0.01 향상 (더 안정적인 fallback)
- 테스트 데이터에서 안정성 증가

작성: 2025-12-11 (Week 2 준비 작업)
실행: Week 4 (D-19~13) 테스트 예정
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(".")

print("=" * 80)
print("K리그 패스 좌표 예측 - Zone Fallback 개선 버전")
print("=" * 80)

# =============================================================================
# 1. 데이터 로드
# =============================================================================
print("\n[1] 데이터 로드...")
train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")
sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")

test_episodes = []
for _, row in test_df.iterrows():
    ep_df = pd.read_csv(DATA_DIR / row['path'])
    test_episodes.append(ep_df)
test_all = pd.concat(test_episodes, ignore_index=True)

print(f"  Train episodes: {train_df['game_episode'].nunique():,}")
print(f"  Test episodes: {test_all['game_episode'].nunique():,}")

# =============================================================================
# 2. 피처 준비
# =============================================================================
print("\n[2] 피처 준비...")

def prepare_features(df):
    df = df.copy()
    df['dx'] = df['end_x'] - df['start_x']
    df['dy'] = df['end_y'] - df['start_y']
    df['prev_dx'] = df.groupby('game_episode')['dx'].shift(1).fillna(0)
    df['prev_dy'] = df.groupby('game_episode')['dy'].shift(1).fillna(0)
    return df

train_df = prepare_features(train_df)
test_all = prepare_features(test_all)

train_last = train_df.groupby('game_episode').last().reset_index()
train_last = train_last.dropna(subset=['end_x', 'end_y'])
train_last['delta_x'] = train_last['end_x'] - train_last['start_x']
train_last['delta_y'] = train_last['end_y'] - train_last['start_y']

test_last = test_all.groupby('game_episode').last().reset_index()

print(f"  Train samples: {len(train_last):,}")
print(f"  Test samples: {len(test_last):,}")

# =============================================================================
# 3. Zone 및 방향 분류 함수
# =============================================================================
print("\n[3] Zone 및 방향 분류 함수...")

def get_zone(x, y, n_x, n_y):
    """NxN Zone 분류"""
    x_zone = min(n_x - 1, int(x / (105 / n_x)))
    y_zone = min(n_y - 1, int(y / (68 / n_y)))
    return x_zone * n_y + y_zone

def get_direction_8way(prev_dx, prev_dy):
    """8방향 분류"""
    if abs(prev_dx) < 1 and abs(prev_dy) < 1:
        return 'none'

    angle = np.arctan2(prev_dy, prev_dx)
    angle_deg = np.degrees(angle)

    if -22.5 <= angle_deg < 22.5:
        return 'forward'
    elif 22.5 <= angle_deg < 67.5:
        return 'forward_up'
    elif 67.5 <= angle_deg < 112.5:
        return 'up'
    elif 112.5 <= angle_deg < 157.5:
        return 'back_up'
    elif angle_deg >= 157.5 or angle_deg < -157.5:
        return 'backward'
    elif -157.5 <= angle_deg < -112.5:
        return 'back_down'
    elif -112.5 <= angle_deg < -67.5:
        return 'down'
    else:  # -67.5 <= angle_deg < -22.5
        return 'forward_down'

# =============================================================================
# 4. 여러 모델 구성 (Zone + Direction 조합)
# =============================================================================
print("\n[4] 여러 모델 구성...")

models = [
    {'name': '5x5_8dir', 'zone': (5, 5), 'direction': True, 'min_samples': 25},
    {'name': '6x6_8dir', 'zone': (6, 6), 'direction': True, 'min_samples': 25},
    {'name': '7x7_8dir', 'zone': (7, 7), 'direction': True, 'min_samples': 20},
    {'name': '6x6_simple', 'zone': (6, 6), 'direction': False, 'min_samples': 30},
]

print(f"  총 {len(models)}개 모델 구성")

# =============================================================================
# 5. GroupKFold 교차 검증 (Fold 1-3만 평가)
# =============================================================================
print("\n[5] GroupKFold 교차 검증 (Fold 1-3 평가)...")

gkf = GroupKFold(n_splits=5)
game_ids = train_last['game_id'].values

all_model_predictions = {m['name']: [] for m in models}
all_model_fold_scores = {m['name']: [] for m in models}

for fold, (train_idx, val_idx) in enumerate(gkf.split(train_last, groups=game_ids)):
    train_fold = train_last.iloc[train_idx]
    val_fold = train_last.iloc[val_idx]

    print(f"\n  Fold {fold+1} (n={len(val_fold)}):")

    for model in models:
        n_x, n_y = model['zone']
        use_dir = model['direction']
        min_s = model['min_samples']

        # Zone 계산
        train_fold_temp = train_fold.copy()
        val_fold_temp = val_fold.copy()

        train_fold_temp['zone'] = train_fold_temp.apply(
            lambda r: get_zone(r['start_x'], r['start_y'], n_x, n_y), axis=1
        )
        val_fold_temp['zone'] = val_fold_temp.apply(
            lambda r: get_zone(r['start_x'], r['start_y'], n_x, n_y), axis=1
        )

        if use_dir:
            train_fold_temp['direction'] = train_fold_temp.apply(
                lambda r: get_direction_8way(r['prev_dx'], r['prev_dy']), axis=1
            )
            val_fold_temp['direction'] = val_fold_temp.apply(
                lambda r: get_direction_8way(r['prev_dx'], r['prev_dy']), axis=1
            )
            train_fold_temp['key'] = train_fold_temp['zone'].astype(str) + '_' + train_fold_temp['direction']
            val_fold_temp['key'] = val_fold_temp['zone'].astype(str) + '_' + val_fold_temp['direction']

            # 통계 계산
            stats = train_fold_temp.groupby('key').agg({
                'delta_x': 'median',
                'delta_y': 'median',
                'game_episode': 'count'
            }).rename(columns={'game_episode': 'count'})
        else:
            train_fold_temp['key'] = train_fold_temp['zone'].astype(str)
            val_fold_temp['key'] = val_fold_temp['zone'].astype(str)

            stats = train_fold_temp.groupby('key').agg({
                'delta_x': 'median',
                'delta_y': 'median',
                'game_episode': 'count'
            }).rename(columns={'game_episode': 'count'})

        # ===== 개선: Zone fallback에 count 추가 =====
        zone_fallback = train_fold_temp.groupby('zone').agg({
            'delta_x': 'median',
            'delta_y': 'median',
            'game_episode': 'count'  # ← 추가!
        }).rename(columns={'game_episode': 'count'})

        global_dx = train_fold_temp['delta_x'].median()
        global_dy = train_fold_temp['delta_y'].median()

        # ===== 개선: min_samples 체크 추가 =====
        def predict_row(row):
            key = row['key']

            # 1순위: Zone+Direction 조합
            if key in stats.index and stats.loc[key, 'count'] >= min_s:
                dx = stats.loc[key, 'delta_x']
                dy = stats.loc[key, 'delta_y']
            # 2순위: Zone만 (min_samples 체크 추가!)
            elif row['zone'] in zone_fallback.index and zone_fallback.loc[row['zone'], 'count'] >= min_s:
                dx = zone_fallback.loc[row['zone'], 'delta_x']
                dy = zone_fallback.loc[row['zone'], 'delta_y']
            # 3순위: Global fallback
            else:
                dx = global_dx
                dy = global_dy

            pred_x = np.clip(row['start_x'] + dx, 0, 105)
            pred_y = np.clip(row['start_y'] + dy, 0, 68)
            return pd.Series({'pred_x': pred_x, 'pred_y': pred_y})

        predictions = val_fold_temp.apply(predict_row, axis=1)
        val_fold_temp['pred_x'] = predictions['pred_x']
        val_fold_temp['pred_y'] = predictions['pred_y']

        # 점수 계산
        dist = np.sqrt((val_fold_temp['pred_x'] - val_fold['end_x'])**2 +
                      (val_fold_temp['pred_y'] - val_fold['end_y'])**2)
        cv = dist.mean()

        all_model_fold_scores[model['name']].append(cv)
        all_model_predictions[model['name']].append((val_idx, val_fold_temp[['pred_x', 'pred_y']].values))

        # Fold 1-3만 출력
        if fold < 3:
            print(f"    {model['name']:15s}: {cv:.4f}")

# Fold별 요약
print("\n" + "=" * 80)
print("Fold별 성능 요약")
print("=" * 80)

for model in models:
    scores = all_model_fold_scores[model['name']]
    print(f"\n{model['name']}:")
    print(f"  Fold 1-3 평균: {np.mean(scores[:3]):.4f} ± {np.std(scores[:3]):.4f}")
    print(f"  Fold 4-5 평균: {np.mean(scores[3:]):.4f} ± {np.std(scores[3:]):.4f}")
    print(f"  차이: {np.mean(scores[3:]) - np.mean(scores[:3]):+.4f}")
    print(f"  전체 평균: {np.mean(scores):.4f} (무시)")

# =============================================================================
# 6. Fold 1-3 기반 앙상블 (Inverse Variance Weighting)
# =============================================================================
print("\n" + "=" * 80)
print("[6] Fold 1-3 기반 앙상블 구성...")
print("=" * 80)

# Fold 1-3 분산 계산
model_variances = {}
for model in models:
    scores_13 = all_model_fold_scores[model['name']][:3]
    var = np.var(scores_13)
    model_variances[model['name']] = var

# Inverse Variance Weighting
total_inv_var = sum(1/v for v in model_variances.values())
weights = {name: (1/v)/total_inv_var for name, v in model_variances.items()}

print("\nFold 1-3 기반 앙상블 가중치:")
for name, w in weights.items():
    print(f"  {name:15s}: {w:.3f} (var={model_variances[name]:.6f})")

# 앙상블 예측 재구성
ensemble_predictions = np.zeros((len(train_last), 2))

for model_name, preds_list in all_model_predictions.items():
    w = weights[model_name]
    for val_idx, preds in preds_list:
        ensemble_predictions[val_idx] += w * preds

# 앙상블 CV 계산
ensemble_fold_scores = []
for fold, (_, val_idx) in enumerate(gkf.split(train_last, groups=game_ids)):
    val_fold = train_last.iloc[val_idx]
    ensemble_pred = ensemble_predictions[val_idx]

    dist = np.sqrt((ensemble_pred[:, 0] - val_fold['end_x'].values)**2 +
                   (ensemble_pred[:, 1] - val_fold['end_y'].values)**2)
    cv = dist.mean()
    ensemble_fold_scores.append(cv)

    if fold < 3:
        print(f"  Fold {fold+1}: {cv:.4f}")

print(f"\n앙상블 성능:")
print(f"  Fold 1-3 평균: {np.mean(ensemble_fold_scores[:3]):.4f} ± {np.std(ensemble_fold_scores[:3]):.4f}")
print(f"  Fold 4-5 평균: {np.mean(ensemble_fold_scores[3:]):.4f}")
print(f"  전체 평균: {np.mean(ensemble_fold_scores):.4f} (무시)")

# =============================================================================
# 7. Test 예측
# =============================================================================
print("\n[7] Test 예측...")

test_predictions = np.zeros((len(test_last), 2))

for model in models:
    n_x, n_y = model['zone']
    use_dir = model['direction']
    min_s = model['min_samples']
    w = weights[model['name']]

    # 전체 Train으로 통계 재계산
    train_temp = train_last.copy()
    test_temp = test_last.copy()

    train_temp['zone'] = train_temp.apply(
        lambda r: get_zone(r['start_x'], r['start_y'], n_x, n_y), axis=1
    )
    test_temp['zone'] = test_temp.apply(
        lambda r: get_zone(r['start_x'], r['start_y'], n_x, n_y), axis=1
    )

    if use_dir:
        train_temp['direction'] = train_temp.apply(
            lambda r: get_direction_8way(r['prev_dx'], r['prev_dy']), axis=1
        )
        test_temp['direction'] = test_temp.apply(
            lambda r: get_direction_8way(r['prev_dx'], r['prev_dy']), axis=1
        )
        train_temp['key'] = train_temp['zone'].astype(str) + '_' + train_temp['direction']
        test_temp['key'] = test_temp['zone'].astype(str) + '_' + test_temp['direction']

        stats = train_temp.groupby('key').agg({
            'delta_x': 'median',
            'delta_y': 'median',
            'game_episode': 'count'
        }).rename(columns={'game_episode': 'count'})
    else:
        train_temp['key'] = train_temp['zone'].astype(str)
        test_temp['key'] = test_temp['zone'].astype(str)

        stats = train_temp.groupby('key').agg({
            'delta_x': 'median',
            'delta_y': 'median',
            'game_episode': 'count'
        }).rename(columns={'game_episode': 'count'})

    # ===== 개선: Zone fallback에 count 추가 =====
    zone_fallback = train_temp.groupby('zone').agg({
        'delta_x': 'median',
        'delta_y': 'median',
        'game_episode': 'count'  # ← 추가!
    }).rename(columns={'game_episode': 'count'})

    global_dx = train_temp['delta_x'].median()
    global_dy = train_temp['delta_y'].median()

    # ===== 개선: min_samples 체크 추가 =====
    def predict_row(row):
        key = row['key']

        # 1순위: Zone+Direction 조합
        if key in stats.index and stats.loc[key, 'count'] >= min_s:
            dx = stats.loc[key, 'delta_x']
            dy = stats.loc[key, 'delta_y']
        # 2순위: Zone만 (min_samples 체크 추가!)
        elif row['zone'] in zone_fallback.index and zone_fallback.loc[row['zone'], 'count'] >= min_s:
            dx = zone_fallback.loc[row['zone'], 'delta_x']
            dy = zone_fallback.loc[row['zone'], 'delta_y']
        # 3순위: Global fallback
        else:
            dx = global_dx
            dy = global_dy

        pred_x = np.clip(row['start_x'] + dx, 0, 105)
        pred_y = np.clip(row['start_y'] + dy, 0, 68)
        return pd.Series({'pred_x': pred_x, 'pred_y': pred_y})

    model_preds = test_temp.apply(predict_row, axis=1)
    test_predictions += w * model_preds.values

print("  Test 예측 완료")

# =============================================================================
# 8. 제출 파일 생성
# =============================================================================
print("\n[8] 제출 파일 생성...")

submission = pd.DataFrame({
    'game_episode': test_last['game_episode'],
    'end_x': test_predictions[:, 0],
    'end_y': test_predictions[:, 1]
})
submission = sample_sub[['game_episode']].merge(submission, on='game_episode', how='left')
submission.to_csv('submission_safe_fold13_improved.csv', index=False)

print("  submission_safe_fold13_improved.csv 저장 완료")

# =============================================================================
# 9. 최종 요약
# =============================================================================
print("\n" + "=" * 80)
print("최종 요약")
print("=" * 80)

fold13_cv = np.mean(ensemble_fold_scores[:3])
fold13_std = np.std(ensemble_fold_scores[:3])

print(f"\n[성능]")
print(f"  Fold 1-3 CV: {fold13_cv:.4f} ± {fold13_std:.4f}")
print(f"  예상 Public: {fold13_cv + 0.03:.2f} (Gap +0.03 가정)")

print(f"\n[개선사항]")
print(f"  ✅ Zone fallback에 min_samples 체크 추가")
print(f"  ✅ 샘플 부족 시 global fallback으로 안전 전환")
print(f"  ✅ 예상 효과: CV ~0.01 향상")

print(f"\n[전략]")
print(f"  ✅ Fold 4-5 제외 (쉬운 데이터, 과적합 위험)")
print(f"  ✅ Fold 1-3 기반 Inverse Variance Weighting")
print(f"  ✅ 여러 Zone 해상도 + 방향 조합")
print(f"  ✅ 계층적 Fallback (안정성 강화)")

print(f"\n[제출 파일]")
print(f"  submission_safe_fold13_improved.csv")

print(f"\n[테스트 시기]")
print(f"  Week 4 (D-19~13): CV 확인 후 제출")
print(f"  목표: CV 16.27-16.34 범위 내")

print("\n" + "=" * 80)
print("완료!")
print("=" * 80)
