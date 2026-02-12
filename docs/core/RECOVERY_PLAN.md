# Recovery Plan: 241위 → 상위 20% 돌파

> **작성일:** 2025-12-15
> **현재:** 241/1006위 (Public 16.36)
> **목표:** 상위 20% (Public < 16.0)
> **전략:** Gradient Boosting
> **확률:** 60-70%

---

## 🎯 목표

### 단계별 목표

| 단계 | 목표 | Public | 순위 | 확률 |
|------|------|--------|------|------|
| **Phase 1** | GBM Baseline | 15.5-16.0 | ~220위 | 80% |
| **Phase 2** | Feature + Tune | 15.0-15.5 | ~180위 | 60% |
| **Phase 3** | Ensemble | < 15.0 | ~150위 | 40% |

### 최종 목표

```
Public < 16.0 (상위 20%)
개선: -0.36점 (2.2%)
기한: 2026-01-12 (D-28)
```

---

## 📅 타임라인

### Week 2 남은 기간 (D-28~D-22) - 현재

**목표:** 준비 완료

**할 일:**

1. ✅ Ultrathink 분석 (완료)
2. ✅ Recovery Plan 작성 (진행 중)
3. 🔄 빠른 실험 시스템 구축
4. 🔄 GBM Baseline 코드 작성 (10% 샘플)
5. ⏸️ 관찰 모드 유지 (제출 0회)

**산출물:**
- `code/utils/fast_experiment.py` - 10% 샘플링, 자동 CV
- `code/models/active/gbm_baseline.py` - XGBoost baseline
- `docs/EXPERIMENT_LOG.md` - 실험 기록 템플릿

### Week 3 (D-21~D-15)

**목표:** Phase 1 완료 (GBM Baseline)

**Day 1-2 (D-21~D-20):**
- XGBoost, LightGBM, CatBoost 비교 (10% 샘플)
- 기본 피처 (Zone 6x6 수준)
- 목표 CV: 15.5-16.5

**Day 3-4 (D-19~D-18):**
- 최고 모델 선택
- Full data 학습
- CV 검증 (5-fold)

**Day 5-6 (D-17~D-16):**
- 첫 제출 (검증 목적)
- Gap 확인
- 전략 조정

**Day 7 (D-15):**
- 휴식 & 정리
- Phase 2 준비

**제출:** 1-2회 (검증 목적만)

**산출물:**
- `code/models/active/gbm_v1.py` - Best GBM baseline
- Submission (검증용)
- 실험 로그

### Week 4 (D-14~D-8)

**목표:** Phase 2 완료 (Feature + Tune)

**Day 1-3 (D-14~D-12):**
- Feature engineering 강화
  - 시간 피처 (period, time_left, pressure)
  - Episode 피처 (position, early/late)
  - Interaction 피처 (zone_time, zone_position)
- 목표 CV: 14.5-15.5

**Day 4-5 (D-11~D-10):**
- Hyperparameter tuning
  - Grid search 또는 Optuna
  - CV 최적화
- 목표 CV: 14.0-15.0

**Day 6-7 (D-9~D-8):**
- Full data 학습
- 제출 & 검증
- Gap 확인

**제출:** 2-3회

**산출물:**
- `code/models/active/gbm_v2_features.py`
- `code/models/active/gbm_v3_tuned.py`
- 실험 비교 표

### Week 5 (D-7~D-0)

**목표:** Phase 3 완료 (Ensemble) & 최종 제출

**Day 1-2 (D-7~D-6):**
- Zone 10x10 실험
- Quantile regression
- 목표 CV: 14.5-15.5

**Day 3-4 (D-5~D-4):**
- Ensemble 구성
  - Zone 6x6 (16.36) 가중치 0.2
  - GBM best (15.0) 가중치 0.6
  - Zone 10x10 (15.5) 가중치 0.2
- 목표 CV: 14.0-14.5

**Day 5-6 (D-3~D-2):**
- 최종 검증
- 여러 조합 시도
- 최고 성능 선택

**Day 7 (D-1~D-0):**
- 최종 제출
- 백업 제출

**제출:** 5-10회 (집중)

**산출물:**
- `code/models/best/gbm_ensemble.py`
- Final submissions
- 최종 보고서

---

## 🛠️ Phase 1: GBM Baseline (상세)

### 목표

```
CV: 15.5-16.5
Public: 15.5-16.5 (Gap < 1.0)
개선: -0.5~-1.0점
순위: ~220위
```

### 1.1 빠른 실험 시스템 구축

**파일:** `code/utils/fast_experiment.py`

```python
"""
빠른 실험 시스템

Carla 조언:
- 10% 샘플로 빠른 테스트
- 메모리 주의
- 자주 저장
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
import time
import json

class FastExperiment:
    """빠른 실험을 위한 유틸리티"""

    def __init__(self, sample_frac=0.1, n_folds=3, random_state=42):
        self.sample_frac = sample_frac
        self.n_folds = n_folds
        self.random_state = random_state

    def load_data(self, sample=True):
        """데이터 로드 (샘플링 옵션)"""
        train_df = pd.read_csv('train.csv')

        if sample:
            # Episode 단위 샘플링 (중요!)
            episodes = train_df['game_episode'].unique()
            sampled_episodes = np.random.choice(
                episodes,
                size=int(len(episodes) * self.sample_frac),
                replace=False
            )
            train_df = train_df[train_df['game_episode'].isin(sampled_episodes)]
            print(f"  Sampled: {len(sampled_episodes)} episodes ({self.sample_frac*100:.0f}%)")

        return train_df

    def create_features(self, df):
        """피처 생성 (Episode 독립성 유지)"""
        df = df.copy()

        # Zone 6x6
        df['zone_x'] = (df['start_x'] / (105/6)).astype(int).clip(0, 5)
        df['zone_y'] = (df['start_y'] / (68/6)).astype(int).clip(0, 5)
        df['zone'] = df['zone_x'].astype(str) + '_' + df['zone_y'].astype(str)

        # Direction 8-way
        df['dx'] = df['end_x'] - df['start_x']
        df['dy'] = df['end_y'] - df['start_y']
        df['prev_dx'] = df.groupby('game_episode')['dx'].shift(1).fillna(0)
        df['prev_dy'] = df.groupby('game_episode')['dy'].shift(1).fillna(0)

        angle = np.degrees(np.arctan2(df['prev_dy'], df['prev_dx']))
        df['direction'] = ((angle + 22.5) // 45).astype(int) % 8

        # Goal
        df['goal_distance'] = np.sqrt((105 - df['start_x'])**2 + (34 - df['start_y'])**2)
        df['goal_angle'] = np.degrees(np.arctan2(34 - df['start_y'], 105 - df['start_x']))

        # Time
        df['time_left'] = 5400 - df['time_seconds']

        # Episode
        df['pass_count'] = df.groupby('game_episode').cumcount() + 1

        return df

    def run_cv(self, model, X, y, groups, feature_names=None):
        """Cross-validation"""
        gkf = GroupKFold(n_splits=self.n_folds)
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model.fit(X_train, y_train)
            pred = model.predict(X_val)

            # Euclidean distance
            dist = np.sqrt((pred[:, 0] - y_val[:, 0])**2 +
                          (pred[:, 1] - y_val[:, 1])**2)
            cv = dist.mean()
            fold_scores.append(cv)

            print(f"  Fold {fold+1}: {cv:.4f}")

        mean_cv = np.mean(fold_scores)
        std_cv = np.std(fold_scores)
        print(f"\n  Mean CV: {mean_cv:.4f} ± {std_cv:.4f}")

        return mean_cv, std_cv, fold_scores

    def log_experiment(self, name, cv, params, features, runtime):
        """실험 로그 저장"""
        log = {
            'name': name,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'cv_mean': cv[0],
            'cv_std': cv[1],
            'cv_folds': cv[2],
            'params': params,
            'features': features,
            'runtime': runtime,
            'sample_frac': self.sample_frac
        }

        # Append to log file
        with open('experiment_log.json', 'a') as f:
            f.write(json.dumps(log) + '\n')

        return log

    def compare_experiments(self, log_file='experiment_log.json'):
        """실험 비교 테이블"""
        logs = []
        with open(log_file, 'r') as f:
            for line in f:
                logs.append(json.loads(line))

        # Sort by CV
        logs = sorted(logs, key=lambda x: x['cv_mean'])

        print("\n" + "=" * 80)
        print("실험 비교")
        print("=" * 80)
        print(f"{'Rank':<5} {'Name':<20} {'CV':<10} {'Runtime':<10} {'Sample':<10}")
        print("-" * 80)

        for i, log in enumerate(logs):
            print(f"{i+1:<5} {log['name']:<20} {log['cv_mean']:<10.4f} "
                  f"{log['runtime']:<10.1f}s {log['sample_frac']*100:<10.0f}%")

        return logs
```

### 1.2 GBM Baseline 구현

**파일:** `code/models/active/gbm_baseline.py`

```python
"""
GBM Baseline

3개 라이브러리 비교:
- XGBoost
- LightGBM
- CatBoost
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import time

import sys
sys.path.append('../../utils')
from fast_experiment import FastExperiment

print("=" * 80)
print("GBM Baseline Comparison")
print("=" * 80)

# Setup
exp = FastExperiment(sample_frac=0.1, n_folds=3)

# Load data
print("\n[1] 데이터 로드...")
train_df = exp.load_data(sample=True)

# Features
print("\n[2] 피처 생성...")
train_df = exp.create_features(train_df)

# Last pass per episode
train_last = train_df.groupby('game_episode').last().reset_index()

# Feature columns
feature_cols = [
    'start_x', 'start_y',
    'zone_x', 'zone_y',
    'direction',
    'goal_distance', 'goal_angle',
    'period_id', 'time_seconds', 'time_left',
    'pass_count',
    'prev_dx', 'prev_dy'
]

X = train_last[feature_cols].values
y = train_last[['end_x', 'end_y']].values
groups = train_last['game_episode'].str.split('_').str[0].values

print(f"  X: {X.shape}")
print(f"  y: {y.shape}")
print(f"  Features: {len(feature_cols)}")

# =============================================================================
# XGBoost
# =============================================================================
print("\n" + "=" * 80)
print("[3] XGBoost")
print("=" * 80)

start = time.time()

# Separate models for x and y
xgb_x = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)

xgb_y = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)

# CV
gkf = GroupKFold(n_splits=3)
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Fit
    xgb_x.fit(X_train, y_train[:, 0])
    xgb_y.fit(X_train, y_train[:, 1])

    # Predict
    pred_x = xgb_x.predict(X_val)
    pred_y = xgb_y.predict(X_val)

    # Clip
    pred_x = np.clip(pred_x, 0, 105)
    pred_y = np.clip(pred_y, 0, 68)

    # Score
    dist = np.sqrt((pred_x - y_val[:, 0])**2 + (pred_y - y_val[:, 1])**2)
    cv = dist.mean()
    fold_scores.append(cv)

    print(f"  Fold {fold+1}: {cv:.4f}")

xgb_cv = np.mean(fold_scores)
xgb_std = np.std(fold_scores)
xgb_time = time.time() - start

print(f"\n  XGBoost CV: {xgb_cv:.4f} ± {xgb_std:.4f}")
print(f"  Runtime: {xgb_time:.1f}s")

# Log
exp.log_experiment(
    name='xgb_baseline',
    cv=(xgb_cv, xgb_std, fold_scores),
    params={'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1},
    features=feature_cols,
    runtime=xgb_time
)

# =============================================================================
# LightGBM
# =============================================================================
print("\n" + "=" * 80)
print("[4] LightGBM")
print("=" * 80)

start = time.time()

lgb_x = lgb.LGBMRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

lgb_y = lgb.LGBMRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

# CV
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    lgb_x.fit(X_train, y_train[:, 0])
    lgb_y.fit(X_train, y_train[:, 1])

    pred_x = np.clip(lgb_x.predict(X_val), 0, 105)
    pred_y = np.clip(lgb_y.predict(X_val), 0, 68)

    dist = np.sqrt((pred_x - y_val[:, 0])**2 + (pred_y - y_val[:, 1])**2)
    cv = dist.mean()
    fold_scores.append(cv)

    print(f"  Fold {fold+1}: {cv:.4f}")

lgb_cv = np.mean(fold_scores)
lgb_std = np.std(fold_scores)
lgb_time = time.time() - start

print(f"\n  LightGBM CV: {lgb_cv:.4f} ± {lgb_std:.4f}")
print(f"  Runtime: {lgb_time:.1f}s")

exp.log_experiment(
    name='lgb_baseline',
    cv=(lgb_cv, lgb_std, fold_scores),
    params={'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1},
    features=feature_cols,
    runtime=lgb_time
)

# =============================================================================
# CatBoost
# =============================================================================
print("\n" + "=" * 80)
print("[5] CatBoost")
print("=" * 80)

start = time.time()

cat_x = cb.CatBoostRegressor(
    iterations=100,
    depth=6,
    learning_rate=0.1,
    random_state=42,
    verbose=0
)

cat_y = cb.CatBoostRegressor(
    iterations=100,
    depth=6,
    learning_rate=0.1,
    random_state=42,
    verbose=0
)

# CV
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    cat_x.fit(X_train, y_train[:, 0])
    cat_y.fit(X_train, y_train[:, 1])

    pred_x = np.clip(cat_x.predict(X_val), 0, 105)
    pred_y = np.clip(cat_y.predict(X_val), 0, 68)

    dist = np.sqrt((pred_x - y_val[:, 0])**2 + (pred_y - y_val[:, 1])**2)
    cv = dist.mean()
    fold_scores.append(cv)

    print(f"  Fold {fold+1}: {cv:.4f}")

cat_cv = np.mean(fold_scores)
cat_std = np.std(fold_scores)
cat_time = time.time() - start

print(f"\n  CatBoost CV: {cat_cv:.4f} ± {cat_std:.4f}")
print(f"  Runtime: {cat_time:.1f}s")

exp.log_experiment(
    name='cat_baseline',
    cv=(cat_cv, cat_std, fold_scores),
    params={'iterations': 100, 'depth': 6, 'learning_rate': 0.1},
    features=feature_cols,
    runtime=cat_time
)

# =============================================================================
# Comparison
# =============================================================================
print("\n" + "=" * 80)
print("최종 비교")
print("=" * 80)

results = [
    ('XGBoost', xgb_cv, xgb_std, xgb_time),
    ('LightGBM', lgb_cv, lgb_std, lgb_time),
    ('CatBoost', cat_cv, cat_std, cat_time)
]

results = sorted(results, key=lambda x: x[1])

print(f"{'Rank':<5} {'Model':<12} {'CV':<12} {'Runtime':<10}")
print("-" * 50)
for i, (name, cv, std, runtime) in enumerate(results):
    print(f"{i+1:<5} {name:<12} {cv:.4f}±{std:.4f}  {runtime:.1f}s")

best = results[0]
print(f"\n✅ Best: {best[0]} (CV {best[1]:.4f})")

print("\n다음 단계:")
print("1. Full data로 학습")
print("2. 제출 & Gap 확인")
print("3. Feature engineering")
```

### 1.3 실행 순서

```bash
# 1. 빠른 실험 시스템 테스트
cd code/utils
python fast_experiment.py

# 2. GBM Baseline (10% 샘플)
cd ../models/active
python gbm_baseline.py

# 예상 결과:
# XGBoost: CV ~15.5-16.5
# LightGBM: CV ~15.5-16.5
# CatBoost: CV ~15.5-16.5
# Runtime: 10-30s (10% 샘플)

# 3. Best 모델로 Full data
# gbm_baseline.py에서 sample_frac=1.0으로 변경
# Runtime: 2-5분 (전체 데이터)
```

---

## 📊 Phase 2: Feature Engineering (Week 4)

### 추가 피처 (Episode 독립성 유지!)

```python
# 1. 시간 피처
df['is_first_half'] = (df['period_id'] == 1).astype(int)
df['is_last_10min'] = (df['time_left'] < 600).astype(int)
df['time_pressure'] = np.clip(600 - df['time_left'], 0, 600) / 600

# 2. 위치 피처
df['is_attacking_third'] = (df['start_x'] > 70).astype(int)
df['is_defensive_third'] = (df['start_x'] < 35).astype(int)
df['is_central'] = ((df['start_y'] > 23) & (df['start_y'] < 45)).astype(int)
df['distance_from_sideline'] = np.minimum(df['start_y'], 68 - df['start_y'])

# 3. Episode 피처
df['episode_length'] = df.groupby('game_episode')['game_episode'].transform('size')
df['episode_position'] = df['pass_count'] / df['episode_length']
df['is_early_pass'] = (df['episode_position'] < 0.2).astype(int)
df['is_late_pass'] = (df['episode_position'] > 0.8).astype(int)

# 4. 이전 패스 피처
df['prev_distance'] = np.sqrt(df['prev_dx']**2 + df['prev_dy']**2)
df['prev_angle'] = np.degrees(np.arctan2(df['prev_dy'], df['prev_dx']))

df['cumulative_dx'] = df.groupby('game_episode')['dx'].cumsum()
df['cumulative_dy'] = df.groupby('game_episode')['dy'].cumsum()

# 5. Interaction 피처
df['zone_time'] = df['zone'].astype(str) + '_' + df['period_id'].astype(str)
df['zone_position_bin'] = df['zone'].astype(str) + '_' + (df['episode_position'] * 5).astype(int).astype(str)
df['goal_dist_time'] = df['goal_distance'] * (1 + df['time_pressure'])
```

**중요:** 모든 피처가 `groupby('game_episode')` 내부에서 계산됨!

---

## 🎯 Phase 3: Ensemble (Week 5)

### Ensemble 전략

```python
# 1. Zone 6x6 (안정적, Gap +0.02)
zone_pred = zone_6x6_model.predict(test)

# 2. GBM Best (성능, Gap ~1.0 예상)
gbm_pred = gbm_best_model.predict(test)

# 3. Zone 10x10 (절충, Gap ~0.5 예상)
zone10_pred = zone_10x10_model.predict(test)

# Weighted ensemble
final_pred = (
    0.2 * zone_pred +
    0.6 * gbm_pred +
    0.2 * zone10_pred
)
```

### 가중치 최적화

```python
from scipy.optimize import minimize

def objective(weights):
    """CV 최소화"""
    pred = (weights[0] * zone_pred +
            weights[1] * gbm_pred +
            weights[2] * zone10_pred)
    cv = euclidean_distance(pred, y_true).mean()
    return cv

# Constraints: sum(weights) = 1, all >= 0
result = minimize(
    objective,
    x0=[0.33, 0.33, 0.33],
    constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1},
    bounds=[(0, 1)] * 3
)

optimal_weights = result.x
```

---

## ✅ 체크리스트

### Phase 1 (GBM Baseline)

- [ ] `fast_experiment.py` 작성 및 테스트
- [ ] `gbm_baseline.py` 작성
- [ ] 10% 샘플 실험 (3개 라이브러리 비교)
- [ ] Best 모델 선택
- [ ] Full data 학습
- [ ] CV 검증 (5-fold)
- [ ] 첫 제출 & Gap 확인
- [ ] 실험 로그 정리

### Phase 2 (Feature + Tune)

- [ ] 시간 피처 추가
- [ ] 위치 피처 추가
- [ ] Episode 피처 추가
- [ ] Interaction 피처 추가
- [ ] CV 개선 확인
- [ ] Hyperparameter tuning
- [ ] 제출 & Gap 확인

### Phase 3 (Ensemble)

- [ ] Zone 10x10 구현
- [ ] Quantile regression 실험
- [ ] Ensemble 구성
- [ ] 가중치 최적화
- [ ] 최종 검증
- [ ] 최종 제출

---

## 📝 제출 전 체크리스트 (필수!)

### Episode 독립성 확인

- [ ] 모든 피처가 `groupby('game_episode')` 사용?
- [ ] Train/Test 동일 방식 처리?
- [ ] 다른 episode 정보 사용 안 함?
- [ ] Cross-validation GroupKFold 사용?

### 대회 규칙 확인

- [ ] 외부 데이터 사용 안 함?
- [ ] API 호출 안 함?
- [ ] 2025.11.23 이전 모델만 사용?
- [ ] 코드 + 가중치 제출 가능?

### 제출 파일 검증

- [ ] 샘플 수: 2,414개?
- [ ] 컬럼: game_episode, end_x, end_y?
- [ ] NaN 없음?
- [ ] 범위: end_x [0, 105], end_y [0, 68]?
- [ ] 중복 game_episode 없음?

---

## 🚨 위험 관리

### Risk 1: GBM Gap 클 경우

**증상:** CV 15.0 → Public 17.0 (Gap +2.0)

**대응:**
1. Feature 단순화
2. Regularization 강화 (max_depth 감소, min_child_weight 증가)
3. Ensemble 비중 조정 (Zone 6x6 비중 증가)

### Risk 2: CV 개선 안 될 경우

**증상:** GBM CV ~16.0 (Zone 6x6 수준)

**대응:**
1. Feature engineering 재검토
2. 다른 모델 시도 (Neural Network, TabNet)
3. Zone 세분화 (10x10, 12x12)

### Risk 3: 시간 부족

**증상:** D-3인데 목표 미달성

**대응:**
1. Phase 3 생략
2. Best single model 제출
3. Zone 6x6 + Best GBM 간단 ensemble

---

## 📈 성공 지표

### Phase 1 성공

```
10% 샘플: CV 15.5-16.5
Full data: CV 15.5-16.5
First submission: Public 15.5-16.5, Gap < 1.0

→ Phase 2 진행 ✅
```

### Phase 2 성공

```
Feature: CV 14.5-15.5 (-1.0 개선)
Tune: CV 14.0-15.0 (-0.5 개선)
Submission: Public 14.5-15.5, Gap < 1.0

→ Phase 3 진행 ✅
```

### 최종 성공

```
Ensemble: CV < 14.5
Final submission: Public < 16.0
순위: 상위 20% (< 200위)

→ 목표 달성! 🎉
```

---

## 🎓 성공 요인

1. **Ultrathink 분석:** 문제 본질 이해 (표준편차 15.9m)
2. **GBM 선택:** Kaggle 표준, tabular data 최강
3. **Episode 독립성:** Data Leakage 방지
4. **빠른 실험:** 10% 샘플로 빠른 반복
5. **체계적 접근:** Phase별 명확한 목표

---

## 🔗 참고 문서

- `docs/ULTRATHINK_ANALYSIS.md` - 문제 분석
- `docs/DATA_LEAKAGE_VERIFICATION.md` - 안전 확인
- `docs/AI_CODING_CONSTRAINTS.md` - 제약 조건
- `docs/COMPETITION_INFO.md` - 대회 규정

---

**작성자:** Claude Sonnet 4.5
**작성일:** 2025-12-15
**다음 리뷰:** Phase 1 완료 시 (Week 3 말)

---

*"The best way to predict the future is to create it."*
*"Zone 6x6은 과거, GBM은 미래. 241위에서 상위 20%로!"*
