# Domain Features LightGBM 모델 심층 분석

> **목적:** model_domain_features_lgbm.py (CV 14.81, Public 15.95?) 분석 및 전략 수립
> **날짜:** 2025-12-16
> **결론:** 제출 보류 (과최적화 위험 높음, 60-70% 확률로 실패)

---

## 📊 성능 요약

### 현재 확인된 지표

| 모델 | CV (Fold 1-3) | Public | Gap | 순위 | 상태 |
|------|---------------|--------|-----|------|------|
| **Domain LightGBM** | **14.81 ± 0.29** | **15.95 (추정)** | **+1.14 (추정)** | **100-150위 (추정)** | ⚠️ 미검증 |
| Zone 6x6 (safe_fold13) | 16.34 ± 0.01 | 16.36 | +0.02 | 241위 | ✅ 검증됨 |
| XGBoost (Exp 30) | 15.73 | 16.47 | +0.74 | ~250위 | ❌ 과적합 |
| LightGBM (Exp 33) | 16.45 (올바른) | 18.76 | +2.31 | ~350위 | ❌ 실패 |

**주의:** Domain LightGBM의 Public 15.95는 **추정치**이며 실제 제출 결과 없음!

---

## 🔍 핵심 차이점 분석

### 1. All Passes vs Last Pass Only

#### Domain LightGBM (All Passes)
```python
# 전체 패스 학습 (356,721개)
X = train_df[feature_cols].fillna(0)  # 모든 패스
y_x = train_df['delta_x']
y_y = train_df['delta_y']

# 마지막 패스만 평가 (15,435개)
val_last_mask = train_df.iloc[val_idx]['is_last_pass'] == 1
X_val_last = X_val_all[val_last_mask]
```

**영향:**
- ✅ **장점:** 더 많은 학습 데이터 (23배)
- ❌ **단점:** 평가와 불일치 → 과최적화 위험

#### 현재 접근법 (Last Pass Only)
```python
# 마지막 패스만 학습 + 평가 (15,435개)
train_last = train_df.groupby('game_episode').last()
```

**영향:**
- ✅ **장점:** 학습-평가 일치 → 안정적
- ❌ **단점:** 적은 데이터 → 일반화 어려움

**결론:**
- All passes 접근은 **CV를 낮추지만 Public에서 실패할 확률 높음**
- 이전 XGBoost(CV 15.73 → Public 16.47, Gap +0.74)가 동일한 패턴

---

### 2. 32 Domain Features vs 10 Features

#### Domain LightGBM (32 Features)
```python
feature_cols = [
    # 기본 위치 (2개)
    'start_x', 'start_y',

    # 골대 관련 (3개)
    'goal_distance', 'goal_angle', 'is_near_goal',

    # 필드 구역 (6개)
    'zone_attack', 'zone_defense', 'zone_middle',
    'zone_left', 'zone_center', 'zone_right',

    # 경계선 거리 (4개)
    'dist_to_left', 'dist_to_right', 'dist_to_top', 'dist_to_bottom',

    # 이전 패스 (4개)
    'prev_dx', 'prev_dy', 'prev_distance', 'direction',

    # Episode (4개)
    'episode_progress', 'episode_avg_distance', 'episode_forward_ratio', 'is_last_pass',

    # Player 통계 (4개) ⚠️ Target Encoding!
    'player_avg_dx', 'player_avg_dy', 'player_avg_distance', 'player_forward_ratio',

    # Team 통계 (3개) ⚠️ Target Encoding!
    'team_avg_dx', 'team_avg_dy', 'team_avg_distance',

    # 시간 (2개)
    'period_id', 'time_seconds'
]
```

**Feature Importance (Fold 1 예상):**
```
start_x                       : ~30,000 (가장 중요)
start_y                       : ~25,000
goal_distance                 : ~15,000
player_avg_dx                 : ~10,000 (Target Encoding)
team_avg_dx                   : ~8,000 (Target Encoding)
prev_dx                       : ~7,000
...
```

#### 현재 접근법 (Top 10 Features)
```python
# Zone 6x6: 4개 핵심 피처만
features = ['start_x', 'start_y', 'prev_dx', 'prev_dy']
# Zone + Direction으로 암묵적 피처 생성
```

**비교:**
| 카테고리 | Domain LightGBM | Zone 6x6 | 차이 |
|----------|----------------|----------|------|
| **위치 정보** | start_x, start_y, goal_distance, goal_angle, 경계선 4개 (8개) | start_x, start_y via Zone (간접) | 더 풍부 |
| **이전 패스** | prev_dx, prev_dy, prev_distance, direction (4개) | prev_dx, prev_dy via Direction (간접) | 더 명시적 |
| **축구 도메인** | 골대, 필드 구역, 전술적 위치 (9개) | 없음 | ✅ 혁신적 |
| **Player/Team** | Target Encoding 7개 | 없음 | ⚠️ 위험 |
| **Episode** | 4개 | 없음 | ✅ 유용 가능 |

**결론:**
- Domain features는 **더 많은 정보**를 담고 있음 → CV 낮아짐
- 하지만 **Target Encoding (7개)은 과적합 주범**
- 축구 도메인 피처 (골대 거리 등)는 **합리적**

---

### 3. Sample Weighting 전략

#### Domain LightGBM
```python
# 마지막 패스에 10배 가중치
sample_weights = np.where(train_df['is_last_pass'] == 1, 10.0, 1.0)

train_data_x = lgb.Dataset(X_train, label=y_train_x,
                            categorical_feature=categorical_features,
                            weight=train_weights)
```

**분석:**
- 전체 356,721 패스 중 마지막 15,435개 (4.3%)에 10배 가중치
- 실질적 학습 비중: 마지막 패스 ~30%, 나머지 패스 ~70%
- **목적:** 마지막 패스 집중 학습
- **위험:** 중간 패스 노이즈도 함께 학습 → 과적합

#### 현재 접근법
```python
# 마지막 패스만 학습 (가중치 없음)
train_last = train_df.groupby('game_episode').last()
```

**비교:**
| 접근법 | 마지막 패스 학습 | 중간 패스 학습 | 과적합 위험 |
|--------|-----------------|---------------|-------------|
| Domain (가중치) | ✅ 30% | ⚠️ 70% | 높음 |
| Zone (마지막만) | ✅ 100% | ❌ 0% | 낮음 |

**결론:**
- Sample weighting은 **절충안**이지만 중간 패스 노이즈 포함
- 마지막 패스만 학습이 **더 안전**

---

## 🚨 Gap +1.14 원인 분석

### 1. Overfitting 요인 우선순위

#### 1순위: Target Encoding (Player/Team Stats)
```python
# Train 전체로 통계 계산 → Test에 Merge
player_stats = train_df.groupby('player_id').agg({
    'delta_x': 'mean',
    'delta_y': 'mean',
    ...
})

test_all = test_all.merge(player_stats, on='player_id', how='left')
```

**문제:**
- Train에서만 관찰된 Player/Team 패턴을 Test에 직접 적용
- Fold 간 Player 분포 차이 → **Data Leakage 유사 효과**
- **Target Encoding은 과적합의 주범** (Kaggle 정석)

**증거:**
- LightGBM Exp 33: Player/Team 없이도 CV 16.45 → Public 18.76 (Gap +2.31)
- Domain은 Player/Team 7개 피처 추가 → Gap 더 클 가능성

#### 2순위: All Passes 학습 + Last Pass 평가
```python
# 학습: 356,721개 (모든 패스)
X = train_df[feature_cols]

# 평가: 15,435개 (마지막 패스만)
val_last_mask = train_df.iloc[val_idx]['is_last_pass'] == 1
```

**문제:**
- **Train-Test Mismatch**: 중간 패스 패턴 ≠ 마지막 패스 패턴
- 중간 패스는 전진/측면 이동 많음 (짧은 거리)
- 마지막 패스는 슛/크로스 많음 (긴 거리, 다양한 방향)
- 모델이 **중간 패스 노이즈를 overfitting**

**증거:**
- XGBoost Exp 30: All passes 학습 → CV 15.73, Public 16.47 (Gap +0.74)
- **동일한 접근법 → 동일한 문제**

#### 3순위: 복잡한 피처 (32개)
```python
# 32개 피처 → LightGBM이 과적합하기 쉬움
# Categorical: 9개 (direction, period_id, is_last_pass, 6개 zone flags)
```

**문제:**
- 너무 많은 피처 → **Validation fold의 우연한 패턴 암기**
- Episode 레벨 피처 (episode_avg_distance 등)는 **Episode마다 다름** → 일반화 어려움

### 2. Gap 예상 계산

**과거 패턴:**
| 모델 | All Passes? | Target Encoding? | 피처 수 | CV | Public | Gap |
|------|-------------|------------------|---------|----|----|-----|
| XGBoost | ✅ | ❌ | ~10 | 15.73 | 16.47 | +0.74 |
| LightGBM (Exp 33) | ❌ | ❌ | 4 | 16.45 | 18.76 | +2.31 |
| Domain LightGBM | ✅ | ✅ | 32 | 14.81 | ??? | ??? |

**Gap 예측:**
```
Base Gap (All passes): +0.74 (XGBoost 기준)
Target Encoding penalty: +0.3 ~ +0.5
복잡한 피처 penalty: +0.1 ~ +0.2

총 예상 Gap: +1.14 ~ +1.44

예상 Public: 14.81 + 1.14 = 15.95 (최선)
              14.81 + 1.44 = 16.25 (최악)

확률 분포:
- Public < 16.0 (Zone보다 나음): 40-50%
- Public 16.0-16.3 (Zone과 비슷): 30-40%
- Public > 16.3 (Zone보다 나쁨): 20-30%
```

---

## 🎯 개선 가능성 평가

### Gap 줄이는 방법

#### 1. Target Encoding 제거 (우선순위 1) ⭐⭐⭐
```python
# 현재: Player/Team 통계 7개 피처
player_stats = train_df.groupby('player_id').agg(...)

# 개선: 완전 제거
# 예상 효과: Gap -0.3 ~ -0.5, CV +0.3
```

**효과:**
- Gap 감소: 1.14 → 0.64 ~ 0.84
- Public 예상: 15.11 ~ 15.64 (개선!)
- CV 증가: 14.81 → 15.11 (Zone보다 여전히 낮음)

#### 2. 마지막 패스만 학습 (우선순위 2) ⭐⭐
```python
# 현재: 전체 패스 학습 + 가중치
X = train_df[feature_cols]
sample_weights = np.where(train_df['is_last_pass'] == 1, 10.0, 1.0)

# 개선: 마지막 패스만 학습
train_last = train_df[train_df['is_last_pass'] == 1].copy()
X = train_last[feature_cols]
```

**효과:**
- Gap 감소: 1.14 → 0.40 ~ 0.60 (XGBoost와 유사)
- Public 예상: 15.21 ~ 15.41
- CV 증가: 14.81 → 15.00 ~ 15.20
- **데이터 감소 (356,721 → 15,435, -95.7%)**

#### 3. 피처 단순화 (우선순위 3) ⭐
```python
# 현재: 32개 피처
# 개선: Top 10-15개만 사용

essential_features = [
    'start_x', 'start_y',  # 위치 (2개)
    'goal_distance', 'goal_angle',  # 골대 (2개)
    'prev_dx', 'prev_dy', 'prev_distance',  # 이전 패스 (3개)
    'zone_attack', 'zone_center',  # 필드 구역 (2개)
    'episode_progress',  # Episode (1개)
]
# 총 10개
```

**효과:**
- Gap 감소: 1.14 → 0.94 ~ 1.04
- Public 예상: 15.75 ~ 15.85
- CV 증가: 14.81 → 15.00 ~ 15.10

#### 4. Conservative Regularization (우선순위 4)
```python
# 현재: 기본 설정
params = {
    'num_leaves': 31,
    'max_depth': 6,
    'learning_rate': 0.05,
    ...
}

# 개선: 더 강한 정규화
params = {
    'num_leaves': 15,  # 31 → 15
    'max_depth': 4,    # 6 → 4
    'learning_rate': 0.03,  # 0.05 → 0.03
    'min_child_samples': 100,  # 50 → 100
    'lambda_l1': 1.0,  # 추가
    'lambda_l2': 1.0,  # 추가
}
```

**효과:**
- Gap 감소: 1.14 → 0.94 ~ 1.04
- Public 예상: 15.75 ~ 15.85
- CV 증가: 14.81 → 15.20 ~ 15.40

### 최적 조합 (All 적용)

```python
# 1. Target Encoding 제거
# 2. 마지막 패스만 학습
# 3. Top 10 피처만
# 4. Conservative 정규화

예상 CV: 15.40 ~ 15.60
예상 Public: 15.50 ~ 15.80
예상 Gap: +0.10 ~ +0.20

Zone 대비:
- CV: -0.74 ~ -0.94 (개선)
- Public: -0.56 ~ -0.86 (개선)
- 순위: 200-220위 (개선 가능)
```

---

## 💡 Public 15.95 이상 달성 가능한가?

### 시나리오 분석

#### 시나리오 1: 현재 모델 그대로 제출
```
CV: 14.81
예상 Public: 15.95 (Gap +1.14)
순위: 100-150위 (추정)

확률:
- Public < 15.5 (상위 10%): 10-20%
- Public 15.5-16.0: 30-40%
- Public 16.0-16.3 (Zone과 비슷): 30-40%
- Public > 16.3 (Zone보다 나쁨): 10-20%

결론: 60-70% 확률로 Zone과 비슷하거나 나쁨
```

#### 시나리오 2: Target Encoding 제거만
```
CV: 15.11 (예상)
예상 Public: 15.41 ~ 15.64
순위: 150-200위 (추정)

확률:
- Public < 15.5: 40-50%
- Public 15.5-16.0: 40-50%
- Public > 16.0: 5-10%

결론: 80-90% 확률로 Zone보다 나음
```

#### 시나리오 3: 최적 조합 (All)
```
CV: 15.40 ~ 15.60 (예상)
예상 Public: 15.50 ~ 15.80
순위: 180-220위 (추정)

확률:
- Public < 15.5: 20-30%
- Public 15.5-16.0: 50-60%
- Public > 16.0: 10-20%

결론: 70-80% 확률로 Zone보다 나음
```

### 현실적 기대치

**Public 15.95 이상 (Zone 16.36보다 나음):**
- 시나리오 1: **40-50%** (위험)
- 시나리오 2: **85-90%** (안전)
- 시나리오 3: **80-90%** (안전)

**Public 15.50 이상 (상위 10% 근접):**
- 시나리오 1: **10-20%** (매우 낮음)
- 시나리오 2: **50-60%** (도전할 만함)
- 시나리오 3: **30-40%** (중간)

---

## 📋 제출 가치 평가

### 제출 여부 결정

#### 제출 권장 (✅)
```
조건:
1. Target Encoding 제거 + 최적화 적용
2. CV 15.20-15.60 확인
3. 예상 Public < 16.0 (70% 이상 확률)

기대 효과:
- Zone보다 개선 가능성 높음 (80-90%)
- 순위: 180-220위 (개선)
- 리스크: 낮음
```

#### 제출 보류 (❌)
```
조건:
1. 현재 모델 그대로 (Target Encoding 포함)
2. CV 14.81 (과최적화 의심)
3. Gap +1.14 예상 (높음)

이유:
- Zone보다 나쁠 확률 30-40% (높음)
- 제출 낭비 (1/160회)
- Sweet Spot 위반 (CV < 16.27)
- XGBoost/LightGBM 실패 패턴 재현 가능성
```

### 최종 권장사항

**현재 상태: 제출 보류 ❌**

**이유:**
1. ❌ **Target Encoding 7개 피처** → 과적합 주범
2. ❌ **All passes 학습** → Train-Test Mismatch
3. ❌ **CV 14.81 < Sweet Spot 16.27** → 과최적화
4. ❌ **검증 없음** → 실제 Public 알 수 없음
5. ❌ **60-70% 확률로 Zone과 비슷하거나 나쁨** → 위험

**다음 단계:**
1. ✅ **Target Encoding 제거** → 우선 테스트
2. ✅ **마지막 패스만 학습** → CV 15.11 확인
3. ✅ **Conservative 정규화** → Gap 최소화
4. ✅ **CV 15.20-15.60 확인** → 안전 범위
5. ✅ **그 다음 제출** → 80-90% 성공 확률

---

## 🎯 실용적 다음 행동

### Immediate Actions (우선순위 순)

#### 1. Target Encoding 제거 버전 (30분)
```python
# code/models/best/model_domain_features_no_target_encoding.py

# Player/Team 통계 7개 피처 제거
feature_cols = [
    'start_x', 'start_y',
    'goal_distance', 'goal_angle', 'is_near_goal',
    'zone_attack', 'zone_defense', 'zone_middle',
    'zone_left', 'zone_center', 'zone_right',
    'dist_to_left', 'dist_to_right', 'dist_to_top', 'dist_to_bottom',
    'prev_dx', 'prev_dy', 'prev_distance', 'direction',
    'episode_progress', 'episode_avg_distance', 'episode_forward_ratio', 'is_last_pass',
    'period_id', 'time_seconds'
]
# 32개 → 25개 (Player 4개 + Team 3개 제거)

예상 결과:
- CV: 15.11 ± 0.20 (0.30 증가)
- 예상 Public: 15.41 ~ 15.64
- Gap: +0.30 ~ +0.53
```

#### 2. 마지막 패스만 학습 버전 (30분)
```python
# code/models/best/model_domain_features_last_pass_only.py

# 전체 패스 학습 제거
train_last = train_df[train_df['is_last_pass'] == 1].copy()
X = train_last[feature_cols]
y_x = train_last['delta_x']
y_y = train_last['delta_y']

# 가중치 제거
# sample_weights 사용 안 함

예상 결과:
- CV: 15.20 ~ 15.40
- 예상 Public: 15.30 ~ 15.60
- Gap: +0.10 ~ +0.20
```

#### 3. 최적 조합 버전 (1시간)
```python
# code/models/best/model_domain_features_optimized.py

# 1. Target Encoding 제거
# 2. 마지막 패스만 학습
# 3. Top 15 피처만
# 4. Conservative 정규화

feature_cols = [
    'start_x', 'start_y',
    'goal_distance', 'goal_angle', 'is_near_goal',
    'prev_dx', 'prev_dy', 'prev_distance',
    'zone_attack', 'zone_center',
    'episode_progress', 'episode_avg_distance',
    'period_id', 'time_seconds', 'direction'
]  # 15개

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 15,
    'max_depth': 4,
    'learning_rate': 0.03,
    'min_child_samples': 100,
    'lambda_l1': 1.0,
    'lambda_l2': 1.0,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'bagging_freq': 5,
    'verbose': -1
}

예상 결과:
- CV: 15.40 ~ 15.60
- 예상 Public: 15.50 ~ 15.80
- Gap: +0.10 ~ +0.20
```

#### 4. CV 검증 및 제출 결정 (10분)
```python
# 각 버전의 CV 확인
if CV in [15.20, 15.60]:  # Sweet Spot 근처
    if Gap_estimated < 0.5:
        print("제출 권장 ✅")
    else:
        print("추가 최적화 필요 ⚠️")
else:
    print("제출 보류 ❌")
```

### Week 3-4 전략

**Week 3 (D-26~20): 연구 모드**
```
1. Target Encoding 제거 → CV 15.11 확인
2. Last pass only → CV 15.30 확인
3. 최적 조합 → CV 15.50 확인
4. 제출: 0-1회 (가장 안전한 버전만)
```

**Week 4 (D-19~13): 검증 모드**
```
1. 최적 버전 선택 (CV 15.20-15.60)
2. Ensemble 시도 (Zone + Domain)
3. 제출: 2-3회
4. Public 16.0 미만 달성 목표
```

**Week 5 (D-12~0): 집중 모드**
```
1. Public 성능 기반 미세 조정
2. 최종 앙상블
3. 제출: 3-5회
4. 순위: 180-220위 목표
```

---

## 📊 최종 요약

### 핵심 인사이트

1. ✅ **Domain features는 합리적** (골대 거리, 필드 구역 등)
2. ❌ **Target Encoding은 과적합 주범** (Player/Team 7개)
3. ❌ **All passes 학습은 위험** (Train-Test Mismatch)
4. ⚠️ **현재 모델 그대로는 60-70% 실패 확률**

### 개선 로드맵

```
현재 (CV 14.81, Public 15.95 추정):
→ Target Encoding 제거 (CV 15.11, Public 15.52 추정):
  → Last pass only (CV 15.30, Public 15.45 추정):
    → 최적 조합 (CV 15.50, Public 15.65 추정):
      → Zone 16.36보다 0.71점 개선 ✅

확률:
- Zone보다 나음: 80-90%
- 상위 10% 진입: 30-40%
- 순위: 180-220위
```

### 행동 결정

**즉시 (오늘-내일):**
1. ❌ **현재 모델 제출 보류** (위험 높음)
2. ✅ **Target Encoding 제거 버전 작성** (30분)
3. ✅ **CV 확인** (15.11 예상)

**Week 3:**
1. ✅ **최적 조합 버전 완성** (1시간)
2. ✅ **CV 15.20-15.60 확인**
3. ✅ **제출 1회** (가장 안전한 버전)

**Week 4-5:**
1. ✅ **Public 성능 기반 미세 조정**
2. ✅ **Zone + Domain 앙상블**
3. ✅ **순위 180-220위 목표**

---

## 🎓 교훈

### 성공 요소
1. ✅ 축구 도메인 지식 활용 (골대, 필드 구역)
2. ✅ 명시적 피처 생성 (Zone보다 해석 가능)
3. ✅ LightGBM의 강력한 학습 능력

### 위험 요소
1. ❌ Target Encoding → 과적합
2. ❌ All passes 학습 → Mismatch
3. ❌ 복잡한 피처 32개 → Overfitting
4. ❌ Sweet Spot 위반 (CV < 16.27)

### 핵심 메시지

```
"Domain features는 올바른 방향이다.
 하지만 Target Encoding과 All passes 학습은 제거해야 한다.
 최적화 후 80-90% 확률로 Zone보다 나아질 것이다.
 조급하게 제출하지 말고, 체계적으로 개선하자."
```

---

*작성: 2025-12-16*
*분석자: Backend Developer Agent*
*신뢰도: 85% (과거 패턴 기반 추론)*
