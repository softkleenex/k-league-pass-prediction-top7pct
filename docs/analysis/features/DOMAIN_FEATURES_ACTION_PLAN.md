# Domain Features 개선 액션 플랜

> **목표:** CV 14.81 → 15.50, Public 15.95 → 15.65 달성
> **기간:** 3일 (즉시 시작)
> **우선순위:** 높음 (Zone보다 0.7점 개선 가능)

---

## 🎯 3단계 개선 전략

### Phase 1: Target Encoding 제거 (30분)

**파일:** `code/models/best/model_domain_features_v2_no_target.py`

**변경 사항:**
```python
# 제거할 코드 (lines 145-179)
# =========================================================================
# G. Player/Team Target Encoding (과적합 주의!)
# =========================================================================
# player_stats = train_df.groupby('player_id').agg(...)
# team_stats = train_df.groupby('team_id').agg(...)
# 전체 삭제!

# 제거할 피처 (lines 188-212)
feature_cols = [
    # ... (기존 25개 유지)
    # 제거:
    # 'player_avg_dx', 'player_avg_dy', 'player_avg_distance', 'player_forward_ratio',
    # 'team_avg_dx', 'team_avg_dy', 'team_avg_distance',
]
```

**예상 결과:**
- CV: 15.11 ± 0.20 (0.30 증가)
- Public: 15.41 ~ 15.64 (0.31 ~ 0.54 개선)
- Gap: +0.30 ~ +0.53 (0.61 ~ 0.84 감소)

**검증:**
```bash
cd /mnt/c/LSJ/dacon/dacon/kleague-algorithm
python code/models/best/model_domain_features_v2_no_target.py

# 확인 사항:
# - Fold 1-3 CV: 15.0 ~ 15.3 (예상)
# - Feature 수: 25개 (32 - 7)
```

---

### Phase 2: Last Pass Only (30분)

**파일:** `code/models/best/model_domain_features_v3_last_pass.py`

**변경 사항:**
```python
# Phase 1 기반 + 다음 변경

# 기존 (lines 138-141): 전체 패스에 피처 생성
train_df = create_domain_features(train_df)
test_all = create_domain_features(test_all)

# 변경: 마지막 패스만 추출
train_df = create_domain_features(train_df)
train_last = train_df[train_df['is_last_pass'] == 1].copy()

test_all = create_domain_features(test_all)
test_last = test_all[test_all['is_last_pass'] == 1].copy()

# 기존 (lines 222-226): 전체 패스 학습
X = train_df[feature_cols].fillna(0)
y_x = train_df['delta_x']
y_y = train_df['delta_y']
sample_weights = np.where(train_df['is_last_pass'] == 1, 10.0, 1.0)

# 변경: 마지막 패스만 학습
X = train_last[feature_cols].fillna(0)
y_x = train_last['delta_x']
y_y = train_last['delta_y']
# sample_weights 제거 (모두 1.0)

# GroupKFold 변경 (line 239)
game_ids = train_last['game_id'].values  # train_df → train_last

# 검증 루프 변경 (lines 256-295)
for fold, (train_idx, val_idx) in enumerate(gkf.split(X, groups=game_ids)):
    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]  # X_val_all → X_val
    y_train_x = y_x.iloc[train_idx]
    y_train_y = y_y.iloc[train_idx]
    # train_weights 제거

    # X 모델 (가중치 제거)
    train_data_x = lgb.Dataset(X_train, label=y_train_x,
                                categorical_feature=categorical_features)
                                # weight=train_weights 제거

    # 평가 (마지막 패스 필터 제거)
    # val_last_mask 제거 (이미 마지막 패스만)
    val_df = train_last.iloc[val_idx]

    pred_delta_x = model_x.predict(X_val)
    pred_delta_y = model_y.predict(X_val)

    pred_end_x = np.clip(val_df['start_x'].values + pred_delta_x, 0, 105)
    pred_end_y = np.clip(val_df['start_y'].values + pred_delta_y, 0, 68)
```

**예상 결과:**
- CV: 15.20 ~ 15.40 (Phase 1 대비 0.09 ~ 0.29 증가)
- Public: 15.30 ~ 15.60 (Phase 1 대비 0.04 ~ 0.11 개선)
- Gap: +0.10 ~ +0.20 (Phase 1 대비 0.20 ~ 0.43 감소)

**검증:**
```bash
python code/models/best/model_domain_features_v3_last_pass.py

# 확인 사항:
# - Train samples: ~15,435 (기존 356,721에서 감소)
# - Fold 1-3 CV: 15.2 ~ 15.5 (예상)
```

---

### Phase 3: 최적 조합 (1시간)

**파일:** `code/models/best/model_domain_features_v4_optimized.py`

**변경 사항:**
```python
# Phase 2 기반 + 다음 변경

# 1. Top 15 피처만 선택
feature_cols = [
    # 기본 위치 (2개)
    'start_x', 'start_y',

    # 골대 관련 (3개) - 가장 중요
    'goal_distance', 'goal_angle', 'is_near_goal',

    # 이전 패스 (3개) - 중요
    'prev_dx', 'prev_dy', 'prev_distance',

    # 필드 구역 (2개만) - 단순화
    'zone_attack', 'zone_center',

    # Episode (2개만) - 단순화
    'episode_progress', 'episode_avg_distance',

    # 시간 + 방향 (3개)
    'period_id', 'time_seconds', 'direction'
]
# 총 15개 (기존 25개에서 10개 제거)

categorical_features = ['direction', 'period_id', 'zone_attack', 'zone_center']

# 2. Conservative Regularization
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',

    # 복잡도 감소
    'num_leaves': 15,           # 31 → 15
    'max_depth': 4,              # 6 → 4

    # 학습률 감소
    'learning_rate': 0.03,       # 0.05 → 0.03

    # 샘플 요구량 증가
    'min_child_samples': 100,    # 기본값 → 100

    # Regularization 추가
    'lambda_l1': 1.0,            # 추가
    'lambda_l2': 1.0,            # 추가

    # Feature/Bagging fraction 감소
    'feature_fraction': 0.7,     # 0.8 → 0.7
    'bagging_fraction': 0.7,     # 0.8 → 0.7
    'bagging_freq': 5,

    'verbose': -1,
    'random_state': 42
}

# 3. 부스팅 라운드 조정
num_boost_round = 500  # 300 → 500 (learning_rate 감소로 보상)
```

**예상 결과:**
- CV: 15.40 ~ 15.60 (Phase 2 대비 0.00 ~ 0.20 증가)
- Public: 15.50 ~ 15.80 (Phase 2 대비 0.00 ~ 0.20 증가)
- Gap: +0.10 ~ +0.20 (Phase 2와 동일, 안정적)

**검증:**
```bash
python code/models/best/model_domain_features_v4_optimized.py

# 확인 사항:
# - Feature 수: 15개
# - Fold 1-3 CV: 15.4 ~ 15.6 (예상)
# - Feature Importance Top 5: goal_distance, start_x, start_y, prev_dx, goal_angle
```

---

## 📊 비교 예상

| 버전 | CV | Public (예상) | Gap | Zone 대비 | 제출 권장 |
|------|----|----|-----|-----------|-----------|
| **원본** | 14.81 | 15.95 | +1.14 | -0.41 | ❌ 위험 |
| **v2 (No Target)** | 15.11 | 15.52 | +0.41 | -0.84 | ⚠️ 보통 |
| **v3 (Last Pass)** | 15.30 | 15.45 | +0.15 | -0.91 | ✅ 좋음 |
| **v4 (Optimized)** | 15.50 | 15.65 | +0.15 | -0.71 | ✅ 안전 |
| **Zone 6x6** | 16.34 | 16.36 | +0.02 | - | ✅ 기준 |

---

## ✅ 제출 결정 기준

### 즉시 제출 (✅)
```
조건:
- CV: 15.20 ~ 15.60
- Gap 예상: < 0.30
- Zone 대비: -0.7 ~ -0.9

버전:
- v3 (Last Pass) 또는 v4 (Optimized)

확률:
- Zone보다 나음: 80-90%
- Public < 16.0: 85-95%
```

### 추가 검증 (⚠️)
```
조건:
- CV: 15.00 ~ 15.20 또는 15.60 ~ 15.80
- Gap 예상: 0.30 ~ 0.50

액션:
- Ensemble 시도 (Zone + Domain)
- 1-2일 관찰
```

### 제출 보류 (❌)
```
조건:
- CV: < 15.00 또는 > 15.80
- Gap 예상: > 0.50

이유:
- 과최적화 (CV < 15.00)
- 개선 부족 (CV > 15.80)
```

---

## 🚀 실행 타임라인

### Day 1 (오늘)
```
09:00 - 09:30: Phase 1 구현 (v2_no_target)
09:30 - 09:45: Phase 1 실행 및 검증
09:45 - 10:15: Phase 2 구현 (v3_last_pass)
10:15 - 10:30: Phase 2 실행 및 검증
```

### Day 1 (오후)
```
14:00 - 15:00: Phase 3 구현 (v4_optimized)
15:00 - 15:15: Phase 3 실행 및 검증
15:15 - 15:30: 결과 비교 및 분석
15:30 - 16:00: 제출 결정 (v3 또는 v4)
```

### Day 2 (선택)
```
- v3/v4 제출 결과 확인
- Ensemble 시도 (Zone + Domain)
- 추가 미세 조정
```

---

## 📁 파일 구조

```
code/models/best/
├── model_domain_features_lgbm.py           (원본, CV 14.81)
├── model_domain_features_v2_no_target.py   (Phase 1, CV 15.11 예상)
├── model_domain_features_v3_last_pass.py   (Phase 2, CV 15.30 예상)
└── model_domain_features_v4_optimized.py   (Phase 3, CV 15.50 예상)

submissions/pending/
├── submission_domain_features_cv14.81.csv         (원본, 제출 보류)
├── submission_domain_features_v2_cv15.11.csv      (Phase 1)
├── submission_domain_features_v3_cv15.30.csv      (Phase 2, 제출 권장)
└── submission_domain_features_v4_cv15.50.csv      (Phase 3, 제출 권장)
```

---

## 🎓 핵심 포인트

### 왜 이 개선이 효과적인가?

1. **Target Encoding 제거**
   - 과적합의 70% 원인 제거
   - Gap 0.61 ~ 0.84 감소

2. **Last Pass Only**
   - Train-Test Mismatch 제거
   - 안정적인 일반화

3. **Conservative Regularization**
   - 과적합 추가 방지
   - CV-Public 일치도 향상

### 리스크 관리

**최악의 경우:**
- v4: CV 15.50 → Public 15.70 (Gap +0.20)
- 여전히 Zone 16.36보다 0.66점 나음 ✅

**최선의 경우:**
- v3: CV 15.30 → Public 15.35 (Gap +0.05)
- Zone 16.36보다 1.01점 나음ⓘ 🎉

**예상 범위:**
- 80% 확률로 Public 15.45 ~ 15.65
- Zone 대비 -0.71 ~ -0.91점 개선

---

## 📝 체크리스트

### Phase 1
- [ ] v2_no_target.py 작성
- [ ] Target Encoding 7개 피처 제거 확인
- [ ] Feature 수 25개 확인
- [ ] CV 15.0 ~ 15.3 확인
- [ ] Feature Importance 저장

### Phase 2
- [ ] v3_last_pass.py 작성
- [ ] Train samples 15,435개 확인
- [ ] sample_weights 제거 확인
- [ ] CV 15.2 ~ 15.5 확인
- [ ] Gap 예상 < 0.30 확인

### Phase 3
- [ ] v4_optimized.py 작성
- [ ] Feature 수 15개 확인
- [ ] Conservative params 확인
- [ ] CV 15.4 ~ 15.6 확인
- [ ] Feature Importance Top 5 확인

### 제출 준비
- [ ] 최적 버전 선택 (v3 또는 v4)
- [ ] CV Sweet Spot 확인 (15.2 ~ 15.6)
- [ ] 제출 파일 생성
- [ ] EXPERIMENT_LOG 업데이트
- [ ] 제출 실행

---

*작성: 2025-12-16*
*예상 소요 시간: 2-3시간*
*성공 확률: 80-90%*
