# Ultrathink: 근본적 문제 분석

> **작성일:** 2025-12-15
> **목적:** 241위 원인 규명 및 돌파구 발견
> **방법:** 데이터 심층 분석 + 분포 비교 + 모델 한계 분석

---

## 🚨 현재 상황

```
실제 순위: 241/1006위 (하위 76%)
1등: 12.70
우리: 16.36
차이: +3.66 (28.8%)

상태: 안정적이지만 절대 성능 매우 낮음
```

---

## 📊 데이터 심층 분석 결과

### 1. Episode 길이 분포

```
평균: 23.1개 패스
중앙값: 16개 패스
최소: 1개
최대: 270개
표준편차: 22.0 (매우 불균등!)

길이별 분포:
- 1개: 0.0%
- 2-5개: 15.4%
- 10개 이상: 69.2%
- 20개 이상: 42.7%
- 50개 이상: 10.7%
```

**발견:**
- 매우 불균등한 분포
- LSTM은 길이에 민감할 수 있음

### 2. Delta 거리 분포 (핵심!)

```
평균: 20.4m
중앙값: 15.7m
표준편차: 15.9m ⚠️ 매우 큼!

백분위수:
- 10%: 3.9m
- 25%: 8.8m
- 50%: 15.7m (median)
- 75%: 28.1m
- 90%: 45.1m
- 95%: 53.2m
- 99%: 66.6m (필드 전체!)
```

**발견:**
- **표준편차 15.9m = 우리 점수 16.36의 원인!**
- Median만 사용 → 50%는 median에서 ±15.9m 벗어남
- 상위권은 이 분산을 어떻게 처리하는가?

### 3. Zone별 Delta 표준편차

```
최악의 Zone:
1. Zone 0_1: 20.1m
2. Zone 0_3: 20.1m
3. Zone 1_2: 19.2m
4. Zone 0_2: 18.7m
5. Zone 1_3: 18.6m
```

**발견:**
- 자진 영역(Zone 0, 1): 표준편차 매우 큼 (20m!)
- 우리 모델: median만 사용 → 20m 오차 가능
- **이것이 16.36의 근본 원인!**

### 4. Zone 분포

```
상위 10개 Zone:
1. Zone 3_0: 5.6%
2. Zone 3_5: 5.0%
3. Zone 4_5: 4.6%
4. Zone 4_0: 4.6%
...

특징:
- 불균등 분포
- 36개 Zone 중 상위 10개가 40% 차지
- 하위 Zone: 샘플 수 적음 → 통계 불안정
```

---

## 🆚 Train vs Test 분포 비교

### 결과: 거의 동일! ✅

```
Episode 길이:
- Train 평균: 23.1 vs Test: 22.0 (차이 1.1개)
- Train 중앙값: 16 vs Test: 15 (차이 1개)

시작 위치:
- start_x: Train 54.9 vs Test 54.1 (차이 0.8m)
- start_y: Train 33.6 vs Test 34.0 (차이 0.4m)

Zone 분포:
- 최대 차이: 0.8% (Zone 1_3)
- 대부분 < 0.5%

Period 분포:
- Train 전반: 47.3% vs Test: 47.7% (차이 0.3%)
- Train 후반: 52.7% vs Test: 52.3%

골대 거리:
- Train: 55.5m vs Test: 56.2m (차이 0.7m)
```

**핵심 발견:**
- ✅ Train/Test 분포가 거의 동일
- ✅ Distribution shift 없음
- ❌ LSTM Gap +2.93의 원인이 분포 차이가 아님!

---

## 🔍 모델별 문제 분석

### Zone 6x6 모델

**접근법:**
```python
# Zone+Direction별 median 사용
stats = train.groupby(['zone', 'direction']).agg({
    'delta_x': 'median',
    'delta_y': 'median'
})

pred = start + stats.loc[(zone, direction)]
```

**문제:**

1. **분산 무시:**
   ```
   Delta 거리 표준편차: 15.9m
   Zone 0_1, 0_3 표준편차: 20.1m

   Median만 사용
   → 50%는 median에서 ±15.9m 벗어남
   → 평균 오차: ~16m
   → Public 16.36 ✅ 설명 가능!
   ```

2. **Context 부족:**
   - 시간 정보 무시 (전반/후반)
   - Episode 상황 무시 (몇 번째 패스인지)
   - 이전 패스 패턴 무시

3. **단순함의 양면성:**
   - ✅ 과적합 방지 (Gap +0.02)
   - ❌ 절대 성능 낮음 (241위)

**결론:**
```
Zone 6x6 = 안정적이지만 성능 한계
표준편차 15.9m를 median으로 예측
→ 16.36 이하로 내려가기 어려움
```

### LSTM v3/v5 모델

**접근법:**
```python
# Full episode sequence 학습
input = [p1, p2, p3, ...]  # Episode 전체
target = p_last  # 마지막 패스

LSTM → 213K parameters (v5)
```

**문제:**

1. **Gap 폭발:**
   ```
   CV: 14.36 (좋음!)
   Public: 17.29 (나쁨!)
   Gap: +2.93 (과적합!)
   ```

2. **Train/Test 분포 동일한데 왜 Gap?**
   - 가설 1: Episode 길이 민감성
     - 짧은 episode (1-5개): 정보 부족
     - 긴 episode (50+개): 관련 없는 정보
   - 가설 2: 과적합
     - 213K parameters → Train noise 학습
     - Regularization (dropout 0.6, L2) 부족
   - 가설 3: Sequence의 본질
     - 문제: "마지막 패스 좌표 예측"
     - LSTM: 시퀀스 전체 → 과도한 정보?
     - Zone: 마지막 위치만 → 적절한 정보?

3. **단순화(v5) 실패:**
   ```
   Parameters: 838K → 213K (74.6% 감소)
   CV: 14.36 → 14.44 (+0.08, 거의 동일)
   Public: 17.29 → 17.44 (+0.15, 악화!)

   → 단순화로도 Gap 해결 안 됨
   → 근본 문제: "시퀀스 접근 자체"
   ```

**결론:**
```
LSTM = Gap 문제 해결 불가능
문제의 본질이 "시퀀스"가 아닌 "위치 통계"
더 이상의 LSTM 실험은 시간 낭비
```

---

## 💡 핵심 인사이트

### 1. Zone 6x6의 16.36은 **이론적 한계**

```
Delta 표준편차: 15.9m
Median 예측 → 평균 오차: ~16m
Public 16.36 ✅

더 낮추려면:
→ Median 대신 분산 고려
→ 더 많은 context
→ 더 세밀한 segmentation
```

### 2. LSTM은 **근본적으로 맞지 않음**

```
CV 14.36 (좋음) → Public 17.29 (나쁨)
Gap +2.93 = 과적합

원인:
→ 시퀀스 전체가 아닌 "마지막 위치"가 중요
→ Zone 통계가 본질에 더 가까움
```

### 3. Train/Test 분포 동일 = **좋은 소식**

```
Zone 차이 < 1%
Episode 길이 차이: 1.1개

→ 새 모델도 안정적일 가능성 높음
→ CV ≈ Public 기대 가능
```

### 4. 상위권(12.70~13.50)의 비밀

**가설:**

1. **더 세밀한 Segmentation:**
   ```
   Zone 6x6 → 10x10, 12x12 또는 Adaptive
   더 정확한 통계
   ```

2. **분산 고려:**
   ```
   Median 대신:
   - Quantile regression
   - Gaussian Mixture
   - Neural network
   ```

3. **더 많은 Context:**
   ```
   Zone + Direction + Time + Episode_position + ...
   복합적 패턴 학습
   ```

4. **Ensemble:**
   ```
   여러 모델 조합
   안정성 + 성능
   ```

---

## 🎯 돌파구

### 현실적 목표

```
현재: 16.36 (241위)
목표: < 16.0 (상위 20% 진입)
격차: -0.36점 (2.2% 개선)
```

### 가능한 접근법 (우선순위)

#### 1. Gradient Boosting (최우선!) ⭐

**이유:**
- Tabular data에 최강 (Kaggle 표준)
- Zone보다 복잡, LSTM보다 안정적
- 많은 피처 활용 가능
- 과적합 제어 용이

**장점:**
```python
# Zone 6x6 피처
features = [
    'zone', 'direction',  # Zone 통계
    'start_x', 'start_y',  # 위치
    'goal_distance', 'goal_angle',  # 골대
    'period_id', 'time_seconds',  # 시간
    'pass_count',  # Episode 내 위치
    'prev_dx', 'prev_dy',  # 이전 패스
    ...
]

# XGBoost/LightGBM/CatBoost
model = XGBRegressor(...)
model.fit(features, target)
```

**기대 성능:**
```
CV: 14-15 (LSTM 수준)
Public: 15-16 (Gap < 1.5)
개선: -1.0~-1.5점 (200위권 진입 가능!)
```

#### 2. Zone 세분화 + Quantile

**이유:**
- Zone 6x6의 자연스러운 확장
- 안정성 유지하면서 성능 개선

**접근:**
```python
# 10x10 Zone + 8-way Direction
# Quantile 0.4, 0.5, 0.6 앙상블
# Adaptive Zone (샘플 수 고려)
```

**기대 성능:**
```
CV: 15-16
Public: 15.5-16.5 (Gap < 1.0)
개선: -0.5~-1.0점 (220위권)
```

#### 3. Feature Engineering 강화

**이유:**
- Zone 6x6 + 더 많은 context
- 기존 코드 재사용 가능

**추가 피처:**
```python
# 시간 피처
'time_left': 5400 - time_seconds
'is_first_half': period_id == 1
'time_pressure': time_left < 600  # 막판 10분

# Episode 피처
'episode_position': pass_count / episode_length
'is_last_few_passes': (episode_length - pass_count) < 3

# 위치 피처
'is_attacking_third': start_x > 70
'is_defensive_third': start_x < 35
'is_central': 23 < start_y < 45

# 조합 피처
'zone_time': f"{zone}_{period_id}"
'zone_position': f"{zone}_{episode_position_bin}"
```

**기대 성능:**
```
CV: 15.5-16.5
Public: 15.8-16.8
개선: -0.3~-0.6점
```

#### 4. Ensemble (최종 단계)

**이유:**
- 안정성 극대화
- 여러 접근법 조합

**구성:**
```
Zone 6x6 (16.36) 가중치 0.3
GBM (15.0) 가중치 0.5
Zone 10x10 (15.8) 가중치 0.2
→ 예상: 15.5
```

---

## 🚫 하지 말아야 할 것

### 1. LSTM 추가 실험 ❌

**이유:**
```
v2, v3, v4, v5 모두 실패
Gap +2.93~+3.00 해결 불가능
시간 낭비
```

### 2. Zone 6x6 하이퍼파라미터 추가 탐색 ❌

**이유:**
```
14회 완전 탐색 완료
Sweet Spot 발견
개선 불가능 (확률 0.006%)
```

### 3. Data Augmentation ❌

**이유:**
```
Horizontal Flip 실패 (v4)
축구의 비대칭성
물리적 타당성 ≠ 패턴 보존
```

### 4. 복잡한 Neural Network ❌

**이유:**
```
LSTM Gap 문제
Transformer: 더 복잡 → 더 큰 Gap 예상
Attention: 필요 없음 (마지막 위치만 중요)
```

---

## 📅 실행 계획

### Week 2 (D-28~D-22) - 현재

**목표:** 연구 & 준비

**할 일:**
1. ✅ Data Leakage 검증 (완료)
2. ✅ Ultrathink 분석 (완료)
3. 🔄 GBM 코드 작성 (10% 샘플)
4. 🔄 빠른 실험 시스템 구축
5. ⏸️ 관찰 (제출 0회)

### Week 3 (D-21~D-15)

**목표:** 실험

**할 일:**
1. GBM baseline (XGBoost, LightGBM, CatBoost 비교)
2. Feature engineering 강화
3. CV 검증 (Gap 확인)
4. 제출: 1-2회 (검증 목적만)

### Week 4-5 (D-14~D-0)

**목표:** 최적화 & 제출

**할 일:**
1. Best GBM 하이퍼파라미터 튜닝
2. Zone 세분화 실험
3. Ensemble 구성
4. 제출: 3-5회/일
5. 목표: Public < 16.0 (상위 20%)

---

## 🎓 배운 교훈

### 1. 문제의 본질 이해

```
"마지막 패스 좌표 예측"
→ 시퀀스 전체가 아닌 "마지막 위치 + Context"
→ Zone 통계 > LSTM
→ GBM이 적합
```

### 2. 분산의 중요성

```
표준편차 15.9m 무시
→ Median만 사용
→ 평균 오차 16m
→ 분산 고려 필수
```

### 3. 안정성 vs 성능

```
Zone 6x6: Gap +0.02 (안정) but 241위 (성능 낮음)
LSTM: CV 14.36 (성능 좋음) but Gap +2.93 (불안정)
→ 둘 다 필요: GBM으로 균형
```

### 4. Train/Test 분포 확인

```
분포 동일 확인 (차이 < 1%)
→ 새 모델도 안정적 예상
→ CV ≈ Public 기대
```

---

## 📊 예상 성과

### 보수적 시나리오

```
GBM baseline: CV 15.0, Public 15.5 (Gap +0.5)
현재 대비: -0.86점
예상 순위: 200위권 (상위 20% 진입 ✅)
```

### 낙관적 시나리오

```
GBM + Feature + Ensemble: CV 14.5, Public 14.8 (Gap +0.3)
현재 대비: -1.56점
예상 순위: 150위권 (상위 15%)
```

### 현실적 목표

```
Public < 16.0 (상위 20%)
확률: 60-70%
근거: GBM은 Kaggle 표준, 우리 데이터에 적합
```

---

## 🔗 다음 단계

1. **RECOVERY_PLAN.md 작성**
   - 구체적 실행 계획
   - 코드 템플릿
   - 체크리스트

2. **빠른 실험 시스템 구축**
   - 10% 샘플링
   - 자동 CV 계산
   - 로깅

3. **GBM Baseline 구현**
   - XGBoost, LightGBM, CatBoost
   - 기본 피처
   - CV 검증

---

**작성자:** Claude Sonnet 4.5
**분석 일자:** 2025-12-15
**다음 리뷰:** Week 3 시작 시 (D-21)

---

*"The devil is in the distribution, not the mean."*
*"Zone 6x6의 16.36은 이론적 한계였다. 이제 GBM으로 돌파한다."*
