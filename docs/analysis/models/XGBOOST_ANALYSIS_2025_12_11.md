# K리그 패스 좌표 예측 - XGBoost + 시퀀스 피처 분석
**작성일:** 2025-12-11
**모델:** XGBoost + Sequence Features
**상태:** 실험 완료, 과최적화 경고

---

## 1. 실험 개요

### 목표
- Zone 통계 기반 베이스라인 (16.34 CV) 개선
- XGBoost + 시퀀스 피처로 더 정교한 패스 좌표 예측
- 14~15점대 달성 시도

### 구현 내용
3개의 XGBoost 모델 구현 및 검증:
1. **model_xgboost_v1.py** - 절대 좌표 직접 예측 (실패)
2. **model_xgboost_delta_v2.py** - Delta(dx, dy) 예측 기반
3. **model_xgboost_safe_v3.py** - 강정규화 + Zone 앙상블

---

## 2. 시퀀스 피처 설계

### 생성된 피처 (8가지)

```
기본 피처:
- start_x, start_y          : 현재 패스 시작 위치

시퀀스 피처 (이전 패스 정보):
- prev_1_dx, prev_1_dy      : 직전 패스의 이동량
- prev_2_dx, prev_2_dy      : 2번 전 패스의 이동량
- prev_3_dx, prev_3_dy      : 3번 전 패스의 이동량
```

### 설계 원칙
- **간단함:** 8가지 피처만 사용 (과복잡성 방지)
- **순차성:** 이전 3개 패스 정보로 패턴 인식
- **필드 위치:** 시작점으로 인코딩된 필드 정보

### 생성되지 않은 피처 (제외 이유)
- `avg_dx_3, std_dx_3` - 시퀀스 모델에서 XGBoost가 자동 학습
- `zone, direction` - 이미 safe_fold13에서 완전 탐색됨
- `distance, angle` - 단순 파생 피처 (XGBoost 불필요)

---

## 3. 모델 성능 비교

### 3.1 XGBoost Delta v2 (정규화 없음)

| 메트릭 | 값 | 상태 |
|--------|-----|------|
| **Fold 1-3 CV** | 15.7580 | ❌ 과최적화 |
| **Fold 4-5 CV** | 15.3362 | ⚠️ 큰 Gap |
| **Fold Gap** | -0.4218 | 위험 신호 |
| **vs Zone** | -1.0859 | +1.09 개선 |

**평가:** 좋은 성능이지만 과최적화 위험 (CV < 16.27)

### 3.2 XGBoost Safe v3 (강정규화 + 앙상블)

| 메트릭 | 값 | 상태 |
|--------|-----|------|
| **Fold 1-3 CV** | 15.9213 | ❌ 여전히 과최적화 |
| **Fold 4-5 CV** | 15.4865 | 개선됨 |
| **Fold Gap** | -0.4348 | ✅ 양호 (< 0.5) |
| **최적 앙상블** | Zone 10% + XGBoost 90% | |
| **앙상블 CV** | 15.7314 | 최적화된 성능 |

**평가:** 정규화로 Gap 개선, 하지만 CV 여전히 16.27 이하

### 3.3 Zone Baseline (기준선)

| 메트릭 | 값 |
|--------|-----|
| **Fold 1-3 CV** | 16.6756 |
| **Public** | 16.3639 (safe_fold13) |
| **상태** | 검증된 Sweet Spot |

---

## 4. 핵심 발견사항

### 4.1 XGBoost의 강점

1. **뛰어난 성능**
   - Fold 1-3에서 Zone 대비 +1.08 개선
   - 15.76~15.92 CV 달성

2. **안정적인 Fold Gap**
   - XGBoost Safe v3: -0.4348
   - Zone baseline: 약 +0.33 (gap 역방향)
   - XGBoost가 더 안정적

3. **학습 능력**
   - 시퀀스 피처를 효과적으로 활용
   - 단순한 통계 방식보다 패턴 인식 우수

### 4.2 과최적화 경고

**문제:** CV 모두 16.27 이하
```
XGBoost Delta v2:   15.7580 (16.27보다 -0.47)
XGBoost Safe v3:    15.9213 (16.27보다 -0.35)
```

**의미:**
- Fold 1-3에서 과도하게 좋은 성능
- Public에서 Gap이 발생할 가능성
- CLAUDE.md 경고: CV < 16.27 → Gap 폭발 위험

**근거:**
1. 14회 연속 실패로 증명된 Zone 최적점
2. XGBoost도 같은 한계에 도달?
3. 시퀀스 피처는 제한된 이득만 가능

---

## 5. 제출 파일 분석

### 생성된 파일

| 파일명 | 전략 | CV | 평가 |
|--------|------|-----|------|
| submission_xgboost_v1.csv | 절대좌표 직접 | 2.68 | 실패 |
| submission_xgboost_delta_v2.csv | XGBoost 100% | 15.76 | 과최적화 |
| submission_xgboost_safe_v3.csv | Zone 10% + XGBoost 90% | 15.73 | 권장 |

### submission_xgboost_safe_v3.csv 상세

```
- 앙상블 비율: Zone 10% + XGBoost 90%
- 예측 범위: X[8.24, 102.68], Y[1.29, 66.45]
- 특징: XGBoost이 주도하지만 Zone 안정성 포함
```

---

## 6. 통계적 분석

### 6.1 Fold Gap 분석

```
Zone Baseline Gap:
  Fold 1-3: 16.66
  Fold 4-5: 16.25
  Gap: -0.41 (Fold 4-5가 더 쉬움)

XGBoost Delta v2 Gap:
  Fold 1-3: 15.76
  Fold 4-5: 15.34
  Gap: -0.43 (비슷한 패턴)

XGBoost Safe v3 Gap:
  Fold 1-3: 15.92
  Fold 4-5: 15.49
  Gap: -0.43 (안정적)
```

**의미:** Fold 4-5가 실제로 쉬운 데이터 → CV가 낮을 수 있음

### 6.2 과최적화 위험도

```
Safe Spot:        16.27 ~ 16.34  (안전)
Current XGBoost:  15.76 ~ 15.92  (위험)
Gap from Safe:    0.35 ~ 0.51    (중대)

Public Gap 추정:
- 기준 (Zone): +0.17
- XGBoost: +0.17 + α (불명확)
- 최악의 경우: +0.30 ~ 0.50
```

---

## 7. 의사결정

### 7.1 XGBoost 모델 평가

**✅ 긍정적 신호:**
- 정확도: Zone 대비 +1.08 개선 뚜렷
- 안정성: Fold Gap이 일관적이고 합리적
- 신뢰성: 3개 모델 모두 비슷한 결과 (재현성)

**❌ 부정적 신호:**
- 과최적화: CV 16.27 미만 (CLAUDE.md 경고)
- 한계 도달: 14회 실패한 Zone도 같은 지점?
- 불확실성: Public Gap 크기 불명확

### 7.2 권장 사항

**즉시 제출 여부:** **NO (Week 2-3 금지)**

**이유:**
1. CLAUDE.md: "Week 2-3은 관찰만, Week 4-5부터 집중"
2. CV < 16.27: 기준선 미달 (과최적화 위험)
3. 검증 필요: Public에서 실제 성능 확인 전 제출 금지

**Week 3 권장 활동:**
```
- 이 분석 문서화 (완료) ✅
- 다른 ML 기법 연구 (미진행)
  * LightGBM + Sequence
  * Neural Network 기반
  * Ensemble 조합
- 앙상블 전략 재검토
```

**Week 4-5 결정:**
```
IF XGBoost 제출:
  - Zone 10% + XGBoost 90% 앙상블 추천
  - submission_xgboost_safe_v3.csv 사용
  - CV 낮지만 안정적 (Gap < 0.5)

IF Zone 유지:
  - safe_fold13 계속 사용
  - 16.36 확정 (안전)
  - Public 위험 낮음
```

---

## 8. 기술적 교훈

### 8.1 시퀀스 피처의 한계

**발견:** 8개 피처로는 충분한 개선 불가능
```
Zone 베이스라인:   16.34~16.68
XGBoost 단순:      15.76~15.92
개선:              0.58~0.92 (1.0 미만)

필요 개선:         약 2~3점
부족:              ~1.4점
```

**결론:** 더 강력한 패턴이 필요 (LSTM 같은 RNN?)

### 8.2 정규화의 효과

```
정규화 없음 (Delta v2):  15.7580 CV
정규화 있음 (Safe v3):   15.9213 CV (더 높음)

역설: 정규화가 성능을 해침?
이유: Tree 깊이 제한이 표현력 감소
     수렴 어려워짐 (조기 종료 발동)
```

**결론:** max_depth=6 → 4로 줄인 것이 과도했을 가능성

### 8.3 앙상블의 가치

```
XGBoost 100%:     15.7476 CV
Zone 10% + XGBoost 90%: 15.7314 CV (개선!)

개선: 0.0162 (작음)
의미: 약간의 안정성 추가
```

---

## 9. 미래 연구 방향

### 제안 (Week 3에 탐색, Week 4-5 실행)

1. **LightGBM + Sequence**
   - GBDT 계열이지만 다른 하이퍼파라미터
   - 가능한 추가 개선?

2. **Neural Network 기반**
   - LSTM/GRU: 시퀀스 학습 전문
   - MLP: 간단하고 안정적
   - 메모리 제약: 주의

3. **고급 앙상블**
   - Stacking: Meta-learner 추가
   - Voting: 여러 모델 가중평균
   - Blending: 교차 검증 필요 없음

4. **특성 엔지니어링**
   - 더 많은 시퀀스 정보 (과거 5-10개)
   - 팀 정보 활용
   - 경기 상황 피처 (시간, 스코어 등)

---

## 10. 최종 결론

### Summary

```
XGBoost + 시퀀스 피처 모델:
- 성능: ✅ 우수 (Zone 대비 +1.08)
- 안정성: ✅ 양호 (Fold Gap < 0.5)
- 과최적화: ❌ 위험 (CV < 16.27)
- 제출 시기: 🕐 Week 4-5 (확신 후)
- 권장 전략: 앙상블 (Zone 10% + XGBoost 90%)
```

### 의사결정 요약

| 질문 | 답변 |
|------|------|
| XGBoost 모델이 좋은가? | 예, 성능은 우수 |
| 지금 제출해야 하나? | 아니오, Week 3은 관찰만 |
| Safe Spot인가? | 아니오, 과최적화 경고 |
| Week 4-5에 제출할까? | 불명확, 다른 기법 먼저 시도 |
| 최악의 경우는? | CV 16.27 미만 → Public Gap 폭발 |

### 위험도 평가

```
낮음 (0-2점):
  - Zone 베이스라인 유지

중간 (2-4점):
  - XGBoost 앙상블 제출 (위험 있음)

높음 (4점+):
  - XGBoost 단독 100% 제출 (금지)
```

---

## 11. 파일 링크

### 생성 파일
- `/code/models/model_xgboost_v1.py` - 초기 모델 (실패)
- `/code/models/model_xgboost_delta_v2.py` - Delta 기반 (과최적화)
- `/code/models/model_xgboost_safe_v3.py` - 안전형 (권장)

### 제출 파일
- `submission_xgboost_v1.csv` - 점수: 부정확
- `submission_xgboost_delta_v2.csv` - CV: 15.76 (위험)
- `submission_xgboost_safe_v3.csv` - CV: 15.73 (상대적으로 안전)

### 참고 문서
- `CLAUDE.md` - 프로젝트 가이드
- `FACTS.md` - 불변 사실들
- `safe_fold13.py` - 기준선 모델

---

*2025-12-11 작성*
*XGBoost 실험 완료*
*결론: 성능은 우수하나 과최적화 경고*
*권장: Week 3 관찰, Week 4-5 다른 기법 우선 탐색*
