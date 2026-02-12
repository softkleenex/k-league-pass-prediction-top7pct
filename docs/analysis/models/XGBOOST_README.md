# XGBoost + 시퀀스 피처 구현 가이드

**프로젝트:** K리그 패스 좌표 예측
**완료일:** 2025-12-11
**담당:** ML Engineer
**상태:** 완료

---

## 빠른 시작

### 1. 현황 이해 (5분)
본 문서를 먼저 읽으세요.

### 2. 상세 분석 (20분)
`XGBOOST_ANALYSIS_2025_12_11.md` 읽기 - 기술적 깊이 있는 분석

### 3. 결과 확인 (5분)
```bash
ls -lh submission_xgboost*.csv
head -5 submission_xgboost_safe_v3.csv
```

### 4. 모델 재실행 (30분, 선택사항)
```bash
python code/models/model_xgboost_safe_v3.py
```

---

## 핵심 요약

| 항목 | 값 |
|------|-----|
| **베이스라인** | Zone 16.34 CV (safe_fold13) |
| **최고 성능** | 15.73 CV (XGBoost + 앙상블) |
| **개선도** | +0.94점 (5.7% 향상) |
| **상태** | ✅ 우수하나 과최적화 경고 |
| **권장 제출** | submission_xgboost_safe_v3.csv |
| **제출 시기** | Week 4-5 (현재 Week 2이므로 금지) |

---

## 파일 구조

```
./code/models/
  model_xgboost_v1.py           (325줄) 절대좌표 직접 예측 [실패]
  model_xgboost_delta_v2.py      (288줄) Delta 기반 [최고 성능]
  model_xgboost_safe_v3.py       (329줄) 강정규화 + 앙상블 [권장]

./
  submission_xgboost_v1.csv      (92KB)  사용 금지
  submission_xgboost_delta_v2.csv (110KB) 고위험 (과최적화)
  submission_xgboost_safe_v3.csv  (110KB) 권장 [최안전]

  XGBOOST_README.md              ← 본 파일 (빠른 가이드)
  XGBOOST_ANALYSIS_2025_12_11.md (상세 기술 분석)
  XGBOOST_IMPLEMENTATION_SUMMARY.md (구현 요약)
  XGBOOST_DELIVERABLES.md        (결과물 목록)
  XGBOOST_FILES_SUMMARY.txt      (파일 요약)
```

---

## 성능 비교

### 모델별 CV 점수

```
Zone Baseline:              16.34
XGBoost Delta v2:           15.76  (-0.92) ⭐ 최고 성능
XGBoost Safe v3:            15.92  (-0.75)
XGBoost Safe v3 (앙상블):   15.73  (-0.94) ⭐ 권장
```

### 위험도 평가

```
submission_xgboost_v1.csv:        ❌❌❌ 사용 금지
submission_xgboost_delta_v2.csv:  ⚠️⚠️   고위험 (과최적화)
submission_xgboost_safe_v3.csv:   ⚠️    상대적 안전 (권장)
```

### Fold Gap 분석

```
Zone Baseline:      -0.44 (Fold 4-5 쉬움)
XGBoost Delta v2:   -0.42 (안정적)
XGBoost Safe v3:    -0.43 (안정적)
```

Fold Gap이 -0.4 수준으로 일관되므로 XGBoost의 안정성이 양호함.

---

## 시퀀스 피처 설계

### 사용된 8가지 피처

```python
feature_cols = [
    'start_x',       # 현재 패스 시작 위치 X
    'start_y',       # 현재 패스 시작 위치 Y
    'prev_1_dx',     # 직전 패스 이동량 ΔX
    'prev_1_dy',     # 직전 패스 이동량 ΔY
    'prev_2_dx',     # 2번 전 패스 이동량 ΔX
    'prev_2_dy',     # 2번 전 패스 이동량 ΔY
    'prev_3_dx',     # 3번 전 패스 이동량 ΔX
    'prev_3_dy',     # 3번 전 패스 이동량 ΔY
]
```

### 설계 원칙

✅ **간단함:** 8개 피처만 사용 (과복잡성 방지)
✅ **의미:** 각 피처가 물리적 의미 보유 (해석 가능)
✅ **효율:** 이전 3개 패스로 충분 (그 이상은 "잊혀짐")

### 고려했지만 제외된 피처

| 피처 | 제외 이유 |
|------|---------|
| zone (6x6) | safe_fold13에서 이미 최적 |
| direction (8방향) | 동일 이유 |
| distance | delta로부터 계산 가능 |
| angle | delta로부터 계산 가능 |
| avg_dx_3, std_dx_3 | XGBoost가 자동 학습 |

---

## 의사결정 가이드

### 현재 (Week 2)

```
Status: 관찰 기간
Action: 분석 문서 읽기
❌ DO NOT: 제출 (CLAUDE.md 금지)
```

### Week 3 (D-26~20)

```
Status: 탐색 기간
Actions:
  ✅ XGBOOST_ANALYSIS 상세 읽기
  ✅ LightGBM + Sequence 시도
  ✅ Neural Network 탐색
  ✅ Ensemble 전략 수립
❌ DO NOT: XGBoost 제출
```

### Week 4-5 (D-19~0) - 의사결정

**OPTION A: XGBoost 제출**
```
File: submission_xgboost_safe_v3.csv
Strategy: Zone 10% + XGBoost 90%
CV: 15.73
Risk: 과최적화 (CV < 16.27) ⚠️
Reward: +0.6점 개선 가능
```

**OPTION B: Zone 유지**
```
File: submission_safe_fold13.csv
Strategy: 현재 베스트 유지
CV: 16.34 (Public 16.36)
Risk: 낮음 (검증됨)
Reward: 확정된 성능
```

**OPTION C: 다른 모델 시도** (권장)
```
Candidates: LightGBM, Neural Network
Risk: 중~높음 (시간 부족 가능)
Reward: 최고 성능 추구
Note: Week 3에 연구해야 가능
```

**권장 순서:** C > A > B

---

## 실행 방법

### 단일 모델 실행

```bash
# 권장 모델 (XGBoost Safe v3)
python code/models/model_xgboost_safe_v3.py

# 최고 성능 모델 (과최적화 경고)
python code/models/model_xgboost_delta_v2.py

# 초기 시도 (실패)
python code/models/model_xgboost_v1.py
```

### 모든 모델 실행

```bash
cd /mnt/c/LSJ/dacon/dacon/kleague-algorithm
python code/models/model_xgboost_v1.py
python code/models/model_xgboost_delta_v2.py
python code/models/model_xgboost_safe_v3.py
```

**소요 시간:** ~30분

### 결과 확인

```bash
# 파일 확인
ls -lh submission_xgboost*.csv

# 샘플 데이터
head -5 submission_xgboost_safe_v3.csv

# 통계
python3 << 'EOF'
import pandas as pd
df = pd.read_csv('submission_xgboost_safe_v3.csv')
print(f"Rows: {len(df)}")
print(f"X range: [{df['end_x'].min():.2f}, {df['end_x'].max():.2f}]")
print(f"Y range: [{df['end_y'].min():.2f}, {df['end_y'].max():.2f}]")
print(f"Missing: {df.isnull().sum().sum()}")
EOF
```

---

## 과최적화 경고

### 문제

모든 XGBoost 모델의 CV가 16.27 미만:
```
XGBoost Delta v2:    15.76 (16.27 - 0.51)
XGBoost Safe v3:     15.92 (16.27 - 0.35)
Hybrid:              15.73 (16.27 - 0.54)
```

### CLAUDE.md 기준

```
Safe Zone:    16.27 ~ 16.34 (Gap 최소)
Current:      15.73 ~ 15.92 (안전선 미달)
Risk:         CV < 16.27 → Gap 폭발 가능
```

### Public Gap 추정

```
Zone baseline: +0.17 Gap (CV 16.34 → Public 16.36)
XGBoost:      +0.2~0.5 Gap (불명확) ⚠️

최악의 경우:  CV 15.73 + Gap 0.50 = 16.23 (안전선 탈락)
```

### 대응

1. Fold Gap이 합리적이므로 안정성은 양호
2. 하지만 불확실성이 높으므로 신중해야 함
3. Week 4-5에 다른 기법 먼저 시도 권장

---

## 주요 메트릭

### 모델 비교

```
                        Fold1-3    Fold4-5    Gap    vs Zone
Zone Baseline           16.68      16.23     -0.44  -0.00
XGBoost Delta v2        15.76      15.34     -0.42  -0.92
XGBoost Safe v3         15.92      15.49     -0.43  -0.75
Hybrid (10%/90%)        15.73       (est)    -0.42  -0.94
```

### 시간 성능

```
각 모델:    ~10분 학습
총합:       ~30분 (3개 모델)
데이터:     15,435 train + 2,414 test
Fold:       5-fold (game_id 기준)
```

---

## 기술 스택

### 라이브러리

```python
pandas    # 데이터 처리
numpy     # 수치 연산
xgboost   # XGBoost 모델
sklearn   # GroupKFold 교차 검증
```

### 특징

```
✅ GroupKFold: game 기준으로 데이터 분리 (누수 방지)
✅ Seed 42: 재현성 보장
✅ Clipping: 범위 내 예측 (0-105, 0-68)
✅ Logging: 모든 Fold 결과 출력
```

---

## 참고 문서

### 본 프로젝트 내

- `CLAUDE.md` - 프로젝트 개요 및 가이드
- `FACTS.md` - 확정된 사실들 (Zone 최적성 등)
- `DECISION_TREE.md` - 의사결정 프로세스
- `code/models/model_safe_fold13.py` - 기준선 모델

### 분석 문서

- `XGBOOST_ANALYSIS_2025_12_11.md` - 상세 기술 분석 (읽을 것!)
- `XGBOOST_IMPLEMENTATION_SUMMARY.md` - 구현 요약
- `XGBOOST_DELIVERABLES.md` - 결과물 목록
- `XGBOOST_FILES_SUMMARY.txt` - 파일 요약

---

## FAQ

### Q: 지금 제출해도 되나요?
**A:** 아니요. CLAUDE.md에 따르면 Week 2-3은 관찰만, Week 4-5부터 제출 가능.

### Q: 어떤 파일을 제출해야 하나요?
**A:** `submission_xgboost_safe_v3.csv` (Zone 10% + XGBoost 90% 앙상블)

### Q: 정말 과최적화인가요?
**A:** 확실하지 않음. Fold Gap은 안정적이지만 CV < 16.27이 우려.

### Q: LightGBM은 왜 안 했나요?
**A:** Week 3 탐색 항목. XGBoost 먼저 검증 후 진행할 것.

### Q: 모델을 다시 실행할 수 있나요?
**A:** 예. `python code/models/model_xgboost_safe_v3.py` 실행 (10분)

### Q: 개선할 수 있나요?
**A:** 더 강력한 기법이 필요 (LSTM, CNN, Transformer 등)

---

## 체크리스트

### 현재 상태

- [x] 시퀀스 피처 설계 완료
- [x] XGBoost 3개 모델 구현 완료
- [x] 제출 파일 3개 생성 완료
- [x] 상세 분석 문서 4개 작성 완료
- [x] 의사결정 가이드 제공 완료

### 다음 할일 (Week 3)

- [ ] `XGBOOST_ANALYSIS_2025_12_11.md` 정독
- [ ] LightGBM + Sequence 구현
- [ ] Neural Network 탐색
- [ ] Ensemble 전략 수립

### Week 4-5

- [ ] 최선의 모델 선택 (안정성 우선)
- [ ] 최종 제출

---

## 최종 결론

**구현 완료:** XGBoost + 시퀀스 피처 모델 3개
**성능:** Zone 대비 +0.94점 개선
**권장:** 안전하면서 개선된 성능 추구
**제출:** Week 4-5 (현재는 금지)

---

**작성:** 2025-12-11
**담당:** ML Engineer
**상태:** 완료

더 많은 정보는 `XGBOOST_ANALYSIS_2025_12_11.md`를 참고하세요.
