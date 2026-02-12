# 필드 영역 기반 모델 개발 분석 보고서

## 개발 목표
TacticAI 논문의 인사이트를 반영하여 필드 영역별로 다른 패턴을 학습하는 모델 개발

## 모델 아키텍처

### 필드 3분할 전략
```
Field Division (105m × 68m):
├── Defensive (x < 35m):   수비 영역
├── Midfield (35 ≤ x < 70m): 중앙 영역
└── Attacking (x ≥ 70m):   공격 영역
```

### 개발 과정

#### Version 1: 영역별 차별화된 Zone 해상도
- **전략**: 영역 특성에 맞는 Zone 크기 적용
  - Defensive: 4x4 (넓은 Zone, 수비는 단순)
  - Midfield: 6x6 (중간 Zone, 중앙 밀집)
  - Attacking: 8x8 (세밀한 Zone, 공격 정교)

- **결과**: CV 16.3812 ± 0.2673
  - Defensive: 20.4593 (4x4)
  - Midfield: 17.8586 (6x6)
  - Attacking: 11.4429 (8x8)

- **문제점**:
  - Defensive: Zone이 너무 넓어 정보 손실
  - Attacking: Zone이 너무 세밀해 샘플 부족 (과소적합)

#### Version 2: 균등 Zone + 최적화된 min_samples
- **전략**: 모든 영역에 6x6 적용, min_samples 조정
  - Defensive: min_samples=18
  - Midfield: min_samples=22
  - Attacking: min_samples=16
  - 5단계 계층적 fallback 로직

- **결과**: CV 16.2664 ± 0.3089
  - Defensive: 19.9760 (6x6)
  - Midfield: 17.8583 (6x6)
  - Attacking: 11.4497 (6x6)

- **개선**:
  - v1 대비: +0.11 (개선)
  - 8방향 대비: -0.24 (악화)

## 성능 비교

| 모델 | CV Score | Train Score | Gap | 영역별 성능 | 비고 |
|------|----------|-------------|-----|-------------|------|
| **8방향 앙상블 (Best)** | **16.03** | - | **+0.33** | - | 현재 Best |
| 필드 영역 v1 | 16.38 | 16.13 | +0.20-0.30 | D:20.5, M:17.9, A:11.4 | 차별화 Zone |
| 필드 영역 v2 | 16.27 | 15.98 | +0.17-0.25 | D:20.0, M:17.9, A:11.5 | 균등 Zone |

## 핵심 발견

### 1. 영역별 성능 차이
```
Attacking (x≥70):  11.4-11.5 (매우 우수)
Midfield (35≤x<70): 17.9      (보통)
Defensive (x<35):  20.0-20.5  (어려움)
```

**인사이트**:
- 공격 지역: 패턴 명확 (슈팅, 크로스 등)
- 중앙 지역: 다양한 플레이로 예측 어려움
- 수비 지역: 클리어런스, 긴 패스로 변동성 큼

### 2. Zone 해상도 영향
- **4x4**: 너무 넓어 세밀한 패턴 포착 실패
- **6x6**: 균형잡힌 성능 (Best)
- **8x8**: 샘플 부족으로 과소적합

### 3. CV-Public Gap 안정성
```
CV 16.27 (안전 구간)
→ 예상 Gap: +0.17-0.25
→ 예상 Public: 16.44-16.52
```

## 제출 전략

### 제출 우선순위

| 우선순위 | 파일 | CV | 예상 Public | 제출 여부 |
|:-------:|------|:--:|:-----------:|:---------:|
| 1 | **submission_8direction_safe.csv** | **16.03** | **16.20-16.36** | ✅ 제출 완료 (Best) |
| 2 | submission_field_region_v2.csv | 16.27 | 16.44-16.52 | ⚠️  보류 |
| 3 | submission_field_region.csv | 16.38 | 16.55-16.65 | ❌ 제출 안함 |

### 제출 권장사항

**필드 영역 v2 (CV 16.27)**:
- ✅ **장점**:
  - CV 안전 구간 (16.2+)
  - Gap 안정적 (+0.17-0.25)
  - 새로운 접근 방식

- ❌ **단점**:
  - 8방향 Best보다 CV 높음 (+0.24)
  - 예상 Public이 현재 Best보다 나쁨

- **결론**: **제출 보류** 권장
  - 이유: Public 개선 가능성 낮음
  - 대안: 앙상블 또는 다른 피처 탐색

## 학습된 교훈

### 1. 영역별 전략의 한계
필드 영역으로 분할하는 것만으로는 성능 개선 불충분
→ 영역별 특성을 반영한 추가 피처 필요

### 2. Zone 해상도는 균등이 유리
영역별로 다른 Zone 크기보다 **균등한 6x6**이 더 안정적

### 3. 계층적 Fallback의 중요성
5단계 fallback 로직으로 안정성 확보:
1. Region + Zone + Direction
2. Region + Zone
3. Region + Direction
4. Zone + Direction
5. Zone only

### 4. CV 16.2+ 유지의 중요성
CV가 16.2 이상이면 Gap이 안정적 (+0.15-0.25)

## 다음 단계 제안

### 1. 앙상블 전략
```python
# 8방향 + 필드 영역 앙상블
pred = 0.7 * pred_8direction + 0.3 * pred_field_region
```

### 2. 추가 피처 탐색
- **영역 전환 패턴**: 수비→공격 전환 시 패턴
- **시간 피처**: 전반/후반, 추가시간
- **팀 특성**: 팀별 플레이 스타일

### 3. 영역별 세분화
```
Defensive → [Clear, Short, Build-up]
Midfield  → [Lateral, Forward, Back]
Attacking → [Cross, Through, Shot]
```

## 코드 위치

### 모델 파일
- `/mnt/c/LSJ/dacon/dacon/kleague-algorithm/code/models/model_field_region.py`
- `/mnt/c/LSJ/dacon/dacon/kleague-algorithm/code/models/model_field_region_optimized.py`

### 제출 파일
- `/mnt/c/LSJ/dacon/dacon/kleague-algorithm/submission_field_region.csv` (CV 16.38)
- `/mnt/c/LSJ/dacon/dacon/kleague-algorithm/submission_field_region_v2.csv` (CV 16.27)

## 결론

필드 영역 기반 모델은 **안전한 CV (16.27)**를 달성했으나, 현재 Best (16.03) 대비 성능 개선은 없었다.

**핵심 인사이트**:
1. 공격 지역 예측이 가장 쉬움 (11.4)
2. 수비 지역 예측이 가장 어려움 (20.0)
3. 6x6 균등 Zone이 최적
4. CV 16.2+ 유지 시 Gap 안정적

**제출 전략**:
- 현재는 **제출 보류**
- 앙상블 또는 추가 피처로 CV 16.0 이하 달성 후 재검토

---

*작성일: 2025-12-04*
*작성자: ML Engineer Agent*
