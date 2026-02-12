# 패스 거리 모델 개발 - 요약 보고서

**날짜**: 2025-12-04
**작업**: 패스 거리 조건화 모델 개발
**상태**: ✅ 완료

---

## 빠른 요약

### 개발 결과

✅ **3개 모델 개발 완료**
- Distance 기본 (6x6): CV 16.23
- Distance 보수적: CV 16.28
- Distance 앙상블: CV 15.99 (과적합)

❌ **현재 Best 대비 개선 실패**
- 현재 Best: 16.3574
- 예상 결과: 16.40-16.53
- 차이: 0.0-0.2 악화

⚠️ **제출 보류 권장**
- 개선 가능성 낮음
- 다른 방향 탐색 우선

---

## 모델 상세

### 1. Distance 6x6 단독 (최고 성능)

**파일**: `/mnt/c/LSJ/dacon/dacon/kleague-algorithm/submission_pass_distance_6x6_only.csv`

```
CV: 16.2275 ± 0.2800
예상 Public: 16.40-16.48
안전성: ✅ 안전 (CV >= 16.2)

구조:
- 8방향 분류
- 3단계 거리 (short < 10m, medium 10-25m, long > 25m)
- 6x6 Zone
- min_samples: 25

장점:
+ 안전 구간 유지
+ 거리 정보 활용

단점:
- 개선 미미 (vs Best 16.03)
- Public 16.40-16.48 예상
```

### 2. Distance 보수적 (가장 안정)

**파일**: `/mnt/c/LSJ/dacon/dacon/kleague-algorithm/submission_pass_distance_conservative.csv`

```
CV: 16.2815 ± 0.3207
예상 Public: 16.43-16.53
안전성: ✅ 안전 (CV >= 16.2)

구조:
- 5방향 분류 (단순화)
- 2단계 거리 (short < 15m, long >= 15m)
- 6x6 Zone
- min_samples: 30-40 (높음)

장점:
+ 높은 안정성
+ 과적합 위험 최소화

단점:
- CV 약간 높음
- 개선 가능성 낮음
```

### 3. Distance 앙상블 (보류)

**파일**: `/mnt/c/LSJ/dacon/dacon/kleague-algorithm/submission_pass_distance.csv`

```
CV: 15.9970 ± 0.2634
예상 Public: > 16.50
안전성: ❌ 위험 (CV < 16.0)

문제:
- 앙상블 시 CV 급락
- 과적합 신호 명확
- 제출 권장하지 않음
```

---

## 주요 발견

### 1. 패스 거리의 예측력

```
패스 거리 분포:
- Median: 9.19m
- Mean: 11.63m
- Range: 0-71.17m

대부분 짧은 패스 (< 15m)
→ 거리 구분이 유의미함
```

### 2. 최적 분류 전략

```
✅ 효과적:
- 2단계 거리 (15m 기준)
- 5방향
- 높은 min_samples (30-40)

❌ 과적합 위험:
- 3단계 거리
- 8방향
- 낮은 min_samples (20-25)
- 앙상블
```

### 3. CV-Public Gap

```
Distance 모델 예측:

CV 16.23 → Public 16.40-16.48 (Gap +0.17-0.25)
CV 16.28 → Public 16.43-53 (Gap +0.15-0.25)

vs 기존 모델:
CV 16.03 → Public 16.36 (Gap +0.33)
CV 16.19 → Public 16.36 (Gap +0.17)

→ Distance 모델이 더 안정적이지만
   절대 성능 개선은 미미
```

---

## 제출 권장사항

### 현재 상황

```
오늘 제출: 3회 / 5회
남은 제출: 2회
현재 Best: 16.3574 (8방향 모델)
```

### 권장: 제출 보류 ⏸️

**이유**:

1. **개선 가능성 낮음**
   - 예상 Public: 16.40-16.53
   - Best 대비: 0.0-0.2 악화

2. **안전성은 확보**
   - CV 16.2+ 유지
   - 과적합 위험 없음

3. **더 좋은 기회 대기**
   - 다른 피처 탐색
   - CV < 16.0 모델 발견 시 제출

### 대안: 조건부 제출

**만약 제출한다면**:

```
우선순위 1: Distance 6x6 단독
- CV 16.23 (가장 낮음)
- 예상 16.40-16.48
- 백업으로 보유

우선순위 2: Distance 보수적
- CV 16.28
- 예상 16.43-16.53
- 더 안정적
```

---

## 파일 위치

### 코드

```
기본 모델:
/mnt/c/LSJ/dacon/dacon/kleague-algorithm/code/models/model_pass_distance.py

보수적 모델:
/mnt/c/LSJ/dacon/dacon/kleague-algorithm/code/models/model_pass_distance_conservative.py
```

### 제출 파일

```
앙상블 (보류):
/mnt/c/LSJ/dacon/dacon/kleague-algorithm/submission_pass_distance.csv

6x6 단독 (백업):
/mnt/c/LSJ/dacon/dacon/kleague-algorithm/submission_pass_distance_6x6_only.csv

보수적 (백업):
/mnt/c/LSJ/dacon/dacon/kleague-algorithm/submission_pass_distance_conservative.csv
```

### 분석 문서

```
상세 분석:
/mnt/c/LSJ/dacon/dacon/kleague-algorithm/analysis_pass_distance_models.md

전체 비교:
/mnt/c/LSJ/dacon/dacon/kleague-algorithm/model_comparison_summary.md
```

---

## Cross Validation 상세

### Distance 6x6 단독

```
Fold 1: 16.4556
Fold 2: 16.4668
Fold 3: 16.4457
Fold 4: 15.8899 ⚠️ (변동)
Fold 5: 15.8794 ⚠️ (변동)

평균: 16.2275 ± 0.2800
```

Fold 4, 5에서 낮은 점수 → 특정 게임 패턴?

### Distance 보수적

```
Fold 1: 16.4423
Fold 2: 16.5143
Fold 3: 16.6529
Fold 4: 15.8569 ⚠️
Fold 5: 15.9412 ⚠️

평균: 16.2815 ± 0.3207
```

유사한 패턴 → Fold 4, 5 특성 확인 필요

---

## 기술적 인사이트

### 계층적 Fallback 전략

```python
# 효과적인 3단계 전략

Level 3: Zone + Direction + Distance
  if count >= 40:
    use detailed statistics
  else:
    ↓

Level 2: Zone + Direction
  if count >= 30:
    use medium statistics
  else:
    ↓

Level 1: Zone only
  always use (fallback)
```

이 구조가 과적합 방지에 효과적.

### min_samples의 임계값

```
threshold=20: CV 16.22, 앙상블 15.99 (과적합)
threshold=30: CV 16.28, 안정적
threshold=40: CV 16.35, 매우 안정

최적: 30-40 범위
```

높을수록 안전하지만 CV 약간 희생.

---

## 다음 단계

### 1. 추가 탐색 방향

**우선순위 높음**:
- 필드 영역 기반 (공격/중간/수비)
- 시퀀스 길이 조건화

**우선순위 중간**:
- 팀 전술 패턴
- 앙상블 개선 (CV 16.2+ 모델들만)

**우선순위 낮음**:
- 시간 요인 (전반/후반)

### 2. 제출 전략

**보수적 (권장)**:
1. 추가 피처 탐색
2. CV < 16.0 모델 발견 시만 제출
3. 리더보드 모니터링

**탐색적**:
1. Distance 6x6 즉시 제출
2. 결과 확인 후 추가 제출

### 3. 리스크 관리

```
현재 포지션: 강함 (Public 16.3574)
남은 시간: 40일
남은 제출: 오늘 2회

전략: 보수적 대기
- 더 좋은 기회 탐색
- 리스크 최소화
```

---

## 결론

### 개발 성과

✅ **성공**:
- 안전한 모델 개발 (CV 16.2+)
- 거리 정보 활용 검증
- 과적합 위험 관리

❌ **한계**:
- 큰 성능 개선 없음
- Public 16.40-16.53 예상 (vs Best 16.36)

### 최종 권장

**제출 보류**, 백업으로 보유

**이유**:
1. 현재 Best 충분히 좋음
2. 개선 가능성 낮음
3. 다른 방향 탐색 우선
4. 남은 시간 충분 (40일)

### 학습 포인트

1. **거리 정보는 유의미하지만 충분하지 않음**
2. **높은 threshold가 안전성 확보**
3. **앙상블이 항상 좋은 것은 아님**
4. **CV < 16.2는 위험 신호**

---

*보고서 작성: 2025-12-04 09:20*
*작성자: Data Scientist Agent*
