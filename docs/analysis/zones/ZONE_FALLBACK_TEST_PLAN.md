# Zone Fallback 개선 - 테스트 계획

**작성:** 2025-12-11 (D-32)
**실행 예정:** Week 4 (D-19~13, 12/23-12/29)
**목적:** Zone fallback 개선 효과 검증 및 제출 준비
**목표:** CV ~0.01 향상, Public 16.36 → 16.35

---

## 🎯 개선 내용 요약

### 기존 코드 (model_safe_fold13.py)
```python
# Zone fallback (180-183행, 333-336행)
zone_fallback = train_fold_temp.groupby('zone').agg({
    'delta_x': 'median',
    'delta_y': 'median'
}).to_dict()

# 예측 로직 (194-200행)
elif row['zone'] in zone_fallback['delta_x']:
    dx = zone_fallback['delta_x'][row['zone']]  # ← 문제: count 체크 없음!
    dy = zone_fallback['delta_y'][row['zone']]
```

**문제점:**
- Zone fallback 사용 시 샘플 수 확인 없음
- min_samples < 25인 Zone도 사용 → 불안정

### 개선 코드 (model_safe_fold13_improved.py)
```python
# Zone fallback (195-199행, 379-383행)
zone_fallback = train_fold_temp.groupby('zone').agg({
    'delta_x': 'median',
    'delta_y': 'median',
    'game_episode': 'count'  # ← 추가!
}).rename(columns={'game_episode': 'count'})

# 예측 로직 (209-215행)
elif row['zone'] in zone_fallback.index and zone_fallback.loc[row['zone'], 'count'] >= min_s:
    dx = zone_fallback.loc[row['zone'], 'delta_x']  # ← min_samples 체크 추가!
    dy = zone_fallback.loc[row['zone'], 'delta_y']
# 3순위: Global fallback (샘플 부족 시)
else:
    dx = global_dx
    dy = global_dy
```

**개선점:**
- ✅ Zone fallback에 count 컬럼 추가
- ✅ min_samples >= 25 체크 추가
- ✅ 샘플 부족 시 global fallback으로 안전 전환

---

## 📋 테스트 체크리스트

### Phase 1: 로컬 검증 (Week 4 초반, 12/23-12/24)

**1.1 코드 리뷰**
- [ ] 문법 오류 확인 (python -m py_compile)
- [ ] import 문 확인
- [ ] 경로 확인 (DATA_DIR)
- [ ] 함수 시그니처 일관성

**1.2 Dry Run (실행하지 않고 검토)**
- [ ] 로직 흐름 검토
- [ ] Edge case 시나리오 확인
  - Zone fallback count = 24 (min_samples 미만)
  - Zone fallback count = 25 (경계값)
  - Zone fallback count = 26 (정상)
  - Zone이 존재하지 않는 경우

**1.3 예상 동작 시나리오**
```
시나리오 1: Zone+Direction 충분 (count >= 25)
→ Zone+Direction 통계 사용 ✅

시나리오 2: Zone+Direction 부족, Zone 충분
→ Zone 통계 사용 (개선: count >= 25 체크) ✅

시나리오 3: Zone+Direction 부족, Zone도 부족
→ Global fallback 사용 (개선: 안전) ✅

시나리오 4: Zone 존재하지 않음
→ Global fallback 사용 ✅
```

### Phase 2: CV 확인 (Week 4 초반, 12/24-12/25)

**2.1 실행 준비**
- [ ] 데이터 확인 (train.csv, test.csv)
- [ ] 환경 확인 (Python, pandas, numpy, sklearn)
- [ ] 백업 (기존 submission 파일)

**2.2 실행**
```bash
# Week 4에 실행
python code/models/model_safe_fold13_improved.py
```

**2.3 CV 기록**
```
예상 결과:
- Fold 1 CV: 16.32-16.34 (기존: 16.34)
- Fold 2 CV: 16.32-16.34 (기존: 16.33)
- Fold 3 CV: 16.32-16.35 (기존: 16.34)
- Fold 1-3 평균: 16.32-16.34 (기존: 16.34)

목표: CV 16.27-16.34 범위 내
최선: CV 16.33 (0.01 향상)
최악: CV 16.34 (변화 없음, 안정성만 향상)
```

**2.4 CV Sweet Spot 확인**
- [ ] CV < 16.27? → ❌ 즉시 중단 (과최적화)
- [ ] CV 16.27-16.34? → ✅ 제출 진행
- [ ] CV > 16.34? → ⚠️ 분석 필요

### Phase 3: 제출 결정 (Week 4 중반, 12/25-12/26)

**3.1 제출 조건 체크**
- [ ] CV 16.27-16.34 범위 ✅
- [ ] DECISION_TREE.md 승인 ✅
- [ ] 제출 5회 이상 남음 ✅
- [ ] 코드 검증 완료 ✅
- [ ] 예측 파일 형식 확인 ✅

**3.2 제출 전 최종 점검**
```bash
# 제출 파일 확인
wc -l submission_safe_fold13_improved.csv  # 2415 라인 (헤더 포함)
head -5 submission_safe_fold13_improved.csv  # 형식 확인
tail -5 submission_safe_fold13_improved.csv  # 끝 확인

# 범위 확인
python -c "
import pandas as pd
df = pd.read_csv('submission_safe_fold13_improved.csv')
print('end_x range:', df['end_x'].min(), '-', df['end_x'].max())
print('end_y range:', df['end_y'].min(), '-', df['end_y'].max())
print('NaN count:', df.isna().sum().sum())
"
# 예상: end_x 0-105, end_y 0-68, NaN 0
```

**3.3 제출 (Week 4)**
- [ ] 제출 파일: submission_safe_fold13_improved.csv
- [ ] 제출 시간: 오후 (리더보드 활동 적은 시간)
- [ ] 제출 메모: "Zone fallback 개선 (min_samples 체크 추가)"

### Phase 4: 결과 분석 (제출 후 즉시)

**4.1 Public Score 분석**
```
시나리오 A: Public < 16.36 (개선!)
→ ✅ Best 갱신!
→ STATUS.md, FACTS.md 업데이트
→ Gap 분석

시나리오 B: Public = 16.36-16.37 (유사)
→ ⚠️ 미미한 개선
→ 안정성 향상으로 간주
→ 유지 또는 교체 검토

시나리오 C: Public > 16.37 (악화)
→ ❌ 개선 실패
→ safe_fold13 유지
→ 원인 분석
```

**4.2 Gap 분석**
```python
gap = public_score - cv_fold13
print(f"Gap: {gap:+.4f}")

# 평가
if gap < 0.05:
    print("✅ 최소 Gap (이상적)")
elif gap < 0.10:
    print("✅ 정상 Gap")
elif gap < 0.15:
    print("⚠️ 높은 Gap (주의)")
else:
    print("❌ 과최적화")
```

**4.3 기록**
- [ ] EXPERIMENT_LOG.md 업데이트
- [ ] FACTS.md 업데이트 (Best 갱신 시)
- [ ] STATUS.md 업데이트
- [ ] 제출 횟수 기록 (14/175)

---

## 🔬 예상 효과 분석

### 긍정적 시나리오 (60% 확률)
**효과:**
- CV 16.33-16.34 (0.00-0.01 향상)
- Public 16.35-16.36 (0.00-0.01 향상)
- 안정성 증가 (Zone fallback 신뢰도 향상)

**결과:**
- Best 유지 또는 미미한 개선
- 안정성 확보
- Week 4-5 기반 모델로 활용

### 중립적 시나리오 (30% 확률)
**효과:**
- CV 16.34 (변화 없음)
- Public 16.36 (변화 없음)
- 안정성 향상만

**결과:**
- safe_fold13 유지
- 개선 없지만 실패도 아님
- 안정성 확보로 만족

### 부정적 시나리오 (10% 확률)
**효과:**
- CV 16.35+ (악화)
- Public 16.37+ (악화)

**결과:**
- safe_fold13 유지
- 개선 포기
- 다른 아이디어 검토 (Velocity features?)

---

## 🚨 리스크 관리

### Risk 1: CV Sweet Spot 이탈
**확률:** 5%
**영향:** 높음 (과최적화)
**대응:**
- CV < 16.27 → 즉시 중단, 코드 폐기
- CV > 16.34 → 원인 분석, 재검토

### Risk 2: Public 악화
**확률:** 10%
**영향:** 중간 (제출 낭비)
**대응:**
- safe_fold13 유지
- 제출 1회 소진 (161회 남음, 문제 없음)

### Risk 3: 코드 오류
**확률:** 5%
**영향:** 중간 (시간 낭비)
**대응:**
- Phase 1 코드 리뷰 철저히
- Dry run으로 사전 검증

### Risk 4: 변화 없음
**확률:** 30%
**영향:** 낮음 (안정성 향상)
**대응:**
- 실패가 아님, 안정성 확보
- 다른 아이디어 검토 (Velocity features)

---

## 📊 성공 지표

### 최소 목표 (100% 달성 필요)
- [ ] CV Sweet Spot 유지 (16.27-16.34)
- [ ] 코드 오류 없음
- [ ] 제출 성공

### 목표 (80% 달성 희망)
- [ ] CV 16.33 이하 (0.01 향상)
- [ ] Public 16.35 이하 (0.01 향상)
- [ ] Gap 0.05 이하

### 최선 (50% 달성 희망)
- [ ] CV 16.32 이하 (0.02 향상)
- [ ] Public 16.34 이하 (0.02 향상)
- [ ] Best 갱신

---

## 📅 실행 타임라인 (Week 4)

**12/23 (월, D-19):**
- 오전: Phase 1 코드 리뷰
- 오후: Phase 2 CV 확인 (실행)
- 저녁: 결과 분석

**12/24 (화, D-18):**
- 오전: Phase 3 제출 결정
- 오후: 제출 (조건 충족 시)
- 저녁: Phase 4 결과 분석

**12/25 (수, D-17):**
- 오전: 기록 업데이트
- 오후: 다음 실험 계획 (필요 시)

---

## ✅ 승인 체크리스트

### 실행 전 (Week 4 시작 시)
- [ ] DECISION_TREE.md 확인
- [ ] Week 4 전략 재확인
- [ ] 제출 횟수 확인 (5회 이상)
- [ ] 리더보드 상황 확인

### 실행 중
- [ ] CV 실시간 모니터링
- [ ] Sweet Spot 엄수
- [ ] 이상 징후 즉시 중단

### 실행 후
- [ ] 문서 업데이트
- [ ] 교훈 기록
- [ ] 다음 단계 계획

---

## 📝 최종 메시지

**Zone fallback 개선 = 안전하고 합리적인 시도**

**근거:**
- ✅ 문헌 조사로 검증된 방법
- ✅ 코드 검증 보고서에서 권장
- ✅ 리스크 낮음 (CV Sweet Spot 이탈 가능성 5%)
- ✅ 예상 효과 합리적 (CV ~0.01)

**원칙:**
- CV Sweet Spot 엄수
- 조급함 금지
- 데이터 기반 판단

**목표:**
- 최소: 안정성 확보
- 목표: CV ~0.01 향상
- 최선: Best 갱신

**자신감: 80%**

---

*작성: 2025-12-11 (Week 2)*
*실행: Week 4 (12/23-12/25)*
*문의: DECISION_TREE.md 참조*
