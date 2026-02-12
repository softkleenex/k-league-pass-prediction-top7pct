# Claude 메모리 전략

> **문제:** 장기 작업 시 메모리/컨텍스트 관리가 혼란스러워짐
> **해결:** 명확한 메모리 계층과 참조 규칙

**작성일:** 2025-12-16

---

## 🧠 메모리 계층

### 1. Hot Memory (세션 메모리)
**현재 작업 중인 정보**
- 지속: 1개 세션 (최대 몇 시간)
- 저장: Claude의 대화 컨텍스트
- 용도: 현재 작업 진행

**예시:**
- 현재 실험 중인 모델
- 방금 계산한 CV 값
- 진행 중인 Todo 리스트

### 2. Warm Memory (파일 캐시)
**자주 참조하는 정보**
- 지속: 며칠~몇 주
- 저장: 프로젝트 루트의 핵심 MD 파일
- 용도: 세션 시작 시 읽기

**파일:**
- `SUBMISSION_LOG.md` - 제출 기록 (SSOT)
- `CLAUDE.md` - Claude 가이드 (간결)
- `README.md` - 프로젝트 개요

**세션 시작 시 읽기:**
```markdown
1. CLAUDE.md (우선 읽기)
   → 현재 상태, Best 모델, 금지 사항

2. SUBMISSION_LOG.md (필요 시)
   → 제출 이력, 성능 비교

3. PIPELINE.md (작업 방법 모를 때)
   → 워크플로우 참조
```

### 3. Cold Memory (아카이브)
**과거 기록**
- 지속: 영구
- 저장: docs/archive/, 실험 폴더
- 용도: 특정 정보 검색 시만

**파일:**
- `docs/archive/EXPERIMENT_LOG.md` - 구 실험 기록
- `docs/archive/SCORES.md` - 구 제출 기록
- `experiments/exp_XXX/EXPERIMENT.md` - 개별 실험

---

## 📝 파일별 역할

### CLAUDE.md (Hot → Warm)
**역할:** Claude가 세션 시작 시 **반드시** 읽는 파일

**내용 (간결하게!):**
```markdown
## TL;DR (3 Lines)
- Best: [모델명] ([점수])
- 현재 작업: [상태]
- 금지: [하지 말아야 할 것]

## Quick Links
- [SUBMISSION_LOG.md] - 제출 기록
- [PIPELINE.md] - 워크플로우

## 핵심 사실
- Best 모델: [정보]
- Safe 모델: [정보]
- 제출 현황: X/175

## 금지 사항
❌ [절대 하지 말 것들]
```

**업데이트 주기:** 주요 변화 시 (Best 갱신, 전략 변경)

### SUBMISSION_LOG.md (Warm)
**역할:** 모든 제출의 **단일 진실 공급원 (SSOT)**

**내용:**
- 제출 이력 (전체)
- 성능 비교
- 전략적 인사이트

**업데이트 주기:** 매 제출 후

### PIPELINE.md (Cold → Warm)
**역할:** 워크플로우 레퍼런스

**내용:**
- 폴더 구조
- 실험/제출 워크플로우
- 파일 명명 규칙

**업데이트 주기:** 워크플로우 변경 시

### README.md (Cold)
**역할:** 프로젝트 개요

**내용:**
- 대회 소개
- 데이터 설명
- 시작 가이드

**업데이트 주기:** 거의 없음

---

## 🔄 세션 워크플로우

### 세션 시작 시

```markdown
1. 사용자 요청 확인

2. CLAUDE.md 읽기 (자동)
   → 현재 상태 파악
   → Best 모델 확인
   → 금지 사항 확인

3. 필요 시 추가 파일 읽기
   - 제출 관련 작업 → SUBMISSION_LOG.md
   - 워크플로우 질문 → PIPELINE.md
   - 특정 실험 → experiments/exp_XXX/

4. 작업 시작
```

### 작업 중

```markdown
Hot Memory 사용:
- 현재 CV 값
- 진행 중인 코드
- Todo 리스트

파일 참조:
- 성능 비교 필요 → SUBMISSION_LOG.md 읽기
- 과거 실험 참조 → experiments/ 읽기
```

### 세션 종료 시

```markdown
1. 작업 결과 저장
   - 코드 → experiments/exp_XXX/
   - 제출 → SUBMISSION_LOG.md 업데이트

2. 중요 변화 시 CLAUDE.md 업데이트
   - Best 모델 갱신
   - 전략 변경
   - 새로운 금지 사항

3. Hot Memory 버림
   - 다음 세션은 파일에서 복원
```

---

## ⚠️ 주의사항

### DON'T ❌

1. **Hot Memory에 과거 기록 저장**
   ```
   ❌ "3주 전 실험 X의 CV는 Y였어"
   → Hot Memory는 현재 작업만!
   → 과거 기록은 파일 참조
   ```

2. **파일 없이 추측**
   ```
   ❌ "Best 모델은 아마 Zone 6x6일거야"
   → SUBMISSION_LOG.md 읽고 확인!
   ```

3. **CLAUDE.md를 너무 길게**
   ```
   ❌ 10,000단어짜리 CLAUDE.md
   → 간결하게! (<200 lines)
   → 상세 내용은 다른 파일로
   ```

4. **여러 파일에 동일 정보**
   ```
   ❌ CLAUDE.md + SUBMISSION_LOG.md에 모두 제출 기록
   → SSOT 원칙! SUBMISSION_LOG.md만!
   → CLAUDE.md는 링크만
   ```

### DO ✅

1. **파일 = 진실**
   ```
   ✅ Hot Memory와 파일 불일치 시
   → 파일이 정답
   → Hot Memory 버리고 파일 재로딩
   ```

2. **CLAUDE.md = 진입점**
   ```
   ✅ 세션 시작 시 CLAUDE.md부터
   → 현재 상태 파악
   → 다른 파일 링크
   ```

3. **즉시 기록**
   ```
   ✅ 중요한 결과 → 즉시 파일 저장
   → Hot Memory는 휘발성!
   ```

4. **간결한 CLAUDE.md**
   ```
   ✅ TL;DR 3 lines
   ✅ Quick Links
   ✅ 핵심 사실만
   → 상세 내용은 링크
   ```

---

## 📊 정보 수명

| 정보 유형 | 수명 | 저장 위치 | 예시 |
|----------|------|-----------|------|
| **진행 중 작업** | 몇 시간 | Hot Memory | 현재 CV 계산 중 |
| **당일 결과** | 하루 | Hot Memory | 오늘 제출한 모델 |
| **현재 상태** | 며칠 | CLAUDE.md | Best 모델, 제출 현황 |
| **제출 기록** | 영구 | SUBMISSION_LOG.md | 전체 제출 이력 |
| **과거 실험** | 영구 | experiments/ | 3주 전 실험 |

---

## 🎯 최적 메모리 전략

### 원칙

1. **Hot Memory = 현재만**
   - 과거는 파일 참조
   - 휘발성 인정

2. **CLAUDE.md = 진입점**
   - 간결하게 (< 200 lines)
   - 링크 중심
   - 자주 업데이트

3. **SUBMISSION_LOG.md = SSOT**
   - 모든 제출 기록
   - 단일 진실 공급원
   - 다른 파일과 불일치 시 이게 정답

4. **파일 > 메모리**
   - 불일치 시 파일이 정답
   - Hot Memory 믿지 말기

### 예시: "Best 모델이 뭐야?" 질문 시

**Bad (Hot Memory 의존):**
```
"음... Zone 6x6이 Best였던 것 같은데..."
→ 틀림! (실제는 domain_features)
```

**Good (파일 참조):**
```
1. SUBMISSION_LOG.md 읽기
2. "Best Model: exp_007_domain_features (15.9508)"
3. 정확한 답변!
```

---

## 🔧 CLAUDE.md 템플릿

```markdown
# K-League 대회 - Claude 가이드

> **진입점:** 세션 시작 시 이 파일부터 읽으세요

**마지막 업데이트:** 2025-12-16

---

## 📌 TL;DR (3 Lines)

```
Best: domain_features (15.95) | 2등: ensemble (16.13)
현재: Week 3 관찰 모드 | 제출: 9/175 (5.1%)
금지: Zone 6x6 변경, CV < 16.0 추구
```

---

## 🔗 Quick Links

**필독:**
- [SUBMISSION_LOG.md](./SUBMISSION_LOG.md) - 제출 기록 (SSOT)
- [PIPELINE.md](./PIPELINE.md) - 워크플로우

**참고:**
- [README.md](./README.md) - 프로젝트 개요
- [대회 페이지](https://dacon.io/competitions/official/236647/)

---

## 🎯 현재 상태

| 항목 | 값 |
|------|-----|
| **Best Model** | exp_007_domain_features |
| **Best Public** | 15.9508 |
| **Safe Model** | exp_001_zone_6x6 |
| **Latest** | exp_015_ensemble (16.13, 2등) |
| **제출** | 9/175 (5.1%) |
| **전략** | Week 3 관찰 모드 |

---

## ⚠️ 금지 사항

```
❌ Zone 6x6 하이퍼파라미터 변경 (14회 탐색 완료)
❌ CV < 16.0 추구 (과최적화)
❌ 제출 기록을 여러 파일에 분산
❌ Best 모델 추측 (SUBMISSION_LOG.md 확인!)
```

---

## 📝 작업 시작 전

**제출 관련:**
→ SUBMISSION_LOG.md 읽기

**실험 관련:**
→ PIPELINE.md 읽기

**과거 실험 참조:**
→ experiments/exp_XXX/ 읽기

---

*상세 내용은 링크된 파일 참조*
```

---

## 📚 요약

**3줄 요약:**
```
1. Hot Memory = 현재 작업만, 파일 = 진실
2. CLAUDE.md = 진입점 (간결!), SUBMISSION_LOG.md = SSOT
3. 불일치 시 파일이 정답, Hot Memory 버리고 재로딩
```

**Golden Rule:**
```
파일에 기록되지 않은 것은 존재하지 않는다.
Hot Memory를 믿지 마라.
```

---

*마지막 업데이트: 2025-12-16*
