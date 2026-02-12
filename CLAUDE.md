# K-League 대회 - Quick Reference

> **컨텍스트 시작 시 이 파일부터 읽으세요**
> **원칙:** 파일 = 진실, 추측 금지

**마지막 업데이트:** 2026-01-06 20:52 (Iterative Pseudo-Labeling NEW BEST!)

---

## 🎯 Current State

```
Best LB: 13.4343 (exp_130 Iterative Pseudo) - NEW!
Best CV: 13.3608 (exp_128 Pseudo-Labeling)
전체 1위: ~12점 | 우리: 13.43 | 격차: ~1.43점
```

### 🔥 최신 실험 결과 (2026-01-06)
```
exp_130 Iterative Pseudo: LB 13.4343 (NEW BEST!)
exp_128 Pseudo-Labeling: CV 13.3608 → LB 13.4390
exp_131 Entity Embeddings: CV 13.52 (실패)
exp_132 Confidence-weighted: CV 13.43 (제출 대기)
```

### 핵심 교훈
```
[실패한 접근법 - 다시 시도 금지]
- Lagged Features (prev1, prev2, prev3): CV 좋아도 LB 폭발
- LSTM (시퀀스 모델): CV 16.79 (baseline보다 나쁨)
- MDN (Mixture Density Network): CV 14.23 (baseline보다 나쁨)

[검증된 접근법]
- Pseudo-Labeling: LB 13.44 ✓ (test 데이터로 재훈련)
- Delta Prediction + 11-fold + 7-seeds: LB 13.54 ✓
- CatBoost + MAE + EMA 피처
```

---

## 📂 Essential Files (SSOT)

| 파일 | 역할 | 언제 읽는가 |
|------|------|-------------|
| **[SUBMISSION_LOG.md](./SUBMISSION_LOG.md)** | 제출 기록 (단일 진실) | 제출 전 **필수!** |
| **[COMPETITION_ESSENTIALS.md](./COMPETITION_ESSENTIALS.md)** | 대회 핵심 정보 | 규칙/일정 확인 시 |
| **[PIPELINE.md](./PIPELINE.md)** | 워크플로우 | 실험/제출 방법 |
| **[GEMINI_COLLABORATION_PROTOCOL.md](./GEMINI_COLLABORATION_PROTOCOL.md)** | Gemini 협업 프로토콜 | 자동 실행 시 |
| **[MEMORY_STRATEGY.md](./MEMORY_STRATEGY.md)** | 메모리 관리 | 컨텍스트 혼란 시 |
| **[QUICK_START.md](./QUICK_START.md)** | 30초 요약 | 빠른 파악 |
| **[FAQ.md](./FAQ.md)** | 자주 묻는 질문 | 질문 있을 때 |

**Golden Rule:**
```
파일과 Hot Memory 불일치 → 파일이 정답 → Hot Memory 버리고 재로딩
```

---

## ⚠️ Critical Rules

```
✅ 필수: 매일 5회 제출 (안 쓰면 영구 소실!)
✅ 필수: SUBMISSION_LOG.md 확인 (Best 모델 추측 금지)
✅ 필수: 제출 기록은 SUBMISSION_LOG.md만 사용

❌ 금지: 같은 접근 반복 (기존 방법 = 15-17점대 = 실패)
❌ 금지: 제출 기회 낭비 (오늘 안 쓰면 영구 소실)
❌ 금지: Gap에만 집중 (절대 성능이 중요, 1위 = 12점)

⚠️ 재평가됨 (필요하면 시도):
   - Zone 수정 (기존 6x6이 최적이었지만 4점 격차 극복 필요)
   - CV < 16.0 (CV 10-11점 목표, Gap 커도 절대 성능 우선)
   - 새로운 시퀀스 모델 (LSTM은 실패했지만 Transformer 등 가능)
```

---

## 🔄 Workflow

### 제출 관련 작업

```
1. SUBMISSION_LOG.md 읽기 (필수!)
   → Best 모델 확인
   → 제출 이력 확인
   → 전체 순위 확인 (현재 ~200등)

2. 매일 5회 확인
   → 오늘 몇 회 남았는지 확인
   → 안 쓰면 영구 소실!

3. 새 제출 시
   → PIPELINE.md > "제출 워크플로우"
   → SUBMISSION_LOG.md 즉시 업데이트

4. 결과 분석
   → 절대 성능 (1위 12점과 비교)
   → Gap 계산 (참고용)
   → 새로운 접근법 효과 평가
```

### 🤖 제출 결과 자동 기록 프로토콜

**사용자가 제출 결과를 제공하면 자동으로 기록:**

```
입력 형식 (사용자):
"제출 결과:
 파일: submission_xxx.csv
 점수: 15.xxxx
 제출 시간: 2025-12-XX HH:MM"

자동 처리 (Claude):
1. SUBMISSION_LOG.md 읽기
2. 다음 exp_XXX 번호 자동 계산
3. 순위 재계산
4. Gap 계산 (CV 정보 있으면)
5. SUBMISSION_LOG.md 업데이트:
   - "현재 상태" 섹션 (Best 모델 갱신 시)
   - "제출 순위" 테이블
   - "제출 타임라인"
6. 결과 요약 출력

필수 업데이트 항목:
- 총 제출 수
- Best 모델 (갱신 시)
- 순위 테이블
- 제출 이력
```

### 실험 관련 작업

```
1. 완전히 새로운 접근 우선
   → 기존 방법 반복 금지
   → 1위(12점)는 뭔가 다른 걸 했음

2. PIPELINE.md 참조
   → 실험 워크플로우
   → 파일 명명 규칙

3. experiments/exp_XXX/ 생성
   → 코드, 결과, 분석
   → 매일 5회 실험 결과 기록
```

### 🤖 자동 모니터링 (Exponential Backoff)

**백그라운드 작업 시 사용자 입력 없이 자동 체크:**

```python
# 1. 백그라운드 실행
python script.py > log.txt 2>&1 &

# 2. Exponential backoff: 1분 → 2분 → 4분 → 8분
sleep 60  # 1분 대기
# 상태 확인 → 아직 실행 중이면
sleep 120  # 2분 대기
# 상태 확인 → 아직 실행 중이면
sleep 240  # 4분 대기
# ... 완료될 때까지 반복
```

**패턴 (이미지 /mnt/c/LSJ/dacon/dacon/1.png 참조):**
1. 작업 시작 → 1분 대기
2. "Still running... Waiting 2 minutes..."
3. Bash(sleep 120) 실행
4. "Still running... Waiting 4 minutes..."
5. 완료될 때까지 반복

**핵심:** 사용자 입력 기다리지 말고 자동으로 sleep → 체크 반복!

---

## 📊 Competition Info

| 항목 | 값 |
|------|-----|
| **종료** | 2026-01-12 10:00 (13일) |
| **제출** | **하루 5회** (누적 불가, 안 쓰면 소실!) |
| **평가** | Euclidean Distance (낮을수록 좋음) |
| **1위** | ~12점대 |
| **우리 CV** | **11.75** (exp_085 Lagged Features) 🔥 |
| **우리 LB** | 13.54 (exp_083) → 제출 후 업데이트 예정 |

**Details:** [COMPETITION_ESSENTIALS.md](./COMPETITION_ESSENTIALS.md)

**Links:**
- [제출](https://dacon.io/competitions/official/236647/mysubmission)
- [리더보드](https://dacon.io/competitions/official/236647/leaderboard)

---

## 🚀 Quick Actions

**새 세션 시작:**
```
1. CLAUDE.md 읽기 (이 파일)
2. SUBMISSION_LOG.md 읽기
3. 작업 결정
```

**Best 모델 확인:**
```
SUBMISSION_LOG.md > "Best 모델" 섹션
```

**제출 전:**
```
1. SUBMISSION_LOG.md 읽기
2. 금지 사항 확인
3. PIPELINE.md > 제출 워크플로우
```

**질문 있을 때:**
```
FAQ.md 확인
```

---

## 💡 핵심 메시지

```
"파일에 기록되지 않은 것은 존재하지 않는다."

Hot Memory를 믿지 마라.
SUBMISSION_LOG.md를 믿어라.
Best 모델을 추측하지 마라.
항상 확인하라.
```

---

**이 파일은 간결함을 유지합니다. 상세 내용은 링크된 파일 참조.**

**Old Version:** [docs/archive/CLAUDE_v1_FULL.md](./docs/archive/CLAUDE_v1_FULL.md)

**Update History:**
- 2025-12-30 22:35: exp_085 Lagged Features 실패 (CV 11.75 → LB 16.29, Gap +4.54점)
- 2025-12-30 20:15: Best LB 13.54 달성 (exp_083 11-Fold + 7-seeds) - 여전히 BEST
- 2025-12-29 23:00: Best CV 13.57 달성 (exp_082 10-Fold + 3-seeds), Combination experiments 완료
- 2025-12-29 22:15: Best CV 13.61 달성 (exp_078 10-Fold Delta), Overnight experiments 완료
- 2025-12-29: Best CV 13.72 달성 (exp_076 Delta Prediction), 핵심 발견 섹션 추가
- 2025-12-16 23:30 (긴급 개편): 위기 상황 반영, 제출 시스템 명확화, 금지 사항 재평가
- 2025-12-16 (Phase 3): Minimal CLAUDE.md (383줄 -> 150줄, 61% 감소)
