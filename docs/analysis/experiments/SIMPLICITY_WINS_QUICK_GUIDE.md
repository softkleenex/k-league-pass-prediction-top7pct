# 단순함의 강점 - 빠른 가이드 (1분)

> **핵심:** Gap = 일반화 능력, Zone 6x6 = Gap 0.02 (완벽)
> **날짜:** 2025-12-16

---

## 1초 결론

```
단순함 > 복잡함
Zone 6x6 (4 features, Gap 0.02) > Domain (32 features, Gap 1.14) > LSTM (70 features, Gap 2.93)
```

---

## 핵심 수치 (10초)

| 지표 | Zone 6x6 | Domain v1 | LSTM v3 |
|------|----------|-----------|---------|
| **Gap** | **0.02** (1st) | 1.14 (5th) | 2.93 (7th) |
| **Gap 비율** | **0.1%** | 7.7% | 20.4% |
| **일반화** | **99.88%** | 92.30% | 79.60% |
| **Fold Std** | **0.006** | 0.29 (49배) | 0.15 (25배) |
| **Features** | **4** | 32 (8배) | 70 (17배) |
| **효율성** | **27.23** | 1.43 | 0.86 |

**결론:** Zone 6x6 = 모든 지표에서 압도적 1위!

---

## 왜 단순함이 이기나? (20초)

### 1. Bias-Variance Trade-off
```
단순 (Zone 6x6):
- High Bias (단순한 median)
- Low Variance (std 0.006) ⭐
→ 안정적!

복잡 (Domain/LSTM):
- Low Bias (복잡한 학습)
- High Variance (std 0.15-0.29) ❌
→ 불안정!
```

### 2. Overfitting 회피
```
Zone 6x6:  4 features → 노이즈 무시 → Gap 0.02
Domain v1: 32 features → 노이즈 학습 → Gap 1.14
LSTM v3:   70 features → 노이즈 과학습 → Gap 2.93
```

### 3. 게임 독립성
```
Zone 6x6:  위치 통계 → 게임 독립 → 99.88% 일반화
Domain v1: Player/Team → 게임 종속 → 92.30% 일반화
LSTM v3:   시퀀스 패턴 → 시간 종속 → 79.60% 일반화
```

---

## 증거 (30초)

### Feature 수 vs Gap (상관관계)
```
Pearson r = +0.67 (강한 양의 상관)
→ 복잡할수록 Gap 증가!

 4 features → Gap 0.02
10 features → Gap 0.74
32 features → Gap 1.14
70 features → Gap 2.93
```

### Zone 해상도 (U-Shaped)
```
5x5: Gap 1.14 (과적합)
6x6: Gap 0.02 (최적) ⭐
7x7: Gap 0.80 (과소적합)

→ 6x6 = U-Shaped 최저점!
```

### min_samples (통계학)
```
15: Gap 증가 (과적합)
25: Gap 0.02 (최적) ⭐
30: Gap 증가 (과소적합)

→ 25 = 중심극한정리 임계값!
```

### Direction 각도 (도메인 지식)
```
40°: CV 16.52 (과세분)
45°: CV 16.34 (최적) ⭐
50°: CV 16.49 (과단순)

→ 45° (8방향) = 축구 전술 일치!
```

---

## 실패 사례 (Domain/LSTM) (30초)

### Domain v1 실패 원인
```
1. Target Encoding (7개 피처) → Gap +0.4
2. All passes 학습 → Gap +0.5
3. 복잡한 피처 (32개) → Gap +0.2

총 Gap: 1.14
```

### LSTM v3 실패 원인
```
1. 잘못된 추상화 (시퀀스 ≠ 위치) → Gap +2.0
2. Bidirectional (cheating) → Gap +0.5
3. 과적합 (838K params) → Gap +0.4

총 Gap: 2.93
```

### 공통점
```
복잡도 ↑ → Validation fold 암기 → Public fold 실패
→ Gap 폭발!
```

---

## 의사결정 플로우차트

```
새로운 아이디어가 생겼다!
         │
         ▼
    Feature 수는?
         │
    ┌────┴────┐
    │         │
  ≤ 10      > 10
    │         │
    ▼         ▼
  진행     "단순화 가능?" → Yes → 단순화 후 진행
              │
              No
              │
              ▼
         "CV < 16.27?" → Yes → 과적합 위험! 중단
              │
              No
              │
              ▼
         "Gap 예상 < 0.5?" → Yes → 신중히 진행
              │
              No
              │
              ▼
           중단 (Gap 폭발 예상)
```

---

## 체크리스트

### 새 모델 제출 전 확인
```
□ Feature 수 ≤ 10개? (많을수록 위험)
□ CV ≥ 16.27? (낮으면 과적합)
□ Fold std < 0.1? (높으면 불안정)
□ Gap 예상 < 0.5? (높으면 일반화 실패)
□ 게임 독립적? (종속적이면 위험)
□ Train = Test 분포? (불일치하면 실패)
□ Target Encoding 없음? (있으면 과적합)

→ 모두 ✓ → 제출 고려
→ 하나라도 ✗ → 재검토
```

---

## 금칙 사항

```
❌ Feature 수 무조건 늘리기
   → 복잡할수록 Gap 증가 (r=+0.67)

❌ CV만 최적화
   → Gap 무시하면 Public 실패

❌ Target Encoding 사용
   → Player/Team 통계는 과적합 주범

❌ All passes 학습 + Last pass 평가
   → Train-Test Mismatch → Gap 폭발

❌ LSTM/Transformer 시도
   → 시퀀스 추상화는 문제 오해

❌ 데이터 증강 (Flip, Rotation)
   → 축구 비대칭성 위반

❌ Bidirectional LSTM
   → Test time cheating (미래 정보 사용)
```

---

## 권장 사항

```
✅ 단순함 추구
   → 4-10 features가 최적

✅ Gap 최소화 집중
   → CV보다 Gap이 중요

✅ Fold 일관성 확인
   → Std < 0.1 목표

✅ 게임 독립적 피처
   → 위치, 필드 구역, 골대 거리

✅ 도메인 지식 활용
   → 8방향, 필드 구역, 전술적 위치

✅ 통계적 엄밀성
   → min_samples ≥ 25

✅ Train = Test 분포
   → Last pass only 학습
```

---

## 수치로 보는 Zone 6x6 우수성

### 안정성 (Gap)
```
Zone 6x6: 0.02 (기준)
평균:     2.16 (108배 높음!)
LSTM v2:  6.90 (345배 높음!)

→ Zone 6x6 = 압도적 안정성!
```

### 일관성 (Fold Std)
```
Zone 6x6: 0.006 (기준)
Domain:   0.29 (49배 높음!)
LSTM v3:  0.15 (25배 높음!)

→ Zone 6x6 = 극도로 일관적!
```

### 효율성 (Public/log(Complexity))
```
Zone 6x6: 27.23 (1st)
XGBoost:   3.65 (2nd, 7.5배 낮음)
Domain:    1.43 (3rd, 19배 낮음)
LSTM v3:   0.86 (4th, 32배 낮음)

→ Zone 6x6 = 가장 효율적!
```

---

## 핵심 공식

### Gap = 일반화 능력
```
Gap = Public - CV

Gap → 0: 완벽한 일반화 (Zone 6x6: 0.02)
Gap > 1: 과적합 (Domain/LSTM: 1.14-6.90)
```

### Generalization Score
```
GS = (1 - Gap/CV) × (1 - Std/CV) × (1 - Range/CV)

Zone 6x6: 0.9978 (99.78%)
Domain:   0.8698 (86.98%)
LSTM v3:  0.7716 (77.16%)

→ Zone 6x6 = 거의 완벽!
```

### Occam's Razor
```
단순한 모델 > 복잡한 모델 (일반화)

증명:
- Feature 4개 vs 32개 vs 70개
- Gap 0.02 vs 1.14 vs 2.93
- r = +0.67 (강한 양의 상관)
```

---

## 1줄 요약

```
"Zone 6x6 (4 features, Gap 0.02) = 단순하지만 완벽한 일반화.
 Domain/LSTM (32-70 features, Gap 1.14-2.93) = 복잡하지만 과적합.
 → 단순함이 이긴다. Occam's Razor가 옳다."
```

---

## 다음 행동

### 즉시 (오늘)
```
1. 새로운 아이디어 → 체크리스트 확인
2. Feature 수 확인 → ≤ 10개 목표
3. Gap 예상 계산 → < 0.5 확인
```

### Week 3-4
```
1. Domain features 정제 (Target Encoding 제거)
2. Zone + Domain 앙상블
3. Gap < 0.3 목표
```

### Week 4-5
```
1. 검증된 접근만 시도
2. Public 성능 기반 미세 조정
3. 순위 100위 이내 목표
```

---

## 참고 자료

**빠른 요약 (2분):**
- [ZONE_6x6_EXECUTIVE_SUMMARY.md](./ZONE_6x6_EXECUTIVE_SUMMARY.md)

**상세 증명 (20분):**
- [ZONE_6x6_STABILITY_PROOF.md](./ZONE_6x6_STABILITY_PROOF.md)

**관련 문서:**
- [CLAUDE.md](../../CLAUDE.md) - 빠른 가이드
- [FACTS.md](../core/FACTS.md) - 확정 사실
- [EXPERIMENT_LOG.md](../core/EXPERIMENT_LOG.md) - 35회 실험

---

*이 가이드는 1분 안에 의사결정을 도와줍니다.*

*"단순함이 이긴다"를 항상 기억하세요!*

*작성: 2025-12-16*
*저자: Data Analyst Agent*
