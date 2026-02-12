# Quick Reference Card - 2025-12-09

## 현재 상태 (한눈에)

```
┌─────────────────────────────────────────────────────────────┐
│                      CURRENT BEST                           │
│                   safe_fold13 모델                          │
├─────────────────────────────────────────────────────────────┤
│  Public:      16.3639  (13회 중 1위)                        │
│  Fold 1-3 CV: 16.3356  (Sweet Spot ✓)                      │
│  Gap:         +0.028   (역대 최소 ✓)                        │
│  상태:        지역 최적점 도달                               │
└─────────────────────────────────────────────────────────────┘

제출 현황: 13/175 (7.4%)
남은 제출: 162회
남은 기간: D-34 (35일)
```

---

## 의사결정 트리 (30초)

```
현재 모델 개선 가능한가?
│
├─ YES (확률 30-40%) ────→ Option B (Hybrid Zone)
│                         - 필드별 가변 Zone
│                         - CV 16.28-16.32 목표
│                         - 제출 8-12회
│
└─ NO/불확실 (확률 60%) ──→ Option D (현재 유지 + 후반전)
                          - Week 2: 1-2회/일
                          - Week 3: 리더보드 재평가
                          - Final: All-in

권장: 80% D + 20% B
```

---

## 제출 전 체크리스트

```
□ Fold 1-3 CV < 16.34?
□ Fold 분산 < 0.015?
□ min_samples ≥ 20?
□ CV Sweet Spot (16.30-16.34) 내?
□ 로컬에서 3-Fold 이상 검증?

5개 모두 YES → 제출 진행
하나라도 NO → 추가 튜닝 or 폐기
```

---

## 절대 금지 (DO NOT)

```
❌ CV < 16.27 추구 (과최적화!)
❌ min_samples < 20 (노이즈 학습!)
❌ Fold 4-5 성능 신뢰
❌ 복잡한 ML 모델 (LightGBM 등) 경솔히 제출
❌ 6개 이상 앙상블
❌ CV만 보고 판단 (Gap도 중요!)
```

---

## 허용 (DO)

```
✓ CV 16.30-16.34 유지
✓ min_samples 20-30
✓ Fold 1-3 기준 평가
✓ Inverse Variance Weighting
✓ 계층적 Fallback
✓ 단순한 모델 우선
✓ 로컬 검증 철저히
```

---

## 오늘 할 일 (12/09)

```
오전:
1. [ ] Hybrid Zone 데이터 분석
2. [ ] 필드 영역별 분포 확인

오후:
3. [ ] Hybrid Zone 모델 구현
4. [ ] 로컬 CV < 16.32 달성 시 → 제출

저녁:
5. [ ] Bayesian 이론 검토
6. [ ] ML 로컬 테스트 (제출 X)
```

---

## 이번 주 목표 (12/09-12/15)

```
제출: 7-14회 (1-2회/일)
목표: Public 16.30-16.35
시도:
  1. Hybrid Zone (50%)
  2. Bayesian Statistics (30%)
  3. Ensemble of Statistics (20%)
```

---

## 긴급 연락처 (문서)

```
상세 분석:    /docs/STRATEGIC_DECISION_ANALYSIS_2025_12_09.md
요약:         /docs/EXECUTIVE_SUMMARY_2025_12_09.md
CV 분석:      /docs/CV_SWEET_SPOT_DISCOVERY.md
Fold 분석:    /code/analysis/analyze_fold_45_pattern.py
Best 모델:    /code/models/model_safe_fold13.py
```

---

## 성공 확률 요약

```
Tier 1 (상금권):     < 5%   (Public < 16.00)
Tier 2 (목표):      25-35%  (Public 16.20-16.30)
Tier 3 (양호):      50-60%  (Public 16.30-16.36)
Tier 4 (현상 유지): 70-80%  (Public 16.36-16.45)
```

---

## 핵심 메시지

**"조급해하지 말라. 현재 모델은 이미 우수하다. 체계적으로 후반전을 준비하라."**

---

*업데이트: 2025-12-09*
*다음 업데이트: 매일 (필요시)*
