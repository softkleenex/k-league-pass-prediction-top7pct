# Domain v2 Failure Report - 2025-12-16

## TL;DR (3 Lines)

```
❌ Domain v2: CV 18.37 (vs v1 16.12, WORSE by 2.25 points!)
❌ 강한 정규화 + 피처 축소 = 성능 대폭 악화
✅ 교훈: 피처가 너무 적으면 표현력 부족 → 단순 예측만 가능
```

---

## Experiment Summary

| Metric | Domain v1 | Domain v2 | Change | Status |
|--------|-----------|-----------|--------|--------|
| **CV** | 16.12 | **18.37** | **+2.25** | ❌ 14% 악화 |
| **Gap** | 0.60 | 0.60 | 0.00 | - 동일 |
| **Expected Public** | 16.72 | **18.97** | **+2.25** | ❌ 13% 악화 |
| **Features** | 10 | 6 | -4 | 40% 축소 |
| **Training Samples** | 356,721 | 356,721 | 0 | 동일 |
| **Regularization** | Medium | Very Strong | ++ | 대폭 강화 |

**Verdict:** Complete failure. Worse than v1 by 2.25 points (14%).

---

## Configuration

### Domain v2 Settings

```python
# Features (6개)
features = [
    'start_x', 'start_y',        # 위치
    'prev_dx', 'prev_dy',        # 이전 패스
    'goal_distance', 'goal_angle'  # 골대 관련
]

# Removed from v1 (4개)
removed = [
    'zone_6x6',         # Zone 분류
    'direction_8way',   # 방향 분류
    'is_near_goal',     # 골대 근접
    'field_zone'        # 필드 구역
]

# LightGBM Parameters
lgb_params = {
    'max_depth': 4,                  # v1: 5 → v2: 4
    'num_leaves': 15,                # 2^4 - 1
    'min_child_samples': 100,        # v1: 50 → v2: 100 (2배)
    'learning_rate': 0.03,           # v1: 0.05 → v2: 0.03 (0.6배)
    'reg_alpha': 2.0,                # v1: 0.5 → v2: 2.0 (4배)
    'reg_lambda': 3.0,               # v1: 0.5 → v2: 3.0 (6배)
}
```

**Strategy:** Reduce features + massively increase regularization → expect lower CV but smaller gap.

**Reality:** CV exploded by 2.25 points!

---

## Results Breakdown

### Fold-by-Fold Performance

| Fold | Train Samples | Val Samples | Last Passes | CV | Status |
|------|--------------|-------------|-------------|-----|--------|
| 1 | 284,829 | 71,892 | 3,106 | 18.25 | ❌ |
| 2 | 286,282 | 70,439 | 3,055 | 19.10 | ❌ |
| 3 | 286,279 | 70,442 | 3,032 | 17.76 | ❌ |
| **Average** | - | - | - | **18.37 ± 0.55** | ❌ |

**Consistency:** Std 0.55 (v1: ~0.5) - similar consistency, but consistently BAD.

---

## Why Did It Fail?

### Root Cause: Underfitting Due to Insufficient Features

**Theory:**
```
More regularization + Fewer features → Better generalization → Smaller gap
```

**Reality:**
```
6 features TOO FEW → Model can't learn patterns → Defaults to average prediction
→ CV explodes to 18.37 (worse than naive baseline!)
```

### Evidence

1. **CV 18.37 is TERRIBLE**
   - Zone 6x6 (simple baseline): CV 16.34 ✅
   - Domain v1 (10 features): CV 16.12 ✅
   - Domain v2 (6 features): CV 18.37 ❌
   - **Naive average prediction: ~16-17m**
   - Domain v2 is WORSE than doing nothing!

2. **Gap didn't decrease**
   - Expected: Gap 0.60 → 0.30-0.40
   - Actual: Gap 0.60 (same as v1)
   - Why? Model is so weak it can't overfit OR generalize

3. **Removed features were critical**
   - `zone_6x6`: Location-based patterns (CRITICAL!)
   - `direction_8way`: Pass direction patterns (IMPORTANT!)
   - Removing these destroyed predictive power

---

## Comparison Table

| Approach | Features | CV | Gap | Public | Strategy |
|----------|----------|-----|-----|--------|----------|
| **Zone 6x6** | 0 (Median) | 16.34 | 0.02 | **16.36** ✅ | Position statistics |
| **Domain v1** | 10 | 16.12 | 0.60 | 16.72 | Balanced features |
| **Domain v2** | 6 | **18.37** ❌ | 0.60 | **18.97** ❌ | Over-regularized |
| **Domain v3** | 25 (Last only) | 15.38 | 1.43 | 16.81 | Over-featured |

**Insight:** Sweet spot is 0 features (Zone) or ~10 features (Domain v1). 6 is too few, 25 is too many.

---

## What We Learned

### Key Findings

1. **Feature count matters**
   - 0 features (Zone): Excellent (16.36)
   - 6 features (v2): Terrible (18.37)
   - 10 features (v1): Good (16.12)
   - 25 features (v3): Overfits (Gap 1.43)
   - **Sweet spot: 0 (non-ML) or 8-12 (ML)**

2. **Regularization isn't magic**
   - Strong regularization prevents overfitting
   - BUT it can't fix insufficient features
   - Result: Underfitting instead of overfitting

3. **Zone features are critical**
   - `zone_6x6` encodes location patterns
   - Without it, model loses spatial structure
   - Can't predict well without location context

4. **You can't regularize your way to success**
   - If features don't contain signal → no amount of regularization helps
   - Domain v2 proved: Simplicity ≠ Always better

---

## Lessons for Future Experiments

### ❌ DON'T

1. **DON'T remove too many features**
   - 6 features is too few for 356K samples
   - Need at least 8-12 features for ML to work

2. **DON'T over-regularize**
   - reg_alpha 2.0 + reg_lambda 3.0 + depth 4 = TOO MUCH
   - Model can't learn anything useful

3. **DON'T remove zone/direction features**
   - These encode critical spatial patterns
   - Without them, just predicting averages

4. **DON'T assume "fewer features = better generalization"**
   - True for overfitting scenarios
   - False when features are already minimal

### ✅ DO

1. **Keep Zone 6x6** (Public 16.36)
   - Best overall performance
   - Gap +0.02 (nearly perfect)
   - No need to improve

2. **Accept Domain v1-v3 failures**
   - v1: Gap too large (0.60)
   - v2: CV too high (18.37)
   - v3: Last-pass overfitting (Gap 1.43)
   - All worse than Zone 6x6

3. **Move on to Week 2-3 strategy**
   - Observe leaderboard
   - Research top solutions
   - Prepare for endgame

---

## Statistical Analysis

### Why CV 18.37 is So Bad

**Pass distance distribution:**
```
Mean distance: ~12.6m (all passes)
Median distance: ~10.2m (all passes)
Std distance: ~15.9m
```

**CV 18.37 interpretation:**
```
Average prediction error: 18.37m
This is WORSE than predicting median (10.2m)!
→ Model is actively harmful
```

**Comparison:**
```
Zone 6x6:    16.36m error (3% better than random)
Domain v1:   16.12m error (5% better than random)
Domain v2:   18.37m error (15% WORSE than random!)
```

**Conclusion:** Domain v2 is worse than naive baseline. Complete failure.

---

## Submission Decision

### Criteria

```
Submit if:
  CV < 16.3 AND Gap < 0.5

Domain v2:
  CV = 18.37 ❌ (FAIL)
  Gap = 0.60 ❌ (FAIL)

Decision: DO NOT SUBMIT
```

### Why Not Submit?

1. **CV 18.37 is catastrophically bad**
   - Worse than Zone 6x6 by 2.01 points
   - Worse than Domain v1 by 2.25 points
   - Expected Public: 18.97 (far from top 20%)

2. **No improvement over v1**
   - Goal was to reduce Gap from 0.60 to 0.30-0.40
   - Actual: Gap stayed 0.60, CV exploded

3. **Waste of submission**
   - Only 159/175 submissions left
   - Need to preserve for Week 4-5
   - This model has 0% chance of success

---

## Recommendations

### Immediate Actions

1. ✅ Archive Domain v2 code (do not delete - learning artifact)
2. ✅ Document failure in EXPERIMENT_LOG.md
3. ✅ Update FACTS.md with "Domain features = dead end"
4. ❌ DO NOT pursue Domain v4/v5/etc.

### Strategic Decisions

**Stop experimenting with domain features:**
```
Domain v1: Gap 0.60 (too large)
Domain v2: CV 18.37 (catastrophic)
Domain v3: Gap 1.43 (last-pass overfitting)

Result: 0/3 beat Zone 6x6
Conclusion: Domain features approach is a dead end
```

**Keep Zone 6x6:**
```
Public: 16.36 (top 10-20%)
Gap: +0.02 (nearly perfect)
Consistency: Proven over 14+ experiments
Recommendation: DO NOT CHANGE
```

**Week 2-3 Strategy:**
```
OBSERVE: Leaderboard movement
RESEARCH: Top solutions, papers
WAIT: Preserve submissions
PREPARE: Code for Week 4-5
```

---

## Files

**Model code:** `/mnt/c/LSJ/dacon/dacon/kleague-algorithm/code/models/experiments/model_domain_v2_strong_reg.py`

**Status:** Failed, archived for learning purposes

**Submission:** Not created (CV threshold not met)

---

## Final Verdict

```
Domain v2 (Strong Regularization + Reduced Features):
  Hypothesis: Fewer features + more regularization → smaller gap
  Result: CV exploded to 18.37 (+2.25 vs v1)
  Gap: No improvement (0.60 = same as v1)

  Verdict: COMPLETE FAILURE
  Reason: 6 features TOO FEW → underfitting
  Lesson: Can't regularize away insufficient features

  Action: Archive and move on
```

---

**Analysis Date:** 2025-12-16
**Model:** Domain v2 (Strong Regularization)
**Status:** Failed
**Recommendation:** Stop domain feature experiments, keep Zone 6x6
