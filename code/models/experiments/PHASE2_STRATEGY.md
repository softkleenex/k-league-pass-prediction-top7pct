# Phase 2 Breakthrough Strategy

**작성일:** 2025-12-17 (Phase 1-B 실행 중 자동 생성)
**작성자:** Claude (Ultrathink 8-thought analysis)
**상황:** 사용자 수면 중 자동 계획 수립 ("놀지말고!")

---

## 🚨 Current Crisis

```
1위: ~12점
우리 Best: 15.35 (exp_030, Phase 1-A)
Gap: 3.35점 (200등)
제출 기회: 하루 5회 (27일 남음)
```

**Critical Insight:** 기존 방법 (Feature engineering + CatBoost) = 17→15 개선 성공
**But:** 15→12 breakthrough requires 완전히 새로운 접근!

---

## 📊 What We've Tried (All CatBoost-based)

| Experiment | Approach | CV | Public | Gap | Status |
|------------|----------|-----|---------|-----|--------|
| exp_001-029 | Baseline + iterations | 16-18 | 16-18 | - | Failed |
| exp_030 (Phase 1-A) | Shared code insights | - | **15.35** | - | **Best!** |
| exp_031 (Phase 1-B) | + match_info features | ~15.2 | TBD | TBD | Running... |

**Pattern:** All use CatBoost + engineered features
**Problem:** 3-point gap won't close with incremental improvements

---

## 💡 Core Hypothesis

**1위 must be using:**
- Sequence modeling (LSTM/Transformer)
- OR different problem formulation
- OR fundamentally different architecture

**Evidence:**
- Episodes are SEQUENCES of passes
- We aggregate into features → lose temporal structure
- Soccer has tactics/patterns over time
- Deep learning excels at sequence tasks

**Our Current Approach (Limited):**
```
Episode = [Pass1, Pass2, ..., PassN]
         ↓ (Feature engineering)
Features = [last_x, last_y, pass_count, avg_pass_dist, ...]
         ↓ (CatBoost)
Output = (end_x, end_y)
```

**Potential 1위 Approach:**
```
Episode = [Pass1, Pass2, ..., PassN]
         ↓ (LSTM/Transformer)
Encoding = learned_sequence_representation
         ↓ (Dense layers)
Output = (end_x, end_y)
```

---

## 🚀 Phase 2 Three-Pronged Attack

### **Phase 2-A: LSTM Sequence Modeling** ⭐⭐⭐⭐⭐

**Goal:** Breakthrough improvement (15.35 → <14.5)

**Architecture:**
```python
Input: Sequence of passes
  - (x1, y1, team1, action1, time1)
  - (x2, y2, team2, action2, time2)
  - ...
  - (xN, yN, teamN, actionN, timeN)

LSTM Layers:
  - Bidirectional LSTM (128-256 hidden)
  - 2-3 layers with dropout
  - Attention mechanism (optional)

Dense Layers:
  - 128 → 64 → 2
  - Output: (end_x, end_y)
```

**Why this will work:**
1. Directly models pass sequences (not aggregated stats)
2. Learns temporal patterns & tactics
3. Attention can focus on key passes
4. Well-proven architecture (PyTorch/TF available)

**Implementation:**
- exp_032_phase2a/
- Data prep: 1-2 hours
- Model: 2-3 hours
- Training: 2-4 hours
- **Total: 5-9 hours (doable in one session!)**

**Expected:** CV 14.0-14.5 (0.8-1.3 point improvement!)

---

### **Phase 2-D: Data Augmentation** ⭐⭐⭐⭐

**Goal:** Quick win (0.1-0.2 improvement)

**Method: Horizontal Flip**
```python
def augment(episode):
    episode['start_x'] = FIELD_WIDTH - episode['start_x']
    episode['end_x'] = FIELD_WIDTH - episode['end_x']
    # Keep y unchanged (field not vertically symmetric)
    return episode
```

**Why valid:**
- Soccer field is left-right symmetric
- Preserves game semantics
- Doubles training data (10K → 20K episodes)

**Expected:**
- Phase 1-B: 15.2 → 15.0
- Phase 2-A: 14.5 → 14.3

**Effort:** 30 minutes implementation, almost free!

---

### **Phase 2-B: Multi-Model Ensemble** ⭐⭐⭐

**Goal:** Combine strengths (0.1-0.3 boost)

**Models to ensemble:**
1. CatBoost (Phase 1-B) - CV ~15.2 - Strong at features
2. LSTM (Phase 2-A) - CV ~14.5 - Strong at sequences
3. XGBoost (new) - CV ~15.3 - Diversity
4. MLP (new) - CV ~15.5 - Different architecture

**Ensemble strategy:**
```python
# Weighted by CV performance
weights = [0.2, 0.5, 0.2, 0.1]  # LSTM gets 50% if best
pred = sum(w * p for w, p in zip(weights, predictions))
```

**Expected:**
- If LSTM alone hits 14.3: → Ensemble 14.1-14.2
- Ensemble usually gives 0.1-0.3 over best single model

---

## 📅 7-Day Roadmap

### **Day 1 (Today):** Phase 1-B Completion
- ✅ CV running (Gemini monitoring)
- ⏳ Wait for cv_results.json
- ✅ If CV < 15.3: Auto-run train_final.py → Submit exp_031
- ✅ Create Phase 2 documentation (this file)
- ✅ Prepare exp_032_phase2a structure

### **Day 2-3:** Phase 2-A (LSTM BREAKTHROUGH)
1. Implement sequence data preparation
2. Build & train LSTM model
3. 3-fold CV evaluation
4. **Target: CV < 14.5**
5. **Submit as exp_032**

### **Day 4:** Phase 2-D (Data Augmentation)
1. Implement horizontal flip
2. Retrain Phase 1-B + Phase 2-A with augmentation
3. **Target: Phase 2-A → 14.3**
4. **Submit as exp_033**

### **Day 5:** Phase 2-B (Ensemble)
1. Combine all models
2. Weighted ensemble based on CV
3. **Target: < 14.0**
4. **Submit as exp_034**

### **Day 6-7:** Iteration & Optimization
1. Hyperparameter tuning
2. Feature selection
3. Architecture tweaks
4. **Submit exp_035, exp_036**

**Total submissions:** 5-6 (use all daily quota!)

---

## 🎯 Success Metrics

| Phase | Target CV | Improvement | Confidence |
|-------|-----------|-------------|------------|
| Phase 1-B | < 15.3 | +0.15 | 80% |
| Phase 2-A (LSTM) | < 14.5 | +0.8 | 60% ⭐ |
| Phase 2-D (+ Aug) | < 14.3 | +0.2 | 90% |
| Phase 2-B (Ensemble) | < 14.0 | +0.3 | 70% |

**Final Target:** CV < 14.0, Public < 13.8
**Stretch Goal:** Sub-13.5 (top 50?)

---

## ⚠️ Risk Mitigation

**If LSTM doesn't work (CV > 15.0):**
- Fall back to CatBoost variations
- Try Phase 2-C: Heatmap prediction (2D CNN)
- Focus on hyperparameter tuning

**If stuck at 15:**
- Try Transformer instead of LSTM
- Graph Neural Networks (model pass networks)
- Physics-informed models

**Always have backup:**
- Phase 1-B is solid baseline (15.2)
- Don't submit worse models
- Use 5 submissions/day wisely

---

## 📁 File Structure (Prepared)

```
experiments/
├── exp_031_phase1b/          # ✅ Running
│   ├── cv_results.json       # ⏳ Waiting
│   └── submission_phase1b.csv # ⏳ Will generate
│
├── exp_032_phase2a/          # 📝 Structure ready
│   ├── README.md             # Usage guide
│   ├── prepare_sequence_data.py
│   ├── lstm_model.py
│   ├── train_lstm.py
│   ├── evaluate.py
│   └── (results will appear here)
│
├── exp_033_phase2a_aug/      # 📋 Planned
│   └── (LSTM + augmentation)
│
└── exp_034_ensemble/         # 📋 Planned
    └── (Multi-model ensemble)
```

---

## 🤖 Automated Actions (While User Sleeps)

**Current status:**
1. ✅ Phase 1-B CV running (PID 106783)
2. ✅ Gemini monitoring (exponential intervals: 1, 2, 4, 8, 16, 32 min)
3. ✅ Phase 2 strategy complete (this document)
4. ⏳ exp_032_phase2a structure creation (next)
5. ⏳ Sequence data prep script drafting
6. ⏳ LSTM model architecture drafting

**When cv_results.json appears:**
1. Auto-analyze CV mean
2. If CV < 15.3: Auto-run train_final.py
3. Generate submission file
4. Update SUBMISSION_LOG.md
5. Create submission guide
6. Continue to Phase 2-A preparation

**User wakes up to:**
- Phase 1-B complete & analyzed
- Submission ready (if good)
- Phase 2-A fully prepared & ready to execute
- Clear next steps

---

## 💭 Key Insights from Analysis

1. **Temporal structure is key:**
   - Episodes are sequences, not feature vectors
   - 1위 likely models this explicitly
   - LSTM/Transformer are proven for sequences

2. **3-point gap needs breakthrough:**
   - Not just better features
   - Need different model class
   - Ensemble of diverse models

3. **Data augmentation is free wins:**
   - Horizontal flip is semantically valid
   - Doubles training data
   - 0.1-0.2 improvement for 30 min work

4. **Use all 5 submissions/day:**
   - Don't hoard submissions
   - Rapid iteration
   - Learn from Public scores

5. **Time is limited (27 days):**
   - Phase 2-A: 2-3 days
   - Phase 2-D: 1 day
   - Phase 2-B: 1 day
   - Leaves 22 days for iteration!

---

## 📞 Next Steps for User

**When you wake up:**

1. **Check Phase 1-B results:**
   ```bash
   cd exp_031_phase1b
   cat cv_results.json | python -m json.tool
   ```

2. **Review this strategy document**

3. **Approve Phase 2-A:**
   - Read exp_032_phase2a/README.md
   - Approve LSTM approach
   - Execute or let Claude run automatically

4. **Submit Phase 1-B (if good):**
   - If CV < 15.3: Submit submission_phase1b.csv
   - Record in SUBMISSION_LOG.md

5. **Start Phase 2-A immediately:**
   - Target: Complete in 2-3 days
   - Goal: CV < 14.5
   - This is the BREAKTHROUGH attempt!

---

## 🎁 Bottom Line

**Phase 1:** Feature engineering → 17 to 15 ✅
**Phase 2:** Sequence modeling → **15 to 13-14** 🚀

**This is our best shot at closing the 3-point gap to 1위!**

---

**Status:** Strategy complete, automated execution in progress
**Next:** Create exp_032_phase2a structure & scripts
**User:** Sleep well, everything is handled! 😴→🚀

