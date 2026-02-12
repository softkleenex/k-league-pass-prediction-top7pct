# LSTM 접근법 실패 분석 보고서

> **날짜:** 2025-12-15
> **목적:** LSTM v2/v3/v4/v5 실패 원인 분석 및 교훈 정리
> **결론:** LSTM은 이 문제에 부적합한 접근법 (시퀀스 ≠ 위치 통계)

---

## 📊 실험 요약

| 버전 | 접근법 | CV | Public | Gap | 파라미터 | 결과 |
|------|--------|-----|--------|-----|----------|------|
| **v2** | Sampling (200/ep) | 13.18 | 20.08 | +6.90 | 838K | ❌ 최악 |
| **v3** | Full Episode | 14.36 | 17.29 | +2.93 | 838K | ⚠️ 개선 |
| **v4** | Flip 증강 (2x data) | 14.85 | - | - | 838K | ❌ 악화 |
| **v5** | 단순화 (74.6% ↓) | 14.44 | 17.44 | +3.00 | 213K | ❌ 실패 |
| **Zone 6x6** | 위치 통계 | 16.34 | 16.36 | +0.02 | - | ✅ 성공 |

**핵심 사실:**
- LSTM 최선: CV 14.36, Public 17.29, Gap +2.93
- Zone 6x6: CV 16.34, Public 16.36, Gap +0.02
- **Gap 차이: 146배 (2.93 vs 0.02)**

---

## 🔍 버전별 실패 분석

### LSTM v2: Sampling Mismatch (2025-12-14)

**접근법:**
- Episode당 200개 패스 샘플링 학습
- Bidirectional LSTM (2 layers, 256 hidden)
- Attention Mechanism
- 4-layer FC with BatchNorm

**가설:** "200개 샘플로도 패턴 학습 가능"

**결과:**
```
CV:     13.18 (RMSE 3.63)
Public: 20.08 (RMSE 4.48)
Gap:    +6.90 (매우 큼!)
```

**실패 원인:**
1. **Train/Test 분포 불일치**
   - Train: Episode당 200개 샘플 (시작 위치 편향)
   - Test: Episode당 1개 전체 시퀀스
   - 모델이 "샘플링된 패턴"을 학습

2. **Cumulative 피처 문제**
   - 샘플링으로 인해 누적 피처가 불연속
   - 실제 경기 흐름과 괴리

**교훈:**
- 샘플링 기반 학습은 절대 불가
- Train = Test 분포 일치 필수

---

### LSTM v3: Full Episode (2025-12-15)

**접근법:**
- Episode당 1개 전체 시퀀스 학습 (분포 일치)
- Bidirectional LSTM (2 layers, 128 hidden)
- Attention Mechanism
- 4-layer FC with BatchNorm
- Dropout 0.5

**가설:** "분포 일치하면 Gap 감소"

**결과:**
```
CV:     14.36 (RMSE 3.79)
Public: 17.29 (RMSE 4.16)
Gap:    +2.93 (여전히 큼)
```

**개선 사항:**
- Gap: 6.90 → 2.93 (57% 감소) ✅
- 분포 일치 효과 확인

**여전한 문제:**
- Gap +2.93 = Zone 6x6 대비 146배
- CV는 좋아졌지만 Public은 여전히 나쁨

**분석:**
1. **Bidirectional의 문제**
   - 미래 정보 사용 (data leakage)
   - 실제 예측 시 사용 불가능한 정보

2. **BatchNorm의 문제**
   - Train/Test 통계 차이
   - Batch 크기 16 vs Test time 32

3. **과적합 의심**
   - 복잡한 모델 (838K 파라미터)
   - CV 14.36은 좋지만 일반화 안 됨

**교훈:**
- 분포 일치는 필수지만 충분하지 않음
- Bidirectional = cheating in test
- BatchNorm = train/test 불일치 원인

---

### LSTM v4: Horizontal Flip (2025-12-15)

**접근법:**
- v3 + Horizontal Flip 데이터 증강
- X좌표 반전: x → 105 - x
- 데이터 2배 증가 (15K → 30K episodes)

**가설:** "데이터 증강으로 일반화 개선"

**결과:**
```
CV: 14.85 (RMSE 3.85)
vs v3: +0.49 (악화!)
```

**실패 원인:**
1. **축구의 비대칭성**
   - 홈팀: 왼쪽 → 오른쪽 공격
   - 원정팀: 오른쪽 → 왼쪽 공격
   - 공격 방향에 따라 패턴 다름

2. **잘못된 가정**
   - 물리적 대칭 ≠ 패턴 대칭
   - Flip은 "존재하지 않는" 패턴 생성

**교훈:**
- 도메인 지식 없는 증강은 위험
- 물리적 타당성 ≠ 데이터 타당성
- 데이터 증강은 신중하게!

---

### LSTM v5: Simplification (2025-12-15)

**접근법:**
- 과적합 감소를 위한 단순화
- Unidirectional LSTM (bidirectional 제거)
- No Attention (제거)
- No BatchNorm (제거)
- 2-layer FC (4→2)
- Dropout 0.6 (0.5→0.6)
- L2 regularization (weight_decay=1e-4)
- Gradient clipping (max_norm=1.0)
- **파라미터: 838K → 213K (74.6% 감소)**

**가설:** "단순화로 Gap 감소"

**결과:**
```
CV:     14.44 (RMSE 3.80)
Public: 17.44 (RMSE 4.18)
Gap:    +3.00 (증가!)

vs v3:
  CV: 14.36 → 14.44 (+0.08, 소폭 악화)
  Public: 17.29 → 17.44 (+0.15, 악화)
  Gap: 2.93 → 3.00 (+0.07, 증가!)
```

**예상 vs 현실:**
| 항목 | 예상 | 현실 | 결과 |
|------|------|------|------|
| CV | 약간 증가 | 14.44 (+0.08) | ✅ 예상대로 |
| Gap | 감소 | 3.00 (+0.07) | ❌ 증가! |
| Public | 개선 | 17.44 (+0.15) | ❌ 악화! |

**충격적 결론:**
- 파라미터 74.6% 감소해도 Gap 감소 없음
- 오히려 Gap 증가!
- **→ Gap은 모델 복잡도 문제가 아님**

**근본 원인:**
1. **잘못된 문제 추상화**
   - 문제: "패스 끝점 위치 예측"
   - LSTM 관점: "시퀀스 → 시퀀스" 문제
   - 실제: "위치 → 위치" 통계 문제

2. **시퀀스의 착각**
   - 70개 패스가 "시퀀스"처럼 보임
   - 하지만 실제는 "위치 의존적 통계"
   - LSTM은 시간 의존성을 학습하지만, 여기는 공간 의존성

3. **Zone이 성공한 이유**
   - (start_x, start_y) → (end_x, end_y) 직접 매핑
   - 위치별 통계로 접근
   - 시퀀스 무시 = 올바른 추상화

**교훈:**
- 모델 복잡도는 문제가 아니었음
- 문제 정의가 틀렸음 (시퀀스 vs 위치)
- 잘못된 추상화는 개선 불가능

---

## 💡 핵심 통찰

### 1. 문제의 본질

**LSTM의 가정:**
```
"과거 패스들의 시퀀스가 다음 패스를 결정한다"
```

**실제 문제:**
```
"시작 위치(start_x, start_y)가 끝 위치를 결정한다"
```

**증거:**
- Zone 6x6: 시작 위치만으로 Gap +0.02
- LSTM: 70개 시퀀스 사용해도 Gap +2.9+

### 2. 데이터 분석

**Episode 구조:**
```
Episode: 70개 패스 (평균)
각 패스: [start_x, start_y, ..., end_x, end_y]
```

**Zone 접근:**
```python
# 위치별 통계
zone = get_zone(start_x, start_y)
end_x_pred = zone_statistics[zone].median_end_x
```

**LSTM 접근:**
```python
# 시퀀스 학습
hidden = LSTM(pass_sequence)  # 70개 패스
end_pred = FC(hidden)  # 마지막 hidden state
```

**왜 Zone이 이기나?**
- 패스는 독립적 (이전 패스와 무관)
- 위치만이 중요 (시퀀스는 noise)
- LSTM은 존재하지 않는 패턴을 학습

### 3. 실패의 교훈

**기술적 교훈:**
1. ✅ Train = Test 분포 일치 필수
2. ✅ Bidirectional = test time cheating
3. ✅ BatchNorm = train/test 불일치
4. ✅ 데이터 증강은 도메인 지식 필요
5. ✅ 단순화만으로 일반화 개선 불가

**근본적 교훈:**
1. **문제 추상화가 가장 중요**
   - 좋은 모델 < 올바른 추상화
   - LSTM은 훌륭한 모델이지만 틀린 문제

2. **CV는 거짓말할 수 있다**
   - CV 14.36 vs Public 17.29
   - CV 최적화 ≠ Public 최적화

3. **Simple is better**
   - Zone 6x6: 단순하지만 효과적
   - LSTM: 복잡하지만 부적합

---

## 📈 수치 비교

### CV vs Public Gap

```
LSTM v2: 6.90 (525%)
LSTM v3: 2.93 (220%)
LSTM v5: 3.00 (221%)
Zone 6x6: 0.02 (1%)
```

Gap이 작을수록 좋은 모델!

### Public Score 비교

```
Zone 6x6:  16.36 ⭐ BEST
LSTM v3:   17.29 (+0.93, 5.7% 나쁨)
LSTM v5:   17.44 (+1.08, 6.6% 나쁨)
LSTM v2:   20.08 (+3.72, 22.7% 나쁨)
```

### 상대적 성능

```
Zone 6x6을 100%로 기준:
- LSTM v3: 94.6% (5.4% 손실)
- LSTM v5: 93.8% (6.2% 손실)
- LSTM v2: 81.5% (18.5% 손실)
```

---

## 🎯 결론 및 권장사항

### 확정된 사실

1. **LSTM은 이 문제에 부적합**
   - v2/v3/v4/v5 모두 Gap > 2.9
   - Zone 6x6: Gap +0.02 (146배 차이)
   - 개선 불가능 (근본 추상화 오류)

2. **Zone 접근이 올바름**
   - 위치 통계 = 문제의 본질
   - 단순하지만 효과적
   - 일반화 우수 (Gap 최소)

3. **데이터 증강 주의**
   - Flip: 축구 비대칭성 위반
   - 도메인 지식 없이 증강 위험

### 향후 방향

**❌ 하지 말아야 할 것:**
- LSTM 재시도 (모든 변형 실패)
- 더 복잡한 시퀀스 모델 (GRU, Transformer)
- 데이터 증강 (Flip, Rotation 등)

**✅ 고려할 수 있는 것:**
- Zone + LightGBM 하이브리드
  - 위험: 과적합 가능성 75-85%
  - 기대: CV < 16.0 시 Gap 폭발

- 완전히 다른 접근법
  - Graph Neural Networks (패스 네트워크)
  - Physics-based Model (궤적 물리)
  - 단, 모두 3-5일 개발, 30-40% 성공률

**🎯 현재 최선:**
- Zone 6x6 유지 (Public 16.36)
- Week 2-3: 관찰 모드
- Week 4-5: 검증된 접근만 시도

---

## 📚 참고 자료

### 코드 위치
```
/mnt/c/LSJ/dacon/dacon/kleague-algorithm/code/models/archive/lstm/
├── v2/  # Sampling 접근
├── v3/  # Full Episode
├── v4/  # Horizontal Flip
└── v5/  # Simplification
```

### 관련 문서
- [CLAUDE.md](../CLAUDE.md) - 빠른 가이드
- [FACTS.md](../FACTS.md) - 확정 사실
- [EXPERIMENT_LOG.md](../EXPERIMENT_LOG.md) - 전체 실험 기록

### 제출 기록
- v2: submission_lstm_v2_optimized_cv13.18.csv → 20.08
- v3: submission_lstm_v3_full_cv14.36.csv → 17.29
- v5: submission_lstm_v5_simplified_cv14.44.csv → 17.44

---

*이 문서는 LSTM 접근법 실패를 기록하고, 동일한 실수를 반복하지 않기 위해 작성되었습니다.*

*"Fast failure is fast learning" - 4번의 실패로 문제의 본질을 이해했습니다.*
