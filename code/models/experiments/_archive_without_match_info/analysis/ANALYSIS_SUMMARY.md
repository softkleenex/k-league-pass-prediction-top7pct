# 분석 요약 (2025-12-21)

> 오늘 수행한 모든 분석과 발견 정리

---

## 핵심 발견 요약

| 분석 | 발견 | 개선 효과 |
|------|------|-----------|
| 에러 분석 | 수비/중앙 Zone에서 실패 | 패턴 파악 |
| Zone 후처리 | 효과 없음 | - |
| Train/Test 분포 | 매우 유사 (KL < 0.01) | 일반화 문제 없음 |
| **피처 중요도** | **Top 10으로 충분** | **+0.23점** |
| 예측 분포 | 분산 부족 | +0.07점 |

---

## 1. 에러 분석

**파일:** `error_analysis.py`

### 고에러 특성
- zone_x 낮음 (1.80 vs 3.01) → **-40% 차이**
- goal_distance 높음 (67.36 vs 50.28) → **+34% 차이**
- start_x 낮음 (40.24 vs 61.51) → **-35% 차이**

### Zone별 에러
```
높은 에러: Zone (0,2), (0,3), (1,2) → 수비/중앙
낮은 에러: Zone (5,x) → 골 근처
```

### 시간대별 에러
```
높음: 0-15분, 90+분
낮음: 60-90분
```

---

## 2. Train vs Test 분포

**파일:** `train_test_distribution.py`

### 주요 피처 비교
| 피처 | Train | Test | 차이 |
|------|-------|------|------|
| start_x | 54.92 | 54.10 | -1.5% |
| start_y | 33.60 | 33.96 | +1.1% |
| episode_length | 23.11 | 22.00 | -4.8% |

### 결론
- **분포 매우 유사** (KL divergence < 0.01)
- 일반화 문제 없음
- Zone 분포도 일치

---

## 3. 피처 중요도 분석

**파일:** `feature_importance.py`

### Top 10 피처 (Combined Importance)
1. goal_distance (20.26)
2. zone_y (13.35)
3. goal_angle (11.60)
4. prev_dx (8.40)
5. prev_dy (7.75)
6. zone_x (6.80)
7. final_poss_len (6.28)
8. direction (5.11)
9. result_encoded (3.80)
10. team_possession_pct (3.71)

### 쓸모없는 피처
- type_encoded (0.00)
- is_final_team (0.00)

### X/Y별 차이
- **X 예측:** goal_distance, zone_x 중요
- **Y 예측:** zone_y, goal_angle 중요

---

## 4. 피처 선택 실험

**파일:** `feature_selection_experiment.py`

### 결과
| 피처 수 | CV |
|---------|-----|
| 5개 | 15.93 |
| **10개** | **15.26** |
| 15개 | 15.38 |
| 19개 (전체) | 15.41 |

### 결론
- **Top 10 피처가 최적!**
- 전체 대비 **+0.23점 개선**
- 생성된 파일: `submission_top10_cv15.2569.csv`

---

## 5. 예측값 분포 분석

**파일:** `prediction_distribution.py`

### 분산 비교
| | 실제값 | 예측값 | 차이 |
|--|--------|--------|------|
| X std | 23.85 | 20.22 | -15% |
| Y std | 24.35 | 20.37 | -16% |

### 편향
- X 편향: +0.02 (거의 없음)
- Y 편향: -0.03 (거의 없음)

### 영역별 에러
- X=0~20: 에러 **20.93** (최악)
- X=80~105: 에러 **14.90** (최고)

---

## 6. 분산 스케일링 실험

**파일:** `variance_scaling_experiment.py`

### 결과
| Alpha | CV |
|-------|-----|
| 0.00 (baseline) | 15.49 |
| **0.25** | **15.42** |
| 0.50 | 15.46 |
| 0.75 | 15.62 |
| 1.00 | 15.86 |

### 결론
- alpha=0.25에서 **+0.07점 개선**
- 효과는 있지만 피처 선택보다 작음

---

## 최종 권장 사항

### 내일(12/22) 제출 우선순위

| 순위 | 파일 | CV | 이유 |
|------|------|-----|------|
| **1** | `submission_top10_cv15.2569.csv` | 15.26 | **피처 선택 효과!** |
| 2 | `weighted_57_43.csv` | - | 가중치 미세 조정 |
| 3 | `weighted_60_40.csv` | - | Phase1A 비중 증가 |
| 4 | `tuned_cv15.4271.csv` | 15.43 | lr=0.03 튜닝 |
| 5 | `3model_55_35_10.csv` | - | 3모델 다양성 |

### 추가 실험 아이디어
1. Top 10 피처 + 분산 스케일링 조합
2. Top 10 피처로 앙상블 재구성
3. X/Y별 다른 피처셋 사용

---

## 파일 목록

```
analysis/
├── error_analysis.py
├── error_analysis_results.csv
├── train_test_distribution.py
├── train_test_distribution_results.csv
├── feature_importance.py
├── feature_importance_results.csv
├── feature_selection_experiment.py
├── feature_selection_results.json
├── prediction_distribution.py
├── variance_scaling_experiment.py
├── model_x_top10.pkl
├── model_y_top10.pkl
├── submission_top10_cv15.2569.csv
└── ANALYSIS_SUMMARY.md (이 파일)
```

---

*분석 완료: 2025-12-21*
