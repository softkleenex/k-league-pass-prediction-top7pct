# K-League 실험 기록

> **마지막 업데이트:** 2025-12-21
> **Best Score:** 15.3001 (weighted_55_45)

---

## 실험 요약

| Exp | 이름 | CV | Public | Gap | 설명 |
|-----|------|-----|--------|-----|------|
| 030 | phase1a | 15.45 | **15.35** | -0.10 | 21개 피처, CatBoost |
| 036 | last_pass_only | 15.49 | 15.38 | -0.11 | 마지막 pass만 학습 |
| 037 | ensemble_50_50 | - | 15.30 | - | Phase1A 50% + exp_036 50% |
| 037w | weighted_55_45 | - | **15.30** | - | **NEW BEST!** |
| 037w | weighted_52_48 | - | 15.30 | - | Phase1A 52% + exp_036 48% |
| 038 | xgboost | 15.93 | - | - | 다양성 확보용 |
| 041 | catboost_tuned | 15.43 | - | - | lr=0.03 최적 |
| 042 | new_features | 12.24 | - | - | **과적합 의심!** |

---

## 폴더 구조

```
experiments/
├── analysis/                    # 분석 스크립트
│   ├── error_analysis.py       # 에러 분석
│   ├── train_test_distribution.py
│   ├── feature_importance.py
│   └── feature_selection_experiment.py
│
├── exp_030_phase1a/            # Phase1A 기준 모델
│   ├── train_cv.py
│   ├── predict_submission.py
│   └── cv_results.json
│
├── exp_036_last_pass_only/     # 마지막 pass만 학습
│   ├── train_last_pass_only.py
│   └── submission_*.csv
│
├── exp_037_ensemble_*/         # 앙상블 실험
│   └── *.csv
│
├── exp_038_lightgbm_last_pass/ # LightGBM/XGBoost
│   ├── train_lightgbm_last_pass.py
│   └── train_xgboost_last_pass.py
│
├── exp_039_3model_ensemble/    # 3모델 앙상블
│   └── submission_3model_*.csv
│
├── exp_040_weight_optimization/# 가중치 최적화
│   └── submission_weighted_*.csv
│
├── exp_041_catboost_tuning/    # CatBoost 튜닝
│   ├── train_tuned_catboost.py
│   └── tuning_results.json
│
└── exp_042_new_features/       # 새 피처 실험
    └── train_with_new_features.py
```

---

## 핵심 발견

### 1. 분포 일치가 중요
- Train: 356K passes → 마지막 15K만 사용
- Test와 동일한 분포로 학습 → 일반화 향상

### 2. 앙상블이 효과적
- Phase1A (15.35) + exp_036 (15.38) = 15.30
- 최적 가중치: 55:45

### 3. 피처 중요도
```
Top 5:
1. goal_distance (20.26)
2. zone_y (13.35)
3. goal_angle (11.60)
4. prev_dx (8.40)
5. prev_dy (7.75)

쓸모없는 피처:
- type_encoded (0.00)
- is_final_team (0.00)
```

### 4. Train vs Test 분포
- 매우 유사 (KL divergence < 0.01)
- Zone 분포 일치
- 에피소드 길이 유사

---

## 실패한 접근법

| 접근법 | 결과 | 교훈 |
|--------|------|------|
| LSTM | Gap 3~7 | 시퀀스 모델 부적합 |
| 과도한 피처 | Gap 5+ | 과적합 유발 |
| Zone 후처리 | 효과 없음 | 모델이 이미 학습 |
| Domain v2/v3 | 악화 | v1이 최적 |

---

## 다음 실험 계획

1. **피처 선택** - Top 10으로 CV 15.20 달성 가능
2. **3-모델 앙상블** - XGBoost 다양성 추가
3. **CatBoost 튜닝** - lr=0.03 제출 필요
4. **가중치 미세 조정** - 57:43, 60:40 테스트

---

## 파일 명명 규칙

```
스크립트:
  train_*.py      - 학습 스크립트
  predict_*.py    - 예측 스크립트

결과:
  cv_results.json - CV 결과
  model_*.pkl     - 저장된 모델

제출:
  submission_[모델명]_cv[점수].csv
```

---

*이 문서는 실험 진행에 따라 업데이트됩니다.*
