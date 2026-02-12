# exp_068~076: 실험 결과 종합 (12/29)

## 요약

**NEW BEST: exp_076 Delta Prediction (CV 13.72)**

| 실험 | 접근법 | CV | vs Best |
|------|--------|-----|---------|
| exp_067 | MAE Absolute (3-fold) | 13.79 | +0.07 |
| exp_068 | Sequence Features | 14.14 | +0.42 |
| exp_069 | Path Signatures | 14.16 | +0.44 |
| exp_070 | LightGBM MAE | 14.04 | +0.32 |
| exp_071 | MAE+RMSE Ensemble | 13.95 | +0.23 |
| exp_074 | Shared Code Features | 13.94 | +0.22 |
| exp_075 | Delta Pred (5-fold) | 13.89 | +0.17 |
| **exp_076** | **Delta Pred (3-fold)** | **13.72** | **- (Best)** |

## 상세 결과

### exp_068: Sequence Features (실패)

**아이디어:** 시퀀스 통계 피처 추가 (마지막 패스 제외)
- seq_length, seq_x_mean, seq_x_std, ...
- prev3_x_mean, prev3_dx_mean, ...

**결과:** CV 14.14 (+0.48 악화)
- 새 피처 importance 1-2% (미미)
- 기존 피처가 이미 충분히 좋음

### exp_069: Path Signatures (실패)

**아이디어:** esig 라이브러리로 경로 signature 인코딩
- 시퀀스의 기하학적 특성 수학적 캡처
- order=3 → 15차원 signature

**결과:** CV 14.16 (+0.50 악화)
- Signature importance 12.2% (합계)
- 하지만 예측 성능 개선 없음

### exp_070: LightGBM (실패)

**아이디어:** CatBoost 대신 LightGBM 사용

**결과:** CV 14.04 (+0.38 악화)
- CatBoost가 이 데이터에 더 적합

### XGBoost (실패)

**아이디어:** CatBoost 대신 XGBoost 사용

**결과:** CV 14.22 (+0.56 악화)
- CatBoost >>> XGBoost > LightGBM

### exp_071: MAE+RMSE Ensemble

**아이디어:** MAE와 RMSE 모델 앙상블
- 서로 다른 loss의 관점 조합

**결과:**
| MAE 비율 | CV |
|----------|-----|
| 0% (RMSE only) | 14.35 |
| 30% | 14.15 |
| 50% | 14.05 |
| 70% | 13.98 |
| **100% (MAE only)** | **13.95** ★ |

**결론:** MAE 100%가 최적! RMSE 섞으면 악화됨.

## 핵심 인사이트

1. **MAE Loss >> RMSE Loss**
   - CV 14.25 (RMSE) → 13.66 (MAE) = +0.59 개선
   - 이상치에 덜 민감한 MAE가 이 문제에 적합

2. **TOP_12 피처가 최적**
   - 추가 피처는 노이즈만 증가
   - goal_angle, zone_y, goal_distance가 핵심

3. **CatBoost가 최적 모델**
   - CatBoost > LightGBM > XGBoost
   - 0.2~0.5 차이

4. **앙상블보다 단일 모델**
   - MAE + RMSE 앙상블 = 악화
   - MAE 단독이 최적

---

## 12/29 추가 실험

### exp_074: Shared Code Features (효과 없음)

**배경:** DACON 공유 코드 분석 후 새 feature 테스트

**테스트:**
- goal_open_angle (골대 열린 각도)
- is_final_third (x > 70)
- Data Augmentation (Y-flip)

**결과:** 모두 효과 없음 또는 악화

### exp_075-076: Delta Prediction (성공!)

**핵심 발견:** 공유 코드에서 `target_mode='delta'` 발견

**방법:**
- 기존: end_x, end_y 직접 예측
- 신규: dx, dy 예측 후 start + delta로 변환

**결과:**
| 방식 | CV | 개선 |
|------|-----|------|
| Absolute | 13.79 | - |
| **Delta** | **13.72** | **-0.07** |

---

## 제출 파일 (12/30)

1. **submission_delta_3fold_cv13.72.csv** (NEW BEST!)
2. submission_mae_cv13.79.csv (백업)

## 핵심 교훈

1. **Delta Prediction > Absolute Prediction** (-0.07)
2. 새 feature 추가보다 **예측 방식 변경**이 효과적
3. 공유 코드 분석의 중요성
