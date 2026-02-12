# exp_067: MAE Loss 실험

## 핵심 발견

**MAE Loss가 RMSE보다 훨씬 좋음!**

| 모델 | CV | 개선 |
|------|-----|------|
| RMSE 기준 (exp_047) | 14.25 | - |
| RMSE 5-Fold | 14.12 | +0.13 |
| MAE 기본 | 13.79 | +0.46 |
| MAE 최적 (lr=0.01, l2=7) | 13.74 | +0.51 |
| **MAE 5-Fold 최적** | **13.66** | **+0.59** |

## 최적 파라미터

```python
params_mae_opt = {
    'iterations': 4000,
    'depth': 8,
    'learning_rate': 0.01,
    'l2_leaf_reg': 7.0,
    'loss_function': 'MAE',
    'early_stopping_rounds': 100
}
```

## 시도한 Loss Functions

| Loss | CV | 결과 |
|------|-----|------|
| MAE | **13.66** | Best |
| RMSE | 14.12 | 기준 |
| Quantile(0.5) | 13.79 | MAE와 동일 |
| Huber | 14.58 | 악화 |
| MAPE | 21.85 | 매우 악화 |

## 예측 분포 비교

| 모델 | end_x 평균 | end_y 표준편차 |
|------|-----------|---------------|
| RMSE | 67.79 | 20.86 |
| MAE | 66.80 | 22.51 |

MAE가 y 방향으로 더 분산된 예측 생성

## 내일 제출 순서

1. submission_mae_5fold_cv13.66.csv (Best CV)
2. submission_mae_opt_cv13.74.csv
3. submission_ensemble_mae_rmse_55.csv
4. submission_strongreg_cv14.32.csv (Gap 실험)
5. submission_ensemble_mae_rmse_73.csv

## 주의사항

- CV 개선 ≠ Public 개선 보장
- 기존 패턴: CV 낮추면 Gap 양수 → Public 악화
- MAE Loss는 다를 수 있음 (이상치 처리 방식 다름)
