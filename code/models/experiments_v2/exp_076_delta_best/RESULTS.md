# exp_076: Delta Prediction (Best Settings)

**날짜**: 2025-12-29
**상태**: NEW BEST CV!

## 개요
Delta prediction + exp_067 설정 (TOP_15, 3-fold, 빠른 파라미터)

## 설정
- Features: TOP_15 (zone_x, result_encoded, diff_x, velocity 포함)
- CV: 3-Fold GroupKFold
- Parameters: iterations=1000, lr=0.05, l2_reg=3.0, early_stop=50

## 결과
```
======================================================================
Results:
  Absolute: CV 13.7878
  Delta:    CV 13.7154 (-0.0724)  ★ NEW BEST!
======================================================================
```

## 핵심 발견
| 방식 | CV | 개선 |
|------|-----|------|
| Absolute (end_x, end_y) | 13.79 | - |
| **Delta (dx, dy)** | **13.72** | **-0.07** |

## 생성된 파일
- `submission_delta_3fold_cv13.72.csv`

## 결론
- **Delta prediction이 Absolute보다 -0.07 개선**
- 공유 코드에서 발견한 핵심 인사이트
- 다음 제출 시 사용 권장
