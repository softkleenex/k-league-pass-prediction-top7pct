# exp_075: Delta Prediction (5-Fold)

**날짜**: 2025-12-29

## 개요
Delta (dx, dy) 예측 방식 테스트 - 5-Fold CV, TOP_12 features

## 방법
- 기존: end_x, end_y 직접 예측
- 신규: dx, dy (변화량) 예측 후 start_x + dx, start_y + dy로 변환

## 결과
```
Delta CV: 13.8893
vs Baseline (13.66 추정): +0.23
```

## 결론
- Delta prediction이 효과적임을 확인
- 하지만 exp_067 설정 (3-fold, TOP_15)과 비교 필요 → exp_076
