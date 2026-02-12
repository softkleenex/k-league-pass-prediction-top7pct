# exp_074: Shared Code Features Analysis

**날짜**: 2025-12-29

## 배경
DACON 공유 코드 (Deep Gaussian Process) 분석 후 새로운 feature들 테스트

## 테스트한 Feature들
- `goal_open_angle`: 골대 열린 각도 (포스트 간 각도 차이)
- `is_final_third`: 공격 1/3 지역 (x > 70)
- `min_dist_to_touchline`: 터치라인까지 최소 거리
- `match_hour`, `is_weekend`, `home_rest`, `away_rest`: 경기 맥락

## 결과

### exp_074a: Feature 추가 테스트
| 실험 | CV | vs Baseline |
|------|-----|-------------|
| Baseline (TOP_12) | 13.9486 | - |
| + New Features | 14.0088 | **+0.06 (악화)** |
| + Data Augmentation | 13.9363 | -0.01 (미미) |

### exp_074b: Zone & Delta 테스트
| 실험 | CV | vs Baseline |
|------|-----|-------------|
| Zone_y=6 (baseline) | 13.9486 | - |
| Zone_y=3 | 13.9595 | +0.01 (악화) |
| Zone_y=4 | 13.9426 | -0.006 (미미) |
| **Delta (dx,dy)** | **13.8893** | **-0.06 (개선!)** |

## 결론
- 공유 코드의 새 feature들 (goal_open_angle 등): **효과 없음**
- Data Augmentation (Y-flip): **미미한 효과**
- Zone_y 변경: **효과 없음**
- **Delta prediction: 효과 있음!** → exp_075, exp_076에서 추가 검증
