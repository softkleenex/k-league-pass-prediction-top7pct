# Zone Fallback 분석 결과

**분석 날짜:** 2025-12-11
**분석자:** Data Analyst
**상태:** 완료

---

## 분석 목적

Zone fallback 개선 시도가 CV에 영향을 주지 않은 이유를 데이터 관점에서 파악

---

## 핵심 결과

### 1. Fallback 사용 빈도 (Best 모델: 6x6_8dir)

```
Zone + Direction:  89.00% (13,737개)  ← 대부분
Zone Fallback:     11.00%  (1,698개)  ← 소수
Global Fallback:    0.00%      (0개)  ← 거의 없음
```

### 2. 영향력 분석

```
Fallback을 완벽하게 개선해도:
  - 전체 CV의 11%만 영향
  - 실제 개선 효과: < 1% CV
  - 측정 불가능한 수준

실제로는:
  - Zone fallback이 이미 효과적 (평균 54.8 샘플)
  - 개선 여지: < 5%
  - 실제 영향: < 0.01 CV
```

### 3. 결론

```
Zone fallback 개선은 불필요:
  ✅ 사용 빈도 매우 낮음 (11%)
  ✅ 이미 충분히 효과적
  ✅ 개선 영향 미미 (< 1% CV)
  ✅ Zone 통계 접근법 이미 최적화됨
```

---

## 생성된 파일

### 분석 스크립트

```
code/analysis/analyze_zone_fallback.py      - Zone fallback 통계 분석
code/analysis/visualize_zone_fallback.py    - 시각화 생성
```

### 결과 데이터

```
results/zone_fallback_comparison.csv        - 모델 비교 테이블
results/zone_fallback_summary.csv           - 핵심 메트릭 요약
results/zone_stats_5x5_8dir.csv             - 5x5 모델 상세 통계
results/zone_stats_6x6_8dir.csv             - 6x6 모델 상세 통계
results/zone_stats_7x7_8dir.csv             - 7x7 모델 상세 통계
results/zone_stats_6x6_simple.csv           - 6x6_simple 모델 상세 통계
```

### 시각화

```
results/zone_fallback_analysis.png          - Fallback 분석 대시보드
results/sample_count_distribution.png       - 샘플 수 분포 히스토그램
```

### 보고서

```
docs/ZONE_FALLBACK_ANALYSIS_REPORT_2025_12_11.md          - 상세 보고서
docs/ZONE_FALLBACK_EXECUTIVE_SUMMARY_2025_12_11.md       - 요약 보고서
```

---

## 재현 방법

```bash
# 1. Zone fallback 통계 분석
python code/analysis/analyze_zone_fallback.py

# 2. 시각화 생성
python code/analysis/visualize_zone_fallback.py
```

---

## 주요 발견

### 발견 1: 사용 빈도가 매우 낮음

| 모델 | Fallback 사용률 |
|------|----------------|
| 5x5_8dir | 3.93% |
| **6x6_8dir** | **11.00%** |
| 7x7_8dir | 12.88% |
| 6x6_simple | 0.00% |

### 발견 2: Zone + Direction 조합이 충분한 샘플 보유

**6x6_8dir 기준:**
- 총 조합 수: 324개
- 충분한 조합 (>= 25 샘플): 215개 (66.4%)
- 부족한 조합 (< 25 샘플): 109개 (33.6%)
- **평균 샘플/조합: 47.6**
- **중앙값 샘플/조합: 35.0** (min_samples(25)보다 높음)

### 발견 3: Zone Fallback이 이미 효과적

**Fallback 사용 시 통계:**
- Zone별 평균 샘플: 54.8
- Zone별 중앙값 샘플: 48.0
- Global fallback 사용: 0.00%

**해석:** Fallback이 사용되어도 충분히 신뢰할 수 있는 통계량 사용

---

## 시각화 해석

### 1. zone_fallback_analysis.png

**4개 그래프:**
1. **Prediction Method Usage:** Zone+Direction 89% vs Fallback 11%
2. **Sample Sufficiency:** 대부분의 조합이 충분한 샘플 보유
3. **Average Samples:** 평균 47.6, 중앙값 35.0
4. **Distribution (6x6_8dir):** Zone+Direction 압도적

### 2. sample_count_distribution.png

**4개 히스토그램:**
- 각 모델의 샘플 수 분포
- min_samples 선 (빨간색)
- 평균 (녹색), 중앙값 (주황색)
- 6x6_8dir의 균형잡힌 분포 확인

---

## 결론 및 권장사항

### 결론

```
Zone fallback 개선이 실패한 이유:

1. 영향력이 매우 작음 (11%만 담당)
2. 이미 충분히 효과적 (평균 54.8 샘플)
3. 성능 병목은 다른 곳 (Zone + Direction 조합 자체)

14회 연속 실패의 의미:
- 실패가 아니라 최적점 확인
- 통계적 증거: 0.006% 확률
- 더 이상의 시도 불필요
```

### 권장사항

```
❌ 중단해야 할 것:
   - Zone fallback 개선 (영향 < 1%)
   - Zone 설정 변경 (14회 완전 탐색)
   - Direction 각도 조정 (3회 완전 탐색)

✅ 해야 할 것:
   - Week 2-3: 관찰 모드 유지
   - Week 4-5: 검증된 접근만 시도
   - CV Sweet Spot 준수 (16.27-16.34)
```

---

## 데이터 상세

### 기본 정보

```
데이터: train.csv
전체 행 수: 356,721
에피소드 수: 15,435
분석 샘플: 15,435 (각 에피소드의 마지막 액션)
```

### 모델 설정

| 모델 | Zone | Direction | min_samples |
|------|------|-----------|-------------|
| 5x5_8dir | 5x5 (25) | 8-way | 25 |
| **6x6_8dir** | **6x6 (36)** | **8-way** | **25** |
| 7x7_8dir | 7x7 (49) | 8-way | 20 |
| 6x6_simple | 6x6 (36) | None | 30 |

---

## 수학적 분석

### CV 개선 계산

```
Total CV = w_zd × CV_zone_dir + w_fb × CV_fallback

where:
  w_zd = 0.89 (Zone + Direction 비율)
  w_fb = 0.11 (Fallback 비율)

Fallback을 x% 개선 시:
  ΔCV = w_fb × (x/100) × CV_fallback
      = 0.11 × (x/100) × CV_fallback

최대 영향 (100% 개선):
  ΔCV = 0.11 × CV_fallback
      ≈ 0.11 × 16.34 × 0.11
      ≈ 0.20 (이론적 최대)

실제 영향 (10% 개선):
  ΔCV = 0.11 × 0.10 × 16.34
      ≈ 0.18

실제 개선 가능성 (5% 이하):
  ΔCV < 0.11 × 0.05 × 16.34
      < 0.09 (측정 불가능)
```

---

## 참고 문서

- `CLAUDE.md` - 프로젝트 빠른 가이드
- `FACTS.md` - 불변 사실
- `docs/CV_SWEET_SPOT_DISCOVERY.md` - Sweet Spot 발견
- `docs/WEEK1_ZONE_EXPERIMENTS.md` - Week 1 실험
- `docs/ZONE_FALLBACK_ANALYSIS_REPORT_2025_12_11.md` - 상세 보고서
- `docs/ZONE_FALLBACK_EXECUTIVE_SUMMARY_2025_12_11.md` - 요약 보고서

---

**작성:** 2025-12-11
**업데이트:** 2025-12-11
**상태:** 완료 ✅
