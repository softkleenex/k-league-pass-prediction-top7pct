# Analysis 디렉토리

> **목적:** 모델, 피처, Zone, 실험 분석 문서 통합 관리
> **업데이트:** 2025-12-16

---

## 📂 폴더 구조

### models/ (7개)
**모델별 분석 문서**
- LSTM 분석 (3개): 실험 실패 기록, 데이터 분석
- XGBoost 분석 (4개): 구현, 성능, 딜리버러블

**주요 파일:**
- `LSTM_FAILURE_ANALYSIS.md` - LSTM 실패 종합 분석
- `XGBOOST_ANALYSIS_2025_12_11.md` - XGBoost 상세 분석

---

### zones/ (8개)
**Zone 기반 접근 분석**
- Zone 6x6 안정성 증명
- Zone Fallback 실험
- Field Region 분석

**주요 파일:**
- `ZONE_6x6_STABILITY_PROOF.md` - 14회 실험 통계 증명
- `ZONE_STABILITY_ANALYSIS_INDEX.md` - Zone 분석 인덱스

---

### features/ (4개)
**Feature 엔지니어링 분석**
- Domain Features 분석
- Pass Distance 분석

**주요 파일:**
- `DOMAIN_FEATURES_ANALYSIS.md` - Domain 피처 상세 분석
- `PASS_DISTANCE_SUMMARY.md` - Pass 거리 통계

---

### experiments/ (7개)
**실험 및 전략 분석**
- CV-Public Gap 분석
- Ensemble 실험
- 제출 전략

**주요 파일:**
- `CV_PUBLIC_GAP_ANALYSIS_2025_12_12.md` - Gap 분석
- `SIMPLICITY_WINS_QUICK_GUIDE.md` - 단순성 원칙

---

## 🔍 사용법

### 특정 주제 찾기

**모델 성능 분석:**
```bash
ls docs/analysis/models/
```

**Zone 실험 결과:**
```bash
ls docs/analysis/zones/
```

**Feature 엔지니어링:**
```bash
ls docs/analysis/features/
```

**실험 전략:**
```bash
ls docs/analysis/experiments/
```

---

## 📊 통계

| 폴더 | 파일 수 | 주요 내용 |
|------|---------|-----------|
| models/ | 7 | LSTM, XGBoost 분석 |
| zones/ | 8 | Zone 6x6, Fallback |
| features/ | 4 | Domain, Pass Distance |
| experiments/ | 7 | CV Gap, Ensemble |
| **합계** | **26** | - |

---

**마지막 업데이트:** 2025-12-16
**정리 작업:** Phase 2 완료
