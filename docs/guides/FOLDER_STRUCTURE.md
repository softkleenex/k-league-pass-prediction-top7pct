# 폴더 구조 가이드

정리 완료: 2025-12-12

---

## 📁 디렉토리 구조

```
kleague-algorithm/
├── competition_info/          # 대회 정보
│   ├── overview.txt          # 대회 개요
│   └── key_findings.txt      # 핵심 발견사항
│
├── submissions/               # 제출 파일
│   ├── submitted/            # 제출한 파일들
│   │   ├── SCORES.md        # 점수 기록 ⭐
│   │   ├── submission_safe_fold13.csv (Public 16.36) - BEST
│   │   ├── submission_lightgbm_cv12.15.csv (Public 18.76)
│   │   ├── submission_catboost_cv12.15.csv (Public 18.80)
│   │   ├── submission_zone_player_lgbm_cv15.94.csv (Public 16.58)
│   │   ├── submission_zone_sequence_lgbm_cv15.95.csv (Public 16.36)
│   │   └── submission_all_passes_cv15.88.csv (Public 16.30)
│   │
│   └── pending/              # 미제출 파일들
│       ├── submission_domain_features_cv14.81.csv (NEW!)
│       ├── submission_randomforest_cv12.59.csv
│       ├── submission_knn_cv12.94.csv
│       ├── submission_zone_10x10_cv16.88.csv
│       └── submission_zone_20x20_cv17.28.csv
│
├── code/
│   └── models/
│       ├── best/             # Best 모델
│       │   ├── model_safe_fold13.py (Zone 앙상블)
│       │   └── model_domain_features_lgbm.py (도메인 피처)
│       │
│       ├── active/           # 현재 실험
│       │   └── model_all_passes_lgbm.py (전체 패스 학습)
│       │
│       ├── archive/          # 옛날 실험
│       │   ├── model_zone_10x10.py
│       │   ├── model_zone_20x20.py
│       │   ├── model_zone_player_lgbm.py
│       │   └── model_zone_sequence_lgbm.py
│       │
│       └── utils/            # 유틸리티
│           └── generate_submissions.py
│
├── logs/
│   ├── recent/              # 최근 로그
│   │   ├── domain_features_lgbm.log
│   │   └── all_passes_lgbm.log
│   │
│   └── archive/             # 옛날 로그
│       └── [30+ 로그 파일들]
│
├── docs/                    # 문서
│   ├── WEEK2_5_ACTION_PLAN.md
│   ├── VERIFICATION_REPORT_2025_12_09.md
│   ├── CV_SWEET_SPOT_DISCOVERY.md
│   └── ...
│
├── CLAUDE.md               # 빠른 가이드 ⭐
├── STATUS.md               # 오늘의 상태 ⭐
├── FACTS.md                # 불변 사실 ⭐
├── EXPERIMENT_LOG.md       # 실험 로그 ⭐
└── DECISION_TREE.md        # 의사결정 가이드

```

---

## 🎯 빠른 참조

### 제출 파일 확인
```bash
cat submissions/submitted/SCORES.md
```

### Best 모델 확인
```bash
ls code/models/best/
```

### 대회 정보 확인
```bash
cat competition_info/overview.txt
cat competition_info/key_findings.txt
```

### 최신 실험 확인
```bash
cat logs/recent/domain_features_lgbm.log
```

---

## 📊 현재 상태 (2025-12-12)

**Best 성능:**
- Zone 6x6: Public 16.36 (241위)
- 도메인 피처: CV 14.81 (제출 대기!)

**제출 현황:**
- 사용: 6/175회 (3.4%)
- 남음: 169회

**다음 단계:**
1. 도메인 피처 제출
2. 결과 확인
3. 전략 수정

---

## 🔗 주요 문서

매일 확인:
- **CLAUDE.md** - 빠른 가이드
- **STATUS.md** - 오늘의 상태
- **submissions/submitted/SCORES.md** - 점수 기록

상세 정보:
- **FACTS.md** - 불변 사실
- **EXPERIMENT_LOG.md** - 실험 로그
- **competition_info/** - 대회 정보
