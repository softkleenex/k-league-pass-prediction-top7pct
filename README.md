# ⚽ K-League Pass Coordinate Prediction (LifeTwin AI)

> **DACON K리그-서울시립대 AI 경진대회 (Track1 알고리즘 부문)**
> **Team LifeTwin AI (이상재)**
> 
> **🏆 최종 결과: 121위 (상위 7%) / 1,740팀 참여**
> **Private Score: 13.5100**

---

## 📌 대회 개요 (Competition Info)
이 프로젝트는 K리그 경기 데이터(Episode)를 분석하여 **마지막 패스의 도착 좌표(x, y)**를 예측하는 AI 모델을 개발하는 것을 목표로 합니다.

- **주최**: DACON, 한국프로축구연맹, 서울시립대학교
- **데이터**: 2024 시즌 K리그 경기 데이터 (패스 성공/실패, 좌표, 선수 정보 등)
- **평가 지표**: Euclidean Distance (유클리드 거리)
- **핵심 과제**: 경기장 내 공의 움직임 패턴을 학습하여 정확한 위치 예측 및 일반화 성능 확보

---

## 💡 핵심 접근 방법 (Key Approaches)

대회 기간 동안 **Private Score 13.5100**을 달성하기 위해 사용한 주요 전략입니다.

### 1. Delta Prediction (변화량 예측)
절대 좌표(`end_x`, `end_y`)를 직접 예측하는 대신, 패스 시작점으로부터의 **변화량(`dx`, `dy`)**을 예측하는 방식을 채택했습니다.
- 축구 패스의 물리적 특성(방향, 거리)을 더 잘 반영하며, 모델 학습의 난이도를 낮추는 효과가 있었습니다.

### 2. Iterative Pseudo-Labeling (반복적 의사 라벨링)
Public Leaderboard 점수가 높은 Test 데이터의 예측값을 학습 데이터에 추가하여 재학습하는 **Pseudo-Labeling** 기법을 적용했습니다.
- **반복 적용**: 1회성에 그치지 않고, 신뢰도가 높은 데이터부터 단계적으로 적용하여 점진적인 성능 향상을 이끌어냈습니다. (Leaderboard 기준 약 -0.11 점수 개선)

### 3. Simplicity vs Robustness (단순함과 강건함의 균형)
모델이 복잡할수록 특정 경기에 과적합(Overfitting)되는 현상을 발견했습니다. 이를 방지하기 위해 **LOGO (Leave-One-Game-Out) CV**를 통해 모델의 일반화 성능을 정량적으로 검증했습니다.
- **Zone 6x6 전략**: 경기장을 6x6 그리드로 나누어 분석했을 때 가장 안정적인 성능(Lowest Gap)을 보임을 확인하고, 이를 최종 모델의 핵심 구조로 채택했습니다.

---

## 📂 프로젝트 구조 (Project Structure)

```bash
kleague-algorithm/
├── code/                        # 전체 소스 코드
│   ├── analysis/                # 데이터 분석 및 시각화 스크립트
│   ├── baseline/                # 베이스라인 모델
│   ├── models/                  # 다양한 실험 모델 (CatBoost, LGBM, LSTM 등)
│   ├── pipeline/                # 데이터 전처리 및 파이프라인
│   └── utils/                   # 유틸리티 함수
├── analysis_results/            # 실험 결과 및 분석 리포트
├── competition_info/            # 대회 관련 문서 (규칙, 설명)
├── docs/                        # 전략 문서 및 개발 로그
├── submissions/                 # (Git 제외) 제출 파일 히스토리
└── data/                        # (Git 제외) 학습/테스트 데이터
```

---

## 🔍 주목할 만한 파일 (Noteworthy Files)

이 프로젝트의 핵심 로직이 담긴 파일들입니다.

| 파일 경로 | 설명 |
|---|---|
| `code/models/model_catboost.py` | 메인 예측 모델 (CatBoost 기반) |
| `code/models/model_6x6_single.py` | 안정성이 검증된 Zone 6x6 전략 구현체 |
| `code/pipeline/data_pipeline.py` | 데이터 전처리 및 피처 엔지니어링 파이프라인 |
| `docs/FINAL_STRATEGY_2025_12_16.md` | 최종 전략 수립 과정 및 의사결정 로그 |
| `analysis_results/cv_public_gap_analysis.csv` | CV 점수와 Public 점수 간의 괴리 분석 |

---

## 🛠 기술 스택 (Tech Stack)

- **Language**: Python 3.9+
- **Machine Learning**: CatBoost, LightGBM, Random Forest, XGBoost
- **Data Processing**: Pandas, NumPy
- **Analysis**: Matplotlib, Seaborn (EDA)

---

## 🚀 설치 및 실행 (Installation)

이 레포지토리는 데이터(`data/`) 및 대용량 파일(`catboost_info/`, `logs/`)을 포함하지 않습니다.

1. **환경 설정**
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: `requirements.txt`가 없는 경우 주요 라이브러리 설치 필요: `pandas`, `numpy`, `catboost`, `lightgbm`, `scikit-learn`)*

2. **데이터 준비**
   - `data/` 폴더에 `train.csv`, `test.csv`, `match_info.csv`를 위치시킵니다.

3. **모델 학습 및 추론**
   ```bash
   # 예시 실행 커맨드
   python code/models/model_catboost.py
   ```

---

**Last Updated:** 2026-01-18
**Author:** Sangjae Lee (LifeTwin AI)