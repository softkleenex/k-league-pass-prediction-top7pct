# ⚽ K-League Pass Coordinate Prediction (LifeTwin AI)

> **DACON K리그-서울시립대 AI 경진대회 (Track1 알고리즘 부문)**  
> **Team LifeTwin AI (이상재)**  
>
> **🏆 Final Results (Private Leaderboard)**
> - **Rank:** **121st** / 1,782 Teams (상위 6.8%)
> - **Private Score:** **13.5100** (1위: 12.16102, 격차: 1.34898)
> - **Public Score:** 13.4342

---

## 📌 Competition Overview

이 프로젝트는 2024 시즌 K리그 경기 데이터를 기반으로 **마지막 패스의 도착 좌표(x, y)**를 예측하는 AI 모델을 개발하는 대회입니다. 단순한 통계적 접근을 넘어, 경기장 내 공의 흐름과 선수들의 움직임 패턴을 학습하여 정밀한 위치 예측을 목표로 했습니다.

### 🎯 Goal
- **Input:** 경기 내 패스 이벤트 시퀀스 (Episode), 선수/팀 정보, 시간 등
- **Output:** 해당 Episode의 마지막 패스가 도착할 **(x, y) 좌표**
- **Evaluation:** Euclidean Distance (유클리드 거리, 낮을수록 우수)

### 📊 Data Description
- **Train:** 15,435 Episodes (356,721 Passes)
- **Test:** 2,414 Episodes (마지막 패스 예측 대상)
- **Key Features:** `start_x`, `start_y`, `end_x`, `end_y`, `player_id`, `team_id`, `time_seconds`

---

## 💡 Core Strategy (핵심 전략)

Private Score 13.5100을 달성한 3가지 핵심 전략입니다.

### 1. Delta Prediction (변화량 예측)
절대 좌표(`end_x`, `end_y`)를 직접 예측하는 대신, **(시작점 대비 변화량 `dx`, `dy`)**을 예측하는 방식으로 문제를 재정의했습니다.
- **Why?** 패스의 방향과 거리는 시작 위치보다 '상황'에 더 의존적입니다.
- **Effect:** 모델이 경기장의 특정 위치에 과적합되는 것을 막고, 패스의 물리적 특성(벡터)을 더 잘 학습했습니다.

### 2. Iterative Pseudo-Labeling (반복적 의사 라벨링)
대회 후반부, Test 데이터에 대한 예측값을 다시 학습 데이터로 활용하는 **Semi-Supervised Learning** 기법을 도입했습니다.
- **Process:**
    1. Base Model로 Test 데이터 예측
    2. 예측된 Test 데이터를 Train 데이터에 병합 (Pseudo-Label)
    3. 확장된 데이터로 모델 재학습
    4. 위 과정을 반복하며 신뢰도 향상
- **Result:** Public Leaderboard 점수 기준 **약 -0.11 개선** (13.54 → 13.43)

### 3. Simplicity & Robustness (Zone 6x6)
초기 실험에서 복잡한 딥러닝(LSTM) 모델이 특정 경기에 심각하게 과적합(Gap > 3.0)되는 문제를 발견했습니다. 이를 해결하기 위해 **LOGO (Leave-One-Game-Out) CV**를 통해 가장 일반화 성능이 뛰어난 **Zone 6x6 (경기장을 36개 구역으로 분할)** 접근법을 채택했습니다.
- **Outcome:** 어떠한 새로운 경기 데이터(OOD)가 들어와도 성능 저하가 거의 없는 견고한(Robust) 모델을 구축했습니다.

---

## 🏆 Submission History (Top Records)

대회 기간 중 의미 있었던 주요 제출 기록입니다.

| Rank | Submission File | Public Score | CV Score | Method | Date |
|:---:|---|:---:|:---:|---|:---:|
| **1** | `submission_iterative_pseudo.csv` | **13.4343** | - | **Iterative Pseudo-Labeling (Final Best)** | 01-06 |
| **2** | `submission_pseudo_3seed_fixed.csv` | 13.4390 | 13.36 | Pseudo-Labeling (3-Seed Ensemble) | 01-05 |
| **3** | `submission_multimodel_cv13.52.csv` | 13.4924 | 13.52 | Multi-Model Ensemble | 12-31 |
| **4** | `submission_l2_30_cv13.51.csv` | 13.4958 | 13.51 | L2 Regularization Tuned | 01-01 |
| **5** | `submission_optimized_ensemble.csv` | 16.3502 | 16.35 | Initial Zone 6x6 Baseline | 12-04 |

---

## 💡 Lessons Learned & Failure Analysis (비판적 회고)

이 프로젝트는 반복적인 실험과 실패를 통해 **Out-of-Distribution (OOD)** 데이터에 대한 모델의 일반화 성능을 높이는 과정이었습니다.

### 1. 피처 엔지니어링의 역설 (Domain v2의 실패)
- **가설**: "도메인 지식을 활용한 복잡한 피처와 강한 정규화(Regularization)를 결합하면 OOD 일반화 성능이 좋아질 것이다."
- **결과**: 오히려 예측 오차가 순진한 베이스라인(Naive Baseline)보다 대폭 상승(CV 18.37)하는 치명적 실패를 겪었습니다.
- **원인**: 너무 적은 피처(6개)에 과도한 정규화를 적용해 모델이 심각하게 과소적합(Underfitting)되었으며, 경기장 위치 같은 핵심 공간 패턴(Zone)을 제거한 것이 패착이었습니다.
- **교훈**: 모델의 단순성이 OOD 강인성에 도움이 되지만, 데이터의 도메인 핵심 패턴(공간적 특성)까지 제거할 수준으로 피처를 깎아내면 예측 능력을 완전히 상실합니다. 정규화는 만능이 아닙니다.

### 2. "Simplicity is the ultimate sophistication"
- 복잡한 딥러닝(LSTM)이나 수십 개의 파생 변수를 추가한 GBDT 앙상블 모델들은 학습 환경(특정 경기 패턴)에 심각하게 과적합(CV-Public Gap > 3.0)되는 현상을 반복적으로 보였습니다.
- 결론적으로 가장 단순한 형태의 통계적 **Zone 6x6 모델(최소 파라미터 288개)**이 완전히 처음 보는 게임 데이터(LOGO CV)에 대해서도 거의 성능 저하 없이 가장 압도적이고 안정적인 성능(Gap 0.02)을 보였습니다. 

---

## 📂 Repository Structure

```bash
kleague-algorithm/
├── code/
│   ├── models/                  # 예측 모델 소스 코드
│   │   ├── model_catboost.py    # [Main] CatBoost 기반 예측 모델
│   │   ├── model_6x6_single.py  # [Baseline] Zone 6x6 안정성 검증 모델
│   │   └── ...
│   ├── pipeline/                # 데이터 처리 파이프라인
│   │   ├── data_pipeline.py     # 전처리 및 피처 엔지니어링
│   │   └── ...
│   └── analysis/                # 데이터 분석 및 시각화
│       └── ...
├── analysis_results/            # 실험 결과 및 리포트
│   ├── cv_public_gap_analysis.csv
│   └── game_level_cv_scores.csv
├── docs/                        # 프로젝트 문서 및 전략 로그
│   ├── FINAL_STRATEGY_2025_12_16.md
│   └── ...
└── submissions/                 # (Git Ignored) 제출 파일 아카이브
```

---

## 🛠 Tech Stack

- **Core:** Python 3.9+
- **Modeling:** CatBoost, LightGBM, XGBoost, Random Forest
- **DataOps:** Pandas, NumPy, Scikit-learn
- **Tools:** Git, Jupyter Notebook

---

## 🚀 How to Run

이 레포지토리는 코드와 설정 파일만 포함하며, 데이터셋은 포함되어 있지 않습니다.

1. **Install Dependencies**
   ```bash
   pip install pandas numpy catboost lightgbm scikit-learn matplotlib seaborn
   ```

2. **Prepare Data**
   - `data/` 폴더 생성 후 대회 데이터 (`train.csv`, `test.csv`, `match_info.csv`) 배치

3. **Train & Predict**
   ```bash
   # CatBoost 메인 모델 실행
   python code/models/model_catboost.py
   ```

---

**Author:** Sangjae Lee (LifeTwin AI)  
**Last Updated:** 2026-01-18
