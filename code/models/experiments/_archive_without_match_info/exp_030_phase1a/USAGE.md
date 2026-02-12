# predict_test.py 빠른 시작 가이드

## ⚡ 30초 요약

Phase 1-A 학습 모델로 test 데이터를 예측하고 submission을 생성합니다.

```bash
# 1단계: 프로젝트 루트로 이동
cd /mnt/c/LSJ/dacon/dacon/kleague-algorithm

# 2단계: 예측 스크립트 실행
python code/models/experiments/exp_030_phase1a/predict_test.py

# 3단계: Submission 파일 확인
ls -lh submissions/submission_phase1a_*.csv
```

---

## 🔄 전체 프로세스

### Phase 1: 모델 학습 (Agent 1)

```bash
# Agent 1에서 모델을 생성하면:
# ✅ code/models/experiments/exp_030_phase1a/model_x.cbm
# ✅ code/models/experiments/exp_030_phase1a/model_y.cbm

# 모델 확인
ls -lh code/models/experiments/exp_030_phase1a/model_*.cbm
```

**예상 파일 크기**: 5-10MB (각각)

**생성 방법** (Agent 1 스크립트):
```python
from code.utils.fast_experiment_phase1a import FastExperimentPhase1A
from catboost import CatBoostRegressor

exp = FastExperimentPhase1A(sample_frac=1.0, n_folds=3)
train_df = exp.load_data(sample=False)
train_df = exp.create_features(train_df)
X, y, groups, feature_cols = exp.prepare_data(train_df)

# 모델 학습
model_x = CatBoostRegressor(iterations=100, verbose=100)
model_y = CatBoostRegressor(iterations=100, verbose=100)

model_x.fit(X, y[:, 0])
model_y.fit(X, y[:, 1])

# 모델 저장
model_x.save_model('code/models/experiments/exp_030_phase1a/model_x.cbm')
model_y.save_model('code/models/experiments/exp_030_phase1a/model_y.cbm')

print("✅ 모델 저장 완료")
```

---

### Phase 2: Test 예측 (Agent 2 - 현재)

#### 방법 1: 커맨드 라인 (가장 간단)

```bash
# 기본 실행
python code/models/experiments/exp_030_phase1a/predict_test.py
```

**실행 순서**:
1. 모델 로드 (model_x.cbm, model_y.cbm)
2. Test 메타데이터 로드 (data/test.csv)
3. 각 episode 데이터 로드 (data/test/{game_id}/*.csv)
4. 21개 피처 생성
5. 마지막 패스 추출
6. 좌표 예측
7. Submission 생성

**예상 시간**: 5-10분

**출력 예시**:
```
================================================================================
Phase 1-A Test Prediction
================================================================================

================================================================================
1. 모델 로드
================================================================================
  Loading model_x.cbm... ✓
  Loading model_y.cbm... ✓
  ✅ 모델 로드 완료

================================================================================
2. Test 데이터 로드
================================================================================
  로드된 데이터: 3,627개 episode

================================================================================
3. Episode별 데이터 로드
================================================================================
  로드 완료:
    - 성공: 3,627개 episode
    - 실패: 0개 episode
    - 총 패스: 123,456개

================================================================================
4. 피처 생성 (FastExperimentPhase1A)
================================================================================
[FastExperimentPhase1A 상세 로그...]

================================================================================
5. Test 데이터 준비
================================================================================
  마지막 패스 추출: 3,627개 episode

================================================================================
6. 예측 수행
================================================================================
  X 좌표 예측 중... 완료 (2.5s)
  Y 좌표 예측 중... 완료 (2.3s)

  예측 결과:
    - 총 예측: 3,627개
    - X 범위: [0.00, 105.00]
    - Y 범위: [0.00, 68.00]

================================================================================
7. Submission 생성
================================================================================
  ✅ Submission 저장:
    경로: /path/to/submissions/submission_phase1a_cv15_95.csv
    파일명: submission_phase1a_cv15_95.csv
    파일 크기: 125.3 KB

================================================================================
✅ 예측 완료!
================================================================================
  총 실행 시간: 234.5초
  예측 수: 3,627개
  Submission: /path/to/submissions/submission_phase1a_cv15_95.csv
```

---

#### 방법 2: Python 스크립트에서 호출

```python
from pathlib import Path
from code.models.experiments.exp_030_phase1a.predict_test import Phase1APredictor

# 경로 설정
EXP_DIR = Path('code/models/experiments/exp_030_phase1a')
DATA_DIR = Path('data')

# Predictor 생성
predictor = Phase1APredictor(exp_dir=EXP_DIR, data_dir=DATA_DIR)

# 예측 실행 (CV 점수 선택사항)
try:
    results = predictor.run(cv_score=15.95)

    # 결과 활용
    print(f"✅ Success")
    print(f"   File: {results['submission_path']}")
    print(f"   Predictions: {results['n_predictions']}")
    print(f"   Time: {results['elapsed_time']:.1f}s")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
```

---

#### 방법 3: 단계별 실행 (커스터마이징)

```python
from pathlib import Path
from code.models.experiments.exp_030_phase1a.predict_test import Phase1APredictor

# Predictor 생성
exp_dir = Path('code/models/experiments/exp_030_phase1a')
data_dir = Path('data')
predictor = Phase1APredictor(exp_dir=exp_dir, data_dir=data_dir)

# Step 1: 모델 로드
model_x, model_y = predictor.load_models()

# Step 2: Test 데이터 로드
test_csv = predictor.load_test_data()

# Step 3: Episode 데이터 로드
test_df = predictor.load_episode_data(test_csv)

# Step 4: 피처 생성
test_df = predictor.create_features(test_df)

# Step 5: 데이터 준비
X, game_episodes, feature_cols = predictor.prepare_test_data(test_df)

# Step 6: 예측
predictions = predictor.predict(model_x, model_y, X)

# Step 7: Submission 생성
submission_path = predictor.create_submission(
    game_episodes=game_episodes,
    predictions=predictions,
    cv_score=15.95
)

print(f"✅ Submission: {submission_path}")
```

---

## 📊 입출력 상세

### 입력 파일

#### 1. 학습 모델
```
code/models/experiments/exp_030_phase1a/
├── model_x.cbm          # X 좌표 예측 모델 (CatBoost)
└── model_y.cbm          # Y 좌표 예측 모델 (CatBoost)
```

**파일 크기**: 각각 5-10MB
**형식**: CatBoost 바이너리 포맷 (.cbm)

#### 2. Test 메타데이터
```
data/test.csv
```

**형식**:
```csv
game_id,game_episode,path
153363,153363_1,./test/153363/153363_1.csv
153363,153363_2,./test/153363/153363_2.csv
...
```

**행 수**: 3,627개

#### 3. Test 데이터 (Episode별)
```
data/test/{game_id}/{game_episode}.csv
```

**샘플 경로**:
- data/test/153363/153363_1.csv
- data/test/153363/153363_2.csv
- ...

**형식** (각 파일):
```csv
game_id,team_id,player_id,start_x,start_y,end_x,end_y,
period_id,time_seconds,type_name,result_name,is_home,player_position,jersey_number
```

---

### 출력 파일

#### Submission CSV
```
submissions/submission_phase1a_cv15_95.csv
```

**형식**:
```csv
game_episode,end_x,end_y
153363_1,50.12,34.56
153363_2,52.34,35.78
153363_6,48.90,32.10
...
```

**구성**:
- 행: 3,627개 (test의 모든 episode)
- 열: 3 (game_episode, end_x, end_y)
- 범위: x=[0, 105], y=[0, 68]

---

## 🔍 결과 확인

### Submission 파일 확인
```bash
# 파일 존재 확인
ls -lh submissions/submission_phase1a_*.csv

# 파일 내용 확인 (처음 10줄)
head -10 submissions/submission_phase1a_cv15_95.csv

# 행 수 확인 (3,627 + header = 3,628)
wc -l submissions/submission_phase1a_cv15_95.csv

# 통계 확인
tail -1 submissions/submission_phase1a_cv15_95.csv
```

### 데이터 검증
```python
import pandas as pd
import numpy as np

# Submission 로드
submission = pd.read_csv('submissions/submission_phase1a_cv15_95.csv')

print(f"Shape: {submission.shape}")
print(f"Columns: {submission.columns.tolist()}")
print(f"\nFirst rows:")
print(submission.head())

# 범위 확인
print(f"\nX range: [{submission['end_x'].min():.2f}, {submission['end_x'].max():.2f}]")
print(f"Y range: [{submission['end_y'].min():.2f}, {submission['end_y'].max():.2f}]")

# 결측 확인
print(f"\nMissing values: {submission.isnull().sum().sum()}")
```

---

## ⚠️ 문제 해결

### 1. "Model not found"

**증상**:
```
FileNotFoundError: Model not found: .../model_x.cbm
```

**원인**: Agent 1에서 모델을 저장하지 않았음

**해결**:
```bash
# 모델 파일 확인
ls -la code/models/experiments/exp_030_phase1a/model_*.cbm

# 없으면 Agent 1 스크립트 실행
python code/models/experiments/exp_030_phase1a/train_phase1a.py
```

---

### 2. "Episode 파일 없음"

**증상**:
```
WARNING: Episode 파일 없음: 153363_1
```

**원인**: data/test/{game_id}/{game_episode}.csv 파일 경로 오류

**해결**:
```bash
# 데이터 구조 확인
find data/test -name "*.csv" | head -20

# 또는 특정 game_id 확인
ls -la data/test/153363/ | head -10
```

---

### 3. "CatBoost not installed"

**증상**:
```
ModuleNotFoundError: No module named 'catboost'
```

**해결**:
```bash
# CatBoost 설치
pip install catboost

# 또는 conda 사용
conda install -c conda-forge catboost
```

---

### 4. "메모리 부족"

**증상**: 메모리 부족 오류 또는 실행 중단

**원인**: Test 데이터가 큼 (약 50MB)

**해결**: 증분 처리 (필요시)
```python
# 분할 로드 (선택사항)
for game_id in unique_game_ids:
    test_subset = test_df[test_df['game_id'] == game_id]
    # 처리...
```

---

### 5. "ImportError: fast_experiment_phase1a"

**증상**:
```
ModuleNotFoundError: No module named 'fast_experiment_phase1a'
```

**원인**: Python path에 utils 디렉토리가 없음

**해결**: 스크립트가 자동으로 처리하지만, 수동으로도 가능
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path('code/utils')))
from fast_experiment_phase1a import FastExperimentPhase1A
```

---

## 📋 체크리스트

실행 전 확인사항:

- [ ] 모델 파일 존재 확인
  ```bash
  ls -l code/models/experiments/exp_030_phase1a/model_*.cbm
  ```

- [ ] Test 데이터 구조 확인
  ```bash
  ls -l data/test.csv
  ls -l data/test/153363/ | head -5
  ```

- [ ] CatBoost 설치 확인
  ```bash
  python -c "import catboost; print(catboost.__version__)"
  ```

- [ ] FastExperimentPhase1A 임포트 확인
  ```bash
  python -c "from code.utils.fast_experiment_phase1a import FastExperimentPhase1A; print('OK')"
  ```

- [ ] Submissions 디렉토리 생성
  ```bash
  mkdir -p submissions
  ```

---

## 📈 성능 예상

| 항목 | 예상값 | 범위 |
|------|--------|------|
| 실행 시간 | 5-10분 | 3-15분 |
| 메모리 사용 | 2-4GB | 1-8GB |
| Submission 크기 | 100-150KB | 80-200KB |
| 예측 수 | 3,627개 | 3,000-4,000 |
| CV 점수 | 15.3-15.5 | 15.0-16.0 |

---

## 🎯 다음 단계

1. **스크립트 실행**
   ```bash
   python code/models/experiments/exp_030_phase1a/predict_test.py
   ```

2. **Submission 확인**
   ```bash
   ls -lh submissions/submission_phase1a_*.csv
   head -5 submissions/submission_phase1a_*.csv
   ```

3. **DACON 제출**
   - submissions/submission_phase1a_cv15_95.csv 파일을 DACON 웹사이트에 제출
   - 링크: https://dacon.io/competitions/official/236647/mysubmission

4. **결과 기록**
   - SUBMISSION_LOG.md에 결과 기록
   - 점수, 제출 시간, 모델 정보 등 기록

5. **분석**
   - CV vs Public LB 점수 비교
   - 개선 효과 평가
   - 다음 실험 계획 수립

---

## 📞 문제 발생 시

1. **에러 메시지 전체 복사**
   ```bash
   python code/models/experiments/exp_030_phase1a/predict_test.py 2>&1 | tee prediction.log
   ```

2. **로그 파일 확인**
   ```bash
   cat prediction.log
   ```

3. **중요 파일 확인**
   ```bash
   # 모델
   ls -lh code/models/experiments/exp_030_phase1a/model_*.cbm

   # 데이터
   ls -lh data/test.csv
   ls -lh data/test/153363/ | head -5
   ```

---

## 📚 추가 정보

- **README.md**: 상세 문서
- **fast_experiment_phase1a.py**: 피처 생성 클래스
- **predict_test.py**: 이 스크립트

---

**작성일**: 2025-12-17
**버전**: 1.0
**상태**: 프로덕션 준비 완료
