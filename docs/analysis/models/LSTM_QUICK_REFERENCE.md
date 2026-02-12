# LSTM 모델 설계 - 빠른 참조 가이드

> 상세한 분석은 LSTM_DATA_ANALYSIS_2025_12_11.md 참조

---

## 핵심 수치 (One-page Summary)

```
데이터 준비
├─ 총 샘플: 114,602개 (시퀀스 길이 5)
├─ 에피소드: 9,579개 (전체 62%)
├─ Pass 개수: 162,497개 (전체 91%)
└─ 상태: ✓ LSTM 학습 충분

입력 피처 (4D)
├─ start_x: [0, 105] → ÷105 정규화
├─ start_y: [0, 68] → ÷68 정규화
├─ end_x: [0, 105] → ÷105 정규화
└─ end_y: [0, 68] → ÷68 정규화

출력 (Classification)
├─ 클래스 0: Unsuccessful (14%)
└─ 클래스 1: Successful (86%)
    └─ ⚠ 불균형 → class_weight={0: 1.0, 1: 0.16}

시간 정보
├─ 평균 간격: 3.764초
├─ 중앙값: 2.956초
└─ 전처리: log(1 + time_diff)
```

---

## 추천 모델 구조

### 기본 모델

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input

model = Sequential([
    Input(shape=(5, 4)),           # seq_length=5, features=4
    LSTM(64, return_sequences=True, dropout=0.2),
    LSTM(32, return_sequences=False, dropout=0.2),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(2, activation='softmax')  # 2 classes
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy', Precision(), Recall(), AUC()]
)
```

### 학습 설정

```python
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=64,
    epochs=50,
    class_weight={0: 1.0, 1: 0.16},
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=10),
        ReduceLROnPlateau(factor=0.5, patience=5)
    ]
)
```

---

## 데이터 전처리 (Python Snippet)

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 1. 데이터 로드
df = pd.read_csv('train.csv')
pass_df = df[df['type_name'] == 'Pass'].copy()

# 2. 에피소드별 필터링 (Pass >= 6)
episode_counts = pass_df.groupby('game_episode').size()
valid_episodes = episode_counts[episode_counts >= 6].index
pass_df = pass_df[pass_df['game_episode'].isin(valid_episodes)]

# 3. 좌표 정규화
pass_df['start_x_norm'] = pass_df['start_x'] / 105.0
pass_df['start_y_norm'] = pass_df['start_y'] / 68.0
pass_df['end_x_norm'] = pass_df['end_x'] / 105.0
pass_df['end_y_norm'] = pass_df['end_y'] / 68.0

# 4. 결과 인코딩
pass_df['is_successful'] = (pass_df['result_name'] == 'Successful').astype(int)

# 5. 시퀀스 생성
def create_sequences(episode_group, seq_length=5):
    sequences = []
    targets = []

    group_sorted = episode_group.sort_values('time_seconds')
    coords = group_sorted[['start_x_norm', 'start_y_norm',
                           'end_x_norm', 'end_y_norm']].values
    results = group_sorted['is_successful'].values

    for i in range(len(coords) - seq_length):
        sequences.append(coords[i:i+seq_length])
        targets.append(results[i+seq_length])

    return sequences, targets

all_sequences = []
all_targets = []

for episode, group in pass_df.groupby('game_episode'):
    seqs, targets = create_sequences(group, seq_length=5)
    all_sequences.extend(seqs)
    all_targets.extend(targets)

X = np.array(all_sequences)  # (114602, 5, 4)
y = np.array(all_targets)    # (114602,)

# 6. 원-핫 인코딩
y_onehot = np.eye(2)[y]  # (114602, 2)

print(f"X shape: {X.shape}")
print(f"y shape: {y_onehot.shape}")
```

---

## 성능 평가 메트릭

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc
)

# 예측
y_pred = model.predict(X_test)
y_pred_class = np.argmax(y_pred, axis=1)

# 메트릭
print("Accuracy:", accuracy_score(y_test_class, y_pred_class))
print("Precision:", precision_score(y_test_class, y_pred_class))
print("Recall:", recall_score(y_test_class, y_pred_class))
print("F1-Score:", f1_score(y_test_class, y_pred_class))
print("AUC-ROC:", roc_auc_score(y_test[:, 1], y_pred[:, 1]))

# Confusion Matrix
cm = confusion_matrix(y_test_class, y_pred_class)
print("\nConfusion Matrix:")
print(cm)

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test[:, 1], y_pred[:, 1])
plt.plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.3f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
```

---

## 시퀀스 길이 선택 기준

| Seq Len | 샘플 수 | 에피소드 | 선택 기준 |
|---------|--------|--------|---------|
| 3 | 136,601 | 11,496 | 최대 데이터 활용, 단순 패턴 |
| **5** | **114,602** | **9,579** | **권장: 균형잡힘** |
| 7 | 96,274 | 7,959 | 복잡한 패턴, 데이터 손실 커짐 |

**권장: 시퀀스 길이 = 5**

---

## 피처 엔지니어링 옵션

### Option 1: 기본 (권장)
```python
features = [start_x_norm, start_y_norm, end_x_norm, end_y_norm]
```

### Option 2: 거리/각도 추가
```python
distance = sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
angle = arctan2(end_y - start_y, end_x - start_x)
features = [distance_norm, angle_norm, time_diff_norm, is_successful_prev]
```

### Option 3: 혼합
```python
features = [start_x_norm, start_y_norm, end_x_norm, end_y_norm,
            distance_norm, log_time_diff]
```

---

## 하이퍼파라미터 튜닝

```python
from optuna import create_study
from optuna.samplers import TPESampler

def objective(trial):
    lstm1_units = trial.suggest_int('lstm1_units', 32, 256, step=32)
    lstm2_units = trial.suggest_int('lstm2_units', 16, 128, step=16)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.1)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)

    model = Sequential([
        Input(shape=(5, 4)),
        LSTM(lstm1_units, return_sequences=True, dropout=dropout_rate),
        LSTM(lstm2_units, dropout=dropout_rate),
        Dense(16, activation='relu'),
        Dense(2, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                       batch_size=64, epochs=30, verbose=0)

    return history.history['val_accuracy'][-1]

study = create_study(direction='maximize', sampler=TPESampler())
study.optimize(objective, n_trials=20)
```

---

## 주의사항

### 반드시 피할 사항
- ✗ 에피소드 경계 무시 (Pass 시퀀스가 에피소드를 넘으면 안 됨)
- ✗ 좌표 정규화 누락 (스케일 차이로 인한 학습 불안정)
- ✗ 시간 정보 무시 (Pass는 시간 순서가 있는 시계열 데이터)
- ✗ 클래스 불균형 무시 (Successful 86% → class_weight 필수)

### 필수 체크사항
- ✓ 데이터 정규화 (입력 [0, 1])
- ✓ 에피소드 검증 (Pass >= 6)
- ✓ 시간 순서 정렬 (sort_by('time_seconds'))
- ✓ 훈련/검증/테스트 분할 (에피소드 기준)
- ✓ 클래스 가중치 설정

---

## 예상 성능

| 모델 | Accuracy | Precision | Recall | F1 |
|------|----------|-----------|--------|-----|
| 기본 LSTM | 85-87% | 0.82-0.85 | 0.80-0.85 | 0.83-0.85 |
| 개선 LSTM | 88-90% | 0.86-0.88 | 0.85-0.88 | 0.86-0.88 |
| 고급 모델 | 90-92% | 0.88-0.90 | 0.88-0.90 | 0.89-0.91 |

---

**최종 정리:** 114,602개의 충분한 시퀀스 샘플로 LSTM 모델 구축 가능. 기본 구조 (LSTM 64→32)부터 시작하여 하이퍼파라미터 튜닝 진행 권장.
