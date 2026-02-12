# Data Leakage 검증 보고서

> **작성일:** 2025-12-15
> **목적:** Episode 독립성 검증 및 대회 규정 준수 확인
> **동기:** Nikhil Mishra의 RedBus 대회 사례에서 배운 교훈 적용

---

## 🎯 검증 목적

### Nikhil의 사례 (RedBus 대회, 400+ 참가자 중 우승)

**문제:**
3번 연속 실패 - Claude Code가 temporal constraint를 위반하여 미래 데이터 사용

**원인:**
```python
# 잘못된 코드 (실패 3회)
transactions_df.filter(...)  # 15일 이후 데이터 필터링 실수

# 올바른 코드 (1등)
trans_filt = transactions_df.filter((pl.col('dbd') >= 15))  # 명시적 필터링
```

**교훈:**
- **"Assume you're working with a smart junior engineer who needs explicit constraints"**
- Data Leakage는 성능을 망칠 수 있는 치명적 실수
- 규칙을 명시적으로 코드에 반영해야 함

---

## 📋 대회 규정 (K리그 패스 예측)

**핵심 규칙:**

> "모든 예측은 game_id-episode 단위로 독립적으로 수행되어야 합니다.
> 예측은 해당 에피소드 내부의 시퀀스 데이터만을 입력으로 사용하여야 하며,
> 다른 에피소드(동일 경기 내 다른 episode 포함)의 데이터를 활용한 추론은 금지됩니다."

**해석:**

| 허용 ✅ | 금지 ❌ |
|---------|---------|
| Episode 내부 시퀀스 (start_x, start_y, prev_dx, ...) | 다른 episode의 정보 |
| Train에서 배운 패턴 (통계, 가중치) | 동일 game_id의 다른 episode 정보 |
| Episode 독립적 피처 (goal_distance, ...) | Test episode 간 정보 공유 |

---

## ✅ Zone 6x6 모델 검증

**파일:** `code/models/best/model_safe_fold13.py`

### 1. Training 데이터 처리

```python
# Line 61-62: Episode별 마지막 pass 추출
train_last = train_df.groupby('game_episode').last()

# Line 132-178: Cross-validation
gkf = GroupKFold(n_splits=5)
for fold, (train_idx, val_idx) in enumerate(gkf.split(train_last, groups=game_ids)):
    train_fold = train_last.iloc[train_idx]

    # Zone+Direction별 통계 계산
    stats = train_fold_temp.groupby('key').agg({
        'delta_x': 'median',
        'delta_y': 'median',
        'game_episode': 'count'
    })
```

**분석:**
- ✅ `groupby('game_episode')`: Episode별 독립 처리
- ✅ GroupKFold: Game-level 분리 (같은 game의 episode는 같은 fold)
- ✅ 통계 계산: 여러 episode의 패턴 학습 (정상)

### 2. Test 예측

```python
# Line 297-360: Test 예측
for model in models:
    # 전체 train에서 통계 계산 (학습된 패턴)
    stats = train_temp.groupby('key').agg({
        'delta_x': 'median',
        'delta_y': 'median',
        'game_episode': 'count'
    })

    # 각 test episode 독립 예측
    def predict_row(row):
        key = row['key']  # 이 episode의 zone+direction
        dx = stats.loc[key, 'delta_x']  # train에서 배운 패턴
        pred_x = np.clip(row['start_x'] + dx, 0, 105)  # 이 episode의 좌표
        return pred_x
```

**분석:**
- ✅ Train 통계: 학습된 패턴 (Data Leakage 아님)
- ✅ Episode 독립 예측: 각 row는 하나의 episode
- ✅ Episode 내부 정보만 사용: start_x, start_y, zone, direction

### 결론: Zone 6x6 모델 ✅ Data Leakage 없음

---

## ✅ LSTM 모델 검증

**파일:**
- `code/models/archive/lstm/v3/lstm_data_preprocessing_v3_full.py`
- `code/models/archive/lstm/v5/train_lstm_v5.py`
- `code/models/archive/lstm/v5/predict_test_v5.py`

### 1. Preprocessing (피처 생성)

```python
# Line 66-90: create_features_v3()
def create_features_v3(df):
    # 이전 패스 (Episode별)
    df['prev_dx'] = df.groupby('game_episode')['dx'].shift(1).fillna(0)
    df['prev_dy'] = df.groupby('game_episode')['dy'].shift(1).fillna(0)

    # Cumulative (Episode별)
    df['cumulative_dx'] = df.groupby('game_episode')['dx'].cumsum()
    df['cumulative_dy'] = df.groupby('game_episode')['dy'].cumsum()

    # Pass count (Episode별)
    df['pass_count'] = df.groupby('game_episode').cumcount() + 1

    return df
```

**분석:**
- ✅ 모든 피처가 `groupby('game_episode')`로 계산
- ✅ Episode 간 정보 공유 없음
- ✅ 각 episode 내부의 시퀀스 정보만 사용

### 2. Sequence 생성

```python
# Line 127-203: create_full_episode_sequences()
def create_full_episode_sequences(episodes, max_length, feature_cols, include_target=True):
    sequences = []
    targets = []

    for episode_id, group in episodes:  # 각 episode 독립 처리
        # 전체 시퀀스 features
        features = group[feature_cols].values

        # Cumulative forward fill (Episode 내부)
        if len(features) > 1:
            features[-1, 8] = features[-2, 8]  # cumulative_dx
            features[-1, 9] = features[-2, 9]  # cumulative_dy

        # Input: 마지막 제외 모든 pass
        input_seq = features[:-1]

        # Target: 마지막 pass
        target = [group.iloc[-1]['delta_x'], group.iloc[-1]['delta_y']]

        sequences.append(input_seq)
        targets.append(target)
```

**분석:**
- ✅ `for episode_id, group in episodes`: Episode별 독립 처리
- ✅ Cumulative forward fill: Episode 내부 정보만 사용
- ✅ 다른 episode 정보 사용 안 함

### 3. Training

```python
# Line 76-86: GroupKFold (train_lstm_v5.py)
game_ids = np.array([ep_id.split('_')[0] for ep_id in train_episode_ids])
gkf = GroupKFold(n_splits=3)
folds = list(gkf.split(X_train, y_train, groups=game_ids))

# Line 132-200: Training loop
for fold_idx, (train_idx, val_idx) in enumerate(folds):
    # 각 episode를 독립적으로 학습
    for X_batch, y_batch, lengths_batch, _ in train_loader:
        output = model(X_batch, lengths_batch)
```

**분석:**
- ✅ GroupKFold: Game-level 분리 (Zone 6x6와 동일)
- ✅ 각 episode를 독립적인 sample로 학습
- ✅ Variable length sequence 처리 (lengths_batch)

### 4. Test 예측

```python
# Line 32-44: 데이터 로드 (predict_test_v5.py)
X_test = np.load('X_test_lstm_v3.npy')  # Preprocessing에서 생성
starts_test = np.load('starts_test_lstm_v3.npy')

# Line 71-93: 예측
for fold in range(3):
    model.load_state_dict(torch.load(f'best_model_v5_fold{fold}.pth'))

    for X_batch, lengths_batch, _ in test_loader:
        output = model(X_batch, lengths_batch)  # 각 episode 독립 예측

# 실제 좌표로 변환
pred_end_x = np.clip(starts_test[:, 0] + final_preds[:, 0], 0, 105)
pred_end_y = np.clip(starts_test[:, 1] + final_preds[:, 1], 0, 68)
```

**분석:**
- ✅ Preprocessing에서 이미 episode별로 독립 처리한 데이터
- ✅ 각 episode의 start_x, start_y만 사용
- ✅ 다른 episode 정보 사용 안 함

### 결론: LSTM v3/v5 모델 ✅ Data Leakage 없음

---

## 📊 검증 요약

| 모델 | Episode 독립성 | Data Leakage | 대회 규정 |
|------|----------------|--------------|-----------|
| **Zone 6x6** | ✅ 완벽 | ✅ 없음 | ✅ 준수 |
| **LSTM v3** | ✅ 완벽 | ✅ 없음 | ✅ 준수 |
| **LSTM v5** | ✅ 완벽 | ✅ 없음 | ✅ 준수 |

### 공통 특징

**모든 모델이:**

1. ✅ Episode별로 독립적으로 피처 생성 (`groupby('game_episode')`)
2. ✅ Episode별로 독립적으로 예측
3. ✅ 다른 episode의 정보 사용 안 함
4. ✅ Train에서 배운 패턴(통계/가중치)만 사용
5. ✅ GroupKFold로 game-level 분리

---

## 🆚 Nikhil 사례와 비교

| 항목 | Nikhil (RedBus) | 우리 (K리그) |
|------|-----------------|--------------|
| **Constraint** | Temporal (15일 이후만) | Episode independence |
| **위반 여부** | ❌ 3회 실패 (미래 데이터 사용) | ✅ 완벽 준수 |
| **원인** | 필터링 로직 실수 | 애초에 독립적으로 설계 |
| **해결** | 명시적 필터링 추가 | 이미 구현됨 ✅ |

### 우리가 안전한 이유

```python
# Nikhil의 문제: Temporal constraint
transactions_df.filter(...)  # 실수: 미래 데이터 포함

# 우리의 설계: Episode independence (자연스럽게 구현)
df.groupby('game_episode')  # 모든 피처가 episode별
for episode_id, group in episodes:  # 각 episode 독립 처리
```

**핵심 차이:**
- Nikhil: 명시적 제약(15일)을 코드에 반영 실패
- 우리: Episode 구조 자체가 독립성을 강제

---

## ⚠️ 향후 주의사항

### 새 모델 개발 시 체크리스트

**필수 확인:**

- [ ] 피처 생성 시 `groupby('game_episode')` 사용
- [ ] Train/Test 동일한 방식으로 처리
- [ ] 예측 시 각 episode 독립적으로 처리
- [ ] 다른 episode 정보 사용 안 함

**금지 사항:**

```python
# ❌ 금지: 다른 episode 정보 사용
train_df['other_episode_info'] = train_df.groupby('game_id')['feature'].transform('mean')

# ❌ 금지: Test episode 간 정보 공유
test_df['global_avg'] = test_df['feature'].mean()  # 모든 test episode 평균

# ✅ 허용: Episode 내부 정보
train_df['cumulative'] = train_df.groupby('game_episode')['feature'].cumsum()

# ✅ 허용: Train에서 배운 패턴
stats = train_df.groupby('zone').agg({'delta_x': 'median'})
```

### CLAUDE.md에 추가할 제약 조건

```markdown
## DATA LEAKAGE 방지 (최우선!)

### Episode 독립성 규칙

**ALWAYS:**
- Predict each episode INDEPENDENTLY
- Use only episode-internal sequence data
- Use patterns learned from training data

**NEVER:**
- Use data from other episodes (even same game_id)
- Share information between test episodes
- Access future data within episode

### Code Template

```python
# ✅ 올바른 피처 생성
df['feature'] = df.groupby('game_episode')['col'].transform(...)

# ✅ 올바른 예측
for episode_id, group in test_df.groupby('game_episode'):
    pred = model.predict(group)  # 각 episode 독립 예측
```
```

---

## 🎯 최종 결론

### 검증 결과

```
✅ 모든 기존 모델이 Episode 독립성을 완벽히 유지
✅ Data Leakage 없음
✅ 대회 규정 준수
✅ Nikhil의 교훈을 이미 반영한 설계
```

### 안전성 평가

**Zone 6x6 모델:**
- Episode 독립성: ✅ 완벽
- 위험도: 🟢 매우 낮음
- 설명 가능성: ✅ 높음 (단순 통계)

**LSTM v3/v5 모델:**
- Episode 독립성: ✅ 완벽
- 위험도: 🟢 매우 낮음
- 설명 가능성: ⚠️ 중간 (Neural Network)

### 향후 전략

1. **새 모델 개발 시:**
   - CLAUDE.md에 명시적 제약 조건 추가
   - Episode 독립성 체크리스트 준수
   - 검증 스크립트 작성

2. **기존 모델:**
   - 추가 수정 불필요 (이미 안전)
   - 제출 전 최종 검증만 수행

3. **문서화:**
   - 이 보고서를 향후 참고 자료로 유지
   - 새 팀원에게 공유

---

## 📚 참고 자료

- **Nikhil 사례:** "I beat 400+ data scientists using an AI that kept trying to cheat" by Nikhil Mishra
- **대회 규정:** `docs/COMPETITION_INFO.md`
- **코드 위치:**
  - Zone 6x6: `code/models/best/model_safe_fold13.py`
  - LSTM preprocessing: `code/models/archive/lstm/v3/lstm_data_preprocessing_v3_full.py`
  - LSTM training: `code/models/archive/lstm/v5/train_lstm_v5.py`
  - LSTM prediction: `code/models/archive/lstm/v5/predict_test_v5.py`

---

**작성자:** Claude Sonnet 4.5
**검증 일자:** 2025-12-15
**다음 검토:** 새 모델 개발 시

---

*"Assume you're working with a smart junior engineer who needs explicit constraints."*
*- Nikhil Mishra*
