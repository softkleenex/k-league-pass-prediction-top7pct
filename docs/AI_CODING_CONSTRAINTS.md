# AI Coding Constraints

> **목적:** AI(Claude Code)가 코드 작성 시 반드시 지켜야 할 명시적 제약 조건
> **동기:** Nikhil Mishra의 조언 - "Assume you're working with a smart junior engineer who needs explicit constraints"
> **작성일:** 2025-12-15

---

## 🎯 핵심 원칙

```
"Claude Code는 매우 똑똑한 주니어 엔지니어입니다.
하지만 명시적인 제약 조건이 없으면 실수할 수 있습니다.
모든 규칙을 명확하고 구체적으로 작성하세요."
```

---

## 🚨 최우선 규칙: Episode 독립성

### 대회 규정 (절대 위반 금지!)

> "모든 예측은 game_id-episode 단위로 독립적으로 수행되어야 합니다.
> 예측은 해당 에피소드 내부의 시퀀스 데이터만을 입력으로 사용하여야 하며,
> 다른 에피소드(동일 경기 내 다른 episode 포함)의 데이터를 활용한 추론은 금지됩니다."

### ✅ ALWAYS (반드시 해야 할 것)

```python
# 1. Episode별로 피처 생성
df['feature'] = df.groupby('game_episode')['col'].transform(...)

# 2. Episode별로 독립 처리
for episode_id, group in df.groupby('game_episode'):
    process_episode(group)

# 3. Episode 내부 시퀀스만 사용
episode_data = df[df['game_episode'] == target_episode]
features = create_features(episode_data)  # 이 episode만

# 4. Train에서 배운 패턴 사용 (OK!)
stats = train_df.groupby('zone').agg({'delta_x': 'median'})
pred = start_x + stats.loc[zone, 'delta_x']  # 학습된 통계
```

### ❌ NEVER (절대 하지 말아야 할 것)

```python
# 1. 다른 episode 정보 사용
df['avg_across_episodes'] = df.groupby('game_id')['feature'].transform('mean')  # ❌

# 2. Test episode 간 정보 공유
test_df['global_avg'] = test_df['feature'].mean()  # ❌ 모든 test episode 평균

# 3. 동일 game_id의 다른 episode 정보
same_game = df[df['game_id'] == current_game_id]
avg_feature = same_game['feature'].mean()  # ❌ 다른 episode 포함

# 4. Episode 경계 넘는 Rolling/Shift
df['rolling'] = df['feature'].rolling(window=5).mean()  # ❌ episode 경계 무시
```

---

## 📋 피처 생성 규칙

### Template: Episode-Safe Feature Engineering

```python
def create_features_episode_safe(df):
    """
    Episode 독립성을 유지하는 피처 생성
    """
    df = df.copy()

    # ✅ Episode별 Shift (이전 값)
    df['prev_value'] = df.groupby('game_episode')['value'].shift(1).fillna(0)

    # ✅ Episode별 Cumulative
    df['cumulative'] = df.groupby('game_episode')['value'].cumsum()

    # ✅ Episode별 Count
    df['pass_count'] = df.groupby('game_episode').cumcount() + 1

    # ✅ Episode별 통계
    df['episode_mean'] = df.groupby('game_episode')['value'].transform('mean')

    # ✅ 독립적 계산 (episode 무관)
    df['goal_distance'] = np.sqrt((105 - df['x'])**2 + (34 - df['y'])**2)

    return df
```

### 금지 패턴

```python
# ❌ Episode 경계 무시
df['rolling_avg'] = df['value'].rolling(window=5).mean()

# ❌ Game-level aggregation (다른 episode 포함)
df['game_avg'] = df.groupby('game_id')['value'].transform('mean')

# ❌ Global statistics (train+test 혼합)
df['normalized'] = (df['value'] - df['value'].mean()) / df['value'].std()

# ✅ Train-only statistics (OK!)
train_mean = train_df['value'].mean()
df['normalized'] = (df['value'] - train_mean) / train_std
```

---

## 🔄 Train/Test 데이터 처리

### Template: Episode-Safe Train/Test Split

```python
# ✅ Episode별 독립 처리 (Train/Test 동일)
def preprocess_episodes(episodes, stats_from_train=None):
    """
    Args:
        episodes: list of (episode_id, group)
        stats_from_train: 학습된 통계 (Test에만 제공)
    """
    processed = []

    for episode_id, group in episodes:
        # Episode 내부 피처
        features = create_episode_features(group)

        # Train에서 배운 패턴 적용 (Test만)
        if stats_from_train is not None:
            features = apply_learned_stats(features, stats_from_train)

        processed.append((episode_id, features))

    return processed

# Train
train_episodes = list(train_df.groupby('game_episode'))
train_processed = preprocess_episodes(train_episodes)

# Train 통계 학습
train_stats = learn_statistics(train_processed)

# Test (Train 통계 사용)
test_episodes = list(test_df.groupby('game_episode'))
test_processed = preprocess_episodes(test_episodes, stats_from_train=train_stats)
```

### Cross-Validation

```python
from sklearn.model_selection import GroupKFold

# ✅ GroupKFold로 game-level 분리
game_ids = np.array([ep_id.split('_')[0] for ep_id in episode_ids])
gkf = GroupKFold(n_splits=5)

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=game_ids)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # 같은 game_id의 episode는 같은 fold
    # Episode 독립성 유지
```

---

## 🎯 모델 예측 규칙

### Template: Episode-Independent Prediction

```python
def predict_episodes(test_episodes, model, train_stats):
    """
    각 episode를 독립적으로 예측
    """
    predictions = []

    for episode_id, group in test_episodes:
        # ✅ 이 episode의 데이터만 사용
        episode_features = group[feature_cols].values

        # ✅ 이 episode의 start 좌표
        start_x = group.iloc[-1]['start_x']
        start_y = group.iloc[-1]['start_y']

        # ✅ Train에서 배운 패턴 사용
        zone = get_zone(start_x, start_y)
        delta = train_stats.loc[zone]

        # ✅ 예측
        pred_x = np.clip(start_x + delta['x'], 0, 105)
        pred_y = np.clip(start_y + delta['y'], 0, 68)

        predictions.append({
            'game_episode': episode_id,
            'end_x': pred_x,
            'end_y': pred_y
        })

    return pd.DataFrame(predictions)
```

### 금지 패턴

```python
# ❌ Batch 예측에서 정보 공유
test_batch = test_df.iloc[batch_idx]
batch_mean = test_batch['feature'].mean()  # ❌ 여러 episode 정보 혼합
predictions = model.predict(test_batch)

# ✅ 올바른 Batch 예측
for batch in test_batches:
    # Batch 내 각 episode는 독립적
    # Model은 각 sample을 독립적으로 처리
    predictions = model.predict(batch)
```

---

## 🧪 검증 체크리스트

### 새 코드 작성 후 필수 확인

- [ ] **피처 생성:** 모든 피처가 `groupby('game_episode')` 사용?
- [ ] **Train/Test:** 동일한 방식으로 처리?
- [ ] **예측:** 각 episode 독립적으로 예측?
- [ ] **정보 공유:** 다른 episode 정보 사용 안 함?
- [ ] **Cross-validation:** GroupKFold 사용?

### 자가 검증 질문

1. **"이 코드가 다른 episode의 정보를 사용하는가?"**
   - 사용하면 ❌ 위반

2. **"Train과 Test를 다르게 처리하는가?"**
   - 다르면 ❌ 분포 불일치

3. **"Test episode 간 정보를 공유하는가?"**
   - 공유하면 ❌ 위반

4. **"Episode 경계를 넘는 연산이 있는가?"**
   - 있으면 ❌ 위반

---

## 🚫 대회 규칙 위반 금지

### 외부 데이터 금지

```python
# ❌ 금지
import requests
external_data = requests.get('https://api.example.com/data')

# ❌ 금지
weather_df = pd.read_csv('external_weather_data.csv')

# ✅ 허용 (주어진 데이터만)
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
```

### API 호출 금지

```python
# ❌ 금지
import openai
response = openai.ChatCompletion.create(...)

# ❌ 금지
from anthropic import Anthropic
client = Anthropic()

# ✅ 허용 (로컬 모델만)
from transformers import AutoModel
model = AutoModel.from_pretrained('model_name')  # 2025.11.23 이전 버전만
```

### 2025.11.23 이전 모델만 허용

```python
# ✅ 허용
from transformers import AutoModel
model = AutoModel.from_pretrained('bert-base-uncased')  # 2023년 모델

# ❌ 금지
model = AutoModel.from_pretrained('new-model-2025-12')  # 2025.11.23 이후
```

---

## 📝 코드 작성 가이드

### 1. 피처 생성

```python
# Template
def create_features(df):
    """
    CONSTRAINT: Episode 독립성 유지
    """
    df = df.copy()

    # 모든 groupby는 'game_episode' 사용
    df['feature1'] = df.groupby('game_episode')['col1'].transform(...)
    df['feature2'] = df.groupby('game_episode')['col2'].shift(1).fillna(0)

    # 독립적 계산 (episode 무관)
    df['feature3'] = some_calculation(df['col3'])

    return df
```

### 2. 모델 학습

```python
# Template
def train_model(train_df):
    """
    CONSTRAINT: GroupKFold 사용
    """
    # Episode별 처리
    train_episodes = list(train_df.groupby('game_episode'))

    # GroupKFold
    game_ids = np.array([ep_id.split('_')[0] for ep_id in episode_ids])
    gkf = GroupKFold(n_splits=5)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=game_ids)):
        # 학습
        model.fit(X[train_idx], y[train_idx])

    return model
```

### 3. 예측

```python
# Template
def predict(test_df, model, train_stats):
    """
    CONSTRAINT: Episode 독립 예측
    """
    predictions = []

    for episode_id, group in test_df.groupby('game_episode'):
        # 이 episode만 사용
        pred = model.predict(group)

        predictions.append({
            'game_episode': episode_id,
            'end_x': pred[0],
            'end_y': pred[1]
        })

    return pd.DataFrame(predictions)
```

---

## ⚠️ 일반적인 실수 패턴

### 1. Global Statistics

```python
# ❌ 잘못된 예
scaler = StandardScaler()
df['normalized'] = scaler.fit_transform(df[['feature']])  # Train+Test 혼합

# ✅ 올바른 예
scaler = StandardScaler()
scaler.fit(train_df[['feature']])  # Train만

train_df['normalized'] = scaler.transform(train_df[['feature']])
test_df['normalized'] = scaler.transform(test_df[['feature']])
```

### 2. Episode 경계 무시

```python
# ❌ 잘못된 예
df['rolling_avg'] = df['feature'].rolling(window=5).mean()  # Episode 경계 무시

# ✅ 올바른 예
df['rolling_avg'] = df.groupby('game_episode')['feature'].rolling(window=5).mean().reset_index(0, drop=True)
```

### 3. Test Leakage

```python
# ❌ 잘못된 예
all_data = pd.concat([train_df, test_df])
all_data['feature'] = all_data['col'].transform(...)  # Train+Test 혼합

# ✅ 올바른 예
train_df['feature'] = create_features(train_df)
test_df['feature'] = create_features(test_df)  # 동일 함수, 독립 처리
```

---

## 🎓 학습한 교훈 (Nikhil 사례)

### 문제 상황

```python
# Nikhil의 실수 (3번 실패)
# Constraint: 15일 이후 데이터만 사용
transactions_df.filter(...)  # 잘못된 필터링 → 미래 데이터 포함

# 해결 (1등)
trans_filt = transactions_df.filter((pl.col('dbd') >= 15))  # 명시적 필터링
```

### 교훈

1. **명시적 제약 조건:**
   - "15일 이후"를 코드에 명확히 표현: `>= 15`
   - "Episode 독립"을 코드에 명확히 표현: `groupby('game_episode')`

2. **가정하지 말 것:**
   - AI는 암묵적 규칙을 모를 수 있음
   - 모든 제약을 명시적으로 작성

3. **검증 철저히:**
   - 각 단계마다 규칙 준수 확인
   - 제출 전 최종 검증

---

## ✅ 성공 사례 (우리 프로젝트)

### Zone 6x6 모델

```python
# Episode별 독립 처리
train_last = train_df.groupby('game_episode').last()

# GroupKFold
gkf = GroupKFold(n_splits=5)
for fold, (train_idx, val_idx) in enumerate(gkf.split(train_last, groups=game_ids)):
    ...

# Episode 독립 예측
def predict_row(row):
    pred_x = row['start_x'] + stats.loc[row['key'], 'delta_x']
    return pred_x
```

**결과:** ✅ Data Leakage 없음, Public 16.36 (241위)

### LSTM v3/v5 모델

```python
# Episode별 피처
df['prev_dx'] = df.groupby('game_episode')['dx'].shift(1)
df['cumulative'] = df.groupby('game_episode')['dx'].cumsum()

# Episode별 Sequence
for episode_id, group in episodes:
    input_seq = group[feature_cols].values[:-1]
    target = group.iloc[-1][['delta_x', 'delta_y']]
```

**결과:** ✅ Data Leakage 없음, Public 17.29 (255위)

---

## 📚 참고 자료

- **검증 보고서:** `docs/DATA_LEAKAGE_VERIFICATION.md`
- **대회 규정:** `docs/COMPETITION_INFO.md`
- **Nikhil 사례:** `docs/COMPETITION_STRATEGIES_FROM_WINNERS.md`

---

## 🔄 업데이트 이력

- **2025-12-15:** 최초 작성 (Data Leakage 검증 후)
- **다음 업데이트:** 새 모델 개발 시

---

**작성자:** Claude Sonnet 4.5
**목적:** AI가 대회 규칙을 명확히 이해하고 준수하도록 지원

---

*"The one line that saved everything: `trans_filt = transactions_df.filter((pl.col('dbd') >= 15))`"*
*- Nikhil Mishra, RedBus 대회 우승자*
