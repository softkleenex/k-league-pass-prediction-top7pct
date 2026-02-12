# 대회 우승자/참가자 전략 분석

> **출처:** Medium 글 분석
> **작성일:** 2025-12-15
> **목적:** 다른 대회 참가자들의 경험에서 배우기

---

## 📚 분석한 글

### 1. My First Kaggle Competition - LLM Classification Finetuning

**저자:** Carla Cotas
**링크:** https://medium.com/@carlacotas/my-first-kaggle-competition-llm-classification-finetuning-476db368b389
**대회:** Kaggle LLM Classification Finetuning
**성과:** 첫 참가, 10주 챌린지 완료

### 2. I Beat 400+ Data Scientists Using an AI That Kept Trying to Cheat ⭐ 필독!

**저자:** Nikhil Mishra (Kaggle Grandmaster, 40+ AI 대회 우승)
**링크:** https://medium.com/@devnikhilmishra/i-beat-400-to-win-lakhs-data-scientists-using-an-ai-that-kept-trying-to-cheat-fcb7add97d8a
**대회:** RedBus 해커톤
**성과:** 우승 (400+ 참가자, 상금 50만 루피)
**도구:** Claude Code

---

## 🎯 대회 개요

### Kaggle LLM Classification Finetuning

**목표:**
- Chatbot Arena 대화에서 사용자 선호도 예측
- 두 LLM의 응답 중 어느 것을 선호할지 예측
- 3-class classification: model_a / model_b / tie

**데이터:**
- train.csv: id, model_a/b, prompt, response_a/b, winner
- test.csv: id, prompt, response_a/b
- 평가 지표: **Log Loss**

**챌린지:**
- Position bias (첫 번째 응답 선호)
- Verbosity bias (장황한 응답 선호)
- Self-enhancement bias (자기 홍보)

---

## 🔬 저자의 접근법

### 1단계: 데이터 이해 (Week 1)

```python
# 데이터 로드
training = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 기본 탐색
training.head(10)
training.tail(10)
```

**소요 시간:** 1주
**어려움:** Kaggle 플랫폼 익숙해지기
**해결:** 토론 포럼 활용

### 2단계: 데이터 클리닝 (Week 2-3)

**텍스트 전처리:**

```python
def clean_text(text):
    # 소문자 변환
    text = text.lower()

    # 숫자 및 특수문자 제거
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)

    # 토큰화
    text = nltk.word_tokenize(text)

    # 불용어 제거
    stop_words = set(stopwords.words('english'))
    text = [word for word in text if word not in stop_words]

    # 표제어 추출 (Lemmatization)
    lemmatizer = nltk.WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text]

    return ' '.join(text)

# 적용
training["prompt"] = training["prompt"].apply(clean_text)
training["response_a"] = training["response_a"].apply(clean_text)
training["response_b"] = training["response_b"].apply(clean_text)
```

**주요 발견:**
- ID 중복 없음 ✅
- NaN/Null 값 없음 ✅
- 64개 LLM 모델
- 5,743개 중복 prompt (정상, 다른 모델 조합)

**실수 및 교훈:**
- **노트북 크래시** → 자주 저장하기!
- Matplotlib 서브플롯 실수 → 각각 따로 그리기

### 3단계: 데이터 탐색 (Week 2-3)

**시각화:**
- LLM 분포 (model_a, model_b)
- Winner 분포
- 특별한 패턴 발견 안 됨

**핵심:**
- LLM 정보는 test 데이터에 없음
- LLM별 분석 중단

### 4단계: 피처 엔지니어링 (Week 4)

**TF-IDF 벡터화:**

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# 중요: max_features 설정 (메모리 제한!)
vectorizer = TfidfVectorizer(max_features=150)

# 각 텍스트 필드 벡터화
vectorizer_prompt = vectorizer.fit_transform(training["prompt"])
vectorizer_response_a = vectorizer.fit_transform(training["response_a"])
vectorizer_response_b = vectorizer.fit_transform(training["response_b"])

# 피처 결합
train_X = np.concatenate((
    temp_prompt.toarray(),
    temp_response_a.toarray(),
    temp_response_b.toarray()
), axis=1)

# 타겟
train_y = training["winner"].values
```

**Critical Issue:**
- **메모리 제한!** max_features 없이 실행 → 크래시
- **해결:** max_features=150 설정

### 5단계: 모델 학습 (Week 5-6)

**Logistic Regression:**

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    max_iter=500,
    multi_class='multinomial',  # 3-class
    solver='saga'               # 대규모 데이터셋에 빠름
)

model.fit(train_X, train_y)
```

**실행 시간 측정:**

```python
from datetime import datetime

start = datetime.now()
# ... 모델 학습 ...
end = datetime.now()

execution_time = (end - start).total_seconds() / 60
print(f"Execution time: {execution_time} minutes")
```

### 6단계: 모델 평가 (Week 5-6)

```python
# Train/Validation Split
train_X_train, train_X_val, train_y_train, train_y_val = train_test_split(
    train_X, train_y, test_size=0.2, random_state=42
)

# 평가
value_y_predict = model.predict(train_X_val)
value_y_probabilities = model.predict_proba(train_X_val)

# Confusion Matrix
cm = confusion_matrix(train_y_val, value_y_predict)

# Accuracy
score = model.score(train_X_val, train_y_val)  # < 50%

# Precision & Recall
macro_precision = precision_score(train_y_val, value_y_predict, average='macro')
macro_recall = recall_score(train_y_val, value_y_predict, average='macro')

# Log Loss (핵심 지표!)
model_log_loss = log_loss(train_y_val, value_y_probabilities)
# Result: 1.05 (baseline: log(1/3) = 1.10)
```

**핵심 인사이트:**
- Accuracy < 50% → 처음엔 혼란
- **Log Loss가 더 중요!** 1.05 < 1.10 (good!)
- 토론 포럼에서 Log Loss 중요성 학습

### 7단계: 제출 (Week 6)

```python
# 예측
test_X = np.concatenate((
    temp_test_prompt.toarray(),
    temp_test_response_a.toarray(),
    temp_test_response_b.toarray()
), axis=1)

value_test_y_probabilities = model.predict_proba(test_X)

# 제출 파일
output = pd.DataFrame({
    'id': test.id,
    'winner_model_a': value_test_y_probabilities[:, 0],
    'winner_model_b': value_test_y_probabilities[:, 1],
    'winner_tie': value_test_y_probabilities[:, 2]
})

output.to_csv('submission.csv', index=False)
```

**첫 제출 점수:** 1.11623 (하위권)

---

## 💡 핵심 교훈

### 1. 플랫폼 익숙해지기

```
✅ Kaggle/DACON 토론 포럼 적극 활용
✅ 노트북 자주 저장 (크래시 대비)
✅ 샘플 제출 파일 확인
```

### 2. 텍스트 데이터 처리

```
✅ 전처리 필수 (소문자, 특수문자 제거, 불용어, 표제어)
✅ TF-IDF 같은 벡터화 기법
✅ 메모리 제한 주의 (max_features 설정)
```

### 3. 모델 선택

```
✅ Multi-class → Logistic Regression (multinomial)
✅ 대규모 데이터 → solver='saga'
✅ 실행 시간 측정으로 병목 파악
```

### 4. 평가 지표 이해

```
⚠️ Accuracy가 낮아도 괜찮을 수 있음
✅ 대회의 핵심 지표 집중 (Log Loss, Euclidean Distance 등)
✅ 토론 포럼에서 지표 해석 학습
```

### 5. 반복적 개선

```
✅ 첫 제출이 하위권이어도 학습 과정이 중요
✅ 토론 포럼에서 다른 참가자 전략 학습
✅ 작은 변경의 영향 테스트
```

---

## 🎯 우리 K리그 대회에 적용

### 직접 적용 가능

1. **토론/코드 공유 활용**
   ```
   현재: 거의 활용 안 함
   개선: DACON 토론 게시판, 코드 공유 적극 확인
   ```

2. **실행 시간 측정**
   ```python
   # 병목 구간 파악
   start = datetime.now()
   # ... 학습 ...
   end = datetime.now()
   print(f"Time: {(end-start).total_seconds()} sec")
   ```

3. **메모리 최적화**
   ```
   유사 사례: LSTM v5 파라미터 74.6% 감소
   교훈: 메모리만이 아니라 성능도 고려
   ```

4. **체계적 단계별 접근**
   ```
   Week 1: 데이터 이해 ✅
   Week 2-3: 클리닝 & 탐색 ✅
   Week 4: 피처 엔지니어링 ✅
   Week 5-6: 모델링 ✅

   우리: 이미 잘하고 있음! ✅
   ```

### 차이점 & 배울 점

| 항목 | Carla의 대회 | 우리 대회 | 적용 |
|------|--------------|-----------|------|
| **데이터** | 텍스트 (LLM 응답) | 수치 (좌표) | - |
| **평가** | Log Loss | Euclidean Distance | 지표 이해 중요 |
| **접근** | TF-IDF + Logistic Regression | Zone 통계, LSTM | 단순함 우선 |
| **기간** | 10주 챌린지 | 6주 (43일) | 집중 필요 |
| **첫 제출** | 하위권 (1.11623) | 중하위권 (16.36) | 괜찮음! |

### 우리가 더 잘하는 점

```
✅ 체계적인 문서화 (28회 실험 기록)
✅ CV/Public Gap 분석 (Sweet Spot 발견)
✅ 14회 하이퍼파라미터 완전 탐색
✅ 실패 분석 문서화 (LSTM 5개 버전)
```

### 우리가 개선할 점

```
❌ 토론 게시판 활용 부족
   → DACON 토크, 코드 공유 확인

❌ 다른 참가자 접근법 분석 부족
   → 상위권 공개 노트북 학습

❌ 작은 변경 테스트 부족
   → Week 4-5에 작은 실험들
```

---

## 📋 실행 계획

### Week 2-3 (현재)

```
✅ 문서화 완료
□ DACON 토론 게시판 확인 (매일 10분)
□ 상위권 코드 공유 1-2개 분석
□ 다른 참가자 접근법 요약
```

### Week 4-5 (후반전)

```
□ 학습한 전략 테스트 (작은 변경)
□ 실행 시간 측정으로 병목 파악
□ 토론에서 배운 팁 적용
□ 제출 2-4회/일
```

---

## 🔗 참고 자료

### Carla의 자료

- **Medium 글:** [링크](https://medium.com/@carlacotas/my-first-kaggle-competition-llm-classification-finetuning-476db368b389)
- **Kaggle 노트북:** 공개됨
- **GitHub:** 공개됨

### 우리 자료

- **대회 토론:** https://dacon.io/competitions/official/236647/talkboard
- **코드 공유:** https://dacon.io/competitions/official/236647/codeshare

---

## 📝 핵심 메시지

```
"첫 제출이 하위권이어도 괜찮다.
 10주 챌린지를 완료하며 배운 것이 더 중요하다.

 우리도 마찬가지다.
 현재 241위지만, 6주간 체계적으로 접근했고
 많은 것을 배웠다.

 이제 토론을 활용하고, 다른 참가자에게 배우며
 후반전을 준비하자."
```

---

## 🏆 Nikhil의 우승 전략 (RedBus 해커톤)

### 대회 개요

**문제:**
- 15일 후 버스 좌석 예약 현황 예측
- 평가 지표: RMSE
- 제약: **15일 전 데이터만 사용 가능**

**핵심 도전:**
- 사람들은 버스표를 막판에 예약 (20% only 며칠 전)
- 수학으로 인간의 자발성 예측해야 함
- **시간적 제약 위반 = Data Leakage = 실패**

---

### 🚨 Claude Code의 3번 연속 Data Leakage 실패!

#### 1-3차 제출: 참패

**문제:**
```python
# Claude Code가 만든 잘못된 피처
"향후 7일간 평균 예약 건수"  # ❌ 미래 데이터 사용!
"주중 최대 예약 요일"          # ❌ 여정 날짜 이후 예약까지 포함!
```

**증상:**
- Validation RMSE: 훌륭 ✅
- Public Leaderboard: 형편없음 ❌

**원인:**
- Claude Code가 **모든 데이터**를 사용해서 피처 생성
- 시간적 제약 완전히 무시
- **전형적인 Data Leakage**

#### 깨달음

> "Claude Code는 모든 데이터를 활용하여 피처를 개발했습니다.
> 여기에는 **여정 날짜 이후에 발생한 예약까지 포함**되었습니다.
>
> 전형적인 데이터 유출 사례. 경쟁사 자멸 행위."

---

### ✅ 해결: 명시적 시간 제약

#### 모든 것을 구한 한 줄

```python
# The line that saved everything
trans_filt = transactions_df.filter((pl.col('dbd') >= 15))

# Then ALL feature engineering on this filtered data
features = trans_filt.group_by(['route', 'source', 'destination']).agg([
    pl.col('seats_booked').mean().alias('avg_seats'),
    pl.col('seats_booked').std().alias('std_seats'),
    # ... eventually 35,000+ features
])
```

**핵심:**
```
⚠️ 피처 엔지니어링 **전에** 시간 제약 필터링!
✅ 35,000+ 피처 모두 15일 전 데이터만 사용
```

---

### 💡 Claude Code = 매우 똑똑한 Junior Engineer

#### Claude Code가 잘하는 것

```
✅ 보일러플레이트 코드 초고속 작성
✅ 잘 정의된 구체적 작업 구현
✅ 실험 프레임워크 구축
✅ 최신 라이브러리 지식 (TabDPT 같은 것)
```

#### Claude Code가 못하는 것

```
❌ ML 문제의 암묵적 제약 이해
❌ 시간적 검증 및 Data Leakage 방지
❌ 최적화된 코드 작성 (pandas 기본, Polars 10배 빠름)
❌ 비슷한 라이브러리 구문 혼동
```

---

### 📝 CLAUDE.md: 게임 체인저

#### Nikhil의 CLAUDE.md

```markdown
# CRITICAL CONSTRAINTS:
- ALWAYS filter data with temporal constraints BEFORE feature creation
- Use only data from >= 15 days before prediction date
- No data leakage: future cannot predict past

# CODE PREFERENCES:
- Use Polars for large datasets, not pandas
- Iterate on smaller faster code
```

**효과:**
> "Claude Code가 도메인 전문 지식을 미리 로드하자
> 엄청나게 효과적으로 변했습니다."

---

### 🔬 피처 엔지니어링 폭발

#### 35,000+ 피처 생성

**Nikhil의 지시:**
```
"시간적 패턴을 포착하는 temporal 피처를 만들어줘.
중요: 모든 피처는 시간 제약을 준수해야 함.
df_filt를 베이스로 사용.
휴일, 요일 효과, 계절 트렌드를 생각해봐."
```

**Claude Code 결과:**
```python
✅ Cyclical encoding (sine/cosine 변환)
✅ 휴일 근접도 피처 (인도 특정 휴일)
✅ 10개 다른 시간 구간의 롤링 윈도우 통계
✅ 모멘텀 점수, 효율성 비율 같은 2차 피처
```

**핵심:**
```
제약은 명시적으로
구현 세부사항은 Claude에게
```

---

### 🧪 실험 시스템

#### 빠른 반복 프레임워크

```python
FEATURE_CONFIGS = [
    {'name': '1K_features', 'top_n_features': 1000},
    {'name': '2K_features', 'top_n_features': 2000},
    {'name': '3K_features', 'top_n_features': 3000},
    {'name': '6K_features', 'top_n_features': 6000}
]
```

**전략:**
1. **10% 데이터로 빠른 실험**
2. 승자만 full-scale 학습
3. 계산 시간 절약, 10배 빠른 아이디어 테스트

---

### 🎯 TabDPT: 서프라이즈 무기

**Claude Code 제안:**
> "TabDPT는 tabular 데이터에서 훌륭한 결과를 보입니다.
> 통합해드릴까요?"

**결과:**
- RMSE 455.68
- 단일 GBM 모델보다 훨씬 좋음

**교훈:**
```
✅ AI는 당신이 (아직) 모르는 기법을 알고 있음
```

---

### 🏅 최종 앙상블

```python
# 가중 앙상블
75% weight: Gradient Boosting ensemble (12 models)
25% weight: TabDPT predictions
```

**전략:**
- 전통적 모델의 안정성 활용
- Transformer 혁신 결합

---

### 💼 Nikhil의 실제 워크플로우

#### 1. CLAUDE.md 먼저 설정

```markdown
제약 조건, 검증 전략, 선호 라이브러리, 흔한 실수 문서화
```

#### 2. 빠른 실험 루프 구축

```
10% 데이터 샘플로 빠른 반복
→ 승자만 full-scale 학습으로 승격
```

#### 3. 구체적인 요청

**나쁨:**
> "우승 솔루션을 만들어줘"

**좋음:**
> "7, 14, 30일 롤링 통계를 만들어줘. 시간 제약 준수해야 함."

#### 4. 3단계 파이프라인

```
탐색 (Exploration):
  Claude가 초기 피처 빠르게 생성

실험 (Experimentation):
  작은 데이터로 접근법 테스트

프로덕션 (Production):
  검증된 접근만 스케일업
```

---

## 🎯 우리 K리그 대회에 적용 (중요!)

### 🚨 Data Leakage 방지 - 최우선!

**우리 대회 규칙:**
```
❌ 모든 예측은 game_id-episode 단위로 독립적
❌ 다른 에피소드 데이터 사용 금지 (동일 경기 내 다른 episode 포함)
```

**Nikhil의 교훈 적용:**
```python
# 우리도 필요한 필터!
# 각 episode는 독립적으로 예측
# 다른 episode 정보 절대 사용 금지

# 예: LSTM 학습 시
for episode_id in episodes:
    episode_data = data[data['episode'] == episode_id]  # ✅ 이 episode만
    # NOT: data[data['game_id'] == game_id]  # ❌ 같은 경기 전체
```

**우리가 이미 위반했을 가능성:**
```
⚠️ 전체 패스 학습 (356K) - 다른 episode 정보 사용?
⚠️ 확인 필요: episode 독립성 보장되는지?
```

### CLAUDE.md 작성 (즉시!)

#### 우리의 CLAUDE.md

```markdown
# K리그 패스 좌표 예측 - CRITICAL CONSTRAINTS

## DATA LEAKAGE 방지 (최우선!)
- ALWAYS predict each episode INDEPENDENTLY
- NEVER use data from other episodes (even same game_id)
- Filter by episode_id BEFORE any feature engineering
- No future information: only use passes BEFORE the target pass

## 대회 규칙
- No API calls (OpenAI, Gemini, etc.)
- No external data
- Only pretrained models from before 2025.11.23
- Local execution only

## 코드 선호
- Use Polars for large datasets
- Measure execution time for bottlenecks
- Document experiments in EXPERIMENT_LOG.md

## 검증 전략
- CV Sweet Spot: 16.27-16.34
- CV < 16.27 = Overfitting (Gap explosion)
- Target: Gap < 0.1

## 금지 사항
- NO LSTM (4번 실패)
- NO data augmentation (Flip, Rotation)
- NO CV < 16.27 추구
```

### 빠른 실험 시스템

**10% 데이터 샘플링:**
```python
# Nikhil처럼
sample_episodes = train.sample(frac=0.1, random_state=42)

# 빠른 실험
for config in CONFIGS:
    model = train_model(sample_episodes, config)
    score = evaluate(model)
    if score < threshold:
        # Full scale 학습
        full_model = train_model(train, config)
```

### TabDPT 같은 새 기법 탐색

```
□ DACON 토론에서 상위권 기법 확인
□ Tabular Transformer (TabDPT, TabNet, FT-Transformer)
□ 작은 실험으로 먼저 테스트
```

---

## 📊 두 글 비교

| 항목 | Carla (첫 참가) | Nikhil (우승) | 우리 |
|------|-----------------|---------------|------|
| **경험** | 초보 | Kaggle Grandmaster | 중급 |
| **도구** | 수동 코딩 | Claude Code | 수동 + 일부 AI |
| **핵심 실수** | 지표 이해 부족 | Data Leakage (3번!) | TBD (확인 필요) |
| **해결** | 토론 포럼 | CLAUDE.md | ? |
| **결과** | 하위권 (학습) | 우승 | 241위 (학습 중) |
| **기간** | 10주 | ? | 6주 (진행 중) |

---

## 🚨 즉시 실행 항목

### 1. Data Leakage 확인 (최우선!)

```
□ Zone 6x6 모델: episode 독립성 확인
□ 전체 패스 학습 (356K): 다른 episode 정보 사용했나?
□ LSTM: episode 독립 예측 보장되는지?
```

**코드 리뷰:**
```python
# code/models/best/model_safe_fold13.py
# 이 모델이 episode별로 독립적으로 예측하는지 확인!
```

### 2. CLAUDE.md 작성 (오늘)

```
□ 위의 템플릿 사용
□ 대회 규칙 명시
□ Data Leakage 방지 규칙
□ 코드 선호도
```

### 3. 실험 시스템 개선

```
□ 10% 샘플로 빠른 테스트 프레임워크
□ 실행 시간 측정 추가
□ 승자만 full-scale
```

### 4. 토론/코드 활용

```
□ DACON 토론 매일 10분
□ 상위권 코드 1-2개 분석
□ 새로운 기법 탐색 (TabNet 등)
```

---

## 📝 핵심 메시지 (업데이트)

```
"Nikhil의 가장 큰 교훈: Data Leakage

3번 제출, 3번 모두 실패. Claude Code가 미래 데이터를 사용했기 때문.
검증 점수는 훌륭했지만, 리더보드는 형편없었다.

한 줄의 필터링 코드가 모든 것을 바꿨다:
trans_filt = transactions_df.filter((pl.col('dbd') >= 15))

우리도 마찬가지다.
Episode 독립성을 보장하는가?
다른 episode 정보를 사용하지 않았는가?

이것부터 확인하자. 지금 당장."
```

---

**작성일:** 2025-12-15
**업데이트:** Medium 글 2개 분석 완료
**다음:** Data Leakage 확인, CLAUDE.md 작성
