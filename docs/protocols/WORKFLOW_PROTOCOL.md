# Workflow Protocol: Claude ↔ Gemini 작업 분할

**작성일:** 2025-12-17
**목적:** 효율적인 작업 분할로 각 AI의 강점 최대화

---

## 🎯 핵심 원칙

**Claude (코드 작성 & 설계)**
- 코드 작성 및 구현
- 디버깅 및 에러 수정
- 아키텍처 설계
- 문서 작성
- 전략 수립 및 계획

**Gemini (파일 실행 & 모니터링)**
- Python 스크립트 실행
- 학습 프로세스 모니터링
- 결과 파일 확인 및 리포트
- 로그 분석
- 장시간 실행 작업

---

## 📋 상세 작업 분할

### Claude의 역할 ⭐

**1. 코드 작성:**
```python
# Claude가 작성
- 새로운 피처 추출 스크립트
- 모델 학습 코드
- 평가 및 제출 스크립트
- 유틸리티 함수
```

**2. 디버깅:**
```python
# Claude가 수정
- 에러 메시지 분석
- 버그 수정
- 경로 문제 해결
- Import 오류 수정
```

**3. 아키텍처 & 설계:**
- 실험 구조 설계
- 피처 엔지니어링 전략
- 모델 아키텍처 선택
- 파이프라인 설계

**4. 문서화:**
- README.md 작성
- 주석 및 docstring
- 실험 보고서
- 전략 문서

**5. 전략 수립:**
- Ultrathink / Plan agent 사용
- 우선순위 결정
- 리스크 분석

---

### Gemini의 역할 ⭐

**1. 파일 실행:**
```bash
# Gemini가 실행
python extract_features.py
python train_model.py
python generate_submission.py
```

**2. 모니터링:**
- 학습 진행 상황 체크
- 로그 파일 확인
- CV 결과 추출
- 에러 발생 시 알림

**3. 장시간 작업:**
- 30분+ 학습 프로세스
- 대용량 데이터 처리
- 여러 fold CV 실행
- 하이퍼파라미터 튜닝

**4. 결과 리포트:**
- cv_results.json 읽기
- 주요 메트릭 추출
- 실행 시간 보고
- 성공/실패 여부 판단

---

## 🔄 표준 워크플로우

### 새 실험 시작 시

**Step 1: Claude - 코드 작성**
```
1. exp_XXX 폴더 생성
2. 피처 추출 스크립트 작성
3. 학습 스크립트 작성
4. README.md 작성
5. 디버깅 편의성 고려:
   - 명확한 에러 메시지
   - 진행 상황 로깅
   - 중간 결과 저장
   - 체크포인트 기능
```

**Step 2: Gemini - 실행**
```bash
cd exp_XXX
python step1_extract.py
python step2_train.py
python step3_evaluate.py
```

**Step 3: Gemini - 결과 보고**
```
CV Mean: XX.XX
실행 시간: XX분
에러: 없음/있음
```

**Step 4: Claude - 분석 & 다음 단계**
```
결과 분석
제출 여부 결정
다음 실험 계획
```

---

## 🐛 디버깅 워크플로우

### 에러 발생 시

**Gemini가 할 일:**
1. 에러 메시지 전체 복사
2. 마지막 50줄 로그 제공
3. 실행 환경 정보 (Python 버전, 패키지 등)

**Claude가 할 일:**
1. 에러 분석
2. 코드 수정
3. 수정된 파일 생성
4. 재실행 지시

**예시:**
```
[Gemini 보고]
Error: FileNotFoundError: train.csv
Log: (마지막 50줄)

[Claude 수정]
경로 오류 발견 → 수정
새 파일 생성: extract_features_v2.py

[Gemini 재실행]
python extract_features_v2.py
→ 성공!
```

---

## 📝 코드 작성 가이드라인 (Claude용)

### 디버깅 편의성을 위한 코드 작성

**1. 명확한 에러 메시지:**
```python
# Good
if not os.path.exists(data_path):
    raise FileNotFoundError(
        f"Training data not found at: {data_path}\n"
        f"Current directory: {os.getcwd()}\n"
        f"Expected location: {os.path.abspath(data_path)}"
    )

# Bad
if not os.path.exists(data_path):
    raise FileNotFoundError("File not found")
```

**2. 진행 상황 로깅:**
```python
# Good
print(f"[1/5] Loading data from {data_path}...")
print(f"  → Loaded {len(df):,} rows")
print(f"[2/5] Extracting features...")
print(f"  → Created {len(features)} features")

# Bad
# (아무것도 출력 안 함)
```

**3. 중간 결과 저장:**
```python
# Good
df.to_csv('intermediate_features.csv', index=False)
print(f"✓ Saved intermediate results to intermediate_features.csv")

# 이렇게 하면 에러 발생 시 다시 시작 안 해도 됨
```

**4. 체크포인트:**
```python
# Good
for fold in range(n_folds):
    model.fit(X_train, y_train)
    joblib.dump(model, f'model_fold{fold}.pkl')
    print(f"✓ Saved checkpoint: model_fold{fold}.pkl")
```

**5. Try-Except with Context:**
```python
# Good
try:
    df = pd.read_csv(data_path)
except Exception as e:
    print(f"❌ Error loading data:")
    print(f"   Path: {data_path}")
    print(f"   Error type: {type(e).__name__}")
    print(f"   Error message: {str(e)}")
    print(f"   Current directory: {os.getcwd()}")
    raise
```

---

## ⚡ 실행 가이드라인 (Gemini용)

### 실행 시 제공할 정보

**1. 시작 시:**
```
실행 시작: 2025-12-17 09:30:00
스크립트: train_model.py
작업 디렉토리: /path/to/exp_XXX
```

**2. 진행 중:**
```
[5분 경과] 진행률 20% - Feature extraction complete
[10분 경과] 진행률 40% - Fold 1/3 training...
[15분 경과] 진행률 60% - Fold 2/3 training...
```

**3. 완료 시:**
```
실행 완료: 2025-12-17 09:50:00
소요 시간: 20분
생성된 파일:
  - cv_results.json
  - model_fold0.pkl
  - model_fold1.pkl
  - model_fold2.pkl

주요 결과:
  CV Mean: 15.00
  CV Std: 0.12
```

**4. 에러 시:**
```
실행 실패: 2025-12-17 09:35:00
에러 발생 위치: train_model.py:145
에러 타입: ValueError
에러 메시지: (전체 복사)
로그 (마지막 50줄): (전체 복사)
```

---

## 🎯 성공 사례 (exp_033)

**Claude 작업 (구현):**
```python
# 1. 파일 작성
- extract_player_features.py (310줄)
- train_player_model.py (369줄)
- predict_submission.py (141줄)

# 2. 디버깅 고려사항
- 명확한 진행 상황 출력
- 중간 결과 저장 (player_stats.csv)
- 체크포인트 (모델 파일)
```

**Gemini 작업 (실행):**
```bash
# 1. 실행
python extract_player_features.py  # 2분
python train_player_model.py       # 18분
python predict_submission.py       # 1분

# 2. 결과 보고
CV Mean: 15.1234
Total time: 21분
Files created: 3
```

**결과:**
- 21분 만에 완료 ✅
- CV 15.35 → 15.12 (0.22 개선) ✅
- 에러 없이 한 번에 성공 ✅

---

## 📊 효율성 비교

### Before (작업 분할 없음)

```
Claude가 모두 수행:
- 코드 작성: 2시간
- 실행 대기: 30분 (컨텍스트 낭비!)
- 디버깅: 1시간
- 재실행: 30분
총 시간: 4시간, 컨텍스트 과다 사용
```

### After (작업 분할)

```
Claude (병렬 작업 가능):
- 코드 작성: 2시간
- 다음 실험 계획: 1시간
- 총 컨텍스트: 50K tokens

Gemini (백그라운드 실행):
- 실행 모니터링: 30분
- 결과 보고: 5분

총 시간: 2시간 30분 (40% 단축!)
컨텍스트: 50% 절약!
```

---

## 🔧 도구별 사용

### Claude Tools

**코드 작성:**
- Write (새 파일)
- Edit (수정)
- Read (확인)

**분석:**
- Task (agent 사용)
- Ultrathink (전략)
- Plan (계획)

**문서화:**
- Write (문서)
- Edit (업데이트)

### Gemini Tools (Zen MCP)

**실행:**
- mcp__zen__chat (실행 위임)
- continuation_id (대화 연속)

**보고:**
- 실행 결과
- 에러 메시지
- 로그 분석

---

## 📝 체크리스트

### Claude (코드 작성 전)

- [ ] 디버깅 편의성 고려했는가?
- [ ] 에러 메시지가 명확한가?
- [ ] 진행 상황 로깅 포함했는가?
- [ ] 중간 결과 저장하는가?
- [ ] 체크포인트 기능 있는가?
- [ ] README.md 작성했는가?

### Gemini (실행 전)

- [ ] 작업 디렉토리 확인했는가?
- [ ] 필요한 파일 모두 있는가?
- [ ] 실행 시간 예상했는가?
- [ ] 모니터링 간격 정했는가?

### Gemini (실행 후)

- [ ] CV 결과 확인했는가?
- [ ] 생성 파일 확인했는가?
- [ ] 에러 없었는가?
- [ ] 실행 시간 보고했는가?

---

## 🎯 요약

**핵심 규칙:**
1. **Claude = 코드 작성 & 디버깅**
2. **Gemini = 파일 실행 & 모니터링**
3. **디버깅 편의성 = 최우선**
4. **명확한 에러 메시지 & 로깅**
5. **중간 결과 저장 & 체크포인트**

**기대 효과:**
- ⚡ 40% 시간 단축
- 💰 50% 컨텍스트 절약
- 🎯 에러 빠른 해결
- 🚀 병렬 작업 가능

---

---

## 🔄 장시간 작업 자동 모니터링 (Tip 16)

### Exponential Backoff 패턴

**문제:** 45-55분 걸리는 작업을 자동 모니터링하고 싶다

**해결:** Bash sleep + 지수적 간격 체크

**구현:**
```python
# 한 응답 안에서:
Check #1 (T+0min)
sleep 60  # 1분 대기
Check #2 (T+1min)
sleep 120  # 2분 대기
Check #3 (T+3min)
sleep 240  # 4분 대기
Check #4 (T+7min)
sleep 480  # 8분 대기
Check #5 (T+15min)
sleep 960  # 16분 대기
Check #6 (T+31min)
sleep 960  # 16분 더
Check #7 (T+47min) → 대부분 완료!
```

**장점:**
- ✅ 사용자 입력 불필요 (47분간 자동)
- ✅ 토큰 효율적 (지수적 간격)
- ✅ 진행 상황 점진적 보고
- ✅ 타임아웃 안전 (47분 < 일반적 한계)

**사용 시기:**
- Gemini 장시간 실행 (30분+)
- Docker build
- CI/CD 파이프라인
- 모델 학습

**예시:**
```bash
# Claude가 자동으로:
mcp__zen__chat: 상태 체크
Bash: sleep 60
mcp__zen__chat: 상태 체크
Bash: sleep 120
...
최종 결과 보고
```

---

**이 프로토콜은 모든 실험에 적용됩니다.**

**작성일:** 2025-12-17 09:00
**버전:** 1.1 (Exponential backoff 추가)
**상태:** ✅ Active

