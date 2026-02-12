# Google Drive 완전 자동 동기화

> **목표:** 로컬 파일 수정 → 자동으로 Google Drive 업로드 → Colab에서 자동 접근
> **방법:** Google Drive 데스크톱 앱 사용
> **설정 시간:** 5분 (한 번만!)

---

## 🎯 방법 1: Google Drive 데스크톱 (추천!)

### 1. Google Drive 데스크톱 설치

**다운로드:** https://www.google.com/drive/download/

**Windows:**
- 설치 후 Google 계정 로그인
- 자동으로 `G:\` 드라이브 생성

**Mac:**
- 설치 후 Google 계정 로그인
- `/Volumes/GoogleDrive/` 생성

### 2. 프로젝트 폴더 이동/복사

```bash
# Windows (현재 위치)
로컬: C:\LSJ\dacon\dacon\kleague-algorithm\

# Google Drive로 복사
G:\내 드라이브\kleague-algorithm\
```

**구조:**
```
G:\내 드라이브\kleague-algorithm\
├── data/
│   ├── train.csv          ← 자동 업로드!
│   ├── test.csv
│   ├── sample_submission.csv
│   └── test/
├── models/                ← 자동 생성
├── submissions/           ← 자동 생성
└── logs/
```

### 3. 로컬에서 작업

```bash
# 로컬에서 파일 수정
vim G:\내 드라이브\kleague-algorithm\data\train.csv

# 자동으로 Google Drive 업로드! ✅
# Colab에서 자동으로 접근 가능! ✅
```

### 4. Colab에서 접근

```python
# Colab 노트북
drive.mount('/content/drive')

# 바로 접근 가능!
data = pd.read_csv('/content/drive/MyDrive/kleague-algorithm/data/train.csv')
```

**완전 자동화!** 🎉

---

## 🎯 방법 2: Colab에서 직접 업로드 (한 번만)

### Colab 노트북에 추가:

```python
# 첫 실행 시에만 (데이터 없으면)
from google.colab import files
import shutil

# train.csv 업로드
if not (DATA_DIR / 'train.csv').exists():
    print("train.csv를 선택하세요...")
    uploaded = files.upload()

    for filename, content in uploaded.items():
        with open(DATA_DIR / filename, 'wb') as f:
            f.write(content)

    print(f"✅ {filename} 업로드 완료!")

# test.csv, sample_submission.csv도 동일
```

**장점:**
- 설치 불필요
- Colab에서 직접 업로드

**단점:**
- 매번 업로드 (대용량 파일은 느림)
- test/ 폴더는 zip으로 압축 필요

---

## 🎯 방법 3: rclone (고급 사용자)

### 1. rclone 설치

```bash
# Linux/Mac
curl https://rclone.org/install.sh | sudo bash

# Windows
choco install rclone
```

### 2. Google Drive 설정

```bash
rclone config

# n (new remote)
# name: gdrive
# storage: drive (Google Drive)
# 브라우저에서 인증
```

### 3. 자동 동기화

```bash
# 로컬 → Google Drive 동기화
rclone sync /mnt/c/LSJ/dacon/dacon/kleague-algorithm/ \
            gdrive:kleague-algorithm/

# 또는 양방향
rclone bisync /mnt/c/LSJ/dacon/dacon/kleague-algorithm/ \
              gdrive:kleague-algorithm/
```

### 4. 자동화 스크립트

```bash
# sync.sh
#!/bin/bash

while true; do
    rclone sync /mnt/c/LSJ/dacon/dacon/kleague-algorithm/ \
                gdrive:kleague-algorithm/ \
                --exclude "*.pyc" \
                --exclude "__pycache__/"

    echo "Synced at $(date)"
    sleep 300  # 5분마다
done
```

**장점:**
- 완전 자동화
- 양방향 동기화
- 선택적 파일 제외

**단점:**
- 설정 복잡
- 명령줄 도구

---

## ✅ 추천 방법

### 🏆 1위: Google Drive 데스크톱

```
이유:
✅ 가장 간단
✅ GUI 제공
✅ 자동 동기화
✅ 양방향 지원
✅ 설정 5분

설치: https://www.google.com/drive/download/
```

### 🥈 2위: Colab 직접 업로드

```
이유:
✅ 설치 불필요
✅ 한 번만 업로드
✅ 간단

단점:
⚠️ 대용량 파일 느림
⚠️ test/ 폴더 zip 필요
```

### 🥉 3위: rclone

```
이유:
✅ 완전 자동화
✅ 스크립트 가능

단점:
⚠️ 고급 사용자용
⚠️ 설정 복잡
```

---

## 🚀 실행 순서 (방법 1)

```
1. Google Drive 데스크톱 설치 (5분)
   https://www.google.com/drive/download/

2. 로그인 & G:\ 드라이브 확인

3. 파일 복사
   C:\LSJ\dacon\dacon\kleague-algorithm\
   →
   G:\내 드라이브\kleague-algorithm\

4. Colab 노트북 실행
   kleague_colab_auto.ipynb

5. "Run All" 클릭

6. 끝! 🎉
```

---

## 📁 최종 구조

```
# 로컬 (선택)
C:\LSJ\dacon\dacon\kleague-algorithm\

# Google Drive (자동 동기화!)
G:\내 드라이브\kleague-algorithm\
├── data/
│   ├── train.csv          ← 한 번 복사
│   ├── test.csv
│   ├── sample_submission.csv
│   └── test/
├── models/                ← Colab에서 자동 생성
│   ├── catboost_x.cbm
│   └── catboost_y.cbm
├── submissions/           ← Colab에서 자동 생성
│   └── submission_*.csv
├── logs/
└── kleague_colab_auto.ipynb  ← 노트북

# Colab (자동 마운트!)
/content/drive/MyDrive/kleague-algorithm/
└── (동일 구조)
```

---

## 💡 팁

### 1. 선택적 동기화

Google Drive 데스크톱 설정:
```
- ✅ data/ (작은 파일만)
- ✅ code/
- ✅ notebooks/
- ❌ models/ (큰 파일, 필요시만)
- ❌ logs/ (불필요)
```

### 2. .gitignore 활용

```gitignore
# .gitignore
*.cbm
*.pkl
models/
submissions/
logs/
__pycache__/
```

### 3. 대용량 파일

```bash
# test/ 폴더 압축
cd data
zip -r test.zip test/

# Colab에서 자동 해제
import zipfile
with zipfile.ZipFile('data/test.zip', 'r') as zip_ref:
    zip_ref.extractall('data/')
```

---

**설치하고 5분이면 완전 자동화!** 🚀

---

*작성: 2025-12-15*
*다음: Colab에서 "Run All" 클릭!*
