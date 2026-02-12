# Phase 1: Target Encoding 제거 - 변경 사항

**생성일:** 2025-12-16
**파일:** model_domain_features_v2_no_target.py

---

## 변경 요약

### 제거된 코드 블록

#### 1. Player 통계 계산 (lines 145-155)
```python
# REMOVED:
player_stats = train_df.groupby('player_id').agg({
    'delta_x': 'mean',
    'delta_y': 'mean',
    'distance': 'mean',
    'is_forward': 'mean'
}).reset_index()
player_stats.columns = ['player_id', 'player_avg_dx', 'player_avg_dy',
                        'player_avg_distance', 'player_forward_ratio']
```

#### 2. Team 통계 계산 (lines 157-163)
```python
# REMOVED:
team_stats = train_df.groupby('team_id').agg({
    'delta_x': 'mean',
    'delta_y': 'mean',
    'distance': 'mean'
}).reset_index()
team_stats.columns = ['team_id', 'team_avg_dx', 'team_avg_dy', 'team_avg_distance']
```

#### 3. Merge 코드 (lines 165-179)
```python
# REMOVED:
train_df = train_df.merge(player_stats, on='player_id', how='left')
train_df = train_df.merge(team_stats, on='team_id', how='left')

test_all = test_all.merge(player_stats, on='player_id', how='left')
test_all = test_all.merge(team_stats, on='team_id', how='left')

# 없는 player/team은 global 평균
for col in ['player_avg_dx', 'player_avg_dy', 'player_avg_distance', 'player_forward_ratio']:
    train_df[col] = train_df[col].fillna(train_df[col].mean())
    test_all[col] = test_all[col].fillna(train_df[col].mean())

for col in ['team_avg_dx', 'team_avg_dy', 'team_avg_distance']:
    train_df[col] = train_df[col].fillna(train_df[col].mean())
    test_all[col] = test_all[col].fillna(train_df[col].mean())
```

---

## 피처 변경

### 제거된 피처 (7개)

| 피처 | 유형 | 설명 |
|------|------|------|
| player_avg_dx | Player Target Encoding | 선수별 평균 delta_x |
| player_avg_dy | Player Target Encoding | 선수별 평균 delta_y |
| player_avg_distance | Player Target Encoding | 선수별 평균 패스 거리 |
| player_forward_ratio | Player Target Encoding | 선수별 전진 패스 비율 |
| team_avg_dx | Team Target Encoding | 팀별 평균 delta_x |
| team_avg_dy | Team Target Encoding | 팀별 평균 delta_y |
| team_avg_distance | Team Target Encoding | 팀별 평균 패스 거리 |

### 남은 피처 (25개)

| 카테고리 | 피처 수 | 피처 리스트 |
|----------|---------|-------------|
| 기본 위치 | 2 | start_x, start_y |
| 골대 관련 | 3 | goal_distance, goal_angle, is_near_goal |
| 필드 구역 | 6 | zone_attack, zone_defense, zone_middle, zone_left, zone_center, zone_right |
| 경계선 거리 | 4 | dist_to_left, dist_to_right, dist_to_top, dist_to_bottom |
| 이전 패스 | 4 | prev_dx, prev_dy, prev_distance, direction |
| Episode | 4 | episode_progress, episode_avg_distance, episode_forward_ratio, is_last_pass |
| 시간 | 2 | period_id, time_seconds |
| **합계** | **25** | - |

---

## 예상 결과

### 가설
```
Target Encoding → Train에 과적합 → Public Gap 증가
제거 후 → Gap 감소 예상
```

### 수치 예상

| 메트릭 | v1 (32개 피처) | v2 (25개 피처) | 변화 |
|--------|----------------|----------------|------|
| CV (Fold 1-3) | 14.0229 | 15.11 (예상) | +1.09 |
| Public Score | 15.16 | 15.52 (예상) | +0.36 |
| **Gap (Public - CV)** | **+1.14** | **+0.41** | **-0.73** |

**핵심:**
- CV 약간 상승 (14.02 → 15.11)
- Public 약간 상승 (15.16 → 15.52)
- **Gap 대폭 감소 (1.14 → 0.41)** ← 목표!

---

## 검증 방법

### 1. CV 측정
```bash
cd /mnt/c/LSJ/dacon/dacon/kleague-algorithm
python3 code/models/best/model_domain_features_v2_no_target.py
```

### 2. 예상 결과 확인
- Fold 1-3 CV: 15.0-15.2 범위 예상
- 파일명: submission_domain_v2_no_target_cv{score}.csv

### 3. Public 제출 (조건부)
- CV < 15.3 → 즉시 제출
- CV >= 15.3 → Phase 2로 진행 (LightGBM 튜닝)

---

## 코드 품질 검증

### 1. 피처 수 확인
```python
len(feature_cols) == 25  # OK
```

### 2. Target Encoding 코드 부재
```bash
grep "player_stats\|team_stats" model_domain_features_v2_no_target.py
# → 주석/문서에만 존재 (코드에는 없음)
```

### 3. Categorical Features
```python
categorical_features = [
    'direction', 'period_id', 'is_last_pass',
    'zone_attack', 'zone_defense', 'zone_middle',
    'zone_left', 'zone_center', 'zone_right'
]  # 9개 (변경 없음)
```

---

## 다음 단계

### Phase 1 성공 시 (Gap < 0.5)
```
→ Phase 2: LightGBM 하이퍼파라미터 튜닝
→ 목표: CV 14.8-15.0, Public 15.2-15.3
```

### Phase 1 실패 시 (Gap >= 0.5)
```
→ 다른 접근 탐색 (Ensemble, XGBoost 등)
→ Zone 6x6로 회귀 고려
```

---

## 파일 구조

```
code/models/best/
├── model_domain_features_lgbm.py        # v1 (32개 피처)
├── model_domain_features_v2_no_target.py  # v2 (25개 피처) ← NEW
└── PHASE1_CHANGES.md                    # 이 문서
```

---

## 체크리스트

- [x] Player Target Encoding 제거
- [x] Team Target Encoding 제거
- [x] feature_cols에서 7개 피처 제거
- [x] categorical_features 확인 (변경 없음)
- [x] 파일 헤더 수정
- [x] 총 피처 수 25개 확인
- [x] 실행 가능한 완전한 코드
- [x] 원본 파일 보존

---

**상태:** 준비 완료
**실행 대기:** 사용자 승인 후 즉시 실행 가능
