"""
데이터 품질 검증 스크립트

code-reviewer가 발견한 잠재적 이슈 검증:
1. 음수 좌표 존재 여부
2. 범위 초과 좌표 존재 여부
3. NaN 값 존재 여부
4. 중복 데이터 존재 여부

2025-12-09 검증용
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(".")

print("=" * 80)
print("데이터 품질 검증")
print("=" * 80)

# =============================================================================
# 1. Train 데이터 검증
# =============================================================================
print("\n[1] Train 데이터 로드...")
train_df = pd.read_csv(DATA_DIR / "train.csv")
print(f"  행: {len(train_df):,}, 열: {len(train_df.columns)}")

print("\n[2] Train 데이터 품질 체크...")

# 2.1 NaN 체크
print("\n2.1 NaN 값 체크:")
nan_counts = train_df.isna().sum()
if nan_counts.sum() == 0:
    print("  ✅ NaN 없음")
else:
    print("  ⚠️ NaN 발견:")
    print(nan_counts[nan_counts > 0])

# 2.2 음수 좌표 체크
print("\n2.2 음수 좌표 체크:")
neg_start_x = (train_df['start_x'] < 0).sum()
neg_start_y = (train_df['start_y'] < 0).sum()
neg_end_x = (train_df['end_x'] < 0).sum()
neg_end_y = (train_df['end_y'] < 0).sum()

if neg_start_x + neg_start_y + neg_end_x + neg_end_y == 0:
    print("  ✅ 음수 좌표 없음")
else:
    print(f"  ⚠️ 음수 좌표 발견:")
    if neg_start_x > 0:
        print(f"    start_x < 0: {neg_start_x:,}개")
    if neg_start_y > 0:
        print(f"    start_y < 0: {neg_start_y:,}개")
    if neg_end_x > 0:
        print(f"    end_x < 0: {neg_end_x:,}개")
    if neg_end_y > 0:
        print(f"    end_y < 0: {neg_end_y:,}개")

# 2.3 범위 초과 체크
print("\n2.3 범위 초과 체크:")
over_start_x = (train_df['start_x'] > 105).sum()
over_start_y = (train_df['start_y'] > 68).sum()
over_end_x = (train_df['end_x'] > 105).sum()
over_end_y = (train_df['end_y'] > 68).sum()

if over_start_x + over_start_y + over_end_x + over_end_y == 0:
    print("  ✅ 범위 초과 없음")
else:
    print(f"  ⚠️ 범위 초과 발견:")
    if over_start_x > 0:
        print(f"    start_x > 105: {over_start_x:,}개")
    if over_start_y > 0:
        print(f"    start_y > 68: {over_start_y:,}개")
    if over_end_x > 0:
        print(f"    end_x > 105: {over_end_x:,}개")
    if over_end_y > 0:
        print(f"    end_y > 68: {over_end_y:,}개")

# 2.4 중복 체크
print("\n2.4 중복 game_episode 체크:")
dup_episodes = train_df['game_episode'].duplicated().sum()
if dup_episodes == 0:
    print("  ⚠️ 주의: 모든 game_episode가 고유함 (시퀀스 데이터인데?)")
else:
    print(f"  ✅ 중복 존재: {dup_episodes:,}개 (시퀀스 데이터 정상)")

# =============================================================================
# 3. Test 데이터 검증
# =============================================================================
print("\n" + "=" * 80)
print("[3] Test 데이터 로드...")
test_df = pd.read_csv(DATA_DIR / "test.csv")
print(f"  행: {len(test_df):,}")

# Test 에피소드 로드
test_episodes = []
for _, row in test_df.iterrows():
    ep_df = pd.read_csv(DATA_DIR / row['path'])
    test_episodes.append(ep_df)
test_all = pd.concat(test_episodes, ignore_index=True)
print(f"  전체 패스: {len(test_all):,}, 열: {len(test_all.columns)}")

print("\n[4] Test 데이터 품질 체크...")

# 4.1 NaN 체크
print("\n4.1 NaN 값 체크:")
nan_counts = test_all.isna().sum()
if nan_counts.sum() == 0:
    print("  ✅ NaN 없음")
else:
    print("  ⚠️ NaN 발견:")
    print(nan_counts[nan_counts > 0])

# 4.2 음수 좌표 체크
print("\n4.2 음수 좌표 체크:")
neg_start_x = (test_all['start_x'] < 0).sum()
neg_start_y = (test_all['start_y'] < 0).sum()

if neg_start_x + neg_start_y == 0:
    print("  ✅ 음수 좌표 없음")
else:
    print(f"  ⚠️ 음수 좌표 발견:")
    if neg_start_x > 0:
        print(f"    start_x < 0: {neg_start_x:,}개")
    if neg_start_y > 0:
        print(f"    start_y < 0: {neg_start_y:,}개")

# 4.3 범위 초과 체크
print("\n4.3 범위 초과 체크:")
over_start_x = (test_all['start_x'] > 105).sum()
over_start_y = (test_all['start_y'] > 68).sum()

if over_start_x + over_start_y == 0:
    print("  ✅ 범위 초과 없음")
else:
    print(f"  ⚠️ 범위 초과 발견:")
    if over_start_x > 0:
        print(f"    start_x > 105: {over_start_x:,}개")
    if over_start_y > 0:
        print(f"    start_y > 68: {over_start_y:,}개")

# =============================================================================
# 5. 최종 요약
# =============================================================================
print("\n" + "=" * 80)
print("검증 요약")
print("=" * 80)

# code-reviewer 이슈 검증
print("\n[code-reviewer 이슈 검증]")

print("\n1. 음수 좌표 처리 (High Priority):")
if (train_df['start_x'] < 0).sum() + (train_df['start_y'] < 0).sum() == 0:
    print("   ✅ Train 데이터: 음수 좌표 없음 → 이슈 없음")
else:
    print("   🚨 Train 데이터: 음수 좌표 존재 → 수정 필요!")

if (test_all['start_x'] < 0).sum() + (test_all['start_y'] < 0).sum() == 0:
    print("   ✅ Test 데이터: 음수 좌표 없음 → 이슈 없음")
else:
    print("   🚨 Test 데이터: 음수 좌표 존재 → 수정 필요!")

print("\n2. Zone fallback min_samples 체크 (High Priority):")
print("   ⚠️ 코드 로직 이슈 → 별도 수정 필요")
print("   - 현재: Zone fallback이 min_samples 체크 안함")
print("   - 영향: 소수 샘플 Zone도 fallback으로 사용")

print("\n3. Division by Zero (Medium Priority):")
print("   ⚠️ 이론적 가능성 → 방어 코드 추가 권장")
print("   - Inverse variance 계산 시")

print("\n[데이터 품질 종합]")
train_ok = (train_df['start_x'] >= 0).all() and (train_df['start_x'] <= 105).all() and \
           (train_df['start_y'] >= 0).all() and (train_df['start_y'] <= 68).all() and \
           not train_df[['start_x', 'start_y', 'end_x', 'end_y']].isna().any().any()

test_ok = (test_all['start_x'] >= 0).all() and (test_all['start_x'] <= 105).all() and \
          (test_all['start_y'] >= 0).all() and (test_all['start_y'] <= 68).all() and \
          not test_all[['start_x', 'start_y']].isna().any().any()

if train_ok and test_ok:
    print("✅ 모든 데이터 품질 정상")
    print("✅ safe_fold13.py는 현재 데이터셋에서 정상 작동")
    print("⚠️ 단, Zone fallback 로직 개선 권장")
else:
    print("🚨 데이터 품질 이슈 발견")
    print("🚨 코드 수정 필요")

print("\n" + "=" * 80)
print("검증 완료!")
print("=" * 80)
