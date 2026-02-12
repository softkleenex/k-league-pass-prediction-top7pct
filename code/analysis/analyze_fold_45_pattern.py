"""
Fold 4-5 이상치 패턴 분석
모든 모델에서 Fold 4-5가 낮은 CV를 보이는 이유 규명
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(".")

print("=" * 80)
print("Fold 4-5 이상치 패턴 분석")
print("=" * 80)

# 데이터 로드
train_df = pd.read_csv(DATA_DIR / "train.csv")

# 마지막 패스만 추출
train_last = train_df.groupby('game_episode').last().reset_index()
train_last = train_last.dropna(subset=['end_x', 'end_y'])
train_last['delta_x'] = train_last['end_x'] - train_last['start_x']
train_last['delta_y'] = train_last['end_y'] - train_last['start_y']
train_last['distance'] = np.sqrt(train_last['delta_x']**2 + train_last['delta_y']**2)

print(f"\n총 샘플 수: {len(train_last):,}")

# GroupKFold
gkf = GroupKFold(n_splits=5)
game_ids = train_last['game_id'].values

print("\n" + "=" * 80)
print("Fold별 데이터 분포 상세 분석")
print("=" * 80)

fold_info = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(train_last, groups=game_ids)):
    val_fold = train_last.iloc[val_idx]

    print(f"\n{'='*80}")
    print(f"Fold {fold+1}")
    print(f"{'='*80}")

    # 기본 통계
    print(f"\n[기본 정보]")
    print(f"  샘플 수: {len(val_fold):,}")
    print(f"  게임 수: {val_fold['game_id'].nunique()}")

    # 위치 분포
    print(f"\n[시작 위치 분포]")
    print(f"  start_x: 평균={val_fold['start_x'].mean():.2f}, 중앙={val_fold['start_x'].median():.2f}, std={val_fold['start_x'].std():.2f}")
    print(f"  start_y: 평균={val_fold['start_y'].mean():.2f}, 중앙={val_fold['start_y'].median():.2f}, std={val_fold['start_y'].std():.2f}")

    # 도착 위치 분포
    print(f"\n[도착 위치 분포]")
    print(f"  end_x: 평균={val_fold['end_x'].mean():.2f}, 중앙={val_fold['end_x'].median():.2f}, std={val_fold['end_x'].std():.2f}")
    print(f"  end_y: 평균={val_fold['end_y'].mean():.2f}, 중앙={val_fold['end_y'].median():.2f}, std={val_fold['end_y'].std():.2f}")

    # Delta 분포
    print(f"\n[Delta 분포]")
    print(f"  delta_x: 평균={val_fold['delta_x'].mean():.2f}, 중앙={val_fold['delta_x'].median():.2f}, std={val_fold['delta_x'].std():.2f}")
    print(f"  delta_y: 평균={val_fold['delta_y'].mean():.2f}, 중앙={val_fold['delta_y'].median():.2f}, std={val_fold['delta_y'].std():.2f}")
    print(f"  distance: 평균={val_fold['distance'].mean():.2f}, 중앙={val_fold['distance'].median():.2f}, std={val_fold['distance'].std():.2f}")

    # Zone 분포
    def get_zone_6x6(x, y):
        x_zone = min(5, int(x / (105 / 6)))
        y_zone = min(5, int(y / (68 / 6)))
        return x_zone * 6 + y_zone

    val_fold_temp = val_fold.copy()
    val_fold_temp['zone'] = val_fold_temp.apply(lambda r: get_zone_6x6(r['start_x'], r['start_y']), axis=1)
    zone_dist = val_fold_temp['zone'].value_counts()

    print(f"\n[Zone 분포 (6x6)]")
    print(f"  총 Zone 수: {len(zone_dist)}")
    print(f"  Zone당 평균 샘플: {zone_dist.mean():.1f}")
    print(f"  Zone당 중앙 샘플: {zone_dist.median():.1f}")
    print(f"  최대 샘플 Zone: {zone_dist.max()}")
    print(f"  최소 샘플 Zone: {zone_dist.min()}")

    # 필드 영역별 분포
    print(f"\n[필드 영역별 분포]")
    defensive = val_fold[val_fold['start_x'] < 35]
    midfield = val_fold[(val_fold['start_x'] >= 35) & (val_fold['start_x'] < 70)]
    attacking = val_fold[val_fold['start_x'] >= 70]

    print(f"  수비 (x<35):    {len(defensive):4d} ({len(defensive)/len(val_fold)*100:5.1f}%)")
    print(f"  미드 (35≤x<70): {len(midfield):4d} ({len(midfield)/len(val_fold)*100:5.1f}%)")
    print(f"  공격 (x≥70):    {len(attacking):4d} ({len(attacking)/len(val_fold)*100:5.1f}%)")

    # 게임 ID 분포
    game_counts = val_fold['game_id'].value_counts()
    print(f"\n[게임 분포]")
    print(f"  게임당 평균 에피소드: {game_counts.mean():.1f}")
    print(f"  게임당 중앙 에피소드: {game_counts.median():.1f}")
    print(f"  최대 에피소드 게임: {game_counts.max()}")
    print(f"  최소 에피소드 게임: {game_counts.min()}")

    # Fold 정보 저장
    fold_info.append({
        'fold': fold + 1,
        'samples': len(val_fold),
        'games': val_fold['game_id'].nunique(),
        'start_x_mean': val_fold['start_x'].mean(),
        'start_y_mean': val_fold['start_y'].mean(),
        'delta_x_mean': val_fold['delta_x'].mean(),
        'delta_y_mean': val_fold['delta_y'].mean(),
        'distance_mean': val_fold['distance'].mean(),
        'delta_x_std': val_fold['delta_x'].std(),
        'delta_y_std': val_fold['delta_y'].std(),
        'defensive_pct': len(defensive)/len(val_fold)*100,
        'midfield_pct': len(midfield)/len(val_fold)*100,
        'attacking_pct': len(attacking)/len(val_fold)*100,
    })

# Fold 비교 분석
print("\n" + "=" * 80)
print("Fold 간 비교 분석")
print("=" * 80)

fold_df = pd.DataFrame(fold_info)

print("\n[샘플 수]")
print(fold_df[['fold', 'samples', 'games']].to_string(index=False))

print("\n[Delta 평균 비교]")
print(fold_df[['fold', 'delta_x_mean', 'delta_y_mean', 'distance_mean']].to_string(index=False))

print("\n[Delta 표준편차 비교]")
print(fold_df[['fold', 'delta_x_std', 'delta_y_std']].to_string(index=False))

print("\n[필드 영역 비율 비교]")
print(fold_df[['fold', 'defensive_pct', 'midfield_pct', 'attacking_pct']].to_string(index=False))

# Fold 4-5 vs Fold 1-3 비교
print("\n" + "=" * 80)
print("Fold 4-5 vs Fold 1-3 비교")
print("=" * 80)

fold_13 = fold_df[fold_df['fold'].isin([1, 2, 3])]
fold_45 = fold_df[fold_df['fold'].isin([4, 5])]

print("\n[Delta 평균]")
print(f"  Fold 1-3 delta_x: {fold_13['delta_x_mean'].mean():.4f} ± {fold_13['delta_x_mean'].std():.4f}")
print(f"  Fold 4-5 delta_x: {fold_45['delta_x_mean'].mean():.4f} ± {fold_45['delta_x_mean'].std():.4f}")
print(f"  차이: {fold_45['delta_x_mean'].mean() - fold_13['delta_x_mean'].mean():.4f}")

print(f"\n  Fold 1-3 delta_y: {fold_13['delta_y_mean'].mean():.4f} ± {fold_13['delta_y_mean'].std():.4f}")
print(f"  Fold 4-5 delta_y: {fold_45['delta_y_mean'].mean():.4f} ± {fold_45['delta_y_mean'].std():.4f}")
print(f"  차이: {fold_45['delta_y_mean'].mean() - fold_13['delta_y_mean'].mean():.4f}")

print(f"\n  Fold 1-3 distance: {fold_13['distance_mean'].mean():.4f} ± {fold_13['distance_mean'].std():.4f}")
print(f"  Fold 4-5 distance: {fold_45['distance_mean'].mean():.4f} ± {fold_45['distance_mean'].std():.4f}")
print(f"  차이: {fold_45['distance_mean'].mean() - fold_13['distance_mean'].mean():.4f}")

print("\n[Delta 표준편차]")
print(f"  Fold 1-3 delta_x_std: {fold_13['delta_x_std'].mean():.4f}")
print(f"  Fold 4-5 delta_x_std: {fold_45['delta_x_std'].mean():.4f}")
print(f"  차이: {fold_45['delta_x_std'].mean() - fold_13['delta_x_std'].mean():.4f}")

print(f"\n  Fold 1-3 delta_y_std: {fold_13['delta_y_std'].mean():.4f}")
print(f"  Fold 4-5 delta_y_std: {fold_45['delta_y_std'].mean():.4f}")
print(f"  차이: {fold_45['delta_y_std'].mean() - fold_13['delta_y_std'].mean():.4f}")

print("\n[필드 영역 비율]")
print(f"  Fold 1-3 defensive: {fold_13['defensive_pct'].mean():.2f}%")
print(f"  Fold 4-5 defensive: {fold_45['defensive_pct'].mean():.2f}%")
print(f"  차이: {fold_45['defensive_pct'].mean() - fold_13['defensive_pct'].mean():.2f}%")

print(f"\n  Fold 1-3 attacking: {fold_13['attacking_pct'].mean():.2f}%")
print(f"  Fold 4-5 attacking: {fold_45['attacking_pct'].mean():.2f}%")
print(f"  차이: {fold_45['attacking_pct'].mean() - fold_13['attacking_pct'].mean():.2f}%")

# 간단한 Zone 모델로 예측 성능 비교
print("\n" + "=" * 80)
print("간단한 6x6 Zone 모델 성능 비교")
print("=" * 80)

def get_zone_6x6(x, y):
    x_zone = min(5, int(x / (105 / 6)))
    y_zone = min(5, int(y / (68 / 6)))
    return x_zone * 6 + y_zone

train_last['zone'] = train_last.apply(lambda r: get_zone_6x6(r['start_x'], r['start_y']), axis=1)

fold_scores = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(train_last, groups=game_ids)):
    train_fold = train_last.iloc[train_idx]
    val_fold = train_last.iloc[val_idx]

    # Zone 통계 계산
    zone_stats = train_fold.groupby('zone').agg({
        'delta_x': 'median',
        'delta_y': 'median'
    }).to_dict()

    # 예측
    val_fold = val_fold.copy()
    val_fold['pred_x'] = val_fold.apply(
        lambda r: np.clip(r['start_x'] + zone_stats['delta_x'].get(r['zone'], 0), 0, 105), axis=1
    )
    val_fold['pred_y'] = val_fold.apply(
        lambda r: np.clip(r['start_y'] + zone_stats['delta_y'].get(r['zone'], 0), 0, 68), axis=1
    )

    # 점수 계산
    dist = np.sqrt((val_fold['pred_x'] - val_fold['end_x'])**2 + (val_fold['pred_y'] - val_fold['end_y'])**2)
    cv = dist.mean()
    fold_scores.append(cv)

    print(f"  Fold {fold+1}: CV = {cv:.4f}")

print(f"\n평균 CV: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
print(f"Fold 1-3 평균: {np.mean(fold_scores[:3]):.4f}")
print(f"Fold 4-5 평균: {np.mean(fold_scores[3:]):.4f}")
print(f"차이: {np.mean(fold_scores[3:]) - np.mean(fold_scores[:3]):.4f}")

print("\n" + "=" * 80)
print("결론")
print("=" * 80)

if np.mean(fold_scores[3:]) < np.mean(fold_scores[:3]):
    print("\n✅ Fold 4-5의 데이터가 실제로 예측하기 더 쉬운 특성을 가짐")
    print("   → 모델이 Fold 4-5의 쉬운 패턴에 과적합")
    print("   → Public 데이터는 Fold 1-3과 유사한 분포일 가능성 높음")
    print("\n권장 전략:")
    print("   1. Fold 1-3 평균 CV를 진짜 성능으로 간주")
    print("   2. 모델 선택 시 Fold 4-5 CV는 무시")
    print("   3. 앙상블 가중치 계산 시 Fold 1-3만 사용")
else:
    print("\n⚠️ Fold 4-5의 데이터 분포는 정상이지만 모델이 과적합")
    print("   → 복잡한 모델일수록 Fold 4-5에 과적합 경향")
    print("\n권장 전략:")
    print("   1. 더 단순한 모델 사용")
    print("   2. 정규화 강화")
    print("   3. Fold 수 증가 (7-Fold 이상)")

print("\n" + "=" * 80)
