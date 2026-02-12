"""
리더보드 모니터링 헬퍼 스크립트

목적: 리더보드 기록 및 분석 자동화
사용: python monitor_leaderboard.py --score 16.20 --rank 25

참고: 웹 크롤링 없이 수동 입력 방식
"""

import argparse
from datetime import datetime
import pandas as pd
from pathlib import Path

def add_record(rank_1st, avg_top10, my_rank, my_score, memo=""):
    """리더보드 기록 추가"""

    # 파일 경로
    csv_file = Path("leaderboard_history.csv")

    # 새 기록
    new_record = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'time': datetime.now().strftime('%H:%M'),
        'rank_1st': rank_1st,
        'avg_top10': avg_top10,
        'my_rank': my_rank,
        'my_score': my_score,
        'memo': memo
    }

    # 기존 파일 로드 또는 새로 생성
    if csv_file.exists():
        df = pd.read_csv(csv_file)
        df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
    else:
        df = pd.DataFrame([new_record])

    # 저장
    df.to_csv(csv_file, index=False)

    print(f"✅ 기록 추가 완료: {new_record['date']} {new_record['time']}")
    print(f"   1위: {rank_1st}, Top 10: {avg_top10}, 내 순위: {my_rank}, 내 점수: {my_score}")

    # 알림 체크
    check_alerts(rank_1st, avg_top10, my_rank)

    return df

def check_alerts(rank_1st, avg_top10, my_rank):
    """알림 기준 체크"""

    alerts = []

    # Critical
    if rank_1st < 16.00:
        alerts.append("🚨 CRITICAL: 1위 < 16.00 (새로운 접근법 등장!)")
    if avg_top10 < 16.20:
        alerts.append("🚨 CRITICAL: Top 10 평균 < 16.20 (전체 수준 급상승!)")
    if my_rank > 100:
        alerts.append("🚨 CRITICAL: 내 순위 > 100위 (크게 뒤처짐!)")

    # Warning
    if 16.00 <= rank_1st < 16.20:
        alerts.append("⚠️ WARNING: 1위 < 16.20 (강력한 경쟁자)")
    if 16.20 <= avg_top10 < 16.30:
        alerts.append("⚠️ WARNING: Top 10 평균 < 16.30 (경쟁 심화)")
    if 50 < my_rank <= 100:
        alerts.append("⚠️ WARNING: 내 순위 > 50위 (주의 필요)")

    # 알림 출력
    if alerts:
        print("\n📢 알림:")
        for alert in alerts:
            print(f"   {alert}")
    else:
        print("\n✅ 정상 범위 (현상 유지)")

def show_stats():
    """통계 표시"""

    csv_file = Path("leaderboard_history.csv")

    if not csv_file.exists():
        print("❌ 기록 없음. 먼저 기록을 추가하세요.")
        return

    df = pd.read_csv(csv_file)

    print("\n" + "=" * 60)
    print("리더보드 추적 통계")
    print("=" * 60)

    print(f"\n기록 수: {len(df)}개")
    print(f"기간: {df['date'].iloc[0]} ~ {df['date'].iloc[-1]}")

    print(f"\n1위 점수:")
    print(f"  최저: {df['rank_1st'].min():.4f}")
    print(f"  최고: {df['rank_1st'].max():.4f}")
    print(f"  평균: {df['rank_1st'].mean():.4f}")

    print(f"\nTop 10 평균:")
    print(f"  최저: {df['avg_top10'].min():.4f}")
    print(f"  최고: {df['avg_top10'].max():.4f}")
    print(f"  평균: {df['avg_top10'].mean():.4f}")

    print(f"\n내 순위:")
    print(f"  최고: {df['my_rank'].min():.0f}위")
    print(f"  최저: {df['my_rank'].max():.0f}위")
    print(f"  평균: {df['my_rank'].mean():.1f}위")

    print(f"\n내 점수:")
    print(f"  최고: {df['my_score'].min():.4f}")
    print(f"  최저: {df['my_score'].max():.4f}")

    print("\n최근 5개 기록:")
    print(df.tail(5).to_string(index=False))

def main():
    parser = argparse.ArgumentParser(description='리더보드 모니터링')
    parser.add_argument('--add', action='store_true', help='기록 추가')
    parser.add_argument('--rank-1st', type=float, help='1위 점수')
    parser.add_argument('--avg-top10', type=float, help='Top 10 평균')
    parser.add_argument('--my-rank', type=int, help='내 순위')
    parser.add_argument('--my-score', type=float, help='내 점수')
    parser.add_argument('--memo', type=str, default="", help='메모')
    parser.add_argument('--stats', action='store_true', help='통계 표시')

    args = parser.parse_args()

    if args.add:
        if not all([args.rank_1st, args.avg_top10, args.my_rank, args.my_score]):
            print("❌ 오류: --rank-1st, --avg-top10, --my-rank, --my-score 모두 필요")
            return

        add_record(args.rank_1st, args.avg_top10, args.my_rank, args.my_score, args.memo)

    elif args.stats:
        show_stats()

    else:
        print("사용법:")
        print("  기록 추가: python monitor_leaderboard.py --add --rank-1st 16.20 --avg-top10 16.35 --my-rank 25 --my-score 16.3639")
        print("  통계 보기: python monitor_leaderboard.py --stats")

if __name__ == "__main__":
    main()
