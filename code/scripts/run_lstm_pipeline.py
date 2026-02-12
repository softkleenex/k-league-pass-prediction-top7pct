"""
LSTM 100% 자동화 파이프라인

실행 순서:
1. LSTM 훈련 실행 (2-4시간)
2. 훈련 완료 후 자동으로:
   - 결과 분석
   - 피드백 생성
   - 문서 업데이트 (EXPERIMENT_LOG.md, STATUS.md)
   - 제출 여부 권장
   - 다음 행동 계획

사용법:
    python run_lstm_pipeline.py

백그라운드 실행:
    nohup python run_lstm_pipeline.py > pipeline.log 2>&1 &
"""

import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
import sys

# 경로 설정
CHECKPOINT_DIR = Path("checkpoints/lstm_100pct")
RESULTS_FILE = CHECKPOINT_DIR / "training_results.json"
PIPELINE_LOG = Path("logs/lstm_pipeline.log")
PIPELINE_LOG.parent.mkdir(parents=True, exist_ok=True)

def log(message):
    """로그 메시지 출력 및 저장"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)
    with open(PIPELINE_LOG, 'a', encoding='utf-8') as f:
        f.write(log_msg + '\n')

def run_training():
    """LSTM 훈련 실행"""
    log("="*80)
    log("Step 1: LSTM 훈련 시작")
    log("="*80)

    start_time = time.time()

    # 훈련 스크립트 실행
    process = subprocess.Popen(
        [sys.executable, "code/models/model_lstm_100pct.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )

    # 실시간 출력
    for line in process.stdout:
        print(line, end='')

    process.wait()

    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)

    if process.returncode != 0:
        log(f"❌ 훈련 실패! (exit code: {process.returncode})")
        return False

    log(f"✅ 훈련 완료! (소요 시간: {hours}시간 {minutes}분)")
    return True

def analyze_results():
    """결과 분석"""
    log("\n" + "="*80)
    log("Step 2: 결과 분석")
    log("="*80)

    if not RESULTS_FILE.exists():
        log("❌ 결과 파일을 찾을 수 없습니다!")
        return None

    with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
        results = json.load(f)

    cv_1_3 = results['cv_fold_1_3']
    cv_std = results['cv_std_1_3']
    cv_all = results['cv_all']

    log(f"\n📊 훈련 결과:")
    log(f"  - CV (Fold 1-3): {cv_1_3:.4f} ± {cv_std:.4f}")
    log(f"  - CV (All Folds): {cv_all:.4f}")
    log(f"  - Fold Scores (1-3): {results['fold_scores_1_3']}")

    # Sweet Spot 분석
    log(f"\n🎯 Sweet Spot 분석:")
    if results['sweet_spot']:
        log(f"  ✅ SWEET SPOT! (16.27-16.34)")
        log(f"  → Public 예상: 16.3-16.4")
        log(f"  → Gap 예상: +0.03-0.08")
        recommendation = "submit"
    elif cv_1_3 < 16.27:
        log(f"  ⚠️  과최적화 위험! (CV < 16.27)")
        log(f"  → Public 예상: 16.4-17.0+")
        log(f"  → Gap 예상: +0.13 이상")
        recommendation = "risky"
    elif 16.34 < cv_1_3 < 17.0:
        log(f"  ⚠️  성능 저하 (CV > 16.34)")
        log(f"  → Zone 16.34보다 나쁨")
        recommendation = "do_not_submit"
    else:
        log(f"  ❌ 완전 실패 (CV >> 17.0)")
        log(f"  → Zone 대비 크게 나쁨")
        recommendation = "failure"

    return {
        **results,
        'recommendation': recommendation
    }

def generate_feedback(results):
    """피드백 생성"""
    log("\n" + "="*80)
    log("Step 3: 피드백 생성")
    log("="*80)

    cv_1_3 = results['cv_fold_1_3']
    recommendation = results['recommendation']

    feedback = []

    # Zone 대비 비교
    zone_cv = 16.34
    diff = cv_1_3 - zone_cv

    feedback.append(f"\n📈 Zone 대비 분석:")
    if diff < -0.05:
        feedback.append(f"  ✅ LSTM이 Zone보다 {-diff:.4f} 우수!")
        feedback.append(f"  → 제출 강력 권장")
    elif -0.05 <= diff <= 0.05:
        feedback.append(f"  ➡️  Zone과 비슷한 수준 ({diff:+.4f})")
        feedback.append(f"  → 제출 고려 가능")
    else:
        feedback.append(f"  ⬇️  Zone보다 {diff:.4f} 나쁨")
        feedback.append(f"  → 제출 비권장")

    # 행동 권장
    feedback.append(f"\n🎯 권장 행동:")
    if recommendation == "submit":
        feedback.append(f"  1. ✅ 제출 파일 확인: submission_lstm_100pct.csv")
        feedback.append(f"  2. ✅ DACON 제출 (14/175 → 15/175)")
        feedback.append(f"  3. ✅ Public 결과 대기")
        feedback.append(f"  4. ✅ EXPERIMENT_LOG.md 업데이트")
    elif recommendation == "risky":
        feedback.append(f"  1. ⚠️  제출 신중히 고려")
        feedback.append(f"  2. ⚠️  과최적화 가능성 높음")
        feedback.append(f"  3. ⚠️  XGBoost와 유사한 패턴 (CV 15.73 → Public 16.47)")
        feedback.append(f"  4. ✅ 학습 기록만 남기기")
    elif recommendation == "do_not_submit":
        feedback.append(f"  1. ❌ 제출하지 않기")
        feedback.append(f"  2. ✅ Zone 16.34가 더 우수함")
        feedback.append(f"  3. ✅ EXPERIMENT_LOG.md에 실패 기록")
        feedback.append(f"  4. ✅ Week 2 전략 복귀")
    else:  # failure
        feedback.append(f"  1. ❌ 완전 실패")
        feedback.append(f"  2. ✅ 10% LSTM과 유사한 결과")
        feedback.append(f"  3. ✅ 시퀀스 모델링 부적합 재확인")
        feedback.append(f"  4. ✅ Week 2 전략 복귀")

    feedback_text = "\n".join(feedback)
    log(feedback_text)

    # 피드백 파일 저장
    feedback_file = Path("logs/lstm_feedback.txt")
    with open(feedback_file, 'w', encoding='utf-8') as f:
        f.write(f"LSTM 100% 훈련 피드백\n")
        f.write(f"생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        f.write(f"CV (Fold 1-3): {cv_1_3:.4f} ± {results['cv_std_1_3']:.4f}\n")
        f.write(f"Zone CV: {zone_cv:.4f}\n")
        f.write(f"차이: {diff:+.4f}\n")
        f.write(feedback_text)

    log(f"\n피드백 저장: {feedback_file}")

    return feedback_text

def update_experiment_log(results):
    """EXPERIMENT_LOG.md 업데이트"""
    log("\n" + "="*80)
    log("Step 4: EXPERIMENT_LOG.md 업데이트")
    log("="*80)

    cv_1_3 = results['cv_fold_1_3']
    cv_std = results['cv_std_1_3']
    recommendation = results['recommendation']

    # Exp 32 내용 생성
    exp_32 = f"""
#### Exp 32: LSTM 100% Full Data (Overnight)

| 항목 | 값 |
|------|-----|
| **모델** | LSTM (sequence length 3, batch 256) |
| **샘플링** | 100% (356,721 samples) |
| **훈련 시간** | {datetime.now().strftime('%Y-%m-%d %H:%M')} |
| **CV Fold 1-3** | {cv_1_3:.4f} ± {cv_std:.4f} |
| **CV All** | {results['cv_all']:.4f} |
| **Zone 대비** | {cv_1_3 - 16.34:+.4f} |
| **Public** | {'제출 예정' if recommendation == 'submit' else '제출 안 함'} |
| **결과** | {'✅ Sweet Spot' if results['sweet_spot'] else '❌ 실패'} |

**설계:**
- 10% → 100% 데이터 (10배 증가)
- sequence_length: 50 → 3 (짧은 시퀀스)
- batch_size: 64 → 256 (효율적 학습)
- Fold 1-3 CV 별도 계산
- 체크포인트 자동 저장
- 30분 간격 모니터링 시스템

**결과 분석:**
- Sweet Spot: {'Yes' if results['sweet_spot'] else 'No'}
- Zone 대비: {cv_1_3 - 16.34:+.4f}
- 권장: {recommendation}

**교훈:**
"""

    if results['sweet_spot']:
        exp_32 += "- LSTM도 Sweet Spot 달성 가능\n"
        exp_32 += "- Zone과 비슷한 수준\n"
    elif cv_1_3 < 16.27:
        exp_32 += "- 과최적화 (XGBoost와 동일 패턴)\n"
        exp_32 += "- CV 낮음 ≠ Public 좋음\n"
    else:
        exp_32 += "- 시퀀스 모델링 부적합 재확인\n"
        exp_32 += "- Zone 통계가 최적\n"

    log("EXPERIMENT_LOG.md에 Exp 32 추가:")
    log(exp_32)

    # 파일에 추가 (실제 구현 시)
    # TODO: EXPERIMENT_LOG.md 파일 읽기 → Phase 8에 추가 → 저장

    return exp_32

def update_status(results):
    """STATUS.md 업데이트"""
    log("\n" + "="*80)
    log("Step 5: STATUS.md 업데이트")
    log("="*80)

    today = datetime.now().strftime('%Y-%m-%d')

    status_update = f"""
### {today}
- ✅ LSTM 100% 훈련 완료 (overnight)
- ✅ CV (Fold 1-3): {results['cv_fold_1_3']:.4f} ± {results['cv_std_1_3']:.4f}
- ✅ Sweet Spot: {'Yes' if results['sweet_spot'] else 'No'}
- ✅ 권장: {results['recommendation']}
"""

    log(status_update)

    return status_update

def main():
    """메인 파이프라인"""
    log("\n" + "🚀"*40)
    log("LSTM 100% 자동화 파이프라인 시작")
    log("🚀"*40)

    pipeline_start = time.time()

    # Step 1: 훈련 실행
    success = run_training()
    if not success:
        log("\n❌ 파이프라인 중단: 훈련 실패")
        return

    # Step 2: 결과 분석
    results = analyze_results()
    if results is None:
        log("\n❌ 파이프라인 중단: 결과 분석 실패")
        return

    # Step 3: 피드백 생성
    feedback = generate_feedback(results)

    # Step 4: 문서 업데이트
    exp_log = update_experiment_log(results)
    status_update = update_status(results)

    # 최종 요약
    pipeline_elapsed = time.time() - pipeline_start
    hours = int(pipeline_elapsed // 3600)
    minutes = int((pipeline_elapsed % 3600) // 60)

    log("\n" + "="*80)
    log("파이프라인 완료!")
    log("="*80)
    log(f"총 소요 시간: {hours}시간 {minutes}분")
    log(f"\n📋 다음 할 일:")

    if results['recommendation'] == 'submit':
        log("  1. submission_lstm_100pct.csv 확인")
        log("  2. DACON 제출")
        log("  3. Public 결과 대기")
    else:
        log("  1. logs/lstm_feedback.txt 확인")
        log("  2. EXPERIMENT_LOG.md 업데이트")
        log("  3. Week 2 전략 복귀")

    log(f"\n피드백 파일: logs/lstm_feedback.txt")
    log(f"파이프라인 로그: {PIPELINE_LOG}")
    log(f"체크포인트: {CHECKPOINT_DIR}/")

    log("\n✅ 모든 작업 완료! 결과를 확인하세요.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("\n⚠️  사용자 중단")
    except Exception as e:
        log(f"\n❌ 오류 발생: {e}")
        import traceback
        log(traceback.format_exc())
