#!/bin/bash

# K리그 야간 실험 배치 (2025-12-08)
# Week 2 전략: Direction 각도 및 min_samples 최적화

echo "================================================================================"
echo "K리그 야간 실험 배치 시작"
echo "시작 시간: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"
echo ""

# 로그 디렉토리
LOG_DIR="/mnt/c/LSJ/dacon/dacon/kleague-algorithm/logs"
RESULT_FILE="${LOG_DIR}/overnight_results.txt"

# 초기화
> "${RESULT_FILE}"
echo "야간 실험 결과 요약" > "${RESULT_FILE}"
echo "실행 시작: $(date '+%Y-%m-%d %H:%M:%S')" >> "${RESULT_FILE}"
echo "" >> "${RESULT_FILE}"

# 프로젝트 루트로 이동
cd /mnt/c/LSJ/dacon/dacon/kleague-algorithm

# =============================================================================
# 실험 1: Direction 40도
# =============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "실험 1/4: Direction 40도 간격"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

START_TIME=$(date +%s)

python code/models/model_direction_40deg.py > "${LOG_DIR}/exp1_direction_40deg.log" 2>&1

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# 결과 추출
if [ $? -eq 0 ]; then
    CV_FOLD13=$(grep "Fold 1-3 CV:" "${LOG_DIR}/exp1_direction_40deg.log" | tail -1 | awk '{print $4}')
    VERDICT=$(grep "최종 판정" "${LOG_DIR}/exp1_direction_40deg.log" -A 1 | tail -1 | awk '{print $1}')

    echo "✅ 실험 1 완료 (${DURATION}초)" | tee -a "${RESULT_FILE}"
    echo "   Fold 1-3 CV: ${CV_FOLD13}" | tee -a "${RESULT_FILE}"
    echo "   판정: ${VERDICT}" | tee -a "${RESULT_FILE}"
    echo "" | tee -a "${RESULT_FILE}"
else
    echo "❌ 실험 1 실패" | tee -a "${RESULT_FILE}"
    echo "" | tee -a "${RESULT_FILE}"
fi

# =============================================================================
# 실험 2: Direction 50도
# =============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "실험 2/4: Direction 50도 간격"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

START_TIME=$(date +%s)

python code/models/model_direction_50deg.py > "${LOG_DIR}/exp2_direction_50deg.log" 2>&1

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

if [ $? -eq 0 ]; then
    CV_FOLD13=$(grep "Fold 1-3 CV:" "${LOG_DIR}/exp2_direction_50deg.log" | tail -1 | awk '{print $4}')
    VERDICT=$(grep "최종 판정" "${LOG_DIR}/exp2_direction_50deg.log" -A 1 | tail -1 | awk '{print $1}')

    echo "✅ 실험 2 완료 (${DURATION}초)" | tee -a "${RESULT_FILE}"
    echo "   Fold 1-3 CV: ${CV_FOLD13}" | tee -a "${RESULT_FILE}"
    echo "   판정: ${VERDICT}" | tee -a "${RESULT_FILE}"
    echo "" | tee -a "${RESULT_FILE}"
else
    echo "❌ 실험 2 실패" | tee -a "${RESULT_FILE}"
    echo "" | tee -a "${RESULT_FILE}"
fi

# =============================================================================
# 실험 3: min_samples=22
# =============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "실험 3/4: 6x6 min_samples=22"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

START_TIME=$(date +%s)

python code/models/model_6x6_min22.py > "${LOG_DIR}/exp3_6x6_min22.log" 2>&1

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

if [ $? -eq 0 ]; then
    CV_FOLD13=$(grep "Fold 1-3 CV:" "${LOG_DIR}/exp3_6x6_min22.log" | tail -1 | awk '{print $4}')
    VERDICT=$(grep "판정:" "${LOG_DIR}/exp3_6x6_min22.log" | tail -1 | awk '{print $2}')

    echo "✅ 실험 3 완료 (${DURATION}초)" | tee -a "${RESULT_FILE}"
    echo "   Fold 1-3 CV: ${CV_FOLD13}" | tee -a "${RESULT_FILE}"
    echo "   판정: ${VERDICT}" | tee -a "${RESULT_FILE}"
    echo "" | tee -a "${RESULT_FILE}"
else
    echo "❌ 실험 3 실패" | tee -a "${RESULT_FILE}"
    echo "" | tee -a "${RESULT_FILE}"
fi

# =============================================================================
# 실험 4: min_samples=24
# =============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "실험 4/4: 6x6 min_samples=24"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

START_TIME=$(date +%s)

python code/models/model_6x6_min24.py > "${LOG_DIR}/exp4_6x6_min24.log" 2>&1

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

if [ $? -eq 0 ]; then
    CV_FOLD13=$(grep "Fold 1-3 CV:" "${LOG_DIR}/exp4_6x6_min24.log" | tail -1 | awk '{print $4}')
    VERDICT=$(grep "판정:" "${LOG_DIR}/exp4_6x6_min24.log" | tail -1 | awk '{print $2}')

    echo "✅ 실험 4 완료 (${DURATION}초)" | tee -a "${RESULT_FILE}"
    echo "   Fold 1-3 CV: ${CV_FOLD13}" | tee -a "${RESULT_FILE}"
    echo "   판정: ${VERDICT}" | tee -a "${RESULT_FILE}"
    echo "" | tee -a "${RESULT_FILE}"
else
    echo "❌ 실험 4 실패" | tee -a "${RESULT_FILE}"
    echo "" | tee -a "${RESULT_FILE}"
fi

# =============================================================================
# 최종 요약
# =============================================================================
echo ""
echo "================================================================================"
echo "야간 실험 배치 완료"
echo "종료 시간: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"
echo ""

echo "종료 시간: $(date '+%Y-%m-%d %H:%M:%S')" >> "${RESULT_FILE}"
echo "" >> "${RESULT_FILE}"
echo "상세 로그:" >> "${RESULT_FILE}"
echo "  - ${LOG_DIR}/exp1_direction_40deg.log" >> "${RESULT_FILE}"
echo "  - ${LOG_DIR}/exp2_direction_50deg.log" >> "${RESULT_FILE}"
echo "  - ${LOG_DIR}/exp3_6x6_min22.log" >> "${RESULT_FILE}"
echo "  - ${LOG_DIR}/exp4_6x6_min24.log" >> "${RESULT_FILE}"
echo "" >> "${RESULT_FILE}"

# 제출 파일 위치 안내
echo "제출 파일 생성:" >> "${RESULT_FILE}"
echo "  - submission_direction_40deg.csv" >> "${RESULT_FILE}"
echo "  - submission_direction_50deg.csv" >> "${RESULT_FILE}"
echo "  - submission_6x6_min22.csv" >> "${RESULT_FILE}"
echo "  - submission_6x6_min24.csv" >> "${RESULT_FILE}"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "결과 요약 확인:"
echo "  cat ${RESULT_FILE}"
echo ""
echo "개별 로그 확인:"
echo "  tail -100 ${LOG_DIR}/exp1_direction_40deg.log"
echo "  tail -100 ${LOG_DIR}/exp2_direction_50deg.log"
echo "  tail -100 ${LOG_DIR}/exp3_6x6_min22.log"
echo "  tail -100 ${LOG_DIR}/exp4_6x6_min24.log"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
