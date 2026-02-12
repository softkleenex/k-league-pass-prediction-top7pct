#!/bin/bash
# Overnight experiments - run sequentially to avoid memory issues

echo "========================================"
echo "Starting overnight experiments..."
echo "Started at: $(date)"
echo "========================================"

cd /mnt/c/LSJ/dacon/dacon/kleague-algorithm/code/models/experiments_v2

# exp_078: 5-fold test
echo ""
echo "[1/4] exp_078: 5-Fold Test"
echo "Started at: $(date)"
cd exp_078_5fold
python train_5fold.py > exp078.log 2>&1
echo "Completed at: $(date)"
cd ..

# exp_079: Multi-seed ensemble
echo ""
echo "[2/4] exp_079: Multi-Seed Ensemble"
echo "Started at: $(date)"
cd exp_079_multiseed
python train_multiseed.py > exp079.log 2>&1
echo "Completed at: $(date)"
cd ..

# exp_080: Hyperparameter tuning
echo ""
echo "[3/4] exp_080: Hyperparameter Tuning"
echo "Started at: $(date)"
cd exp_080_hparam
python train_hparam.py > exp080.log 2>&1
echo "Completed at: $(date)"
cd ..

# exp_081: Post-processing
echo ""
echo "[4/4] exp_081: Post-Processing"
echo "Started at: $(date)"
cd exp_081_postprocess
python train_postprocess.py > exp081.log 2>&1
echo "Completed at: $(date)"
cd ..

echo ""
echo "========================================"
echo "All experiments completed!"
echo "Finished at: $(date)"
echo "========================================"

# Summary
echo ""
echo "=== RESULTS SUMMARY ==="
echo ""
echo "exp_078 (5-Fold):"
tail -15 exp_078_5fold/exp078.log 2>/dev/null || echo "No results yet"
echo ""
echo "exp_079 (Multi-Seed):"
tail -10 exp_079_multiseed/exp079.log 2>/dev/null || echo "No results yet"
echo ""
echo "exp_080 (Hyperparameter):"
tail -15 exp_080_hparam/exp080.log 2>/dev/null || echo "No results yet"
echo ""
echo "exp_081 (Post-Processing):"
tail -15 exp_081_postprocess/exp081.log 2>/dev/null || echo "No results yet"
