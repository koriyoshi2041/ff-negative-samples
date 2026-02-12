#!/bin/bash
# FF Research - A100 Training Script
# Comprehensive experiments with detailed logging

set -e

echo "=============================================="
echo "FF Research - A100 Full Experiments"
echo "Started at: $(date)"
echo "=============================================="

# Create directories
mkdir -p results
mkdir -p logs

# Log everything
exec > >(tee -a logs/run_all_$(date +%Y%m%d_%H%M%S).log) 2>&1

echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if python -c 'import torch; print(torch.cuda.is_available())' | grep -q True; then
    echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi

# ============================================================
# 1. Negative Strategy Comparison (CRITICAL: 1000 epochs)
# ============================================================
echo ""
echo "============================================================"
echo "[1/7] Negative Strategy Comparison (1000 epochs/layer)"
echo "============================================================"
python experiments/strategy_comparison_full.py \
    --epochs 1000 \
    --output results/strategy_comparison_1000ep.json \
    2>&1 | tee logs/strategy_comparison.log

# ============================================================
# 2. Transfer Learning Comparison (FF vs BP)
# ============================================================
echo ""
echo "============================================================"
echo "[2/7] Transfer Learning Comparison"
echo "============================================================"
python experiments/transfer_comparison.py \
    --epochs 500 \
    --output results/transfer_comparison.json \
    2>&1 | tee logs/transfer_comparison.log

# ============================================================
# 3. Dendritic FF (Wright et al. Science 2025)
# ============================================================
echo ""
echo "============================================================"
echo "[3/7] Dendritic FF Experiment"
echo "============================================================"
python experiments/dendritic_ff_experiment.py \
    --epochs 500 \
    --output results/dendritic_ff.json \
    2>&1 | tee logs/dendritic_ff.log

# ============================================================
# 4. Three-Factor FF (Neuromodulation)
# ============================================================
echo ""
echo "============================================================"
echo "[4/7] Three-Factor FF Experiment"
echo "============================================================"
python experiments/three_factor_experiment.py \
    --pretrain-epochs 500 \
    --batch-size 50000 \
    --output results/three_factor_ff.json \
    2>&1 | tee logs/three_factor_ff.log

# ============================================================
# 5. PCL FF (Predictive Coding Light)
# ============================================================
echo ""
echo "============================================================"
echo "[5/7] PCL-FF Experiment"
echo "============================================================"
python experiments/pcl_ff_experiment.py \
    --epochs 500 \
    --batch-size 50000 \
    --output results/pcl_ff.json \
    2>&1 | tee logs/pcl_ff.log

# ============================================================
# 6. Prospective FF (Nature Neuroscience 2024)
# ============================================================
echo ""
echo "============================================================"
echo "[6/7] Prospective FF Experiment"
echo "============================================================"
python experiments/prospective_ff_experiment.py \
    --epochs 500 \
    --output results/prospective_ff.json \
    2>&1 | tee logs/prospective_ff.log

# ============================================================
# 7. Layer Collaboration Transfer Test
# ============================================================
echo ""
echo "============================================================"
echo "[7/7] Layer Collaboration Transfer Test"
echo "============================================================"
if [ -f experiments/layer_collab_transfer.py ]; then
    python experiments/layer_collab_transfer.py \
        --pretrain-epochs 500 \
        --output results/layer_collab_transfer.json \
        2>&1 | tee logs/layer_collab_transfer.log
else
    echo "Skipped: layer_collab_transfer.py not found"
fi

# ============================================================
# Summary
# ============================================================
echo ""
echo "=============================================="
echo "All experiments completed!"
echo "Finished at: $(date)"
echo "=============================================="
echo ""
echo "Results saved in results/:"
ls -lh results/*.json 2>/dev/null || echo "No JSON files found"
echo ""
echo "Logs saved in logs/:"
ls -lh logs/*.log 2>/dev/null || echo "No log files found"
echo ""
echo "To copy results back:"
echo "  scp -r results/ your-local-machine:~/ff-results/"
