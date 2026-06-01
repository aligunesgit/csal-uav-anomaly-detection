#!/usr/bin/env bash
# ============================================================
# Reproduce all experiments from:
# "A Risk-Weighted Active Learning Framework for
#  Annotation-Constrained UAV Multispectral Anomaly Detection"
#
# Run from the repository root:
#   bash run_all.sh
#
# Prerequisites:
#   - Python 3.9+  (pip install -r requirements.txt)
#   - Dataset in data/  (see data/README.md for Zenodo download)
#
# Total runtime: ~3–5 hours on CPU
# ============================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

echo "========================================================"
echo "  cAL Experiment Runner"
echo "  Working directory: $REPO"
echo "========================================================"

# ---------- 0. Check data -------------------------------------
if [ ! -f "data/z1/z1.raw" ]; then
    echo ""
    echo "ERROR: Dataset not found at data/z1/z1.raw"
    echo "       Download from https://doi.org/10.5281/zenodo.14852117"
    echo "       and place scene folders (z1/ z2/ e1/ e2/) inside data/"
    exit 1
fi
echo "[OK] Dataset found"

# ---------- 1. SHAP band importance (Fig. 2) ------------------
echo ""
echo "[1/6] SHAP band importance analysis..."
python3 experiment_band_selection.py
echo "      -> results/  (band_importance_*.png, shap_results.json)"

# ---------- 2. Main cAL experiment (Tables III, IV; Figs 4-8) -
echo ""
echo "[2/6] Main cAL experiment (this takes ~2-4 hours)..."
python3 al_cal_experiment.py
echo "      -> results/al_cal/"

# ---------- 3. Baseline comparison ----------------------------
echo ""
echo "[3/6] Baseline query strategy comparison..."
python3 active_learning_comparison.py
echo "      -> results/al/"

# ---------- 4. Feature set ablation ---------------------------
echo ""
echo "[4/6] Feature set ablation (Raw-5D / Raw-10D / Full-19D)..."
python3 ablation_feature_sets.py
echo "      -> results/ablation/"

# ---------- 5. Generate figures from saved results ------------
echo ""
echo "[5/6] Generating learning curve figures (Figs. 4, 5, 6, 7)..."
python3 generate_all_plots.py
echo "      -> results/al/"

echo ""
echo "[6/6] Generating confusion matrix figures (Fig. 8)..."
python3 generate_cal_confusion_matrices.py
echo "      -> results/al_cal/"

echo ""
echo "========================================================"
echo "  All done. Figures are in results/"
echo "========================================================"
