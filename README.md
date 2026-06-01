# Cost-Sensitive Active Learning for UAV Multispectral Anomaly Detection

Official code repository for the paper:

> **A Risk-Weighted Active Learning Framework for Annotation-Constrained UAV Multispectral Anomaly Detection**
> Ali Güneş, *Member, IEEE*, et al.
> *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, 2025 (under review)

---

## Overview

This repository implements the **cost-sensitive active learning (cAL)** framework proposed in the paper. The method reduces missed environmental anomalies in UAV multispectral imagery under limited annotation budgets by incorporating asymmetric misclassification cost at every stage of the active learning pipeline.

**Core idea:** a single configurable risk coefficient `r+` controls the false-negative penalty simultaneously in (i) the cost-sensitive SVM (cSVM) training objective, (ii) the risk-sensitive margin uncertainty query score, and (iii) the risk-weighted evaluation criterion `C(r+)`.

### Key results (4 UAV riparian scenes, r+ = 4)

| Metric | Standard AL | cAL (ours) | Improvement |
|--------|------------|------------|-------------|
| Aggregate FNR | 0.214 | 0.078 | −63.6% |
| Anomaly Recall | 0.786 | 0.922 | +13.6 pp |
| False Negatives | 493.0 | 179.3 | −63.6% |

---

## Repository structure

```
.
├── al_cal_experiment.py          # Main cAL framework (Section IV) — core paper contribution
├── active_learning_comparison.py # Baseline comparison (Random, Entropy, Margin, BADGE, etc.)
├── ablation_feature_sets.py      # Feature set ablation (Raw-5D / Raw-10D / Full-19D)
├── experiment_band_selection.py  # SHAP-based band importance analysis (Fig. 2)
├── al_novel_experiments.py       # Supplementary: RX warm-start & cross-scene transfer
├── pipeline_gat_conformal.py     # Supplementary: SpectralGAT + conformal prediction baseline
├── generate_all_plots.py         # Reproduce learning curve figures (Fig. 4, 5, 6, 7)
├── generate_cal_confusion_matrices.py  # Reproduce confusion matrix figures (Fig. 8)
├── generate_missing_plots.py     # Additional diagnostic plots
├── data/                         # Dataset directory (download separately — see data/README.md)
├── results/                      # Experiment outputs (JSON + figures written here)
├── requirements.txt
└── .gitignore
```

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/aligunesgit/csal-uav-anomaly-detection.git
cd csal-uav-anomaly-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset → place in data/ (see data/README.md)

# 4. Run everything
bash run_all.sh
```

All scripts must be run from the **repository root** (the directory containing `run_all.sh`).

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/aligunesgit/csal-uav-anomaly-detection.git
cd csal-uav-anomaly-detection
```

### 2. Install dependencies

Python 3.9+ is required.

```bash
pip install -r requirements.txt
```

> `torch` is only needed for `pipeline_gat_conformal.py`. All other scripts run on standard scikit-learn.

### 3. Download the dataset

See [`data/README.md`](data/README.md) for download instructions (Zenodo DOI: [10.5281/zenodo.14852117](https://doi.org/10.5281/zenodo.14852117)).

Place the four scene folders (`z1/`, `z2/`, `e1/`, `e2/`) inside the `data/` directory.

---

## Reproducing the paper results

### Main experiment — cAL vs baselines (Tables III, IV; Figs. 4–8)

```bash
python al_cal_experiment.py
```

Runs cAL at four risk levels `r+ ∈ {1, 2, 3, 4}` and three baselines (Random AL, Standard AL, Unc+KernelKMeans) across all four scenes. Results are saved to `results/al_cal/`.

**Runtime:** ~2–4 hours on CPU (3 repetitions × 4 scenes × 4 risk levels × ~12 AL iterations).

### Baseline comparison (supplementary)

```bash
python active_learning_comparison.py
```

Compares eight query strategies (Random, Entropy, Margin, Least Confidence, CoreSet, BADGE, RX-Guided, Unc+KernelKMeans) using Random Forest. Results saved to `results/al/`.

### Feature ablation (Section III-C justification)

```bash
python ablation_feature_sets.py
```

Compares Raw-5D, Raw-10D, and Full-19D feature representations. Results saved to `results/ablation/`.

### SHAP band importance (Fig. 2)

```bash
python experiment_band_selection.py
```

Computes global SHAP values pooled via LOCO-CV over all four scenes. Results saved to `results/`.

### Generate figures

```bash
# Learning curves (Figs. 4, 5, 6, 7) — requires al_cal_experiment.py to have run first
python generate_all_plots.py

# Confusion matrix panels (Fig. 8)
python generate_cal_confusion_matrices.py
```

---

## Method summary

### Risk-weighted misclassification cost

$$C(r^+) = \frac{r^+ \cdot \text{FN} + \text{FP}}{N} \times 100$$

When `r+ = 1` this reduces to symmetric misclassification cost (complement of accuracy). As `r+` increases, the cost increasingly penalises missed anomalies over false alarms.

### Cost-sensitive SVM (cSVM)

Class-specific misclassification weights: `w_anomaly = r+`, `w_normal = 1`. Trained with RBF kernel; `C` and `γ` tuned by stratified cross-validation on the initial labeled set.

### Two-stage query strategy

1. **Risk-sensitive margin uncertainty** — scores each unlabeled superpixel by `u(x) = r+ |d(x)|` if predicted anomaly, `|d(x)|` otherwise. Selects top-M candidate pool.
2. **Kernel k-means diversity filtering** — partitions the candidate pool into B clusters using the same RBF kernel; selects the most uncertain sample per cluster to form the query batch.

---

## Experimental configuration

| Parameter | Value |
|-----------|-------|
| Initial labeled set | 20 (10 anomaly + 10 normal) |
| Query batch size B | 50 |
| Candidate pool size M | 5B = 250 |
| Maximum budget T | 600 or 5% of pool |
| Train/test split | 70% / 30% (stratified) |
| Repetitions | 3 |
| SVM kernel | RBF |
| Risk levels r+ | {1, 2, 3, 4} |

---

## Dataset

**Galician Rivers Multispectral Anomaly Detection Dataset** — four large-format UAV scenes of riparian ecosystems in Galicia, Spain. Acquired with a MicaSense RedEdge sensor (5 bands, 8.2 cm/pixel). Anthropogenic anomalies (buildings, roads, bridges) constitute 3–13% of each scene.

| Scene | Size | Anomaly % |
|-------|------|-----------|
| Z1 | 3807 × 2141 px | 3.95 |
| Z2 | 2081 × 957 px | 12.51 |
| E1 | 3629 × 961 px | 4.72 |
| E2 | 1094 × 707 px | 3.30 |

Reference: J. López-Fandiño et al., Zenodo, 2025. https://doi.org/10.5281/zenodo.14852117

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{gunes2025csal,
  title   = {A Risk-Weighted Active Learning Framework for
             Annotation-Constrained {UAV} Multispectral Anomaly Detection},
  author  = {G{\"u}ne{\c{s}}, Ali and others},
  journal = {IEEE Journal of Selected Topics in Applied Earth Observations
             and Remote Sensing},
  year    = {2025},
  note    = {Under review}
}
```

---

## License

Code released under the MIT License. Dataset distributed under its original Zenodo license (see dataset DOI for terms).
