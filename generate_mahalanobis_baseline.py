#!/usr/bin/env python3
"""
Mahalanobis Standalone Baseline — cAL Paper
============================================
Evaluates the per-scene Mahalanobis spectral anomaly score as a
threshold-based classifier, independent of any active learning loop.

Design:
  - Uses the same 70/30 stratified train/test split as al_cal_experiment.py
  - Mahalanobis score is the 19th feature (index 18) of the full 19-D set:
        mah_i = sqrt( mean_b( ((mu_i,b - mu_b) / (sigma_b + eps))^2 ) )
    where mu_b and sigma_b are computed over ALL superpixels in the scene
    (unsupervised global statistics), making this a fully annotation-free score.
  - Optimal threshold is found on the POOL set (70%) by maximising Youden's J:
        J = Recall + Specificity - 1
    This simulates access to full pool annotations (upper-bound for Mahalanobis).
  - Final metrics are reported on the held-out TEST set (30%).
  - Cost metric C(r+=4) is included for direct comparison with AL results.

Output:
  - results/mahalanobis/mahalanobis_results.json
  - results/mahalanobis/mahalanobis_roc.png
"""

import sys, warnings, json
warnings.filterwarnings('ignore')

import numpy as np
from pathlib import Path
from sklearn.metrics import roc_curve

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── Paths ────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
# Resolve data root: prefer sibling symlinks, fall back to agentic-ai data dir
_CANDIDATE = _ROOT / "data"
_FALLBACK   = _ROOT.parent / "agentic ai" / "data"
BASE = _CANDIDATE if (_CANDIDATE / "z1").is_dir() else _FALLBACK
OUT  = _ROOT / "results" / "mahalanobis"
OUT.mkdir(parents=True, exist_ok=True)

# ─── Config (identical to al_cal_experiment.py) ───────────────────────────────
IMAGES    = {"z1": (3807, 2141), "z2": (2081, 957), "e1": (3629, 961), "e2": (1094, 707)}
SCENES    = list(IMAGES.keys())
SEED_BASE = 2024

# ─── Data loading ─────────────────────────────────────────────────────────────
def load_scene(name, w, h):
    with open(BASE / name / f"{name}.raw", "rb") as f:
        f.read(12)
        img = np.frombuffer(f.read(), dtype=np.uint32).reshape(h, w, 5).astype(np.float32)
    with open(BASE / name / f"{name}_gt.pgm", "rb") as f:
        f.readline(); f.readline(); f.readline()
        gt = np.frombuffer(f.read()[:h * w], dtype=np.uint8).reshape(h, w)
    with open(BASE / name / f"{name}_seg.raw", "rb") as f:
        f.read(8)
        seg = np.frombuffer(f.read(), dtype=np.uint32).reshape(h, w)
    return img, gt, seg


def extract_mahalanobis(img, gt, seg):
    """Return Mahalanobis score (scalar per superpixel) and binary labels."""
    n    = int(seg.max()) + 1
    sf   = seg.ravel().astype(np.int64)
    imgf = img.reshape(-1, 5).astype(np.float64)
    gtf  = gt.ravel().astype(np.int64)
    cnt  = np.bincount(sf, minlength=n).clip(1).astype(np.float64)

    mu  = np.zeros((n, 5))
    sq  = np.zeros((n, 5))
    for b in range(5):
        mu[:, b] = np.bincount(sf, weights=imgf[:, b],    minlength=n) / cnt
        sq[:, b] = np.bincount(sf, weights=imgf[:, b]**2, minlength=n) / cnt
    sig = np.sqrt(np.clip(sq - mu**2, 0, None))

    E   = 1e-6
    mah = np.sqrt((((mu - mu.mean(0)) / (sig.mean(0) + E))**2).mean(1))

    # Labels: majority-vote ground truth (same as ablation / al_cal)
    ac  = np.bincount(sf, weights=(gtf == 2).astype(np.float64), minlength=n)
    nc  = np.bincount(sf, weights=(gtf == 1).astype(np.float64), minlength=n)
    labels = (ac > nc).astype(np.int64)

    return mah.astype(np.float32), labels


# ─── Threshold selection ──────────────────────────────────────────────────────
def youden_threshold(scores, labels):
    """Find threshold maximising Youden's J on the provided (pool) set."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    j = tpr + (1 - fpr) - 1          # J = Sensitivity + Specificity - 1
    best_idx = int(np.argmax(j))
    return float(thresholds[best_idx]), float(j[best_idx])


# ─── Metrics ──────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, r_plus=4):
    TP = int(np.sum((y_true == 1) & (y_pred == 1)))
    FN = int(np.sum((y_true == 1) & (y_pred == 0)))
    FP = int(np.sum((y_true == 0) & (y_pred == 1)))
    TN = int(np.sum((y_true == 0) & (y_pred == 0)))
    total = len(y_true)
    anom  = int(y_true.sum())

    fnr     = FN / anom if anom > 0 else 0.0
    recall  = TP / anom if anom > 0 else 0.0
    fpr_val = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    prec    = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    f1      = (2 * prec * recall) / (prec + recall) if (prec + recall) > 0 else 0.0
    cost    = (r_plus * FN + FP) / total * 100
    return {
        "TP": TP, "FN": FN, "FP": FP, "TN": TN,
        "fnr":    round(fnr,    3),
        "recall": round(recall, 3),
        "fpr":    round(fpr_val, 3),
        "precision": round(prec, 3),
        "f1":     round(f1,     3),
        "cost_rp4": round(cost, 2),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    results = {}
    roc_data = {}

    print("=" * 60)
    print("Mahalanobis Standalone Baseline  (threshold = Youden J)")
    print("=" * 60)

    for scene, (w, h) in IMAGES.items():
        print(f"\n[{scene.upper()}] Loading...", end=" ", flush=True)
        img, gt, seg = load_scene(scene, w, h)
        scores, labels = extract_mahalanobis(img, gt, seg)
        del img, gt
        print(f"N={len(labels)}, anom={labels.sum()} ({100*labels.mean():.1f}%)")

        # Stratified 70 / 30 split — identical seed to al_cal_experiment.py
        rng      = np.random.default_rng(SEED_BASE)
        anom_idx = np.where(labels == 1)[0]
        norm_idx = np.where(labels == 0)[0]
        pool_anom = rng.choice(anom_idx, int(0.7 * len(anom_idx)), replace=False)
        pool_norm = rng.choice(norm_idx, int(0.7 * len(norm_idx)), replace=False)
        pool_idx  = np.concatenate([pool_anom, pool_norm])
        test_idx  = np.array([i for i in range(len(labels))
                               if i not in set(pool_idx.tolist())])

        sc_pool = scores[pool_idx];  y_pool = labels[pool_idx]
        sc_test = scores[test_idx];  y_test = labels[test_idx]

        # Threshold selection on pool
        thr, j_val = youden_threshold(sc_pool, y_pool)
        print(f"  Optimal threshold: {thr:.4f}  (Youden J on pool = {j_val:.3f})")

        # Evaluation on test set
        y_pred = (sc_test >= thr).astype(np.int64)
        m = compute_metrics(y_test, y_pred, r_plus=4)
        results[scene] = {"threshold": round(thr, 4), "youden_j_pool": round(j_val, 3), **m}

        print(f"  TEST  FNR={m['fnr']:.3f}  Recall={m['recall']:.3f}  "
              f"Prec={m['precision']:.3f}  F1={m['f1']:.3f}  C(r+=4)={m['cost_rp4']:.2f}")

        # ROC curve data for plot
        fpr_arr, tpr_arr, _ = roc_curve(y_test, sc_test)
        roc_data[scene] = {"fpr": fpr_arr.tolist(), "tpr": tpr_arr.tolist()}

    # ─── Save JSON ────────────────────────────────────────────────────────────
    out_json = OUT / "mahalanobis_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_json}")

    # ─── Summary table ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY  (test set, threshold = Youden J on pool)")
    print(f"{'Scene':<6} {'FNR':>6} {'Recall':>8} {'Prec':>7} {'F1':>6} {'C(r+=4)':>9}")
    print("-" * 60)
    for s in SCENES:
        m = results[s]
        print(f"{s.upper():<6} {m['fnr']:>6.3f} {m['recall']:>8.3f} "
              f"{m['precision']:>7.3f} {m['f1']:>6.3f} {m['cost_rp4']:>9.2f}")
    print("=" * 60)

    # ─── ROC plot ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), sharex=True, sharey=True)
    colors = {"z1": "#1f77b4", "z2": "#9467bd", "e1": "#2ca02c", "e2": "#d62728"}
    for ax, scene in zip(axes, SCENES):
        rd = roc_data[scene]
        m  = results[scene]
        ax.plot(rd["fpr"], rd["tpr"], color=colors[scene], lw=1.8)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_title(f"Scene {scene.upper()}\n"
                     f"FNR={m['fnr']:.3f}, Recall={m['recall']:.3f}", fontsize=9)
        ax.set_xlabel("FPR")
    axes[0].set_ylabel("TPR (Recall)")
    fig.suptitle("Mahalanobis Baseline — ROC Curves (test set)", fontsize=10)
    plt.tight_layout()
    out_fig = OUT / "mahalanobis_roc.png"
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    print(f"ROC figure saved to {out_fig}")


if __name__ == "__main__":
    main()
