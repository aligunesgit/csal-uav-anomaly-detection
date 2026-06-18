#!/usr/bin/env python3
"""
Feature Set Ablation — cAL Paper
=================================
Compares three feature representations to justify the 19-D design:

  Raw-5D  : 5 band means only
  Raw-10D : 5 band means + 5 band standard deviations
  Full-19D: current 19-D feature set (means + std + VI + ratios + Mahalanobis)

Runs cAL (r+=4, 3 seeds) on all four scenes for each feature set.
Writes results to results/ablation/ and prints a summary table.

Runtime: ~20–40 min depending on machine.
"""

import sys, warnings, json
warnings.filterwarnings('ignore')

import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from sklearn.kernel_approximation import Nystroem
from sklearn.cluster import MiniBatchKMeans

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── Paths ────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
BASE  = _ROOT / "data"
OUT   = _ROOT / "results" / "ablation"
OUT.mkdir(parents=True, exist_ok=True)

# ─── Config (matches al_cal_experiment.py) ───────────────────────────────────
IMAGES    = {"z1": (3807, 2141), "z2": (2081, 957), "e1": (3629, 961), "e2": (1094, 707)}
SCENES    = list(IMAGES.keys())
N_INIT    = 20
BATCH_Q   = 50
MAX_BUDGET = 600
MAX_PCT   = 0.05
N_RUNS    = 3
POOL_CAP  = 10_000
SEED_BASE = 2024
R_PLUS    = 4          # ablation runs cAL at r+=4 only
# Deterministic per-scene seed offset — matches al_cal_experiment.py
SCENE_SEED_OFFSET = {s: i * 10 for i, s in enumerate(SCENES)}  # z1=0, z2=10, e1=20, e2=30

FEATURE_MODES = ["5d", "10d", "19d"]

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


def extract_features(img, gt, seg, mode="19d"):
    """
    mode='5d'  -> band means only         (5 features)
    mode='10d' -> band means + band std   (10 features)
    mode='19d' -> full 19-D feature set   (19 features)
    """
    n   = int(seg.max()) + 1
    sf  = seg.ravel().astype(np.int64)
    imgf = img.reshape(-1, 5).astype(np.float64)
    gtf  = gt.ravel().astype(np.int64)
    cnt  = np.bincount(sf, minlength=n).clip(1).astype(np.float64)

    mu  = np.zeros((n, 5))
    sq  = np.zeros((n, 5))
    for b in range(5):
        mu[:, b] = np.bincount(sf, weights=imgf[:, b],    minlength=n) / cnt
        sq[:, b] = np.bincount(sf, weights=imgf[:, b]**2, minlength=n) / cnt
    sig = np.sqrt(np.clip(sq - mu**2, 0, None))

    if mode == "5d":
        feats = mu.astype(np.float32)

    elif mode == "10d":
        feats = np.c_[mu, sig].astype(np.float32)

    else:  # "19d" — identical to al_cal_experiment.py
        E = 1e-6
        B, G, R, RE, N_ = mu[:,0], mu[:,1], mu[:,2], mu[:,3], mu[:,4]
        ndvi  = (N_ - R)  / (N_ + R  + E)
        ndre  = (N_ - RE) / (N_ + RE + E)
        exg   = 2*G - R - B
        evi   = 2.5 * (N_ - R) / (N_ + 6*R - 7.5*B + 1 + E)
        bndvi = (N_ - B)  / (N_ + B  + E)
        rb    = R  / (B  + E)
        nr    = N_ / (R  + E)
        nre_  = N_ / (RE + E)
        mah   = np.sqrt((((mu - mu.mean(0)) / (sig.mean(0) + E))**2).mean(1))
        feats = np.c_[mu, sig, ndvi, ndre, exg, evi, bndvi, rb, nr, nre_, mah].astype(np.float32)

    # Labels: majority vote
    ac = np.bincount(sf, weights=(gtf == 2).astype(np.float64), minlength=n)
    nc = np.bincount(sf, weights=(gtf == 1).astype(np.float64), minlength=n)
    labels = (ac > nc).astype(np.int64)
    return feats, labels


# ─── Metrics ──────────────────────────────────────────────────────────────────
def cost_metric(y_true, y_pred, r_plus):
    FN = int(np.sum((y_true == 1) & (y_pred == 0)))
    FP = int(np.sum((y_true == 0) & (y_pred == 1)))
    return (r_plus * FN + FP) / len(y_true) * 100


# ─── cSVM ─────────────────────────────────────────────────────────────────────
def train_csvm(X_lab, y_lab, r_plus):
    clf = SVC(kernel='rbf', C=1.0, gamma='scale',
              class_weight={1: float(r_plus), 0: 1.0}, random_state=42)
    clf.fit(X_lab, y_lab)
    return clf


# ─── cAL query (Stage 1 + Stage 2) ──────────────────────────────────────────
def query_cal(clf, X_unlab, n, r_plus):
    cap   = min(len(X_unlab), POOL_CAP)
    idx0  = np.random.choice(len(X_unlab), cap, replace=False)
    Xu    = X_unlab[idx0]
    dec   = clf.decision_function(Xu)
    # Risk-sensitive uncertainty score (Eq. 8 in paper)
    u     = np.where(dec < 0, r_plus * np.abs(dec), np.abs(dec))
    M     = min(5 * n, len(Xu))
    top_M = np.argsort(u)[:M]
    nys   = Nystroem(kernel='rbf', gamma=1.0 / Xu.shape[1],
                     n_components=min(64, M), random_state=42).fit(Xu[top_M])
    Xk    = nys.transform(Xu[top_M])
    km    = MiniBatchKMeans(n_clusters=n, n_init=3, random_state=42).fit(Xk)
    unc   = u[top_M]
    chosen = []
    for c in range(n):
        mask = (km.labels_ == c)
        if mask.sum() == 0:
            continue
        chosen.append(idx0[top_M[np.where(mask)[0][np.argmin(unc[mask])]]])
    chosen = list(dict.fromkeys(chosen))
    if len(chosen) < n:
        rest = [idx0[i] for i in top_M if idx0[i] not in set(chosen)]
        chosen.extend(rest[:n - len(chosen)])
    return np.array(chosen[:n])


# ─── Single AL run ────────────────────────────────────────────────────────────
def al_run(X_pool, y_pool, X_test, y_test, r_plus, seed, budget):
    rng = np.random.default_rng(seed)
    n_pool = len(X_pool)

    anom_idx = np.where(y_pool == 1)[0]
    norm_idx = np.where(y_pool == 0)[0]
    k = N_INIT // 2
    labeled = set(
        rng.choice(anom_idx, min(k, len(anom_idx)), replace=False).tolist() +
        rng.choice(norm_idx, min(k, len(norm_idx)), replace=False).tolist()
    )

    n_labs, costs, fnrs = [], [], []
    queried = 0

    while queried <= budget:
        labeled_arr   = np.array(sorted(labeled))
        unlabeled_arr = np.array([i for i in range(n_pool) if i not in labeled])
        X_lab = X_pool[labeled_arr]
        y_lab = y_pool[labeled_arr]

        if len(np.unique(y_lab)) < 2:
            n_labs.append(len(labeled)); costs.append(np.nan); fnrs.append(np.nan)
        else:
            clf   = train_csvm(X_lab, y_lab, r_plus)
            dec   = clf.decision_function(X_test)
            y_pred = (dec >= 0).astype(int)
            costs.append(cost_metric(y_test, y_pred, r_plus))
            FN = int(np.sum((y_test == 1) & (y_pred == 0)))
            fnrs.append(FN / max(1, int(y_test.sum())))
            n_labs.append(len(labeled))

        if queried >= budget or len(unlabeled_arr) < BATCH_Q:
            break

        np.random.seed(int(seed) + queried)
        q = min(BATCH_Q, len(unlabeled_arr))
        X_unl = X_pool[unlabeled_arr]

        if len(np.unique(y_lab)) < 2:
            local_q = np.random.choice(len(X_unl), q, replace=False)
        else:
            clf_q   = train_csvm(X_lab, y_lab, r_plus)
            local_q = query_cal(clf_q, X_unl, q, r_plus)

        for li in local_q:
            labeled.add(int(unlabeled_arr[li]))
        queried += q

    return np.array(n_labs), np.array(costs), np.array(fnrs)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    results = {}  # {mode: {scene: {fnr_final, recall_final, cost_final}}}

    for mode in FEATURE_MODES:
        print(f"\n=== Feature mode: {mode.upper()} ===")
        results[mode] = {}

        for scene, (w, h) in IMAGES.items():
            print(f"  Loading {scene}...", end=" ", flush=True)
            img, gt, seg = load_scene(scene, w, h)
            feats, labels = extract_features(img, gt, seg, mode=mode)
            del img, gt  # free memory

            # Stratified 70/30 split
            rng = np.random.default_rng(SEED_BASE)
            anom_idx = np.where(labels == 1)[0]
            norm_idx = np.where(labels == 0)[0]
            n_pool_a = int(0.7 * len(anom_idx))
            n_pool_n = int(0.7 * len(norm_idx))
            pool_idx = np.concatenate([
                rng.choice(anom_idx, n_pool_a, replace=False),
                rng.choice(norm_idx, n_pool_n, replace=False)
            ])
            test_idx = np.array([i for i in range(len(labels)) if i not in set(pool_idx.tolist())])

            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_pool = scaler.fit_transform(feats[pool_idx])
            X_test = scaler.transform(feats[test_idx])
            y_pool = labels[pool_idx]
            y_test = labels[test_idx]

            budget = min(MAX_BUDGET, int(MAX_PCT * len(pool_idx)))

            run_fnrs, run_costs = [], []
            for run in range(N_RUNS):
                seed = SEED_BASE + run * 100 + SCENE_SEED_OFFSET[scene]
                n_labs, costs, fnrs = al_run(X_pool, y_pool, X_test, y_test,
                                             r_plus=R_PLUS, seed=seed, budget=budget)
                run_costs.append(costs[-1] if len(costs) > 0 else np.nan)
                run_fnrs.append(fnrs[-1]  if len(fnrs)  > 0 else np.nan)

            fnr_mean  = float(np.nanmean(run_fnrs))
            recall    = 1.0 - fnr_mean
            cost_mean = float(np.nanmean(run_costs))
            results[mode][scene] = {
                "fnr":    round(fnr_mean, 3),
                "recall": round(recall,   3),
                "cost":   round(cost_mean, 2),
            }
            print(f"FNR={fnr_mean:.3f}  Recall={recall:.3f}  C(r+=4)={cost_mean:.2f}")

    # ─── Save results ─────────────────────────────────────────────────────────
    out_json = OUT / "ablation_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_json}")

    # ─── Print summary table ──────────────────────────────────────────────────
    print("\n" + "="*72)
    print("SUMMARY TABLE  —  cAL (r+=4), Final Budget")
    print(f"{'Scene':<8}", end="")
    for mode in FEATURE_MODES:
        label = {"5d": "Raw-5D", "10d": "Raw-10D", "19d": "Full-19D"}[mode]
        print(f"  {label:>10} FNR  {label:>10} Rec", end="")
    print()
    print("-" * 72)
    for scene in SCENES:
        print(f"{scene.upper():<8}", end="")
        for mode in FEATURE_MODES:
            r = results[mode][scene]
            print(f"  {r['fnr']:>14.3f}  {r['recall']:>13.3f}", end="")
        print()
    print("="*72)

    # ─── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(SCENES))
    width = 0.25
    colors = {"5d": "#888888", "10d": "#1f77b4", "19d": "#9467bd"}
    labels_map = {"5d": "Raw-5D (5 features)", "10d": "Raw-10D (10 features)",
                  "19d": "Full-19D (proposed)"}

    for i, mode in enumerate(FEATURE_MODES):
        fnrs   = [results[mode][s]["fnr"]    for s in SCENES]
        recalls= [results[mode][s]["recall"] for s in SCENES]
        axes[0].bar(x + i*width, fnrs,    width, label=labels_map[mode], color=colors[mode], alpha=0.85)
        axes[1].bar(x + i*width, recalls, width, label=labels_map[mode], color=colors[mode], alpha=0.85)

    for ax, title, ylabel in zip(axes,
                                  ["False Negative Rate (lower is better)",
                                   "Anomaly Recall (higher is better)"],
                                  ["FNR", "Recall"]):
        ax.set_xticks(x + width)
        ax.set_xticklabels([s.upper() for s in SCENES])
        ax.set_xlabel("Scene")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1)

    fig.suptitle(f"Feature Set Ablation — cAL (r+=4), {N_RUNS} runs", fontsize=11)
    plt.tight_layout()
    out_fig = OUT / "ablation_feature_sets.png"
    plt.savefig(out_fig, dpi=150, bbox_inches='tight')
    print(f"Figure saved to {out_fig}")


if __name__ == "__main__":
    main()
