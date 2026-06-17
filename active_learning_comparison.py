#!/usr/bin/env python3
"""
Pool-Based Active Learning Comparison
=======================================
"First UAV Multispectral Riparian Zone Active Learning Study"

Query strategies compared:
  1. Random              — baseline
  2. Entropy             — highest prediction entropy
  3. Margin              — smallest class-probability margin
  4. Least Confidence    — 1 - P(most likely class)
  5. CoreSet             — greedy furthest-first in feature space
  6. BADGE               — gradient-embedding diversity (k-means++ seeding)
  7. RX-Guided           — Reed-Xiaoli anomaly score × model uncertainty
  8. Unc+KernelKMeans    — top-uncertain pool → kernel k-means diversity

Experimental protocol:
  - 4 scenes (z1, z2, e1, e2); features: 19-D per superpixel
  - Each scene split 70/30 → labeled pool / fixed test set
  - Seed: 20 initial labeled (10 anomaly + 10 normal, stratified)
  - Budget: B=50 queries per step, up to 5% of pool or 600 queries
  - R=5 independent runs (different random seeds) → mean ± std
  - Classifier: Random Forest (100 trees, balanced class weight)
  - Metrics: AUC-ROC, Average Precision (AP), F1 @0.5 threshold

Output: learning curves, label-efficiency table, band-importance figure
"""

import sys, warnings, time, json
warnings.filterwarnings('ignore')

import numpy as np
from pathlib import Path
from itertools import combinations

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              f1_score, cohen_kappa_score)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.kernel_approximation import Nystroem
from sklearn.cluster import KMeans, MiniBatchKMeans

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec

# ─── Config ──────────────────────────────────────────────────────────────────

_ROOT  = Path(__file__).resolve().parent
BASE   = _ROOT / "data"
OUT    = _ROOT / "results" / "al"
OUT.mkdir(parents=True, exist_ok=True)

IMAGES = {"z1": (3807, 2141), "z2": (2081, 957),
          "e1": (3629,  961), "e2": (1094,  707)}
BANDS  = ["Blue(475)", "Green(560)", "Red(668)", "RedEdge(717)", "NIR(840)"]

N_INIT     = 20      # initial labeled superpixels (10 anom + 10 normal)
BATCH_Q    = 50      # queries per AL step
MAX_BUDGET = 600     # maximum total queries (beyond seed)
MAX_PCT    = 0.05    # also cap at 5% of pool
N_RUNS     = 5       # independent runs per strategy×scene
RF_TREES   = 100     # random forest size
POOL_CAP   = 15_000  # max unlabeled pool size for expensive strategies
SEED_BASE  = 2024

# Strategy registry — (name, colour, linestyle)
STRATEGIES = [
    ("Random",           "#888888", "-"),
    ("Entropy",          "#2196F3", "-"),
    ("Margin",           "#4CAF50", "-"),
    ("LeastConf",        "#FF9800", "-"),
    ("CoreSet",          "#F44336", "--"),
    ("BADGE",            "#9C27B0", "--"),
    ("RX-Guided",        "#00BCD4", "-."),
    ("Unc+KernelKMeans", "#E91E63", "-."),
]
STRAT_NAMES = [s[0] for s in STRATEGIES]

# ─── 1. Data Loading ─────────────────────────────────────────────────────────

def load_scene(name, w, h):
    with open(BASE/name/f"{name}.raw","rb") as f:
        f.read(12)
        img = np.frombuffer(f.read(), dtype=np.uint32).reshape(h,w,5).astype(np.float32)
    with open(BASE/name/f"{name}_gt.pgm","rb") as f:
        f.readline(); f.readline(); f.readline()
        raw = f.read()
        gt  = np.frombuffer(raw[:h*w], dtype=np.uint8).reshape(h,w)
    with open(BASE/name/f"{name}_seg.raw","rb") as f:
        f.read(8)
        seg = np.frombuffer(f.read(), dtype=np.uint32).reshape(h,w)
    return img, gt, seg


def extract_features(img, gt, seg):
    """19-D per-superpixel feature vector + binary label."""
    n       = int(seg.max())+1
    sf      = seg.ravel().astype(np.int64)
    imgf    = img.reshape(-1,5).astype(np.float64)
    gtf     = gt.ravel().astype(np.int64)
    cnt     = np.bincount(sf, minlength=n).clip(1).astype(np.float64)

    mu = np.zeros((n,5)); sq = np.zeros((n,5))
    for b in range(5):
        mu[:,b] = np.bincount(sf, weights=imgf[:,b], minlength=n)/cnt
        sq[:,b] = np.bincount(sf, weights=imgf[:,b]**2, minlength=n)/cnt
    sig = np.sqrt(np.clip(sq-mu**2, 0, None))

    E = 1e-6
    B,G,R,RE,N_ = mu[:,0],mu[:,1],mu[:,2],mu[:,3],mu[:,4]
    ndvi = (N_-R)/(N_+R+E);   ndre  = (N_-RE)/(N_+RE+E)
    exg  = 2*G-R-B;            evi   = 2.5*(N_-R)/(N_+6*R-7.5*B+1+E)
    bndvi= (N_-B)/(N_+B+E);   rb    = R/(B+E)
    nr   = N_/(R+E);            nre   = N_/(RE+E)
    gmu  = mu.mean(0); gsd = sig.mean(0)+E
    mah  = np.sqrt((((mu-gmu)/gsd)**2).mean(1))

    feats  = np.c_[mu, sig, ndvi, ndre, exg, evi, bndvi, rb, nr, nre, mah].astype(np.float32)
    ac = np.bincount(sf, weights=(gtf==2).astype(np.float64), minlength=n)
    nc = np.bincount(sf, weights=(gtf==1).astype(np.float64), minlength=n)
    labels = (ac > nc).astype(np.int64)
    return feats, labels


# ─── 2. RX Detector ──────────────────────────────────────────────────────────

def rx_scores(X):
    """Reed-Xiaoli (Mahalanobis-based) anomaly scores."""
    mu   = X.mean(0)
    cov  = np.cov(X.T) + 1e-6*np.eye(X.shape[1])
    Ci   = np.linalg.pinv(cov)
    diff = X - mu
    return np.einsum('ij,jk,ik->i', diff, Ci, diff)


# ─── 3. Query Strategies ─────────────────────────────────────────────────────

def query_random(clf, X_unlab, X_lab, n):
    return np.random.choice(len(X_unlab), n, replace=False)


def query_entropy(clf, X_unlab, X_lab, n):
    p   = clf.predict_proba(X_unlab)
    ent = -(p * np.log(p + 1e-12)).sum(1)
    return np.argsort(ent)[::-1][:n]


def query_margin(clf, X_unlab, X_lab, n):
    p   = np.sort(clf.predict_proba(X_unlab), axis=1)[:,::-1]
    mar = p[:,0] - p[:,1]
    return np.argsort(mar)[:n]


def query_leastconf(clf, X_unlab, X_lab, n):
    p    = clf.predict_proba(X_unlab).max(1)
    conf = 1 - p
    return np.argsort(conf)[::-1][:n]


def query_coreset(clf, X_unlab, X_lab, n):
    """Greedy furthest-first (capped pool for speed)."""
    cap  = min(len(X_unlab), POOL_CAP)
    idx0 = np.random.choice(len(X_unlab), cap, replace=False)
    Xu   = X_unlab[idx0]

    selected = []
    X_sel    = X_lab.copy()
    min_dist  = pairwise_distances(Xu, X_sel).min(axis=1)

    for _ in range(n):
        q = int(np.argmax(min_dist))
        selected.append(idx0[q])
        d_new    = pairwise_distances(Xu, Xu[q:q+1]).ravel()
        min_dist = np.minimum(min_dist, d_new)

    return np.array(selected)


def query_badge(clf, X_unlab, X_lab, n):
    """
    BADGE: gradient embeddings via (p_y_hat − 1_{y=y_hat}) × x,
    then k-means++ seeding for diversity.
    Approximated for non-differentiable classifiers (RF).
    """
    cap  = min(len(X_unlab), POOL_CAP)
    idx0 = np.random.choice(len(X_unlab), cap, replace=False)
    Xu   = X_unlab[idx0]

    p      = clf.predict_proba(Xu)          # (M,2)
    y_hat  = p.argmax(1)
    # Gradient magnitude: uncertainty weight on feature vector
    unc    = np.array([p[i, y_hat[i]] - 1 for i in range(len(Xu))])  # < 0
    embeds = unc[:,None] * Xu               # (M, 19)

    # k-means++ seeding
    centres = [embeds[np.random.randint(len(embeds))]]
    for _ in range(n-1):
        dists = np.array([min(np.linalg.norm(e-c)**2 for c in centres) for e in embeds])
        probs = dists / dists.sum()
        centres.append(embeds[np.random.choice(len(embeds), p=probs)])

    # Assign each centre to nearest embedding → pick actual index
    centres = np.array(centres)
    chosen  = set()
    for c in centres:
        d   = np.linalg.norm(embeds - c, axis=1)
        d_sorted = np.argsort(d)
        for j in d_sorted:
            if j not in chosen:
                chosen.add(j); break

    local_idx = np.array(list(chosen))[:n]
    return idx0[local_idx]


def query_rx_guided(clf, X_unlab, X_lab, n, rx_sc):
    """
    Combine RX anomaly score (prior anomaly belief) with model uncertainty.
    score = rx_normalised × entropy
    Query highest combined scores.
    """
    p    = clf.predict_proba(X_unlab)
    ent  = -(p * np.log(p + 1e-12)).sum(1)
    rx_n = (rx_sc - rx_sc.min()) / (rx_sc.max() - rx_sc.min() + 1e-8)
    combined = rx_n * ent
    return np.argsort(combined)[::-1][:n]


def query_unc_kernelkmeans(clf, X_unlab, X_lab, n):
    """
    1. Select top 5×n uncertain points (entropy).
    2. Map them into RBF kernel feature space (Nystroem).
    3. Run k-means to find n diverse clusters.
    4. Return one representative per cluster (closest to centroid).
    """
    p    = clf.predict_proba(X_unlab)
    ent  = -(p * np.log(p + 1e-12)).sum(1)
    top_k = min(5*n, len(X_unlab))
    top_idx = np.argsort(ent)[::-1][:top_k]
    Xu_top  = X_unlab[top_idx]

    n_comp = min(64, top_k)
    gamma  = 1.0 / X_unlab.shape[1]
    nys    = Nystroem(kernel='rbf', gamma=gamma, n_components=n_comp,
                      random_state=42).fit(Xu_top)
    Xk     = nys.transform(Xu_top)

    km = MiniBatchKMeans(n_clusters=n, n_init=3, random_state=42).fit(Xk)

    chosen = []
    for c in range(n):
        mask = (km.labels_ == c)
        if mask.sum() == 0:
            continue
        dists = np.linalg.norm(Xk[mask] - km.cluster_centers_[c], axis=1)
        chosen.append(top_idx[np.where(mask)[0][np.argmin(dists)]])

    # Fill remaining if any cluster was empty
    chosen = list(dict.fromkeys(chosen))    # deduplicate
    if len(chosen) < n:
        rest = [i for i in top_idx if i not in chosen]
        chosen.extend(rest[:n-len(chosen)])

    return np.array(chosen[:n])


QUERY_FNS = {
    "Random":           query_random,
    "Entropy":          query_entropy,
    "Margin":           query_margin,
    "LeastConf":        query_leastconf,
    "CoreSet":          query_coreset,
    "BADGE":            query_badge,
    "RX-Guided":        query_rx_guided,     # needs extra arg
    "Unc+KernelKMeans": query_unc_kernelkmeans,
}


# ─── 4. Single AL Run ────────────────────────────────────────────────────────

def al_run(strategy, X_pool, y_pool, X_test, y_test,
           rx_pool, seed, budget, batch):
    """
    Returns arrays: (n_labeled_per_step, auc_per_step, ap_per_step, f1_per_step)
    """
    rng = np.random.default_rng(seed)
    n_pool = len(X_pool)

    # Stratified initialisation
    anom_idx  = np.where(y_pool == 1)[0]
    norm_idx  = np.where(y_pool == 0)[0]
    k_init    = N_INIT // 2
    init_a    = rng.choice(anom_idx, min(k_init, len(anom_idx)), replace=False)
    init_n    = rng.choice(norm_idx, min(k_init, len(norm_idx)), replace=False)
    labeled   = set(init_a.tolist() + init_n.tolist())

    aucs, aps, f1s, n_labs = [], [], [], []
    queried  = 0

    while queried <= budget:
        labeled_arr  = np.array(sorted(labeled))
        unlabeled_arr= np.array([i for i in range(n_pool) if i not in labeled])

        X_lab = X_pool[labeled_arr];   y_lab = y_pool[labeled_arr]
        X_unl = X_pool[unlabeled_arr]

        # Train RF
        clf = RandomForestClassifier(n_estimators=RF_TREES,
                                     class_weight='balanced',
                                     max_features='sqrt',
                                     n_jobs=-1, random_state=int(seed))
        if len(np.unique(y_lab)) < 2:
            aucs.append(np.nan); aps.append(np.nan)
            f1s.append(np.nan);  n_labs.append(len(labeled))
        else:
            clf.fit(X_lab, y_lab)
            prob_test = clf.predict_proba(X_test)[:,1]
            pred_test = (prob_test >= 0.5).astype(int)
            auc = roc_auc_score(y_test, prob_test) if len(np.unique(y_test))>1 else np.nan
            ap  = average_precision_score(y_test, prob_test)
            f1  = f1_score(y_test, pred_test, zero_division=0)
            aucs.append(auc); aps.append(ap); f1s.append(f1)
            n_labs.append(len(labeled))

        if queried >= budget or len(unlabeled_arr) < batch:
            break

        # Query
        np.random.seed(int(seed) + queried)
        q = min(batch, len(unlabeled_arr))

        if strategy == "RX-Guided":
            rx_unl = rx_pool[unlabeled_arr]
            local_q = QUERY_FNS[strategy](clf, X_unl, X_lab, q, rx_unl)
        else:
            local_q = QUERY_FNS[strategy](clf, X_unl, X_lab, q)

        for li in local_q:
            labeled.add(int(unlabeled_arr[li]))
        queried += q

    return (np.array(n_labs), np.array(aucs),
            np.array(aps),    np.array(f1s))


# ─── 5. Experiment per Scene ─────────────────────────────────────────────────

def run_scene_experiment(name, X, y, scaler):
    """Run all strategies × N_RUNS on a single scene."""
    np.random.seed(SEED_BASE)
    n = len(X)
    # 70/30 pool/test split (stratified by class)
    anom_idx = np.where(y==1)[0]; norm_idx = np.where(y==0)[0]
    np.random.shuffle(anom_idx); np.random.shuffle(norm_idx)
    cut_a = int(0.7*len(anom_idx)); cut_n = int(0.7*len(norm_idx))
    pool_idx = np.concatenate([anom_idx[:cut_a], norm_idx[:cut_n]])
    test_idx  = np.concatenate([anom_idx[cut_a:], norm_idx[cut_n:]])

    Xsc = scaler.transform(X).astype(np.float32)
    X_pool, y_pool = Xsc[pool_idx], y[pool_idx]
    X_test, y_test = Xsc[test_idx], y[test_idx]
    rx_pool        = rx_scores(X_pool).astype(np.float32)

    budget = min(MAX_BUDGET, int(MAX_PCT * len(pool_idx)))
    n_anom = int(y_pool.sum())
    n_norm = int((y_pool==0).sum())
    print(f"  Pool: {len(pool_idx):,} SPs  (anom={n_anom:,}={100*n_anom/len(pool_idx):.1f}%)"
          f"  Test: {len(test_idx):,}  Budget: {budget}")

    scene_results = {}
    for sname, scol, sls in STRATEGIES:
        t0 = time.time()
        run_aucs, run_aps, run_f1s, run_labs = [], [], [], []

        for run_id in range(N_RUNS):
            seed = SEED_BASE + run_id * 100 + hash(name) % 100
            nl, au, ap, f1 = al_run(sname, X_pool, y_pool, X_test, y_test,
                                     rx_pool, seed, budget, BATCH_Q)
            run_labs.append(nl); run_aucs.append(au)
            run_aps.append(ap);   run_f1s.append(f1)

        # Interpolate to common x-axis
        x_common = np.arange(N_INIT, budget+N_INIT+1, BATCH_Q)
        auc_interp = np.array([np.interp(x_common, nl, au)
                                for nl,au in zip(run_labs,run_aucs)])
        ap_interp  = np.array([np.interp(x_common, nl, ap)
                                for nl,ap in zip(run_labs,run_aps)])
        f1_interp  = np.array([np.interp(x_common, nl, f1)
                                for nl,f1 in zip(run_labs,run_f1s)])

        scene_results[sname] = {
            "x": x_common.tolist(),
            "auc_mean": np.nanmean(auc_interp, 0).tolist(),
            "auc_std":  np.nanstd(auc_interp,  0).tolist(),
            "ap_mean":  np.nanmean(ap_interp,  0).tolist(),
            "ap_std":   np.nanstd(ap_interp,   0).tolist(),
            "f1_mean":  np.nanmean(f1_interp,  0).tolist(),
            "f1_std":   np.nanstd(f1_interp,   0).tolist(),
        }
        dt = time.time()-t0
        final_auc = np.nanmean(auc_interp, 0)[-1]
        print(f"    {sname:<20} final AUC={final_auc:.4f}  [{dt:.1f}s]")

    return scene_results, x_common


# ─── 6. Full-data Baseline ───────────────────────────────────────────────────

def full_data_baseline(name, X, y, scaler):
    """RF trained on 70% pool, tested on 30% — upper bound."""
    Xsc = scaler.transform(X).astype(np.float32)
    n   = len(X)
    anom_idx = np.where(y==1)[0]; norm_idx = np.where(y==0)[0]
    np.random.shuffle(anom_idx); np.random.shuffle(norm_idx)
    cut_a = int(0.7*len(anom_idx)); cut_n = int(0.7*len(norm_idx))
    pool_idx = np.concatenate([anom_idx[:cut_a], norm_idx[:cut_n]])
    test_idx  = np.concatenate([anom_idx[cut_a:], norm_idx[cut_n:]])
    X_pool, y_pool = Xsc[pool_idx], y[pool_idx]
    X_test, y_test = Xsc[test_idx], y[test_idx]
    clf = RandomForestClassifier(RF_TREES, class_weight='balanced',
                                 max_features='sqrt', n_jobs=-1, random_state=42)
    clf.fit(X_pool, y_pool)
    prob = clf.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, prob) if len(np.unique(y_test))>1 else np.nan
    ap  = average_precision_score(y_test, prob)
    return auc, ap, clf.feature_importances_


# ─── 7. Label Efficiency Table ───────────────────────────────────────────────

def label_efficiency(scene_results, x_common, full_auc, thresholds=(0.90,0.95,0.99)):
    """
    For each strategy, find minimum n_labeled to reach
    threshold × full_data_AUC.
    """
    table = {}
    for sname, _, _ in STRATEGIES:
        aucs = np.array(scene_results[sname]["auc_mean"])
        row  = {}
        for thr in thresholds:
            target = thr * full_auc
            reached = np.where(aucs >= target)[0]
            row[f"{int(thr*100)}%"] = int(x_common[reached[0]]) if len(reached)>0 else None
        table[sname] = row
    return table


# ─── 8. Visualisation ────────────────────────────────────────────────────────

def plot_learning_curves(all_results, all_full_auc, out_dir):
    """4-panel plot (one per scene) of AUC learning curves."""
    scene_names = list(all_results.keys())
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=False)
    axes = axes.ravel()

    for si, sname_k in enumerate(scene_names):
        ax  = axes[si]
        sr  = all_results[sname_k]
        x   = np.array(sr[STRAT_NAMES[0]]["x"])
        fa  = all_full_auc[sname_k]

        for strat, col, ls in STRATEGIES:
            auc_m = np.array(sr[strat]["auc_mean"])
            auc_s = np.array(sr[strat]["auc_std"])
            ax.plot(x, auc_m, color=col, ls=ls, lw=1.8, label=strat)
            ax.fill_between(x, auc_m-auc_s, auc_m+auc_s,
                            color=col, alpha=0.10)

        ax.axhline(fa, ls=':', color='black', lw=1.2, alpha=0.6,
                   label=f"Full data ({fa:.3f})")
        ax.set_title(f"Scene {sname_k.upper()}", fontweight='bold')
        ax.set_xlabel("Labeled superpixels")
        ax.set_ylabel("AUC-ROC")
        ax.set_ylim(0.45, 1.02)
        ax.grid(True, alpha=0.3)
        if si == 0:
            ax.legend(fontsize=7, ncol=2, loc='lower right')

    plt.suptitle("Pool-Based Active Learning: AUC-ROC Learning Curves\n"
                 "UAV Multispectral Riparian Zone Anomaly Detection",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    p = out_dir / "al_learning_curves_auc.png"
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  → {p.name}")


def plot_ap_curves(all_results, all_full_ap, out_dir):
    """4-panel Average Precision learning curves."""
    scene_names = list(all_results.keys())
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    for si, sname_k in enumerate(scene_names):
        ax = axes[si]
        sr = all_results[sname_k]
        x  = np.array(sr[STRAT_NAMES[0]]["x"])
        fa = all_full_ap[sname_k]

        for strat, col, ls in STRATEGIES:
            ap_m = np.array(sr[strat]["ap_mean"])
            ap_s = np.array(sr[strat]["ap_std"])
            ax.plot(x, ap_m, color=col, ls=ls, lw=1.8, label=strat)
            ax.fill_between(x, ap_m-ap_s, ap_m+ap_s, color=col, alpha=0.10)

        ax.axhline(fa, ls=':', color='black', lw=1.2, alpha=0.6,
                   label=f"Full data ({fa:.3f})")
        ax.set_title(f"Scene {sname_k.upper()}", fontweight='bold')
        ax.set_xlabel("Labeled superpixels")
        ax.set_ylabel("Average Precision")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        if si == 0:
            ax.legend(fontsize=7, ncol=2, loc='lower right')

    plt.suptitle("Pool-Based Active Learning: Average Precision Learning Curves\n"
                 "UAV Multispectral Riparian Zone Anomaly Detection",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    p = out_dir / "al_learning_curves_ap.png"
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  → {p.name}")


def plot_aggregated(all_results, out_dir):
    """Mean AUC/AP across all 4 scenes per strategy."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x_ref = np.array(all_results[list(all_results.keys())[0]][STRAT_NAMES[0]]["x"])

    for metric, ylbl in [("auc","AUC-ROC"), ("ap","Avg Precision")]:
        for strat, col, ls in STRATEGIES:
            means = []
            for sc in all_results.values():
                means.append(sc[strat][f"{metric}_mean"])
            mn = np.nanmean(means, 0)
            sd = np.nanstd(means, 0)
            axes[["auc","ap"].index(metric)].plot(
                x_ref, mn, color=col, ls=ls, lw=2.0, label=strat)
            axes[["auc","ap"].index(metric)].fill_between(
                x_ref, mn-sd, mn+sd, color=col, alpha=0.08)

    for i, ylbl in enumerate(["AUC-ROC", "Average Precision"]):
        axes[i].set_xlabel("Labeled superpixels")
        axes[i].set_ylabel(ylbl)
        axes[i].set_title(f"Aggregated {ylbl}\n(Mean ± std across 4 scenes)")
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(fontsize=8, ncol=2)

    plt.suptitle("Pool-Based Active Learning — Aggregated Performance\n"
                 "UAV Multispectral Riparian Zone (Galicia, Spain)",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    p = out_dir / "al_aggregated.png"
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  → {p.name}")


def plot_final_bar(all_results, out_dir, step_idx=-1):
    """Bar chart of final AUC and AP per strategy, averaged across scenes."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    col_map = {s[0]:s[1] for s in STRATEGIES}

    for mi, (metric, _title) in enumerate([("auc_mean","AUC-ROC"), ("ap_mean","Avg Precision")]):
        ax = axes[mi]
        vals_by_strat = {}
        for strat in STRAT_NAMES:
            scene_vals = [np.array(sc[strat][metric])[step_idx]
                          for sc in all_results.values()]
            vals_by_strat[strat] = (np.nanmean(scene_vals), np.nanstd(scene_vals))

        names  = list(vals_by_strat.keys())
        means  = [vals_by_strat[n][0] for n in names]
        stds   = [vals_by_strat[n][1] for n in names]
        colors = [col_map[n] for n in names]

        bars = ax.bar(names, means, color=colors, alpha=0.85,
                      yerr=stds, capsize=4)
        ax.set_xticklabels(names, rotation=30, ha='right')
        ax.set_ylabel(metric.split("_")[0].upper() + "-ROC" if "auc" in metric else "Average Precision")
        ax.set_title(f"Final {metric.split('_')[0].upper()} (at max budget)\nMean ± std across 4 scenes")
        ax.set_ylim(0, 1.1); ax.grid(True, axis='y', alpha=0.3)

        # Annotate bars
        for bar, m in zip(bars, means):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                    f"{m:.3f}", ha='center', va='bottom', fontsize=8)

    plt.suptitle("Active Learning Strategy Comparison — Final Performance",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    p = out_dir / "al_final_bar.png"
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  → {p.name}")


def plot_label_efficiency_heatmap(eff_all, out_dir):
    """
    Heatmap: rows=strategies, cols=scenes×thresholds,
    value=n_labeled to reach threshold of full-data AUC.
    """
    scenes     = list(eff_all.keys())
    thresholds = ["90%", "95%", "99%"]
    col_labels = [f"{sc.upper()}\n{t}" for sc in scenes for t in thresholds]

    matrix = np.full((len(STRAT_NAMES), len(col_labels)), np.nan)
    for si, strat in enumerate(STRAT_NAMES):
        col = 0
        for sc in scenes:
            for t in thresholds:
                v = eff_all[sc].get(strat, {}).get(t, None)
                matrix[si, col] = v if v is not None else np.nan
                col += 1

    fig, ax = plt.subplots(figsize=(14, 5))
    vmax = np.nanmax(matrix)
    im   = ax.imshow(np.nan_to_num(matrix, nan=vmax*1.1),
                     cmap='RdYlGn_r', aspect='auto', vmin=N_INIT, vmax=vmax)
    plt.colorbar(im, ax=ax, label='Labeled superpixels needed')

    ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels, fontsize=8)
    ax.set_yticks(range(len(STRAT_NAMES))); ax.set_yticklabels(STRAT_NAMES)
    ax.set_title("Label Efficiency: Queries needed to reach X% of full-data AUC\n"
                 "(lower = more efficient; NaN = threshold not reached)",
                 fontweight='bold')

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i,j]
            txt = str(int(v)) if not np.isnan(v) else "×"
            ax.text(j, i, txt, ha='center', va='center', fontsize=7,
                    color='white' if v > vmax*0.6 else 'black')

    plt.tight_layout()
    p = out_dir / "al_label_efficiency.png"
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  → {p.name}")


def plot_rx_vs_entropy(all_results, out_dir):
    """Direct comparison: RX-Guided vs Entropy vs Unc+KernelKMeans vs Random."""
    highlight = ["Random", "Entropy", "RX-Guided", "Unc+KernelKMeans"]
    col_map   = {s[0]: (s[1], s[2]) for s in STRATEGIES}
    scene_names = list(all_results.keys())

    fig, axes = plt.subplots(1, len(scene_names), figsize=(16, 4), sharey=False)
    for si, (sname_k, ax) in enumerate(zip(scene_names, axes)):
        sr = all_results[sname_k]
        x  = np.array(sr[highlight[0]]["x"])
        for strat in highlight:
            col, ls = col_map[strat]
            m = np.array(sr[strat]["auc_mean"])
            s = np.array(sr[strat]["auc_std"])
            ax.plot(x, m, color=col, ls=ls, lw=2.2, label=strat)
            ax.fill_between(x, m-s, m+s, color=col, alpha=0.12)
        ax.set_title(f"Scene {sname_k.upper()}", fontweight='bold')
        ax.set_xlabel("Labeled superpixels")
        if si == 0:
            ax.set_ylabel("AUC-ROC")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.45, 1.02)
    axes[0].legend(fontsize=9)
    plt.suptitle("Novel vs. Baseline Strategies — AUC-ROC Comparison",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    p = out_dir / "al_novel_comparison.png"
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  → {p.name}")


# ─── 9. Main ─────────────────────────────────────────────────────────────────

def main():
    print("="*65)
    print("  Pool-Based Active Learning Comparison")
    print("  UAV Multispectral Riparian Zone — 8 Query Strategies")
    print("="*65)

    # ── Load & featurise ──────────────────────────────────────────────────
    print("\n[1/4] Loading scenes …")
    scenes_X, scenes_y = {}, {}
    for name, (w, h) in IMAGES.items():
        t0 = time.time()
        img, gt, seg = load_scene(name, w, h)
        X, y = extract_features(img, gt, seg)
        scenes_X[name] = X; scenes_y[name] = y
        print(f"  {name}: nodes={len(X):,}  anom={y.sum():,} ({100*y.mean():.1f}%)"
              f"  [{time.time()-t0:.1f}s]")

    # Shared scaler fitted on all data
    scaler = StandardScaler().fit(np.vstack(list(scenes_X.values())))

    # ── Full-data baselines ───────────────────────────────────────────────
    print("\n[2/4] Full-data baselines …")
    full_auc, full_ap = {}, {}
    for name in IMAGES:
        auc, ap, imp = full_data_baseline(name, scenes_X[name], scenes_y[name], scaler)
        full_auc[name] = auc; full_ap[name] = ap
        print(f"  {name}: AUC={auc:.4f}  AP={ap:.4f}")

    # ── AL experiments ────────────────────────────────────────────────────
    print("\n[3/4] Active learning experiments …")
    print(f"  Strategies: {STRAT_NAMES}")
    print(f"  N_INIT={N_INIT}  BATCH={BATCH_Q}  MAX={MAX_BUDGET}  RUNS={N_RUNS}\n")

    all_results = {}
    eff_all     = {}

    for name in IMAGES:
        print(f"\n─── Scene {name.upper()} ────────────────────────────")
        sr, x_common = run_scene_experiment(
            name, scenes_X[name], scenes_y[name], scaler)
        all_results[name] = sr

        # Label efficiency per scene
        eff = {}
        for strat in STRAT_NAMES:
            aucs   = np.array(sr[strat]["auc_mean"])
            row = {}
            for thr in [0.90, 0.95, 0.99]:
                target  = thr * full_auc[name]
                reached = np.where(aucs >= target)[0]
                row[f"{int(thr*100)}%"] = int(x_common[reached[0]]) if len(reached)>0 else None
            eff[strat] = row
        eff_all[name] = eff

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n[4/4] Results & Figures")
    print("\n  Final AUC (at max budget), mean across 4 scenes:")
    print(f"  {'Strategy':<22} {'AUC':>8} {'AP':>8}")
    print("  " + "-"*40)
    final_step = -1
    for strat in STRAT_NAMES:
        aucs = [np.array(all_results[sc][strat]["auc_mean"])[final_step]
                for sc in IMAGES]
        aps  = [np.array(all_results[sc][strat]["ap_mean"])[final_step]
                for sc in IMAGES]
        print(f"  {strat:<22} {np.nanmean(aucs):>8.4f} {np.nanmean(aps):>8.4f}")

    print("\n  Label efficiency (queries to reach 90% of full-data AUC):")
    print(f"  {'Strategy':<22}", end="")
    for sc in IMAGES: print(f"  {sc.upper():>6}", end="")
    print()
    print("  "+"-"*50)
    for strat in STRAT_NAMES:
        print(f"  {strat:<22}", end="")
        for sc in IMAGES:
            v = eff_all[sc][strat].get("90%")
            print(f"  {str(v) if v else '  N/A':>6}", end="")
        print()

    # Save JSON
    out_data = {
        "config": {"N_INIT": N_INIT, "BATCH_Q": BATCH_Q,
                   "MAX_BUDGET": MAX_BUDGET, "N_RUNS": N_RUNS,
                   "RF_TREES": RF_TREES},
        "full_data_baseline": {"auc": full_auc, "ap": full_ap},
        "results": {sc: {st: all_results[sc][st] for st in STRAT_NAMES}
                    for sc in IMAGES},
        "label_efficiency": eff_all
    }
    with open(OUT/"al_results.json","w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\n  Saved → {OUT}/al_results.json")

    # Figures
    print("\n  Generating figures …")
    plot_learning_curves(all_results, full_auc, OUT)
    plot_ap_curves(all_results, full_ap, OUT)
    plot_aggregated(all_results, OUT)
    plot_final_bar(all_results, OUT)
    plot_label_efficiency_heatmap(eff_all, OUT)
    plot_rx_vs_entropy(all_results, OUT)

    print("\nDONE ✓")


if __name__ == "__main__":
    main()
