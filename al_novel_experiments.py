#!/usr/bin/env python3
"""
Novel AL Experiments
=====================
Experiment 1 — Anomaly-Biased Initialization (RX warm-start)
    Compare: random stratified init (baseline) vs RX-guided init
    (top-N_INIT by RX score, no labels needed at init time)
    across all 4 strategies × 4 scenes.

Experiment 2 — Cross-Scene Transfer
    Train RF on full source-scene pool.
    Use source RF to select initial labeled set on target scene
    (top uncertain = warm start). Compare vs cold start (random init).
    All 4×3 = 12 directed source→target pairs.

Output: results/al_novel/
  exp1_results.json        — biased vs random init curves
  exp2_results.json        — transfer vs cold start curves
  exp1_*.png               — figures
  exp2_*.png               — figures
"""
import sys, warnings, time, json
warnings.filterwarnings('ignore')

import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import Nystroem
from sklearn.cluster import MiniBatchKMeans

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── Paths & Config ────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
BASE  = _ROOT / "data"
OUT   = _ROOT / "results" / "al_novel"
OUT.mkdir(parents=True, exist_ok=True)

IMAGES     = {"z1": (3807,2141), "z2": (2081,957), "e1": (3629,961), "e2": (1094,707)}
SCENES     = list(IMAGES.keys())
N_INIT     = 20
BATCH_Q    = 50
MAX_BUDGET = 600
MAX_PCT    = 0.05
N_RUNS     = 3
RF_TREES   = 100
SEED_BASE  = 2024

STRATEGIES = ["Random", "Entropy", "BADGE", "Unc+KernelKMeans"]
COLORS     = {
    "Random":           "#888888",
    "Entropy":          "#1f77b4",
    "BADGE":            "#9467bd",
    "Unc+KernelKMeans": "#e377c2",
}

# ─── Data Loading ──────────────────────────────────────────────────────────────
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
    n   = int(seg.max())+1
    sf  = seg.ravel().astype(np.int64)
    imgf= img.reshape(-1,5).astype(np.float64)
    gtf = gt.ravel().astype(np.int64)
    cnt = np.bincount(sf, minlength=n).clip(1).astype(np.float64)
    mu  = np.zeros((n,5)); sq = np.zeros((n,5))
    for b in range(5):
        mu[:,b] = np.bincount(sf, weights=imgf[:,b], minlength=n)/cnt
        sq[:,b] = np.bincount(sf, weights=imgf[:,b]**2, minlength=n)/cnt
    sig = np.sqrt(np.clip(sq-mu**2, 0, None))
    E   = 1e-6
    B,G,R,RE,N_ = mu[:,0],mu[:,1],mu[:,2],mu[:,3],mu[:,4]
    ndvi = (N_-R)/(N_+R+E); ndre = (N_-RE)/(N_+RE+E)
    exg  = 2*G-R-B; evi  = 2.5*(N_-R)/(N_+6*R-7.5*B+1+E)
    bndvi= (N_-B)/(N_+B+E); rb = R/(B+E); nr = N_/(R+E); nre = N_/(RE+E)
    gmu  = mu.mean(0); gsd = sig.mean(0)+E
    mah  = np.sqrt((((mu-gmu)/gsd)**2).mean(1))
    feats= np.c_[mu,sig,ndvi,ndre,exg,evi,bndvi,rb,nr,nre,mah].astype(np.float32)
    ac   = np.bincount(sf, weights=(gtf==2).astype(np.float64), minlength=n)
    nc   = np.bincount(sf, weights=(gtf==1).astype(np.float64), minlength=n)
    labels = (ac > nc).astype(np.int64)
    return feats, labels

def rx_scores(X):
    mu  = X.mean(0)
    cov = np.cov(X.T) + 1e-6*np.eye(X.shape[1])
    Ci  = np.linalg.pinv(cov)
    d   = X - mu
    return np.einsum('ij,jk,ik->i', d, Ci, d)

# ─── Query Functions ───────────────────────────────────────────────────────────
def query_random(clf, X_unlab, X_lab, n):
    return np.random.choice(len(X_unlab), n, replace=False)

def query_entropy(clf, X_unlab, X_lab, n):
    p   = clf.predict_proba(X_unlab)
    ent = -(p * np.log(p + 1e-12)).sum(1)
    return np.argsort(ent)[::-1][:n]

def query_badge(clf, X_unlab, X_lab, n):
    cap  = min(len(X_unlab), 15000)
    idx0 = np.random.choice(len(X_unlab), cap, replace=False)
    Xu   = X_unlab[idx0]
    p    = clf.predict_proba(Xu)
    y_hat= p.argmax(1)
    unc  = p[np.arange(len(Xu)), y_hat] - 1   # vectorized
    embeds = unc[:, None] * Xu                 # (M, D)

    # Vectorized k-means++ seeding
    first = np.random.randint(len(embeds))
    centre_idx = [first]
    min_dists = np.sum((embeds - embeds[first]) ** 2, axis=1)
    for _ in range(n - 1):
        probs = min_dists / (min_dists.sum() + 1e-12)
        new_c = np.random.choice(len(embeds), p=probs)
        centre_idx.append(new_c)
        d = np.sum((embeds - embeds[new_c]) ** 2, axis=1)
        min_dists = np.minimum(min_dists, d)

    # Assign each centre to nearest unique embedding
    chosen = set()
    for ci in centre_idx:
        d = np.sum((embeds - embeds[ci]) ** 2, axis=1)
        for j in np.argsort(d):
            if j not in chosen:
                chosen.add(j); break
    return idx0[np.array(list(chosen))[:n]]

def query_unc_kernelkmeans(clf, X_unlab, X_lab, n):
    p    = clf.predict_proba(X_unlab)
    ent  = -(p * np.log(p + 1e-12)).sum(1)
    top_k= min(5*n, len(X_unlab))
    tidx = np.argsort(ent)[::-1][:top_k]
    Xt   = X_unlab[tidx]
    n_comp = min(64, top_k)
    nys  = Nystroem(kernel='rbf', gamma=1.0/X_unlab.shape[1],
                    n_components=n_comp, random_state=42).fit(Xt)
    Xk   = nys.transform(Xt)
    km   = MiniBatchKMeans(n_clusters=n, n_init=3, random_state=42).fit(Xk)
    chosen = []
    for c in range(n):
        mask = (km.labels_ == c)
        if mask.sum() == 0: continue
        dists = np.linalg.norm(Xk[mask] - km.cluster_centers_[c], axis=1)
        chosen.append(tidx[np.where(mask)[0][np.argmin(dists)]])
    chosen = list(dict.fromkeys(chosen))
    if len(chosen) < n:
        rest = [i for i in tidx if i not in chosen]
        chosen.extend(rest[:n-len(chosen)])
    return np.array(chosen[:n])

QUERY_FNS = {
    "Random":           query_random,
    "Entropy":          query_entropy,
    "BADGE":            query_badge,
    "Unc+KernelKMeans": query_unc_kernelkmeans,
}

# ─── AL Runner ────────────────────────────────────────────────────────────────
def al_run(strategy, X_pool, y_pool, X_test, y_test, seed, budget,
           init_labeled=None):
    """
    init_labeled: set of pool indices to use as seed (if None → random stratified)
    """
    rng = np.random.default_rng(seed)
    n_pool = len(X_pool)

    if init_labeled is not None:
        labeled = set(int(i) for i in init_labeled)
        # Ensure both classes present; fill with random if needed
        if len(np.unique(y_pool[list(labeled)])) < 2:
            for cls in [0, 1]:
                if cls not in y_pool[list(labeled)]:
                    cands = np.where(y_pool == cls)[0]
                    cands = [c for c in cands if c not in labeled]
                    if len(cands):
                        labeled.add(int(rng.choice(cands)))
    else:
        # Standard stratified random init
        anom_idx = np.where(y_pool == 1)[0]
        norm_idx = np.where(y_pool == 0)[0]
        k = N_INIT // 2
        labeled = set(
            rng.choice(anom_idx, min(k, len(anom_idx)), replace=False).tolist() +
            rng.choice(norm_idx, min(k, len(norm_idx)), replace=False).tolist()
        )

    aucs, aps, f1s, n_labs = [], [], [], []
    queried = 0

    while queried <= budget:
        labeled_arr   = np.array(sorted(labeled))
        unlabeled_arr = np.array([i for i in range(n_pool) if i not in labeled])
        X_lab = X_pool[labeled_arr]; y_lab = y_pool[labeled_arr]
        X_unl = X_pool[unlabeled_arr]

        clf = RandomForestClassifier(n_estimators=RF_TREES, class_weight='balanced',
                                     max_features='sqrt', n_jobs=-1, random_state=int(seed))
        if len(np.unique(y_lab)) < 2:
            aucs.append(np.nan); aps.append(np.nan)
            f1s.append(np.nan);  n_labs.append(len(labeled))
        else:
            clf.fit(X_lab, y_lab)
            prob = clf.predict_proba(X_test)[:,1]
            pred = (prob >= 0.5).astype(int)
            auc = roc_auc_score(y_test, prob) if len(np.unique(y_test))>1 else np.nan
            ap  = average_precision_score(y_test, prob)
            f1  = f1_score(y_test, pred, zero_division=0)
            aucs.append(auc); aps.append(ap); f1s.append(f1)
            n_labs.append(len(labeled))

        if queried >= budget or len(unlabeled_arr) < BATCH_Q:
            break

        np.random.seed(int(seed) + queried)
        q = min(BATCH_Q, len(unlabeled_arr))
        local_q = QUERY_FNS[strategy](clf, X_unl, X_lab, q)
        for li in local_q:
            labeled.add(int(unlabeled_arr[li]))
        queried += q

    return np.array(n_labs), np.array(aucs), np.array(aps), np.array(f1s)


def interp_curves(run_labs, run_aucs, run_aps, budget):
    x_common = np.arange(N_INIT, budget + N_INIT + 1, BATCH_Q)
    auc_i = np.array([np.interp(x_common, nl, au) for nl,au in zip(run_labs, run_aucs)])
    ap_i  = np.array([np.interp(x_common, nl, ap) for nl,ap in zip(run_labs, run_aps)])
    return (x_common,
            np.nanmean(auc_i,0), np.nanstd(auc_i,0),
            np.nanmean(ap_i,0),  np.nanstd(ap_i,0))


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1 — Anomaly-Biased Initialization
# ══════════════════════════════════════════════════════════════════════════════
def rx_biased_init(X_pool, n=N_INIT):
    """
    Unsupervised warm-start: pick the top-n superpixels by RX anomaly score.
    No ground-truth labels needed at selection time.
    """
    scores = rx_scores(X_pool.astype(np.float64))
    return np.argsort(scores)[::-1][:n]


def run_exp1(scenes_X, scenes_y, scaler):
    print("\n" + "="*60)
    print("  EXPERIMENT 1: Anomaly-Biased Initialization")
    print("="*60)
    exp1 = {}

    for scene in SCENES:
        print(f"\n── Scene {scene.upper()} ──")
        X, y = scenes_X[scene], scenes_y[scene]
        Xsc  = scaler.transform(X).astype(np.float32)
        budget = min(MAX_BUDGET, int(MAX_PCT * len(X)))

        # Stratified pool/test split
        np.random.seed(SEED_BASE)
        anom_idx = np.where(y==1)[0]; norm_idx = np.where(y==0)[0]
        np.random.shuffle(anom_idx); np.random.shuffle(norm_idx)
        ca, cn = int(0.7*len(anom_idx)), int(0.7*len(norm_idx))
        pool_idx = np.concatenate([anom_idx[:ca], norm_idx[:cn]])
        test_idx = np.concatenate([anom_idx[ca:], norm_idx[cn:]])
        X_pool, y_pool = Xsc[pool_idx], y[pool_idx]
        X_test, y_test = Xsc[test_idx], y[test_idx]

        # RX-biased init indices (same for all runs — deterministic)
        biased_init = rx_biased_init(X_pool, N_INIT)

        exp1[scene] = {}
        for strat in STRATEGIES:
            print(f"  {strat}")
            for init_mode, init_arg in [("random", None), ("rx_biased", biased_init)]:
                run_labs, run_aucs, run_aps = [], [], []
                for run_id in range(N_RUNS):
                    seed = SEED_BASE + run_id*100 + hash(scene)%100
                    nl, au, ap, _ = al_run(strat, X_pool, y_pool, X_test, y_test,
                                           seed, budget,
                                           init_labeled=init_arg if init_mode=="rx_biased" else None)
                    run_labs.append(nl); run_aucs.append(au); run_aps.append(ap)
                x, am, astd, pm, pstd = interp_curves(run_labs, run_aucs, run_aps, budget)
                key = f"{strat}__{init_mode}"
                exp1[scene][key] = {
                    "x": x.tolist(),
                    "auc_mean": am.tolist(), "auc_std": astd.tolist(),
                    "ap_mean":  pm.tolist(), "ap_std":  pstd.tolist(),
                }
            # Print delta at final step
            am_rand = np.array(exp1[scene][f"{strat}__random"]["auc_mean"])[-1]
            am_bias = np.array(exp1[scene][f"{strat}__rx_biased"]["auc_mean"])[-1]
            am_rand_early = np.array(exp1[scene][f"{strat}__random"]["auc_mean"])[1]   # step 2
            am_bias_early = np.array(exp1[scene][f"{strat}__rx_biased"]["auc_mean"])[1]
            print(f"    AUC early (+50): random={am_rand_early:.4f} rx_biased={am_bias_early:.4f}"
                  f"  Δ={am_bias_early-am_rand_early:+.4f}")
            print(f"    AUC final:       random={am_rand:.4f} rx_biased={am_bias:.4f}"
                  f"  Δ={am_bias-am_rand:+.4f}")

    with open(OUT/"exp1_results.json","w") as f:
        json.dump(exp1, f, indent=2)
    print(f"\nSaved → {OUT}/exp1_results.json")
    return exp1


def plot_exp1(exp1):
    """Per-scene, per-strategy: random vs rx_biased init (AUC curves)."""
    fig, axes = plt.subplots(len(STRATEGIES), len(SCENES),
                              figsize=(4*len(SCENES), 3.5*len(STRATEGIES)))
    fig.suptitle("Exp 1 — RX-Biased Init vs Random Init\nAUC-ROC Learning Curves",
                 fontsize=13, fontweight="bold")

    for ri, strat in enumerate(STRATEGIES):
        for ci, scene in enumerate(SCENES):
            ax = axes[ri][ci]
            col = COLORS[strat]
            for mode, ls, lbl in [("random","--","Random init"),
                                   ("rx_biased","-","RX-biased init ★")]:
                key = f"{strat}__{mode}"
                x  = np.array(exp1[scene][key]["x"])
                am = np.array(exp1[scene][key]["auc_mean"])
                as_ = np.array(exp1[scene][key]["auc_std"])
                line, = ax.plot(x, am, color=col, ls=ls, lw=2.2, label=lbl)
                ax.fill_between(x, am-as_, am+as_, color=col, alpha=0.12)
            if ri == 0:
                ax.set_title(scene.upper(), fontsize=11, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(strat, fontsize=9)
            ax.set_xlabel("Budget" if ri == len(STRATEGIES)-1 else "", fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUT/"exp1_learning_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")

    # Summary: early-budget AUC gain (RX-biased - Random), aggregated over 4 scenes
    fig, ax = plt.subplots(figsize=(10, 5))
    x_ref = np.array(exp1[SCENES[0]][f"{STRATEGIES[0]}__random"]["x"])
    for strat in STRATEGIES:
        deltas = []
        for scene in SCENES:
            am_rand = np.array(exp1[scene][f"{strat}__random"]["auc_mean"])
            am_bias = np.array(exp1[scene][f"{strat}__rx_biased"]["auc_mean"])
            deltas.append(am_bias - am_rand)
        delta_mean = np.mean(deltas, axis=0)
        ax.plot(x_ref, delta_mean, color=COLORS[strat], lw=2.2, marker="o",
                markersize=3, label=strat)
    ax.axhline(0, color="black", lw=1, ls=":")
    ax.fill_between(x_ref, 0, 0, alpha=0)
    ax.set_xlabel("Labeled Budget (# pixels)", fontsize=11)
    ax.set_ylabel("ΔAUC (RX-biased − Random)", fontsize=11)
    ax.set_title("RX-Biased Init Gain Over Random Init\n(Mean over 4 Scenes)", fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = OUT/"exp1_gain_curve.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2 — Cross-Scene Transfer
# ══════════════════════════════════════════════════════════════════════════════
def transfer_init(source_clf, X_target_pool, n=N_INIT):
    """
    Use source RF to find the most uncertain target-pool samples.
    Top-n by entropy = warm-start seed for target scene AL.
    """
    p   = source_clf.predict_proba(X_target_pool)
    ent = -(p * np.log(p + 1e-12)).sum(1)
    return np.argsort(ent)[::-1][:n]


def run_exp2(scenes_X, scenes_y, scaler):
    print("\n" + "="*60)
    print("  EXPERIMENT 2: Cross-Scene Transfer")
    print("="*60)
    exp2 = {}

    # Build full-pool RF for each scene (source models)
    print("\nTraining source models …")
    source_clfs = {}
    for scene in SCENES:
        Xsc = scaler.transform(scenes_X[scene]).astype(np.float32)
        clf = RandomForestClassifier(RF_TREES, class_weight='balanced',
                                     max_features='sqrt', n_jobs=-1, random_state=42)
        clf.fit(Xsc, scenes_y[scene])
        source_clfs[scene] = clf
        print(f"  {scene}: AUC (self) = {roc_auc_score(scenes_y[scene], clf.predict_proba(Xsc)[:,1]):.4f}")

    for target in SCENES:
        X_tgt, y_tgt = scenes_X[target], scenes_y[target]
        Xsc_tgt = scaler.transform(X_tgt).astype(np.float32)
        budget = min(MAX_BUDGET, int(MAX_PCT * len(X_tgt)))

        np.random.seed(SEED_BASE)
        anom_idx = np.where(y_tgt==1)[0]; norm_idx = np.where(y_tgt==0)[0]
        np.random.shuffle(anom_idx); np.random.shuffle(norm_idx)
        ca, cn = int(0.7*len(anom_idx)), int(0.7*len(norm_idx))
        pool_idx = np.concatenate([anom_idx[:ca], norm_idx[:cn]])
        test_idx = np.concatenate([anom_idx[ca:], norm_idx[cn:]])
        X_pool, y_pool = Xsc_tgt[pool_idx], y_tgt[pool_idx]
        X_test, y_test = Xsc_tgt[test_idx], y_tgt[test_idx]

        print(f"\n── Target: {target.upper()} ──")
        exp2[target] = {}

        for source in SCENES:
            if source == target:
                continue
            pair = f"{source}→{target}"
            print(f"  Transfer {pair}")

            # Warm-start init using source clf
            warm_init = transfer_init(source_clfs[source], X_pool, N_INIT)

            exp2[target][pair] = {}
            for strat in STRATEGIES:
                for init_mode, init_arg in [("cold", None), ("warm", warm_init)]:
                    run_labs, run_aucs, run_aps = [], [], []
                    for run_id in range(N_RUNS):
                        seed = SEED_BASE + run_id*100 + hash(target+source)%100
                        nl, au, ap, _ = al_run(strat, X_pool, y_pool, X_test, y_test,
                                               seed, budget,
                                               init_labeled=init_arg if init_mode=="warm" else None)
                        run_labs.append(nl); run_aucs.append(au); run_aps.append(ap)
                    x, am, astd, pm, pstd = interp_curves(run_labs, run_aucs, run_aps, budget)
                    key = f"{strat}__{init_mode}"
                    exp2[target][pair][key] = {
                        "x": x.tolist(),
                        "auc_mean": am.tolist(), "auc_std": astd.tolist(),
                        "ap_mean":  pm.tolist(), "ap_std":  pstd.tolist(),
                    }
                am_cold = np.array(exp2[target][pair][f"{strat}__cold"]["auc_mean"])[-1]
                am_warm = np.array(exp2[target][pair][f"{strat}__warm"]["auc_mean"])[-1]
                print(f"    {strat:<20} cold={am_cold:.4f} warm={am_warm:.4f} Δ={am_warm-am_cold:+.4f}")

    with open(OUT/"exp2_results.json","w") as f:
        json.dump(exp2, f, indent=2)
    print(f"\nSaved → {OUT}/exp2_results.json")
    return exp2


def plot_exp2(exp2):
    """For each target scene: warm vs cold start (Unc+KernelKMeans, all 3 sources)."""
    strat = "Unc+KernelKMeans"

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Exp 2 — Cross-Scene Transfer ({strat})\nWarm Start vs Cold Start AUC",
                 fontsize=13, fontweight="bold")

    source_colors = {"z1":"#e41a1c","z2":"#377eb8","e1":"#4daf4a","e2":"#984ea3"}

    for ai, target in enumerate(SCENES):
        ax = axes.flat[ai]
        sources = [s for s in SCENES if s != target]
        for source in sources:
            pair = f"{source}→{target}"
            col  = source_colors[source]
            for mode, ls, lbl_sfx in [("cold","--"," (cold)"), ("warm","-"," (warm ★)")]:
                key = f"{strat}__{mode}"
                x  = np.array(exp2[target][pair][key]["x"])
                am = np.array(exp2[target][pair][key]["auc_mean"])
                as_ = np.array(exp2[target][pair][key]["auc_std"])
                ax.plot(x, am, color=col, ls=ls, lw=2.0,
                        label=f"{source.upper()}→{target.upper()}{lbl_sfx}")
                ax.fill_between(x, am-as_, am+as_, color=col, alpha=0.08)
        ax.set_title(f"Target: {target.upper()}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Budget", fontsize=9)
        ax.set_ylabel("ROC-AUC", fontsize=9)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUT/"exp2_transfer_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")

    # Summary heatmap: mean AUC gain (warm−cold) at final step, all pairs × all strategies
    pairs = [f"{s}→{t}" for t in SCENES for s in SCENES if s != t]
    matrix = np.zeros((len(STRATEGIES), len(pairs)))
    for pi, pair in enumerate(pairs):
        src, tgt = pair.split("→")
        for si, strat_k in enumerate(STRATEGIES):
            am_cold = np.array(exp2[tgt][pair][f"{strat_k}__cold"]["auc_mean"])[-1]
            am_warm = np.array(exp2[tgt][pair][f"{strat_k}__warm"]["auc_mean"])[-1]
            matrix[si, pi] = am_warm - am_cold

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto",
                   vmin=-0.005, vmax=0.02)
    plt.colorbar(im, ax=ax, label="ΔAUC (warm − cold)")
    ax.set_xticks(range(len(pairs))); ax.set_xticklabels(pairs, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(STRATEGIES))); ax.set_yticklabels(STRATEGIES, fontsize=9)
    ax.set_title("Cross-Scene Transfer Gain — Final AUC (Warm − Cold Start)\n"
                 "Green = warm start helps, Red = hurts", fontsize=11)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i,j]:+.3f}", ha="center", va="center",
                    fontsize=7.5, color="black")
    plt.tight_layout()
    out = OUT/"exp2_transfer_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("="*60)
    print("  Novel AL Experiments")
    print("="*60)

    print("\n[1/2] Loading scenes …")
    scenes_X, scenes_y = {}, {}
    for name, (w, h) in IMAGES.items():
        img, gt, seg = load_scene(name, w, h)
        X, y = extract_features(img, gt, seg)
        scenes_X[name] = X; scenes_y[name] = y
        print(f"  {name}: {len(X):,} SPs  anom={y.sum():,} ({100*y.mean():.1f}%)")

    scaler = StandardScaler().fit(np.vstack(list(scenes_X.values())))

    exp1 = run_exp1(scenes_X, scenes_y, scaler)
    plot_exp1(exp1)

    exp2 = run_exp2(scenes_X, scenes_y, scaler)
    plot_exp2(exp2)

    print("\nDONE ✓  →  results saved to", OUT)

if __name__ == "__main__":
    main()
