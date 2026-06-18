#!/usr/bin/env python3
"""
Cost-Sensitive Active Learning (cAL) — UAV Remote Sensing
===========================================================
Adaptation of Ali Güneş' cAL framework to UAV multispectral riparian
anomaly detection.

CORRECT EVALUATION DESIGN:
  - Baselines (Random, Standard AL, Unc+KernelKMeans):
      trained with standard SVM (r_+=1)
      evaluated at C(r_+) for ALL r_+ ∈ {1,2,3,4}
  - cAL(r_+=k):
      trained with cSVM(r_+=k)
      evaluated at C(r_+=k)
  - Full-pool baseline for r_+=k:
      full-pool cSVM(r_+=k) → C(r_+=k)

  For each r_+ panel, all methods are compared on the SAME cost function.

Metric: C(r_+) = (r_+ · FN + FP) / N × 100
"""

import sys, warnings, time, json
warnings.filterwarnings('ignore')

import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import Nystroem
from sklearn.cluster import MiniBatchKMeans

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── Paths & Config ────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
BASE  = _ROOT / "data"
OUT   = _ROOT / "results" / "al_cal"
OUT.mkdir(parents=True, exist_ok=True)

IMAGES   = {"z1":(3807,2141), "z2":(2081,957), "e1":(3629,961), "e2":(1094,707)}
SCENES   = list(IMAGES.keys())
# Deterministic per-scene seed offset — avoids hash() which varies with PYTHONHASHSEED
SCENE_SEED_OFFSET = {s: i * 10 for i, s in enumerate(SCENES)}  # z1=0, z2=10, e1=20, e2=30
N_INIT   = 20
BATCH_Q  = 50
MAX_BUDGET = 600
MAX_PCT  = 0.05
N_RUNS   = 3
POOL_CAP = 10_000
SEED_BASE = 2024
R_PLUS_LEVELS = [1, 2, 3, 4]

METHOD_COLORS = {
    "Random":           "#888888",
    "Standard AL":      "#1f77b4",
    "Unc+KernelKMeans": "#e377c2",
    "cAL r+=1":         "#d62728",
    "cAL r+=2":         "#ff7f0e",
    "cAL r+=3":         "#2ca02c",
    "cAL r+=4":         "#9467bd",
}
METHOD_LS = {
    "Random":           "--",
    "Standard AL":      "-.",
    "Unc+KernelKMeans": ":",
    "cAL r+=1":         "-",
    "cAL r+=2":         "-",
    "cAL r+=3":         "-",
    "cAL r+=4":         "-",
}

# ─── Data ─────────────────────────────────────────────────────────────────────
def load_scene(name, w, h):
    with open(BASE/name/f"{name}.raw","rb") as f:
        f.read(12)
        img = np.frombuffer(f.read(), dtype=np.uint32).reshape(h,w,5).astype(np.float32)
    with open(BASE/name/f"{name}_gt.pgm","rb") as f:
        f.readline(); f.readline(); f.readline()
        gt = np.frombuffer(f.read()[:h*w], dtype=np.uint8).reshape(h,w)
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
    E=1e-6
    B,G,R,RE,N_=mu[:,0],mu[:,1],mu[:,2],mu[:,3],mu[:,4]
    ndvi=(N_-R)/(N_+R+E); ndre=(N_-RE)/(N_+RE+E)
    exg=2*G-R-B; evi=2.5*(N_-R)/(N_+6*R-7.5*B+1+E)
    bndvi=(N_-B)/(N_+B+E); rb=R/(B+E); nr=N_/(R+E); nre_=N_/(RE+E)
    mah=np.sqrt((((mu-mu.mean(0))/(sig.mean(0)+E))**2).mean(1))
    feats=np.c_[mu,sig,ndvi,ndre,exg,evi,bndvi,rb,nr,nre_,mah].astype(np.float32)
    ac=np.bincount(sf,weights=(gtf==2).astype(np.float64),minlength=n)
    nc=np.bincount(sf,weights=(gtf==1).astype(np.float64),minlength=n)
    return feats, (ac>nc).astype(np.int64)

# ─── Cost Metric ──────────────────────────────────────────────────────────────
def cost_metric(y_true, y_pred, r_plus):
    """C(r_+) = (r_+ · FN + FP) / N × 100  [Eq. 9]"""
    FN = int(np.sum((y_true==1)&(y_pred==0)))
    FP = int(np.sum((y_true==0)&(y_pred==1)))
    return (r_plus * FN + FP) / len(y_true) * 100

# ─── Classifier ───────────────────────────────────────────────────────────────
def train_csvm(X_lab, y_lab, r_plus=1):
    cw = {1: float(r_plus), 0: 1.0}
    clf = SVC(kernel='rbf', C=1.0, gamma='scale',
              class_weight=cw, random_state=42)
    clf.fit(X_lab, y_lab)
    return clf

# ─── Query Strategies ─────────────────────────────────────────────────────────
def query_random(clf, X_unlab, n):
    return np.random.choice(len(X_unlab), n, replace=False)

def query_standard_al(clf, X_unlab, n):
    cap  = min(len(X_unlab), POOL_CAP)
    idx0 = np.random.choice(len(X_unlab), cap, replace=False)
    scores = np.abs(clf.decision_function(X_unlab[idx0]))
    return idx0[np.argsort(scores)[:n]]

def query_unc_kernelkmeans(clf, X_unlab, n):
    cap  = min(len(X_unlab), POOL_CAP)
    idx0 = np.random.choice(len(X_unlab), cap, replace=False)
    Xu   = X_unlab[idx0]
    scores = np.abs(clf.decision_function(Xu))
    top_k  = min(5*n, len(Xu))
    tidx   = np.argsort(scores)[:top_k]
    nys = Nystroem(kernel='rbf', gamma=1.0/Xu.shape[1],
                   n_components=min(64,top_k), random_state=42).fit(Xu[tidx])
    Xk  = nys.transform(Xu[tidx])
    km  = MiniBatchKMeans(n_clusters=n, n_init=3, random_state=42).fit(Xk)
    unc = scores[tidx]
    chosen = []
    for c in range(n):
        mask = (km.labels_==c)
        if mask.sum()==0: continue
        chosen.append(idx0[tidx[np.where(mask)[0][np.argmin(unc[mask])]]])
    chosen = list(dict.fromkeys(chosen))
    if len(chosen)<n:
        rest=[idx0[i] for i in tidx if idx0[i] not in set(chosen)]
        chosen.extend(rest[:n-len(chosen)])
    return np.array(chosen[:n])

def query_cal(clf, X_unlab, n, r_plus=1):
    """Algorithm 1: Stage 1 risk-sensitive uncertainty → Stage 2 kernel k-means.

    Risk-sensitive score (Eq. 4):
      u(x) = r+ * |d(x)|  if d(x) < 0  (predicted anomaly — higher FN cost)
      u(x) = |d(x)|        otherwise    (predicted normal)
    Selects M candidates with smallest u(x), then applies kernel k-means diversity.
    """
    cap  = min(len(X_unlab), POOL_CAP)
    idx0 = np.random.choice(len(X_unlab), cap, replace=False)
    Xu   = X_unlab[idx0]
    dec  = clf.decision_function(Xu)
    u    = np.where(dec < 0, r_plus * np.abs(dec), np.abs(dec))  # Eq. 4
    M    = min(5*n, len(Xu))
    top_M= np.argsort(u)[:M]
    nys  = Nystroem(kernel='rbf', gamma=1.0/Xu.shape[1],
                    n_components=min(64,M), random_state=42).fit(Xu[top_M])
    Xk   = nys.transform(Xu[top_M])
    km   = MiniBatchKMeans(n_clusters=n, n_init=3, random_state=42).fit(Xk)
    unc  = u[top_M]
    chosen = []
    for c in range(n):
        mask = (km.labels_==c)
        if mask.sum()==0: continue
        chosen.append(idx0[top_M[np.where(mask)[0][np.argmin(unc[mask])]]])
    chosen = list(dict.fromkeys(chosen))
    if len(chosen)<n:
        rest=[idx0[i] for i in top_M if idx0[i] not in set(chosen)]
        chosen.extend(rest[:n-len(chosen)])
    return np.array(chosen[:n])

# ─── Single AL Run ────────────────────────────────────────────────────────────
def al_run(method, r_plus_train, X_pool, y_pool, X_test, y_test, seed, budget):
    """
    Returns per-step arrays:
      n_labs, costs_dict {r_+: array}, auc_arr, fnr_arr
    costs_dict contains C(r_+) at ALL r_+ levels from this method's predictions.
    """
    rng = np.random.default_rng(seed)
    n_pool = len(X_pool)

    anom_idx = np.where(y_pool==1)[0]; norm_idx = np.where(y_pool==0)[0]
    k = N_INIT // 2
    labeled = set(
        rng.choice(anom_idx, min(k, len(anom_idx)), replace=False).tolist() +
        rng.choice(norm_idx, min(k, len(norm_idx)), replace=False).tolist()
    )

    n_labs = []
    costs_all = {r: [] for r in R_PLUS_LEVELS}
    aucs, fnrs = [], []
    queried = 0

    while queried <= budget:
        labeled_arr   = np.array(sorted(labeled))
        unlabeled_arr = np.array([i for i in range(n_pool) if i not in labeled])
        X_lab = X_pool[labeled_arr]; y_lab = y_pool[labeled_arr]

        if len(np.unique(y_lab)) < 2:
            n_labs.append(len(labeled))
            for r in R_PLUS_LEVELS: costs_all[r].append(np.nan)
            aucs.append(np.nan); fnrs.append(np.nan)
        else:
            clf = train_csvm(X_lab, y_lab, r_plus=r_plus_train)
            dec   = clf.decision_function(X_test)
            y_pred= (dec >= 0).astype(int)
            # Evaluate at ALL r_+ levels (key fix)
            for r in R_PLUS_LEVELS:
                costs_all[r].append(cost_metric(y_test, y_pred, r))
            try:
                aucs.append(roc_auc_score(y_test, dec))
            except Exception:
                aucs.append(np.nan)
            FN = int(np.sum((y_test==1)&(y_pred==0)))
            fnrs.append(FN / max(1, int(y_test.sum())))
            n_labs.append(len(labeled))

        if queried >= budget or len(unlabeled_arr) < BATCH_Q:
            break

        np.random.seed(int(seed) + queried)
        q = min(BATCH_Q, len(unlabeled_arr))
        X_unl = X_pool[unlabeled_arr]

        if len(np.unique(y_lab)) < 2 or method == "Random":
            local_q = query_random(None, X_unl, q)
        else:
            clf_q = train_csvm(X_lab, y_lab, r_plus=r_plus_train)
            if method == "Standard AL":
                local_q = query_standard_al(clf_q, X_unl, q)
            elif method == "Unc+KernelKMeans":
                local_q = query_unc_kernelkmeans(clf_q, X_unl, q)
            else:  # cAL
                local_q = query_cal(clf_q, X_unl, q, r_plus=r_plus_train)

        for li in local_q:
            labeled.add(int(unlabeled_arr[li]))
        queried += q

    return (np.array(n_labs),
            {r: np.array(v) for r, v in costs_all.items()},
            np.array(aucs), np.array(fnrs))


def interp_metric(run_labs, run_vals, budget):
    x = np.arange(N_INIT, budget + N_INIT + 1, BATCH_Q)
    interped = np.array([np.interp(x, nl, v) for nl, v in zip(run_labs, run_vals)])
    return x, np.nanmean(interped, 0), np.nanstd(interped, 0)


# ─── Experiments ──────────────────────────────────────────────────────────────
def build_methods():
    return (["Random", "Standard AL", "Unc+KernelKMeans"] +
            [f"cAL r+={r}" for r in R_PLUS_LEVELS])

def run_experiments(scenes_X, scenes_y):
    methods = build_methods()
    all_results = {}

    for scene in SCENES:
        print(f"\n{'='*55}\n  Scene {scene.upper()}\n{'='*55}")
        X, y = scenes_X[scene], scenes_y[scene]
        budget = min(MAX_BUDGET, int(MAX_PCT * len(X)))

        # Stratified 70/30 split — consistent with ablation_feature_sets.py
        rng_split = np.random.default_rng(SEED_BASE)
        ai = np.where(y==1)[0]; ni = np.where(y==0)[0]
        ca, cn = int(0.7*len(ai)), int(0.7*len(ni))
        pool_idx = np.concatenate([
            rng_split.choice(ai, ca, replace=False),
            rng_split.choice(ni, cn, replace=False),
        ])
        test_idx = np.array([i for i in range(len(y)) if i not in set(pool_idx.tolist())])

        # Per-scene scaler fitted on pool only (no test leakage)
        scaler = StandardScaler().fit(X[pool_idx])
        X_pool = scaler.transform(X[pool_idx]).astype(np.float32)
        X_test = scaler.transform(X[test_idx]).astype(np.float32)
        y_pool, y_test = y[pool_idx], y[test_idx]

        # Full-pool baseline: separate cSVM per r_+ level (KEY FIX)
        full_costs = {}
        for r in R_PLUS_LEVELS:
            clf_f = train_csvm(X_pool, y_pool, r_plus=r)
            yp_f  = (clf_f.decision_function(X_test) >= 0).astype(int)
            full_costs[r] = cost_metric(y_test, yp_f, r)
        print("  Full-pool: " + "  ".join(f"C(r+={r})={full_costs[r]:.2f}"
                                           for r in R_PLUS_LEVELS))

        all_results[scene] = {"full_costs": full_costs, "methods": {}}

        for method in methods:
            r_train = int(method.split("=")[1]) if method.startswith("cAL") else 1
            t0 = time.time()
            run_labs, run_costs_all, run_aucs, run_fnrs = [], [], [], []

            for run_id in range(N_RUNS):
                seed = SEED_BASE + run_id * 100 + SCENE_SEED_OFFSET[scene]
                nl, costs_d, au, fn = al_run(method, r_train,
                                              X_pool, y_pool, X_test, y_test,
                                              seed, budget)
                run_labs.append(nl)
                run_costs_all.append(costs_d)
                run_aucs.append(au)
                run_fnrs.append(fn)

            # Interpolate costs at ALL r_+ levels
            cost_curves = {}
            for r in R_PLUS_LEVELS:
                vals = [rc[r] for rc in run_costs_all]
                x, cm, cs = interp_metric(run_labs, vals, budget)
                cost_curves[r] = {"mean": cm.tolist(), "std": cs.tolist()}

            x, am, _ = interp_metric(run_labs, run_aucs, budget)
            x, fm, _ = interp_metric(run_labs, run_fnrs, budget)

            all_results[scene]["methods"][method] = {
                "x":      x.tolist(),
                "costs":  cost_curves,   # C(r_+) at each r_+ level
                "auc":    am.tolist(),
                "fnr":    fm.tolist(),
            }
            dt = time.time()-t0
            c1 = cost_curves[1]["mean"][-1]
            c4 = cost_curves[4]["mean"][-1]
            print(f"  {method:<22}  C(1)={c1:.2f}  C(4)={c4:.2f}  AUC={am[-1]:.4f}"
                  f"  FNR={fm[-1]:.3f}  [{dt:.1f}s]")

    with open(OUT/"cal_results.json","w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {OUT}/cal_results.json")
    return all_results


# ─── Plots ────────────────────────────────────────────────────────────────────
def plot_cal_vs_baselines(all_results):
    """
    4 rows (r_+ levels) × 4 cols (scenes).
    Each panel: cAL(r_+) vs baselines, all evaluated at C(r_+).
    """
    methods = build_methods()
    baselines = ["Random", "Standard AL", "Unc+KernelKMeans"]

    fig, axes = plt.subplots(len(R_PLUS_LEVELS), len(SCENES),
                              figsize=(4.5*len(SCENES), 3.5*len(R_PLUS_LEVELS)))
    fig.suptitle("cAL vs Baselines — C(r₊) per Risk Level per Scene\n"
                 "(All methods evaluated at same C(r₊))",
                 fontsize=13, fontweight="bold")

    for ri, r in enumerate(R_PLUS_LEVELS):
        cal_m = f"cAL r+={r}"
        for ci, scene in enumerate(SCENES):
            ax = axes[ri][ci]
            sr = all_results[scene]
            x  = np.array(sr["methods"]["Random"]["x"])

            for method in baselines + [cal_m]:
                cm = np.array(sr["methods"][method]["costs"][r]["mean"])
                cs = np.array(sr["methods"][method]["costs"][r]["std"])
                lw = 2.8 if method == cal_m else 1.6
                ax.plot(x, cm, color=METHOD_COLORS[method],
                        ls=METHOD_LS[method], lw=lw, label=method)
                ax.fill_between(x, cm-cs, cm+cs,
                                color=METHOD_COLORS[method], alpha=0.1)

            # Full-pool baseline for this r_+
            fc = sr["full_costs"][r]
            ax.axhline(fc, color="black", lw=1.2, ls=":",
                       label=f"Full-pool ({fc:.2f})")

            if ri == 0:
                ax.set_title(scene.upper(), fontsize=11, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(f"r₊={r}\nC(r₊)", fontsize=9)
            if ri == len(R_PLUS_LEVELS)-1:
                ax.set_xlabel("Budget (# pixels)", fontsize=8)
            ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUT/"cal_vs_baselines.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")


def plot_label_efficiency(all_results):
    """Budget to reach full-pool C(r_+), mean over 4 scenes."""
    baselines = ["Random", "Standard AL", "Unc+KernelKMeans"]
    fig, axes = plt.subplots(1, len(R_PLUS_LEVELS),
                              figsize=(5*len(R_PLUS_LEVELS), 5))
    fig.suptitle("Label Efficiency — Budget to Reach Full-Pool C(r₊)\n"
                 "(Lower = More Label-Efficient)", fontsize=13, fontweight="bold")

    for ax, r in zip(axes, R_PLUS_LEVELS):
        cal_m = f"cAL r+={r}"
        show  = baselines + [cal_m]
        budgets = {m: [] for m in show}

        for scene in SCENES:
            sr   = all_results[scene]
            full = sr["full_costs"][r]
            x    = np.array(sr["methods"]["Random"]["x"])
            for method in show:
                cm = np.array(sr["methods"][method]["costs"][r]["mean"])
                reached = next((int(xi) for xi, ci in zip(x, cm) if ci <= full),
                               int(x[-1]) + 50)
                budgets[method].append(reached)

        means  = [np.mean(budgets[m]) for m in show]
        colors = [METHOD_COLORS[m] for m in show]
        bars   = ax.bar(range(len(show)), means, color=colors, edgecolor="white")
        ax.set_xticks(range(len(show)))
        ax.set_xticklabels(show, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Mean Budget (pixels)", fontsize=9)
        ax.set_title(f"r₊ = {r}", fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        for bar, v in zip(bars, means):
            lbl = f"{int(v)}" if v <= MAX_BUDGET+N_INIT else "N/A"
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+3,
                    lbl, ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out = OUT/"cal_label_efficiency.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")


def plot_fnr_curves(all_results):
    """False Negative Rate curves — how quickly each method stops missing anomalies."""
    show = ["Random", "Standard AL", "Unc+KernelKMeans", "cAL r+=2", "cAL r+=4"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("False Negative Rate During Active Learning\n"
                 "(Lower = Fewer Missed Anomalies — Critical for Environmental Monitoring)",
                 fontsize=13, fontweight="bold")
    for ax, scene in zip(axes.flat, SCENES):
        x = np.array(all_results[scene]["methods"]["Random"]["x"])
        for method in show:
            fm = np.array(all_results[scene]["methods"][method]["fnr"])
            lw = 2.5 if method.startswith("cAL") else 1.6
            ax.plot(x, fm, color=METHOD_COLORS[method],
                    ls=METHOD_LS[method], lw=lw, label=method)
        ax.set_title(f"Scene {scene.upper()}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Labeled Budget (# pixels)", fontsize=10)
        ax.set_ylabel("False Negative Rate", fontsize=10)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = OUT/"cal_fnr_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")


def plot_cost_aggregated(all_results):
    """Mean C(r_+) over 4 scenes — one panel per r_+."""
    baselines = ["Random", "Standard AL", "Unc+KernelKMeans"]
    fig, axes = plt.subplots(1, len(R_PLUS_LEVELS),
                              figsize=(5.5*len(R_PLUS_LEVELS), 5))
    fig.suptitle("Aggregated C(r₊) Learning Curves (Mean over 4 Scenes)\n"
                 "cAL vs Baselines at Each Risk Level",
                 fontsize=13, fontweight="bold")

    for ax, r in zip(axes, R_PLUS_LEVELS):
        cal_m = f"cAL r+={r}"
        x_ref = np.array(all_results[SCENES[0]]["methods"]["Random"]["x"])
        for method in baselines + [cal_m]:
            scene_means = np.array([
                all_results[sc]["methods"][method]["costs"][r]["mean"]
                for sc in SCENES])
            agg_mean = np.nanmean(scene_means, axis=0)
            agg_std  = np.nanstd(scene_means, axis=0)
            lw = 2.5 if method == cal_m else 1.6
            line, = ax.plot(x_ref, agg_mean,
                            color=METHOD_COLORS[method], ls=METHOD_LS[method],
                            lw=lw, label=method)
            ax.fill_between(x_ref, agg_mean-agg_std, agg_mean+agg_std,
                            color=line.get_color(), alpha=0.1)

        # Mean full-pool baseline
        fc_mean = np.mean([all_results[sc]["full_costs"][r] for sc in SCENES])
        ax.axhline(fc_mean, color="black", lw=1.2, ls=":",
                   label=f"Full-pool ({fc_mean:.2f})")
        ax.set_title(f"r₊ = {r}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Budget (# pixels)", fontsize=10)
        ax.set_ylabel("C(r₊)" if r==1 else "", fontsize=10)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUT/"cal_cost_aggregated.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")


def print_summary(all_results):
    baselines = ["Random", "Standard AL", "Unc+KernelKMeans"]
    print("\n" + "="*65)
    print("  LABEL EFFICIENCY — Budget to reach full-pool C(r_+)")
    print("  Mean over 4 scenes")
    print("="*65)
    header = f"  {'Method':<22}" + "".join(f"  r+={r}" for r in R_PLUS_LEVELS)
    print(header); print("  "+"-"*55)
    for method in baselines + [f"cAL r+={r}" for r in R_PLUS_LEVELS]:
        r_train = int(method.split("=")[1]) if method.startswith("cAL") else 1
        row = f"  {method:<22}"
        for r in R_PLUS_LEVELS:
            buds = []
            for scene in SCENES:
                sr   = all_results[scene]
                full = sr["full_costs"][r]
                x    = np.array(sr["methods"][method]["x"])
                cm   = np.array(sr["methods"][method]["costs"][r]["mean"])
                reached = next((int(xi) for xi, ci in zip(x, cm) if ci<=full), None)
                buds.append(reached if reached else MAX_BUDGET+50)
            v = int(np.mean(buds))
            row += f"  {'N/A':>5}" if v > MAX_BUDGET+N_INIT else f"  {v:>5}"
        print(row)

    print("\n  FINAL C(r_+) at max budget, mean over 4 scenes")
    print("  "+"-"*55)
    print(header)
    for method in baselines + [f"cAL r+={r}" for r in R_PLUS_LEVELS]:
        row = f"  {method:<22}"
        for r in R_PLUS_LEVELS:
            vals = [np.array(all_results[sc]["methods"][method]["costs"][r]["mean"])[-1]
                    for sc in SCENES]
            row += f"  {np.nanmean(vals):>5.2f}"
        print(row)


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("="*55)
    print("  cAL — UAV Remote Sensing (Fixed Evaluation)")
    print("="*55)

    print("\n[1/3] Loading …")
    scenes_X, scenes_y = {}, {}
    for name, (w, h) in IMAGES.items():
        img, gt, seg = load_scene(name, w, h)
        X, y = extract_features(img, gt, seg)
        scenes_X[name]=X; scenes_y[name]=y
        print(f"  {name}: {len(X):,} SPs  anom={y.sum():,} ({100*y.mean():.1f}%)")

    print("\n[2/3] Running …")
    all_results = run_experiments(scenes_X, scenes_y)

    print("\n[3/3] Figures …")
    plot_cal_vs_baselines(all_results)
    plot_label_efficiency(all_results)
    plot_fnr_curves(all_results)
    plot_cost_aggregated(all_results)
    print_summary(all_results)

    print(f"\nDONE ✓  →  {OUT}")

if __name__ == "__main__":
    main()
