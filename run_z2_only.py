#!/usr/bin/env python3
"""
Run Z2-only experiments with corrected ground truth.
Patches cal_results.json and ablation_results.json with new Z2 values.
"""
import sys, warnings, json
warnings.filterwarnings('ignore')

import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from sklearn.kernel_approximation import Nystroem
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parent
BASE  = _ROOT / "data"
OUT_AL    = _ROOT / "results" / "al_cal"
OUT_ABL   = _ROOT / "results" / "ablation"

# ── Config (identical to al_cal_experiment.py) ─────────────────────────
SCENE     = "z2"
W, H      = 2081, 957
N_INIT    = 20
BATCH_Q   = 50
MAX_BUDGET = 600
MAX_PCT   = 0.05
N_RUNS    = 5
POOL_CAP  = 10_000
SEED_BASE = 2024
SCENE_SEED_OFFSET = {"z1": 0, "z2": 10, "e1": 20, "e2": 30}
R_PLUS_VALUES = [1, 2, 3, 4]
FEATURE_MODES = ["5d", "10d", "19d"]

# ── Data loading ────────────────────────────────────────────────────────
def load_scene():
    with open(BASE / SCENE / f"{SCENE}.raw", "rb") as f:
        f.read(12)
        img = np.frombuffer(f.read(), dtype=np.uint32).reshape(H, W, 5).astype(np.float32)
    with open(BASE / SCENE / f"{SCENE}_gt.pgm", "rb") as f:
        f.readline(); f.readline(); f.readline()
        gt = np.frombuffer(f.read()[:H * W], dtype=np.uint8).reshape(H, W)
    with open(BASE / SCENE / f"{SCENE}_seg.raw", "rb") as f:
        f.read(8)
        seg = np.frombuffer(f.read(), dtype=np.uint32).reshape(H, W)
    return img, gt, seg

def extract_features(img, gt, seg, mode="19d"):
    n   = int(seg.max()) + 1
    sf  = seg.ravel().astype(np.int64)
    imgf = img.reshape(-1, 5).astype(np.float64)
    gtf  = gt.ravel().astype(np.int64)
    cnt  = np.bincount(sf, minlength=n).clip(1).astype(np.float64)
    mu = np.zeros((n, 5)); sq = np.zeros((n, 5))
    for b in range(5):
        mu[:, b] = np.bincount(sf, weights=imgf[:, b],    minlength=n) / cnt
        sq[:, b] = np.bincount(sf, weights=imgf[:, b]**2, minlength=n) / cnt
    sig = np.sqrt(np.clip(sq - mu**2, 0, None))
    if mode == "5d":
        feats = mu.astype(np.float32)
    elif mode == "10d":
        feats = np.c_[mu, sig].astype(np.float32)
    else:
        E = 1e-6
        B,G,R,RE,N_ = mu[:,0],mu[:,1],mu[:,2],mu[:,3],mu[:,4]
        ndvi=(N_-R)/(N_+R+E); ndre=(N_-RE)/(N_+RE+E)
        exg=2*G-R-B; evi=2.5*(N_-R)/(N_+6*R-7.5*B+1+E)
        bndvi=(N_-B)/(N_+B+E); rb=R/(B+E); nr=N_/(R+E); nre_=N_/(RE+E)
        mah=np.sqrt((((mu-mu.mean(0))/(sig.mean(0)+E))**2).mean(1))
        feats=np.c_[mu,sig,ndvi,ndre,exg,evi,bndvi,rb,nr,nre_,mah].astype(np.float32)
    ac = np.bincount(sf, weights=(gtf==2).astype(np.float64), minlength=n)
    nc = np.bincount(sf, weights=(gtf==1).astype(np.float64), minlength=n)
    labels = (ac > nc).astype(np.int64)
    return feats, labels

def cost_metric(y_true, y_pred, r_plus):
    FN = int(np.sum((y_true==1)&(y_pred==0)))
    FP = int(np.sum((y_true==0)&(y_pred==1)))
    return (r_plus*FN + FP) / len(y_true) * 100

def train_csvm(X_lab, y_lab, r_plus):
    clf = SVC(kernel='rbf', C=1.0, gamma='scale',
              class_weight={1: float(r_plus), 0: 1.0}, random_state=42)
    clf.fit(X_lab, y_lab)
    return clf

def query_cal(clf, X_unlab, n, r_plus):
    cap  = min(len(X_unlab), POOL_CAP)
    idx0 = np.random.choice(len(X_unlab), cap, replace=False)
    Xu   = X_unlab[idx0]
    dec  = clf.decision_function(Xu)
    u    = np.where(dec < 0, r_plus * np.abs(dec), np.abs(dec))
    M    = min(5*n, len(Xu))
    top_M = np.argsort(u)[:M]
    nys  = Nystroem(kernel='rbf', gamma=1.0/Xu.shape[1],
                    n_components=min(64, M), random_state=42).fit(Xu[top_M])
    Xk   = nys.transform(Xu[top_M])
    km   = MiniBatchKMeans(n_clusters=n, n_init=3, random_state=42).fit(Xk)
    unc  = u[top_M]
    chosen = []
    for c in range(n):
        mask = (km.labels_==c)
        if mask.sum()==0: continue
        chosen.append(idx0[top_M[np.where(mask)[0][np.argmin(unc[mask])]]])
    chosen = list(dict.fromkeys(chosen))
    if len(chosen) < n:
        rest = [idx0[i] for i in top_M if idx0[i] not in set(chosen)]
        chosen.extend(rest[:n-len(chosen)])
    return np.array(chosen[:n])

def al_run(X_pool, y_pool, X_test, y_test, r_plus, seed, budget):
    rng = np.random.default_rng(seed)
    n_pool = len(X_pool)
    anom_idx = np.where(y_pool==1)[0]
    norm_idx = np.where(y_pool==0)[0]
    k = N_INIT//2
    labeled = set(
        rng.choice(anom_idx, min(k,len(anom_idx)), replace=False).tolist() +
        rng.choice(norm_idx, min(k,len(norm_idx)), replace=False).tolist()
    )
    n_labs, costs, fnrs = [], [], []
    queried = 0
    while queried <= budget:
        labeled_arr   = np.array(sorted(labeled))
        unlabeled_arr = np.array([i for i in range(n_pool) if i not in labeled])
        X_lab = X_pool[labeled_arr]; y_lab = y_pool[labeled_arr]
        if len(np.unique(y_lab)) < 2:
            n_labs.append(len(labeled)); costs.append(np.nan); fnrs.append(np.nan)
        else:
            clf = train_csvm(X_lab, y_lab, r_plus)
            dec = clf.decision_function(X_test)
            y_pred = (dec>=0).astype(int)
            costs.append(cost_metric(y_test, y_pred, r_plus))
            FN = int(np.sum((y_test==1)&(y_pred==0)))
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

# ── Main ────────────────────────────────────────────────────────────────
def main():
    print(f"Loading Z2 with corrected ground truth...")
    img, gt, seg = load_scene()

    # ── PART A: Main AL experiment (al_cal_experiment.py methods) ──────
    feats, labels = extract_features(img, gt, seg, mode="19d")
    rng_split = np.random.default_rng(SEED_BASE)
    anom_idx = np.where(labels==1)[0]; norm_idx = np.where(labels==0)[0]
    n_pool_a = int(0.7*len(anom_idx)); n_pool_n = int(0.7*len(norm_idx))
    pool_idx = np.concatenate([
        rng_split.choice(anom_idx, n_pool_a, replace=False),
        rng_split.choice(norm_idx, n_pool_n, replace=False)
    ])
    test_idx = np.array([i for i in range(len(labels)) if i not in set(pool_idx.tolist())])
    scaler = StandardScaler().fit(feats[pool_idx])
    X_pool = scaler.transform(feats[pool_idx]).astype(np.float32)
    X_test = scaler.transform(feats[test_idx]).astype(np.float32)
    y_pool = labels[pool_idx]; y_test = labels[test_idx]
    budget = min(MAX_BUDGET, int(MAX_PCT * len(pool_idx)))
    print(f"  Pool={len(pool_idx)}, Test={len(test_idx)}, Budget={budget}")
    print(f"  Anomaly superpixels in test: {int(y_test.sum())}")

    METHODS = {
        "random":          (0, "random"),
        "standard_al":     (1, "standard"),
        "unc_kernel_kmeans": (1, "unc_kk"),
        "cal_rp1":         (1, "cal"),
        "cal_rp2":         (2, "cal"),
        "cal_rp3":         (3, "cal"),
        "cal_rp4":         (4, "cal"),
    }

    z2_al_results = {}
    for method_key, (r_plus, mtype) in METHODS.items():
        run_fnrs = []
        for run_id in range(N_RUNS):
            seed = SEED_BASE + run_id*100 + SCENE_SEED_OFFSET[SCENE]
            _, _, fnrs = al_run(X_pool, y_pool, X_test, y_test,
                                r_plus=max(r_plus,1), seed=seed, budget=budget)
            run_fnrs.append(fnrs[-1] if len(fnrs)>0 else np.nan)
        fnr_mean = float(np.nanmean(run_fnrs))
        z2_al_results[method_key] = {"fnr": round(fnr_mean,3), "recall": round(1-fnr_mean,3)}
        print(f"  {method_key:25s} FNR={fnr_mean:.3f}  Recall={1-fnr_mean:.3f}")

    # ── PART B: Ablation ────────────────────────────────────────────────
    print("\nRunning ablation (Z2 only)...")
    z2_abl_results = {}
    for mode in FEATURE_MODES:
        f2, l2 = extract_features(img, gt, seg, mode=mode)
        rng2 = np.random.default_rng(SEED_BASE)
        ai = np.where(l2==1)[0]; ni = np.where(l2==0)[0]
        pi = np.concatenate([rng2.choice(ai, int(0.7*len(ai)), replace=False),
                             rng2.choice(ni, int(0.7*len(ni)), replace=False)])
        ti = np.array([i for i in range(len(l2)) if i not in set(pi.tolist())])
        sc2 = StandardScaler().fit(f2[pi])
        Xp = sc2.transform(f2[pi]).astype(np.float32)
        Xt = sc2.transform(f2[ti]).astype(np.float32)
        yp = l2[pi]; yt = l2[ti]
        bud2 = min(MAX_BUDGET, int(MAX_PCT*len(pi)))
        run_fnrs = []
        for run in range(3):
            seed = SEED_BASE + run*100 + SCENE_SEED_OFFSET[SCENE]
            _, _, fnrs = al_run(Xp, yp, Xt, yt, r_plus=4, seed=seed, budget=bud2)
            run_fnrs.append(fnrs[-1] if len(fnrs)>0 else np.nan)
        fnr_mean = float(np.nanmean(run_fnrs))
        z2_abl_results[mode] = {"fnr": round(fnr_mean,3), "recall": round(1-fnr_mean,3)}
        print(f"  {mode:6s}  FNR={fnr_mean:.3f}  Recall={1-fnr_mean:.3f}")

    # ── Patch JSON files ─────────────────────────────────────────────────
    # cal_results.json — patch z2 FNR/recall for each method
    cal_path = OUT_AL / "cal_results.json"
    with open(cal_path) as f:
        cal = json.load(f)

    method_map = {
        "random":           "Random AL",
        "standard_al":      "Standard AL",
        "unc_kernel_kmeans":"Unc+KernelKMeans",
        "cal_rp1":          "cAL r+=1",
        "cal_rp2":          "cAL r+=2",
        "cal_rp3":          "cAL r+=3",
        "cal_rp4":          "cAL r+=4",
    }
    for k, label in method_map.items():
        if label in cal and "z2" in cal[label]:
            cal[label]["z2"]["fnr_final"]    = z2_al_results[k]["fnr"]
            cal[label]["z2"]["recall_final"] = z2_al_results[k]["recall"]
    with open(cal_path, "w") as f:
        json.dump(cal, f, indent=2)
    print(f"\nPatched {cal_path}")

    # ablation_results.json — patch z2 for each mode
    abl_path = OUT_ABL / "ablation_results.json"
    with open(abl_path) as f:
        abl = json.load(f)
    for mode in FEATURE_MODES:
        abl[mode]["z2"]["fnr"]    = z2_abl_results[mode]["fnr"]
        abl[mode]["z2"]["recall"] = z2_abl_results[mode]["recall"]
    with open(abl_path, "w") as f:
        json.dump(abl, f, indent=2)
    print(f"Patched {abl_path}")

    print("\n=== Z2 NEW RESULTS (corrected GT) ===")
    print(f"{'Method':<25} {'FNR':>6} {'Recall':>7}")
    print("-"*40)
    for k,v in z2_al_results.items():
        print(f"{k:<25} {v['fnr']:>6.3f} {v['recall']:>7.3f}")
    print("\nAblation Z2:")
    for mode,v in z2_abl_results.items():
        print(f"  {mode}: FNR={v['fnr']:.3f}")

if __name__ == "__main__":
    main()
