#!/usr/bin/env python3
"""
Generate pixel-level anomaly detection maps comparing Standard AL vs cAL (r+=4).

Color convention (López-Fandiño et al. style):
  Green  = TP  — correctly detected anomaly
  Red    = FN  — missed anomaly (false negative)
  Blue   = FP  — false alarm (false positive)
  Black  = TN  — correctly identified background
  Gray   = Pool pixels (used for training, not evaluated)

Outputs:
  results/maps/anomaly_maps_all_scenes.png   — 4 scenes × 3 cols
  results/maps/anomaly_maps_z2_e2.png        — Z2 & E2 only (paper figure)
  results/maps/anomaly_maps_{scene}.png      — one per scene
"""

import sys, warnings, time
warnings.filterwarnings('ignore')

import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import Nystroem
from sklearn.cluster import MiniBatchKMeans
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─── Config — must match al_cal_experiment.py exactly ─────────────────────────
_ROOT     = Path(__file__).resolve().parent
BASE      = Path("/Users/aligunes/Desktop/IEEE Transactions REmote Sensign/agentic ai/data")
OUT       = _ROOT / "results" / "maps"
OUT.mkdir(parents=True, exist_ok=True)

IMAGES    = {"z1":(3807,2141), "z2":(2081,957), "e1":(3629,961), "e2":(1094,707)}
SCENES    = list(IMAGES.keys())
N_INIT    = 20
BATCH_Q   = 50
MAX_BUDGET = 600
MAX_PCT   = 0.05
POOL_CAP  = 1_000   # reduced for speed (map generation only)
SEED_BASE = 2024
RUN_ID    = 0        # single deterministic run for map visualisation

# ─── Data Loading ──────────────────────────────────────────────────────────────
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
    E   = 1e-6
    B,G,R,RE,N_ = mu[:,0],mu[:,1],mu[:,2],mu[:,3],mu[:,4]
    ndvi  = (N_-R)/(N_+R+E);     ndre = (N_-RE)/(N_+RE+E)
    exg   = 2*G-R-B;              evi  = 2.5*(N_-R)/(N_+6*R-7.5*B+1+E)
    bndvi = (N_-B)/(N_+B+E);     rb   = R/(B+E)
    nr    = N_/(R+E);             nre_ = N_/(RE+E)
    mah   = np.sqrt((((mu-mu.mean(0))/(sig.mean(0)+E))**2).mean(1))
    feats = np.c_[mu,sig,ndvi,ndre,exg,evi,bndvi,rb,nr,nre_,mah].astype(np.float32)
    ac = np.bincount(sf, weights=(gtf==2).astype(np.float64), minlength=n)
    nc = np.bincount(sf, weights=(gtf==1).astype(np.float64), minlength=n)
    return feats, (ac>nc).astype(np.int64)

# ─── Classifier ───────────────────────────────────────────────────────────────
def train_csvm(X_lab, y_lab, r_plus=1):
    cw  = {1: float(r_plus), 0: 1.0}
    clf = SVC(kernel='rbf', C=1.0, gamma='scale',
              class_weight=cw, random_state=42)
    clf.fit(X_lab, y_lab)
    return clf

# ─── Query Strategies ─────────────────────────────────────────────────────────
def query_random(X_unlab, n):
    return np.random.choice(len(X_unlab), n, replace=False)

def query_standard_al(clf, X_unlab, n):
    cap  = min(len(X_unlab), POOL_CAP)
    idx0 = np.random.choice(len(X_unlab), cap, replace=False)
    sc   = np.abs(clf.decision_function(X_unlab[idx0]))
    return idx0[np.argsort(sc)[:n]]

def query_cal(clf, X_unlab, n):
    cap  = min(len(X_unlab), POOL_CAP)
    idx0 = np.random.choice(len(X_unlab), cap, replace=False)
    Xu   = X_unlab[idx0]
    sc   = np.abs(clf.decision_function(Xu))
    M    = min(5*n, len(Xu))
    top  = np.argsort(sc)[:M]
    nys  = Nystroem(kernel='rbf', gamma=1.0/Xu.shape[1],
                    n_components=min(64,M), random_state=42).fit(Xu[top])
    Xk   = nys.transform(Xu[top])
    km   = MiniBatchKMeans(n_clusters=n, n_init=3, random_state=42).fit(Xk)
    unc  = sc[top]
    chosen = []
    for c in range(n):
        mask = (km.labels_==c)
        if mask.sum()==0: continue
        chosen.append(idx0[top[np.where(mask)[0][np.argmin(unc[mask])]]])
    chosen = list(dict.fromkeys(chosen))
    if len(chosen)<n:
        rest = [idx0[i] for i in top if idx0[i] not in set(chosen)]
        chosen.extend(rest[:n-len(chosen)])
    return np.array(chosen[:n])

# ─── AL Runner ────────────────────────────────────────────────────────────────
def run_al(method, r_plus_train, X_pool, y_pool, seed, budget):
    """Run AL to max budget; return final trained classifier."""
    rng  = np.random.default_rng(seed)
    n_pool = len(X_pool)
    ai   = np.where(y_pool==1)[0];  ni = np.where(y_pool==0)[0]
    k    = N_INIT // 2
    labeled = set(
        rng.choice(ai, min(k, len(ai)), replace=False).tolist() +
        rng.choice(ni, min(k, len(ni)), replace=False).tolist()
    )
    queried = 0
    while True:
        labeled_arr   = np.array(sorted(labeled))
        unlabeled_arr = np.array([i for i in range(n_pool) if i not in labeled])
        X_lab = X_pool[labeled_arr];  y_lab = y_pool[labeled_arr]

        if queried >= budget or len(unlabeled_arr) < BATCH_Q:
            break

        np.random.seed(int(seed) + queried)
        q    = min(BATCH_Q, len(unlabeled_arr))
        X_unl= X_pool[unlabeled_arr]

        if len(np.unique(y_lab)) < 2 or method == "Random":
            lq = query_random(X_unl, q)
        else:
            clf_q = train_csvm(X_lab, y_lab, r_plus=r_plus_train)
            if method == "Standard AL":
                lq = query_standard_al(clf_q, X_unl, q)
            else:
                lq = query_cal(clf_q, X_unl, q)

        for li in lq:
            labeled.add(int(unlabeled_arr[li]))
        queried += q

    # Final model trained on all labeled data
    labeled_arr = np.array(sorted(labeled))
    X_lab = X_pool[labeled_arr];  y_lab = y_pool[labeled_arr]
    clf_final = train_csvm(X_lab, y_lab, r_plus=r_plus_train)
    return clf_final, labeled_arr

# ─── Pixel Map Builder ────────────────────────────────────────────────────────
COLORS = {
    'TP':   np.array([0,   200,  0  ], dtype=np.uint8),   # bright green
    'FN':   np.array([220,  30,  30 ], dtype=np.uint8),   # red
    'FP':   np.array([30,   80,  220], dtype=np.uint8),   # blue
    'TN':   np.array([15,   15,  15 ], dtype=np.uint8),   # near-black
    'pool': np.array([85,   85,  85 ], dtype=np.uint8),   # gray
}

def build_overlay_map(fci, seg, y_all, y_pred_all, h, w, alpha=0.55):
    """
    Overlay detection outcomes on false-colour image.
    Predicts on ALL superpixels — no gray pool, no speckles.
    TN = transparent (background shows through).
    TP = green, FN = red, FP = blue overlaid semi-transparently.
    """
    n_sp = int(seg.max()) + 1

    # outcome per superpixel: 0=TN,1=FP,2=FN,3=TP
    outcomes = np.zeros(n_sp, dtype=np.int32)   # default TN
    gt  = y_all.astype(int)
    pr  = y_pred_all.astype(int)
    out = gt * 2 + pr   # 0=TN,1=FP,2=FN,3=TP
    outcomes[:len(out)] = out   # superpixels not in all_idx keep TN

    # Pixel-level outcome map (vectorised)
    out_pix = outcomes[seg.ravel()].reshape(h, w)   # 0-3

    # Start from false-colour base
    result = fci.astype(np.float32).copy()

    overlay_colors = {
        1: np.array([30,  80, 220], dtype=np.float32),   # FP — blue
        2: np.array([220, 30,  30], dtype=np.float32),   # FN — red
        3: np.array([0,  200,   0], dtype=np.float32),   # TP — green
    }
    for code, color in overlay_colors.items():
        mask = (out_pix == code)
        result[mask] = (1 - alpha) * result[mask] + alpha * color

    return np.clip(result, 0, 255).astype(np.uint8)

def false_colour(img, gamma=0.45):
    """NIR/R/G false colour (band indices 4,2,1), gamma-corrected, uint8."""
    fci = np.stack([img[:,:,4], img[:,:,2], img[:,:,1]], axis=2).astype(np.float32)
    fci -= fci.min()
    mx   = fci.max()
    if mx > 0:
        fci /= mx
    return (np.power(fci, gamma) * 255).clip(0, 255).astype(np.uint8)

def rgb_colour(img, gamma=0.45):
    """True RGB composite (band indices 2,1,0 = R,G,B), gamma-corrected, uint8."""
    rgb = np.stack([img[:,:,2], img[:,:,1], img[:,:,0]], axis=2).astype(np.float32)
    rgb -= rgb.min()
    mx   = rgb.max()
    if mx > 0:
        rgb /= mx
    return (np.power(rgb, gamma) * 255).clip(0, 255).astype(np.uint8)

# Mahalanobis thresholds (from generate_mahalanobis_baseline.py, Youden's J)
MAH_THRESHOLDS = {"z1": 5.5633, "z2": 4.6594, "e1": 4.1169, "e2": 5.2268}

# ─── Legend ───────────────────────────────────────────────────────────────────
def make_legend():
    return [
        mpatches.Patch(color=(0, 200/255, 0),     label='TP — detected anomaly'),
        mpatches.Patch(color=(220/255, 30/255, 30/255), label='FN — missed anomaly'),
        mpatches.Patch(color=(30/255, 80/255, 220/255), label='FP — false alarm'),
        mpatches.Patch(color=(0.4, 0.4, 0.4),     label='TN — background (no overlay)'),
    ]

# ─── Per-scene stats ──────────────────────────────────────────────────────────
def compute_stats(y_test, y_pred):
    TP = int(np.sum((y_test==1)&(y_pred==1)))
    FN = int(np.sum((y_test==1)&(y_pred==0)))
    FP = int(np.sum((y_test==0)&(y_pred==1)))
    TN = int(np.sum((y_test==0)&(y_pred==0)))
    P  = TP + FN
    fnr = FN/P if P>0 else float('nan')
    return TP, FN, FP, TN, fnr

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("="*60)
    print("  Anomaly Map Generation — Standard AL vs cAL (r+=4)")
    print("="*60)

    # ── 1. Load all scenes ────────────────────────────────────────
    print("\n[1/3] Loading scenes...")
    scenes_data = {}
    for name, (w, h) in IMAGES.items():
        t0 = time.time()
        img, gt, seg = load_scene(name, w, h)
        X, y = extract_features(img, gt, seg)
        fci  = rgb_colour(img)
        # Mahalanobis overlay (feature index 18 = mah score in 19-D)
        mah_score = X[:, 18]
        mah_pred  = (mah_score > MAH_THRESHOLDS[name]).astype(int)
        pmap_mah  = build_overlay_map(fci, seg, y, mah_pred, h, w)
        scenes_data[name] = dict(X=X, y=y, seg=seg, img=img, fci=fci,
                                 pmap_mah=pmap_mah, w=w, h=h)
        print(f"  {name}: {len(X):,} superpixels  [{time.time()-t0:.1f}s]")

    SCENE_SEED_OFFSET = {"z1": 0, "z2": 10, "e1": 20, "e2": 30}

    # ── 3. Run AL per scene ───────────────────────────────────────
    print("\n[2/3] Running AL experiments (single seed for reproducibility)...")
    scene_maps = {}

    for name, (w, h) in IMAGES.items():
        d = scenes_data[name]
        X, y, seg = d['X'], d['y'], d['seg']

        # Per-scene scaler (matches al_cal_experiment.py)
        rng_split = np.random.default_rng(SEED_BASE)
        ai = np.where(y==1)[0]; ni = np.where(y==0)[0]
        ca, cn = int(0.7*len(ai)), int(0.7*len(ni))
        pool_idx = np.concatenate([
            rng_split.choice(ai, ca, replace=False),
            rng_split.choice(ni, cn, replace=False)
        ])
        test_idx = np.array([i for i in range(len(y)) if i not in set(pool_idx.tolist())])
        scaler = StandardScaler().fit(X[pool_idx])
        Xsc  = scaler.transform(X).astype(np.float32)
        X_pool, y_pool = Xsc[pool_idx], y[pool_idx]
        X_test, y_test = Xsc[test_idx], y[test_idx]
        budget = min(MAX_BUDGET, int(MAX_PCT * len(pool_idx)))

        seed = SEED_BASE + RUN_ID*100 + SCENE_SEED_OFFSET[name]
        print(f"\n  {name.upper()}  budget={budget}  seed={seed}")

        # Standard AL (r+=1, symmetric SVM)
        print(f"    Standard AL...", end='', flush=True)
        t0 = time.time()
        clf_std, _ = run_al("Standard AL", 1, X_pool, y_pool, seed, budget)
        # Predict on TEST set for stats, ALL superpixels for map
        y_pred_std_test = (clf_std.decision_function(X_test) >= 0).astype(int)
        y_pred_std_all  = (clf_std.decision_function(Xsc) >= 0).astype(int)
        TP,FN,FP,TN,fnr = compute_stats(y_test, y_pred_std_test)
        print(f" FNR={fnr:.3f}  TP={TP} FN={FN} FP={FP}  [{time.time()-t0:.1f}s]")

        # cAL r+=4
        print(f"    cAL r+=4...", end='', flush=True)
        t0 = time.time()
        clf_cal, _ = run_al("cAL", 4, X_pool, y_pool, seed, budget)
        y_pred_cal_test = (clf_cal.decision_function(X_test) >= 0).astype(int)
        y_pred_cal_all  = (clf_cal.decision_function(Xsc) >= 0).astype(int)
        TP,FN,FP,TN,fnr = compute_stats(y_test, y_pred_cal_test)
        print(f" FNR={fnr:.3f}  TP={TP} FN={FN} FP={FP}  [{time.time()-t0:.1f}s]")

        # Build overlay maps (all superpixels, on false-colour background)
        all_idx = np.arange(len(y))
        pmap_std = build_overlay_map(d['fci'], seg, y[all_idx], y_pred_std_all, h, w)
        pmap_cal = build_overlay_map(d['fci'], seg, y[all_idx], y_pred_cal_all, h, w)

        scene_maps[name] = dict(
            fci=d['fci'], pmap_mah=d['pmap_mah'],
            pmap_std=pmap_std, pmap_cal=pmap_cal,
            std_stats=compute_stats(y_test, y_pred_std_test),
            cal_stats=compute_stats(y_test, y_pred_cal_test),
        )

    # ── 4. Generate figures ───────────────────────────────────────
    print("\n[3/3] Generating figures...")
    legend_patches = make_legend()

    # ── 4a. All-scenes figure (4 rows × 3 cols) ──────────────────
    fig, axes = plt.subplots(4, 3, figsize=(18, 24))
    fig.patch.set_facecolor('#111111')

    scene_labels = {"z1":"Z1 (Oitavén Sep)", "z2":"Z2 (Oitavén Oct)",
                    "e1":"E1 (Ermidas 1)",   "e2":"E2 (Ermidas 2)"}

    for row, name in enumerate(SCENES):
        m = scene_maps[name]
        st, cal = m['std_stats'], m['cal_stats']

        ax_fci = axes[row, 0]
        ax_std = axes[row, 1]
        ax_cal = axes[row, 2]

        ax_fci.imshow(m['pmap_mah'])
        ax_fci.set_title(f"{scene_labels[name]}\nMahalanobis Baseline",
                         color='white', fontsize=10, fontweight='bold')
        ax_fci.axis('off')

        ax_std.imshow(m['pmap_std'])
        ax_std.set_title(
            f"Standard AL\nFNR={st[4]:.3f}  TP={st[0]} FN={st[1]} FP={st[2]}",
            color='white', fontsize=10, fontweight='bold')
        ax_std.axis('off')

        ax_cal.imshow(m['pmap_cal'])
        fnr_red = (st[4]-cal[4])/st[4]*100 if st[4]>0 else 0
        ax_cal.set_title(
            f"cAL  r⁺=4\nFNR={cal[4]:.3f}  TP={cal[0]} FN={cal[1]} FP={cal[2]}"
            f"  (↓{fnr_red:.0f}% FNR)",
            color='white', fontsize=10, fontweight='bold')
        ax_cal.axis('off')

    fig.legend(handles=legend_patches, loc='lower center', ncol=5,
               fontsize=10, facecolor='#222222', labelcolor='white',
               framealpha=0.9, bbox_to_anchor=(0.5, 0.01))

    plt.suptitle("Anomaly Detection Maps — Standard AL vs cAL (r⁺=4)\n"
                 "UAV Multispectral Riparian Zone  ·  Final Budget (600 queries)",
                 color='white', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0.04, 1, 0.99])
    p = OUT / "anomaly_maps_all_scenes.png"
    plt.savefig(p, dpi=150, bbox_inches='tight', facecolor='#111111')
    plt.close()
    print(f"  → {p}")

    # ── 4b. Z2 + E2 focused figure (2 rows × 3 cols) for paper ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.patch.set_facecolor('#111111')

    for row, name in enumerate(["z2", "e2"]):
        m = scene_maps[name]
        st, cal = m['std_stats'], m['cal_stats']

        ax_fci = axes[row, 0]
        ax_std = axes[row, 1]
        ax_cal = axes[row, 2]

        ax_fci.imshow(m['pmap_mah'])
        ax_fci.set_title(f"{scene_labels[name]}\nMahalanobis Baseline",
                         color='white', fontsize=12, fontweight='bold')
        ax_fci.axis('off')

        ax_std.imshow(m['pmap_std'])
        ax_std.set_title(
            f"Standard AL\nFNR = {st[4]:.3f}   FN = {st[1]}",
            color='white', fontsize=12, fontweight='bold')
        ax_std.axis('off')

        ax_cal.imshow(m['pmap_cal'])
        fnr_red = (st[4]-cal[4])/st[4]*100 if st[4]>0 else 0
        ax_cal.set_title(
            f"cAL   r⁺ = 4\nFNR = {cal[4]:.3f}   FN = {cal[1]}   (↓{fnr_red:.0f}% FNR)",
            color='white', fontsize=12, fontweight='bold')
        ax_cal.axis('off')

    fig.legend(handles=legend_patches, loc='lower center', ncol=5,
               fontsize=11, facecolor='#222222', labelcolor='white',
               framealpha=0.9, bbox_to_anchor=(0.5, 0.01))

    plt.suptitle("Anomaly Detection Maps — Challenging Scenes\n"
                 "Standard AL vs Cost-Sensitive AL (r⁺=4)  ·  600 Labeled Superpixels",
                 color='white', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0.06, 1, 0.99])
    p = OUT / "anomaly_maps_z2_e2.png"
    plt.savefig(p, dpi=180, bbox_inches='tight', facecolor='#111111')
    plt.close()
    print(f"  → {p}")

    # ── 4c. Individual scene figures ─────────────────────────────
    for name in SCENES:
        m   = scene_maps[name]
        st, cal = m['std_stats'], m['cal_stats']
        fnr_red = (st[4]-cal[4])/st[4]*100 if st[4]>0 else 0

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.patch.set_facecolor('#111111')

        axes[0].imshow(m['fci'])
        axes[0].set_title(f"{scene_labels[name]}\nFalse Colour (NIR/R/G)",
                          color='white', fontsize=12, fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(m['pmap_std'])
        axes[1].set_title(
            f"Standard AL\nFNR={st[4]:.3f}  TP={st[0]} FN={st[1]} FP={st[2]}",
            color='white', fontsize=12, fontweight='bold')
        axes[1].axis('off')

        axes[2].imshow(m['pmap_cal'])
        axes[2].set_title(
            f"cAL  r⁺=4\nFNR={cal[4]:.3f}  TP={cal[0]} FN={cal[1]} FP={cal[2]}"
            f"   ↓{fnr_red:.0f}% FNR",
            color='white', fontsize=12, fontweight='bold')
        axes[2].axis('off')

        fig.legend(handles=legend_patches, loc='lower center', ncol=5,
                   fontsize=10, facecolor='#222222', labelcolor='white',
                   framealpha=0.9, bbox_to_anchor=(0.5, 0.00))

        plt.tight_layout(rect=[0, 0.08, 1, 1.0])
        p = OUT / f"anomaly_map_{name}.png"
        plt.savefig(p, dpi=150, bbox_inches='tight', facecolor='#111111')
        plt.close()
        print(f"  → {p}")

    print("\nDONE ✓")
    print(f"Outputs saved to: {OUT}/")


if __name__ == "__main__":
    main()
