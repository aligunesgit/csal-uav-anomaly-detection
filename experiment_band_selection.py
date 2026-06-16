#!/usr/bin/env python3
"""
Explainable Band Selection for Multispectral Anomaly Detection
in UAV-Acquired Fluvial Imagery — Main Experiment Script
"""
import sys
import numpy as np
import json, os, time
from itertools import combinations
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                              recall_score, accuracy_score, cohen_kappa_score)
import shap

_ROOT = Path(__file__).resolve().parent
BASE  = _ROOT / "data"
OUT   = _ROOT / "results"
OUT.mkdir(exist_ok=True)

IMAGES = {"z1": (3807,2141), "z2": (2081,957), "e1": (3629,961), "e2": (1094,707)}
BANDS  = ["Blue(475)", "Green(560)", "Red(668)", "RedEdge(717)", "NIR(840)"]
BS     = ["B","G","R","RE","N"]

def load_image(name, w, h):
    with open(BASE / name / f"{name}.raw", 'rb') as f:
        f.read(12)
        data = np.frombuffer(f.read(), dtype=np.uint32).reshape(h, w, 5).astype(np.float32)
    with open(BASE / name / f"{name}_gt.pgm", 'rb') as f:
        f.readline(); f.readline(); f.readline()
        gt = np.frombuffer(f.read(), dtype=np.uint8).reshape(h, w)
    return data, (gt == 2).astype(np.uint8)

def sample_pixels(data, gt, band_idx, n_per_class=8000, seed=42):
    rng = np.random.default_rng(seed)
    ia = np.argwhere(gt==1); ino = np.argwhere(gt==0)
    na = min(n_per_class, len(ia)); nn = min(n_per_class, len(ino))
    sa = ia[rng.choice(len(ia),na,replace=False)]
    sn = ino[rng.choice(len(ino),nn,replace=False)]
    sel = np.vstack([sa,sn])
    X = data[sel[:,0], sel[:,1], :][:, band_idx]
    y = gt[sel[:,0], sel[:,1]]
    return X.astype(np.float32), y

# Load all images
print("Loading images...")
ALL = {}
for name,(w,h) in IMAGES.items():
    data,gt = load_image(name,w,h)
    ALL[name] = (data,gt)
    print(f"  {name}: anomaly={gt.sum():,} ({100*gt.mean():.2f}%)")

# ── EXP 1: 31 Combinations LOCO-CV ───────────────────────────────────────────
print("\n=== EXP 1: 31 Band Combinations (Leave-One-Image-Out) ===")
names = list(IMAGES.keys())
all_combos = []
for r in range(1,6):
    for c in combinations(range(5),r):
        all_combos.append(list(c))

results = []
for ci, bidx in enumerate(all_combos):
    label = "+".join(BS[b] for b in bidx)
    folds = []
    for test in names:
        train_names = [n for n in names if n!=test]
        Xtr,ytr=[],[]
        for n in train_names:
            X,y = sample_pixels(ALL[n][0], ALL[n][1], bidx, 8000)
            Xtr.append(X); ytr.append(y)
        Xtr = np.vstack(Xtr); ytr = np.concatenate(ytr)
        Xte,yte = sample_pixels(ALL[test][0], ALL[test][1], bidx, 15000)
        rf = RandomForestClassifier(150,max_depth=15,n_jobs=-1,random_state=42,class_weight='balanced')
        rf.fit(Xtr,ytr)
        yp = rf.predict_proba(Xte)[:,1]; ypred = rf.predict(Xte)
        folds.append({
            "test": test,
            "auc":  float(roc_auc_score(yte,yp)),
            "f1":   float(f1_score(yte,ypred,zero_division=0)),
            "prec": float(precision_score(yte,ypred,zero_division=0)),
            "rec":  float(recall_score(yte,ypred,zero_division=0)),
            "oa":   float(accuracy_score(yte,ypred)),
            "kappa":float(cohen_kappa_score(yte,ypred))
        })
    avg = {k: float(np.mean([m[k] for m in folds])) for k in ["auc","f1","prec","rec","oa","kappa"]}
    std = {k: float(np.std( [m[k] for m in folds])) for k in ["auc","f1","prec","rec","oa","kappa"]}
    results.append({"idx":ci+1,"bands":bidx,"label":label,"n_bands":len(bidx),"avg":avg,"std":std,"folds":folds})
    mark = "★★★" if len(bidx)==5 else ("★★ " if avg["auc"]>=0.95 else ("★  " if avg["auc"]>=0.90 else "   "))
    print(f"[{ci+1:2d}/31] {mark} {label:<12} AUC={avg['auc']:.4f}±{std['auc']:.4f}  F1={avg['f1']:.4f}  K={avg['kappa']:.4f}")

with open(OUT/"band_combination_results.json","w") as f: json.dump(results,f,indent=2)
print(f"\nSaved → {OUT}/band_combination_results.json")

# Summary
sres = sorted(results, key=lambda x: x["avg"]["auc"], reverse=True)
print("\n--- TOP 10 by AUC ---")
for i,r in enumerate(sres[:10]):
    print(f"  #{i+1:2d} {r['label']:<14} n={r['n_bands']} AUC={r['avg']['auc']:.4f} F1={r['avg']['f1']:.4f} K={r['avg']['kappa']:.4f}")

full5 = next(r for r in results if r['n_bands']==5)
print(f"\nFull 5-band baseline: AUC={full5['avg']['auc']:.4f}")
for nb in range(1,5):
    best = max([r for r in results if r['n_bands']==nb], key=lambda x:x['avg']['auc'])
    d = best['avg']['auc']-full5['avg']['auc']
    print(f"  Best {nb}-band: {best['label']:<12} AUC={best['avg']['auc']:.4f}  Δ={d:+.4f}")

# ── EXP 2: SHAP Analysis ──────────────────────────────────────────────────────
print("\n=== EXP 2: SHAP Analysis (full 5-band) ===")
Xp,yp=[],[]
for name,(data,gt) in ALL.items():
    X,y = sample_pixels(data,gt,list(range(5)),5000)
    Xp.append(X); yp.append(y)
Xp=np.vstack(Xp); yp=np.concatenate(yp)

rf_full = RandomForestClassifier(200,max_depth=15,n_jobs=-1,random_state=42,class_weight='balanced')
rf_full.fit(Xp,yp)
print(f"RF trained on {len(Xp):,} pixels")

rng=np.random.default_rng(0)
idx_s=rng.choice(len(Xp),min(3000,len(Xp)),replace=False)
Xs=Xp[idx_s]; ys=yp[idx_s]
exp = shap.TreeExplainer(rf_full)
sv = exp.shap_values(Xs)
if isinstance(sv, list):
    sv_a = sv[1]
elif sv.ndim == 3:
    sv_a = sv[:, :, 1]
else:
    sv_a = sv

mean_shap = np.abs(sv_a).mean(axis=0)
shap_imp = {BANDS[b]:float(mean_shap[b]) for b in range(5)}
shap_rank = sorted(shap_imp.items(),key=lambda x:x[1],reverse=True)
total_shap = sum(shap_imp.values())

print("\nGlobal SHAP (anomaly class, mean |SHAP|):")
for i,(bn,v) in enumerate(shap_rank):
    bar="█"*int(v/mean_shap.max()*30)
    print(f"  #{i+1} {bn:<18} {v:7.4f} ({100*v/total_shap:5.1f}%)  {bar}")

# Per-image SHAP
print("\nPer-image SHAP ranking:")
img_shap={}
for name,(data,gt) in ALL.items():
    Xi,yi = sample_pixels(data,gt,list(range(5)),2000)
    svi = exp.shap_values(Xi)
    if isinstance(svi, list):
        svi_a = svi[1]
    elif svi.ndim == 3:
        svi_a = svi[:, :, 1]
    else:
        svi_a = svi
    imp = np.abs(svi_a).mean(axis=0)
    img_shap[name]={BANDS[b]:float(imp[b]) for b in range(5)}
    ranked=sorted(enumerate(imp),key=lambda x:x[1],reverse=True)
    print(f"  {name}: {' > '.join(BS[b] for b,_ in ranked)}")

# Anomaly vs Normal SHAP
msk_a=(ys==1); msk_n=(ys==0)
sv_aa=np.abs(sv_a[msk_a]).mean(axis=0)
sv_na=np.abs(sv_a[msk_n]).mean(axis=0)
print("\nSHAP — Anomaly pixels vs Normal pixels:")
print(f"  {'Band':<20} {'Anomaly':>10} {'Normal':>10} {'Ratio':>8}")
for b in range(5):
    print(f"  {BANDS[b]:<20} {sv_aa[b]:>10.4f} {sv_na[b]:>10.4f} {sv_aa[b]/(sv_na[b]+1e-8):>8.2f}x")

# ── EXP 3: Minimum Band Set ───────────────────────────────────────────────────
print("\n=== EXP 3: Minimum Sufficient Band Set ===")
fa = full5['avg']['auc']
print(f"Full 5-band AUC: {fa:.4f}")
for thr,label in [(0.99,"99%"),(0.97,"97%"),(0.95,"95%")]:
    passing = [(r['label'],r['n_bands'],r['avg']['auc'])
               for r in results if r['avg']['auc']>=fa*thr]
    min_n = min(p[1] for p in passing) if passing else 5
    best_p= min((p for p in passing if p[1]==min_n), key=lambda x:-x[2], default=None)
    print(f"  Threshold {label} (≥{fa*thr:.4f}): min bands={min_n}, best={best_p}")

# Save SHAP
with open(OUT/"shap_results.json","w") as f:
    json.dump({"global":shap_imp,"rank":[b for b,_ in shap_rank],
               "per_image":img_shap,
               "anomaly_shap":{BANDS[b]:float(sv_aa[b]) for b in range(5)},
               "normal_shap": {BANDS[b]:float(sv_na[b]) for b in range(5)}},f,indent=2)
print(f"\nAll saved to {OUT}/")
print("DONE.")
