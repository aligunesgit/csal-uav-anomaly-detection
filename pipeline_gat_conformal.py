#!/usr/bin/env python3
"""
SpectralGAT-Conformal Pipeline  v2
====================================
Multi-Task Graph Attention Network + Split Conformal Prediction
for Riparian Zone Anomaly Detection in UAV Multispectral Imagery.

Key fixes in v2:
  - Focal loss (γ=2) to combat ~96% normal / ~4% anomaly imbalance
  - Balanced mini-batch sampling (equal anomaly / normal per step)
  - Proper marginal conformal calibration across all training scenes
  - Correct conformal coverage metric (marginal, not conditional)
  - Domain-invariant features: ratio bands + normalised spectral distances
  - Inductive node embedding via GraphSAGE-style mean aggregation before GAT

Architecture:
  Superpixel Features (14-D) per scene
  → Spectral Attention Module (learnable band weights, 5→5)
  → GraphSAGE mean aggregation (neighbourhood smoothing)
  → GAT Layer 1  (64-D, 4 heads)
  → GAT Layer 2  (64-D, 4 heads)
  → Anomaly Detection Head   (binary, focal-loss trained)
  → Reconstruction Head      (spectral self-supervision)
  → Split Conformal           (marginal coverage ≥ 1-α)
  → Gradient × Input Band Attribution

Evaluation: Leave-One-Image-Out Cross-Validation (LOCO-CV)
"""

import sys, warnings, time, json
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                              recall_score, accuracy_score, cohen_kappa_score,
                              average_precision_score)
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─── Configuration ────────────────────────────────────────────────────────────

_ROOT = Path(__file__).resolve().parent
BASE  = _ROOT / "data"
OUT   = _ROOT / "results"
OUT.mkdir(exist_ok=True)

IMAGES = {"z1": (3807, 2141), "z2": (2081, 957),
          "e1": (3629,  961), "e2": (1094,  707)}
BANDS  = ["Blue(475)", "Green(560)", "Red(668)", "RedEdge(717)", "NIR(840)"]
BS     = ["B", "G", "R", "RE", "N"]

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALPHA        = 0.10    # conformal miscoverage → 90% marginal coverage target
HIDDEN_DIM   = 64
N_HEADS      = 4
EPOCHS       = 150     # CPU-feasible; ~20-30 min total for 4 folds
PATIENCE     = 30      # early stopping patience
LR           = 3e-3
LAM_REC      = 0.2
FOCAL_GAMMA  = 2.0     # focal loss: down-weight easy negatives
BATCH_NODES  = 512     # balanced mini-batch size (per class) per step

print(f"Device : {DEVICE}")
print(f"Conformal α={ALPHA}  →  target marginal coverage ≥ {100*(1-ALPHA):.0f}%\n")

# ─── 1. Data Loading ──────────────────────────────────────────────────────────

def load_scene_raw(name, w, h):
    with open(BASE / name / f"{name}.raw", 'rb') as f:
        f.read(12)
        img = (np.frombuffer(f.read(), dtype=np.uint32)
               .reshape(h, w, 5).astype(np.float32))
    with open(BASE / name / f"{name}_gt.pgm", 'rb') as f:
        f.readline(); f.readline(); f.readline()
        raw = f.read()
        gt  = np.frombuffer(raw[:h * w], dtype=np.uint8).reshape(h, w)
    with open(BASE / name / f"{name}_seg.raw", 'rb') as f:
        f.read(8)
        seg = (np.frombuffer(f.read(), dtype=np.uint32).reshape(h, w))
    return img, gt, seg


def extract_features(img, gt, seg):
    """
    Vectorised per-superpixel feature extraction (21 features):
      5 band means, 5 band stds,
      NDVI, NDRE, ExG, EVI, BNDVI,
      R/B ratio, NIR/R ratio, NIR/RE ratio,
      Mahalanobis-like anomaly score (||mu - global_mu|| / global_std)

    Labels: 0=normal (gt==1), 1=anomaly (gt==2)
    """
    n_segs   = int(seg.max()) + 1
    seg_flat = seg.ravel().astype(np.int64)
    img_flat = img.reshape(-1, 5).astype(np.float64)
    gt_flat  = gt.ravel().astype(np.int64)
    counts   = np.bincount(seg_flat, minlength=n_segs).clip(1).astype(np.float64)

    # Band means & stds
    mu = np.zeros((n_segs, 5))
    sq = np.zeros((n_segs, 5))
    for b in range(5):
        mu[:, b] = np.bincount(seg_flat, weights=img_flat[:, b],  minlength=n_segs) / counts
        sq[:, b] = np.bincount(seg_flat, weights=img_flat[:, b]**2, minlength=n_segs) / counts
    sigma = np.sqrt(np.clip(sq - mu**2, 0, None))

    EPS  = 1e-6
    B, G, R, RE, N_ = mu[:,0], mu[:,1], mu[:,2], mu[:,3], mu[:,4]

    ndvi  = (N_ - R)  / (N_ + R  + EPS)
    ndre  = (N_ - RE) / (N_ + RE + EPS)
    exg   = 2*G - R - B
    evi   = 2.5*(N_-R) / (N_ + 6*R - 7.5*B + 1 + EPS)
    bndvi = (N_ - B)  / (N_ + B  + EPS)
    rb    = R  / (B  + EPS)
    nr    = N_ / (R  + EPS)
    nre   = N_ / (RE + EPS)

    # Global mean/std for anomaly scoring
    gmu  = mu.mean(0);  gsd = sigma.mean(0) + EPS
    mah  = np.sqrt((((mu - gmu) / gsd) ** 2).mean(1))

    feats = np.concatenate([
        mu, sigma,
        ndvi[:,None], ndre[:,None], exg[:,None], evi[:,None], bndvi[:,None],
        rb[:,None], nr[:,None], nre[:,None],
        mah[:,None]
    ], axis=1).astype(np.float32)              # (N, 21)

    # Labels: majority vote
    anom_c  = np.bincount(seg_flat, weights=(gt_flat==2).astype(np.float64), minlength=n_segs)
    norm_c  = np.bincount(seg_flat, weights=(gt_flat==1).astype(np.float64), minlength=n_segs)
    labels  = (anom_c > norm_c).astype(np.int64)

    return feats, labels


def build_adjacency(seg):
    """Fully vectorised superpixel boundary adjacency + self-loops."""
    n  = int(seg.max()) + 1
    M  = n + 1

    dh = seg[:, :-1] != seg[:, 1:]
    sh = seg[:, :-1][dh].astype(np.int64)
    dh_ = seg[:, 1:][dh].astype(np.int64)

    dv = seg[:-1, :] != seg[1:, :]
    sv = seg[:-1, :][dv].astype(np.int64)
    dv_ = seg[1:,  :][dv].astype(np.int64)

    src = np.concatenate([sh, dh_, sv, dv_])
    dst = np.concatenate([dh_, sh, dv_, sv])
    packed = np.unique(src * M + dst)
    src, dst = packed // M, packed % M

    self_ = np.arange(n, dtype=np.int64)
    src = np.concatenate([src, self_])
    dst = np.concatenate([dst, self_])
    return torch.from_numpy(np.stack([src, dst])).long()


def prepare_scene(name, w, h):
    print(f"  {name} ({w}×{h})", end=" … ", flush=True)
    t0 = time.time()
    img, gt, seg = load_scene_raw(name, w, h)
    feats, labels = extract_features(img, gt, seg)
    edge_index    = build_adjacency(seg)

    n = feats.shape[0]; na = int(labels.sum())
    print(f"nodes={n:,} edges={edge_index.shape[1]:,} "
          f"anom={na:,} ({100*na/n:.1f}%) [{time.time()-t0:.1f}s]")

    return {"name": name, "feats": feats, "labels": labels,
            "edge_index": edge_index, "seg_idx": seg.astype(np.int64),
            "img": img, "gt": gt, "seg": seg}


# ─── 2. Model ─────────────────────────────────────────────────────────────────

IN_DIM = 19        # 5 means + 5 stds + 5 indices + 3 ratios + 1 Mah = 19

class SpectralAttention(nn.Module):
    def __init__(self, n=5):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n, n*2), nn.Tanh(),
            nn.Linear(n*2, n), nn.Softmax(dim=-1))

    def forward(self, x):
        w  = self.fc(x[:, :5])
        xm = x[:, :5] * w
        xs = x[:, 5:10] * w
        return torch.cat([xm, xs, x[:, 10:], w], dim=1), w  # (N, 21+5=26)


class SAGEMean(nn.Module):
    """
    GraphSAGE-style mean aggregation — neighbourhood smoothing before GAT.
    Concatenates node's own features with mean of neighbours.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim * 2, out_dim)

    def forward(self, x, edge_index):
        N   = x.size(0)
        s, d = edge_index
        agg = torch.zeros_like(x)
        agg.scatter_add_(0, d.unsqueeze(1).expand_as(x[s]), x[s])
        deg = torch.zeros(N, 1, device=x.device)
        deg.scatter_add_(0, d.unsqueeze(1), torch.ones(len(s), 1, device=x.device))
        mean = agg / deg.clamp(1)
        return F.relu(self.fc(torch.cat([x, mean], dim=1)))


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, H=4, drop=0.3):
        super().__init__()
        assert out_dim % H == 0
        self.H, self.D = H, out_dim // H
        self.W     = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(1, H, self.D))
        self.a_dst = nn.Parameter(torch.empty(1, H, self.D))
        self.bias  = nn.Parameter(torch.zeros(out_dim))
        self.drop  = nn.Dropout(drop)
        self.leaky = nn.LeakyReLU(0.2)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, x, ei):
        N = x.size(0); s, d = ei
        h = self.W(x).view(N, self.H, self.D)
        e = self.leaky((h*self.a_src).sum(-1)[s] + (h*self.a_dst).sum(-1)[d])
        e = e - e.max()
        ae = torch.exp(e)
        Z  = torch.zeros(N, self.H, device=x.device)
        Z.scatter_add_(0, d.unsqueeze(1).expand_as(ae), ae)
        w  = self.drop(ae / (Z[d] + 1e-8))
        agg = torch.zeros(N, self.H, self.D, device=x.device)
        agg.scatter_add_(0, d.view(-1,1,1).expand_as(h[s]*w.unsqueeze(-1)),
                         h[s]*w.unsqueeze(-1))
        return agg.view(N, -1) + self.bias


class SpectralGAT(nn.Module):
    def __init__(self, in_dim=IN_DIM, hidden=HIDDEN_DIM, H=N_HEADS, drop=0.3):
        super().__init__()
        self.spec = SpectralAttention(5)
        aug = in_dim + 5          # 21 + 5 = 26

        self.sage = SAGEMean(aug, hidden)
        self.gat1 = GATLayer(hidden, hidden, H, drop)
        self.bn1  = nn.BatchNorm1d(hidden)
        self.gat2 = GATLayer(hidden, hidden, H, drop)
        self.bn2  = nn.BatchNorm1d(hidden)

        self.anomaly_head = nn.Sequential(
            nn.Linear(hidden, hidden//2), nn.ReLU(),
            nn.Dropout(drop), nn.Linear(hidden//2, 2))
        self.recon_head = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 5))

    def forward(self, x, ei):
        x_aug, bw = self.spec(x)
        h = self.sage(x_aug, ei)
        h = F.elu(self.bn1(self.gat1(h, ei)))
        h = F.elu(self.bn2(self.gat2(h, ei)))
        return self.anomaly_head(h), self.recon_head(h), bw


# ─── 3. Focal Loss ────────────────────────────────────────────────────────────

def focal_loss(logits, targets, gamma=FOCAL_GAMMA, w_pos=1.0):
    """Focal loss for binary classification with class weighting."""
    weights = torch.where(targets == 1,
                          torch.tensor(w_pos, device=logits.device),
                          torch.tensor(1.0,   device=logits.device))
    ce   = F.cross_entropy(logits, targets, reduction='none')
    pt   = torch.exp(-ce)
    loss = weights * (1 - pt) ** gamma * ce
    return loss.mean()


# ─── 4. Conformal Prediction (Marginal) ───────────────────────────────────────

class SplitConformal:
    """
    Split (inductive) conformal prediction.
    Marginal guarantee: P(true_label ∈ C(X)) ≥ 1 - alpha.

    Nonconformity score: s(x, y) = 1 - softmax_prob[y]
    Calibrate on all labeled calibration nodes regardless of class.
    """
    def __init__(self, alpha=ALPHA):
        self.alpha = alpha
        self.q     = 0.5          # will be overwritten after calibrate()

    def calibrate(self, probs, true_labels):
        """
        probs       : (N, 2) softmax probabilities
        true_labels : (N,)   0 or 1
        """
        scores = 1.0 - probs[np.arange(len(true_labels)), true_labels]
        n      = len(scores)
        q_lev  = min(np.ceil((n+1)*(1-self.alpha)) / n, 1.0)
        self.q = float(np.quantile(scores, q_lev))
        cov    = float((scores <= self.q).mean())
        print(f"    Conformal q={self.q:.4f}  "
              f"cal-nodes={n:,}  cal-empirical-cov={cov:.3f}")

    def predict(self, probs):
        """
        Returns:
          pred      : hard label argmax
          uncertain : nodes where BOTH classes have score ≤ q (prediction set size 2)
        """
        in_set_0 = (1 - probs[:, 0]) <= self.q
        in_set_1 = (1 - probs[:, 1]) <= self.q
        uncertain = in_set_0 & in_set_1          # both in set → uncertain
        pred      = np.argmax(probs, axis=1)
        return pred, uncertain

    def marginal_coverage(self, probs, true_labels):
        scores = 1.0 - probs[np.arange(len(true_labels)), true_labels]
        return float((scores <= self.q).mean())


# ─── 5. Training ──────────────────────────────────────────────────────────────

def balanced_sample(labels, k=BATCH_NODES, rng=None):
    """Return indices: k anomaly + k normal (oversamples if needed)."""
    if rng is None: rng = np.random.default_rng(42)
    anom = np.where(labels == 1)[0]
    norm = np.where(labels == 0)[0]
    k_a  = min(k, len(anom))
    k_n  = min(k, len(norm))
    idx_a = rng.choice(anom, k_a, replace=len(anom) < k)
    idx_n = rng.choice(norm, k_n, replace=len(norm) < k)
    return np.concatenate([idx_a, idx_n])


def train_model(train_scenes, scaler, device, epochs=EPOCHS, seed=0):
    model = SpectralGAT(in_dim=IN_DIM).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sch   = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10,
                                                        factor=0.5, min_lr=1e-5)
    rng   = np.random.default_rng(seed)

    # Pre-move tensors to device
    scene_tensors = []
    for sc in train_scenes:
        X   = torch.tensor(scaler.transform(sc["feats"]),
                           dtype=torch.float32, device=device)
        y   = torch.tensor(sc["labels"], dtype=torch.long, device=device)
        ei  = sc["edge_index"].to(device)
        mu5 = torch.tensor(sc["feats"][:, :5], dtype=torch.float32, device=device)
        scene_tensors.append((X, y, ei, mu5, sc["labels"]))

    # Class weight
    all_y = np.concatenate([sc["labels"] for sc in train_scenes])
    w_pos = min(float((all_y==0).sum()) / float((all_y==1).sum()+1), 20.0)
    print(f"    w_pos={w_pos:.1f}  anom={int((all_y==1).sum()):,}  normal={int((all_y==0).sum()):,}")

    best_loss, best_state, no_improve = float('inf'), None, 0

    for ep in range(epochs):
        model.train()
        total_cls = total_rec = 0.0

        for X, y_t, ei, mu5, y_np in scene_tensors:
            idx   = balanced_sample(y_np, BATCH_NODES, rng)
            idx_t = torch.tensor(idx, dtype=torch.long, device=device)
            logits, recon, bw = model(X, ei)
            loss_cls = focal_loss(logits[idx_t], y_t[idx_t], gamma=FOCAL_GAMMA, w_pos=w_pos)
            tgt      = mu5 / (mu5.amax() + 1e-8)
            loss_rec = F.l1_loss(recon[idx_t], tgt[idx_t])
            loss     = loss_cls + LAM_REC * loss_rec
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_cls += loss_cls.item(); total_rec += loss_rec.item()

        total = total_cls + LAM_REC * total_rec
        sch.step(total)

        if total < best_loss - 1e-5:
            best_loss = total
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (ep+1) % 50 == 0:
            print(f"    ep {ep+1}/{epochs}  cls={total_cls:.4f}  "
                  f"rec={total_rec:.4f}  best={best_loss:.4f}")

        if no_improve >= PATIENCE:
            print(f"    Early stop at ep {ep+1}  (patience={PATIENCE})")
            break

    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def infer(model, scene, scaler, device):
    model.eval()
    X  = torch.tensor(scaler.transform(scene["feats"]),
                      dtype=torch.float32, device=device)
    ei = scene["edge_index"].to(device)
    logits, _, bw = model(X, ei)
    probs = F.softmax(logits, dim=-1).cpu().numpy()   # (N, 2)
    bw    = bw.cpu().numpy()                           # (N, 5)
    return probs, bw


# ─── 6. Band Attribution ──────────────────────────────────────────────────────

def grad_band_importance(model, scene, scaler, device):
    """Gradient × Input attribution per band (mean + std columns)."""
    model.eval()
    X = torch.tensor(scaler.transform(scene["feats"]),
                     dtype=torch.float32, device=device).requires_grad_(True)
    ei = scene["edge_index"].to(device)
    logits, _, _ = model(X, ei)
    F.softmax(logits, dim=-1)[:, 1].mean().backward()
    gi  = (X.grad.detach() * X.detach()).abs().cpu().numpy()
    imp = np.array([gi[:, b].mean() + gi[:, b+5].mean() for b in range(5)])
    return imp / (imp.sum() + 1e-12)


# ─── 7. Visualisation ─────────────────────────────────────────────────────────

def plot_importance(imp_arr, bw_arr, test_name):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    col = ['#2196F3','#4CAF50','#F44336','#FF9800','#9C27B0']

    axes[0].barh(BANDS[::-1], imp_arr[::-1], color=col[::-1], alpha=0.85)
    axes[0].axvline(0.2, ls='--', color='gray', alpha=0.5, label='Uniform')
    axes[0].set_xlabel('Norm. Gradient × Input')
    axes[0].set_title('Band Importance (Gradient × Input)')
    axes[0].legend()

    bw_mean = bw_arr.mean(0); bw_std = bw_arr.std(0)
    axes[1].bar(BS, bw_mean, color=col, alpha=0.85, yerr=bw_std, capsize=4)
    axes[1].axhline(0.2, ls='--', color='gray', alpha=0.5, label='Uniform')
    axes[1].set_ylabel('Attention Weight'); axes[1].legend()
    axes[1].set_title('Spectral Attention Weights (mean ± std)')

    plt.suptitle(f'Spectral Analysis — test: {test_name.upper()}', fontweight='bold')
    plt.tight_layout()
    p = OUT / f"band_importance_{test_name}.png"
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"    → {p.name}")


def plot_map(scene, probs, uncertain):
    name = scene["name"]
    prob_map = probs[:, 1][scene["seg_idx"]]
    unc_map  = uncertain[scene["seg_idx"]]
    gt       = scene["gt"]
    img      = scene["img"]

    fc = np.stack([img[:,:,4], img[:,:,2], img[:,:,1]], -1)
    fc = (fc - fc.min()) / (fc.max() - fc.min() + 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    axes[0].imshow(fc); axes[0].set_title('False-Colour (NIR/R/G)'); axes[0].axis('off')

    im = axes[1].imshow(prob_map, cmap='RdYlGn_r', vmin=0, vmax=1)
    axes[1].contour((gt==2).astype(float), levels=[0.5], colors=['cyan'], linewidths=0.7)
    plt.colorbar(im, ax=axes[1], fraction=0.03)
    axes[1].set_title('SpectralGAT Anomaly Probability\n(cyan = GT boundary)')
    axes[1].axis('off')

    rgb = np.zeros((*gt.shape, 3))
    pred = (prob_map >= 0.5)
    rgb[pred  & ~unc_map] = [0.90, 0.15, 0.15]
    rgb[~pred & ~unc_map] = [0.20, 0.70, 0.20]
    rgb[unc_map]          = [1.00, 0.85, 0.00]
    axes[2].imshow(rgb)
    axes[2].legend(handles=[
        mpatches.Patch(color=[.9,.15,.15], label='Anomaly (certain)'),
        mpatches.Patch(color=[.2,.70,.20], label='Normal (certain)'),
        mpatches.Patch(color=[1.,.85,.00], label='Uncertain (both classes in set)')
    ], loc='lower right', fontsize=7.5)
    axes[2].set_title(f'Conformal Map  (α={ALPHA}, marg. cov≥{100*(1-ALPHA):.0f}%)')
    axes[2].axis('off')

    plt.suptitle(f'Scene {name.upper()}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    p = OUT / f"map_{name}.png"
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"    → {p.name}")


def plot_summary(results):
    names   = [r['name'] for r in results]
    metrics = ['auc','ap','f1','kappa','coverage']
    labels  = ['AUC-ROC','Avg Precision','F1','Cohen κ','Marg. Coverage']
    colors  = ['#2196F3','#00BCD4','#4CAF50','#FF9800','#9C27B0']

    x, w = np.arange(len(names)), 0.15
    fig, ax = plt.subplots(figsize=(12, 5))
    for i,(m,lbl,col) in enumerate(zip(metrics,labels,colors)):
        vals = [r[m] for r in results]
        ax.bar(x + (i-2)*w, vals, w, label=lbl, color=col, alpha=0.85)

    ax.axhline(1-ALPHA, ls=':', color='purple', lw=1.5,
               label=f'Conformal target ({1-ALPHA:.0%})')
    ax.axhline(0.90, ls='--', color='gray', lw=0.8, alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels([n.upper() for n in names])
    ax.set_ylim(0, 1.12); ax.set_ylabel('Score')
    ax.set_title('SpectralGAT-Conformal: LOCO-CV Results  (v2)', fontweight='bold')
    ax.legend(ncol=3, fontsize=8.5)

    avg = {m: np.nanmean([r[m] for r in results]) for m in metrics}
    ax.text(0.99, 0.03,
            f"Mean  AUC={avg['auc']:.3f}  AP={avg['ap']:.3f}  "
            f"F1={avg['f1']:.3f}  κ={avg['kappa']:.3f}  Cov={avg['coverage']:.3f}",
            transform=ax.transAxes, ha='right', fontsize=8.5,
            bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.85))

    plt.tight_layout()
    p = OUT / "gat_summary_results.png"
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"→ {p.name}")


# ─── 8. Main ──────────────────────────────────────────────────────────────────

def main():
    print("="*65)
    print("  SpectralGAT-Conformal v2 — Focal Loss + Marginal Conformal")
    print("="*65)

    print("\n[1/4] Loading & preprocessing scenes …")
    scenes = {n: prepare_scene(n, w, h) for n,(w,h) in IMAGES.items()}
    names  = list(scenes.keys())

    all_results = []
    g_imp       = np.zeros(5)
    g_bw        = []

    print("\n[2/4] LOCO-CV …")
    for fold_i, test_name in enumerate(names, 1):
        tr_names = [n for n in names if n != test_name]
        print(f"\n  ── Fold {fold_i}/{len(names)}  "
              f"test={test_name.upper()}  train={[n.upper() for n in tr_names]}")

        tr_scenes = [scenes[n] for n in tr_names]
        te_scene  = scenes[test_name]

        # Scaler
        scaler = StandardScaler().fit(np.vstack([s["feats"] for s in tr_scenes]))

        # Conformal calibration set: 20% random nodes from EACH training scene
        rng = np.random.default_rng(2024)
        cal_probs_list, cal_labs_list = [], []

        print("  Training …")
        model = train_model(tr_scenes, scaler, DEVICE, EPOCHS, seed=fold_i)

        # Collect calibration predictions (post-training, on training scenes)
        print("  Calibrating …")
        for sc in tr_scenes:
            n = len(sc["labels"])
            idx = rng.choice(n, n // 5, replace=False)
            pr, _ = infer(model, sc, scaler, DEVICE)
            cal_probs_list.append(pr[idx])
            cal_labs_list.append(sc["labels"][idx])

        cal_probs = np.vstack(cal_probs_list)
        cal_labs  = np.concatenate(cal_labs_list)
        conformal = SplitConformal(alpha=ALPHA)
        conformal.calibrate(cal_probs, cal_labs)

        # Test inference
        print(f"  Predicting on {test_name.upper()} …")
        te_probs, bw = infer(model, te_scene, scaler, DEVICE)
        g_bw.append(bw)

        y_true         = te_scene["labels"]
        y_prob_anom    = te_probs[:, 1]
        y_pred, unc    = conformal.predict(te_probs)
        cov            = conformal.marginal_coverage(te_probs, y_true)

        auc = roc_auc_score(y_true, y_prob_anom) if len(np.unique(y_true))>1 else np.nan
        ap  = average_precision_score(y_true, y_prob_anom)
        f1  = f1_score(y_true, y_pred, zero_division=0)
        kap = cohen_kappa_score(y_true, y_pred)
        pre = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        oa  = accuracy_score(y_true, y_pred)
        unc_pct = 100 * unc.mean()

        print(f"\n  ► {test_name.upper()}: "
              f"AUC={auc:.4f}  AP={ap:.4f}  F1={f1:.4f}  κ={kap:.4f}")
        print(f"    Prec={pre:.4f}  Rec={rec:.4f}  OA={oa:.4f}  "
              f"Uncertain={unc_pct:.1f}%")
        print(f"    Conformal coverage = {cov:.4f}  "
              f"(target ≥ {1-ALPHA:.2f})  {'✓' if cov >= 1-ALPHA else '✗'}")

        all_results.append({"name": test_name, "auc": auc, "ap": ap,
                            "f1": f1, "kappa": kap, "prec": pre, "rec": rec,
                            "oa": oa, "coverage": cov, "uncertain_pct": unc_pct})

        imp = grad_band_importance(model, te_scene, scaler, DEVICE)
        g_imp += imp
        plot_importance(imp, bw, test_name)
        plot_map(te_scene, te_probs, unc)

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n[3/4] Summary")
    print("="*65)
    hdr = f"  {'Scene':<8} {'AUC':>7} {'AP':>7} {'F1':>7} {'κ':>7} {'Cov':>8} {'Unc%':>6}"
    print(hdr); print("  " + "-"*55)
    for r in all_results:
        ok = '✓' if r['coverage'] >= 1-ALPHA else '✗'
        print(f"  {r['name'].upper():<8} {r['auc']:>7.4f} {r['ap']:>7.4f} "
              f"{r['f1']:>7.4f} {r['kappa']:>7.4f} "
              f"{r['coverage']:>7.4f}{ok} {r['uncertain_pct']:>5.1f}%")
    avg = {k: np.nanmean([r[k] for r in all_results])
           for k in ['auc','ap','f1','kappa','coverage','uncertain_pct']}
    print("  " + "-"*55)
    print(f"  {'Average':<8} {avg['auc']:>7.4f} {avg['ap']:>7.4f} "
          f"{avg['f1']:>7.4f} {avg['kappa']:>7.4f} "
          f"{avg['coverage']:>7.4f}  {avg['uncertain_pct']:>5.1f}%")

    g_imp /= len(names)
    print("\n  Band Importance (Gradient × Input, LOCO-CV mean):")
    for b, v in sorted(zip(BANDS, g_imp), key=lambda x: -x[1]):
        print(f"    {b:<18} {v:.4f} ({100*v:.1f}%)  {'█'*int(v*70)}")

    g_bw_all = np.vstack(g_bw)
    print("\n  Spectral Attention Weights (mean across all folds):")
    for b, m, s in zip(BS, g_bw_all.mean(0), g_bw_all.std(0)):
        print(f"    {b:<6} {m:.4f} ± {s:.4f}  {'█'*int(m*80)}")

    # ── Save ─────────────────────────────────────────────────────────────
    print("\n[4/4] Saving …")
    out = {
        "model": "SpectralGAT-Conformal-v2",
        "config": {"alpha": ALPHA, "hidden": HIDDEN_DIM, "heads": N_HEADS,
                   "epochs": EPOCHS, "focal_gamma": FOCAL_GAMMA,
                   "batch_nodes": BATCH_NODES, "lam_rec": LAM_REC},
        "loco_cv": all_results, "averages": avg,
        "band_importance": {b: float(v) for b,v in zip(BANDS, g_imp)},
        "spectral_attention": {
            "mean": {b: float(m) for b,m in zip(BANDS, g_bw_all.mean(0))},
            "std":  {b: float(s) for b,s in zip(BANDS, g_bw_all.std(0))}
        }
    }
    with open(OUT / "gat_conformal_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"  → {OUT}/gat_conformal_results.json")
    plot_summary(all_results)
    print("\nDONE ✓")
    return out


if __name__ == "__main__":
    main()
