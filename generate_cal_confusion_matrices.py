#!/usr/bin/env python3
"""Generate final-budget confusion matrices for the cAL experiment."""

import json
import sys
from pathlib import Path

# ensure repo root is on the path regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

import al_cal_experiment as cal


METHODS = {
    "Random": ("Random", 1),
    "Standard AL": ("Standard AL", 1),
    "Unc+KernelKMeans": ("Unc+KernelKMeans", 1),
    "cAL r+=4": ("cAL", 4),
}


def final_prediction(method_key, r_plus_train, X_pool, y_pool, X_test, seed, budget):
    rng = np.random.default_rng(seed)
    anom_idx = np.where(y_pool == 1)[0]
    norm_idx = np.where(y_pool == 0)[0]
    k = cal.N_INIT // 2
    labeled = set(
        rng.choice(anom_idx, min(k, len(anom_idx)), replace=False).tolist()
        + rng.choice(norm_idx, min(k, len(norm_idx)), replace=False).tolist()
    )

    queried = 0
    while queried < budget:
        labeled_arr = np.array(sorted(labeled))
        unlabeled_arr = np.array([i for i in range(len(X_pool)) if i not in labeled])
        X_lab = X_pool[labeled_arr]
        y_lab = y_pool[labeled_arr]
        X_unl = X_pool[unlabeled_arr]

        np.random.seed(int(seed) + queried)
        q = min(cal.BATCH_Q, len(unlabeled_arr), budget - queried)
        if q <= 0:
            break

        if len(np.unique(y_lab)) < 2 or method_key == "Random":
            local_q = cal.query_random(None, X_unl, q)
        else:
            clf_q = cal.train_csvm(X_lab, y_lab, r_plus=r_plus_train)
            if method_key == "Standard AL":
                local_q = cal.query_standard_al(clf_q, X_unl, q)
            elif method_key == "Unc+KernelKMeans":
                local_q = cal.query_unc_kernelkmeans(clf_q, X_unl, q)
            else:
                local_q = cal.query_cal(clf_q, X_unl, q)

        for li in local_q:
            labeled.add(int(unlabeled_arr[li]))
        queried += q

    labeled_arr = np.array(sorted(labeled))
    clf = cal.train_csvm(X_pool[labeled_arr], y_pool[labeled_arr], r_plus=r_plus_train)
    return (clf.decision_function(X_test) >= 0).astype(int)


def plot_matrix(cm, title, out_path):
    fig, ax = plt.subplots(figsize=(4.2, 3.7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks([0, 1], labels=["Pred Normal", "Pred Anomaly"])
    ax.set_yticks([0, 1], labels=["True Normal", "True Anomaly"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:.1f}", ha="center", va="center", fontsize=12)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    scenes_X, scenes_y = {}, {}
    for name, (w, h) in cal.IMAGES.items():
        img, gt, seg = cal.load_scene(name, w, h)
        X, y = cal.extract_features(img, gt, seg)
        scenes_X[name] = X
        scenes_y[name] = y

    scaler = StandardScaler().fit(np.vstack(list(scenes_X.values())))
    out_dir = cal.OUT
    summary = {}

    for scene in cal.SCENES:
        X = scaler.transform(scenes_X[scene]).astype(np.float32)
        y = scenes_y[scene]
        budget = min(cal.MAX_BUDGET, int(cal.MAX_PCT * len(X)))

        np.random.seed(cal.SEED_BASE)
        ai = np.where(y == 1)[0]
        ni = np.where(y == 0)[0]
        np.random.shuffle(ai)
        np.random.shuffle(ni)
        ca, cn = int(0.7 * len(ai)), int(0.7 * len(ni))
        pool_idx = np.concatenate([ai[:ca], ni[:cn]])
        test_idx = np.concatenate([ai[ca:], ni[cn:]])
        X_pool, y_pool = X[pool_idx], y[pool_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        summary[scene] = {}
        for label, (method_key, r_train) in METHODS.items():
            cms = []
            for run_id in range(cal.N_RUNS):
                seed = cal.SEED_BASE + run_id * 100 + hash(scene) % 100
                y_pred = final_prediction(
                    method_key, r_train, X_pool, y_pool, X_test, seed, budget
                )
                cms.append(confusion_matrix(y_test, y_pred, labels=[0, 1]))
            mean_cm = np.mean(cms, axis=0)
            tn, fp, fn, tp = mean_cm.ravel()
            summary[scene][label] = {
                "cm_mean": mean_cm.tolist(),
                "tn": float(tn),
                "fp": float(fp),
                "fn": float(fn),
                "tp": float(tp),
                "fpr": float(fp / max(1.0, fp + tn)),
                "fnr": float(fn / max(1.0, fn + tp)),
                "recall": float(tp / max(1.0, fn + tp)),
                "precision": float(tp / max(1.0, fp + tp)),
            }
            plot_matrix(
                mean_cm,
                f"{scene.upper()} - {label}",
                out_dir / f"confusion_{scene}_{label.lower().replace(' ', '_').replace('+', 'p').replace('=', '')}.png",
            )

    aggregate = {}
    for label in METHODS:
        cms = np.array([summary[scene][label]["cm_mean"] for scene in cal.SCENES])
        mean_cm = np.sum(cms, axis=0)
        tn, fp, fn, tp = mean_cm.ravel()
        aggregate[label] = {
            "cm_sum_mean_runs": mean_cm.tolist(),
            "tn": float(tn),
            "fp": float(fp),
            "fn": float(fn),
            "tp": float(tp),
            "fpr": float(fp / max(1.0, fp + tn)),
            "fnr": float(fn / max(1.0, fn + tp)),
            "recall": float(tp / max(1.0, fn + tp)),
            "precision": float(tp / max(1.0, fp + tp)),
        }
        plot_matrix(mean_cm, f"Aggregate - {label}", out_dir / f"confusion_aggregate_{label.lower().replace(' ', '_').replace('+', 'p').replace('=', '')}.png")

    payload = {"per_scene": summary, "aggregate": aggregate}
    with open(out_dir / "cal_confusion_matrices.json", "w") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
