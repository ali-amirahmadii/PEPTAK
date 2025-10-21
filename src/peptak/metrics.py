from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, brier_score_loss, confusion_matrix,
    precision_recall_curve, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve

def compute_ece(y_true, y_prob, n_bins=30, strategy="uniform"):
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(y_prob, qs)
        edges[0], edges[-1] = 0.0, 1.0
    else:
        raise ValueError
    bins = np.digitize(y_prob, edges[1:-1], right=True)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if not np.any(mask):
            continue
        conf = y_prob[mask].mean()
        acc  = y_true[mask].mean()
        ece += mask.mean() * abs(acc - conf)
    return float(ece)

def evaluate_classification_with_calibration(
    y_true, y_prob, threshold=0.5, pampa=None, model_name="Model", n_bins=30, ece_strategy="uniform"
):
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    y_pred = (y_prob >= float(threshold)).astype(int)

    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = np.nan
    brier = brier_score_loss(y_true, y_prob)
    ece   = compute_ece(y_true, y_prob, n_bins=n_bins, strategy=ece_strategy)

    print(f"{model_name} @ t={threshold:.2f} | ACC {acc:.3f} F1 {f1:.3f} AUC {auc:.3f} | Brier {brier:.4f} ECE {ece:.4f}")
    print("Confusion:\n", confusion_matrix(y_true, y_pred))

    # 1) prob dist
    plt.figure(figsize=(6,4))
    sns.histplot(y_prob, bins=20, stat="probability")
    try: sns.kdeplot(y_prob, lw=1)
    except: pass
    plt.xlabel("Predicted P(positive)")
    plt.title(f"{model_name} — prob. distribution")
    plt.tight_layout(); plt.show()

    # 2) reliability
    pt, pp = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy=ece_strategy)
    plt.figure(figsize=(5,5))
    plt.plot([0,1], [0,1], "--", lw=1, label="perfect")
    plt.plot(pp, pt, marker="o", lw=1.5, label=model_name)
    plt.xlabel("Mean predicted prob"); plt.ylabel("Empirical positive rate")
    plt.title(f"Reliability ({ece_strategy}, {n_bins}) — ECE={ece:.4f}")
    plt.legend(); plt.tight_layout(); plt.show()

    # 3) optional scatter vs PAMPA
    if pampa is not None:
        p = np.asarray(pampa, dtype=float).ravel()
        plt.figure(figsize=(6,4))
        plt.scatter(p, y_prob, s=16, alpha=0.7)
        plt.xlabel("PAMPA"); plt.ylabel("Pred prob")
        if len(y_prob)>1 and np.std(y_prob)>0 and np.std(p)>0:
            r = np.corrcoef(y_prob, p)[0,1]; plt.title(f"{model_name}: corr={r:.3f}")
        plt.tight_layout(); plt.show()

    return {"accuracy":acc, "f1":f1, "auroc":auc, "brier":brier, "ece":ece}

def tune_threshold_on_validation(y_valid, p_valid, objective="f1", beta=1.0):
    yv = np.asarray(y_valid, dtype=int).ravel()
    pv = np.asarray(p_valid, dtype=float).ravel()
    if np.allclose(pv, pv[0]):  # degenerate
        return 0.5, 0.0
    if objective in ("f1", "fbeta"):
        prec, rec, thr = precision_recall_curve(yv, pv)
        prec, rec = prec[:-1], rec[:-1]
        if objective == "f1":
            score = 2 * prec * rec / (prec + rec + 1e-12)
        else:
            b2 = beta**2
            score = (1 + b2) * prec * rec / (b2 * prec + rec + 1e-12)
        idx = int(np.nanargmax(score))
        return float(thr[idx]), float(score[idx])
    if objective == "youden":
        fpr, tpr, thr = roc_curve(yv, pv)
        j = tpr - fpr
        idx = int(np.argmax(j))
        return float(thr[idx]), float(j[idx])
    # fallback
    best_t, best_s = 0.5, -np.inf
    for t in np.linspace(0.01, 0.99, 99):
        s = f1_score(yv, (pv >= t).astype(int))
        if s > best_s: best_t, best_s = float(t), float(s)
    return best_t, best_s
