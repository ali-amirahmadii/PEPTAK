# src/peptak/cli.py
from __future__ import annotations

import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

from peptak.data import (
    load_cycpeptmpdb,
    deduplicate_by_smiles,
    group_labels_achiral,
)
from peptak.kernels import md_gak, pmd_gak
from peptak.gram import compute_gram, tanimoto_kernel_matrix
from peptak.cv import (
    make_nested_stratified_folds,
    make_nested_groupwise_folds,
)
from peptak.gp import make_gpc
from peptak.metrics import (
    evaluate_classification_with_calibration,
)


# ---------------------------
# shared core for nested runs
# ---------------------------
def _run_nested_gp_core(
    K: np.ndarray,
    y_cont: np.ndarray,
    folds: list,
    *,
    threshold: float | None = None,
    save_path: str | None = None,
    normalize: bool = True,
    title_prefix: str = "GPC(D1)",
):
    """
    Run nested CV over provided folds.
    - Model: GaussianProcessClassifier with precomputed global kernel (K)
    - Selection: grid over amp/noise; inner F1 @ 0.5
    - Evaluation: metrics + calibration plot per outer fold
    """
    from sklearn.metrics import f1_score

    y_bin = (y_cont < -6).astype(int)

    # modest grids (adjust as needed)
    amps = np.logspace(-3, 3, 3)
    noises = np.logspace(-8, 1, 3)

    results = []

    outer_bar = tqdm(
        list(enumerate(folds, start=1)),
        total=len(folds),
        desc="Outer folds",
        position=0,
    )

    for k_outer, fold in outer_bar:
        tr_idx = fold["outer"]["train"]
        te_idx = fold["outer"]["test"]

        best_mean = -1.0
        best_amp, best_noise = None, None
        oof_prob = None
        oof_idx = None

        grid = [(a, n) for a in amps for n in noises]
        grid_bar = tqdm(grid, desc=f"Grid o{k_outer}", position=1, leave=False)

        for a, n in grid_bar:
            inner_f1s = []
            tmp_prob, tmp_idx = [], []

            for sub in fold["inner"]:
                in_tr = sub["train"]
                in_va = sub["valid"]

                clf = make_gpc(K, amp=a, noise=n, normalize=normalize)
                clf.fit(in_tr[:, None], y_bin[in_tr])

                p = clf.predict_proba(in_va[:, None])[:, 1]
                inner_f1s.append(f1_score(y_bin[in_va], (p >= 0.5).astype(int)))
                tmp_prob.append(p)
                tmp_idx.append(in_va)

            meanF1 = float(np.mean(inner_f1s)) if inner_f1s else -1.0
            grid_bar.set_postfix(meanF1=f"{meanF1:.3f}")

            if meanF1 > best_mean:
                best_mean = meanF1
                best_amp, best_noise = float(a), float(n)
                oof_prob = np.concatenate(tmp_prob) if tmp_prob else None
                oof_idx = np.concatenate(tmp_idx) if tmp_idx else None

        # Build OOF prob vector aligned with tr_idx (if we collected OOF)
        if oof_prob is not None and oof_idx is not None:
            pos = {idx: i for i, idx in enumerate(tr_idx)}
            p_valid = np.empty(tr_idx.size, float)
            for idx, p in zip(oof_idx, oof_prob):
                p_valid[pos[int(idx)]] = p
        else:
            p_valid = None

        thr = 0.5 if threshold is None else float(threshold)

        # Final fit on full outer-train with best amp/noise
        clf = make_gpc(K, amp=best_amp, noise=best_noise, normalize=normalize)
        clf.fit(tr_idx[:, None], y_bin[tr_idx])

        p_test = clf.predict_proba(te_idx[:, None])[:, 1]

        metrics = evaluate_classification_with_calibration(
            y_true=y_bin[te_idx],
            y_prob=p_test,
            threshold=thr,
            pampa=y_cont[te_idx],
            model_name=f"{title_prefix} o{k_outer} "
                       f"(amp={best_amp:.2g}, noise={best_noise:.2g}, t={thr:.2f})",
        )
        metrics.update(
            {
                "fold": k_outer,
                "amp": best_amp,
                "noise": best_noise,
                "threshold": thr,
            }
        )
        results.append(metrics)

        # live, short per-fold summary
        print(
            f"[outer {k_outer}] F1={metrics['f1']:.3f} "
            f"ACC={metrics['accuracy']:.3f} AUC={metrics['auroc']:.3f} "
            f"Brier={metrics['brier']:.4f} ECE={metrics['ece']:.4f}",
            flush=True,
        )

        # optional running save
        if save_path:
            Path(save_path).write_text(json.dumps(results, indent=2))

    # final JSON to stdout (so user can pipe/tee)
    print(json.dumps(results, indent=2))


# -------------
# CLI entrypoint
# -------------
def main():
    ap = argparse.ArgumentParser(prog="peptak")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # --- compute-gram ---
    g = sub.add_parser("compute-gram", help="Compute Gram matrix (MD-GAK | PMD-GAK | TAN)")
    g.add_argument("--assay", default="PAMPA")
    g.add_argument("--db", required=True)
    g.add_argument("--monomers", required=True)
    g.add_argument("--kernel", choices=["mdgak", "pmdgak", "tanimoto"], required=True)
    g.add_argument("--gap", type=float, default=0.5)
    g.add_argument("--beta", type=float, default=1.0)
    g.add_argument("--T", type=int, default=None)
    g.add_argument("--out", required=True)

    # --- nested-gp (label-stratified) ---
    r = sub.add_parser("nested-gp", help="Nested CV with label stratification (Table 1)")
    r.add_argument("--gram", required=True)
    r.add_argument("--labels", required=True)  # .npy continuous labels (same order as gram)
    r.add_argument("--outer", type=int, default=5)
    r.add_argument("--inner", type=int, default=5)
    r.add_argument("--threshold", type=float, default=None)
    r.add_argument("--save", default=None, help="Path to save JSON results (optional)")

    # --- canonical-nested-gp (achiral canonical group-stratified) ---
    c = sub.add_parser(
        "canonical-nested-gp",
        help="Nested CV with canonical achiral group stratification (Table 2)",
    )
    c.add_argument("--gram", required=True)
    c.add_argument("--labels", required=True)
    # Either provide groups directly, or provide smiles to compute them
    c.add_argument("--groups", default=None, help=".npy of precomputed groups")
    c.add_argument("--smiles", default=None, help=".npy SMILES (compute groups on-the-fly if --groups not given)")
    c.add_argument("--outer", type=int, default=5)
    c.add_argument("--inner", type=int, default=5)
    c.add_argument("--threshold", type=float, default=None)
    c.add_argument("--save", default=None, help="Path to save JSON results (optional)")

    args = ap.parse_args()

    # --- compute-gram command ---
    if args.cmd == "compute-gram":
        X, y, smi, seq = load_cycpeptmpdb(args.db, args.monomers, assay_col=args.assay)
        X, y, smi, seq = deduplicate_by_smiles(X, y, smi, seq)

        if args.kernel == "tanimoto":
            K = tanimoto_kernel_matrix(smi)
        else:
            seqs = X
            if args.kernel == "mdgak":
                def k(a, b): return md_gak(a, b, gap_decay=args.gap)
            else:
                def k(a, b): return pmd_gak(a, b, beta=args.beta, gap_decay=args.gap, T=args.T)
            K = compute_gram(seqs, k, normalize=True, desc=args.kernel.upper())

        np.save(args.out, K)
        np.save(Path(args.out).with_suffix(".labels.npy"), y)
        np.save(Path(args.out).with_suffix(".smiles.npy"), np.asarray(smi))
        print(f"Saved: {args.out}")
        return

    # --- nested-gp (label-stratified) ---
    if args.cmd == "nested-gp":
        K = np.load(args.gram)
        y_cont = np.load(args.labels)
        folds = make_nested_stratified_folds(
            (y_cont < -6).astype(int),
            n_outer=args.outer,
            n_inner=args.inner,
            shuffle=True,
            random_state=42,
        )
        _run_nested_gp_core(
            K,
            y_cont,
            folds,
            threshold=args.threshold,
            save_path=args.save,
            normalize=True,
            title_prefix="GPC(D1)",
        )
        return

    # --- canonical-nested-gp (achiral canonical group-stratified) ---
    if args.cmd == "canonical-nested-gp":
        K = np.load(args.gram)
        y_cont = np.load(args.labels)

        if args.groups:
            groups = np.load(args.groups, allow_pickle=True)
        elif args.smiles:
            smiles = np.load(args.smiles, allow_pickle=True)
            groups = group_labels_achiral(list(smiles))
        else:
            raise SystemExit("Please pass either --groups or --smiles to build canonical groups.")

        folds = make_nested_groupwise_folds(
            groups=groups,
            y=(y_cont < -6).astype(int),
            n_outer=args.outer,
            n_inner=args.inner,
            shuffle=True,
            random_state=42,
        )

        _run_nested_gp_core(
            K,
            y_cont,
            folds,
            threshold=args.threshold,
            save_path=args.save,
            normalize=True,
            title_prefix="GPC(D1)-GROUP",
        )
        return
