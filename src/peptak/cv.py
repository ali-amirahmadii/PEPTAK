from __future__ import annotations
import numpy as np
from sklearn.model_selection import StratifiedKFold, GroupKFold

def make_nested_stratified_folds(y, n_outer=5, n_inner=5, shuffle=True, random_state=42):
    y = np.asarray(y).astype(int)
    X_dummy = np.zeros_like(y)
    outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=shuffle, random_state=random_state)
    nested = []
    for ofold, (trval_idx, test_idx) in enumerate(outer_cv.split(X_dummy, y), start=1):
        inner = []
        y_trval = y[trval_idx]
        X_dummy_inner = np.zeros_like(y_trval)
        inner_cv = StratifiedKFold(
            n_splits=n_inner, shuffle=shuffle,
            random_state=None if random_state is None else (random_state + ofold)
        )
        for tr_rel, va_rel in inner_cv.split(X_dummy_inner, y_trval):
            inner.append({"train": trval_idx[tr_rel], "valid": trval_idx[va_rel]})
        nested.append({"outer": {"train": trval_idx, "test": test_idx}, "inner": inner})
    return nested

def make_nested_groupwise_folds(groups, y=None, n_outer=5, n_inner=5, shuffle=True, random_state=42):
    groups = np.asarray(groups)
    n_unique = np.unique(groups).size
    if n_unique < n_outer or n_unique < n_inner:
        raise ValueError("Not enough unique groups for requested splits.")
    # Try StratifiedGroupKFold if present (sklearn >=1.1)
    try:
        from sklearn.model_selection import StratifiedGroupKFold
        outer_cv = StratifiedGroupKFold(n_splits=n_outer, shuffle=shuffle, random_state=random_state)
        y_outer = np.zeros(len(groups), dtype=int) if y is None else np.asarray(y).astype(int)
    except Exception:
        outer_cv = GroupKFold(n_splits=n_outer)
        y_outer = np.zeros(len(groups), dtype=int)
    X_dummy = np.zeros(len(groups))
    nested = []
    for ofold, (trval_idx, test_idx) in enumerate(outer_cv.split(X_dummy, y_outer, groups=groups), start=1):
        # inner within outer-train
        g_trval = groups[trval_idx]
        X_inner = np.zeros(len(trval_idx))
        try:
            inner_cv = StratifiedGroupKFold(
                n_splits=n_inner, shuffle=shuffle,
                random_state=None if random_state is None else (random_state + ofold)
            )
            y_inner = np.zeros(len(trval_idx), dtype=int) if y is None else np.asarray(y)[trval_idx].astype(int)
            inner_splits = inner_cv.split(X_inner, y_inner, groups=g_trval)
        except Exception:
            inner_cv = GroupKFold(n_splits=n_inner)
            inner_splits = inner_cv.split(X_inner, groups=g_trval)
        inner = [{"train": trval_idx[tr], "valid": trval_idx[va]} for tr, va in inner_splits]
        nested.append({"outer": {"train": trval_idx, "test": test_idx}, "inner": inner})
    return nested
