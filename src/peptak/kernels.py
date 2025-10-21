from __future__ import annotations
import numpy as np
from rdkit.DataStructs import TanimotoSimilarity

def _safe_tanimoto(x, y) -> float:
    try:
        return float(TanimotoSimilarity(x, y))
    except Exception:
        return 0.0

def md_gak(seq1_fp, seq2_fp, gap_decay: float = 0.5) -> float:
    """
    Monomer-Decoupled Global Alignment Kernel (MD-GAK) a.k.a. your D1.
    Recurrence:
      M[i,j] = k0(i,j)*M[i-1,j-1] + λ*M[i-1,j] + λ*M[i,j-1]
    where k0 is Tanimoto between monomer FPs. (Unitless, in [0,1])
    """
    n, m = len(seq1_fp), len(seq2_fp)
    M = np.zeros((n+1, m+1), dtype=float)
    M[0, 0] = 1.0
    for i in range(1, n+1):
        fi = seq1_fp[i-1]
        for j in range(1, m+1):
            k0 = _safe_tanimoto(fi, seq2_fp[j-1])
            M[i, j] = k0 * M[i-1, j-1] + gap_decay * (M[i-1, j] + M[i, j-1])
    return float(M[n, m])

def pmd_gak(seq1_fp, seq2_fp, beta: float = 1.0, gap_decay: float = 0.5, T: int | None = None) -> float:
    """
    Position-aware MD-GAK (PMD-GAK), triangular Toeplitz window of width T.
    Soft match: loc = exp(-beta * (1 - k0)) with k0 in [0,1]; zero-out if |i-j|>T.
    See manuscript for triangular GA discussion. :contentReference[oaicite:1]{index=1}
    """
    n, m = len(seq1_fp), len(seq2_fp)
    M = np.zeros((n+1, m+1), dtype=float)
    M[0, 0] = 1.0
    for i in range(1, n+1):
        fi = seq1_fp[i-1]
        for j in range(1, m+1):
            if T is not None and abs(i - j) > T:
                # outside band -> no contribution (speeds up and encodes positional prior)
                M[i, j] = gap_decay * (M[i-1, j] + M[i, j-1])
                continue
            k0 = _safe_tanimoto(fi, seq2_fp[j-1])
            loc = np.exp(-beta * (1.0 - k0))
            if T is not None:
                w = max(0.0, 1.0 - abs(i - j) / T)
                if w == 0.0:
                    M[i, j] = gap_decay * (M[i-1, j] + M[i, j-1])
                    continue
                loc *= w
            M[i, j] = loc * M[i-1, j-1] + gap_decay * (M[i-1, j] + M[i, j-1])
    return float(M[n, m])
