from __future__ import annotations
import numpy as np
from typing import Callable, Sequence
from tqdm import tqdm
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.DataStructs import TanimotoSimilarity
from joblib import Memory
from joblib import Parallel, delayed
from itertools import combinations


memory = Memory(location=".peptak_cache", verbose=0)

def _normalize_to_unit_diag(K: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    d = np.clip(np.diag(K), eps, None)
    S = 1.0 / np.sqrt(d)
    return (K * S[:, None]) * S[None, :]

@memory.cache
def compute_gram(objects, kernel_fn, normalize=True, desc="Gram", n_jobs=1):
    n = len(objects)
    K = np.zeros((n, n), dtype=float)

    # diagonals
    for i in range(n):
        K[i, i] = kernel_fn(objects[i], objects[i])

    # upper-triangle pairs
    pairs = list(combinations(range(n), 2))

    def _work(i, j):
        return i, j, float(kernel_fn(objects[i], objects[j]))

    for i, j, kij in Parallel(n_jobs=n_jobs, prefer="processes")(delayed(_work)(i, j) for i, j in pairs):
        K[i, j] = kij
        K[j, i] = kij

    if normalize:
        K = _normalize_to_unit_diag(K)
    return K

def tanimoto_kernel_matrix(smiles: Sequence[str], radius: int = 3, useChirality: bool = True) -> np.ndarray:
    fps = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprint(mol, radius, useChirality=useChirality)
        fps.append(fp)
    n = len(fps)
    K = np.eye(n, dtype=float)
    for i in tqdm(range(n-1), desc="Tanimoto"):
        for j in range(i+1, n):
            kij = float(TanimotoSimilarity(fps[i], fps[j]))
            K[i, j] = K[j, i] = kij
    return K  # already has unit diagonal
