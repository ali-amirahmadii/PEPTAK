# src/peptak/gp.py
from __future__ import annotations
import numpy as np
from sklearn.gaussian_process.kernels import Kernel, ConstantKernel, WhiteKernel
from sklearn.gaussian_process import GaussianProcessClassifier

class GlobalPrecomputedKernel(Kernel):
    """
    Gives sklearn's GPC a view into a *global* Gram matrix, by indexing rows/cols with sample indices.
    Supports optional cosine normalization on the fly; caches the diagonal.
    """
    def __init__(self, K_global: np.ndarray, normalize: bool = True, eps: float = 1e-15):
        self.K_global = np.asarray(K_global, dtype=float)
        self.normalize = bool(normalize)
        # IMPORTANT: expose 'eps' as a public attribute so sklearn.clone() can find it
        self.eps = float(eps)
        # cache diagonal once
        self._diag = np.maximum(np.diag(self.K_global).astype(float), self.eps)

    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            Y = X
        Xi = X.ravel().astype(int)
        Yi = Y.ravel().astype(int)
        K = self.K_global[np.ix_(Xi, Yi)]
        if self.normalize:
            di = self._diag[Xi]
            dj = self._diag[Yi]
            K = K / (np.sqrt(np.outer(di, dj)) + self.eps)
        if eval_gradient:
            return K, np.empty((K.shape[0], K.shape[1], 0))
        return K

    def diag(self, X):
        Xi = X.ravel().astype(int)
        return np.ones_like(Xi) if self.normalize else self._diag[Xi]

    def is_stationary(self):
        return False

def make_gpc(K_global: np.ndarray, amp: float, noise: float, normalize: bool = True, seed: int = 42):
    base = GlobalPrecomputedKernel(K_global, normalize=normalize)
    kernel = ConstantKernel(constant_value=amp, constant_value_bounds="fixed") * base \
             + WhiteKernel(noise_level=noise, noise_level_bounds="fixed")
    return GaussianProcessClassifier(kernel=kernel, optimizer=None, max_iter_predict=200, random_state=seed)
