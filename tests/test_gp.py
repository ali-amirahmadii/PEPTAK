import numpy as np
from peptak.gp import make_gpc

def test_gpc_runs():
    rng = np.random.default_rng(0)
    K = rng.random((10,10)); K=(K+K.T)/2; np.fill_diagonal(K,1.0)
    y = rng.integers(0,2, size=10)
    clf = make_gpc(K, amp=1.0, noise=1e-4)
    clf.fit(np.arange(10)[:,None], y)
    p = clf.predict_proba(np.arange(10)[:,None])[:,1]
    assert p.shape==(10,)
