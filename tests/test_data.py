import pandas as pd, numpy as np
from peptak.data import deduplicate_by_smiles

def test_dedup_simple(tmp_path):
    X = [[1],[2]]
    y = np.array([0.0, -7.0])
    smi = ["CCO","CCO"]  # duplicates
    seq = [["A"],["A"]]
    X2, y2, smi2, seq2 = deduplicate_by_smiles(X,y,smi,seq)
    assert len(X2)==1 and smi2[0]=="CCO" and np.isclose(y2[0], (-7.0+0.0)/2)
