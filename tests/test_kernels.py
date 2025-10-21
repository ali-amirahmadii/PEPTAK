import numpy as np
from rdkit.Chem import AllChem, MolFromSmiles
from peptak.kernels import md_gak, pmd_gak

def fp(smi):
    return AllChem.GetMorganFingerprint(MolFromSmiles(smi), 3, useChirality=True)

def test_mdgak_symmetry():
    a = [fp("CCO"), fp("CCN")]
    b = [fp("CCO"), fp("CCN")]
    k = md_gak(a,b)
    assert k == md_gak(b,a)
    assert k >= md_gak(a,[fp("c1ccccc1")])  # self-similarity larger than dissimilar

def test_pmdgak_band():
    a = [fp("CCO")] * 4
    b = [fp("CCO")] * 4
    k_full = pmd_gak(a,b,beta=1.0,gap_decay=0.5,T=None)
    k_band = pmd_gak(a,b,beta=1.0,gap_decay=0.5,T=0)
    assert k_band <= k_full
