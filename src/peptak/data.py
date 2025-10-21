from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Dict
import ast
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

def _morgan_fp_from_smiles(smi: str, radius: int = 3, useChirality: bool = True):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprint(mol, radius, useChirality=useChirality)

def load_cycpeptmpdb(
    peptide_assay_csv: str,
    monomer_csv: str,
    assay_col: str = "PAMPA",
) -> Tuple[List[list], np.ndarray, List[str], List[List[str]]]:
    """
    Returns:
      X: list of list of monomer fingerprints per peptide (sequence of RDKit FPs)
      y: continuous assay values (float array)
      smiles_list: whole-peptide SMILES strings
      sequences: list of monomer SMILES sequences (canonical, achiral)
    Fixes:
      - uses ast.literal_eval instead of eval() for safety
      - canonicalizes monomer SMILES
    """
    db = pd.read_csv(peptide_assay_csv)
    mono = pd.read_csv(monomer_csv, index_col="Symbol")

    X, y, smiles_list, sequences = [], [], [], []
    subset = db[["SMILES", "Sequence", assay_col]].values
    for smiles, seq_field, label in subset:
        smiles_list.append(smiles)
        # parse sequence safely
        if isinstance(seq_field, str):
            sequence_tokens = ast.literal_eval(seq_field)
        elif isinstance(seq_field, (list, tuple, np.ndarray, pd.Series)):
            sequence_tokens = list(seq_field)
        else:
            raise ValueError(f"Unsupported sequence field type: {type(seq_field)}")

        fps, seq_smis = [], []
        for token in sequence_tokens:
            msmi = mono.loc[token]["replaced_SMILES"]
            fp = _morgan_fp_from_smiles(msmi)
            if fp is None:
                continue
            fps.append(fp)
            mol = Chem.MolFromSmiles(msmi)
            Chem.RemoveStereochemistry(mol)  # achiral canonical for the token string
            seq_smis.append(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False))

        X.append(fps)
        y.append(label)
        sequences.append(seq_smis)

    return X, np.asarray(y, dtype=float), smiles_list, sequences

def deduplicate_by_smiles(
    X: List[list], y: np.ndarray, smiles: List[str], sequences: List[List[str]]
) -> Tuple[List[list], np.ndarray, List[str], List[List[str]]]:
    """Group by whole-peptide SMILES (using Tanimoto=1 as exact match)."""
    from rdkit.DataStructs import TanimotoSimilarity
    dedup: Dict[str, Dict[str, list]] = {}
    seen = []
    fps = []
    for i, smi in enumerate(smiles):
        fp = _morgan_fp_from_smiles(smi)
        found = False
        for k, fp2 in fps:
            if TanimotoSimilarity(fp, fp2) >= 1.0:
                dedup[k]["X"].append(X[i])
                dedup[k]["y"].append(y[i])
                dedup[k]["seq"].append(sequences[i])
                found = True
                break
        if not found:
            dedup[smi] = {"X": [X[i]], "y": [y[i]], "seq": [sequences[i]]}
            seen.append(smi)
            fps.append((smi, fp))

    new_X, new_y, new_smiles, new_seq = [], [], [], []
    for k, v in dedup.items():
        new_X.append(v["X"][0])
        new_y.append(float(np.mean(v["y"])))
        new_seq.append(v["seq"][0])
        new_smiles.append(k)
    return new_X, np.asarray(new_y, dtype=float), new_smiles, new_seq

def group_labels_achiral(smiles: List[str]) -> np.ndarray:
    """Achiral canonical SMILES for grouping (no stereochem)."""
    labels = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            labels.append("__INVALID__")
            continue
        Chem.RemoveStereochemistry(mol)
        labels.append(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False))
    return np.asarray(labels)
