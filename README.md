# PeptAK — Peptide Kernels & GP Classification

Model cyclic peptide permeability using **MD-GAK** (Monomer-Decoupled Global Alignment Kernel), **PMD-GAK** (position-aware), and a **Tanimoto** baseline, with nested cross-validation.

This guide shows how to:

- Install the environment (RDKit via conda-forge)
- Build kernel Gram matrices (D1 = MD-GAK, PMD-GAK, D2 = Tanimoto)
- Reproduce results with **label-stratified** and **canonical achiral group-stratified** splits
- Summarize results (mean ± std) for manuscript tables

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Create and Activate the Environment](#create-and-activate-the-environment)
- [Install the Package](#install-the-package)
- [Prepare the Data](#prepare-the-data)
- [Build Kernels (Gram Matrices)](#build-kernels-gram-matrices)
  - [MD-GAK (D1)](#md-gak-d1)
  - [PMD-GAK (optional)](#pmd-gak-optional)
  - [Tanimoto (D2)](#tanimoto-d2)
- [Run Nested Cross-Validation](#run-nested-cross-validation)
  - [Label-Stratified (Table 1)](#label-stratified-table-1)
  - [Canonical Achiral Group-Stratified (Table 2)](#canonical-achiral-group-stratified-table-2)
- [Summarize Results](#summarize-results)
- [Using Precomputed Kernels](#using-precomputed-kernels)
- [Tips & Troubleshooting](#tips--troubleshooting)
- [Windows Notes](#windows-notes)
- [Reproducibility](#reproducibility)
- [Citation](#citation)
- [License](#license)

---

## Prerequisites

- **Conda (recommended):** Mambaforge / Miniconda / Anaconda  
- OS: Linux, macOS, or Windows (PowerShell/CMD)

> RDKit is easiest via **conda-forge**. If you must use pip-only, use Python ≤ 3.11 and install `rdkit-pypi`. This project targets Python 3.12 with RDKit from conda-forge.

---

## Create and Activate the Environment

```bash
# Create env with Python 3.12 and RDKit from conda-forge
conda create -n peptak_env python=3.12 rdkit -c conda-forge -y
conda activate peptak_env

# Verify RDKit
python -c "import rdkit; from rdkit import Chem; print('RDKit OK')"
```

---

## Install the Package

From the repository root (where `pyproject.toml` lives):

```bash
pip install --upgrade pip
pip install -e .
```

> If the `peptak` command isn’t found on your PATH, you can always run commands as `python -m peptak.cli ...`.

---

## Prepare the Data

Place these CSVs at the repository root (or adjust paths in commands below):

- `CycPeptMPDB_Peptide_Assay_PAMPA.csv`  
- `CycPeptMPDB_Monomer_All.csv`

---

## Build Kernels (Gram Matrices)

Create an output folder once:

```bash
mkdir -p out
```

### MD-GAK (D1)

Matches the manuscript recurrence with `gap_decay = 1.0`.

```bash
peptak compute-gram \
  --db CycPeptMPDB_Peptide_Assay_PAMPA.csv \
  --monomers CycPeptMPDB_Monomer_All.csv \
  --kernel mdgak \
  --gap 1.0 \
  --out out/D1.npy
```

Outputs:

- `out/D1.npy` — normalized MD-GAK Gram matrix  
- `out/D1.labels.npy` — continuous PAMPA values (same order)  
- `out/D1.smiles.npy` — whole-peptide SMILES (same order)

### PMD-GAK (optional)

```bash
peptak compute-gram \
  --db CycPeptMPDB_Peptide_Assay_PAMPA.csv \
  --monomers CycPeptMPDB_Monomer_All.csv \
  --kernel pmdgak \
  --gap 1.0 \
  --beta 1.0 \
  --T 5 \
  --out out/PMD1.npy
```

- `--T` controls the positional window (triangular band).  
- `--beta` controls the softness of the local match.

### Tanimoto (D2)

```bash
peptak compute-gram \
  --db CycPeptMPDB_Peptide_Assay_PAMPA.csv \
  --monomers CycPeptMPDB_Monomer_All.csv \
  --kernel tanimoto \
  --out out/D2.npy
```

---

## Run Nested Cross-Validation

Binary label is derived from PAMPA using the manuscript threshold: **positive = (PAMPA < −6)**.  
Both commands display **progress bars** in the terminal and **print JSON** to stdout so you can `tee` to a file.

### Label-Stratified (Table 1)

```bash
peptak nested-gp \
  --gram out/D1.npy \
  --labels out/D1.labels.npy \
  --outer 5 \
  --inner 5 \
  | tee out/results_d1_label_strat.json
```

Notes:

- `--outer` and `--inner` control the number of outer/inner folds (5×5 in the paper).
- Add `--threshold 0.5` to force a decision threshold (default is 0.5).
- Add `--save out/live_label_strat.json` to write an updating JSON during the run (optional).

### Canonical Achiral Group-Stratified (Table 2)

Keeps achiral-canonical **whole-peptide SMILES groups** intact across folds (no leakage) between splits, and balances labels when supported.

**Option A — compute groups on the fly from saved SMILES:**

```bash
peptak canonical-nested-gp \
  --gram out/D1.npy \
  --labels out/D1.labels.npy \
  --smiles out/D1.smiles.npy \
  --outer 5 \
  --inner 5 \
  | tee out/results_d1_group_strat.json
```

**Option B — precompute groups once and reuse:**

```bash
python - << 'PY'
import numpy as np
from peptak.data import group_labels_achiral
smiles = np.load("out/D1.smiles.npy", allow_pickle=True)
groups = group_labels_achiral(list(smiles))
np.save("out/groups_achiral.npy", groups)
print("Saved out/groups_achiral.npy")
PY

peptak canonical-nested-gp \
  --gram out/D1.npy \
  --labels out/D1.labels.npy \
  --groups out/groups_achiral.npy \
  --outer 5 \
  --inner 5 \
  | tee out/results_d1_group_strat.json
```

> If you see `Not enough unique groups for requested splits`, reduce `--outer/--inner` to ≤ the number of unique groups.

---

## Summarize Results

Compute mean ± std across outer folds for any saved JSON (replace the path as needed):

```bash
python - << 'PY'
import json, numpy as np, pathlib
p = pathlib.Path("out/results_d1_label_strat.json")  # or out/results_d1_group_strat.json
res = json.loads(p.read_text())

def ms(key):
    v = np.array([r[key] for r in res], float)
    return f"{v.mean():.3f} ± {v.std():.3f}"

print("Summary:")
for k in ["accuracy","f1","auroc","brier","ece"]:
    print(f"{k:8s}: {ms(k)}")
PY
```

Paste these lines directly into your manuscript tables.

---

## Using Precomputed Kernels

If you already have `D1.npy`, `D1.labels.npy`, and `D1.smiles.npy`, place them in `out/` and jump straight to [Run Nested Cross-Validation](#run-nested-cross-validation).

+ D1_pos.npy for PMD-GAK kernel

---

## Tips & Troubleshooting

**Progress bars & JSON together**

- Progress bars (`tqdm`) write to **stderr**; JSON prints to **stdout**.  
- To see progress bars and save JSON:
  ```bash
  peptak nested-gp ... | tee out/results.json
  ```
- To log bars to a file and save JSON to another:
  ```bash
  peptak nested-gp ... 2> out/progress.log | tee out/results.json
  ```

**CLI not found**

- Use module form (always works in the active env):
  ```bash
  python -m peptak.cli nested-gp --gram ... --labels ...
  ```
- Or reinstall after activating the env:
  ```bash
  pip install -e .
  ```

**RDKit not found**

- Ensure you’re in the env where RDKit is installed:
  ```bash
  conda activate peptak_env
  python -c "from rdkit import Chem; print('OK')"
  ```
- If missing:
  ```bash
  conda install -c conda-forge rdkit -y
  ```

---



## Reproducibility

- Heavy kernel work is cached in `.peptak_cache/` (joblib) to speed up re-runs.  
- Splits use fixed seeds (`random_state=42`) where supported.  
- Saved arrays keep consistent ordering across `*.npy` files (`D1.npy`, `D1.labels.npy`, `D1.smiles.npy`).

---

## Citation

If you use this code or kernels in your work, please cite the accompanying manuscript (update with your details):

```bibtex
@article{YourPaper2025,
  title   = {Gaussian process with Molecular Fingerprint: Kernel for peptide membrane permeability},
  author  = {Ali and Alessandro},
  year    = {2025},
  note    = {Preprint}
}
```

---

