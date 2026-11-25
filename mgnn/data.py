"""Tox21 loading, featurisation and honest handling of its two awkward facts.

Provenance
----------
``data/tox21.csv.gz`` is the MoleculeNet distribution of the Tox21 Data
Challenge 2014 set, downloaded from DeepChem's public bucket
``https://deepchemdata.s3.us-west-1.amazonaws.com/datasets/tox21.csv.gz``
(7831 compounds, 12 assay columns + ``mol_id`` + ``smiles``).  It is vendored
in this repository so every notebook runs offline on a laptop; see
``data/README.md``.

The two awkward facts
---------------------
1. **Labels are missing, not zero.**  Every compound was not run in every
   assay.  A blank cell means "not measured".  Filling blanks with 0 would
   invent thousands of negatives and quietly inflate your metrics, so this
   module carries a boolean mask alongside ``y`` and the loss in
   :mod:`mgnn.train` only scores the entries the mask allows.
2. **The classes are very unbalanced.**  Actives are a few percent of each
   assay.  A model that answers "non-toxic" for everything scores >90%
   accuracy, which is why this repository never reports accuracy and uses
   ROC-AUC plus PR-AUC (the second one is the honest metric under imbalance)
   and an optional positive-class weight in the loss.
"""

from __future__ import annotations

import gzip
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from mgnn.features import MolGraph, mol_to_graph

RDLogger.DisableLog("rdApp.*")

TOX21_TASKS: list[str] = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
    "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]

# The two assays the notebooks use as worked examples.
WORKED_TASKS: list[str] = ["NR-AR", "SR-MMP"]

TOX21_SOURCE_URL = (
    "https://deepchemdata.s3.us-west-1.amazonaws.com/datasets/tox21.csv.gz"
)

__all__ = [
    "TOX21_TASKS", "WORKED_TASKS", "TOX21_SOURCE_URL",
    "repo_root", "load_tox21_frame", "task_summary", "featurise",
    "load_tox21_graphs", "class_balance",
]


def repo_root() -> Path:
    """Directory containing ``mgnn/`` -- works from a notebook or a test."""
    return Path(__file__).resolve().parent.parent


def default_data_path() -> Path:
    return repo_root() / "data" / "tox21.csv.gz"


def load_tox21_frame(path: str | os.PathLike | None = None) -> pd.DataFrame:
    """Read the vendored Tox21 CSV into a DataFrame, blanks kept as ``NaN``."""
    path = Path(path) if path is not None else default_data_path()
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. It ships with the repository; re-download with\n"
            f"    curl -L -o {path} {TOX21_SOURCE_URL}"
        )
    with gzip.open(path, "rt") as fh:
        df = pd.read_csv(fh)
    missing = [c for c in TOX21_TASKS + ["smiles"] if c not in df.columns]
    if missing:
        raise ValueError(f"unexpected Tox21 file, missing columns: {missing}")
    return df


def task_summary(df: pd.DataFrame, tasks: list[str] | None = None) -> pd.DataFrame:
    """Per-assay counts: measured, active, inactive, % active, % missing.

    Printing this table before training is the cheapest way to stay honest
    about what the numbers later in the notebook can possibly mean.
    """
    tasks = tasks or TOX21_TASKS
    rows = []
    for t in tasks:
        col = df[t]
        measured = int(col.notna().sum())
        actives = int((col == 1).sum())
        rows.append({
            "task": t,
            "measured": measured,
            "missing": int(col.isna().sum()),
            "actives": actives,
            "inactives": measured - actives,
            "pct_active": round(100.0 * actives / measured, 2) if measured else float("nan"),
            "pct_missing": round(100.0 * col.isna().sum() / len(df), 2),
        })
    return pd.DataFrame(rows)


def class_balance(y: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Active/inactive counts and the ``pos_weight`` they imply."""
    labelled = y[mask]
    pos = float((labelled == 1).sum())
    neg = float((labelled == 0).sum())
    return {
        "n_labelled": pos + neg,
        "actives": pos,
        "inactives": neg,
        "pct_active": 100.0 * pos / (pos + neg) if pos + neg else float("nan"),
        "pos_weight": neg / pos if pos else float("nan"),
    }


@dataclass
class Tox21Dataset:
    """Featurised Tox21: a list of :class:`~mgnn.features.MolGraph` plus labels."""

    graphs: list[MolGraph]
    smiles: list[str]
    y: np.ndarray        # [n_mols, n_tasks], NaN where unmeasured
    y_mask: np.ndarray   # [n_mols, n_tasks] bool
    tasks: list[str]
    n_failed_parse: int = 0

    def __len__(self) -> int:
        return len(self.graphs)

    def subset(self, idx: list[int]) -> Tox21Dataset:
        return Tox21Dataset(
            graphs=[self.graphs[i] for i in idx],
            smiles=[self.smiles[i] for i in idx],
            y=self.y[idx],
            y_mask=self.y_mask[idx],
            tasks=self.tasks,
        )

    def task_column(self, task: str) -> tuple[np.ndarray, np.ndarray]:
        """``(y, mask)`` for a single assay."""
        j = self.tasks.index(task)
        return self.y[:, j], self.y_mask[:, j]


def featurise(df: pd.DataFrame, tasks: list[str],
              limit: int | None = None, verbose: bool = False) -> Tox21Dataset:
    """SMILES -> graphs.  Molecules RDKit cannot parse are dropped and counted."""
    smiles_col = df["smiles"].tolist()
    label_block = df[tasks].to_numpy(dtype=np.float32)   # NaN where unmeasured

    graphs: list[MolGraph] = []
    smiles: list[str] = []
    ys: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    failed = 0

    for i, smi in enumerate(smiles_col):
        if limit is not None and len(graphs) >= limit:
            break
        mol = Chem.MolFromSmiles(smi)
        if mol is None or mol.GetNumAtoms() == 0:
            failed += 1
            continue
        raw = label_block[i]
        mask = ~np.isnan(raw)
        y = np.nan_to_num(raw, nan=0.0)
        graphs.append(mol_to_graph(mol, y=y.tolist(), y_mask=mask.tolist()))
        smiles.append(smi)
        ys.append(raw)
        masks.append(mask)
        if verbose and len(graphs) % 2000 == 0:
            print(f"  featurised {len(graphs)} molecules...")

    return Tox21Dataset(
        graphs=graphs,
        smiles=smiles,
        y=np.asarray(ys, dtype=np.float32),
        y_mask=np.asarray(masks, dtype=bool),
        tasks=list(tasks),
        n_failed_parse=failed,
    )


def load_tox21_graphs(tasks: list[str] | None = None, limit: int | None = None,
                      cache: bool = True, verbose: bool = True) -> Tox21Dataset:
    """Load + featurise Tox21, caching the result so notebooks re-run fast.

    The cache lands in ``data/cache/`` which is git-ignored; delete it to
    re-featurise from scratch.
    """
    tasks = tasks or TOX21_TASKS
    cache_dir = repo_root() / "data" / "cache"
    key = f"tox21_{'-'.join(tasks)}_{limit}.pkl"
    cache_file = cache_dir / key

    if cache and cache_file.exists():
        with open(cache_file, "rb") as fh:
            ds = pickle.load(fh)
        if verbose:
            print(f"Loaded {len(ds)} featurised molecules from cache ({cache_file.name})")
        return ds

    df = load_tox21_frame()
    if verbose:
        print(f"Featurising {len(df) if limit is None else limit} molecules with RDKit...")
    ds = featurise(df, tasks, limit=limit, verbose=verbose)
    if verbose:
        print(f"  {len(ds)} graphs built, {ds.n_failed_parse} SMILES unparseable")
    if cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "wb") as fh:
            pickle.dump(ds, fh)
    return ds
