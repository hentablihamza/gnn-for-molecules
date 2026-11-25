"""Turn an RDKit molecule into the tensors a GNN consumes.

A molecular graph is three things:

* ``x``          -- node (atom) feature matrix, shape ``[n_atoms, n_atom_features]``
* ``edge_index`` -- COO connectivity, shape ``[2, n_directed_edges]``
* ``edge_attr``  -- edge (bond) feature matrix, shape ``[n_directed_edges, n_bond_features]``

Bonds are undirected in chemistry but GNN message passing is directed, so every
bond becomes **two** entries in ``edge_index``: ``i -> j`` and ``j -> i``.
That detail is the single most common source of confusion for newcomers, so the
code below is deliberately explicit about it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from rdkit import Chem

# --------------------------------------------------------------------------
# Vocabularies.  Keeping them module-level constants means the feature layout
# is documented, reproducible and testable.
# --------------------------------------------------------------------------

ATOM_SYMBOLS: list[str] = [
    "C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "Si", "Se", "other",
]
DEGREES: list[int] = [0, 1, 2, 3, 4, 5]
FORMAL_CHARGES: list[int] = [-2, -1, 0, 1, 2]
NUM_HS: list[int] = [0, 1, 2, 3, 4]
HYBRIDIZATIONS: list[Chem.rdchem.HybridizationType] = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
BOND_TYPES: list[Chem.rdchem.BondType] = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
BOND_STEREO: list[Chem.rdchem.BondStereo] = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
]


def one_hot(value, allowable: list, allow_other: bool = True) -> list[float]:
    """One-hot encode ``value`` against ``allowable``.

    If ``allow_other`` the vector gets one extra slot that catches anything not
    in the vocabulary, which keeps the feature width fixed when an unexpected
    element shows up in a new data set.
    """
    width = len(allowable) + (1 if allow_other else 0)
    vec = [0.0] * width
    if value in allowable:
        vec[allowable.index(value)] = 1.0
    elif allow_other:
        vec[-1] = 1.0
    return vec


def atom_features(atom: Chem.rdchem.Atom) -> list[float]:
    """Feature vector for one atom.

    The blocks are, in order: element, degree, formal charge, attached
    hydrogens, hybridisation, aromaticity flag, ring flag, and scaled mass.
    """
    symbol = atom.GetSymbol()
    if symbol not in ATOM_SYMBOLS:
        symbol = "other"
    return (
        one_hot(symbol, ATOM_SYMBOLS, allow_other=False)
        + one_hot(atom.GetTotalDegree(), DEGREES)
        + one_hot(atom.GetFormalCharge(), FORMAL_CHARGES)
        + one_hot(atom.GetTotalNumHs(), NUM_HS)
        + one_hot(atom.GetHybridization(), HYBRIDIZATIONS)
        + [1.0 if atom.GetIsAromatic() else 0.0]
        + [1.0 if atom.IsInRing() else 0.0]
        + [atom.GetMass() * 0.01]
    )


def bond_features(bond: Chem.rdchem.Bond) -> list[float]:
    """Feature vector for one bond: type, stereo, conjugation, ring membership."""
    return (
        one_hot(bond.GetBondType(), BOND_TYPES)
        + one_hot(bond.GetStereo(), BOND_STEREO)
        + [1.0 if bond.GetIsConjugated() else 0.0]
        + [1.0 if bond.IsInRing() else 0.0]
    )


ATOM_FEATURE_NAMES: list[str] = (
    [f"element={s}" for s in ATOM_SYMBOLS]
    + [f"degree={d}" for d in DEGREES] + ["degree=other"]
    + [f"charge={c}" for c in FORMAL_CHARGES] + ["charge=other"]
    + [f"numH={h}" for h in NUM_HS] + ["numH=other"]
    + [f"hyb={str(h)}" for h in HYBRIDIZATIONS] + ["hyb=other"]
    + ["aromatic", "in_ring", "mass*0.01"]
)
BOND_FEATURE_NAMES: list[str] = (
    [f"bond={str(b)}" for b in BOND_TYPES] + ["bond=other"]
    + [f"stereo={str(s)}" for s in BOND_STEREO] + ["stereo=other"]
    + ["conjugated", "in_ring"]
)

ATOM_FEATURE_DIM: int = len(ATOM_FEATURE_NAMES)
BOND_FEATURE_DIM: int = len(BOND_FEATURE_NAMES)


@dataclass
class MolGraph:
    """A molecule as graph tensors, plus the bookkeeping notebooks need."""

    x: torch.Tensor           # [n_atoms, ATOM_FEATURE_DIM]
    edge_index: torch.Tensor  # [2, 2 * n_bonds]
    edge_attr: torch.Tensor   # [2 * n_bonds, BOND_FEATURE_DIM]
    smiles: str
    y: torch.Tensor | None = None       # [n_tasks]
    y_mask: torch.Tensor | None = None  # [n_tasks], True where a label exists

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def atom_symbols(self) -> list[str]:
        mol = Chem.MolFromSmiles(self.smiles)
        return [a.GetSymbol() for a in mol.GetAtoms()]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"MolGraph(smiles={self.smiles!r}, nodes={self.num_nodes}, "
            f"directed_edges={self.num_edges})"
        )


def mol_to_graph(
    mol: Chem.rdchem.Mol,
    y: list[float] | None = None,
    y_mask: list[bool] | None = None,
) -> MolGraph:
    """Convert an RDKit mol into :class:`MolGraph`.

    Hydrogens are left implicit (the usual choice for 2-D molecular GNNs); the
    hydrogen count rides along inside each heavy atom's feature vector.
    """
    if mol is None:
        raise ValueError("mol_to_graph received None (did MolFromSmiles fail?)")

    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

    src: list[int] = []
    dst: list[int] = []
    eattr: list[list[float]] = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        # One chemical bond, two directed messages.
        src += [i, j]
        dst += [j, i]
        eattr += [bf, bf]

    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(eattr, dtype=torch.float)
    else:  # a single atom, e.g. "[Na+]": no bonds at all
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, BOND_FEATURE_DIM), dtype=torch.float)

    return MolGraph(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        smiles=Chem.MolToSmiles(mol),
        y=None if y is None else torch.tensor(y, dtype=torch.float),
        y_mask=None if y_mask is None else torch.tensor(y_mask, dtype=torch.bool),
    )


def smiles_to_graph(smiles: str, **kwargs) -> MolGraph:
    """Parse SMILES then delegate to :func:`mol_to_graph`."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    return mol_to_graph(mol, **kwargs)


def adjacency_matrix(graph: MolGraph, add_self_loops: bool = False) -> np.ndarray:
    """Dense adjacency matrix ``A`` (optionally ``A + I``) as a NumPy array.

    Dense matrices are how the textbook writes message passing and how
    notebooks 02 and 03 show it; sparse ``edge_index`` is how real code does it.
    """
    n = graph.num_nodes
    A = np.zeros((n, n), dtype=np.float64)
    ei = graph.edge_index.numpy()
    for k in range(ei.shape[1]):
        A[ei[0, k], ei[1, k]] = 1.0
    if add_self_loops:
        A = A + np.eye(n)
    return A


def degree_vector(A: np.ndarray) -> np.ndarray:
    """Row sums of ``A``: how many messages each node receives."""
    return A.sum(axis=1)
