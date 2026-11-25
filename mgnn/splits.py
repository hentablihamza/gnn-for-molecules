"""Bemis-Murcko scaffold splitting, and why Tox21 needs it.

A random split puts close analogues of the same chemical series on both sides
of the train/test line.  The model then only has to recognise "this looks like
compound 412, which was toxic" -- a task it can solve from a fingerprint
nearest neighbour.  Reported ROC-AUC goes up and tells you nothing about the
next series your chemists will make.

A **scaffold split** groups molecules by their Bemis-Murcko framework (ring
systems plus the linkers between them, substituents stripped) and keeps whole
groups together.  Test compounds then have frameworks the model has never seen,
which is the question you actually care about: does it generalise to new
chemistry?

Reference: Bemis, G. W.; Murcko, M. A. "The Properties of Known Drugs. 1.
Molecular Frameworks." J. Med. Chem. 1996, 39 (15), 2887-2893.
DOI: 10.1021/jm9602928.
"""

from __future__ import annotations

from collections import defaultdict

from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

__all__ = ["murcko_scaffold", "scaffold_groups", "scaffold_split", "UNPARSEABLE"]

#: Group label for SMILES RDKit refuses (Tox21 contains a handful, e.g. some
#: aluminium and tin species with valences RDKit rejects).  They are kept in one
#: group rather than dropped silently, so a split always covers every input.
UNPARSEABLE = "<unparseable>"


def murcko_scaffold(smiles: str, include_chirality: bool = False) -> str:
    """Canonical Bemis-Murcko scaffold SMILES for one molecule.

    Acyclic molecules have an empty framework; they all collapse to ``""`` and
    therefore travel together, which is the honest behaviour (they are in a
    sense one "no-ring" class).  SMILES RDKit cannot parse return
    :data:`UNPARSEABLE` so that callers can decide what to do rather than
    having a split blow up halfway through a data set.
    """
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(
            smiles=smiles, includeChirality=include_chirality
        )
    except Exception:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return UNPARSEABLE
        try:
            return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
        except Exception:
            return UNPARSEABLE


def scaffold_groups(smiles_list: list[str],
                    include_chirality: bool = False) -> dict[str, list[int]]:
    """Map scaffold SMILES -> indices of the molecules that share it."""
    groups: dict[str, list[int]] = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        groups[murcko_scaffold(smi, include_chirality)].append(i)
    return dict(groups)


def scaffold_split(
    smiles_list: list[str],
    frac_train: float = 0.8,
    frac_valid: float = 0.1,
    frac_test: float = 0.1,
    include_chirality: bool = False,
) -> tuple[list[int], list[int], list[int]]:
    """Deterministic scaffold split, largest scaffold groups first.

    Filling train first with the big, well-populated frameworks and pushing the
    singletons into validation/test is the convention MoleculeNet uses.  It
    makes the test set deliberately harder and more structurally diverse than
    the training set -- that is the point, not a flaw.

    Returns three index lists.  No scaffold appears in more than one of them,
    which ``tests/test_splits.py`` asserts.
    """
    total = frac_train + frac_valid + frac_test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1.0, got {total}")

    groups = scaffold_groups(smiles_list, include_chirality)
    # sort by size (descending), then by scaffold string so ties are reproducible
    ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))

    n = len(smiles_list)
    n_train, n_valid = int(frac_train * n), int(frac_valid * n)
    train: list[int] = []
    valid: list[int] = []
    test: list[int] = []
    for _scaffold, idx in ordered:
        # A group is never cut in half -- that is the invariant that makes the
        # split leak-free.  Quotas are therefore approximate by design.
        if len(train) + len(idx) <= n_train:
            train += idx
        elif len(valid) + len(idx) <= n_valid:
            valid += idx
        else:
            test += idx
    return sorted(train), sorted(valid), sorted(test)
