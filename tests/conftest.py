"""Shared fixtures.  ``pyg`` is imported lazily so the suite reports a clear
skip reason rather than a collection error if PyTorch Geometric is missing."""

import pytest
import torch

from mgnn.features import smiles_to_graph

# A small, chemically varied set: a 5-atom molecule for hand-checking, a ring,
# an amide, a charged species, a salt (two fragments) and a lone ion.
TEST_SMILES = [
    "CC(=O)O",                   # acetic acid, 4 heavy atoms
    "CCO",                       # ethanol
    "c1ccccc1",                  # benzene, a ring with uniform degrees
    "CC(=O)Nc1ccccc1",           # acetanilide
    "C[N+](C)(C)C",              # quaternary ammonium, high degree centre
    "CC(=O)[O-].[Na+]",          # two fragments
    "[Na+]",                     # single atom, no bonds at all
    "C1CC2CCC1CC2",              # bridged bicycle
]


@pytest.fixture(scope="session")
def graphs():
    return [smiles_to_graph(s) for s in TEST_SMILES]


@pytest.fixture(scope="session")
def small_graph():
    """Propanoic acid, CCC(=O)O: five heavy atoms, the molecule notebook 02 traces."""
    return smiles_to_graph("CCC(=O)O")


@pytest.fixture(autouse=True)
def _deterministic():
    torch.manual_seed(1234)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: deselect with -m 'not slow'")
