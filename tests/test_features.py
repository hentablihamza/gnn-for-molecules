"""Featurisation invariants.  If these break, every notebook is wrong."""

import numpy as np
import torch
from rdkit import Chem

from mgnn import features as F


def test_feature_widths_match_names():
    assert F.ATOM_FEATURE_DIM == len(F.ATOM_FEATURE_NAMES)
    assert F.BOND_FEATURE_DIM == len(F.BOND_FEATURE_NAMES)
    g = F.smiles_to_graph("CC(=O)O")
    assert g.x.shape[1] == F.ATOM_FEATURE_DIM
    assert g.edge_attr.shape[1] == F.BOND_FEATURE_DIM


def test_node_count_equals_heavy_atoms(graphs):
    for g in graphs:
        mol = Chem.MolFromSmiles(g.smiles)
        assert g.num_nodes == mol.GetNumAtoms()


def test_every_bond_becomes_two_directed_edges(graphs):
    for g in graphs:
        mol = Chem.MolFromSmiles(g.smiles)
        assert g.num_edges == 2 * mol.GetNumBonds()


def test_edges_come_in_reverse_pairs(graphs):
    for g in graphs:
        edges = {(int(a), int(b)) for a, b in zip(*g.edge_index.tolist(), strict=False)}
        for a, b in edges:
            assert (b, a) in edges, f"missing reverse of {(a, b)} in {g.smiles}"


def test_no_self_loops_from_featuriser(graphs):
    for g in graphs:
        if g.num_edges:
            assert not bool((g.edge_index[0] == g.edge_index[1]).any())


def test_single_atom_molecule_has_empty_edge_tensors():
    g = F.smiles_to_graph("[Na+]")
    assert g.num_nodes == 1
    assert g.edge_index.shape == (2, 0)
    assert g.edge_attr.shape == (0, F.BOND_FEATURE_DIM)


def test_one_hot_blocks_are_one_hot():
    g = F.smiles_to_graph("CC(=O)Nc1ccccc1")
    # element block is the first len(ATOM_SYMBOLS) columns, exactly one hot
    block = g.x[:, : len(F.ATOM_SYMBOLS)]
    assert torch.allclose(block.sum(dim=1), torch.ones(g.num_nodes))


def test_unknown_element_falls_into_other_bucket():
    g = F.smiles_to_graph("[Pt]")   # platinum is not in ATOM_SYMBOLS
    other_col = F.ATOM_SYMBOLS.index("other")
    assert g.x[0, other_col] == 1.0


def test_adjacency_matrix_is_symmetric_for_molecules(graphs):
    for g in graphs:
        A = F.adjacency_matrix(g)
        assert np.allclose(A, A.T)
        assert np.allclose(np.diag(A), 0.0)
        assert A.sum() == g.num_edges


def test_adjacency_degree_matches_rdkit(graphs):
    for g in graphs:
        mol = Chem.MolFromSmiles(g.smiles)
        A = F.adjacency_matrix(g)
        deg = F.degree_vector(A)
        rdkit_deg = [a.GetDegree() for a in mol.GetAtoms()]
        assert list(deg.astype(int)) == rdkit_deg


def test_labels_and_mask_round_trip():
    g = F.smiles_to_graph("CCO", y=[1.0, 0.0], y_mask=[True, False])
    assert torch.equal(g.y, torch.tensor([1.0, 0.0]))
    assert torch.equal(g.y_mask, torch.tensor([True, False]))
