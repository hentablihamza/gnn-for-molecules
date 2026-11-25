"""The central claim of this repository: the from-scratch layers are the real thing.

Each test copies parameters from the PyTorch Geometric layer into the
hand-written one and asserts the outputs agree to float32 tolerance on real
molecular graphs.  ``1e-6`` is the right tolerance: these are the same
arithmetic in a different order, not an approximation.
"""

import pytest
import torch
import torch.nn as nn

from mgnn import layers as L

pyg_nn = pytest.importorskip(
    "torch_geometric.nn", reason="PyTorch Geometric not installed"
)

TOL = 1e-6


def test_gcn_matches_pyg(graphs):
    for g in graphs:
        pyg = pyg_nn.GCNConv(g.x.shape[1], 8)
        mine = L.GCNLayer(g.x.shape[1], 8).copy_weights_from_pyg(pyg)
        assert torch.allclose(pyg(g.x, g.edge_index), mine(g.x, g.edge_index), atol=TOL)


def test_sage_matches_pyg(graphs):
    for g in graphs:
        pyg = pyg_nn.SAGEConv(g.x.shape[1], 8)
        mine = L.SAGELayer(g.x.shape[1], 8).copy_weights_from_pyg(pyg)
        assert torch.allclose(pyg(g.x, g.edge_index), mine(g.x, g.edge_index), atol=TOL)


@pytest.mark.parametrize("heads,concat", [(1, True), (2, True), (4, True), (4, False)])
def test_gat_matches_pyg_including_attention(graphs, heads, concat):
    for g in graphs:
        pyg = pyg_nn.GATConv(g.x.shape[1], 6, heads=heads, concat=concat)
        mine = L.GATLayer(g.x.shape[1], 6, heads=heads,
                          concat=concat).copy_weights_from_pyg(pyg)
        out_pyg, (ei_pyg, a_pyg) = pyg(g.x, g.edge_index, return_attention_weights=True)
        out_mine, (ei_mine, a_mine) = mine(g.x, g.edge_index, return_attention=True)
        assert torch.allclose(out_pyg, out_mine, atol=TOL), g.smiles
        assert torch.equal(ei_pyg, ei_mine), g.smiles
        assert torch.allclose(a_pyg, a_mine, atol=TOL), g.smiles


def test_gat_attention_rows_sum_to_one(graphs):
    """Every atom's incoming attention is a probability distribution."""
    for g in graphs:
        layer = L.GATLayer(g.x.shape[1], 5, heads=3)
        _out, (ei, alpha) = layer(g.x, g.edge_index, return_attention=True)
        received = L.scatter_sum(alpha, ei[1], g.num_nodes)
        assert torch.allclose(received, torch.ones_like(received), atol=1e-5)


def test_gin_matches_pyg(graphs):
    for g in graphs:
        mlp = nn.Sequential(nn.Linear(g.x.shape[1], 10), nn.ReLU(), nn.Linear(10, 8))
        pyg = pyg_nn.GINConv(mlp, eps=0.0)
        mine = L.GINLayer(mlp, eps=0.0)   # same MLP object, so same weights
        assert torch.allclose(pyg(g.x, g.edge_index), mine(g.x, g.edge_index), atol=TOL)


def test_gin_matches_pyg_with_trained_eps(graphs):
    for g in graphs:
        mlp = nn.Sequential(nn.Linear(g.x.shape[1], 10), nn.ReLU(), nn.Linear(10, 8))
        pyg = pyg_nn.GINConv(mlp, eps=0.3, train_eps=True)
        mine = L.GINLayer(mlp, eps=0.3, train_eps=True)
        assert torch.allclose(pyg(g.x, g.edge_index), mine(g.x, g.edge_index), atol=TOL)


def test_scatter_helpers_match_pyg_pooling(graphs):
    from mgnn.train import collate_graphs
    batch = collate_graphs(list(graphs))
    h = torch.randn(batch.x.shape[0], 7)
    assert torch.allclose(
        L.scatter_sum(h, batch.batch, batch.n_graphs),
        pyg_nn.global_add_pool(h, batch.batch), atol=TOL)
    assert torch.allclose(
        L.scatter_mean(h, batch.batch, batch.n_graphs),
        pyg_nn.global_mean_pool(h, batch.batch), atol=TOL)


def test_gcn_normalised_adjacency_properties(graphs):
    """Self-loops present, symmetric, and rows of a regular graph sum to 1."""
    for g in graphs:
        A_hat = L.gcn_normalised_adjacency(g.edge_index, g.num_nodes)
        assert torch.allclose(A_hat, A_hat.T, atol=TOL)
        assert bool((A_hat.diagonal() > 0).all())
    benzene = [g for g in graphs if g.smiles == "c1ccccc1"][0]
    A_hat = L.gcn_normalised_adjacency(benzene.edge_index, benzene.num_nodes)
    # every carbon has the same degree, so normalisation makes rows sum to 1
    assert torch.allclose(A_hat.sum(dim=1), torch.ones(6), atol=1e-5)
