"""The ChemNet reconstruction: distances, invariance, and the GCN special case."""

import numpy as np
import torch

from mgnn.chemnet import (
    ChemNetLayer,
    ChemNetModel,
    distance_buckets,
    topological_distance_matrix,
)
from mgnn.features import smiles_to_graph
from mgnn.layers import gcn_normalised_adjacency


def test_distance_matrix_matches_hand_count():
    """Propanal C0-C1-C2=O3 (RDKit order for CCC=O): distances along the chain."""
    D = topological_distance_matrix("CCC=O")
    assert D.shape == (4, 4)
    assert (np.diag(D) == 0).all()
    assert np.array_equal(D, D.T)
    assert D[0, 1] == 1 and D[0, 2] == 2 and D[0, 3] == 3


def test_separate_fragments_are_unreachable():
    D = topological_distance_matrix("CCO.CCO")
    assert (D == -1).any(), "salt fragments should be marked unreachable"


def test_buckets_are_capped():
    D = topological_distance_matrix("CCCCCCCCCC")       # decane, distances up to 9
    b = distance_buckets(D, max_distance=3)
    assert int(b.max()) == 4                            # 0..3 plus the overflow bucket
    assert int(b.min()) == 0


def test_unreachable_shares_the_overflow_bucket():
    D = topological_distance_matrix("CCO.CCO")
    b = distance_buckets(D, max_distance=5)
    assert int(b.max()) == 6


def test_layer_is_permutation_equivariant():
    g = smiles_to_graph("CCC=O")
    D = topological_distance_matrix(g.smiles)
    b = distance_buckets(D, 5)
    layer = ChemNetLayer(g.x.shape[1], 8, max_distance=5)
    out = layer(g.x, b)

    perm = torch.tensor([2, 0, 3, 1])
    b_p = b[perm][:, perm]
    out_p = layer(g.x[perm], b_p)
    assert torch.allclose(out_p, out[perm], atol=1e-5)


def test_one_hop_only_reduces_to_a_gcn_style_layer():
    """Zero every distance shell past one and ChemNet becomes a graph convolution.

    With mean normalisation per shell the coefficients are row-stochastic
    rather than symmetric, so this checks the *support* -- which atoms can
    reach which -- against the GCN normalised adjacency, not the exact weights.
    """
    g = smiles_to_graph("CC(=O)Nc1ccccc1")
    b = distance_buckets(topological_distance_matrix(g.smiles), 5)
    layer = ChemNetLayer(g.x.shape[1], 6, max_distance=5, bias=False)
    with torch.no_grad():
        for d in range(2, layer.n_buckets):
            layer.weight[d].zero_()
        layer.weight[0].copy_(layer.weight[1])      # tie "self" to "neighbour"
    out = layer(g.x, b)

    A_hat = gcn_normalised_adjacency(g.edge_index, g.num_nodes)
    reach_gcn = (A_hat > 0)
    reach_chemnet = ((b == 0) | (b == 1))
    assert torch.equal(reach_gcn, reach_chemnet)
    assert torch.isfinite(out).all()


def test_all_pairs_in_a_single_layer():
    """One ChemNet layer already mixes distant atoms; one GCN layer cannot."""
    g = smiles_to_graph("CCCCCCCC")       # octane: atoms 0 and 7 are 7 bonds apart
    b = distance_buckets(topological_distance_matrix(g.smiles), max_distance=10)
    layer = ChemNetLayer(g.x.shape[1], 4, max_distance=10, bias=False)
    x = torch.zeros_like(g.x)
    x[7] = g.x[7]                          # only the far atom carries a signal
    out = layer(x, b)
    assert out[0].abs().sum() > 0, "atom 0 should hear from atom 7 in one layer"


def test_model_forward_and_training_step():
    smiles = ["CCO", "c1ccccc1", "CC(=O)Nc1ccccc1"]
    graphs = [smiles_to_graph(s) for s in smiles]
    buckets = [distance_buckets(topological_distance_matrix(s), 5) for s in smiles]
    model = ChemNetModel(n_tasks=1, hidden_dim=16, n_layers=2, max_distance=5)
    out = model([g.x for g in graphs], buckets)
    assert out.shape == (3, 1)

    y = torch.tensor([[1.0], [0.0], [1.0]])
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    losses = []
    for _ in range(40):
        logits = model([g.x for g in graphs], buckets)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert losses[-1] < losses[0]
