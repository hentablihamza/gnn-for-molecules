"""D-MPNN has no PyG counterpart, so it is pinned by a reference implementation.

``dmpnn_reference_forward`` is written straight from the equations in Yang et
al. (2019) with Python loops; the production path is vectorised.  Agreement
between two independently written versions is the evidence here.
"""

import torch

from mgnn.features import smiles_to_graph
from mgnn.layers import DMPNNLayer, dmpnn_reference_forward


def _layer(g, hidden=8, depth=3):
    return DMPNNLayer(g.x.shape[1], g.edge_attr.shape[1], hidden, depth=depth)


def test_matches_loop_reference(graphs):
    for g in graphs:
        layer = _layer(g)
        fast = layer(g.x, g.edge_index, g.edge_attr)
        slow = dmpnn_reference_forward(layer, g.x, g.edge_index, g.edge_attr)
        assert torch.allclose(fast, slow, atol=1e-5), g.smiles


def test_matches_reference_at_several_depths(small_graph):
    for depth in (1, 2, 4, 6):
        layer = _layer(small_graph, depth=depth)
        fast = layer(small_graph.x, small_graph.edge_index, small_graph.edge_attr)
        slow = dmpnn_reference_forward(layer, small_graph.x, small_graph.edge_index,
                                       small_graph.edge_attr)
        assert torch.allclose(fast, slow, atol=1e-5), f"depth={depth}"


def test_reverse_edge_lookup_is_an_involution(graphs):
    for g in graphs:
        if not g.num_edges:
            continue
        rev = DMPNNLayer.reverse_edge_index(g.edge_index)
        assert bool((rev >= 0).all()), g.smiles
        assert torch.equal(rev[rev], torch.arange(g.num_edges))


def test_permutation_invariance(small_graph):
    """Relabelling the atoms must permute the output the same way, nothing more."""
    g = small_graph
    layer = _layer(g)
    out = layer(g.x, g.edge_index, g.edge_attr)

    perm = torch.tensor([3, 0, 4, 1, 2])  # relabel the five atoms
    inv = torch.argsort(perm)
    x_p = g.x[perm]
    ei_p = inv[g.edge_index]
    out_p = layer(x_p, ei_p, g.edge_attr)
    assert torch.allclose(out_p, out[perm], atol=1e-5)


def test_single_atom_molecule_does_not_crash():
    g = smiles_to_graph("[Na+]")
    layer = _layer(g)
    out = layer(g.x, g.edge_index, g.edge_attr)
    assert out.shape == (1, 8)
    assert torch.isfinite(out).all()
