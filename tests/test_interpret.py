"""Attribution plumbing: shapes, normalisation, and that it reacts to the model."""

import numpy as np
import torch

from mgnn.features import smiles_to_graph
from mgnn.interpret import (
    aggregate_gat_attention,
    attention_atom_weights,
    gradient_atom_attribution,
    integrated_gradients_atom_attribution,
)
from mgnn.layers import GATLayer
from mgnn.models import MolecularGNN


def _model(**kw):
    kw.setdefault("pooling", "attention")
    kw.setdefault("dropout", 0.0)
    return MolecularGNN(n_tasks=1, hidden_dim=16, n_layers=2, **kw)


def test_attention_weights_one_per_atom_and_normalised():
    g = smiles_to_graph("CC(=O)Nc1ccccc1")
    w = attention_atom_weights(_model(), g)
    assert w.shape == (g.num_nodes,)
    assert np.isclose(w.sum(), 1.0, atol=1e-5)
    assert (w >= 0).all()


def test_attention_requires_attention_pooling():
    g = smiles_to_graph("CCO")
    try:
        attention_atom_weights(_model(pooling="sum"), g)
    except ValueError as exc:
        assert "attention" in str(exc)
    else:
        raise AssertionError("should refuse a model without attention pooling")


def test_gradient_attribution_shape_and_sign():
    g = smiles_to_graph("CC(=O)Nc1ccccc1")
    a = gradient_atom_attribution(_model(), g)
    assert a.shape == (g.num_nodes,)
    assert np.isfinite(a).all()
    assert np.abs(a).sum() > 0
    unsigned = gradient_atom_attribution(_model(), g, signed=False)
    assert (unsigned >= 0).all()


def test_integrated_gradients_shape_and_stability():
    g = smiles_to_graph("CC(=O)Nc1ccccc1")
    model = _model()
    a1 = integrated_gradients_atom_attribution(model, g, steps=16)
    a2 = integrated_gradients_atom_attribution(model, g, steps=16)
    assert a1.shape == (g.num_nodes,)
    assert np.allclose(a1, a2, atol=1e-5)       # deterministic in eval mode


def test_attribution_changes_when_the_model_changes():
    g = smiles_to_graph("CC(=O)Nc1ccccc1")
    torch.manual_seed(0)
    a = gradient_atom_attribution(_model(), g)
    torch.manual_seed(7)
    b = gradient_atom_attribution(_model(), g)
    assert not np.allclose(a, b)


def test_gat_attention_aggregation():
    g = smiles_to_graph("CC(=O)Nc1ccccc1")
    layer = GATLayer(g.x.shape[1], 8, heads=4)
    per_atom, per_edge = aggregate_gat_attention(layer, g.x, g.edge_index)
    assert per_atom.shape == (g.num_nodes,)
    # self-loops were added, so there is one extra edge per atom
    assert per_edge.shape == (g.num_edges + g.num_nodes,)
    assert (per_atom > 0).all() and (per_atom <= 1).all()


def test_attribution_does_not_leave_gradients_behind():
    """An explanation call must not quietly dirty the model's .grad buffers."""
    g = smiles_to_graph("CCO")
    model = _model()
    gradient_atom_attribution(model, g)
    assert all(p.grad is None or torch.all(p.grad == 0) for p in model.parameters())
