"""Whole-model behaviour: shapes, batch independence, and that it can learn."""

import pytest
import torch

from mgnn.models import LAYER_TYPES, POOLINGS, MolecularGNN
from mgnn.train import collate_graphs, evaluate, masked_bce_loss


@pytest.mark.parametrize("kind", LAYER_TYPES)
def test_forward_shape_for_every_layer_type(graphs, kind):
    model = MolecularGNN(n_tasks=2, hidden_dim=16, n_layers=2, layer_kind=kind)
    batch = collate_graphs(list(graphs))
    edge_attr = batch.edge_attr if model.uses_edge_features else None
    out = model(batch.x, batch.edge_index, batch.batch,
                edge_attr=edge_attr, n_graphs=batch.n_graphs)
    assert out.shape == (len(graphs), 2)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("pooling", POOLINGS)
def test_forward_shape_for_every_pooling(graphs, pooling):
    model = MolecularGNN(n_tasks=1, hidden_dim=16, n_layers=2, pooling=pooling)
    batch = collate_graphs(list(graphs))
    out = model(batch.x, batch.edge_index, batch.batch, n_graphs=batch.n_graphs)
    assert out.shape == (len(graphs), 1)


@pytest.mark.parametrize("kind", LAYER_TYPES)
def test_prediction_independent_of_batch_composition(graphs, kind):
    """Same molecule, different batch mates, same answer.

    This is the test that proves the batching trick (offset edge_index, one
    disconnected graph) does not leak information between molecules.
    """
    model = MolecularGNN(n_tasks=1, hidden_dim=16, n_layers=2, layer_kind=kind,
                         pooling="attention", dropout=0.0)
    model.eval()
    target = graphs[3]
    alone = collate_graphs([target])
    crowd = collate_graphs(list(graphs))
    with torch.no_grad():
        ea_a = alone.edge_attr if model.uses_edge_features else None
        ea_c = crowd.edge_attr if model.uses_edge_features else None
        p_alone = model(alone.x, alone.edge_index, alone.batch,
                        edge_attr=ea_a, n_graphs=1)
        p_crowd = model(crowd.x, crowd.edge_index, crowd.batch,
                        edge_attr=ea_c, n_graphs=crowd.n_graphs)
    assert torch.allclose(p_alone[0], p_crowd[3], atol=1e-5)


def test_node_embeddings_return_one_state_per_layer(graphs):
    model = MolecularGNN(hidden_dim=12, n_layers=4)
    batch = collate_graphs(list(graphs))
    states, final = model.node_embeddings(batch.x, batch.edge_index,
                                         return_all_layers=True)
    assert len(states) == 5          # input embedding + one per conv
    assert torch.equal(states[-1], final)
    assert all(s.shape == (batch.x.shape[0], 12) for s in states)


def test_gat_requires_divisible_width():
    with pytest.raises(ValueError):
        MolecularGNN(hidden_dim=10, layer_kind="gat", heads=4)


def test_model_can_overfit_a_tiny_set(graphs):
    """Sanity: a GNN with enough steps must be able to memorise 8 molecules."""
    model = MolecularGNN(n_tasks=1, hidden_dim=32, n_layers=2, dropout=0.0)
    batch = collate_graphs(list(graphs))
    y = torch.tensor([[1.0], [0.0], [1.0], [0.0], [1.0], [0.0], [1.0], [0.0]])
    mask = torch.ones_like(y, dtype=torch.bool)
    opt = torch.optim.Adam(model.parameters(), lr=0.02)
    first = None
    for step in range(150):
        logits = model(batch.x, batch.edge_index, batch.batch, n_graphs=batch.n_graphs)
        loss = masked_bce_loss(logits, y, mask)
        if step == 0:
            first = float(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss) < 0.25 * first


def test_evaluate_returns_nan_rather_than_a_fake_number(graphs):
    """A single-class test split cannot have an ROC-AUC; say so, do not invent one."""
    model = MolecularGNN(n_tasks=1, hidden_dim=8, n_layers=1)
    for g in graphs:
        g.y = torch.tensor([1.0])
        g.y_mask = torch.tensor([True])
    metrics = evaluate(model, list(graphs), ["only-actives"])
    assert metrics["per_task"]["only-actives"]["roc_auc"] != \
        metrics["per_task"]["only-actives"]["roc_auc"]   # NaN
    for g in graphs:                                     # leave fixtures clean
        g.y = None
        g.y_mask = None
