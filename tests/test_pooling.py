"""Readout behaviour, including the size-sensitivity argument from notebook 05."""

import pytest
import torch

from mgnn.features import smiles_to_graph
from mgnn.pooling import AttentionPooling, global_mean_pool, global_sum_pool
from mgnn.train import collate_graphs


def test_sum_is_size_extensive_mean_is_not():
    """A molecule and its exact dimer: sum doubles, mean does not move."""
    mono = smiles_to_graph("CCO")
    dimer = smiles_to_graph("CCO.CCO")
    b1, b2 = collate_graphs([mono]), collate_graphs([dimer])
    h1 = torch.ones(b1.x.shape[0], 4)
    h2 = torch.ones(b2.x.shape[0], 4)
    assert torch.allclose(global_sum_pool(h2, b2.batch, 1),
                          2 * global_sum_pool(h1, b1.batch, 1))
    assert torch.allclose(global_mean_pool(h2, b2.batch, 1),
                          global_mean_pool(h1, b1.batch, 1))


def test_pooling_separates_molecules_in_a_batch(graphs):
    batch = collate_graphs(list(graphs))
    h = torch.randn(batch.x.shape[0], 5)
    pooled = global_sum_pool(h, batch.batch, batch.n_graphs)
    assert pooled.shape == (len(graphs), 5)
    # molecule k's pooled vector must equal the sum of only its own atom rows
    for k in range(len(graphs)):
        assert torch.allclose(pooled[k], h[batch.batch == k].sum(dim=0), atol=1e-5)


def test_attention_weights_sum_to_one_per_molecule(graphs):
    batch = collate_graphs(list(graphs))
    pool = AttentionPooling(batch.x.shape[1], 6)
    _pooled, alpha = pool(batch.x, batch.batch, batch.n_graphs, return_weights=True)
    for k in range(batch.n_graphs):
        assert float(alpha[batch.batch == k].sum()) == pytest.approx(1.0, abs=1e-5)


def test_attention_pooling_is_batch_independent(graphs):
    """A molecule's readout must not depend on who else is in the batch.

    The softmax has to run within each molecule; this test is what catches the
    classic bug of softmaxing across the whole batch.
    """
    pool = AttentionPooling(graphs[0].x.shape[1], 6)
    pool.eval()
    alone = collate_graphs([graphs[3]])
    together = collate_graphs(list(graphs))
    with torch.no_grad():
        p_alone = pool(alone.x, alone.batch, 1)
        p_together = pool(together.x, together.batch, together.n_graphs)
    assert torch.allclose(p_alone[0], p_together[3], atol=1e-5)
