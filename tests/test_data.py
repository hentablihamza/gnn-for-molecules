"""Tox21 loading, missing labels and the masked loss."""

import numpy as np
import torch

from mgnn.data import (
    TOX21_TASKS,
    WORKED_TASKS,
    class_balance,
    featurise,
    load_tox21_frame,
    task_summary,
)
from mgnn.train import collate_graphs, masked_bce_loss


def test_frame_shape_and_columns():
    df = load_tox21_frame()
    assert len(df) > 7000
    for t in TOX21_TASKS:
        assert t in df.columns
    assert "smiles" in df.columns


def test_labels_are_missing_not_zero():
    """Blank cells must survive as NaN. If they became 0 every metric lies."""
    df = load_tox21_frame()
    assert df[TOX21_TASKS].isna().to_numpy().sum() > 0
    for t in TOX21_TASKS:
        assert set(df[t].dropna().unique()) <= {0.0, 1.0}


def test_task_summary_counts_add_up():
    df = load_tox21_frame()
    summary = task_summary(df)
    for row in summary.itertuples():
        assert row.actives + row.inactives == row.measured
        assert row.measured + row.missing == len(df)


def test_tox21_is_strongly_imbalanced():
    df = load_tox21_frame()
    summary = task_summary(df, WORKED_TASKS).set_index("task")
    for t in WORKED_TASKS:
        assert summary.loc[t, "pct_active"] < 20.0, "assay unexpectedly balanced"


def test_featurise_propagates_mask():
    df = load_tox21_frame().head(200)
    ds = featurise(df, WORKED_TASKS)
    assert len(ds) <= 200
    assert ds.y.shape == (len(ds), 2)
    assert ds.y_mask.shape == (len(ds), 2)
    assert np.isnan(ds.y[~ds.y_mask]).all()          # unmeasured stay NaN
    assert not np.isnan(ds.y[ds.y_mask]).any()       # measured never NaN
    for g in ds.graphs:
        assert g.y_mask is not None and g.y_mask.shape == (2,)


def test_class_balance_pos_weight():
    y = np.array([1, 0, 0, 0, 0, np.nan])
    mask = np.array([True, True, True, True, True, False])
    bal = class_balance(y, mask)
    assert bal["actives"] == 1 and bal["inactives"] == 4
    assert bal["pos_weight"] == 4.0


def test_masked_loss_ignores_unmeasured_labels():
    logits = torch.tensor([[5.0, -5.0]])
    y = torch.tensor([[1.0, 1.0]])
    both = torch.tensor([[True, True]])
    first_only = torch.tensor([[True, False]])
    # the second entry is badly wrong; masking it out must lower the loss a lot
    assert float(masked_bce_loss(logits, y, first_only)) < float(
        masked_bce_loss(logits, y, both))
    # and an all-false mask contributes nothing at all
    none = torch.tensor([[False, False]])
    assert float(masked_bce_loss(logits, y, none)) == 0.0


def test_masked_loss_gradient_is_zero_for_unmeasured():
    logits = torch.tensor([[0.5, 0.5]], requires_grad=True)
    y = torch.tensor([[1.0, 0.0]])
    mask = torch.tensor([[True, False]])
    masked_bce_loss(logits, y, mask).backward()
    assert logits.grad[0, 1].item() == 0.0
    assert logits.grad[0, 0].item() != 0.0


def test_collate_offsets_edges_so_molecules_stay_disconnected(graphs):
    batch = collate_graphs(list(graphs))
    assert batch.x.shape[0] == sum(g.num_nodes for g in graphs)
    assert batch.edge_index.shape[1] == sum(g.num_edges for g in graphs)
    # an edge may never join two different molecules
    src_mol = batch.batch[batch.edge_index[0]]
    dst_mol = batch.batch[batch.edge_index[1]]
    assert torch.equal(src_mol, dst_mol)
    assert int(batch.edge_index.max()) < batch.x.shape[0]
