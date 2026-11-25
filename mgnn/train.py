"""Batching, masked loss, training loop and the metrics worth reporting.

Batching graphs
---------------
Molecules have different numbers of atoms, so you cannot stack them into a
tensor.  The standard trick (and what PyTorch Geometric does under the hood) is
to build one big **disconnected** graph: concatenate the node features, shift
each molecule's ``edge_index`` by the number of atoms already placed, and keep
a ``batch`` vector saying which molecule each atom came from.  Because the
molecules share no edges, message passing cannot leak between them.
:func:`collate_graphs` does exactly that, in twenty readable lines.

Metrics
-------
With a few percent actives, accuracy is meaningless, so :func:`evaluate`
reports ROC-AUC and average precision (PR-AUC), and refuses to score a task
whose test split contains only one class rather than returning a misleading
number.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from mgnn.features import MolGraph

__all__ = [
    "Batch", "collate_graphs", "iterate_minibatches", "masked_bce_loss",
    "evaluate", "train_model", "set_seed", "TrainHistory",
]


def set_seed(seed: int = 0) -> None:
    """Seed Python, NumPy and torch so a notebook re-runs to the same numbers."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class Batch:
    """One batch of molecules packed into a single disconnected graph."""

    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    batch: torch.Tensor
    y: torch.Tensor
    y_mask: torch.Tensor
    n_graphs: int
    smiles: list[str] = field(default_factory=list)


def collate_graphs(graphs: list[MolGraph]) -> Batch:
    """Concatenate molecular graphs, offsetting ``edge_index`` as we go."""
    xs, eis, eas, batches, ys, masks, smiles = [], [], [], [], [], [], []
    offset = 0
    for gi, g in enumerate(graphs):
        xs.append(g.x)
        eis.append(g.edge_index + offset)   # <-- the only subtle line
        eas.append(g.edge_attr)
        batches.append(torch.full((g.num_nodes,), gi, dtype=torch.long))
        if g.y is not None:
            ys.append(g.y)
            masks.append(g.y_mask if g.y_mask is not None
                         else torch.ones_like(g.y, dtype=torch.bool))
        smiles.append(g.smiles)
        offset += g.num_nodes

    return Batch(
        x=torch.cat(xs, dim=0),
        edge_index=torch.cat(eis, dim=1),
        edge_attr=torch.cat(eas, dim=0),
        batch=torch.cat(batches, dim=0),
        y=torch.stack(ys) if ys else torch.zeros(0),
        y_mask=torch.stack(masks) if masks else torch.zeros(0, dtype=torch.bool),
        n_graphs=len(graphs),
        smiles=smiles,
    )


def iterate_minibatches(graphs: list[MolGraph], batch_size: int = 64,
                        shuffle: bool = True, seed: int | None = None):
    """Yield :class:`Batch` objects; shuffling is seeded for reproducibility."""
    idx = list(range(len(graphs)))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        chunk = idx[start:start + batch_size]
        yield collate_graphs([graphs[i] for i in chunk])


def masked_bce_loss(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor,
                    pos_weight: torch.Tensor | None = None) -> torch.Tensor:
    """Binary cross-entropy over measured labels only.

    Unmeasured assay/compound pairs contribute exactly zero gradient -- they are
    not treated as negatives.  ``pos_weight`` multiplies the loss on actives,
    which is how you tell the optimiser that the 4% matter as much as the 96%.
    """
    if mask.sum() == 0:
        return logits.sum() * 0.0
    per_element = F.binary_cross_entropy_with_logits(
        logits, y, reduction="none", pos_weight=pos_weight
    )
    return (per_element * mask.float()).sum() / mask.float().sum()


@torch.no_grad()
def predict(model: nn.Module, graphs: list[MolGraph], batch_size: int = 128):
    """Return ``(logits, y, mask)`` stacked over the whole set."""
    model.eval()
    logits, ys, masks = [], [], []
    for batch in iterate_minibatches(graphs, batch_size, shuffle=False):
        edge_attr = batch.edge_attr if getattr(model, "uses_edge_features", False) else None
        out = model(batch.x, batch.edge_index, batch.batch,
                    edge_attr=edge_attr, n_graphs=batch.n_graphs)
        logits.append(out)
        ys.append(batch.y)
        masks.append(batch.y_mask)
    return torch.cat(logits), torch.cat(ys), torch.cat(masks)


def evaluate(model: nn.Module, graphs: list[MolGraph], tasks: list[str],
             batch_size: int = 128) -> dict:
    """ROC-AUC and PR-AUC per task, plus the macro average over scorable tasks."""
    logits, y, mask = predict(model, graphs, batch_size)
    probs = torch.sigmoid(logits).numpy()
    y_np, mask_np = y.numpy(), mask.numpy()

    per_task = {}
    for j, task in enumerate(tasks):
        m = mask_np[:, j]
        yt, pt = y_np[m, j], probs[m, j]
        if m.sum() == 0 or len(np.unique(yt)) < 2:
            per_task[task] = {"roc_auc": float("nan"), "pr_auc": float("nan"),
                              "n": int(m.sum()), "n_active": int((yt == 1).sum())}
            continue
        per_task[task] = {
            "roc_auc": float(roc_auc_score(yt, pt)),
            "pr_auc": float(average_precision_score(yt, pt)),
            "n": int(m.sum()),
            "n_active": int((yt == 1).sum()),
            "base_rate": float((yt == 1).mean()),
        }
    scorable = [v["roc_auc"] for v in per_task.values() if v["roc_auc"] == v["roc_auc"]]
    scorable_pr = [v["pr_auc"] for v in per_task.values() if v["pr_auc"] == v["pr_auc"]]
    return {
        "per_task": per_task,
        "mean_roc_auc": float(np.mean(scorable)) if scorable else float("nan"),
        "mean_pr_auc": float(np.mean(scorable_pr)) if scorable_pr else float("nan"),
    }


@dataclass
class TrainHistory:
    """Per-epoch record so notebooks can plot learning curves."""

    train_loss: list[float] = field(default_factory=list)
    valid_roc_auc: list[float] = field(default_factory=list)
    valid_pr_auc: list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_valid_roc_auc: float = float("-inf")


def train_model(
    model: nn.Module,
    train_graphs: list[MolGraph],
    valid_graphs: list[MolGraph],
    tasks: list[str],
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    pos_weight: torch.Tensor | None = None,
    seed: int = 0,
    verbose: bool = True,
    early_stopping_patience: int | None = 10,
) -> TrainHistory:
    """Plain Adam training loop with validation-ROC-AUC model selection.

    The best state dict (by validation ROC-AUC) is restored at the end, so the
    test number you report is not the number from an arbitrary last epoch.
    """
    set_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = TrainHistory()
    best_state = None
    since_improved = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total, n_batches = 0.0, 0
        for batch in iterate_minibatches(train_graphs, batch_size, shuffle=True,
                                         seed=seed + epoch):
            edge_attr = batch.edge_attr if getattr(model, "uses_edge_features", False) else None
            logits = model(batch.x, batch.edge_index, batch.batch,
                           edge_attr=edge_attr, n_graphs=batch.n_graphs)
            loss = masked_bce_loss(logits, batch.y, batch.y_mask, pos_weight)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss)
            n_batches += 1

        train_loss = total / max(n_batches, 1)
        metrics = evaluate(model, valid_graphs, tasks)
        history.train_loss.append(train_loss)
        history.valid_roc_auc.append(metrics["mean_roc_auc"])
        history.valid_pr_auc.append(metrics["mean_pr_auc"])

        if metrics["mean_roc_auc"] > history.best_valid_roc_auc:
            history.best_valid_roc_auc = metrics["mean_roc_auc"]
            history.best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            since_improved = 0
        else:
            since_improved += 1

        if verbose:
            print(f"epoch {epoch:3d} | train loss {train_loss:.4f} "
                  f"| valid ROC-AUC {metrics['mean_roc_auc']:.3f} "
                  f"| valid PR-AUC {metrics['mean_pr_auc']:.3f}")

        if early_stopping_patience and since_improved >= early_stopping_patience:
            if verbose:
                print(f"early stop at epoch {epoch} "
                      f"(no improvement for {since_improved} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history
