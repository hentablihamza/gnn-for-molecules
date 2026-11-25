"""Which atoms drove the prediction?

Two complementary answers, both per-atom scores you can paint on the structure:

* **Attention weights** (:func:`attention_atom_weights`) -- read straight out of
  an attention-pooled readout.  Cheap, but they tell you where the model
  *looked*, which is not the same as what changed the answer.  Treat them as a
  hypothesis, not an explanation.
* **Gradient attribution** (:func:`gradient_atom_attribution`) -- how much the
  logit moves when an atom's features move.  Signed, so it distinguishes atoms
  pushing *towards* toxic from atoms pushing away, and it is a property of the
  decision rather than of the architecture.  Also noisier.

When the two agree on a substructure, that substructure is worth showing a
chemist.  When they disagree, the model is probably leaning on a dataset
artefact -- which is itself useful to know.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = [
    "attention_atom_weights",
    "gradient_atom_attribution",
    "integrated_gradients_atom_attribution",
    "aggregate_gat_attention",
]


def _single_graph_inputs(model, graph):
    batch = torch.zeros(graph.num_nodes, dtype=torch.long)
    edge_attr = graph.edge_attr if getattr(model, "uses_edge_features", False) else None
    return batch, edge_attr


@torch.no_grad()
def attention_atom_weights(model, graph) -> np.ndarray:
    """Per-atom readout attention weights, summing to 1 over the molecule.

    Requires a model built with ``pooling='attention'``.
    """
    if getattr(model, "att_pool", None) is None:
        raise ValueError(
            "this model has no attention pooling; build it with pooling='attention'"
        )
    model.eval()
    batch, edge_attr = _single_graph_inputs(model, graph)
    _logit, weights = model(graph.x, graph.edge_index, batch,
                            edge_attr=edge_attr, n_graphs=1, return_attention=True)
    return weights.detach().numpy()


def gradient_atom_attribution(model, graph, task_index: int = 0,
                              signed: bool = True) -> np.ndarray:
    r"""Saliency: :math:`\sum_f x_{if} \cdot \partial \mathrm{logit} / \partial x_{if}`.

    The feature-times-gradient (rather than bare gradient) form answers "how
    much did this atom's actual features contribute", which is the question a
    chemist is asking.  Set ``signed=False`` for magnitudes only.
    """
    model.eval()
    x = graph.x.clone().requires_grad_(True)
    batch, edge_attr = _single_graph_inputs(model, graph)
    logits = model(x, graph.edge_index, batch, edge_attr=edge_attr, n_graphs=1)
    target = logits[0, task_index]
    model.zero_grad(set_to_none=True)
    target.backward()
    contrib = (x * x.grad).sum(dim=1).detach().numpy()
    model.zero_grad(set_to_none=True)   # leave the model as we found it
    return contrib if signed else np.abs(contrib)


def integrated_gradients_atom_attribution(model, graph, task_index: int = 0,
                                          steps: int = 32) -> np.ndarray:
    """Integrated gradients from an all-zero baseline (a "molecule" with no features).

    Steadier than plain saliency because it averages gradients along a path
    instead of trusting one point, at ``steps`` times the cost.  Sundararajan,
    Taly & Yan, ICML 2017, DOI: 10.48550/arXiv.1703.01365.
    """
    model.eval()
    baseline = torch.zeros_like(graph.x)
    batch, edge_attr = _single_graph_inputs(model, graph)
    total = torch.zeros_like(graph.x)
    for alpha in np.linspace(1.0 / steps, 1.0, steps):
        x = (baseline + alpha * (graph.x - baseline)).detach().requires_grad_(True)
        logits = model(x, graph.edge_index, batch, edge_attr=edge_attr, n_graphs=1)
        model.zero_grad(set_to_none=True)
        logits[0, task_index].backward()
        total = total + x.grad
    avg_grad = total / steps
    model.zero_grad(set_to_none=True)
    return ((graph.x - baseline) * avg_grad).sum(dim=1).detach().numpy()


@torch.no_grad()
def aggregate_gat_attention(layer, x: torch.Tensor,
                            edge_index: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Run one :class:`~mgnn.layers.GATLayer` and reduce its edge attention to atoms.

    Returns ``(per_atom, per_edge)``: ``per_atom[i]`` is the mean attention that
    atom ``i`` *received* across heads and incoming edges, which is the number
    notebook 04 paints on the structure; ``per_edge`` is head-averaged
    :math:`\\alpha_{ij}` in the order of the (self-loop-extended) edge list the
    layer used, returned so you can draw bond-level arrows instead.
    """
    _out, (ei, alpha) = layer(x, edge_index, return_attention=True)
    alpha_mean = alpha.mean(dim=1)                       # average the heads
    n = x.shape[0]
    received = np.zeros(n)
    counts = np.zeros(n)
    dst = ei[1].numpy()
    for k, a in enumerate(alpha_mean.numpy()):
        received[dst[k]] += a
        counts[dst[k]] += 1
    return received / np.maximum(counts, 1), alpha_mean.numpy()
