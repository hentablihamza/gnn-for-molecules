"""Whole models: stack layers, pool, predict.

One configurable class (:class:`MolecularGNN`) covers every architecture the
notebooks compare, so that a depth study (notebook 06) or a layer comparison
(notebook 04) is a change of two arguments rather than five copies of a
training script.
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F

from mgnn.features import ATOM_FEATURE_DIM, BOND_FEATURE_DIM
from mgnn.layers import DMPNNLayer, GATLayer, GCNLayer, GINLayer, SAGELayer
from mgnn.pooling import AttentionPooling, global_mean_pool, global_sum_pool

__all__ = ["MolecularGNN", "make_layer", "LAYER_TYPES", "POOLINGS"]

LAYER_TYPES = ("gcn", "sage", "gat", "gin", "dmpnn")
POOLINGS = ("sum", "mean", "attention")


def make_layer(kind: str, in_dim: int, out_dim: int, heads: int = 1) -> nn.Module:
    """Factory so the notebooks can say ``make_layer('gat', 64, 64)``."""
    kind = kind.lower()
    if kind == "gcn":
        return GCNLayer(in_dim, out_dim)
    if kind == "sage":
        return SAGELayer(in_dim, out_dim)
    if kind == "gat":
        # out_dim is the *total* width, split across heads, so every layer type
        # in a comparison has the same parameter budget per node.
        if out_dim % heads:
            raise ValueError(f"out_dim {out_dim} not divisible by heads {heads}")
        return GATLayer(in_dim, out_dim // heads, heads=heads, concat=True)
    if kind == "gin":
        mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)
        )
        return GINLayer(mlp, eps=0.0, train_eps=True)
    raise ValueError(f"unknown layer kind {kind!r}; choose from {LAYER_TYPES}")


class MolecularGNN(nn.Module):
    """``embed -> n_layers x (conv, norm, ReLU) -> readout -> MLP head``.

    Parameters
    ----------
    layer_kind
        One of :data:`LAYER_TYPES`.  ``'dmpnn'`` replaces the whole conv stack
        with a single :class:`~mgnn.layers.DMPNNLayer` of depth ``n_layers``,
        because in that architecture depth is a property of the layer.
    pooling
        One of :data:`POOLINGS`.
    residual
        Add the input of each conv to its output.  Notebook 06 uses this to
        show that residual connections delay, but do not remove, oversmoothing.
    """

    def __init__(
        self,
        n_tasks: int = 1,
        hidden_dim: int = 64,
        n_layers: int = 3,
        layer_kind: str = "gcn",
        pooling: str = "sum",
        heads: int = 4,
        dropout: float = 0.1,
        residual: bool = False,
        node_dim: int = ATOM_FEATURE_DIM,
        edge_dim: int = BOND_FEATURE_DIM,
    ):
        super().__init__()
        self.layer_kind = layer_kind.lower()
        self.pooling_kind = pooling.lower()
        self.n_layers = n_layers
        self.dropout = dropout
        self.residual = residual
        self.uses_edge_features = self.layer_kind == "dmpnn"

        if self.pooling_kind not in POOLINGS:
            raise ValueError(f"unknown pooling {pooling!r}; choose from {POOLINGS}")

        if self.layer_kind == "dmpnn":
            self.encoder = DMPNNLayer(node_dim, edge_dim, hidden_dim, depth=n_layers)
            self.convs = nn.ModuleList()
            self.norms = nn.ModuleList()
            self.embed = None
        else:
            self.embed = nn.Linear(node_dim, hidden_dim)
            self.convs = nn.ModuleList(
                [make_layer(self.layer_kind, hidden_dim, hidden_dim, heads=heads)
                 for _ in range(n_layers)]
            )
            self.norms = nn.ModuleList(
                [nn.LayerNorm(hidden_dim) for _ in range(n_layers)]
            )
            self.encoder = None

        self.att_pool = AttentionPooling(hidden_dim) if self.pooling_kind == "attention" else None
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, n_tasks),
        )

    # -- node representations -------------------------------------------
    def node_embeddings(self, x, edge_index, edge_attr=None,
                        return_all_layers: bool = False):
        """Atom vectors after message passing (the thing notebooks 06 and 08 poke at)."""
        if self.layer_kind == "dmpnn":
            h = self.encoder(x, edge_index, edge_attr)
            return ([h], h) if return_all_layers else h

        h = self.embed(x)
        states = [h]
        for conv, norm in zip(self.convs, self.norms, strict=False):
            h_in = h
            h = conv(h, edge_index)
            h = norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            if self.residual:
                h = h + h_in
            states.append(h)
        return (states, h) if return_all_layers else h

    # -- readout ---------------------------------------------------------
    def readout(self, h, batch, n_graphs=None, return_weights: bool = False):
        if self.pooling_kind == "sum":
            pooled = global_sum_pool(h, batch, n_graphs)
            weights = None
        elif self.pooling_kind == "mean":
            pooled = global_mean_pool(h, batch, n_graphs)
            weights = None
        else:
            pooled, weights = self.att_pool(h, batch, n_graphs, return_weights=True)
        return (pooled, weights) if return_weights else pooled

    def forward(self, x, edge_index, batch, edge_attr=None, n_graphs=None,
                return_attention: bool = False):
        h = self.node_embeddings(x, edge_index, edge_attr)
        if return_attention:
            pooled, weights = self.readout(h, batch, n_graphs, return_weights=True)
            return self.head(pooled), weights
        pooled = self.readout(h, batch, n_graphs)
        return self.head(pooled)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
