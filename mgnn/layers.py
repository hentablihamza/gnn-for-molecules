"""Graph neural network layers written from scratch.

Each class here is a readable re-implementation of a published layer.  The
parameter names mirror PyTorch Geometric so that ``tests/test_layers_match_pyg.py``
can copy weights across and assert the two produce the *same numbers*.

Layers
------
``GCNLayer``       Kipf & Welling (2017) graph convolution, dense normalised adjacency.
``SAGELayer``      Hamilton, Ying & Leskovec (2017), mean aggregator.
``GATLayer``       Velickovic et al. (2018) attention, multi-head.
``GINLayer``       Xu et al. (2019) sum aggregation + MLP.
``DMPNNLayer``     Yang et al. (2019) directed edge-centred message passing (Chemprop).

Teaching conventions
--------------------
* ``edge_index[0]`` is the **source** (the node a message comes *from*) and
  ``edge_index[1]`` is the **target** (the node it arrives *at*).
* The GCN layer is written with a dense adjacency matrix because that is how
  the equation is written on paper.  The others use scatter-style indexing,
  which is how production code works.  Both are shown on purpose.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "GCNLayer",
    "SAGELayer",
    "GATLayer",
    "GINLayer",
    "DMPNNLayer",
    "gcn_normalised_adjacency",
    "scatter_sum",
    "scatter_mean",
    "message_passing_step_numpy",
]


# --------------------------------------------------------------------------
# Small primitives
# --------------------------------------------------------------------------

def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Sum rows of ``src`` into bucket ``index[k]``; empty buckets stay zero.

    This is the whole of "aggregation" in message passing.  Everything else is
    linear algebra around it.
    """
    out_shape = (dim_size,) + tuple(src.shape[1:])
    out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
    if src.numel() == 0:
        return out
    idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    return out.scatter_add_(0, idx, src)


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Mean over each bucket; buckets that receive nothing are zero (as in PyG)."""
    summed = scatter_sum(src, index, dim_size)
    counts = scatter_sum(
        torch.ones(src.shape[0], 1, dtype=src.dtype, device=src.device), index, dim_size
    )
    return summed / counts.clamp(min=1.0)


def scatter_softmax(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Softmax computed *within* each target node's incoming-edge group.

    ``src`` is ``[n_edges, heads]``.  Max-subtraction keeps it numerically safe,
    exactly as PyG's ``utils.softmax`` does.
    """
    if src.numel() == 0:
        return src
    out = torch.full((dim_size,) + tuple(src.shape[1:]), float("-inf"),
                     dtype=src.dtype, device=src.device)
    idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    maxes = out.scatter_reduce(0, idx, src, reduce="amax", include_self=True)
    maxes = torch.nan_to_num(maxes, neginf=0.0)
    exp = (src - maxes[index]).exp()
    denom = scatter_sum(exp, index, dim_size)[index]
    return exp / denom.clamp(min=1e-16)


def add_self_loops(edge_index: torch.Tensor, num_nodes: int,
                   remove_existing: bool = True) -> torch.Tensor:
    """Append ``i -> i`` for every node, the way GCN and GAT both require.

    ``remove_existing=True`` drops any self-loop already present before adding
    a fresh set, so a node never ends up with two.  That is exactly what
    ``GATConv`` does internally (``remove_self_loops`` then ``add_self_loops``),
    and matching it is what lets the tests compare numbers edge-for-edge.
    """
    ei = edge_index
    if remove_existing and ei.numel():
        keep = ei[0] != ei[1]
        ei = ei[:, keep]
    loop = torch.arange(num_nodes, device=edge_index.device)
    loops = torch.stack([loop, loop], dim=0)
    return torch.cat([ei, loops], dim=1)


def gcn_normalised_adjacency(
    edge_index: torch.Tensor, num_nodes: int, add_loops: bool = True
) -> torch.Tensor:
    r"""Build :math:`\hat{A} = \tilde{D}^{-1/2}(A + I)\tilde{D}^{-1/2}` densely.

    Two ideas live in this one line, and notebook 03 pulls them apart:

    * **self-loops** (``+ I``) let a node keep its own features; without them a
      layer replaces an atom by its neighbourhood and forgets the atom.
    A dense ``A`` cannot represent a repeated edge, which is fine for molecular
    graphs built by :func:`mgnn.features.mol_to_graph` (one bond, one pair of
    directed edges, no duplicates, no self-loops) but is worth knowing if you
    feed this function an arbitrary edge list.

    * **symmetric degree normalisation** stops high-degree atoms (a quaternary
      carbon) from producing activations an order of magnitude larger than a
      terminal methyl, which would otherwise swamp training.
    """
    # Row = receiving atom, column = sending atom, so that ``A_hat @ XW`` reads
    # left to right as "each atom's new features are a weighted sum of what
    # arrives".  For a molecule the adjacency is symmetric and the choice does
    # not matter; for a general directed edge list it does, and this convention
    # is the one PyTorch Geometric's ``gcn_norm`` implements (it normalises by
    # in-degree).
    A = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
    if edge_index.numel():
        A[edge_index[1], edge_index[0]] = 1.0
    if add_loops:
        # Set (not add) the diagonal: a node that already carried a self-loop
        # keeps exactly one, which mirrors PyG's ``add_remaining_self_loops``.
        A.fill_diagonal_(1.0)
    deg = A.sum(dim=1)                     # how many messages each atom receives
    dinv = deg.pow(-0.5)
    dinv[torch.isinf(dinv)] = 0.0
    return dinv.view(-1, 1) * A * dinv.view(1, -1)


# --------------------------------------------------------------------------
# GCN -- Kipf & Welling, ICLR 2017
# --------------------------------------------------------------------------

class GCNLayer(nn.Module):
    r"""Graph convolution: :math:`H' = \hat{A} H W + b`.

    Written with the dense normalised adjacency so the notebook can print
    :math:`\hat{A}` and multiply by hand.  Mathematically identical to
    ``torch_geometric.nn.GCNConv`` with its default settings.
    """

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True,
                 add_self_loops: bool = True):
        super().__init__()
        self.in_dim, self.out_dim = in_dim, out_dim
        self.add_self_loops = add_self_loops
        # bias lives outside `lin` so the parameter layout matches GCNConv
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        A_hat = gcn_normalised_adjacency(edge_index, x.shape[0], self.add_self_loops)
        out = A_hat @ self.lin(x)
        if self.bias is not None:
            out = out + self.bias
        return out

    @torch.no_grad()
    def copy_weights_from_pyg(self, conv) -> GCNLayer:
        """Copy parameters out of a ``GCNConv`` so outputs can be compared."""
        self.lin.weight.copy_(conv.lin.weight)
        if self.bias is not None and conv.bias is not None:
            self.bias.copy_(conv.bias)
        return self


# --------------------------------------------------------------------------
# GraphSAGE -- Hamilton, Ying & Leskovec, NeurIPS 2017
# --------------------------------------------------------------------------

class SAGELayer(nn.Module):
    r"""Mean aggregator: :math:`h'_i = W_l \cdot \mathrm{mean}_{j \in N(i)} h_j + W_r h_i`.

    Note what is different from GCN: the node's own features go through a
    *separate* weight matrix (the "root weight") instead of being mixed into the
    neighbourhood by a self-loop.  The layer can therefore learn to treat
    "me" and "my neighbours" differently -- useful when the centre atom's
    identity matters more than its environment.

    There is no degree normalisation beyond the mean, so this layer is blind to
    how many neighbours an atom has.  Compare with :class:`GINLayer`.
    """

    def __init__(self, in_dim: int, out_dim: int, root_weight: bool = True,
                 bias: bool = True):
        super().__init__()
        self.lin_l = nn.Linear(in_dim, out_dim, bias=bias)   # neighbourhood
        self.lin_r = nn.Linear(in_dim, out_dim, bias=False) if root_weight else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.lin_l.reset_parameters()
        if self.lin_r is not None:
            self.lin_r.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        messages = x[src]                                   # what each neighbour sends
        agg = scatter_mean(messages, dst, x.shape[0])       # average per target atom
        out = self.lin_l(agg)
        if self.lin_r is not None:
            out = out + self.lin_r(x)
        return out

    @torch.no_grad()
    def copy_weights_from_pyg(self, conv) -> SAGELayer:
        self.lin_l.weight.copy_(conv.lin_l.weight)
        if conv.lin_l.bias is not None and self.lin_l.bias is not None:
            self.lin_l.bias.copy_(conv.lin_l.bias)
        if self.lin_r is not None:
            self.lin_r.weight.copy_(conv.lin_r.weight)
        return self


# --------------------------------------------------------------------------
# GAT -- Velickovic et al., ICLR 2018
# --------------------------------------------------------------------------

class GATLayer(nn.Module):
    r"""Attention-weighted aggregation.

    For the directed edge :math:`j \rightarrow i`

    .. math::
        e_{ij} = \mathrm{LeakyReLU}(a_{src}^\top W h_j + a_{dst}^\top W h_i),
        \quad \alpha_{ij} = \mathrm{softmax}_{j \in N(i) \cup \{i\}}(e_{ij})

    and :math:`h'_i = \sum_j \alpha_{ij} W h_j`.

    The attention coefficients are the reason this layer is the one to reach for
    when you want to *show a chemist* which atoms the model listened to:
    :meth:`forward` can hand them back (``return_attention=True``) and notebooks
    04 and 08 paint them straight onto the structure.
    """

    def __init__(self, in_dim: int, out_dim: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, bias: bool = True,
                 add_self_loops: bool = True):
        super().__init__()
        self.in_dim, self.out_dim, self.heads = in_dim, out_dim, heads
        self.concat, self.negative_slope = concat, negative_slope
        self.add_self_loops = add_self_loops

        self.lin = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(1, heads, out_dim))
        self.att_dst = nn.Parameter(torch.empty(1, heads, out_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(heads * out_dim if concat else out_dim))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                return_attention: bool = False):
        n = x.shape[0]
        ei = (add_self_loops(edge_index, n, remove_existing=True)
              if self.add_self_loops else edge_index)
        src, dst = ei[0], ei[1]

        h = self.lin(x).view(n, self.heads, self.out_dim)       # [N, H, F']
        alpha_src = (h * self.att_src).sum(-1)                  # [N, H]
        alpha_dst = (h * self.att_dst).sum(-1)                  # [N, H]

        e = F.leaky_relu(alpha_src[src] + alpha_dst[dst], self.negative_slope)
        alpha = scatter_softmax(e, dst, n)                      # [E, H]

        msg = h[src] * alpha.unsqueeze(-1)                      # [E, H, F']
        out = scatter_sum(msg, dst, n)                          # [N, H, F']

        out = out.reshape(n, self.heads * self.out_dim) if self.concat else out.mean(dim=1)
        if self.bias is not None:
            out = out + self.bias
        if return_attention:
            return out, (ei, alpha)
        return out

    @torch.no_grad()
    def copy_weights_from_pyg(self, conv) -> GATLayer:
        self.lin.weight.copy_(conv.lin.weight)
        self.att_src.copy_(conv.att_src)
        self.att_dst.copy_(conv.att_dst)
        if self.bias is not None and conv.bias is not None:
            self.bias.copy_(conv.bias)
        return self


# --------------------------------------------------------------------------
# GIN -- Xu et al., ICLR 2019
# --------------------------------------------------------------------------

class GINLayer(nn.Module):
    r""":math:`h'_i = \mathrm{MLP}\big((1 + \epsilon) h_i + \sum_{j \in N(i)} h_j\big)`.

    **Sum**, not mean: a nitrogen with three neighbours and a nitrogen with one
    produce different aggregates even if the neighbours are identical, which is
    what gives GIN its discriminative power (it is as strong as the
    1-Weisfeiler-Lehman test).  Mean aggregation throws that count away.
    """

    def __init__(self, mlp: nn.Module, eps: float = 0.0, train_eps: bool = False):
        super().__init__()
        self.mlp = mlp
        self.initial_eps = eps
        if train_eps:
            self.eps = nn.Parameter(torch.tensor([eps]))
        else:
            self.register_buffer("eps", torch.tensor([eps]))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        agg = scatter_sum(x[src], dst, x.shape[0])
        return self.mlp((1.0 + self.eps) * x + agg)


# --------------------------------------------------------------------------
# D-MPNN -- Yang et al., J. Chem. Inf. Model. 2019 (Chemprop)
# --------------------------------------------------------------------------

class DMPNNLayer(nn.Module):
    r"""Directed, **edge-centred** message passing.

    The hidden state lives on the directed bond :math:`(v \rightarrow w)` rather
    than on the atom.  The update for that bond sums the incoming bonds of
    :math:`v` while *excluding the reverse bond* :math:`(w \rightarrow v)`:

    .. math::
        m_{vw}^{t+1} = \sum_{u \in N(v) \setminus \{w\}} h_{uv}^{t},
        \qquad h_{vw}^{t+1} = \mathrm{ReLU}(h_{vw}^{0} + W_h m_{vw}^{t+1})

    Dropping the reverse bond is the whole point: it stops a feature from
    bouncing back and forth across one bond, which in an atom-centred network
    makes a two-layer model partly an echo of its own first layer.  Chemprop
    reports this helps most on small molecular data sets.

    There is no exact PyTorch Geometric equivalent of this layer, so the test
    suite checks it against an independent, deliberately naive reference
    implementation (:func:`dmpnn_reference_forward`) and against the structural
    invariants (atom-order permutation invariance, no self-echo).
    """

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, depth: int = 3,
                 bias: bool = False):
        super().__init__()
        self.depth = depth
        self.hidden_dim = hidden_dim
        self.W_i = nn.Linear(node_dim + edge_dim, hidden_dim, bias=bias)
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.W_o = nn.Linear(node_dim + hidden_dim, hidden_dim, bias=True)

    @staticmethod
    def reverse_edge_index(edge_index: torch.Tensor) -> torch.Tensor:
        """Position of the reverse partner of every directed edge.

        :func:`mgnn.features.mol_to_graph` emits bonds in ``(i->j, j->i)`` pairs,
        so the partner of edge ``2k`` is ``2k+1`` -- but we find it by lookup so
        the layer also works on edge lists built elsewhere.
        """
        n_e = edge_index.shape[1]
        lookup = {(int(edge_index[0, k]), int(edge_index[1, k])): k for k in range(n_e)}
        rev = torch.full((n_e,), -1, dtype=torch.long)
        for k in range(n_e):
            key = (int(edge_index[1, k]), int(edge_index[0, k]))
            if key in lookup:
                rev[k] = lookup[key]
        return rev

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        src, dst = edge_index[0], edge_index[1]
        n_e = edge_index.shape[1]

        if n_e == 0:  # lone ion: no bonds, so the atom only sees itself
            zeros = torch.zeros(n, self.hidden_dim, dtype=x.dtype)
            return F.relu(self.W_o(torch.cat([x, zeros], dim=1)))

        rev = self.reverse_edge_index(edge_index).to(x.device)
        h0 = F.relu(self.W_i(torch.cat([x[src], edge_attr], dim=1)))  # [E, hidden]
        h = h0

        for _ in range(self.depth - 1):
            # sum of bonds arriving at the *source* atom of this bond ...
            agg_at_src = scatter_sum(h, dst, n)[src]
            # ... minus the message coming straight back down this same bond
            back = torch.where((rev >= 0).unsqueeze(-1), h[rev.clamp(min=0)],
                               torch.zeros_like(h))
            m = agg_at_src - back
            h = F.relu(h0 + self.W_h(m))

        node_msg = scatter_sum(h, dst, n)                 # bonds into each atom
        return F.relu(self.W_o(torch.cat([x, node_msg], dim=1)))


def dmpnn_reference_forward(layer: DMPNNLayer, x: torch.Tensor,
                            edge_index: torch.Tensor,
                            edge_attr: torch.Tensor) -> torch.Tensor:
    """Independent slow reference for :class:`DMPNNLayer`, written with loops.

    Used only by the tests.  Deliberately written from the equations in the
    paper rather than by refactoring the vectorised code, so that agreement
    between the two is real evidence and not a tautology.
    """
    n = x.shape[0]
    n_e = edge_index.shape[1]
    if n_e == 0:
        zeros = torch.zeros(n, layer.hidden_dim, dtype=x.dtype)
        return F.relu(layer.W_o(torch.cat([x, zeros], dim=1)))

    edges = [(int(edge_index[0, k]), int(edge_index[1, k])) for k in range(n_e)]
    h0 = [F.relu(layer.W_i(torch.cat([x[v], edge_attr[k]]))) for k, (v, w) in enumerate(edges)]
    h = list(h0)
    for _ in range(layer.depth - 1):
        new_h = []
        for k, (v, w) in enumerate(edges):
            m = torch.zeros(layer.hidden_dim, dtype=x.dtype)
            for k2, (u, v2) in enumerate(edges):
                if v2 == v and u != w:          # bonds into v, excluding w->v
                    m = m + h[k2]
            new_h.append(F.relu(h0[k] + layer.W_h(m)))
        h = new_h

    out = []
    for v in range(n):
        m = torch.zeros(layer.hidden_dim, dtype=x.dtype)
        for k, (_a, b) in enumerate(edges):
            if b == v:
                m = m + h[k]
        out.append(F.relu(layer.W_o(torch.cat([x[v], m]))))
    return torch.stack(out)


# --------------------------------------------------------------------------
# The hand-traced step used by notebook 02
# --------------------------------------------------------------------------

def message_passing_step_numpy(
    X: np.ndarray, A: np.ndarray, W: np.ndarray, b: np.ndarray | None = None,
    normalise: bool = True,
) -> dict[str, np.ndarray]:
    """One graph-convolution step in NumPy, returning *every* intermediate.

    Returned keys: ``A_tilde`` (with self-loops), ``degree``, ``A_hat``,
    ``XW`` (transformed features), ``aggregated`` and ``out``.  Notebook 02
    prints these one by one so the reader can watch a feature vector leave one
    atom and arrive at its neighbour.
    """
    n = X.shape[0]
    A_tilde = A + np.eye(n)
    degree = A_tilde.sum(axis=1)
    if normalise:
        dinv = np.where(degree > 0, degree ** -0.5, 0.0)
        A_hat = dinv[:, None] * A_tilde * dinv[None, :]
    else:
        A_hat = A_tilde
    XW = X @ W
    aggregated = A_hat @ XW
    out = aggregated if b is None else aggregated + b
    return {
        "A_tilde": A_tilde,
        "degree": degree,
        "A_hat": A_hat,
        "XW": XW,
        "aggregated": aggregated,
        "out": out,
    }
