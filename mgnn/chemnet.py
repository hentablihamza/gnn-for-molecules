r"""A modern reconstruction of the ChemNet idea (Kireev, 1995).

Citation
--------
Kireev, D. B. "ChemNet: A Novel Neural Network Based Method for Graph/Property
Mapping." *J. Chem. Inf. Comput. Sci.* **1995**, *35* (2), 175-180.
DOI: `10.1021/ci00024a001 <https://doi.org/10.1021/ci00024a001>`_.

What ChemNet did
----------------
ChemNet was **among the first** neural networks to map a molecular graph
*directly* into the architecture of the network itself, rather than first
compressing the molecule into a fixed vector of descriptors.  Each non-hydrogen
atom got its own neuron, and the connection weights between neurons were tied
to the **topological distance** between the corresponding atoms: every pair of
atoms three bonds apart shared one weight, every pair two bonds apart shared
another, and so on.  Learning therefore happened over a handful of
distance-indexed weights that were reused across every molecule, which is
precisely the weight-sharing argument that modern graph convolutions rest on.

The University of Missouri Chemistry department page for Prof. Kireev describes
ChemNet as "the first published instance of a graph-convolutional network
architecture" (https://chemistry.missouri.edu/people/kireev).  Kireev himself
has put it more cautiously in public, writing in 2025 that it "might have been
among the first", and describing how in a single layer messages were passed to
a node from all other nodes, in a Transformer-like fashion.  This module follows
his framing.

Contemporaneous related work
----------------------------
* Baskin, I. I.; Palyulin, V. A.; Zefirov, N. S. "A Neural Device for Searching
  Direct Correlations between Structures and Properties of Chemical Compounds."
  *J. Chem. Inf. Comput. Sci.* **1997**, *37* (4), 715-721.
  DOI: 10.1021/ci940128y.
* Ivanciuc, O. MolNet, a multilayer perceptron whose topology encodes the
  molecular graph (input-hidden connections encode the topological distance
  matrix, hidden-output connections are grouped by atom type); reviewed in
  Ivanciuc, O. "Quantitative Structure-Activity Relationships (QSAR) with the
  MolNet Molecular Graph Machine." *Curr. Comput.-Aided Drug Des.* **2011**,
  *6* (2), 261-268. DOI: 10.2174/1574893611106020261.

Ivanciuc's review chapter "Molecular Graph Descriptors Used in Neural Network
Models" (in *Topological Indices and Related Descriptors in QSAR and QSPR*,
Gordon & Breach, 1999, pp. 697-777) discusses all three architectures -- the
Baskin-Palyulin-Zefirov neural device, Kireev's ChemNet, and MolNet -- together
as networks generated from the molecular graph.

What this module gives you
--------------------------
:class:`ChemNetLayer` is the idea in modern clothes: a node is updated from
**all** other nodes at once, with the contribution of each pair weighted by a
learned, topological-distance-indexed weight matrix.  Notebook 09 trains it and
shows two things:

1. a one-hop GCN layer is the special case where every weight for distance
   greater than one is set to zero, and
2. the "attend to everything, bias the attention by graph distance" pattern is
   what Graphormer-style graph transformers rediscovered two decades later.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem

from mgnn.features import ATOM_FEATURE_DIM
from mgnn.pooling import global_sum_pool

__all__ = [
    "topological_distance_matrix",
    "distance_buckets",
    "ChemNetLayer",
    "ChemNetModel",
]

UNREACHABLE = -1


def topological_distance_matrix(mol_or_smiles) -> np.ndarray:
    """Shortest-path bond counts between every pair of heavy atoms.

    This is the matrix ChemNet's weight sharing was indexed by.  RDKit returns
    a very large number (1e8) for atoms in different fragments (salts); those
    are mapped to :data:`UNREACHABLE` so they can get their own weight.
    """
    mol = (Chem.MolFromSmiles(mol_or_smiles)
           if isinstance(mol_or_smiles, str) else mol_or_smiles)
    if mol is None:
        raise ValueError("could not parse molecule")
    D = Chem.GetDistanceMatrix(mol)
    D = np.where(D > 1e6, UNREACHABLE, D)
    return D.astype(np.int64)


def distance_buckets(D: np.ndarray, max_distance: int) -> torch.Tensor:
    """Map a distance matrix to weight indices ``0..max_distance+1``.

    Bucket ``d`` is "exactly ``d`` bonds apart" for ``d <= max_distance``;
    bucket ``max_distance + 1`` catches both "further than that" and
    "different fragment".  Capping is what keeps the parameter count fixed when
    molecules differ in size -- the same trick as a relative-position bias.
    """
    idx = np.where((D < 0) | (D > max_distance), max_distance + 1, D)
    return torch.as_tensor(idx, dtype=torch.long)


class ChemNetLayer(nn.Module):
    r"""All-pairs message passing with distance-tied weights.

    .. math::
        h'_i = b + \sum_{d=0}^{D} \; \sum_{j \,:\, \mathrm{dist}(i,j)=d} W_d \, h_j

    Every atom receives a message from every other atom in one layer -- no
    multi-hop stacking required -- but *which* weight matrix carries that
    message depends only on how many bonds apart the two atoms are.  Note what
    the two limits give you:

    * keep only :math:`W_0` and :math:`W_1`, tie them, and add degree
      normalisation and you have recovered a GCN layer;
    * replace the fixed :math:`W_d` by a learned attention score that is
      *biased* by :math:`d` and you have a graph transformer.

    ``mean_normalise=True`` divides each distance shell by how many atoms are in
    it, which the 1995 formulation did not do and which makes the layer far
    easier to train on molecules of mixed size.
    """

    def __init__(self, in_dim: int, out_dim: int, max_distance: int = 5,
                 bias: bool = True, mean_normalise: bool = True):
        super().__init__()
        self.in_dim, self.out_dim = in_dim, out_dim
        self.max_distance = max_distance
        self.n_buckets = max_distance + 2        # 0..max_distance, plus "far/other"
        self.mean_normalise = mean_normalise
        # One weight matrix per distance shell: the weight sharing is the method.
        self.weight = nn.Parameter(torch.empty(self.n_buckets, in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for d in range(self.n_buckets):
            nn.init.xavier_uniform_(self.weight[d])

    def forward(self, x: torch.Tensor, dist_bucket: torch.Tensor) -> torch.Tensor:
        """``x`` is ``[N, in_dim]``; ``dist_bucket`` is ``[N, N]`` from
        :func:`distance_buckets` (one molecule at a time, which keeps the
        notebook honest about the ``N**2`` cost)."""
        n = x.shape[0]
        out = x.new_zeros(n, self.out_dim)
        for d in range(self.n_buckets):
            shell = (dist_bucket == d).to(x.dtype)      # [N, N] indicator
            if shell.sum() == 0:
                continue
            if self.mean_normalise:
                shell = shell / shell.sum(dim=1, keepdim=True).clamp(min=1.0)
            out = out + shell @ (x @ self.weight[d])
        if self.bias is not None:
            out = out + self.bias
        return out


class ChemNetModel(nn.Module):
    """Small ChemNet-style property predictor used by notebook 09.

    Deliberately tiny: the point is to show the architecture works and where it
    sits in the lineage, not to compete with notebook 07's model.
    """

    def __init__(self, n_tasks: int = 1, hidden_dim: int = 32, n_layers: int = 2,
                 max_distance: int = 5, node_dim: int = ATOM_FEATURE_DIM,
                 dropout: float = 0.1):
        super().__init__()
        dims = [node_dim] + [hidden_dim] * n_layers
        self.layers = nn.ModuleList(
            [ChemNetLayer(dims[i], dims[i + 1], max_distance=max_distance)
             for i in range(n_layers)]
        )
        self.dropout = dropout
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, n_tasks),
        )

    def node_embeddings(self, x: torch.Tensor, dist_bucket: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = F.relu(layer(h, dist_bucket))
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def forward_single(self, x: torch.Tensor, dist_bucket: torch.Tensor) -> torch.Tensor:
        """One molecule in, one prediction out."""
        h = self.node_embeddings(x, dist_bucket)
        batch = torch.zeros(h.shape[0], dtype=torch.long)
        return self.head(global_sum_pool(h, batch, 1))

    def forward(self, xs: list[torch.Tensor],
                buckets: list[torch.Tensor]) -> torch.Tensor:
        """A "batch" is just a loop here -- all-pairs attention per molecule is
        the cost ChemNet paid for seeing everything in one layer."""
        return torch.cat([self.forward_single(x, b) for x, b in zip(xs, buckets, strict=False)], dim=0)
