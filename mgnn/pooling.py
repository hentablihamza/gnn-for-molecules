"""Readout: turning a bag of atom vectors into one molecule vector.

Message passing gives you ``[n_atoms, hidden]``.  A property like toxicity is
one number per *molecule*, so something has to collapse the atom axis.  That
step is the readout, and the choice you make there is not cosmetic:

* **mean** -- size-invariant.  Two molecules with the same average atom
  environment look identical, so a dimer looks like its monomer.
* **sum** -- size-*extensive*.  Doubling the molecule doubles the readout,
  which matches how many molecular properties actually behave (molar mass,
  number of toxicophores, lipophilicity contributions).  This is why most
  molecular GNNs sum.
* **attention** -- a learned weighted sum.  Lets a few atoms dominate, which is
  the right inductive bias when one substructure drives the label, and it hands
  you per-atom weights you can draw on the structure (notebook 08).

All three accept a ``batch`` vector in the PyTorch Geometric convention:
``batch[i]`` is the index of the molecule atom ``i`` belongs to.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mgnn.layers import scatter_mean, scatter_softmax, scatter_sum

__all__ = ["global_sum_pool", "global_mean_pool", "AttentionPooling"]


def _dim_size(batch: torch.Tensor, size: int | None) -> int:
    if size is not None:
        return size
    return int(batch.max()) + 1 if batch.numel() else 0


def global_sum_pool(x: torch.Tensor, batch: torch.Tensor,
                    size: int | None = None) -> torch.Tensor:
    """Sum atom vectors per molecule. Matches ``global_add_pool``."""
    return scatter_sum(x, batch, _dim_size(batch, size))


def global_mean_pool(x: torch.Tensor, batch: torch.Tensor,
                     size: int | None = None) -> torch.Tensor:
    """Average atom vectors per molecule. Matches ``global_mean_pool``."""
    return scatter_mean(x, batch, _dim_size(batch, size))


class AttentionPooling(nn.Module):
    r"""Gated readout: :math:`z = \sum_i \alpha_i \, (W h_i)` with
    :math:`\alpha = \mathrm{softmax}_{i \in \text{molecule}}(w^\top h_i)`.

    The softmax runs *within* each molecule, never across the batch -- getting
    that wrong is a classic silent bug that makes a model's predictions depend
    on which other molecules happened to share its batch.
    """

    def __init__(self, in_dim: int, out_dim: int | None = None):
        super().__init__()
        out_dim = out_dim or in_dim
        self.score = nn.Linear(in_dim, 1)
        self.transform = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, batch: torch.Tensor,
                size: int | None = None, return_weights: bool = False):
        n_graphs = _dim_size(batch, size)
        logits = self.score(x)                              # [n_atoms, 1]
        alpha = scatter_softmax(logits, batch, n_graphs)    # [n_atoms, 1]
        pooled = scatter_sum(alpha * self.transform(x), batch, n_graphs)
        if return_weights:
            return pooled, alpha.squeeze(-1)
        return pooled
