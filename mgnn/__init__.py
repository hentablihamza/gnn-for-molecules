"""mgnn -- teaching implementations of graph neural networks for molecules.

Every layer in this package is written from scratch so that you can read the
arithmetic, and (where an equivalent exists) it is checked numerically against
the PyTorch Geometric implementation in ``tests/``.

Author: Hentabli Hamza, Ph.D.
License: MIT
"""

__version__ = "0.1.0"
__author__ = "Hentabli Hamza"

from mgnn import (  # noqa: F401
    chemnet,
    data,
    features,
    interpret,
    layers,
    models,
    pooling,
    splits,
    train,
    viz,
)

__all__ = [
    "chemnet",
    "data",
    "features",
    "interpret",
    "layers",
    "models",
    "pooling",
    "splits",
    "train",
    "viz",
]
