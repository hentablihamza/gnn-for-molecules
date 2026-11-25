"""Drawing molecules with per-atom numbers on them.

Everything here returns a matplotlib figure or a PIL image so it renders in a
notebook and can be saved for the README.  The colour convention is fixed
across the whole repository: **red = pushes the prediction up**, **blue =
pushes it down**, colour intensity = magnitude.
"""

from __future__ import annotations

import io

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

__all__ = [
    "draw_molecule_with_atom_weights",
    "draw_molecule_with_indices",
    "draw_graph_schematic",
    "show",
]


def _weights_to_colours(weights: np.ndarray, diverging: bool,
                        alpha: float = 0.85) -> dict[int, tuple]:
    w = np.asarray(weights, dtype=float)
    if diverging:
        scale = np.abs(w).max() or 1.0
        norm = w / scale
        colours = {}
        for i, v in enumerate(norm):
            if v >= 0:   # red for positive contribution
                colours[i] = (1.0, 1.0 - alpha * v, 1.0 - alpha * v)
            else:        # blue for negative
                colours[i] = (1.0 + alpha * v, 1.0 + alpha * v, 1.0)
        return colours
    lo, hi = float(w.min()), float(w.max())
    rng = (hi - lo) or 1.0
    norm = (w - lo) / rng
    return {i: (1.0, 1.0 - alpha * v, 1.0 - alpha * v) for i, v in enumerate(norm)}


def draw_molecule_with_atom_weights(
    mol_or_smiles,
    weights,
    size: tuple[int, int] = (500, 400),
    diverging: bool = True,
    legend: str = "",
    annotate_values: bool = False,
) -> Image.Image:
    """Highlight every atom by its score.

    ``diverging=True`` uses the red/blue signed scale (right for gradient
    attribution); ``False`` uses a one-sided red scale (right for attention
    weights, which are non-negative).
    """
    mol = (Chem.MolFromSmiles(mol_or_smiles)
           if isinstance(mol_or_smiles, str) else mol_or_smiles)
    if mol is None:
        raise ValueError("could not parse molecule")
    weights = np.asarray(weights, dtype=float)
    if len(weights) != mol.GetNumAtoms():
        raise ValueError(
            f"{len(weights)} weights for {mol.GetNumAtoms()} atoms -- these must match "
            "(remember the graph has no explicit hydrogens)"
        )

    colours = _weights_to_colours(weights, diverging)
    mol = Chem.Mol(mol)
    Chem.rdDepictor.Compute2DCoords(mol)
    if annotate_values:
        for atom, w in zip(mol.GetAtoms(), weights, strict=False):
            atom.SetProp("atomNote", f"{w:.2f}")

    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    opts = drawer.drawOptions()
    opts.annotationFontScale = 0.7
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, mol,
        highlightAtoms=list(range(mol.GetNumAtoms())),
        highlightAtomColors=colours,
        highlightBonds=[],
        legend=legend,
    )
    drawer.FinishDrawing()
    return Image.open(io.BytesIO(drawer.GetDrawingText()))


def draw_molecule_with_indices(mol_or_smiles, size: tuple[int, int] = (400, 300),
                               legend: str = "") -> Image.Image:
    """Same molecule, atoms labelled with their graph index.

    Needed constantly: an adjacency matrix is unreadable unless you can see
    which row is which atom.
    """
    mol = (Chem.MolFromSmiles(mol_or_smiles)
           if isinstance(mol_or_smiles, str) else mol_or_smiles)
    mol = Chem.Mol(mol)
    for atom in mol.GetAtoms():
        atom.SetProp("atomNote", str(atom.GetIdx()))
    Chem.rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    drawer.drawOptions().annotationFontScale = 0.9
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol, legend=legend)
    drawer.FinishDrawing()
    return Image.open(io.BytesIO(drawer.GetDrawingText()))


def draw_graph_schematic(graph, node_values=None, title: str = "",
                         figsize: tuple[float, float] = (5.0, 4.0)):
    """Plot the molecule as an abstract node-and-edge diagram.

    Uses a simple spring layout computed with NumPy (no networkx dependency) so
    the reader sees the *graph*, stripped of chemical drawing conventions.
    """
    n = graph.num_nodes
    rng = np.random.default_rng(0)
    pos = rng.normal(scale=0.5, size=(n, 2))
    A = np.zeros((n, n))
    ei = graph.edge_index.numpy()
    A[ei[0], ei[1]] = 1.0

    for _ in range(300):  # Fruchterman-Reingold-ish relaxation
        diff = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(diff, axis=-1) + 1e-9
        repulse = (diff / dist[..., None] ** 2 * 0.05).sum(axis=1)
        attract = -(diff * A[..., None] * 0.08).sum(axis=1)
        pos = pos + repulse + attract

    fig, ax = plt.subplots(figsize=figsize)
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j]:
                ax.plot(*zip(pos[i], pos[j], strict=False), color="0.6", lw=1.2, zorder=1)
    values = np.zeros(n) if node_values is None else np.asarray(node_values, float)
    sc = ax.scatter(pos[:, 0], pos[:, 1], c=values, cmap="coolwarm", s=420,
                    edgecolors="k", zorder=2)
    for i, sym in enumerate(graph.atom_symbols()):
        ax.annotate(f"{sym}{i}", pos[i], ha="center", va="center", fontsize=8, zorder=3)
    if node_values is not None:
        fig.colorbar(sc, ax=ax, shrink=0.8)
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    return fig


def show(img: Image.Image, figsize: tuple[float, float] = (6, 5), title: str = ""):
    """Render a PIL image as a matplotlib figure (so nbconvert captures it)."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig
