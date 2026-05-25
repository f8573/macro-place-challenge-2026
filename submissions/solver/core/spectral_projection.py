"""
Spectral projection: convert a spectral embedding into macro center positions.

Steps:
  1. Build the macro-net clique adjacency (reuses core/hypergraph.py).
  2. Build the normalized Laplacian (reuses core/laplacian.py).
  3. Compute the 2nd and 3rd smallest eigenvectors (first two nontrivial).
  4. Normalize the embedding to the usable die area.
  5. Return center coordinates clamped to canvas bounds.

Isolated nodes (no edges in the clique graph) are placed at the die center.

Failure modes are caught and logged; callers receive an empty list on failure.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from macro_place.benchmark import Benchmark


def _build_spectral_embedding(
    benchmark: Benchmark,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return (xs, ys) normalized 2D spectral embedding, or None on failure."""
    try:
        from submissions.solver.core.hypergraph import clique_adjacency
        from submissions.solver.core.laplacian import normalized_laplacian
        from submissions.solver.core.spectral import spectral_eigenvectors
    except ImportError:
        return None

    try:
        adj = clique_adjacency(benchmark)
        n = benchmark.num_hard_macros

        if n < 3:
            return None

        L = normalized_laplacian(adj)

        # Request 3 eigenvectors (index 0 = constant, 1 = Fiedler, 2 = second)
        k = min(3, n - 1)
        if k < 2:
            return None

        vals, vecs = spectral_eigenvectors(L, k=k)

        if vecs.shape[1] < 2:
            return None

        # Skip the constant eigenvector; use indices 1 and 2 if available
        # For connected graphs, vals[0] ≈ 0, vecs[:,0] ≈ const
        # For disconnected graphs, there may be multiple zero eigenvalues
        # Use the two eigenvectors with the largest variance as embedding dims
        variances = [vecs[:, j].var() for j in range(vecs.shape[1])]
        sorted_cols = sorted(range(len(variances)), key=lambda j: -variances[j])

        if len(sorted_cols) < 2:
            return None

        xs = vecs[:, sorted_cols[0]]
        ys = vecs[:, sorted_cols[1]]

        # Check for degenerate embedding
        if np.ptp(xs) < 1e-10 or np.ptp(ys) < 1e-10:
            return None

        return xs, ys

    except Exception:
        return None


def _apply_transform(
    xs: np.ndarray, ys: np.ndarray, transform: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply a named geometric transform to (xs, ys) in [-1, 1]^2 space."""
    xs = xs.copy()
    ys = ys.copy()
    if transform == "identity":
        pass
    elif transform == "flip_x":
        xs = -xs
    elif transform == "flip_y":
        ys = -ys
    elif transform == "flip_xy":
        xs = -xs
        ys = -ys
    elif transform == "swap_xy":
        xs, ys = ys, xs
    elif transform == "swap_flip_x":
        xs, ys = ys, xs
        xs = -xs
    elif transform == "center_scale_085":
        xs = xs * 0.85
        ys = ys * 0.85
    elif transform == "center_scale_070":
        xs = xs * 0.70
        ys = ys * 0.70
    return xs, ys


def _embedding_to_centers(
    xs: np.ndarray,
    ys: np.ndarray,
    canvas_w: float,
    canvas_h: float,
    widths: np.ndarray,
    heights: np.ndarray,
) -> torch.Tensor:
    """Normalize 1D embedding arrays to canvas center positions.

    Returns [n_hard, 2] float32 tensor of center coordinates, clamped to canvas.
    """
    n = len(xs)

    # Normalize to [0, 1] with 5% margin
    margin = 0.05
    lo, hi = margin, 1.0 - margin

    def norm(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-10:
            return np.full_like(arr, 0.5)
        return lo + (arr - mn) / (mx - mn) * (hi - lo)

    xs_n = norm(xs)
    ys_n = norm(ys)

    centers = torch.zeros(n, 2, dtype=torch.float32)
    for i in range(n):
        cx = float(xs_n[i]) * canvas_w
        cy = float(ys_n[i]) * canvas_h
        # Clamp to keep macro inside canvas
        cx = max(float(widths[i]) / 2.0, min(canvas_w - float(widths[i]) / 2.0, cx))
        cy = max(float(heights[i]) / 2.0, min(canvas_h - float(heights[i]) / 2.0, cy))
        centers[i, 0] = cx
        centers[i, 1] = cy

    return centers


def generate_spectral_candidates(
    benchmark: Benchmark,
) -> List["CandidatePlacement"]:
    """Generate spectral projection candidates.

    Returns a list of CandidatePlacement objects. Returns [] on failure.
    Failure never propagates as an exception.
    """
    # Import here to avoid circular imports
    from submissions.solver.core.candidate_types import CandidatePlacement

    result = _build_spectral_embedding(benchmark)
    if result is None:
        return []

    xs_base, ys_base = result
    canvas_w = benchmark.canvas_width
    canvas_h = benchmark.canvas_height
    n_hard = benchmark.num_hard_macros
    widths = benchmark.macro_sizes[:n_hard, 0].numpy()
    heights = benchmark.macro_sizes[:n_hard, 1].numpy()

    # Normalize base embedding to [-1, 1]
    def to_pm1(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-10:
            return np.zeros_like(arr)
        return 2.0 * (arr - mn) / (mx - mn) - 1.0

    xs_n = to_pm1(xs_base)
    ys_n = to_pm1(ys_base)

    transforms = [
        ("spectral_xy", "identity"),
        ("spectral_flip_x", "flip_x"),
        ("spectral_flip_y", "flip_y"),
        ("spectral_flip_xy", "flip_xy"),
        ("spectral_swap_xy", "swap_xy"),
        ("spectral_swap_flip_x", "swap_flip_x"),
        ("spectral_center_scale_085", "center_scale_085"),
        ("spectral_center_scale_070", "center_scale_070"),
    ]

    candidates = []
    for name, transform in transforms:
        try:
            txs, tys = _apply_transform(xs_n, ys_n, transform)
            centers_hard = _embedding_to_centers(
                txs, tys, canvas_w, canvas_h, widths, heights
            )
            # Extend to full macro count (keep soft macros at original positions)
            positions = benchmark.macro_positions.clone().float()
            positions[:n_hard] = centers_hard
            candidates.append(
                CandidatePlacement(
                    name=name,
                    family="spectral",
                    positions=positions,
                )
            )
        except Exception as exc:
            # Non-fatal: skip this variant
            _ = exc

    return candidates
