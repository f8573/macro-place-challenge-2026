"""
Connected-component diagnostics and spectral eigensolve helpers.

These are diagnostic/inspection tools; they are not part of the placement
pipeline and do not affect placer.py behavior.

Requires scipy (listed under [baselines] optional dependencies).
"""

from typing import Tuple

import numpy as np


def connected_components(adj) -> Tuple[int, np.ndarray]:
    """Count connected components and label each node.

    Args:
        adj: scipy.sparse matrix, shape (n, n).  Treated as undirected.

    Returns:
        (n_components, labels) where labels[i] is the 0-based component id
        of node i.

    Raises:
        ImportError: if scipy is not installed.
    """
    try:
        from scipy.sparse.csgraph import connected_components as _cc
    except ImportError as exc:
        raise ImportError(
            "scipy is required for spectral helpers. "
            "Install with: pip install 'macro-place[baselines]'"
        ) from exc

    return _cc(adj, directed=False, return_labels=True)


def spectral_eigenvectors(L, k: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the k smallest eigenvalues and eigenvectors of a graph Laplacian.

    Uses ARPACK eigsh for large sparse matrices and falls back to dense
    scipy.linalg.eigh for small matrices (n <= 200), near-full-rank requests,
    or sparse-solver failures.

    The first returned eigenvector corresponds to the Fiedler direction when the
    graph is connected (eigenvalue near 0).

    Args:
        L: scipy.sparse matrix, symmetric PSD Laplacian of shape (n, n).
        k: number of eigenpairs to compute (clamped to [0, n-1]).

    Returns:
        (eigenvalues, eigenvectors) with shapes (k,) and (n, k).
        Eigenvalues are sorted ascending.

    Raises:
        ImportError: if scipy is not installed.
    """
    try:
        import scipy.linalg
        import scipy.sparse.linalg as spla
    except ImportError as exc:
        raise ImportError(
            "scipy is required for spectral helpers. "
            "Install with: pip install 'macro-place[baselines]'"
        ) from exc

    n = L.shape[0]
    k = min(k, max(n - 1, 0))

    if k == 0 or n == 0:
        return np.zeros(0, dtype=np.float64), np.zeros((n, 0), dtype=np.float64)

    # Zero Laplacian (no edges) - all eigenvalues are 0
    if hasattr(L, "nnz") and L.nnz == 0:
        return np.zeros(k, dtype=np.float64), np.zeros((n, k), dtype=np.float64)

    # Dense path for small graphs or near-full rank requests
    if n <= 200 or k >= n - 1:
        L_dense = L.toarray() if hasattr(L, "toarray") else np.asarray(L)
        vals, vecs = scipy.linalg.eigh(L_dense, subset_by_index=[0, k - 1])
        return vals, vecs

    try:
        vals, vecs = spla.eigsh(L, k=k, which="SM", tol=1e-8)
    except (RuntimeError, spla.ArpackNoConvergence):
        # Diagnostic helpers should not crash on singular/disconnected
        # Laplacians; use a deterministic dense fallback instead.
        L_dense = L.toarray() if hasattr(L, "toarray") else np.asarray(L)
        vals, vecs = scipy.linalg.eigh(L_dense, subset_by_index=[0, k - 1])

    idx = np.argsort(vals)
    return vals[idx], vecs[:, idx]


def compute_spectral_embedding(L, k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Compatibility wrapper for diagnostic spectral eigenpairs."""
    return spectral_eigenvectors(L, k=k)
