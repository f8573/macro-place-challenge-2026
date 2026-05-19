"""
Graph Laplacian construction helpers for macro netlists.

    combinatorial:  L     = D - A
    normalized:     L_sym = D^{-1/2} L D^{-1/2}

Both functions accept and return scipy.sparse.csr_matrix.  Isolated nodes
(degree 0) are handled gracefully: their rows/columns remain zero in L_sym.

Requires scipy (listed under [baselines] optional dependencies).
"""

import numpy as np


def graph_laplacian(adj):
    """Build combinatorial Laplacian L = D - A.

    Args:
        adj: scipy.sparse.csr_matrix, symmetric, shape (n, n).

    Returns:
        scipy.sparse.csr_matrix - symmetric, PSD Laplacian.

    Raises:
        ImportError: if scipy is not installed.
    """
    try:
        from scipy.sparse import diags
    except ImportError as exc:
        raise ImportError(
            "scipy is required for spectral helpers. "
            "Install with: pip install 'macro-place[baselines]'"
        ) from exc

    degrees = np.asarray(adj.sum(axis=1)).ravel()
    D = diags(degrees, format="csr")
    return D - adj


def normalized_laplacian(adj):
    """Build symmetric normalized Laplacian L_sym = D^{-1/2} (D - A) D^{-1/2}.

    Eigenvalues lie in [0, 2].  For isolated nodes (degree 0), D^{-1/2} is
    set to 0, leaving those rows/columns as zero vectors.

    Args:
        adj: scipy.sparse.csr_matrix, symmetric, shape (n, n).

    Returns:
        scipy.sparse.csr_matrix - symmetric PSD matrix.

    Raises:
        ImportError: if scipy is not installed.
    """
    try:
        from scipy.sparse import diags
    except ImportError as exc:
        raise ImportError(
            "scipy is required for spectral helpers. "
            "Install with: pip install 'macro-place[baselines]'"
        ) from exc

    degrees = np.asarray(adj.sum(axis=1)).ravel()
    safe = np.where(degrees > 0, degrees, 1.0)
    d_inv_sqrt = np.where(degrees > 0, 1.0 / np.sqrt(safe), 0.0)
    D_inv_sqrt = diags(d_inv_sqrt, format="csr")
    L = graph_laplacian(adj)
    return D_inv_sqrt @ L @ D_inv_sqrt
