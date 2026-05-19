"""
Focused deterministic tests for core/laplacian.py.

All inputs are synthetic scipy.sparse matrices built inline.
scipy is required; tests are skipped if it is absent.
"""

import numpy as np
import pytest

pytest.importorskip("scipy")

from scipy.sparse import csr_matrix

from submissions.solver.core.laplacian import graph_laplacian, normalized_laplacian


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _path_adj(n: int) -> csr_matrix:
    """Unweighted path graph adjacency: 0-1-2-...(n-1)."""
    from scipy.sparse import diags

    data = np.ones(n - 1)
    off = diags(data, offsets=1, shape=(n, n))
    adj = off + off.T
    return adj.tocsr()


def _complete_adj(n: int) -> csr_matrix:
    """Unweighted complete graph adjacency K_n."""
    A = np.ones((n, n)) - np.eye(n)
    return csr_matrix(A)


def _disconnected_adj(n1: int, n2: int) -> csr_matrix:
    """Block-diagonal path graph: two disconnected components."""
    from scipy.sparse import block_diag

    return block_diag([_path_adj(n1), _path_adj(n2)], format="csr")


# ---------------------------------------------------------------------------
# graph_laplacian
# ---------------------------------------------------------------------------


def test_laplacian_shape():
    adj = _path_adj(5)
    L = graph_laplacian(adj)
    assert L.shape == (5, 5)


def test_laplacian_symmetric():
    adj = _path_adj(6)
    L = graph_laplacian(adj)
    diff = (L - L.T).toarray()
    assert np.allclose(diff, 0.0, atol=1e-12)


def test_laplacian_row_sum_zero():
    # Each row of L sums to 0 for any graph Laplacian
    adj = _path_adj(5)
    L = graph_laplacian(adj)
    row_sums = np.asarray(L.sum(axis=1)).ravel()
    assert np.allclose(row_sums, 0.0, atol=1e-12)


def test_laplacian_diagonal_equals_degree():
    # For an unweighted path graph: endpoints have degree 1, interior have 2
    adj = _path_adj(5)
    L = graph_laplacian(adj)
    diag = L.diagonal()
    assert diag[0] == pytest.approx(1.0)
    assert diag[1] == pytest.approx(2.0)
    assert diag[4] == pytest.approx(1.0)


def test_laplacian_psd():
    # All eigenvalues of L should be >= 0
    adj = _path_adj(6)
    L = graph_laplacian(adj)
    vals = np.linalg.eigvalsh(L.toarray())
    assert np.all(vals >= -1e-10)


def test_laplacian_smallest_eigenvalue_zero():
    # Connected graph: smallest eigenvalue = 0 (constant eigenvector)
    adj = _path_adj(5)
    L = graph_laplacian(adj)
    vals = np.linalg.eigvalsh(L.toarray())
    assert vals[0] == pytest.approx(0.0, abs=1e-10)


def test_laplacian_num_zero_eigenvalues_matches_components():
    # Two disconnected components -> two zero eigenvalues
    adj = _disconnected_adj(3, 4)
    L = graph_laplacian(adj)
    vals = np.linalg.eigvalsh(L.toarray())
    n_zero = int((np.abs(vals) < 1e-9).sum())
    assert n_zero == 2


def test_laplacian_complete_graph_k4():
    # K4: all non-zero eigenvalues = n = 4
    adj = _complete_adj(4)
    L = graph_laplacian(adj)
    vals = sorted(np.linalg.eigvalsh(L.toarray()))
    assert vals[0] == pytest.approx(0.0, abs=1e-10)
    assert vals[1] == pytest.approx(4.0, abs=1e-10)
    assert vals[2] == pytest.approx(4.0, abs=1e-10)
    assert vals[3] == pytest.approx(4.0, abs=1e-10)


def test_laplacian_empty_graph():
    # No edges -> L = 0
    adj = csr_matrix((4, 4), dtype=np.float64)
    L = graph_laplacian(adj)
    assert np.allclose(L.toarray(), 0.0)


# ---------------------------------------------------------------------------
# normalized_laplacian
# ---------------------------------------------------------------------------


def test_normalized_laplacian_shape():
    adj = _path_adj(5)
    Ln = normalized_laplacian(adj)
    assert Ln.shape == (5, 5)


def test_normalized_laplacian_symmetric():
    adj = _path_adj(6)
    Ln = normalized_laplacian(adj)
    diff = (Ln - Ln.T).toarray()
    assert np.allclose(diff, 0.0, atol=1e-12)


def test_normalized_laplacian_eigenvalues_in_range():
    # Eigenvalues of L_sym lie in [0, 2]
    adj = _path_adj(8)
    Ln = normalized_laplacian(adj)
    vals = np.linalg.eigvalsh(Ln.toarray())
    assert np.all(vals >= -1e-10)
    assert np.all(vals <= 2.0 + 1e-10)


def test_normalized_laplacian_diagonal_connected():
    # For a connected regular graph (all degrees equal), diagonal entries = 1
    adj = _complete_adj(4)
    Ln = normalized_laplacian(adj)
    diag = Ln.diagonal()
    assert np.allclose(diag, 1.0, atol=1e-12)


def test_normalized_laplacian_isolated_node_zero_row():
    # Isolated node (no edges) -> corresponding row and column are zero
    A = np.zeros((3, 3))
    A[0, 1] = A[1, 0] = 1.0  # node 2 is isolated
    adj = csr_matrix(A)
    Ln = normalized_laplacian(adj)
    arr = Ln.toarray()
    assert np.allclose(arr[2, :], 0.0)
    assert np.allclose(arr[:, 2], 0.0)


def test_normalized_laplacian_psd():
    adj = _path_adj(7)
    Ln = normalized_laplacian(adj)
    vals = np.linalg.eigvalsh(Ln.toarray())
    assert np.all(vals >= -1e-10)
