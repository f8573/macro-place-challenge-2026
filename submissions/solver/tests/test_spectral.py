"""
Focused deterministic tests for core/spectral.py.

Uses synthetic scipy.sparse matrices - no filesystem access required.
scipy is required; tests are skipped if it is absent.
"""

import numpy as np
import pytest

pytest.importorskip("scipy")

from scipy.sparse import csr_matrix

from submissions.solver.core.laplacian import graph_laplacian, normalized_laplacian
from submissions.solver.core.spectral import connected_components, spectral_eigenvectors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _path_adj(n: int) -> csr_matrix:
    from scipy.sparse import diags

    data = np.ones(n - 1)
    off = diags(data, offsets=1, shape=(n, n))
    return (off + off.T).tocsr()


def _disconnected_adj(n1: int, n2: int) -> csr_matrix:
    from scipy.sparse import block_diag

    return block_diag([_path_adj(n1), _path_adj(n2)], format="csr")


def _isolated_adj(n: int) -> csr_matrix:
    """Graph with n isolated nodes (no edges)."""
    return csr_matrix((n, n), dtype=np.float64)


# ---------------------------------------------------------------------------
# connected_components
# ---------------------------------------------------------------------------


def test_cc_connected_path():
    adj = _path_adj(5)
    n_comp, labels = connected_components(adj)
    assert n_comp == 1
    assert labels.shape == (5,)
    assert len(set(labels.tolist())) == 1


def test_cc_two_components():
    adj = _disconnected_adj(3, 4)
    n_comp, labels = connected_components(adj)
    assert n_comp == 2
    assert labels.shape == (7,)
    assert len(set(labels.tolist())) == 2


def test_cc_three_components():
    from scipy.sparse import block_diag

    adj = block_diag([_path_adj(2), _path_adj(3), _path_adj(2)], format="csr")
    n_comp, labels = connected_components(adj)
    assert n_comp == 3


def test_cc_all_isolated():
    adj = _isolated_adj(4)
    n_comp, labels = connected_components(adj)
    assert n_comp == 4
    assert len(set(labels.tolist())) == 4


def test_cc_single_node():
    adj = csr_matrix((1, 1), dtype=np.float64)
    n_comp, labels = connected_components(adj)
    assert n_comp == 1
    assert labels[0] == 0


def test_cc_labels_non_negative():
    adj = _disconnected_adj(3, 3)
    _, labels = connected_components(adj)
    assert np.all(labels >= 0)


def test_cc_component_sizes_sum_to_n():
    adj = _disconnected_adj(3, 5)
    _, labels = connected_components(adj)
    sizes = np.bincount(labels)
    assert sizes.sum() == 8


# ---------------------------------------------------------------------------
# spectral_eigenvectors
# ---------------------------------------------------------------------------


def test_eigenvectors_shape_k3():
    adj = _path_adj(8)
    L = graph_laplacian(adj)
    vals, vecs = spectral_eigenvectors(L, k=3)
    assert vals.shape == (3,)
    assert vecs.shape == (8, 3)


def test_eigenvectors_sorted_ascending():
    adj = _path_adj(8)
    L = graph_laplacian(adj)
    vals, _ = spectral_eigenvectors(L, k=4)
    assert np.all(np.diff(vals) >= -1e-10)


def test_eigenvectors_smallest_near_zero_connected():
    # Connected graph -> smallest eigenvalue of L is 0
    adj = _path_adj(10)
    L = graph_laplacian(adj)
    vals, _ = spectral_eigenvectors(L, k=2)
    assert vals[0] == pytest.approx(0.0, abs=1e-8)


def test_eigenvectors_two_zeros_disconnected():
    # Disconnected graph -> two eigenvalues near 0
    adj = _disconnected_adj(5, 5)
    L = graph_laplacian(adj)
    vals, _ = spectral_eigenvectors(L, k=3)
    assert vals[0] == pytest.approx(0.0, abs=1e-8)
    assert vals[1] == pytest.approx(0.0, abs=1e-8)
    assert vals[2] > 1e-8


def test_eigenvectors_psd_all_non_negative():
    adj = _path_adj(12)
    L = normalized_laplacian(adj)
    vals, _ = spectral_eigenvectors(L, k=6)
    assert np.all(vals >= -1e-8)


def test_eigenvectors_k_clamped_to_n_minus_1():
    # Requesting k > n-1 should not error - it is clamped
    adj = _path_adj(4)
    L = graph_laplacian(adj)
    vals, vecs = spectral_eigenvectors(L, k=100)
    assert vals.shape[0] <= 3  # n-1 = 3


def test_eigenvectors_empty_k_zero():
    adj = _path_adj(5)
    L = graph_laplacian(adj)
    vals, vecs = spectral_eigenvectors(L, k=0)
    assert vals.shape == (0,)
    assert vecs.shape[0] == 5
    assert vecs.shape[1] == 0


def test_eigenvectors_normalized_laplacian_in_0_2():
    adj = _path_adj(10)
    Ln = normalized_laplacian(adj)
    vals, _ = spectral_eigenvectors(Ln, k=5)
    assert np.all(vals >= -1e-8)
    assert np.all(vals <= 2.0 + 1e-8)


def test_eigenvectors_fiedler_vector_bipartite():
    # For a path graph (bipartite), eigenvectors are known to have structure
    adj = _path_adj(6)
    L = graph_laplacian(adj)
    vals, vecs = spectral_eigenvectors(L, k=2)
    # Fiedler vector (index 1) should be non-constant
    fiedler = vecs[:, 1]
    assert fiedler.max() - fiedler.min() > 1e-6


def test_eigenvectors_single_node_no_error():
    adj = csr_matrix((1, 1), dtype=np.float64)
    L = graph_laplacian(adj)
    vals, vecs = spectral_eigenvectors(L, k=3)
    assert vals.shape[0] == 0  # k clamped to n-1 = 0


def test_eigenvectors_sparse_disconnected_graph_uses_deterministic_path(monkeypatch):
    adj = _disconnected_adj(120, 95)
    L = normalized_laplacian(adj)
    import scipy.sparse.linalg as spla

    def _raise_singular(*args, **kwargs):
        raise RuntimeError("Factor is exactly singular")

    monkeypatch.setattr(spla, "eigsh", _raise_singular)
    vals, vecs = spectral_eigenvectors(L, k=4)
    assert vals.shape == (4,)
    assert vecs.shape == (215, 4)
    assert np.all(np.diff(vals) >= -1e-10)
    assert vals[0] == pytest.approx(0.0, abs=1e-8)
    assert vals[1] == pytest.approx(0.0, abs=1e-8)
