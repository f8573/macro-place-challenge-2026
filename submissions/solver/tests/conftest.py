"""
Test configuration for submissions/solver.

1. Injects a minimal plc_client_os stub so macro_place can be imported without
   the TILOS submodule (macro_place.objective monkey-patches PlacementCost at
   import time; the stub satisfies that patch).
2. Adds the repo root to sys.path so tests can use canonical package imports
   like `from submissions.solver.core.geometry import ...`.
"""

import sys
import types
from pathlib import Path

# ── plc_client_os stub ────────────────────────────────────────────────────────
# Must be injected BEFORE any macro_place import triggers _plc.py.
# Class name must be PlacementCost so the name-mangled attribute
# _PlacementCost__get_grid_cell_location exists for objective.py's patch.

if "plc_client_os" not in sys.modules:

    class PlacementCost:  # noqa: N801 — name required for mangling
        def __get_grid_cell_location(self, x_pos, y_pos):
            return 0, 0

    _stub = types.ModuleType("plc_client_os")
    _stub.PlacementCost = PlacementCost
    sys.modules["plc_client_os"] = _stub

# ── repo root on path ─────────────────────────────────────────────────────────
# submissions/solver/tests/conftest.py -> solver -> submissions -> repo root

_SOLVER_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _SOLVER_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ── shared benchmark factory (used by M2B and M3A tests) ─────────────────────

import torch  # noqa: E402 — after sys.path is set


def make_benchmark(
    n_hard: int = 4,
    canvas: float = 100.0,
    macro_size: float = 10.0,
    net_nodes=None,
    fixed_mask=None,
    n_soft: int = 0,
    positions=None,
    name: str = "test",
):
    """Create a minimal synthetic benchmark for unit tests."""
    from macro_place.benchmark import Benchmark

    n_total = n_hard + n_soft
    if positions is None:
        base_positions = torch.zeros(n_total, 2, dtype=torch.float32)
        for i in range(n_hard):
            base_positions[i, 0] = (i % 4) * 20.0 + 10.0
            base_positions[i, 1] = (i // 4) * 20.0 + 10.0
        for i in range(n_hard, n_total):
            base_positions[i, 0] = 50.0
            base_positions[i, 1] = 50.0
    else:
        base_positions = torch.tensor(positions, dtype=torch.float32) if not isinstance(positions, torch.Tensor) else positions.float()

    sizes = torch.full((n_total, 2), macro_size, dtype=torch.float32)

    if fixed_mask is None:
        fixed = torch.zeros(n_total, dtype=torch.bool)
    else:
        fixed = torch.tensor(fixed_mask, dtype=torch.bool)

    if net_nodes is None:
        nn = []
        nw = torch.zeros(0)
    else:
        nn = [torch.tensor(ns, dtype=torch.long) for ns in net_nodes]
        nw = torch.ones(len(nn))

    return Benchmark(
        name=name,
        canvas_width=canvas,
        canvas_height=canvas,
        num_macros=n_total,
        num_hard_macros=n_hard,
        num_soft_macros=n_soft,
        macro_positions=base_positions,
        macro_sizes=sizes,
        macro_fixed=fixed,
        macro_names=[f"m{i}" for i in range(n_total)],
        num_nets=len(nn),
        net_nodes=nn,
        net_weights=nw,
        grid_rows=8,
        grid_cols=8,
    )
