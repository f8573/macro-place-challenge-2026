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
