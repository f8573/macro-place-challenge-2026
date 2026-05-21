"""
SolverPlacer — Milestone 2B.

Generates deterministic candidate placements, legalizes each, scores
valid candidates using proxy cost (or HPWL when plc is unavailable),
and returns the valid candidate with the lowest cost.

The original placement is always included as the first candidate and
serves as the fallback if all generated candidates are invalid.
"""

import sys
from pathlib import Path

import torch
from macro_place.benchmark import Benchmark

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))


class SolverPlacer:
    """M2B candidate-search placer."""

    def place(self, benchmark: Benchmark, plc=None) -> torch.Tensor:
        """Generate candidates, legalize, score, and return the best placement.

        Args:
            benchmark: Benchmark object.
            plc:       Optional PlacementCost object for proxy scoring.
                       When None, HPWL is used as a scoring surrogate.

        Returns:
            torch.Tensor [num_macros, 2] center positions.
        """
        from core.candidates import generate_candidates
        from core.candidate_scoring import score_and_select

        candidates = generate_candidates(benchmark)
        best, _ = score_and_select(candidates, benchmark, plc=plc)

        if best is None or best.positions is None:
            # Emergency fallback: return original positions
            return benchmark.macro_positions.clone()

        return best.positions.float()
