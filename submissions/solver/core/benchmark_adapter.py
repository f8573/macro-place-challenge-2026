"""
Thin helpers for inspecting Benchmark objects.

These wrap the official Benchmark API and expose derived statistics.
No challenge-specific parsing logic lives here — that belongs in macro_place.loader.
"""

from macro_place.benchmark import Benchmark


def canvas_area(benchmark: Benchmark) -> float:
    """Canvas area in μm²."""
    return benchmark.canvas_width * benchmark.canvas_height


def hard_macro_area(benchmark: Benchmark) -> float:
    """Total area of all hard macros in μm²."""
    sizes = benchmark.macro_sizes[: benchmark.num_hard_macros]
    return float((sizes[:, 0] * sizes[:, 1]).sum().item())


def utilization(benchmark: Benchmark) -> float:
    """Hard macro area / canvas area (0.0–1.0)."""
    ca = canvas_area(benchmark)
    return hard_macro_area(benchmark) / ca if ca > 0.0 else 0.0


def inspect(benchmark: Benchmark) -> dict:
    """Return a flat dict of benchmark statistics useful for logging."""
    fixed_count = int(benchmark.macro_fixed.sum().item())
    movable_hard = int(
        (benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()).sum().item()
    )
    return {
        "name": benchmark.name,
        "canvas_width": benchmark.canvas_width,
        "canvas_height": benchmark.canvas_height,
        "canvas_area_um2": canvas_area(benchmark),
        "num_macros": benchmark.num_macros,
        "num_hard_macros": benchmark.num_hard_macros,
        "num_soft_macros": benchmark.num_soft_macros,
        "num_fixed": fixed_count,
        "num_movable_hard": movable_hard,
        "num_nets": benchmark.num_nets,
        "hard_macro_area_um2": hard_macro_area(benchmark),
        "utilization": utilization(benchmark),
        "grid_rows": benchmark.grid_rows,
        "grid_cols": benchmark.grid_cols,
    }
