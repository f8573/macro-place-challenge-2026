# ARCHITECTURE.md

## Goal

This solver is a modular macro placement submission for the Partcl/HRT Macro Placement Challenge.

The immediate goal is M1 Infrastructure: produce a valid baseline placement, with clean wrappers for validation, scoring, visualization, scripts, and tests.

M1 should not attempt serious optimization.

---

## Evaluator Contract

The official evaluator loads `submissions/solver/placer.py`, finds a class with a `place()` method, instantiates it, and calls:

```python
placement = placer.place(benchmark)
```

The returned placement must be:

```text
torch.Tensor[num_macros, 2]
```

Coordinates are macro centers in microns.

Requirements:

* one row per macro
* fixed macros preserved
* soft macros unchanged in M1
* no hard-macro overlaps
* finite coordinates
* in bounds

---

## Coordinate System

Internal solver coordinates are always center-based.

```text
left   = x - width / 2
right  = x + width / 2
bottom = y - height / 2
top    = y + height / 2
```

Touching edges are legal and are not overlaps.

Lower-left coordinates may only appear at adapter or visualization boundaries.

---

## Module Boundaries

```text
placer.py
  thin evaluator entrypoint

core/benchmark_adapter.py
  challenge Benchmark helpers and translation

core/geometry.py
  center-coordinate geometry helpers

core/validation.py
  wrapper around official validation plus local checks

core/scoring.py
  wrapper around official proxy scoring when plc is available

core/io.py
  artifact serialization

viz/
  visualization wrappers only

scripts/
  smoke, inspect, and sweep commands

tests/
  focused deterministic tests
```

---

## Official Utility Policy

Reuse official challenge utilities where practical.

Use:

* `macro_place.utils` for validation
* `macro_place.objective` for proxy scoring
* official visualization helpers where suitable

Do not modify official evaluator/scoring behavior.

Do not duplicate evaluator semantics unless the local version is clearly marked as debug-only.

---

## Milestone Boundary

M1 may implement:

* thin baseline `placer.py`
* benchmark inspection helpers
* geometry helpers
* validation/scoring wrappers
* artifact logging
* visualization wrappers
* smoke scripts
* focused tests

M1 must not implement:

* spectral placement
* analytical optimization
* RUDY
* LNS
* simulated annealing
* learned policies
* benchmark-specific hardcoded placements

---

## Design Decisions

1. Internal coordinates are centers.
2. `placer.py` stays thin.
3. Challenge API assumptions stay near `benchmark_adapter.py`.
4. Official scoring remains the source of truth.
5. Local fallback metrics are debug-only.
