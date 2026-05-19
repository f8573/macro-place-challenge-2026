# ARCHITECTURE.md

## Goal

This solver is a modular macro placement submission for the Partcl/HRT Macro Placement Challenge.

Milestone 1 infrastructure is complete and hardened.

The current active milestone is:

```text
M2A — Hypergraph Extraction and Spectral Diagnostics
```

M2A should prove that the solver can:

- extract macro-net connectivity from the official `Benchmark`
- build a weighted sparse macro graph
- construct a Laplacian
- compute diagnostic spectral embeddings

M2A should not attempt:

- final placement optimization
- legalization
- candidate ranking
- official `placer.py` integration

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

- one row per macro
- fixed macros preserved
- soft macros unchanged unless a later milestone explicitly handles them
- no hard-macro overlaps for final/evaluated placements
- finite coordinates
- in bounds

M2A must not change official `placer.py` behavior except for tiny evaluator import-boundary fixes if required.

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

M2A spectral coordinates are diagnostic embedding coordinates, not legal placement coordinates.

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
  smoke, inspect, and diagnostic commands

tests/
  focused deterministic tests
```

---

## M2A Module Additions

M2A may add:

```text
core/hypergraph.py
core/laplacian.py
core/spectral.py

scripts/inspect_spectral_graph.py

tests/test_hypergraph.py
tests/test_laplacian.py
tests/test_spectral.py
```

These modules are diagnostic infrastructure only.

They should not modify official placement behavior.

---

## Official Utility Policy

Reuse official challenge utilities where practical.

Use:

- `macro_place.utils` for validation
- `macro_place.objective` for proxy scoring
- official visualization helpers where suitable

Do not modify official evaluator/scoring behavior.

Do not duplicate evaluator semantics unless the local version is clearly marked as debug-only.

---

## Current Milestone Boundary

Current active milestone:

```text
M2A — Hypergraph Extraction and Spectral Diagnostics
```

M2A may implement:

- hypergraph/net extraction helpers
- normalized clique expansion
- sparse adjacency construction
- Laplacian construction
- connected-component diagnostics
- diagnostic eigensolve wrappers
- graph/spectral inspection scripts
- focused deterministic graph/spectral tests

M2A must not implement:

- full spectral candidate generation
- official `placer.py` behavior changes
- legalization
- overlap repair
- analytical optimization
- RUDY
- LNS
- coordinate descent
- simulated annealing
- learned policies
- benchmark-specific hardcoding

M1 baseline behavior and regression tests must continue to pass.

---

## M2A Design Constraints

The official `Benchmark` remains the source of truth.

Do not introduce large duplicate models such as:

- `Macro`
- `Pin`
- `Net`
- `Placement`
- `BenchmarkData`

Use small helper types only when they reduce ambiguity.

M2A graph/eigen outputs are diagnostics. They are not final placements.

---

## Design Decisions

1. Internal coordinates are centers.
2. `placer.py` stays thin.
3. Challenge API assumptions stay near `benchmark_adapter.py`.
4. Official scoring remains the source of truth.
5. Local fallback metrics are debug-only.
6. M2A begins with normalized clique expansion only.
7. Star expansion, hybrid expansion, candidate variants, and legalization are deferred.
