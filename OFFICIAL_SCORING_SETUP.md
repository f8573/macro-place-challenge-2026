# Official Scoring Setup

M2B candidate selection requires **plc_client_os** (the TILOS PlacementCost evaluator)
and the **IBM ICCAD04 testcase netlists** to produce real, non-degenerate proxy costs.
Both come from the `external/MacroPlacement` git submodule.

**Current status (as of 2026-05-21): WORKING locally.**

---

## Required dependencies

| Dependency | Source |
|---|---|
| `plc_client_os` Python module | `external/MacroPlacement/CodeElements/Plc_client/` |
| IBM ICCAD04 netlist + placement files | `external/MacroPlacement/Testcases/ICCAD04/<name>/` |

Both are pure Python / protobuf — no Bazel or C++ compilation needed.

---

## Setup

### 1. Initialize the submodule

```bash
git submodule update --init external/MacroPlacement
```

This clones `https://github.com/partcleda/MacroPlacement.git`
(branch `fix-scientific-notation-parsing`, commit `45a721d`, ~900 MB shallow clone).

### 2. Verify plc_client_os is importable

```bash
python -c "
import sys
sys.path.insert(0, 'external/MacroPlacement/CodeElements/Plc_client')
from plc_client_os import PlacementCost
print('OK')
"
```

### 3. Verify IBM testcases exist

```
external/MacroPlacement/Testcases/ICCAD04/
├── ibm01/
│   ├── netlist.pb.txt   ✓
│   └── initial.plc      ✓
├── ibm02/
│   ├── netlist.pb.txt   ✓
│   └── initial.plc      ✓
└── ibm03/
    ├── netlist.pb.txt   ✓
    └── initial.plc      ✓
```

---

## Running official scoring

### Smoke test

```bash
python -m submissions.solver.scripts.run_official_scoring_smoke -b ibm01 -b ibm02 -b ibm03
```

### Full benchmark run (after smoke passes)

```bash
python -m submissions.solver.scripts.run_benchmarks --profile official-smoke
```

---

## Verified results (2026-05-21, commit 45a721d)

| Benchmark | net_nodes | scoring_mode | spectral | terminal_anchor | selected_due_to |
|---|---|---|---|---|---|
| ibm01 | 5993 nets, 16815 pins | official | True | True | proxy_cost |
| ibm02 | 9668 nets, 31694 pins | official | True | True | proxy_cost |
| ibm03 | 7674 nets, 24884 pins | official | True | True | proxy_cost |

### Proxy cost results

| Benchmark | Raw input cost | Best candidate (legalized) | Delta | Winner |
|---|---|---|---|---|
| ibm01 | 1.038498 | 1.140784 | +0.102 | original (legalized) |
| ibm02 | 1.565849 | 1.625169 | +0.059 | original (legalized) |
| ibm03 | 1.325486 | 1.353598 | +0.028 | original (legalized) |

**No generated M2B candidate beats the legalized original on any benchmark.**

### Key diagnostic finding

The legalizer degrades the original `initial.plc` placement by 3–10%.
The raw input positions (from `initial.plc` via plc_client_os) score lower than the
legalized positions our pipeline produces. This means:

- All 28 candidates (including the legalized original) are worse than the raw input
- The candidate generation and selection pipeline is working correctly (non-degenerate scores,
  real ranking, proxy_cost-based selection)
- The bottleneck is the legalizer modifying the original placement unnecessarily

### Candidate diversity at ibm01

- 28 candidates generated, 28 valid, 0 invalid
- 21 geometrically distinct placements (7 hash collisions)
- Families: original, spectral, area_degree, terminal_anchor
- avg_displacement=11.64 µm, max_displacement=28.24 µm

---

## What changes with official scoring active vs local .pt only

| Field | Local (.pt only) | Official (plc_client_os) |
|---|---|---|
| `net_nodes` | `[]` (empty) | List of per-net node tensors |
| `scoring_mode` | `unavailable` | `official` |
| `selected_due_to` | `validity_only` | `proxy_cost` |
| `spectral_available` | `False` | `True` |
| `terminal_anchor_available` | `False` | `True` |
| Candidate ranking | arbitrary | by proxy cost |
| M2B improvement claim | cannot be made | measurable |

---

## .pt files: net_nodes still empty

The `.pt` benchmark files in `benchmarks/processed/public/` still have `net_nodes=[]`
because they were converted before the submodule was available. To regenerate them:

```bash
python scripts/convert_ibm_benchmarks.py
```

The regenerated files will have full connectivity. This is optional — the official
smoke test and `--profile official-smoke` load directly from the testcase directory.
