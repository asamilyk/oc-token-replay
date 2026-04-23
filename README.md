# OC-TBR: Object-Centric Token-Based Replay

Implementation and experiments for the paper:

> **Checking the Conformance of Object-Centric Petri Nets and Event Logs
> using a Token Replay Approach**

> Anastasia Samilyk, HSE University, 2026.

## Description

Classical token-based replay checks how well a process log conforms to
a Petri net model. This project extends that technique to
**object-centric Petri nets (OC-PNs)** and **object-centric event logs
(OCEL)**, where a single event can involve multiple objects of different
types simultaneously.

The algorithm (OC-TBR) produces **three fitness metrics**:

| Metric | Scope | Use |
|--------|-------|-----|
| `f` (global) | Whole log | Overall conformance |
| `f_o` (per-object) | Single object | Root-cause: *which* object violates? |
| `f_τ` (per-type) | Object type | *Which category* of objects deviates most? |

## Quick start

```bash
git clone https://github.com/apsamilyk/oc-token-replay
cd oc-token-replay
pip install -r requirements.txt

# Run all experiments (reproduces Tables II–III and Figure 4 in the paper)
python experiments/run_experiments.py --all

# Run only the counterexample (Section IV, Proposition 1)
python experiments/run_experiments.py --exp0

# Use your own OCEL 2.0 JSON log
python experiments/run_experiments.py --exp2 --ocel path/to/log.json
```

## Repository structure

```
oc-token-replay/
├── src/
│   ├── __init__.py      # Public API
│   ├── model.py         # OC-PN data structures (Place, Transition, OCPetriNet)
│   ├── log.py           # OCEL structures + JSON/CSV importers
│   ├── marking.py       # Per-object token bag
│   └── replay.py        # OC-TBR algorithm (Algorithm 1 in paper)
│
├── experiments/
│   ├── nets.py          # Reference OC-PNs (Figure 1 in paper)
│   ├── logs.py          # Synthetic log generators (Logs A, B, C)
│   └── run_experiments.py  # Reproduces all tables and figures
│
├── tests/
│   └── test_replay.py   # 17 unit tests covering all paper claims
│
├── data/                # Place your OCEL 2.0 JSON files here
│   └── README.md
│
├── results/             # Auto-generated CSV results (created on run)
│
├── requirements.txt
└── README.md
```

## Algorithm (paper Section V)

```
Input:  OC-PN N, OCEL log L
Output: counters P_g, C_g, M_g, R_g; per-object stats

Step 1 — Object layout:   group objects by type
Step 2 — Init markings:   place 1 token at source place for each object
Step 3 — Replay loop:     for each event e = (activity, binding):
  Step 3a — find transition t matching activity
  Step 3b — for each object o bound to e:
             consume from preset(t, type(o))   → if missing: m_o++
             produce into postset(t, type(o))
Step 4 — Remaining:       count leftover tokens per object

Fitness formulae (backward-compatible with classical replay):
  f     = 1 - ½·(M_g/C_g + R_g/P_g)          [global,   Eq. 1]
  f_o   = 1 - ½·(m_o/c_o + r_o/p_o)           [per-obj,  Eq. 2]
  f_τ   = 1 - ½·(Σm_o/Σc_o + Σr_o/Σp_o)      [per-type, Eq. 3]

Time complexity: O(|L| · |O| · |P|)  — polynomial
```

## Usage as a library

```python
from src import OCPetriNet, Place, Transition
from src import make_synthetic_log, OCTokenReplay

# Build a minimal OC-PN
net = OCPetriNet()
src = Place("src", "order", is_source=True)
end = Place("end", "order", is_sink=True)
t   = Transition("t1", "Process", preset={"order": [src]},
                                   postset={"order": [end]})
net.places      = [src, end]
net.transitions = [t]

# Build a log
log = make_synthetic_log([
    {"id": "e1", "activity": "Process", "timestamp": 1,
     "objects": [["o1", "order"]]},
])

# Run replay
result = OCTokenReplay(net).run(log)
print(result.summary())
print(result.per_object_table())
```

## Reproducing paper results

| Paper item | Command |
|------------|---------|
| Section IV counterexample | `--exp0` |
| Table II (synthetic logs) | `--exp1` |
| Table III (OCEL 2.0)      | `--exp2 [--ocel path]` |
| Figure 4 (scalability)    | `--scale` |

Results are saved as CSV files in `results/`.

## Running tests

```bash
pytest tests/ -v
```

All 17 tests cover:
- Proposition 1 (soundness of per-object projection failure)
- Perfect conformance → no missing tokens
- Known violations → correct per-object flagging
- Fitness formula correctness (all three equations)
- Edge cases (empty log, unknown activities)

## OCEL 2.0 data

The paper uses the **order management** log from the
[OCEL 2.0 benchmark](https://ocel-standard.org/).
Download `order-management.jsonocel` and pass it via `--ocel`.

## Requirements

- Python 3.10+
- No mandatory dependencies for the core library
- `pytest` for tests
- `pm4py` (optional) for OCEL discovery and alignment baseline

## Citation

```bibtex
@article{samilyk2026octbr,
  author  = {Samilyk, Anastasia},
  title   = {Checking the Conformance of Object-Centric Petri Nets
             and Event Logs using a Token Replay Approach},
  journal = {HSE University, Faculty of Computer Science},
  year    = {2026},
  note    = {IEEE Computer Society Journal format}
}
```
