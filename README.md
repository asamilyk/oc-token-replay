# OC-TBR: Object-Centric Token-Based Replay

Implementation and experiments for the paper:

> **Conformance Checking of Object-Centric Petri Nets and Event Logs
> using a Token Replay Approach**
>
> Anastasia Samilyk, HSE University, 2026.

---

## Description

Classical token-based replay checks how well a process log conforms to
a Petri net model. This project extends that technique to
**object-centric Petri nets (OC-PNs)** and **object-centric event logs
(OCEL)**, where a single event can involve multiple objects of different
types simultaneously.

The algorithm **OC-TBR** produces three fitness metrics:

| Metric | Scope | Use |
|--------|-------|-----|
| `f` (global) | Whole log | Overall conformance |
| `f_o` (per-object) | Single object | Root-cause: *which* object violates? |
| `f_τ` (per-type) | Object type | *Which category* of objects deviates most? |

---

## Quick start

```bash
git clone https://github.com/asamilyk/oc-token-replay
cd oc-token-replay
pip install -r requirements.txt

# download real-world OCEL logs (once)
python scripts/download_data.py

# run on any OCEL log with auto-discovered OC-PN
python run_conformance.py data/example_log.jsonocel
python run_conformance.py data/p2p_normal.jsonocel --threshold 0.9
python run_conformance.py data/your_log.jsonocel --show-net
```

---

## Reproducing paper results

```bash
# Experiments 1 + 2 — synthetic logs + parsing validation
python -m experiments.run_experiments

# Experiments 3 + 4 — three real OCEL logs + auto vs manual OC-PN
python -m experiments.run_real_logs

# Experiment 5 — comparison with OC Alignments [Liss et al., 2023]
# requires Python 3.9 venv — see alignment_baseline/README.md
python -m experiments.run_comparison

# generate all figures (PDF + PNG)
python -m experiments.plot_results
```

| Paper item | Script | Output |
|------------|--------|--------|
| Table II — synthetic logs | `run_experiments` | `results/experiment1.csv` |
| Table III — parsing validation | `run_experiments` | `results/experiment2_top_violators.csv` |
| Table IV — three real logs | `run_real_logs` | `results/experiment3_real_logs.csv` |
| Table V — auto vs manual OC-PN | `run_real_logs` | `results/running_example_auto_vs_manual.csv` |
| Table VI — alignment comparison | `run_comparison` | `results/comparison_oc_alignments.csv` |
| Fig. 1 — fitness vs deviation | `plot_results` | `figures/fitness_vs_deviation.pdf` |
| Fig. 2 — auto vs manual | `plot_results` | `figures/auto_vs_manual.pdf` |
| Fig. 3 — fitness comparison | `plot_results` | `figures/comparison_alignments.pdf` |
| Fig. 4 — runtime comparison | `plot_results` | `figures/runtime_comparison.pdf` |
| Fig. 5 — scalability | `plot_results` | `figures/scalability.pdf` |

---

## Repository structure

```
oc-token-replay/
│
├── src/                        # OC-TBR algorithm — core library
│   ├── __init__.py
│   ├── model.py                # Place, Transition, OCPetriNet
│   ├── log.py                  # OCELLog, OCEvent + OCEL 1.0 parser
│   ├── marking.py              # per-object token bag
│   └── replay.py               # OC-TBR algorithm (Algorithm 1)
│
├── experiments/                # experiment scripts
│   ├── nets.py                 # OC-PN definitions (manual + discovered)
│   ├── logs.py                 # synthetic log generator
│   ├── run_experiments.py      # Experiments 1 + 2
│   ├── run_real_logs.py        # Experiments 3 + 4
│   ├── run_comparison.py       # Experiment 5 — vs OC alignments
│   └── plot_results.py         # all five figures
│
├── alignment_baseline/         # OC alignments [Liss et al., 2023]
│   ├── oc_align_worker.py      # worker (Python 3.9 + pm4py 2.2)
│   └── README.md               # setup instructions
│
├── scripts/
│   └── download_data.py        # downloads OCEL logs before experiments
│
├── data/                       # OCEL log files
│   ├── README.md               # data sources and download links
│   └── example_log.jsonocel    # included — order management, 23 events
│
├── results/                    # auto-generated CSV results (git-ignored)
├── figures/                    # auto-generated PDF/PNG figures (git-ignored)
│
├── tests/
│   └── test_replay.py          # 17 unit tests
│
├── run_conformance.py          # universal runner for any OCEL log
├── requirements.txt
└── README.md
```

---

## Algorithm (paper Section IV)

```
Input:  OC-PN N, OCEL log L
Output: global counters P_g, C_g, M_g, R_g; per-object stats

Step 1 — Object layout:   group objects by type (byType[τ])
Step 2 — Init markings:   place 1 token at source place per object
Step 3 — Replay loop:     for each event e = (activity, binding):
  3a — find transition t matching activity label
  3b — for each object o bound to e:
       consume from preset(t, type(o))  → if missing: m_o++
       produce into postset(t, type(o))
Step 4 — Remaining:       count leftover tokens in non-sink places
         (sink-place tokens excluded — they indicate normal completion)

Fitness formulae — generalisations of Rozinat & van der Aalst (2008):
  f     = 1 − ½·(M_g/C_g + R_g/P_g)        [global,     Eq. 1]
  f_o   = 1 − ½·(m_o/c_o + r_o/p_o)         [per-object, Eq. 2]
  f_τ   = 1 − ½·(Σm_o/Σc_o + Σr_o/Σp_o)    [per-type,   Eq. 3]

Time complexity: O(|L| · |O| · |P|)  — polynomial
```

---

## Usage as a library

```python
from src import OCTokenReplay, OCELLog, OCEvent
from src.model import OCPetriNet, Place, Transition

# build a minimal OC-PN
net = OCPetriNet()
src = Place("src", "order", is_source=True)
end = Place("end", "order", is_sink=True)
t   = Transition("t1", "Process",
                 preset ={"order": [src]},
                 postset={"order": [end]})
net.places      = [src, end]
net.transitions = [t]
net.build_index()

# build a log
log = OCELLog(events=[
    OCEvent("e1", "Process", 1.0, [("o1", "order")]),
])

# run replay
result = OCTokenReplay(net).run(log)
print(result.summary())
print(result.per_object_table())
```

---

## Running tests

```bash
pytest tests/ -v
```

Tests cover:
- Proposition 1 — unsoundness of per-object projection (Section IV)
- Perfect conformance → f = 1.0, no missing tokens
- Known violations → correct per-object flagging
- All three fitness formulae (Eq. 1–3)
- Edge cases: empty log, unknown activities

---

## Data

`data/example_log.jsonocel` is included in the repository.
All other logs are downloaded automatically:

```bash
python scripts/download_data.py
```

See `data/README.md` for sources and download links.

---

## Alignment baseline setup

The comparison with OC Alignments [Liss et al., 2023] requires a
separate Python 3.9 environment.
See `alignment_baseline/README.md` for step-by-step setup.

---

## Requirements

- Python 3.10+ — core library and experiments
- `pm4py` — OC-PN discovery and OCEL loading (Experiments 3–4)
- `matplotlib`, `pandas`, `numpy` — figures
- `pytest` — tests
- Python 3.9 + `pm4py==2.2.19.1` — alignment baseline only

```bash
pip install -r requirements.txt
```

---

## Citation

```bibtex
@article{samilyk2026octbr,
  author  = {Samilyk, Anastasia},
  title   = {Conformance Checking of Object-Centric Petri Nets
             and Event Logs using a Token Replay Approach},
  journal = {HSE University, Faculty of Computer Science},
  year    = {2026},
}
```
