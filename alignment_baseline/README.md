# Alignment Baseline

This folder contains the worker script for running the OC alignment
algorithm of Liss et al. (2023) as a subprocess baseline for
Experiment 5. The implementation is taken from the original authors'
repository: https://github.com/LukasLiss/object-centric-alignments

The alignment implementation requires **Python 3.9** and **pm4py ≤ 2.2**,
which are incompatible with the main project environment.
It is therefore run in a separate virtual environment.

## Setup (one time)

**Step 1:** Clone the alignments repository into the parent folder:

```bash
cd ..
git clone https://github.com/LukasLiss/object-centric-alignments.git
cd object-centric-alignments
```

**Step 2:** Install Python 3.9 from python.org (do not set as default).

**Step 3:** Create a virtual environment with Python 3.9:

```bash
# Windows
C:\path\to\Python39\python.exe -m venv venv_39
.\venv_39\Scripts\activate

# macOS/Linux
python3.9 -m venv venv_39
source venv_39/bin/activate
```

**Step 4:** Install dependencies:

```bash
pip install "pm4py==2.2.19.1" "packaging==21.3" "deprecation==2.1.0"
pip install networkx matplotlib cvxopt
pip install jsonschema
```

**Step 5:** Verify:

```bash
python -c "from algorithm import calculate_oc_alignments; print('OK')"
```
**Step 6:** Return to the main project folder and switch back to the
main environment:

```powershell
# Windows
cd ..\oc-token-replay
deactivate
.\.venv\Scripts\activate

# macOS / Linux
cd ../oc-token-replay
deactivate
source .venv/bin/activate
```

Then run the comparison from the project root:

```bash
python -m experiments.run_comparison
```

Results are saved to `results/comparison_oc_alignments.csv`.

## Notes

- The worker script `oc_align_worker.py` is called automatically
  via subprocess — you do not need to run it directly.
- If Python 3.9 is installed at a different path, update
  `PYTHON_39` in `experiments/run_comparison.py`.
- Alignments are slow even on small logs (3–5 s per log of 10 orders).
  Do not run on large logs without a timeout.