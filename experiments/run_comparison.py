"""
experiments/run_comparison.py

Runs OC alignments in a separate Python 3.11 subprocess
(required because localocpa needs pm4py <= 2.2).

Usage (from oc-token-replay root):
    python -m experiments.run_oc_alignments_subprocess
"""

import sys
import os
import time
import json
import csv
import subprocess
import tempfile

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from src.replay import OCTokenReplay
from logs import generate_synthetic_log
from nets import build_order_fulfilment_net

# path to Python 3.11 venv inside the alignments repo
OC_ALIGN_ROOT = os.path.abspath(os.path.join(ROOT, "..", "object-centric-alignments"))
PYTHON_39 = os.path.join(OC_ALIGN_ROOT, "venv_39", "Scripts", "python.exe")
WORKER_SCRIPT = os.path.join(ROOT, "alignment_baseline", "oc_align_worker.py")


# ── OCEL 1.0 writer ──────────────────────────────────────────────────

def save_as_ocel(log, path):
    events = {}
    objects = {}

    for i, ev in enumerate(log.events):

        seconds = int(ev.timestamp)
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        ts = f"2024-01-01T{hours:02d}:{minutes:02d}:{secs:02d}+00:00"

        events[str(i)] = {  # ← "0", "1", "2"...
            "ocel:activity": ev.activity,
            "ocel:timestamp": ts,  # ← с timezone
            "ocel:omap": [oid for oid, _ in ev.objects],
            "ocel:vmap": {},
        }
        for oid, otype in ev.objects:
            if oid not in objects:
                objects[oid] = {"ocel:type": otype, "ocel:ovmap": {}}

    obj_types = list({otype for ev in log.events for _, otype in ev.objects})
    data = {
        "ocel:global-event": {"ocel:activity": "__INVALID__"},
        "ocel:global-object": {"ocel:type": "__INVALID__"},
        "ocel:global-log": {
            "ocel:attribute-names": [],
            "ocel:object-types": obj_types,
            "ocel:version": ["1.0"],
            "ocel:ordering": ["timestamp"],
        },
        "ocel:events": events,
        "ocel:objects": objects,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ── run OC-TBR ───────────────────────────────────────────────────────

def run_octbr(net, log):
    t0 = time.perf_counter()
    result = OCTokenReplay(net).run(log)
    return result.fitness, round(time.perf_counter() - t0, 4)


# ── run OC alignments via subprocess ─────────────────────────────────

def run_oc_alignments(log, timeout=300):
    with tempfile.TemporaryDirectory() as tmp:
        log_path = os.path.join(tmp, "log.jsonocel")
        result_path = os.path.join(tmp, "result.json")
        save_as_ocel(log, log_path)

        cmd = [
            PYTHON_39, WORKER_SCRIPT,
            "--log", log_path,
            "--result", result_path,
            "--repo", OC_ALIGN_ROOT,
        ]

        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            elapsed = round(time.perf_counter() - t0, 4)

            if proc.returncode != 0:
                print(f"  Return code: {proc.returncode}")
                return None, elapsed

            if not os.path.exists(result_path):
                print("  Worker did not produce result.json")
                return None, elapsed

            with open(result_path) as f:
                data = json.load(f)

            return data.get("fitness"), elapsed

        except subprocess.TimeoutExpired:
            elapsed = round(time.perf_counter() - t0, 4)
            print(f"  Timed out after {timeout}s")
            return None, elapsed
        except Exception as e:
            print(f"  Subprocess error: {e}")
            return None, None


# ── main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    net = build_order_fulfilment_net()

    configs = [
        ("A", "0%  skip Pack", 0.0, 10),
        ("B", "20% skip Pack", 0.2, 10),
        ("C", "40% skip Pack", 0.4, 10),
    ]

    print("\n" + "=" * 72)
    print("  OC-TBR  vs  OC Alignments  [Liss et al., 2023]")
    print("=" * 72)
    print(f"\n{'Log':<5} {'N':>4}  "
          f"{'OC-TBR f':>9} {'Align f':>9}  "
          f"{'OC-TBR t':>9} {'Align t':>10}")
    print("-" * 55)

    results_dir = os.path.join(ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_rows = []

    for name, desc, rate, n in configs:
        log = generate_synthetic_log(n_orders=n, deviation_rate=rate, seed=42)

        f_tbr, t_tbr = run_octbr(net, log)
        f_aln, t_aln = run_oc_alignments(log)

        f_aln_s = f"{f_aln:.4f}" if f_aln is not None else "N/A"
        t_aln_s = f"{t_aln:.4f}s" if t_aln is not None else "—"

        print(f"Log {name:<2} {n:>4}  "
              f"{f_tbr:>9.4f} {f_aln_s:>9}s  "
              f"{t_tbr:>8.4f} {t_aln_s:>10}")

        csv_rows.append([name, desc, n,
                         round(f_tbr, 4), f_aln,
                         t_tbr, t_aln])

    print("=" * 72)

    csv_path = os.path.join(results_dir, "comparison_oc_alignments.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["log", "description", "n_orders",
                    "octbr_fitness", "oc_align_fitness",
                    "octbr_time_s", "oc_align_time_s"])
        w.writerows(csv_rows)
    print(f"\nSaved → {csv_path}")
