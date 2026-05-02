"""
Run all experiments and save results to results/.

Usage:
    python experiments/run_experiments.py

Output files (written to results/):
    experiment1.csv                — f, f_order, f_item, n_events, time for logs A/B/C
    experiment2_top_violators.csv  — per-object fitness for real OCEL log
    scalability.csv                — runtime vs log size
    run_log.txt                    — full console output saved to file
"""
import csv
import os
import sys
import time
from collections import Counter
from datetime import datetime

# ── paths ─────────────────────────────────────────────────────────────
ROOT    = os.path.dirname(os.path.dirname(__file__))
RESULTS = os.path.join(ROOT, "results")
DATA    = os.path.join(ROOT, "data")

sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from replay import OCTokenReplay
from logs   import generate_synthetic_log, load_ocel1
from nets   import build_order_fulfilment_net


# ── Tee: write to console AND file simultaneously ─────────────────────

class Tee:
    """Duplicates stdout to both the console and a log file."""
    def __init__(self, filepath):
        self.console = sys.stdout
        self.file    = open(filepath, "w", encoding="utf-8")

    def write(self, msg):
        self.console.write(msg)
        self.file.write(msg)

    def flush(self):
        self.console.flush()
        self.file.flush()

    def close(self):
        sys.stdout = self.console   # restore original stdout
        self.file.close()


# ── helpers ───────────────────────────────────────────────────────────

def run_once(net, log):
    t0 = time.perf_counter()
    result = OCTokenReplay(net).run(log)
    elapsed = time.perf_counter() - t0
    return result, elapsed


def save_csv(filename, rows, header):
    path = os.path.join(RESULTS, filename)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  saved → {path}")


def separator(char="=", width=65):
    print(char * width)


# ── Experiment 1: synthetic logs ──────────────────────────────────────

def experiment1():
    separator()
    print("  EXPERIMENT 1 — Synthetic Logs")
    print(f"  OC-PN: order-fulfilment (Fig. 1)")
    print(f"  n_orders = 100  |  seed = 42")
    separator()

    net = build_order_fulfilment_net()

    configs = [
        ("A", "none  (baseline)", 0.0),
        ("B", "20% skip Pack",    0.2),
        ("C", "40% skip Pack",    0.4),
    ]

    header = ["log", "deviation", "f", "f_order", "f_item",
              "n_events", "n_objects", "time_s"]
    rows = []

    print(f"\n{'Log':<5} {'Deviation':<22} {'f':>6} {'f_order':>8} "
          f"{'f_item':>7} {'events':>7} {'objects':>8} {'time':>7}")
    print("-" * 65)

    for name, desc, rate in configs:
        log = generate_synthetic_log(n_orders=100,
                                     deviation_rate=rate,
                                     seed=42)
        result, elapsed = run_once(net, log)
        ft = result.fitness_by_type()

        f       = round(result.fitness, 4)
        f_order = round(ft.get("order", 0.0), 4)
        f_item  = round(ft.get("item",  0.0), 4)
        n_obj   = len(result.per_object)

        print(f"Log {name:<2} {desc:<22} {f:>6.3f} {f_order:>8.3f} "
              f"{f_item:>7.3f} {len(log.events):>7} {n_obj:>8} {elapsed:>6.3f}s")

        rows.append([name, desc, f, f_order, f_item,
                     len(log.events), n_obj, round(elapsed, 4)])

    save_csv("experiment1.csv", rows, header)

    # ── per-object breakdown for Log C ───────────────────────────────
    print()
    separator("-")
    print("  Per-object breakdown — Log C (top violators)")
    separator("-")

    log_c    = generate_synthetic_log(n_orders=100, deviation_rate=0.4, seed=42)
    result_c, _ = run_once(net, log_c)

    violators   = sorted([s for s in result_c.per_object if s.fitness < 1.0],
                         key=lambda s: s.fitness)
    conformant  = [s for s in result_c.per_object if s.fitness == 1.0]

    print(f"\n  Total objects   : {len(result_c.per_object)}")
    print(f"  Conformant      : {len(conformant)}  (f_o = 1.000)")
    print(f"  Violators       : {len(violators)}  (f_o < 1.000)")

    print(f"\n  {'obj_id':<10} {'type':<8} {'fitness':>8} {'missing':>8} {'remaining':>10}")
    print("  " + "-" * 48)
    for s in violators[:12]:
        print(f"  {s.obj_id:<10} {s.obj_type:<8} {s.fitness:>8.3f} "
              f"{s.missing:>8} {s.remaining:>10}")
    if len(violators) > 12:
        print(f"  ... ({len(violators) - 12} more violators not shown)")

    # type breakdown among violators
    print()
    vc = Counter(s.obj_type for s in violators)
    for ot, n in vc.most_common():
        print(f"  Violating {ot:<10}: {n} objects")


# ── Experiment 2: real OCEL log ───────────────────────────────────────

def experiment2():
    print()
    separator()
    print("  EXPERIMENT 2 — Real OCEL Log")
    print(f"  File: data/example_log.jsonocel")
    separator()

    path = os.path.join(DATA, "example_log.jsonocel")
    if not os.path.exists(path):
        print(f"\n  ✗  File not found: {path}")
        print("  Place example_log.jsonocel in the data/ folder and re-run.")
        return

    # parse
    log = load_ocel1(path)

    type_counts = Counter(
        ot for ev in log.events for _, ot in ev.objects
    )
    unique_objs = {oid for ev in log.events for oid, _ in ev.objects}
    activities  = sorted({ev.activity for ev in log.events})

    print(f"\n  Parsed successfully.")
    print(f"  Events     : {len(log.events)}")
    print(f"  Objects    : {len(unique_objs)} unique")
    print(f"  Object types:")
    for ot, n in type_counts.most_common():
        print(f"    {ot:<15}: {n} references")
    print(f"  Activities : {activities}")

    # build accepting net
    from model import OCPetriNet, Place, Transition
    from collections import defaultdict

    obj_types = {ot for ev in log.events for _, ot in ev.objects}
    act_types = defaultdict(set)
    for ev in log.events:
        for _, ot in ev.objects:
            act_types[ev.activity].add(ot)

    net = OCPetriNet()
    places = {}
    for ot in obj_types:
        # только source и sink — никакого mid
        src = Place(f"src_{ot}", ot, is_source=True)
        sink = Place(f"sink_{ot}", ot, is_sink=True)
        net.places += [src, sink]
        places[ot] = (src, sink)

    for act, types in act_types.items():
        # каждый переход потребляет src и производит src (токен возвращается)
        # так объект всегда может участвовать в следующем событии
        net.transitions.append(Transition(
            id=act.replace(" ", "_"),
            activity=act,
            preset={ot: [places[ot][0]] for ot in types},  # src
            postset={ot: [places[ot][0]] for ot in types},  # src обратно
        ))
    net.build_index()

    result, elapsed = run_once(net, log)

    print()
    separator("-")
    print("  Replay results (accepting net — parsing validation)")
    separator("-")
    print(f"\n  Global fitness : {result.fitness:.4f}  "
          f"{'✓  OK' if result.fitness >= 0.99 else '✗  unexpected missing tokens'}")
    print(f"  Runtime        : {elapsed:.4f} s")

    ft = result.fitness_by_type()
    print(f"\n  Per-type fitness:")
    for ot, fv in sorted(ft.items()):
        print(f"    f_tau ({ot:<12}) = {fv:.4f}")

    print(f"\n  Per-object fitness (all {len(result.per_object)} objects):")
    print(f"  {'obj_id':<15} {'type':<12} {'fitness':>8} {'missing':>8} {'remaining':>10}")
    print("  " + "-" * 58)
    for s in sorted(result.per_object, key=lambda x: (x.obj_type, x.obj_id)):
        flag = "  ← !" if s.fitness < 1.0 else ""
        print(f"  {s.obj_id:<15} {s.obj_type:<12} {s.fitness:>8.3f} "
              f"{s.missing:>8} {s.remaining:>10}{flag}")

    # save csv
    header = ["obj_id", "obj_type", "fitness", "missing", "remaining"]
    rows   = [
        [s.obj_id, s.obj_type, round(s.fitness, 4), s.missing, s.remaining]
        for s in sorted(result.per_object, key=lambda x: (x.obj_type, x.obj_id))
    ]
    save_csv("experiment2_top_violators.csv", rows, header)


# ── Scalability ───────────────────────────────────────────────────────

def scalability():
    print()
    separator()
    print("  SCALABILITY TEST")
    print("  deviation_rate = 0.0  |  seed = 42")
    separator()

    net    = build_order_fulfilment_net()
    sizes  = [25, 50, 100, 200, 400, 800]
    header = ["n_orders", "n_events", "n_objects", "time_s"]
    rows   = []

    print(f"\n  {'n_orders':>9} {'n_events':>9} {'n_objects':>10} {'time':>8}")
    print("  " + "-" * 42)

    prev_time = None
    for n in sizes:
        log = generate_synthetic_log(n_orders=n, deviation_rate=0.0, seed=42)
        _, elapsed = run_once(net, log)
        n_obj = len({oid for ev in log.events for oid, _ in ev.objects})

        factor = ""
        if prev_time is not None and prev_time > 0:
            factor = f"  (×{elapsed/prev_time:.2f})"
        prev_time = elapsed

        print(f"  {n:>9} {len(log.events):>9} {n_obj:>10} "
              f"{elapsed:>7.4f}s{factor}")
        rows.append([n, len(log.events), n_obj, round(elapsed, 5)])

    save_csv("scalability.csv", rows, header)
    print("\n  Factor column shows runtime ratio relative to previous row.")
    print("  Expected: close to 2.0 at each doubling (linear scaling).")


# ── Entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(DATA,    exist_ok=True)

    # open Tee — everything printed after this line goes to both
    # console and results/run_log.txt
    log_path = os.path.join(RESULTS, "run_log.txt")
    tee = Tee(log_path)
    sys.stdout = tee

    print(f"OC-TBR Experimental Run")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        experiment1()
        experiment2()
        scalability()
    finally:
        print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Full log saved to: {log_path}")
        tee.close()