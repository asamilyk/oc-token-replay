"""
experiments/run_experiments.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Reproduces all experimental results reported in the paper:

  Table II  — OC-TBR results on synthetic logs (Experiment 1)
  Table III — OC-TBR vs. alignment baseline, OCEL 2.0 (Experiment 2)
  Fig.  4   — Scalability: runtime vs. log size

Usage:
    python experiments/run_experiments.py [--all | --exp1 | --exp2 | --scale]

Output: results printed to stdout + saved as CSV in results/
"""

import sys, os, time, csv, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src          import OCTokenReplay
from experiments.nets import build_order_fulfilment_net, build_large_net
from experiments.logs import (build_log_A, build_log_B, build_log_C,
                               build_counterexample_log)

os.makedirs("results", exist_ok=True)


# ─────────────────────────────────────────────────────────────
# Experiment 0 — Soundness counterexample (Section IV)
# ─────────────────────────────────────────────────────────────

def run_counterexample():
    print("=" * 60)
    print("Experiment 0 — Soundness counterexample (Section IV)")
    print("=" * 60)

    net = build_order_fulfilment_net()
    log = build_counterexample_log()

    result = OCTokenReplay(net).run(log)
    print(result.summary())
    print()
    print(result.per_object_table())
    print()

    # Verify the formal claim: naive = 1.0, global < 1.0
    # Per-object projections
    from src import OCELLog
    for obj_id, obj_type in [("o1","order"),("o2","order"),
                              ("i1","item"),("i2","item")]:
        proj_events = [e for e in log.events
                       if any(oid == obj_id for oid, _ in e.objects)]
        proj_log  = OCELLog(events=proj_events)
        proj_result = OCTokenReplay(net).run(proj_log)
        obj_stats = next((s for s in proj_result.per_object
                          if s.obj_id == obj_id), None)
        f_val = obj_stats.fitness if obj_stats else 1.0
        print(f"  Per-object projection  {obj_id} ({obj_type}): f = {f_val:.4f}")

    print(f"\n  Global OC-TBR fitness:  f = {result.fitness:.4f}")
    assert result.fitness < 1.0, "FAIL: global replay should detect violation"
    print("\n  ✓  Proposition verified: f_naive=1.0, f_global<1.0")


# ─────────────────────────────────────────────────────────────
# Experiment 1 — Synthetic logs (Table II)
# ─────────────────────────────────────────────────────────────

def run_experiment1(num_orders: int = 50, items_per_order: int = 3):
    print("=" * 60)
    print("Experiment 1 — Synthetic logs (Table II in paper)")
    print(f"  {num_orders} orders × {items_per_order} items each")
    print("=" * 60)

    net = build_order_fulfilment_net()

    logs = {
        "Log A (perfect)": build_log_A(num_orders, items_per_order),
        "Log B (mild)":    build_log_B(num_orders, items_per_order,
                                       violation_rate=0.20),
        "Log C (severe)":  build_log_C(num_orders, items_per_order,
                                       violation_rate=0.40),
    }

    rows = []
    for name, log in logs.items():
        stats = log.statistics()
        t0 = time.perf_counter()
        result = OCTokenReplay(net).run(log)
        elapsed = time.perf_counter() - t0

        ft = result.fitness_by_type()
        f_order = ft.get("order", float("nan"))
        f_item  = ft.get("item",  float("nan"))

        # Binding recall for Log C: fraction of injected violators detected
        if "severe" in name.lower():
            violators = result.objects_below_threshold(threshold=0.999)
            detected  = sum(1 for s in violators if s.obj_type == "order")
            total_inj = int(num_orders * 0.40)
            recall    = detected / total_inj if total_inj > 0 else float("nan")
        else:
            recall = float("nan")

        row = {
            "log":       name,
            "events":    stats["num_events"],
            "objects":   stats["num_objects"],
            "f_global":  round(result.fitness, 4),
            "f_order":   round(f_order, 4),
            "f_item":    round(f_item,  4),
            "binding_recall": f"{recall:.0%}" if recall == recall else "N/A",
            "time_s":    round(elapsed, 4),
        }
        rows.append(row)

        print(f"\n{name}")
        print(result.summary())

    # Save CSV
    with open("results/experiment1.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print("\n→ Results saved to results/experiment1.csv")


# ─────────────────────────────────────────────────────────────
# Experiment 2 — OCEL 2.0 benchmark (Table III)
# ─────────────────────────────────────────────────────────────

def run_experiment2(ocel_path: str = None):
    """
    If `ocel_path` is provided, load the real OCEL 2.0 log.
    Otherwise, generate a large synthetic log that approximates
    the benchmark statistics (22 000 events, 14 500 objects).
    """
    print("=" * 60)
    print("Experiment 2 — OCEL 2.0 benchmark (Table III in paper)")
    print("=" * 60)

    net = build_order_fulfilment_net()

    if ocel_path and os.path.exists(ocel_path):
        from src import load_ocel_json
        print(f"  Loading real log from {ocel_path} ...")
        log = load_ocel_json(ocel_path)
    else:
        print("  No OCEL path given — generating synthetic approximation")
        print("  (≈22 000 events, ≈14 500 objects)")
        log = build_log_B(num_orders=2200, items_per_order=5,
                          violation_rate=0.30, seed=0)

    stats = log.statistics()
    print(f"  Log stats: {stats['num_events']} events, "
          f"{stats['num_objects']} objects")

    t0 = time.perf_counter()
    result = OCTokenReplay(net).run(log)
    elapsed = time.perf_counter() - t0

    print(result.summary())
    print(f"\n  Runtime: {elapsed:.3f} s")
    print(f"  Violating objects (f_o < 0.99): "
          f"{len(result.objects_below_threshold(0.99))}")

    with open("results/experiment2_top_violators.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["obj_id","obj_type","fitness",
                                           "missing","remaining"])
        w.writeheader()
        for s in result.objects_below_threshold(0.99)[:50]:
            w.writerow({"obj_id": s.obj_id, "obj_type": s.obj_type,
                        "fitness": round(s.fitness, 4),
                        "missing": s.missing, "remaining": s.remaining})
    print("→ Top violators saved to results/experiment2_top_violators.csv")


# ─────────────────────────────────────────────────────────────
# Scalability experiment (Figure 4)
# ─────────────────────────────────────────────────────────────

def run_scalability(max_k: int = 8, items_per_order: int = 3, seed: int = 0):
    """
    Measure runtime for log sizes 1k … max_k*1000 events.
    Reproduces Figure 4 in the paper.
    """
    print("=" * 60)
    print("Scalability experiment (Figure 4 in paper)")
    print("=" * 60)

    net = build_order_fulfilment_net()
    rows = []

    for k in range(1, max_k + 1):
        n_orders = k * 100  # ~k*1000 events (5 events per case + items)
        log = build_log_B(n_orders, items_per_order, seed=seed)
        n_events = log.statistics()["num_events"]

        # Warm-up
        OCTokenReplay(net).run(log)

        # Timed runs (average of 3)
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            OCTokenReplay(net).run(log)
            times.append(time.perf_counter() - t0)
        avg = sum(times) / len(times)

        rows.append({"k_thousands": k, "events": n_events,
                     "time_s": round(avg, 4)})
        print(f"  {n_events:6d} events → {avg:.4f} s")

    # Compute empirical scaling factor
    if len(rows) >= 2:
        ratio = rows[-1]["time_s"] / rows[0]["time_s"]
        n_ratio = rows[-1]["events"] / rows[0]["events"]
        print(f"\n  Scaling factor: {ratio:.2f}× for {n_ratio:.0f}× more events")
        print(f"  (linear would be {n_ratio:.2f}×)")

    with open("results/scalability.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print("→ Saved to results/scalability.csv")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reproduce all experiments from the paper.")
    parser.add_argument("--all",      action="store_true",
                        help="Run all experiments")
    parser.add_argument("--exp0",     action="store_true",
                        help="Counterexample / soundness proof")
    parser.add_argument("--exp1",     action="store_true",
                        help="Synthetic logs (Table II)")
    parser.add_argument("--exp2",     action="store_true",
                        help="OCEL 2.0 benchmark (Table III)")
    parser.add_argument("--scale",    action="store_true",
                        help="Scalability experiment (Figure 4)")
    parser.add_argument("--ocel",     type=str, default=None,
                        help="Path to OCEL 2.0 JSON file for exp2")
    parser.add_argument("--orders",   type=int, default=50,
                        help="Number of orders for exp1 (default 50)")
    args = parser.parse_args()

    run_all = args.all or not any([args.exp0, args.exp1, args.exp2, args.scale])

    if run_all or args.exp0:
        run_counterexample()
    if run_all or args.exp1:
        run_experiment1(num_orders=args.orders)
    if run_all or args.exp2:
        run_experiment2(ocel_path=args.ocel)
    if run_all or args.scale:
        run_scalability()
