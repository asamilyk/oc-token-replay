"""
experiments/run_real_logs.py

Experiment 3 — OC-TBR on three real OCEL logs.
OC-PN is discovered automatically via pm4py for each log.

Usage:
    python -m experiments.run_real_logs

Downloads logs first if needed:
    python scripts/download_data.py
"""
import sys
import os
import csv
import time

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from run_conformance import load_ocel_file, discover_net
from src.replay import OCTokenReplay

RESULTS = os.path.join(ROOT, "results")
DATA = os.path.join(ROOT, "data")
os.makedirs(RESULTS, exist_ok=True)

LOGS = [
    {
        "file": "example_log.jsonocel",
        "name": "Order Management",
        "desc": "Real order management log, 3 object types",
        "source": "pm4py test data",
    },
    {
        "file": "running_example.jsonocel",
        "name": "Paper Example",
        "desc": "Package fulfilment process, 2 object types",
        "source": "Liss et al. 2023",
        "manual_net": True,
    },
    {
        "file": "p2p_normal.jsonocel",
        "name": "Procure-to-Pay",
        "desc": "P2P process, multiple object types",
        "source": "ocpa library",
    },
]


def run_one(log_config):
    path = os.path.join(DATA, log_config["file"])
    if not os.path.exists(path):
        print(f"  NOT FOUND: {path}")
        print(f"  Run: python scripts/download_data.py")
        return None

    print(f"\n  Loading {log_config['file']} ...")
    our_log, pm4py_ocel, _ = load_ocel_file(path)

    print(f"  Discovering OC-PN via pm4py ...")
    net = discover_net(pm4py_ocel)
    print(f"  Net: {len(net.places)} places, "
          f"{len(net.transitions)} transitions")

    t0 = time.perf_counter()
    result = OCTokenReplay(net).run(our_log)
    elapsed = time.perf_counter() - t0

    ft = result.fitness_by_type()
    violators = [s for s in result.per_object if s.fitness < 0.95]

    print(f"  Events   : {len(our_log.events)}")
    print(f"  Objects  : {len(result.per_object)}")
    print(f"  f global : {result.fitness:.4f}")
    print(f"  Runtime  : {elapsed * 1000:.2f} ms")
    print(f"  Violators: {len(violators)} / {len(result.per_object)}")
    print(f"  Per-type fitness:")
    for ot, fv in sorted(ft.items()):
        print(f"    {ot:<15}: {fv:.4f}")

    # ── сравнение с ручной сетью (только для Paper Example) ──────────
    if log_config.get("manual_net"):
        from experiments.nets import build_paper_example_net
        net_manual = build_paper_example_net()
        print(f"\n  Manual net: {len(net_manual.places)} places, "
              f"{len(net_manual.transitions)} transitions")

        t0_m = time.perf_counter()
        result_manual = OCTokenReplay(net_manual).run(our_log)
        t_manual = time.perf_counter() - t0_m
        ft_manual = result_manual.fitness_by_type()

        print(f"\n  {'Metric':<30} {'Auto (pm4py)':>14} {'Manual':>10}")
        print(f"  {'─' * 56}")
        print(f"  {'Global fitness f':<30} "
              f"{result.fitness:>14.4f} "
              f"{result_manual.fitness:>10.4f}")
        print(f"  {'Runtime':<30} "
              f"{elapsed * 1000:>13.2f}ms "
              f"{t_manual * 1000:>9.2f}ms")

        all_types = sorted(set(list(ft.keys()) + list(ft_manual.keys())))
        for ot in all_types:
            fa = ft.get(ot, 0)
            fm = ft_manual.get(ot, 0)
            print(f"  {'f_tau (' + ot + ')':<30} {fa:>14.4f} {fm:>10.4f}")

        print(f"\n  {'Object':<12} {'Type':<10} "
              f"{'f (auto)':>10} {'f (manual)':>12}")
        print(f"  {'─' * 46}")
        manual_map = {s.obj_id: s for s in result_manual.per_object}
        for s in sorted(result.per_object,
                        key=lambda x: (x.obj_type, x.obj_id)):
            fm = manual_map.get(s.obj_id)
            fm_v = f"{fm.fitness:.4f}" if fm else "N/A"
            diff = ""
            if fm and abs(s.fitness - fm.fitness) > 0.01:
                diff = "  ← differs"
            print(f"  {s.obj_id:<12} {s.obj_type:<10} "
                  f"{s.fitness:>10.4f} {fm_v:>12}{diff}")

        cmp_path = os.path.join(
            RESULTS,
            log_config["file"].replace(".jsonocel", "_auto_vs_manual.csv")
        )
        with open(cmp_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["obj_id", "obj_type",
                        "f_auto", "miss_auto", "rem_auto",
                        "f_manual", "miss_manual", "rem_manual"])
            for s in sorted(result.per_object,
                            key=lambda x: (x.obj_type, x.obj_id)):
                fm = manual_map.get(s.obj_id)
                w.writerow([
                    s.obj_id, s.obj_type,
                    round(s.fitness, 4), s.missing, s.remaining,
                    round(fm.fitness, 4) if fm else "",
                    fm.missing if fm else "",
                    fm.remaining if fm else "",
                ])
        print(f"  Comparison saved → {cmp_path}")
    # per-object CSV
    csv_name = log_config["file"].replace(".jsonocel", "_results.csv")
    csv_path = os.path.join(RESULTS, csv_name)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["obj_id", "obj_type", "fitness",
                    "produced", "consumed", "missing", "remaining"])
        for s in sorted(result.per_object,
                        key=lambda x: (x.obj_type, x.fitness)):
            w.writerow([s.obj_id, s.obj_type, round(s.fitness, 4),
                        s.produced, s.consumed, s.missing, s.remaining])
    print(f"  Saved → {csv_path}")

    return {
        "name": log_config["name"],
        "file": log_config["file"],
        "source": log_config["source"],
        "n_events": len(our_log.events),
        "n_objects": len(result.per_object),
        "n_types": len(ft),
        "f": round(result.fitness, 4),
        "time_ms": round(elapsed * 1000, 2),
        "violators": len(violators),
        "f_manual": round(result_manual.fitness, 4)
        if log_config.get("manual_net") and result_manual
        else "",
        **{f"f_{ot}": round(fv, 4) for ot, fv in ft.items()},
    }


if __name__ == "__main__":
    print("=" * 65)
    print("  Experiment 3 — OC-TBR on Real OCEL Logs")
    print("  OC-PN discovered automatically via pm4py")
    print("=" * 65)

    summary_rows = []

    for log_cfg in LOGS:
        print(f"\n{'─' * 65}")
        print(f"  {log_cfg['name']}  ({log_cfg['source']})")
        print(f"  {log_cfg['desc']}")
        row = run_one(log_cfg)
        if row:
            summary_rows.append(row)

    # summary CSV
    if summary_rows:
        summary_path = os.path.join(RESULTS, "experiment3_real_logs.csv")

        all_fields = []
        for row in summary_rows:
            for key in row.keys():
                if key not in all_fields:
                    all_fields.append(key)

        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(summary_rows)
        print(f"\nSummary saved → {summary_path}")
