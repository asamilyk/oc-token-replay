"""
run_conformance.py

Universal OC-TBR conformance checker.
Discovers OC-PN from the log via pm4py, then runs OC-TBR.

Usage:
    python run_conformance.py data/example_log.jsonocel
    python run_conformance.py path/to/any_log.jsonocel
    python run_conformance.py path/to/log.jsonocel --threshold 0.9
"""

import sys
import os
import json
import argparse
import time
from collections import defaultdict

import pandas as pd
import pm4py
from pm4py.objects.ocel.obj import OCEL

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from src.model import OCPetriNet, Place, Transition
from src.log import OCELLog, OCEvent
from src.replay import OCTokenReplay


# ── Step 1: load any OCEL 1.0 JSON file ──────────────────────────────

def load_ocel_file(path):
    """Load OCEL 1.0 JSON and return both our OCELLog and pm4py OCEL."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    raw_objects = data["ocel:objects"]
    events_raw = data["ocel:events"]

    # ── our format ────────────────────────────────────────────────────
    our_events = []
    for eid, ev in events_raw.items():
        obj_list = [
            (oid, raw_objects[oid]["ocel:type"])
            for oid in ev["ocel:omap"]
            if oid in raw_objects
        ]
        our_events.append(OCEvent(
            id=eid,
            activity=ev["ocel:activity"],
            timestamp=ev["ocel:timestamp"],
            objects=obj_list,
        ))
    our_events.sort(key=lambda e: e.timestamp)
    our_log = OCELLog(events=our_events)

    # ── pm4py format ──────────────────────────────────────────────────
    events_rows = []
    relations_rows = []
    objects_rows = []
    seen = set()

    for eid, ev in events_raw.items():
        ts = pd.Timestamp(ev["ocel:timestamp"])
        events_rows.append({
            "ocel:eid": eid,
            "ocel:activity": ev["ocel:activity"],
            "ocel:timestamp": ts,
        })
        for oid in ev["ocel:omap"]:
            if oid not in raw_objects:
                continue
            otype = raw_objects[oid]["ocel:type"]
            relations_rows.append({
                "ocel:eid": eid,
                "ocel:activity": ev["ocel:activity"],
                "ocel:timestamp": ts,
                "ocel:oid": oid,
                "ocel:type": otype,
            })
            if oid not in seen:
                objects_rows.append({"ocel:oid": oid, "ocel:type": otype})
                seen.add(oid)

    pm4py_ocel = OCEL(
        events=pd.DataFrame(events_rows),
        objects=pd.DataFrame(objects_rows),
        relations=pd.DataFrame(relations_rows),
    )
    pm4py_ocel.event_id_column = "ocel:eid"
    pm4py_ocel.event_activity = "ocel:activity"
    pm4py_ocel.event_timestamp = "ocel:timestamp"
    pm4py_ocel.object_id_column = "ocel:oid"
    pm4py_ocel.object_type_column = "ocel:type"
    pm4py_ocel.event_corr_object_id = "ocel:oid"
    pm4py_ocel.event_corr_object_type = "ocel:type"

    return our_log, pm4py_ocel, raw_objects


# ── Step 2: discover OC-PN and convert to our format ─────────────────

def discover_net(pm4py_ocel):
    """Discover OC-PN via pm4py and translate to our OCPetriNet format."""
    ocpn = pm4py.discover_oc_petri_net(pm4py_ocel)

    net = OCPetriNet()

    # maps (obj_type, pm4py_place_name) -> our Place object
    place_map = {}

    # ── build places ──────────────────────────────────────────────────
    for obj_type, (pn, im, fm) in ocpn["petri_nets"].items():
        for p in pn.places:
            is_source = p in im and im[p] > 0
            is_sink = p in fm and fm[p] > 0
            our_place = Place(
                id=f"{obj_type}__{p.name}",
                object_type=obj_type,
                is_source=is_source,
                is_sink=is_sink,
            )
            place_map[(obj_type, p.name)] = our_place
            net.places.append(our_place)

    # ── build transitions ─────────────────────────────────────────────
    # collect all unique activity labels across all subnets
    all_activities = set()
    for obj_type, (pn, im, fm) in ocpn["petri_nets"].items():
        for t in pn.transitions:
            if t.label:
                all_activities.add(t.label)

    # for each activity, gather preset and postset per object type
    for activity in all_activities:
        preset = defaultdict(list)
        postset = defaultdict(list)

        for obj_type, (pn, im, fm) in ocpn["petri_nets"].items():
            for t in pn.transitions:
                if t.label != activity:
                    continue
                for arc in pn.arcs:
                    if arc.target == t:
                        p = place_map.get((obj_type, arc.source.name))
                        if p:
                            preset[obj_type].append(p)
                    if arc.source == t:
                        p = place_map.get((obj_type, arc.target.name))
                        if p:
                            postset[obj_type].append(p)

        if not preset and not postset:
            continue

        our_t = Transition(
            id=activity.replace(" ", "_"),
            activity=activity,
            preset=dict(preset),
            postset=dict(postset),
        )
        net.transitions.append(our_t)

    net.build_index()
    return net


# ── Step 3: run OC-TBR and print results ─────────────────────────────

def run_and_print(our_log, net, threshold, log_path):
    t0 = time.perf_counter()
    result = OCTokenReplay(net).run(our_log)
    elapsed = time.perf_counter() - t0

    print("\n" + "=" * 65)
    print(f"  OC-TBR Conformance Report")
    print(f"  Log  : {os.path.basename(log_path)}")
    print(f"  Model: discovered via pm4py (inductive miner)")
    print("=" * 65)

    print(f"\n  Events processed  : {len(our_log.events)}")
    print(f"  Objects processed : {len(result.per_object)}")
    print(f"  Runtime           : {elapsed * 1000:.2f} ms")
    print(f"\n  Global fitness  f : {result.fitness:.4f}")

    ft = result.fitness_by_type()
    print(f"\n  Per-type fitness:")
    for ot, fv in sorted(ft.items()):
        bar = "█" * int(fv * 20) + "░" * (20 - int(fv * 20))
        print(f"    {ot:<15} {bar}  {fv:.4f}")

    violators = [s for s in result.per_object if s.fitness < threshold]
    conformant = [s for s in result.per_object if s.fitness >= threshold]

    print(f"\n  Threshold: {threshold}")
    print(f"  Conformant objects : {len(conformant)}")
    print(f"  Violating objects  : {len(violators)}")

    print(f"\n  {'obj_id':<15} {'type':<12} {'fitness':>8} "
          f"{'prod':>6} {'cons':>6} {'miss':>6} {'rem':>6}")
    print("  " + "-" * 68)

    all_objs = sorted(result.per_object,
                      key=lambda s: (s.fitness, s.obj_type, s.obj_id))
    for s in all_objs:
        flag = "  ← violator" if s.fitness < threshold else ""
        print(f"  {s.obj_id:<15} {s.obj_type:<12} {s.fitness:>8.3f} "
              f"{s.produced:>6} {s.consumed:>6} "
              f"{s.missing:>6} {s.remaining:>6}{flag}")

    # save CSV
    out_dir = os.path.join(os.path.dirname(log_path))
    csv_name = os.path.splitext(os.path.basename(log_path))[0] + "_conformance.csv"
    csv_path = os.path.join(out_dir, csv_name)

    import csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["obj_id", "obj_type", "fitness",
                    "produced", "consumed", "missing", "remaining"])
        for s in sorted(result.per_object,
                        key=lambda x: (x.obj_type, x.obj_id)):
            w.writerow([s.obj_id, s.obj_type, round(s.fitness, 4),
                        s.produced, s.consumed, s.missing, s.remaining])

    print(f"\n  Results saved → {csv_path}")
    print("=" * 65)

    return result


# ── Entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run OC-TBR on any OCEL 1.0 log with auto-discovered OC-PN"
    )
    parser.add_argument("log",
                        help="Path to OCEL 1.0 JSON file (.jsonocel)")
    parser.add_argument("--threshold", type=float, default=0.95,
                        help="Fitness threshold below which objects are flagged (default: 0.95)")
    parser.add_argument("--show-net", action="store_true",
                        help="Visualise the discovered OC-PN (requires graphviz)")
    args = parser.parse_args()

    if not os.path.exists(args.log):
        print(f"File not found: {args.log}")
        sys.exit(1)

    print(f"Loading  {args.log} ...")
    our_log, pm4py_ocel, raw_objects = load_ocel_file(args.log)
    print(f"  {len(our_log.events)} events, "
          f"{len(raw_objects)} objects")

    print("Discovering OC-PN via pm4py ...")
    net = discover_net(pm4py_ocel)
    print(f"  {len(net.places)} places, {len(net.transitions)} transitions")

    if args.show_net:
        pm4py.view_ocpn(pm4py.discover_oc_petri_net(pm4py_ocel))

    run_and_print(our_log, net, args.threshold, args.log)
