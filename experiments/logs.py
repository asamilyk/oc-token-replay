"""
experiments/logs.py — Synthetic log generators for Experiments 1 and 2.

Three log types as described in Section VI of the paper:
  - Log A (perfect):  all objects follow the conformant trace
  - Log B (mild):     20% of orders skip Pack before Pay
  - Log C (severe):   40% of cases contain cross-object binding violation

Reproduces the exact setup used in Table II of the paper.
"""

import sys, os, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.log import OCELLog, OCEvent


# ── helpers ──────────────────────────────────────────────────

def _ev(eid, act, ts, *obj_pairs):
    """Shorthand for building an OCEvent."""
    return OCEvent(id=eid, activity=act, timestamp=float(ts),
                   objects=list(obj_pairs))


# ── Log A: perfect conformance ────────────────────────────────

def build_log_A(num_orders: int = 50, items_per_order: int = 2,
                seed: int = 42) -> OCELLog:
    """
    Perfectly conformant log.
    Every order follows: Place -> Pack -> Pay -> Ship -> Close
    Every item follows:  Pack -> Ship -> Deliver
    """
    rng = random.Random(seed)
    events = []
    ts = 1.0

    for k in range(num_orders):
        oid = f"o{k}"
        iids = [f"i{k}_{j}" for j in range(items_per_order)]

        events.append(_ev(f"e_pl_{k}", "Place Order", ts,
                          (oid, "order")))
        ts += 1
        item_pairs = [(i, "item") for i in iids]
        events.append(_ev(f"e_pk_{k}", "Pack Items", ts,
                          (oid, "order"), *item_pairs))
        ts += 1
        events.append(_ev(f"e_pa_{k}", "Pay", ts,
                          (oid, "order")))
        ts += 1
        events.append(_ev(f"e_sh_{k}", "Ship Order", ts,
                          (oid, "order"), *item_pairs))
        ts += 1
        events.append(_ev(f"e_cl_{k}", "Close Order", ts,
                          (oid, "order")))
        ts += 1
        for iid in iids:
            events.append(_ev(f"e_dl_{k}_{iid}", "Deliver Item", ts,
                               (iid, "item")))
            ts += 1

    return OCELLog(events=events)


# ── Log B: mild — 20% of orders skip Pack ─────────────────────

def build_log_B(num_orders: int = 50, items_per_order: int = 2,
                violation_rate: float = 0.2, seed: int = 42) -> OCELLog:
    """
    Mild non-conformance: `violation_rate` fraction of orders skip Pack.
    This introduces one missing token per such order at t_pay.
    """
    rng = random.Random(seed)
    events = []
    ts = 1.0

    for k in range(num_orders):
        oid  = f"o{k}"
        iids = [f"i{k}_{j}" for j in range(items_per_order)]
        skip_pack = rng.random() < violation_rate

        events.append(_ev(f"e_pl_{k}", "Place Order", ts,
                          (oid, "order")))
        ts += 1

        if not skip_pack:
            item_pairs = [(i, "item") for i in iids]
            events.append(_ev(f"e_pk_{k}", "Pack Items", ts,
                              (oid, "order"), *item_pairs))
            ts += 1
        else:
            # Items are still "added" to the log but Pack is skipped for order
            item_pairs = [(i, "item") for i in iids]

        events.append(_ev(f"e_pa_{k}", "Pay", ts,
                          (oid, "order")))
        ts += 1
        events.append(_ev(f"e_sh_{k}", "Ship Order", ts,
                          (oid, "order"), *item_pairs))
        ts += 1
        events.append(_ev(f"e_cl_{k}", "Close Order", ts,
                          (oid, "order")))
        ts += 1
        for iid in iids:
            events.append(_ev(f"e_dl_{k}_{iid}", "Deliver Item", ts,
                               (iid, "item")))
            ts += 1

    return OCELLog(events=events)


# ── Log C: severe — cross-object binding violation ────────────

def build_log_C(num_orders: int = 50, items_per_order: int = 2,
                violation_rate: float = 0.4, seed: int = 42) -> OCELLog:
    """
    Severe non-conformance: in `violation_rate` fraction of *pairs* of
    consecutive orders, the items packed for order k are shipped under
    order k+1 (cross-object binding violation, Table I in the paper).

    This is exactly the counterexample from Section IV, scaled up.
    """
    rng  = random.Random(seed)
    events = []
    ts   = 1.0

    # Build order/item data first
    orders = []
    for k in range(num_orders):
        oid  = f"o{k}"
        iids = [f"i{k}_{j}" for j in range(items_per_order)]
        orders.append((oid, iids))

    # Decide which consecutive pairs will be swapped
    swapped = set()
    k = 0
    while k < num_orders - 1:
        if rng.random() < violation_rate:
            swapped.add(k)
            k += 2   # skip next to avoid overlap
        else:
            k += 1

    for k, (oid, iids) in enumerate(orders):
        events.append(_ev(f"e_pl_{k}", "Place Order", ts, (oid, "order")))
        ts += 1

        pack_item_pairs = [(i, "item") for i in iids]
        events.append(_ev(f"e_pk_{k}", "Pack Items", ts,
                          (oid, "order"), *pack_item_pairs))
        ts += 1

        events.append(_ev(f"e_pa_{k}", "Pay", ts, (oid, "order")))
        ts += 1

    # Ship phase: swapped pairs ship under the wrong order
    for k, (oid, iids) in enumerate(orders):
        if k in swapped:
            # Items of order k are shipped under order k+1
            ship_oid       = orders[k + 1][0]
            ship_item_pairs = [(i, "item") for i in iids]
        else:
            ship_oid       = oid
            ship_item_pairs = [(i, "item") for i in iids]

        events.append(_ev(f"e_sh_{k}", "Ship Order", ts,
                          (ship_oid, "order"), *ship_item_pairs))
        ts += 1
        events.append(_ev(f"e_cl_{k}", "Close Order", ts, (oid, "order")))
        ts += 1
        for iid in iids:
            events.append(_ev(f"e_dl_{k}_{iid}", "Deliver Item", ts,
                               (iid, "item")))
            ts += 1

    return OCELLog(events=events)


# ── Counterexample log (Section IV) ──────────────────────────

def build_counterexample_log() -> OCELLog:
    """
    The exact four-event counterexample from Table I of the paper.
    Used in unit tests and Section IV illustration.
    """
    return OCELLog(events=[
        _ev("e1", "Place Order", 1.0, ("o1", "order")),
        _ev("e2", "Place Order", 2.0, ("o2", "order")),
        _ev("e3", "Pack Items",  3.0, ("o1", "order"),
            ("i1", "item"), ("i2", "item")),
        _ev("e4", "Ship Order",  4.0, ("o2", "order"),
            ("i1", "item"), ("i2", "item")),
    ])
