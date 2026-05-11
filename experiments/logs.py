"""
Log generators and OCEL parsers used in experiments.
"""
import json
import random
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from log import OCELLog, OCEvent


def generate_synthetic_log(n_orders: int,
                           deviation_rate: float,
                           seed: int = 42) -> OCELLog:
    """
    Generate a synthetic order-fulfilment OCEL log.

    Parameters
    ----------
    n_orders : int
        Number of orders to generate.
    deviation_rate : float
        Fraction of orders that skip Pack Items.
        0.0 = Log A (perfect), 0.2 = Log B (mild), 0.4 = Log C (severe).
    seed : int
        Random seed for reproducibility.
    """
    random.seed(seed)
    events = []
    ts = 1.0
    eid = 1
    item_counter = 1

    # assign items to orders up front (1–4 items per order, avg 2.5)
    order_items = {}
    for k in range(1, n_orders + 1):
        oid = f"o{k}"
        n_items = random.randint(1, 4)
        iids = [f"i{item_counter + j}" for j in range(n_items)]
        item_counter += n_items
        order_items[oid] = iids

    order_ids = list(order_items.keys())

    # choose deviating orders deterministically
    n_deviant = int(n_orders * deviation_rate)
    deviant_set = set(random.sample(order_ids, n_deviant))

    def add(activity, objs):
        nonlocal ts, eid
        events.append(OCEvent(f"e{eid}", activity, ts, objs))
        ts += 1.0
        eid += 1

    for oid in order_ids:
        add("Place Order", [(oid, "order")])

    for oid in order_ids:
        if oid not in deviant_set:
            objs = [(oid, "order")] + [(i, "item") for i in order_items[oid]]
            add("Pack Items", objs)

    for oid in order_ids:
        add("Pay", [(oid, "order")])

    for oid in order_ids:
        objs = [(oid, "order")] + [(i, "item") for i in order_items[oid]]
        add("Ship Order", objs)

    return OCELLog(events=events)


def load_ocel1(path: str) -> OCELLog:
    """
    Parse an OCEL 1.0 JSON file into OCELLog.

    In OCEL 1.0 the ocel:omap field contains plain object IDs.
    Object types are resolved from the ocel:objects dictionary.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    raw_objects = data["ocel:objects"]
    events = []

    for eid, ev in data["ocel:events"].items():
        obj_list = [
            (oid, raw_objects[oid]["ocel:type"])
            for oid in ev["ocel:omap"]
            if oid in raw_objects
        ]
        events.append(OCEvent(
            id=eid,
            activity=ev["ocel:activity"],
            timestamp=ev["ocel:timestamp"],
            objects=obj_list,
        ))

    events.sort(key=lambda e: e.timestamp)
    return OCELLog(events=events)
