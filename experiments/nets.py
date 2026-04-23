"""
experiments/nets.py — Reference OC-PNs used in all experiments.

Provides the order-fulfilment net from Figure 1 of the paper,
plus a larger net for scalability experiments.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model import Place, Transition, OCPetriNet


def build_order_fulfilment_net() -> OCPetriNet:
    """
    OC-PN for the order-fulfilment process (Figure 1 in the paper).

    Object types: order, item

    Order subnet:
      p_Os(src) -> Place Order -> p_O1 -> Pack Items -> p_O2
               -> Pay -> p_O3 -> Ship Order -> p_O4
               -> Close Order -> p_Oe(sink)

    Item subnet (variable arcs via Pack Items and Ship Order):
      p_Is(src) --var--> Pack Items --var--> p_I1
                         --var--> Ship Order --var--> p_Ie(sink)
      p_Ie -> Deliver Item -> p_I_sink(sink)

    Transition 'Pack Items' takes one order token and N item tokens
    (variable arc on item side). Same for 'Ship Order'.
    """
    net = OCPetriNet()

    # ── Order places ──────────────────────────────────────────
    p_Os = Place("p_Os", "order", is_source=True)
    p_O1 = Place("p_O1", "order")
    p_O2 = Place("p_O2", "order")
    p_O3 = Place("p_O3", "order")
    p_O4 = Place("p_O4", "order")
    p_Oe = Place("p_Oe", "order", is_sink=True)

    # ── Item places ───────────────────────────────────────────
    p_Is   = Place("p_Is",   "item", is_source=True)
    p_I1   = Place("p_I1",   "item")
    p_Ie   = Place("p_Ie",   "item")
    p_Isink= Place("p_Isink","item", is_sink=True)

    net.places = [p_Os, p_O1, p_O2, p_O3, p_O4, p_Oe,
                  p_Is, p_I1, p_Ie, p_Isink]

    # ── Transitions ───────────────────────────────────────────
    t_place = Transition(
        id="t_place", activity="Place Order",
        preset  = {"order": [p_Os]},
        postset = {"order": [p_O1]},
    )
    t_pack = Transition(
        id="t_pack", activity="Pack Items",
        # order: consume p_O1, produce p_O2 (stays in packing state)
        # item:  consume p_Is (variable), produce p_I1 (variable)
        preset  = {"order": [p_O1], "item": [p_Is]},
        postset = {"order": [p_O2], "item": [p_I1]},
    )
    t_pay = Transition(
        id="t_pay", activity="Pay",
        preset  = {"order": [p_O2]},
        postset = {"order": [p_O3]},
    )
    t_ship = Transition(
        id="t_ship", activity="Ship Order",
        # order: consume p_O3, re-emit p_O4
        # item:  consume p_I1 (variable), produce p_Ie (variable)
        preset  = {"order": [p_O3], "item": [p_I1]},
        postset = {"order": [p_O4], "item": [p_Ie]},
    )
    t_close = Transition(
        id="t_close", activity="Close Order",
        preset  = {"order": [p_O4]},
        postset = {"order": [p_Oe]},
    )
    t_deliver = Transition(
        id="t_deliver", activity="Deliver Item",
        preset  = {"item": [p_Ie]},
        postset = {"item": [p_Isink]},
    )

    net.transitions = [t_place, t_pack, t_pay, t_ship, t_close, t_deliver]

    # Mark p_Isink as sink so remaining tokens there don't count
    # (tokens in sink are expected — they are not "remaining" in spirit,
    #  but the formula counts them; tests must account for this)
    return net


def build_large_net(num_types: int = 4) -> OCPetriNet:
    """
    Parametric OC-PN for scalability experiments.

    Creates `num_types` independent subnets, each with 5 places
    and 4 transitions, sharing one 'Sync' transition that
    consumes/produces from all types simultaneously.
    """
    net = OCPetriNet()
    shared_preset  = {}
    shared_postset = {}

    for i in range(num_types):
        t_name = f"type_{i}"
        src  = Place(f"src_{i}",   t_name, is_source=True)
        p1   = Place(f"p1_{i}",    t_name)
        p2   = Place(f"p2_{i}",    t_name)
        p3   = Place(f"p3_{i}",    t_name)
        sink = Place(f"sink_{i}",  t_name, is_sink=True)
        net.places += [src, p1, p2, p3, sink]

        net.transitions += [
            Transition(f"t_a{i}", f"A_{i}",
                       preset={t_name: [src]}, postset={t_name: [p1]}),
            Transition(f"t_b{i}", f"B_{i}",
                       preset={t_name: [p1]},  postset={t_name: [p2]}),
            Transition(f"t_c{i}", f"C_{i}",
                       preset={t_name: [p3]},  postset={t_name: [sink]}),
        ]
        shared_preset[t_name]  = [p2]
        shared_postset[t_name] = [p3]

    net.transitions.append(Transition(
        "t_sync", "Sync",
        preset=shared_preset, postset=shared_postset,
    ))
    return net
