"""
OC-PN definitions used in experiments.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import OCPetriNet, Place, Transition


def build_order_fulfilment_net() -> OCPetriNet:
    """
    Order-fulfilment OC-PN from Fig. 1 of the paper.
    Object types: order, item.
    Flow:
      Order: src_O -> Place -> p1_O -> Pack -> p2_O -> Pay -> p3_O -> Ship -> sink_O
      Item:  src_I -(var)-> Pack -(var)-> p1_I -(var)-> Ship -(var)-> sink_I
    """
    net = OCPetriNet()

    # Order places
    src_O  = Place("src_O",  "order", is_source=True)
    p1_O   = Place("p1_O",   "order")
    p2_O   = Place("p2_O",   "order")
    p3_O   = Place("p3_O",   "order")
    sink_O = Place("sink_O", "order", is_sink=True)

    # Item places
    src_I  = Place("src_I",  "item", is_source=True)
    p1_I   = Place("p1_I",   "item")
    sink_I = Place("sink_I", "item", is_sink=True)

    net.places = [src_O, p1_O, p2_O, p3_O, sink_O,
                  src_I, p1_I, sink_I]

    t_place = Transition("t_place", "Place Order",
        preset  = {"order": [src_O]},
        postset = {"order": [p1_O]})

    t_pack  = Transition("t_pack", "Pack Items",
        preset  = {"order": [p1_O], "item": [src_I]},
        postset = {"order": [p2_O], "item": [p1_I]})

    t_pay   = Transition("t_pay", "Pay",
        preset  = {"order": [p2_O]},
        postset = {"order": [p3_O]})

    t_ship  = Transition("t_ship", "Ship Order",
        preset  = {"order": [p3_O], "item": [p1_I]},
        postset = {"order": [sink_O], "item": [sink_I]})

    net.transitions = [t_place, t_pack, t_pay, t_ship]
    net.build_index()
    return net