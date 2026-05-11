"""
OC-PN definitions used in experiments.

Functions:
  build_order_fulfilment_net()  — synthetic experiments (Fig. 1 in paper)
  build_real_log_net()          — manual net for example_log.jsonocel
  build_paper_example_net()     — manual net for running_example.jsonocel
  build_discovered_net()        — pm4py-discovered net for example_log.jsonocel
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import OCPetriNet, Place, Transition


# ══════════════════════════════════════════════════════════════════════
# 1. Order-fulfilment net — synthetic experiments (Fig. 1 in paper)
# ══════════════════════════════════════════════════════════════════════

def build_order_fulfilment_net() -> OCPetriNet:
    """
    Order-fulfilment OC-PN from Fig. 1 of the paper.
    Object types: order, item.

    Order: src_O -> [Place Order] -> p1_O -> [Pack Items] -> p2_O
                 -> [Pay]         -> p3_O -> [Ship Order] -> sink_O
    Item:  src_I -(var)-> [Pack Items] -(var)-> p1_I
                -(var)-> [Ship Order]  -(var)-> sink_I
    """
    net = OCPetriNet()

    src_O = Place("src_O", "order", is_source=True)
    p1_O = Place("p1_O", "order")
    p2_O = Place("p2_O", "order")
    p3_O = Place("p3_O", "order")
    sink_O = Place("sink_O", "order", is_sink=True)
    src_I = Place("src_I", "item", is_source=True)
    p1_I = Place("p1_I", "item")
    sink_I = Place("sink_I", "item", is_sink=True)

    net.places = [src_O, p1_O, p2_O, p3_O, sink_O,
                  src_I, p1_I, sink_I]

    net.transitions = [
        Transition("t_place", "Place Order",
                   preset={"order": [src_O]},
                   postset={"order": [p1_O]}),

        Transition("t_pack", "Pack Items",
                   preset={"order": [p1_O], "item": [src_I]},
                   postset={"order": [p2_O], "item": [p1_I]}),

        Transition("t_pay", "Pay",
                   preset={"order": [p2_O]},
                   postset={"order": [p3_O]}),

        Transition("t_ship", "Ship Order",
                   preset={"order": [p3_O], "item": [p1_I]},
                   postset={"order": [sink_O], "item": [sink_I]}),
    ]
    net.build_index()
    return net


# ══════════════════════════════════════════════════════════════════════
# 2. Manual net for example_log.jsonocel
#    (order, element, delivery — real order management process)
# ══════════════════════════════════════════════════════════════════════

def build_real_log_net() -> OCPetriNet:
    """
    OC-PN manually constructed from example_log.jsonocel.
    Object types: order, element, delivery.

    Order:    src_O -> [Create Order] -> p1_O -> [Confirm Order] -> p2_O
                    -> [Invoice Sent] -> p3_O -> [Pay Order]     -> sink_O
                                              -> [Reminder]      -> p4_O
                                              -> [Collection]    -> sink_O
                    -> [Cancel Order] -> sink_O

    Element:  src_E -> [Create Order]      -> p1_E -> [Create Delivery] -> sink_E
                    -> [Add Item to Order] -> p1_E  (loop back)
                    -> [Remove Item]       -> sink_E
                    -> [Item out of Stock] -> p2_E -> [Item back in Stock] -> p1_E

    Delivery: src_D -> [Create Delivery]    -> p1_D -> [Successful] -> sink_D
                                                    -> [Failed]     -> p2_D
                    -> [Retry Delivery]     -> p1_D
    """
    net = OCPetriNet()

    # Order
    src_O = Place("src_O", "order", is_source=True)
    p1_O = Place("p1_O", "order")
    p2_O = Place("p2_O", "order")
    p3_O = Place("p3_O", "order")
    p4_O = Place("p4_O", "order")
    sink_O = Place("sink_O", "order", is_sink=True)

    # Element
    src_E = Place("src_E", "element", is_source=True)
    p1_E = Place("p1_E", "element")
    p2_E = Place("p2_E", "element")
    sink_E = Place("sink_E", "element", is_sink=True)

    # Delivery
    src_D = Place("src_D", "delivery", is_source=True)
    p1_D = Place("p1_D", "delivery")
    p2_D = Place("p2_D", "delivery")
    sink_D = Place("sink_D", "delivery", is_sink=True)

    net.places = [
        src_O, p1_O, p2_O, p3_O, p4_O, sink_O,
        src_E, p1_E, p2_E, sink_E,
        src_D, p1_D, p2_D, sink_D,
    ]

    net.transitions = [
        Transition("t_create", "Create Order",
                   preset={"order": [src_O], "element": [src_E]},
                   postset={"order": [p1_O], "element": [p1_E]}),

        Transition("t_confirm", "Confirm Order",
                   preset={"order": [p1_O]},
                   postset={"order": [p2_O]}),

        Transition("t_add", "Add Item to Order",
                   preset={"order": [p1_O], "element": [src_E]},
                   postset={"order": [p1_O], "element": [p1_E]}),

        Transition("t_remove", "Remove Item",
                   preset={"element": [p1_E]},
                   postset={"element": [sink_E]}),

        Transition("t_out", "Item out of Stock",
                   preset={"element": [p1_E]},
                   postset={"element": [p2_E]}),

        Transition("t_back", "Item back in Stock",
                   preset={"element": [p2_E]},
                   postset={"element": [p1_E]}),

        Transition("t_delivery", "Create Delivery",
                   preset={"delivery": [src_D], "element": [p1_E]},
                   postset={"delivery": [p1_D], "element": [sink_E]}),

        Transition("t_success", "Delivery Successful",
                   preset={"delivery": [p1_D]},
                   postset={"delivery": [sink_D]}),

        Transition("t_failed", "Delivery Failed",
                   preset={"delivery": [p1_D]},
                   postset={"delivery": [p2_D]}),

        Transition("t_retry", "Retry Delivery",
                   preset={"delivery": [p2_D]},
                   postset={"delivery": [p1_D]}),

        Transition("t_invoice", "Invoice Sent",
                   preset={"order": [p2_O]},
                   postset={"order": [p3_O]}),

        Transition("t_pay", "Pay Order",
                   preset={"order": [p3_O]},
                   postset={"order": [sink_O]}),

        Transition("t_reminder", "Payment Reminder",
                   preset={"order": [p3_O]},
                   postset={"order": [p4_O]}),

        Transition("t_collection", "Send for Credit Collection",
                   preset={"order": [p4_O]},
                   postset={"order": [sink_O]}),

        Transition("t_cancel", "Cancel Order",
                   preset={"order": [p1_O]},
                   postset={"order": [sink_O]}),
    ]
    net.build_index()
    return net


# ══════════════════════════════════════════════════════════════════════
# 3. Manual net for running_example.jsonocel
#    (Package + Item — Liss et al. 2023 paper example)
# ══════════════════════════════════════════════════════════════════════

def build_paper_example_net() -> OCPetriNet:
    """
    OC-PN manually constructed from running_example.jsonocel.
    Source: Liss et al. 2023, paper-example.jsonocel
    Object types: Package, Item.

    Package: src_P -> [receive sample order] -> p1_P
                   -> [setup box]            -> p2_P
                   -> [add bill]             -> sink_P

    Item:    src_I -> [receive sample order] -> p1_I
                   -> [prepare sample]       -> p2_I
                   -> [add sample]           -> sink_I

    Note: receive sample order is shared (Package + Items, variable arc).
    """
    net = OCPetriNet()

    src_P = Place("src_P", "Package", is_source=True)
    p1_P = Place("p1_P", "Package")
    p2_P = Place("p2_P", "Package")
    sink_P = Place("sink_P", "Package", is_sink=True)

    src_I = Place("src_I", "Item", is_source=True)
    p1_I = Place("p1_I", "Item")
    p2_I = Place("p2_I", "Item")
    sink_I = Place("sink_I", "Item", is_sink=True)

    net.places = [src_P, p1_P, p2_P, sink_P,
                  src_I, p1_I, p2_I, sink_I]

    net.transitions = [
        Transition("t_receive", "receive sample order",
                   preset={"Package": [src_P], "Item": [src_I]},
                   postset={"Package": [p1_P], "Item": [p1_I]}),

        Transition("t_setup", "setup box",
                   preset={"Package": [p1_P]},
                   postset={"Package": [p2_P]}),

        Transition("t_bill", "add bill",
                   preset={"Package": [p2_P]},
                   postset={"Package": [sink_P]}),

        Transition("t_prepare", "prepare sample",
                   preset={"Item": [p1_I]},
                   postset={"Item": [p2_I]}),

        Transition("t_add", "add sample",
                   preset={"Item": [p2_I]},
                   postset={"Item": [sink_I]}),
    ]
    net.build_index()
    return net


# ══════════════════════════════════════════════════════════════════════
# 4. pm4py-discovered net for example_log.jsonocel
#    (translated from pm4py OCPetriNet format)
# ══════════════════════════════════════════════════════════════════════

def build_discovered_net() -> OCPetriNet:
    """
    OC-PN discovered by pm4py inductive miner from example_log.jsonocel
    and translated into our format.

    Delivery: source -> [Create Delivery] -> p_4 -> [Delivery Failed] -> p_5
                     -> [Retry Delivery]  -> p_3 -> [Delivery Successful] -> sink

    Element:  source -> [Create Order] -> p_7 -> [Add Item to Order] -> p_5
                     -> [Item out of Stock] -> p_6 -> [Item back in Stock] -> p_5
                     -> [Create Delivery]   -> sink
                     -> [Remove Item]       -> sink

    Order:    source -> [Create Order]       -> p_5 -> [Confirm Order] -> p_3
                     -> [Add Item to Order]  -> p_4 -> [Cancel Order]  -> sink
                     -> [Invoice Sent]       -> p_7 -> [Pay Order]     -> sink
                                                    -> [Reminder]      -> p_8
                                                    -> [Collection]    -> sink
    """
    net = OCPetriNet()

    # Delivery
    d_src = Place("d_source", "delivery", is_source=True)
    d_p3 = Place("d_p3", "delivery")
    d_p4 = Place("d_p4", "delivery")
    d_p5 = Place("d_p5", "delivery")
    d_sink = Place("d_sink", "delivery", is_sink=True)

    # Element
    e_src = Place("e_source", "element", is_source=True)
    e_p5 = Place("e_p5", "element")
    e_p6 = Place("e_p6", "element")
    e_p7 = Place("e_p7", "element")
    e_sink = Place("e_sink", "element", is_sink=True)

    # Order
    o_src = Place("o_source", "order", is_source=True)
    o_p3 = Place("o_p3", "order")
    o_p4 = Place("o_p4", "order")
    o_p5 = Place("o_p5", "order")
    o_p7 = Place("o_p7", "order")
    o_p8 = Place("o_p8", "order")
    o_sink = Place("o_sink", "order", is_sink=True)

    net.places = [
        d_src, d_p3, d_p4, d_p5, d_sink,
        e_src, e_p5, e_p6, e_p7, e_sink,
        o_src, o_p3, o_p4, o_p5, o_p7, o_p8, o_sink,
    ]

    net.transitions = [
        Transition("t_create_order", "Create Order",
                   preset={"order": [o_src], "element": [e_src]},
                   postset={"order": [o_p5], "element": [e_p7]}),

        Transition("t_confirm", "Confirm Order",
                   preset={"order": [o_p5]},
                   postset={"order": [o_p3]}),

        Transition("t_add_item", "Add Item to Order",
                   preset={"order": [o_p3], "element": [e_p7]},
                   postset={"order": [o_p4], "element": [e_p5]}),

        Transition("t_cancel", "Cancel Order",
                   preset={"order": [o_p4]},
                   postset={"order": [o_sink]}),

        Transition("t_invoice", "Invoice Sent",
                   preset={"order": [o_p4]},
                   postset={"order": [o_p7]}),

        Transition("t_pay", "Pay Order",
                   preset={"order": [o_p7]},
                   postset={"order": [o_sink]}),

        Transition("t_reminder", "Payment Reminder",
                   preset={"order": [o_p7]},
                   postset={"order": [o_p8]}),

        Transition("t_collection", "Send for Credit Collection",
                   preset={"order": [o_p8]},
                   postset={"order": [o_sink]}),

        Transition("t_out_of_stock", "Item out of Stock",
                   preset={"element": [e_p5]},
                   postset={"element": [e_p6]}),

        Transition("t_back_in_stock", "Item back in Stock",
                   preset={"element": [e_p6]},
                   postset={"element": [e_p5]}),

        Transition("t_remove", "Remove Item",
                   preset={"element": [e_p7]},
                   postset={"element": [e_sink]}),

        Transition("t_create_delivery", "Create Delivery",
                   preset={"delivery": [d_src], "element": [e_p5]},
                   postset={"delivery": [d_p4], "element": [e_sink]}),

        Transition("t_failed", "Delivery Failed",
                   preset={"delivery": [d_p4]},
                   postset={"delivery": [d_p5]}),

        Transition("t_retry", "Retry Delivery",
                   preset={"delivery": [d_p5]},
                   postset={"delivery": [d_p3]}),

        Transition("t_success", "Delivery Successful",
                   preset={"delivery": [d_p3]},
                   postset={"delivery": [d_sink]}),
    ]
    net.build_index()
    return net
