import sys, os, json, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--log", required=True)
parser.add_argument("--result", required=True)
parser.add_argument("--repo", required=True)
args = parser.parse_args()

sys.path.insert(0, args.repo)

from algorithm import calculate_oc_alignments
from localocpa.objects.log.importer.ocel import factory as ocel_import_factory
from localocpa.objects.oc_petri_net.obj import ObjectCentricPetriNet as OCPN


def build_order_fulfilment_ocpn():
    ocpn = OCPN(name="Order Fulfilment")
    src_O = OCPN.Place(name="src_O", object_type="order", initial=True)
    p1_O = OCPN.Place(name="p1_O", object_type="order")
    p2_O = OCPN.Place(name="p2_O", object_type="order")
    p3_O = OCPN.Place(name="p3_O", object_type="order")
    sink_O = OCPN.Place(name="sink_O", object_type="order", final=True)
    src_I = OCPN.Place(name="src_I", object_type="item", initial=True)
    p1_I = OCPN.Place(name="p1_I", object_type="item")
    sink_I = OCPN.Place(name="sink_I", object_type="item", final=True)
    for p in [src_O, p1_O, p2_O, p3_O, sink_O, src_I, p1_I, sink_I]:
        ocpn.places.add(p)

    t_place = OCPN.Transition(name="Place Order", label="Place Order")
    t_pack = OCPN.Transition(name="Pack Items", label="Pack Items")
    t_pay = OCPN.Transition(name="Pay", label="Pay")
    t_ship = OCPN.Transition(name="Ship Order", label="Ship Order")
    for t in [t_place, t_pack, t_pay, t_ship]:
        ocpn.transitions.add(t)

    ocpn.add_arc(OCPN.Arc(src_O, t_place))
    ocpn.add_arc(OCPN.Arc(t_place, p1_O))
    ocpn.add_arc(OCPN.Arc(p1_O, t_pack))
    ocpn.add_arc(OCPN.Arc(t_pack, p2_O))
    ocpn.add_arc(OCPN.Arc(src_I, t_pack, variable=True))
    ocpn.add_arc(OCPN.Arc(t_pack, p1_I, variable=True))
    ocpn.add_arc(OCPN.Arc(p2_O, t_pay))
    ocpn.add_arc(OCPN.Arc(t_pay, p3_O))
    ocpn.add_arc(OCPN.Arc(p3_O, t_ship))
    ocpn.add_arc(OCPN.Arc(t_ship, sink_O))
    ocpn.add_arc(OCPN.Arc(p1_I, t_ship, variable=True))
    ocpn.add_arc(OCPN.Arc(t_ship, sink_I, variable=True))
    return ocpn


ocel = ocel_import_factory.apply(args.log)
ocpn = build_order_fulfilment_ocpn()
alignments = calculate_oc_alignments(ocel, ocpn)

fitnesses = []
for exec_id, aln in alignments.items():
    if aln is None:
        continue
    n = len(aln.moves)
    c = aln.get_cost()
    f = max(0.0, min(1.0, 1.0 - c / n)) if n > 0 else 1.0
    fitnesses.append(f)

avg = sum(fitnesses) / len(fitnesses) if fitnesses else None

with open(args.result, "w") as f:
    json.dump({
        "fitness": avg,
        "n_executions": len(alignments),
        "n_with_result": len(fitnesses),
        "per_execution": fitnesses,
    }, f, indent=2)
