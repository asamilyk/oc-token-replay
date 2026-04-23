"""
model.py — Object-Centric Petri Net (OC-PN) data structures.

Implements Definition 1 from the paper:
  ON = (N, pt, F_var)
  where N = (P, T, F, l) is a labelled Petri net,
        pt : P -> OT maps places to object types,
        F_var ⊆ F is the subset of variable arcs.

References:
  Reijers et al. (2021). Discovering Object-Centric Petri Nets. Petri Nets 2021.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Place:
    """A place in the OC-PN, typed to an object type."""
    id: str
    object_type: str        # pt(p) — maps to object type
    is_source: bool = False # source place of the type's subnet
    is_sink:   bool = False # sink place of the type's subnet
    is_variable: bool = False  # True if all arcs to/from this place are variable

    def __hash__(self):    return hash(self.id)
    def __eq__(self, o):   return isinstance(o, Place) and self.id == o.id
    def __repr__(self):    return f"Place({self.id}, type={self.object_type})"


@dataclass
class Transition:
    """
    A transition in the OC-PN.

    preset  = { object_type: [places] }  — input places grouped by type
    postset = { object_type: [places] }  — output places grouped by type

    This grouped structure allows O(1) lookup during per-object replay.
    """
    id: str
    activity: str           # l(t) — activity label

    # Input/output places split by object type for efficient lookup
    preset:  dict[str, list[Place]] = field(default_factory=dict)
    postset: dict[str, list[Place]] = field(default_factory=dict)

    def __hash__(self):    return hash(self.id)
    def __eq__(self, o):   return isinstance(o, Transition) and self.id == o.id
    def __repr__(self):    return f"Transition({self.id}, act={self.activity!r})"


@dataclass
class OCPetriNet:
    """
    Object-Centric Petri Net.

    Contains multiple type-specific subnets unified through
    shared transitions. Each object type has its own source
    and sink places.
    """
    places:      list[Place]      = field(default_factory=list)
    transitions: list[Transition] = field(default_factory=list)

    # Internal indices — built by build_index()
    _activity_index: dict[str, Transition] = field(
        default_factory=dict, init=False, repr=False)
    _sources: dict[str, Place] = field(
        default_factory=dict, init=False, repr=False)
    _sinks:   dict[str, Place] = field(
        default_factory=dict, init=False, repr=False)

    def build_index(self) -> None:
        """Build internal lookup indices. Must be called after construction."""
        self._activity_index = {t.activity: t for t in self.transitions}
        self._sources = {}
        self._sinks   = {}
        for p in self.places:
            if p.is_source:
                self._sources[p.object_type] = p
            if p.is_sink:
                self._sinks[p.object_type] = p

    def get_transition(self, activity: str) -> Optional[Transition]:
        return self._activity_index.get(activity)

    def source_for(self, obj_type: str) -> Optional[Place]:
        return self._sources.get(obj_type)

    def sink_for(self, obj_type: str) -> Optional[Place]:
        return self._sinks.get(obj_type)

    def object_types(self) -> list[str]:
        return sorted({p.object_type for p in self.places})

    def summary(self) -> str:
        return (f"OCPetriNet: {len(self.places)} places, "
                f"{len(self.transitions)} transitions, "
                f"types={self.object_types()}")
