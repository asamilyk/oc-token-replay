"""
log.py — Object-Centric Event Log (OCEL) data structures and importers.

Implements the OCEL standard (OCEL 2.0):
  L = (E, O, act, time, omap, OT, type)

References:
  Ghahfarokhi et al. (2021). OCEL: A Standard for Object-Centric
  Event Logs. ADBIS 2021 Short Papers, CCIS vol. 1450, pp. 169-175.
"""

from __future__ import annotations
import json
import csv
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional


@dataclass
class OCEvent:
    """
    A single event in an object-centric event log.

    objects: list of (object_id, object_type) pairs — omap(e)
    """
    id:        str
    activity:  str                    # act(e)
    timestamp: float                  # time(e), numeric for sorting
    objects:   list[tuple[str, str]]  # [(obj_id, obj_type), ...]

    def object_ids_of_type(self, obj_type: str) -> list[str]:
        return [oid for oid, ot in self.objects if ot == obj_type]

    def __repr__(self):
        return (f"OCEvent(id={self.id!r}, act={self.activity!r}, "
                f"t={self.timestamp}, objs={self.objects})")


@dataclass
class OCELLog:
    """
    Object-Centric Event Log.

    events: list of OCEvent, not necessarily sorted.
    Use sorted_events() to iterate in timestamp order.
    """
    events: list[OCEvent] = field(default_factory=list)

    def sorted_events(self) -> list[OCEvent]:
        """Return events sorted by timestamp (stable sort)."""
        return sorted(self.events, key=lambda e: e.timestamp)

    def all_objects(self) -> dict[str, set[str]]:
        """
        Returns {obj_type: {obj_id, ...}} across the entire log.
        Used in Step 1 of OC-TBR (object layout).
        """
        result: dict[str, set[str]] = defaultdict(set)
        for e in self.events:
            for obj_id, obj_type in e.objects:
                result[obj_type].add(obj_id)
        return dict(result)

    def statistics(self) -> dict:
        objs = self.all_objects()
        return {
            "num_events":  len(self.events),
            "num_objects": sum(len(v) for v in objs.values()),
            "object_types": {k: len(v) for k, v in objs.items()},
            "activities":  sorted({e.activity for e in self.events}),
        }

    def __repr__(self):
        s = self.statistics()
        return (f"OCELLog({s['num_events']} events, "
                f"{s['num_objects']} objects, "
                f"types={list(s['object_types'].keys())})")


# =============================================================
#  Importers
# =============================================================

def load_ocel_json(path: str) -> OCELLog:
    """
    Load an OCEL 2.0 JSON file.

    Expected schema (OCEL 2.0):
    {
      "ocel:events": {
        "<event_id>": {
          "ocel:activity": str,
          "ocel:timestamp": str | float,
          "ocel:omap": ["<obj_id>", ...]
        }
      },
      "ocel:objects": {
        "<obj_id>": {
          "ocel:type": str
        }
      }
    }
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    # Build object-type map
    obj_type: dict[str, str] = {}
    for oid, odata in raw.get("ocel:objects", {}).items():
        obj_type[oid] = odata.get("ocel:type", "unknown")

    events = []
    for eid, edata in raw.get("ocel:events", {}).items():
        ts_raw = edata.get("ocel:timestamp", 0)
        # Handle ISO timestamp strings
        if isinstance(ts_raw, str):
            try:
                from datetime import datetime
                ts = datetime.fromisoformat(
                    ts_raw.replace("Z", "+00:00")).timestamp()
            except ValueError:
                ts = 0.0
        else:
            ts = float(ts_raw)

        objects = [
            (oid, obj_type.get(oid, "unknown"))
            for oid in edata.get("ocel:omap", [])
        ]
        events.append(OCEvent(
            id=eid,
            activity=edata.get("ocel:activity", ""),
            timestamp=ts,
            objects=objects,
        ))

    log = OCELLog(events=events)
    return log


def load_ocel_csv(events_path: str, objects_path: str) -> OCELLog:
    """
    Load from two CSV files:
      events_path : event_id, activity, timestamp, object_ids (semicolon-sep)
      objects_path: object_id, object_type
    """
    obj_type: dict[str, str] = {}
    with open(objects_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            obj_type[row["object_id"]] = row["object_type"]

    events = []
    with open(events_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            oids = [o.strip() for o in row["object_ids"].split(";") if o.strip()]
            objects = [(oid, obj_type.get(oid, "unknown")) for oid in oids]
            events.append(OCEvent(
                id=row["event_id"],
                activity=row["activity"],
                timestamp=float(row["timestamp"]),
                objects=objects,
            ))
    return OCELLog(events=events)


def make_synthetic_log(events: list[dict]) -> OCELLog:
    """
    Convenience constructor for tests and examples.

    events: list of dicts with keys:
      id, activity, timestamp, objects (list of [obj_id, obj_type])
    """
    return OCELLog(events=[
        OCEvent(
            id=e["id"],
            activity=e["activity"],
            timestamp=float(e["timestamp"]),
            objects=[(o[0], o[1]) for o in e["objects"]],
        )
        for e in events
    ])
