"""
replay.py — OC-TBR: Object-Centric Token-Based Replay.

Implements Algorithm 1 from the paper:

  Input : OC-PN N, OCEL L
  Output: Global counters P_g, C_g, M_g, R_g;
          per-object (p_o, c_o, m_o, r_o);
          three-level fitness f, f_o, f_tau.

Steps:
  1. Object layout  — group objects by type (byType[tau])
  2. Initial markings — one token per source place per object
  3. Replay loop    — for each event, for each bound object:
       3a. find the matching transition
       3b. classical replay step (consume / produce)
  4. Remaining tokens — count leftover tokens per object

Time complexity: O(|L| * |O| * |P|)

References:
  Samilyk A. (2026). Checking the Conformance of Object-Centric
  Petri Nets and Event Logs using a Token Replay Approach. HSE.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict

from .model   import OCPetriNet, Transition, Place
from .log     import OCELLog
from .marking import Marking


# =============================================================
#  Result structures
# =============================================================

@dataclass
class ObjectStats:
    """
    Token counters and fitness for a single object.

    p_o: tokens produced (including initial source token)
    c_o: tokens consumed
    m_o: missing tokens (artificial tokens created)
    r_o: remaining tokens at end of case
    fired: sequence of activities this object participated in
    """
    obj_id:   str
    obj_type: str
    produced:  int = 0
    consumed:  int = 0
    missing:   int = 0
    remaining: int = 0
    fired: list[str] = field(default_factory=list)

    @property
    def fitness(self) -> float:
        """
        Per-object fitness formula (Eq. 2 in the paper):
          f_o = 1 - 0.5 * (m_o / c_o + r_o / p_o)
        """
        if self.consumed == 0 and self.produced == 0:
            return 1.0
        part_m = self.missing   / self.consumed  if self.consumed  > 0 else 0.0
        part_r = self.remaining / self.produced  if self.produced  > 0 else 0.0
        return 1.0 - 0.5 * (part_m + part_r)

    def __repr__(self):
        return (f"ObjectStats(id={self.obj_id!r}, type={self.obj_type!r}, "
                f"fitness={self.fitness:.4f}, "
                f"p={self.produced}, c={self.consumed}, "
                f"m={self.missing}, r={self.remaining})")


@dataclass
class ReplayResult:
    """
    Aggregated result for the entire log.

    Provides:
      - global fitness f  (Eq. 1)
      - per-type fitness f_tau  (Eq. 3)
      - per-object stats (ObjectStats list)
    """
    per_object: list[ObjectStats] = field(default_factory=list)

    # Global counters (accumulated across all objects/cases)
    P_g: int = 0   # total produced
    C_g: int = 0   # total consumed
    M_g: int = 0   # total missing
    R_g: int = 0   # total remaining

    @property
    def fitness(self) -> float:
        """
        Global fitness (Eq. 1):
          f = 1 - 0.5 * (M_g / C_g + R_g / P_g)
        """
        if self.C_g == 0 and self.P_g == 0:
            return 1.0
        part_m = self.M_g / self.C_g if self.C_g > 0 else 0.0
        part_r = self.R_g / self.P_g if self.P_g > 0 else 0.0
        return 1.0 - 0.5 * (part_m + part_r)

    def fitness_by_type(self) -> dict[str, float]:
        """
        Per-type fitness (Eq. 3):
          f_tau = 1 - 0.5 * (sum(m_o) / sum(c_o) + sum(r_o) / sum(p_o))
          for all objects o with type(o) = tau
        """
        buckets: dict[str, dict] = defaultdict(
            lambda: {"p": 0, "c": 0, "m": 0, "r": 0})
        for s in self.per_object:
            b = buckets[s.obj_type]
            b["p"] += s.produced
            b["c"] += s.consumed
            b["m"] += s.missing
            b["r"] += s.remaining

        result = {}
        for tau, b in buckets.items():
            part_m = b["m"] / b["c"] if b["c"] > 0 else 0.0
            part_r = b["r"] / b["p"] if b["p"] > 0 else 0.0
            result[tau] = 1.0 - 0.5 * (part_m + part_r)
        return result

    def objects_below_threshold(self, threshold: float = 0.9) -> list[ObjectStats]:
        """Return objects with fitness < threshold, sorted ascending."""
        return sorted(
            [s for s in self.per_object if s.fitness < threshold],
            key=lambda s: s.fitness)

    def summary(self) -> str:
        lines = [
            "─" * 52,
            "  OC-TBR — Replay Results",
            "─" * 52,
            f"  Objects processed : {len(self.per_object)}",
            f"  Global fitness  f : {self.fitness:.4f}",
            f"  Produced  P_g    : {self.P_g}",
            f"  Consumed  C_g    : {self.C_g}",
            f"  Missing   M_g    : {self.M_g}",
            f"  Remaining R_g    : {self.R_g}",
            "",
            "  Per-type fitness f_tau:",
        ]
        for tau, f in sorted(self.fitness_by_type().items()):
            lines.append(f"    {tau:20s}: {f:.4f}")
        lines.append("─" * 52)
        return "\n".join(lines)

    def per_object_table(self) -> str:
        """Detailed per-object table for inspection."""
        header = f"{'Object':8s} {'Type':8s} {'fitness':>8s} {'miss':>6s} {'rem':>6s}  Trace"
        sep    = "-" * 70
        rows   = [header, sep]
        for s in sorted(self.per_object, key=lambda x: (x.obj_type, x.obj_id)):
            trace = "→".join(s.fired) if s.fired else "(none)"
            rows.append(
                f"{s.obj_id:8s} {s.obj_type:8s} {s.fitness:8.3f} "
                f"{s.missing:6d} {s.remaining:6d}  {trace}"
            )
        return "\n".join(rows)


# =============================================================
#  Main algorithm
# =============================================================

class OCTokenReplay:
    """
    OC-TBR: Object-Centric Token-Based Replay.

    Usage:
        net = build_my_net()
        log = load_ocel_json("my_log.json")
        result = OCTokenReplay(net).run(log)
        print(result.summary())
    """

    def __init__(self, net: OCPetriNet):
        self.net = net

    def run(self, log: OCELLog) -> ReplayResult:
        """Execute OC-TBR and return a ReplayResult."""
        net = self.net
        net.build_index()

        # ── Step 1: Object layout ─────────────────────────────
        objects_by_type: dict[str, set[str]] = log.all_objects()

        # ── Step 2: Initial markings ──────────────────────────
        markings:  dict[str, Marking]     = {}
        stats_map: dict[str, ObjectStats] = {}

        for obj_type, obj_ids in objects_by_type.items():
            source = net.source_for(obj_type)
            for obj_id in obj_ids:
                m  = Marking()
                st = ObjectStats(obj_id=obj_id, obj_type=obj_type)
                if source is not None:
                    m.add(source)
                    st.produced += 1
                markings[obj_id]  = m
                stats_map[obj_id] = st

        # ── Step 3: Replay loop ───────────────────────────────
        for event in log.sorted_events():
            transition = net.get_transition(event.activity)
            if transition is None:
                # Activity not in model — skip silently
                # (could log a warning here if needed)
                continue

            # Step 3a: iterate over objects bound to this event
            for obj_id, obj_type in event.objects:
                if obj_id not in markings:
                    continue   # object not seen during init

                # Step 3b: classical replay step for this object
                self._replay_step(
                    transition=transition,
                    obj_type=obj_type,
                    marking=markings[obj_id],
                    stats=stats_map[obj_id],
                    activity=event.activity,
                )

        # ── Step 4: Remaining tokens ──────────────────────────
        for obj_id, marking in markings.items():
            stats_map[obj_id].remaining = marking.total()

        # ── Accumulate global counters ────────────────────────
        result = ReplayResult(per_object=list(stats_map.values()))
        for s in result.per_object:
            result.P_g += s.produced
            result.C_g += s.consumed
            result.M_g += s.missing
            result.R_g += s.remaining

        return result

    # ─────────────────────────────────────────────────────────
    def _replay_step(
        self,
        transition: Transition,
        obj_type:   str,
        marking:    Marking,
        stats:      ObjectStats,
        activity:   str,
    ) -> None:
        """
        Classical token-replay step for one object.

        Consume from preset(t, type), produce into postset(t, type).
        If a required token is missing, create it artificially and
        increment m_o (permissive replay).
        """
        preset_places  = transition.preset.get(obj_type,  [])
        postset_places = transition.postset.get(obj_type, [])

        # Consume
        for place in preset_places:
            missing = marking.remove(place)
            if missing > 0:
                stats.missing += missing  # artificial token
            stats.consumed += 1

        # Produce
        for place in postset_places:
            marking.add(place)
            stats.produced += 1

        stats.fired.append(activity)
