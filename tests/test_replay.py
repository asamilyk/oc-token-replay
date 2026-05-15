"""
tests/test_replay.py — Unit tests for OC-TBR.

Tests are based on the actual implementation in src/ and
the experiment helpers in experiments/.

Run from project root:
    pytest tests/test_replay.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src import OCTokenReplay, OCELLog, OCEvent
from experiments.nets import build_order_fulfilment_net
from experiments.logs import generate_synthetic_log


# ─────────────────────────────────────────────────────────────
# Helpers: small hand-crafted logs
# ─────────────────────────────────────────────────────────────

def make_conformant_log():
    """
    Minimal conformant log: o1 follows Place->Pack->Pay->Ship correctly.
    i1, i2 are bound to o1 at Pack and Ship.
    """
    return OCELLog(events=[
        OCEvent("e1", "Place Order", 1.0,
                [("o1", "order")]),
        OCEvent("e2", "Pack Items", 2.0,
                [("o1", "order"), ("i1", "item"), ("i2", "item")]),
        OCEvent("e3", "Pay", 3.0,
                [("o1", "order")]),
        OCEvent("e4", "Ship Order", 4.0,
                [("o1", "order"), ("i1", "item"), ("i2", "item")]),
    ])


def make_deviant_log():
    """
    Running example from Section III of the paper.
    o2 skips Pack Items and goes straight to Pay.
    i3 is bound to o2 at Ship but was never packed.
    """
    return OCELLog(events=[
        OCEvent("e1", "Place Order", 1.0, [("o1", "order")]),
        OCEvent("e2", "Place Order", 2.0, [("o2", "order")]),
        OCEvent("e3", "Pack Items", 3.0,
                [("o1", "order"), ("i1", "item"), ("i2", "item")]),
        OCEvent("e4", "Pay", 4.0, [("o1", "order")]),
        OCEvent("e5", "Pay", 5.0, [("o2", "order")]),  # o2 skips Pack
        OCEvent("e6", "Ship Order", 6.0,
                [("o1", "order"), ("i1", "item"), ("i2", "item")]),
        OCEvent("e7", "Ship Order", 7.0,
                [("o2", "order"), ("i3", "item")]),
    ])


def make_swap_log():
    """
    Log D from Experiment 1: o2 executes Pay BEFORE Pack Items.
    This reverses the required activity order and creates both
    a missing token (Pay fires before token reaches p2_O) and
    a remaining token (Pack moves token to p2_O after Pay already
    consumed it artificially).
    """
    return OCELLog(events=[
        OCEvent("e1", "Place Order", 1.0, [("o1", "order")]),
        OCEvent("e2", "Place Order", 2.0, [("o2", "order")]),
        OCEvent("e3", "Pay", 3.0, [("o2", "order")]),  # swap: Pay first
        OCEvent("e4", "Pack Items", 4.0,
                [("o1", "order"), ("i1", "item"),
                 ("o2", "order"), ("i2", "item")]),
        OCEvent("e5", "Pay", 5.0, [("o1", "order")]),
        OCEvent("e6", "Ship Order", 6.0,
                [("o1", "order"), ("i1", "item"),
                 ("o2", "order"), ("i2", "item")]),
    ])


def make_counterexample_log():
    """
    Counterexample for the unsoundness of per-object projection
    (Section IV of the paper).
    o1 packs i1 and i2, but they are shipped under o2.
    Per-object projections of i1 and i2 each see Pack->Ship
    and report missing=0, while global replay detects the
    binding violation.
    """
    return OCELLog(events=[
        OCEvent("e1", "Place Order", 1.0, [("o1", "order")]),
        OCEvent("e2", "Place Order", 2.0, [("o2", "order")]),
        OCEvent("e3", "Pack Items", 3.0,
                [("o1", "order"), ("i1", "item"), ("i2", "item")]),
        OCEvent("e4", "Ship Order", 4.0,
                [("o2", "order"), ("i1", "item"), ("i2", "item")]),
    ])


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def net():
    return build_order_fulfilment_net()


@pytest.fixture
def replay(net):
    return OCTokenReplay(net)


# ─────────────────────────────────────────────────────────────
# 1. Perfect conformance
# ─────────────────────────────────────────────────────────────

class TestPerfectConformance:

    def test_no_missing_tokens(self, replay):
        """Conformant log must produce zero missing tokens."""
        result = replay.run(make_conformant_log())
        total_missing = sum(s.missing for s in result.per_object)
        assert total_missing == 0, \
            f"Expected 0 missing tokens, got {total_missing}"

    def test_global_fitness_is_one(self, replay):
        """Global fitness must be 1.0 for a perfectly conformant log."""
        result = replay.run(make_conformant_log())
        assert result.fitness == pytest.approx(1.0, abs=1e-9)

    def test_all_per_object_fitness_is_one(self, replay):
        """Every individual object must have f_o = 1.0."""
        result = replay.run(make_conformant_log())
        for s in result.per_object:
            assert s.fitness == pytest.approx(1.0, abs=1e-9), \
                f"Object {s.obj_id} has f_o={s.fitness:.4f}, expected 1.0"

    def test_per_type_fitness_is_one(self, replay):
        """All per-type scores must be 1.0 on a conformant log."""
        result = replay.run(make_conformant_log())
        for tau, fv in result.fitness_by_type().items():
            assert fv == pytest.approx(1.0, abs=1e-9), \
                f"f_tau({tau}) = {fv:.4f}, expected 1.0"

    def test_synthetic_log_A_no_missing(self, replay):
        """Synthetically generated Log A (0% deviation) → 0 missing tokens."""
        log = generate_synthetic_log(n_orders=20, deviation_rate=0.0, seed=42)
        result = replay.run(log)
        total_missing = sum(s.missing for s in result.per_object)
        assert total_missing == 0


# ─────────────────────────────────────────────────────────────
# 2. Known violations and per-object detection
# ─────────────────────────────────────────────────────────────

class TestViolations:

    def test_deviant_log_fitness_below_one(self, replay):
        """A log with deviations must yield global fitness < 1.0."""
        result = replay.run(make_deviant_log())
        assert result.fitness < 1.0, \
            f"Deviant log reported perfect fitness {result.fitness}"

    def test_o2_flagged_as_violator(self, replay):
        """o2 (which skipped Pack Items) must have missing > 0."""
        result = replay.run(make_deviant_log())
        o2 = next((s for s in result.per_object if s.obj_id == "o2"), None)
        assert o2 is not None, "o2 not found in per_object results"
        assert o2.missing > 0, \
            f"o2 should be a violator but has missing={o2.missing}"

    def test_i3_flagged_as_violator(self, replay):
        """
        i3 was never packed (o2 skipped Pack Items) so when Ship
        fires for i3, no token exists at p1_I → missing > 0.
        """
        result = replay.run(make_deviant_log())
        i3 = next((s for s in result.per_object if s.obj_id == "i3"), None)
        assert i3 is not None, "i3 not found in per_object results"
        assert i3.missing > 0, \
            f"i3 should be a violator but has missing={i3.missing}"

    def test_conformant_objects_not_penalised(self, replay):
        """o1, i1, i2 follow the model correctly and must have f_o = 1.0."""
        result = replay.run(make_deviant_log())
        for obj_id in ["o1", "i1", "i2"]:
            s = next((s for s in result.per_object if s.obj_id == obj_id), None)
            assert s is not None, f"{obj_id} not found in results"
            assert s.fitness == pytest.approx(1.0, abs=1e-9), \
                f"{obj_id} should be conformant but has f_o={s.fitness:.4f}"

    def test_more_deviations_lower_fitness(self, replay):
        """Log C (40% deviation) must have strictly lower fitness than Log B (20%)."""
        log_b = generate_synthetic_log(n_orders=30, deviation_rate=0.2, seed=42)
        log_c = generate_synthetic_log(n_orders=30, deviation_rate=0.4, seed=42)
        r_b = replay.run(log_b)
        r_c = replay.run(log_c)
        assert r_c.fitness < r_b.fitness, \
            f"Log C fitness ({r_c.fitness:.4f}) should be below " \
            f"Log B ({r_b.fitness:.4f})"

    def test_swap_deviation_detected(self, replay):
        """
        Log D: o2 executes Pay before Pack Items.
        This creates both missing and remaining tokens for o2,
        while o1 (conformant) must retain f_o = 1.0.
        """
        result = replay.run(make_swap_log())
        o2 = next(s for s in result.per_object if s.obj_id == "o2")
        assert o2.missing > 0, \
            f"swap should cause missing token for o2, got missing={o2.missing}"
        assert o2.remaining > 0, \
            f"swap should cause remaining token for o2, got remaining={o2.remaining}"
        o1 = next(s for s in result.per_object if s.obj_id == "o1")
        assert o1.fitness == pytest.approx(1.0, abs=1e-9), \
            f"o1 is conformant and should have f_o=1.0, got {o1.fitness:.4f}"

    def test_swap_global_fitness_below_one(self, replay):
        """Global fitness on the swap log must be < 1.0."""
        result = replay.run(make_swap_log())
        assert result.fitness < 1.0


# ─────────────────────────────────────────────────────────────
# 3. Unsoundness of per-object projection (Section IV)
# ─────────────────────────────────────────────────────────────

class TestSoundness:

    def test_per_object_projection_misses_binding_violation(self, net):
        """
        Per-object projection is unsound: i1 and i2 each see
        Pack->Ship in their individual projection and report
        missing=0, even though they were packed under o1 but
        shipped under o2 — a cross-object binding violation.
        """
        log = make_counterexample_log()
        for obj_id in ["i1", "i2"]:
            proj_events = [
                e for e in log.events
                if any(oid == obj_id for oid, _ in e.objects)
            ]
            proj_log = OCELLog(events=proj_events)
            result = OCTokenReplay(net).run(proj_log)
            s = next((x for x in result.per_object if x.obj_id == obj_id), None)
            missing = s.missing if s else 0
            assert missing == 0, \
                (f"Per-object projection for {obj_id} sees Pack->Ship "
                 f"and should report missing=0 (unsoundness), got {missing}")

    def test_global_replay_detects_binding_violation(self, replay):
        """
        Global OC-TBR must detect the binding violation that
        per-object projection silently misses.
        """
        log = make_counterexample_log()
        result = replay.run(log)
        total_missing = sum(s.missing for s in result.per_object)
        assert total_missing > 0, \
            "Global replay should detect missing tokens in the counterexample"

    def test_global_fitness_below_one_in_counterexample(self, replay):
        """
        Global fitness must be < 1.0 while per-object projections
        would report 1.0 — this is the unsoundness demonstrated
        in Section IV of the paper.
        """
        log = make_counterexample_log()
        result = replay.run(log)
        assert result.fitness < 1.0


# ─────────────────────────────────────────────────────────────
# 4. Fitness formula correctness (Equations 1, 2, 3)
# ─────────────────────────────────────────────────────────────

class TestFormulas:

    def test_per_object_formula(self, replay):
        """
        Verify Eq. (2): f_o = 1 - 0.5 * (m_o/c_o + r_o/p_o)
        for every object in the deviant log.
        """
        result = replay.run(make_deviant_log())
        for s in result.per_object:
            if s.consumed == 0 and s.produced == 0:
                continue
            expected = 1.0 - 0.5 * (
                    (s.missing / s.consumed if s.consumed > 0 else 0.0) +
                    (s.remaining / s.produced if s.produced > 0 else 0.0)
            )
            assert s.fitness == pytest.approx(expected, abs=1e-9), \
                f"Eq.(2) mismatch for {s.obj_id}: " \
                f"got {s.fitness:.6f}, expected {expected:.6f}"

    def test_global_formula(self, replay):
        """
        Verify Eq. (1): f = 1 - 0.5 * (M_g/C_g + R_g/P_g)
        matches result.fitness.
        """
        result = replay.run(make_deviant_log())
        expected = 1.0 - 0.5 * (
                (result.M_g / result.C_g if result.C_g > 0 else 0.0) +
                (result.R_g / result.P_g if result.P_g > 0 else 0.0)
        )
        assert result.fitness == pytest.approx(expected, abs=1e-9), \
            f"Eq.(1) mismatch: got {result.fitness:.6f}, " \
            f"expected {expected:.6f}"

    def test_per_type_formula(self, replay):
        """
        Verify Eq. (3): f_tau aggregates m, c, r, p across all
        objects of the same type.
        """
        result = replay.run(make_deviant_log())
        ft = result.fitness_by_type()
        for tau in ["order", "item"]:
            group = [s for s in result.per_object if s.obj_type == tau]
            if not group:
                continue
            p_sum = sum(s.produced for s in group)
            c_sum = sum(s.consumed for s in group)
            m_sum = sum(s.missing for s in group)
            r_sum = sum(s.remaining for s in group)
            expected = 1.0 - 0.5 * (
                    (m_sum / c_sum if c_sum > 0 else 0.0) +
                    (r_sum / p_sum if p_sum > 0 else 0.0)
            )
            assert ft[tau] == pytest.approx(expected, abs=1e-9), \
                f"Eq.(3) mismatch for type={tau}: " \
                f"got {ft[tau]:.6f}, expected {expected:.6f}"

    def test_fitness_in_unit_interval(self, replay):
        """All fitness values must lie in [0.0, 1.0]."""
        logs = [
            make_conformant_log(),
            make_deviant_log(),
            make_swap_log(),
            make_counterexample_log(),
            generate_synthetic_log(10, 0.0),
            generate_synthetic_log(10, 0.4),
        ]
        for log in logs:
            result = replay.run(log)
            assert 0.0 <= result.fitness <= 1.0, \
                f"Global fitness out of [0,1]: {result.fitness}"
            for s in result.per_object:
                assert 0.0 <= s.fitness <= 1.0, \
                    f"f_o out of [0,1] for {s.obj_id}: {s.fitness}"

    def test_conformant_better_than_deviant(self, replay):
        """Conformant log must always score higher than the deviant log."""
        r_good = replay.run(make_conformant_log())
        r_bad = replay.run(make_deviant_log())
        assert r_good.fitness > r_bad.fitness


# ─────────────────────────────────────────────────────────────
# 5. Global counter consistency
# ─────────────────────────────────────────────────────────────

class TestCounters:

    def test_global_counters_equal_sum_of_per_object(self, replay):
        """
        Global counters P_g, C_g, M_g, R_g must equal the sum of
        the corresponding per-object counters (Section IV, Algorithm 1).
        """
        result = replay.run(make_deviant_log())
        assert result.P_g == sum(s.produced for s in result.per_object), \
            "P_g != sum of per-object produced"
        assert result.C_g == sum(s.consumed for s in result.per_object), \
            "C_g != sum of per-object consumed"
        assert result.M_g == sum(s.missing for s in result.per_object), \
            "M_g != sum of per-object missing"
        assert result.R_g == sum(s.remaining for s in result.per_object), \
            "R_g != sum of per-object remaining"

    def test_produced_geq_consumed(self, replay):
        """
        For each object, produced tokens >= consumed tokens
        (initial marking contributes to produced but not to consumed).
        """
        for log in [make_conformant_log(), make_deviant_log(), make_swap_log()]:
            result = replay.run(log)
            for s in result.per_object:
                assert s.produced >= s.consumed, \
                    f"{s.obj_id}: produced={s.produced} < consumed={s.consumed}"


# ─────────────────────────────────────────────────────────────
# 6. Edge cases
# ─────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_empty_log(self, replay):
        """Empty log must not crash, must return fitness=1.0
        and an empty per_object list."""
        result = replay.run(OCELLog(events=[]))
        assert result.fitness == pytest.approx(1.0, abs=1e-9)
        assert len(result.per_object) == 0

    def test_unknown_activity_skipped(self, replay):
        """
        An event whose activity label is not in the model must be
        skipped silently without raising an exception and without
        creating any missing tokens.
        """
        log = OCELLog(events=[
            OCEvent("e1", "ACTIVITY_NOT_IN_MODEL", 1.0,
                    [("o1", "order")]),
        ])
        result = replay.run(log)
        total_missing = sum(s.missing for s in result.per_object)
        assert total_missing == 0, \
            f"Unknown activity should be skipped, got missing={total_missing}"

    def test_single_event_single_object(self, replay):
        """Single-event log with one object must not crash and
        must return exactly one per-object record."""
        log = OCELLog(events=[
            OCEvent("e1", "Place Order", 1.0, [("o99", "order")]),
        ])
        result = replay.run(log)
        assert len(result.per_object) == 1
        assert result.per_object[0].obj_id == "o99"

    def test_large_log_does_not_crash(self, replay):
        """OC-TBR must complete without error on a large synthetic log."""
        log = generate_synthetic_log(n_orders=200, deviation_rate=0.3, seed=0)
        result = replay.run(log)
        assert 0.0 <= result.fitness <= 1.0
