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
# Helpers: build small logs directly (no external function needed)
# ─────────────────────────────────────────────────────────────

def make_conformant_log():
    """
    Minimal conformant log: o1 follows Place->Pack->Pay->Ship correctly.
    i1, i2 are bound to o1 at Pack and Ship.
    """
    return OCELLog(events=[
        OCEvent("e1", "Place Order", 1.0, [("o1", "order")]),
        OCEvent("e2", "Pack Items", 2.0, [("o1", "order"), ("i1", "item"), ("i2", "item")]),
        OCEvent("e3", "Pay", 3.0, [("o1", "order")]),
        OCEvent("e4", "Ship Order", 4.0, [("o1", "order"), ("i1", "item"), ("i2", "item")]),
    ])


def make_deviant_log():
    """
    Deviant log: o2 skips Pack Items and goes straight to Pay.
    i3 is bound to o2 at Ship but was never packed.
    This is the running example from Section III of the paper.
    """
    return OCELLog(events=[
        OCEvent("e1", "Place Order", 1.0, [("o1", "order")]),
        OCEvent("e2", "Place Order", 2.0, [("o2", "order")]),
        OCEvent("e3", "Pack Items", 3.0, [("o1", "order"), ("i1", "item"), ("i2", "item")]),
        OCEvent("e4", "Pay", 4.0, [("o1", "order")]),
        OCEvent("e5", "Pay", 5.0, [("o2", "order")]),  # o2 skips Pack
        OCEvent("e6", "Ship Order", 6.0, [("o1", "order"), ("i1", "item"), ("i2", "item")]),
        OCEvent("e7", "Ship Order", 7.0, [("o2", "order"), ("i3", "item")]),
    ])


def make_counterexample_log():
    """
    Counterexample for Proposition 1 (Section IV):
    o1 packs i1, i2 — but they get shipped under o2.
    Per-object projections of i1, i2 see Pack->Ship and report f=1.
    Global replay detects the binding violation.
    """
    return OCELLog(events=[
        OCEvent("e1", "Place Order", 1.0, [("o1", "order")]),
        OCEvent("e2", "Place Order", 2.0, [("o2", "order")]),
        OCEvent("e3", "Pack Items", 3.0, [("o1", "order"), ("i1", "item"), ("i2", "item")]),
        OCEvent("e4", "Ship Order", 4.0, [("o2", "order"), ("i1", "item"), ("i2", "item")]),
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

    def test_o2_fitness_value(self, replay):
        """
        o2 has 1 missing token out of 3 consumed, 0 remaining.
        f_o2 = 1 - 0.5*(1/3 + 0/3) = 0.8333...
        """
        result = replay.run(make_deviant_log())
        o2 = next(s for s in result.per_object if s.obj_id == "o2")
        assert o2.fitness == pytest.approx(0.7083, abs=0.001)

    def test_more_deviations_lower_fitness(self, replay):
        """Log C (40% deviation) must have strictly lower fitness than Log B (20%)."""
        log_b = generate_synthetic_log(n_orders=30, deviation_rate=0.2, seed=42)
        log_c = generate_synthetic_log(n_orders=30, deviation_rate=0.4, seed=42)
        r_b = replay.run(log_b)
        r_c = replay.run(log_c)
        assert r_c.fitness < r_b.fitness, \
            f"Log C fitness ({r_c.fitness:.4f}) should be below Log B ({r_b.fitness:.4f})"


# ─────────────────────────────────────────────────────────────
# 3. Proposition 1 — unsoundness of per-object projection
# ─────────────────────────────────────────────────────────────

class TestSoundness:

    def test_per_object_misses_binding_violation(self, net):
        """
        Proposition 1 (Section IV): per-object projection is unsound.
        i1 and i2 each see Pack->Ship in their projection and report
        missing=0, even though they were packed under o1 and shipped under o2.
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
                (f"Per-object projection for {obj_id} sees Pack->Ship and "
                 f"should report missing=0 (unsoundness), got {missing}")

    def test_global_detects_binding_violation(self, replay):
        """
        Global OC-TBR must detect the binding violation that
        per-object projection misses.
        """
        log = make_counterexample_log()
        result = replay.run(log)
        total_missing = sum(s.missing for s in result.per_object)
        assert total_missing > 0, \
            "Global replay should detect missing tokens in the counterexample"

    def test_global_fitness_below_one_in_counterexample(self, replay):
        """Global fitness must be < 1.0 while per-object projections see 1.0."""
        log = make_counterexample_log()
        result = replay.run(log)
        assert result.fitness < 1.0


# ─────────────────────────────────────────────────────────────
# 4. Fitness formula correctness
# ─────────────────────────────────────────────────────────────

class TestFormulas:

    def test_per_object_formula(self, replay):
        """
        Verify Eq. (2): f_o = 1 - 0.5*(m_o/c_o + r_o/p_o)
        for every object in the deviant log.
        """
        result = replay.run(make_deviant_log())
        for s in result.per_object:
            if s.consumed == 0 and s.produced == 0:
                continue
            expected = 1 - 0.5 * (
                    (s.missing / s.consumed if s.consumed > 0 else 0) +
                    (s.remaining / s.produced if s.produced > 0 else 0)
            )
            assert s.fitness == pytest.approx(expected, abs=1e-9), \
                f"f_o formula mismatch for {s.obj_id}: " \
                f"got {s.fitness:.6f}, expected {expected:.6f}"

    def test_per_type_formula(self, replay):
        """
        Verify Eq. (3): f_tau aggregates m, c, r, p across all objects
        of the same type.
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
            expected = 1 - 0.5 * (
                    (m_sum / c_sum if c_sum > 0 else 0) +
                    (r_sum / p_sum if p_sum > 0 else 0)
            )
            assert ft[tau] == pytest.approx(expected, abs=1e-9), \
                f"f_tau formula mismatch for type={tau}"

    def test_fitness_in_unit_interval(self, replay):
        """All fitness values must be in [0.0, 1.0]."""
        logs = [
            make_conformant_log(),
            make_deviant_log(),
            make_counterexample_log(),
            generate_synthetic_log(10, 0.0),
            generate_synthetic_log(10, 0.4),
        ]
        for log in logs:
            result = replay.run(log)
            assert 0.0 <= result.fitness <= 1.0, \
                f"Global fitness out of range: {result.fitness}"
            for s in result.per_object:
                assert 0.0 <= s.fitness <= 1.0, \
                    f"f_o out of range for {s.obj_id}: {s.fitness}"

    def test_conformant_better_than_deviant(self, replay):
        """Conformant log must always score higher than deviant log."""
        r_good = replay.run(make_conformant_log())
        r_bad = replay.run(make_deviant_log())
        assert r_good.fitness > r_bad.fitness


# ─────────────────────────────────────────────────────────────
# 5. Edge cases
# ─────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_empty_log(self, replay):
        """Empty log must not crash and must return fitness = 1.0."""
        result = replay.run(OCELLog(events=[]))
        assert result.fitness == pytest.approx(1.0, abs=1e-9)

    def test_unknown_activity_skipped(self, replay):
        """
        An event whose activity is not in the model must be
        skipped silently without raising an exception.
        """
        log = OCELLog(events=[
            OCEvent("e1", "ACTIVITY_NOT_IN_MODEL", 1.0, [("o1", "order")]),
        ])
        result = replay.run(log)
        # no crash — that is the main assertion
        # unknown event is skipped so no tokens are consumed or produced
        total_missing = sum(s.missing for s in result.per_object)
        assert total_missing == 0, \
            f"Unknown activity should be skipped, got missing={total_missing}"

    def test_single_event_single_object(self, replay):
        """Single-event log with one object must not crash."""
        log = OCELLog(events=[
            OCEvent("e1", "Place Order", 1.0, [("o99", "order")]),
        ])
        result = replay.run(log)
        assert len(result.per_object) == 1
        assert result.per_object[0].obj_id == "o99"

    def test_global_counters_equal_sum_of_per_object(self, replay):
        """
        Global counters (P_g, C_g, M_g, R_g) must equal
        the sum of the corresponding per-object counters.
        """
        result = replay.run(make_deviant_log())
        assert result.P_g == sum(s.produced for s in result.per_object), "P_g mismatch"
        assert result.C_g == sum(s.consumed for s in result.per_object), "C_g mismatch"
        assert result.M_g == sum(s.missing for s in result.per_object), "M_g mismatch"
        assert result.R_g == sum(s.remaining for s in result.per_object), "R_g mismatch"
