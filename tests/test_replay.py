"""
tests/test_replay.py — Unit tests for OC-TBR.

Tests are organised to mirror the paper's claims:
  1. Soundness proof (Section IV, Proposition 1)
  2. Perfect conformance → f = 1.0
  3. Known violations → correct f_o values
  4. Per-type fitness formula
  5. Global counters formula
  6. Empty log edge case
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src import OCTokenReplay, OCELLog
from experiments.nets import build_order_fulfilment_net
from experiments.logs import (build_log_A, build_log_B, build_log_C,
                                build_counterexample_log)


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
# 1. Soundness: Proposition 1 from Section IV
# ─────────────────────────────────────────────────────────────

class TestSoundness:

    def test_per_object_misses_binding_violation(self, net):
        """
        Proposition 1 (Section IV): per-object projection is unsound.

        In the counterexample, items i1 and i2 each see the trace
        Pack -> Ship in their projection and report zero missing tokens,
        even though they were packed under o1 and shipped under o2.

        This demonstrates that per-object replay cannot detect
        cross-object binding violations.
        """
        log = build_counterexample_log()
        for obj_id in ["i1", "i2"]:   # items are the ones misled
            proj_events = [e for e in log.events
                           if any(oid == obj_id for oid, _ in e.objects)]
            proj_log    = OCELLog(events=proj_events)
            result      = OCTokenReplay(net).run(proj_log)
            obj_stats   = next((s for s in result.per_object
                                if s.obj_id == obj_id), None)
            missing = obj_stats.missing if obj_stats else 0
            assert missing == 0, \
                (f"Per-object projection for item {obj_id} sees Pack->Ship "
                 f"and should report missing=0 (unsoundness), got {missing}")

    def test_global_detects_violation(self, replay):
        """
        Global OC-TBR detects the cross-object binding violation:
        M_g > 0 while item per-object projections show M = 0.
        """
        log    = build_counterexample_log()
        result = replay.run(log)
        assert result.M_g > 0, \
            f"Global replay should detect missing tokens, got M_g={result.M_g}"

    def test_o2_is_violator(self, replay):
        """
        In the counterexample, o2 shipped items it never packed.
        Global replay should flag o2 (or the bound items) as violators.
        """
        log    = build_counterexample_log()
        result = replay.run(log)
        o2 = next(s for s in result.per_object if s.obj_id == "o2")
        assert o2.missing > 0, \
            f"o2 should be detected as a violator, got missing={o2.missing}"


# ─────────────────────────────────────────────────────────────
# 2. Perfect conformance
# ─────────────────────────────────────────────────────────────

class TestPerfectConformance:

    def test_log_A_no_missing_tokens(self, replay):
        """
        Log A (perfect) should produce M_g = 0.
        Remaining > 0 is expected: classical token replay counts tokens
        in sink places as 'remaining' (van der Aalst 2016, p.179).
        A conformant trace still ends with a token in the sink place.
        """
        result = replay.run(build_log_A(num_orders=10))
        assert result.M_g == 0, \
            f"Expected no missing tokens for Log A, got M_g={result.M_g}"

    def test_log_A_per_object_no_missing(self, replay):
        """No object in Log A should have any missing tokens."""
        result = replay.run(build_log_A(num_orders=10))
        for s in result.per_object:
            assert s.missing == 0, \
                f"Object {s.obj_id} has unexpected missing={s.missing}"

    def test_log_A_better_than_log_B(self, replay):
        """Log A (perfect) must have strictly higher fitness than Log B."""
        r_a = replay.run(build_log_A(num_orders=20))
        r_b = replay.run(build_log_B(num_orders=20, violation_rate=0.5))
        assert r_a.fitness > r_b.fitness, \
            f"Log A fitness ({r_a.fitness:.4f}) should exceed Log B ({r_b.fitness:.4f})"


# ─────────────────────────────────────────────────────────────
# 3. Known violations and per-object scores
# ─────────────────────────────────────────────────────────────

class TestViolations:

    def test_log_B_global_below_one(self, replay):
        """Log B (20% skip Pack) should yield f < 1.0."""
        result = replay.run(build_log_B(num_orders=20, violation_rate=0.5))
        assert result.fitness < 1.0

    def test_log_B_orders_worse_than_log_A_orders(self, replay):
        """In Log B, orders have more missing tokens than in Log A."""
        r_a = replay.run(build_log_A(num_orders=20))
        r_b = replay.run(build_log_B(num_orders=20, violation_rate=0.5))
        ft_a = r_a.fitness_by_type()
        ft_b = r_b.fitness_by_type()
        assert ft_b["order"] < ft_a["order"], \
            "Orders in Log B should have lower fitness than in Log A"

    def test_log_C_detects_binding_violations(self, replay):
        """
        In Log C, cross-object binding violations must be detected.
        Items involved in a swap should have f_o < 1.0.
        """
        log    = build_log_C(num_orders=20, violation_rate=1.0, seed=1)
        result = replay.run(log)
        assert result.fitness < 1.0, "Log C should not report perfect fitness"

    def test_log_C_violating_objects_flagged(self, replay):
        """At least some objects should be flagged as violators."""
        result = replay.run(build_log_C(num_orders=20, violation_rate=0.5))
        violators = result.objects_below_threshold(0.999)
        assert len(violators) > 0, "No violators detected in Log C"


# ─────────────────────────────────────────────────────────────
# 4. Fitness formula correctness
# ─────────────────────────────────────────────────────────────

class TestFormulas:

    def test_global_formula_manual(self, replay):
        """
        Verify Eq. (1): f = 1 - 0.5*(M_g/C_g + R_g/P_g)
        by computing manually from counters.
        """
        result = replay.run(build_log_B(num_orders=10, violation_rate=0.3))
        expected = (1 - 0.5 * (result.M_g / result.C_g
                                + result.R_g / result.P_g))
        assert result.fitness == pytest.approx(expected, abs=1e-9)

    def test_per_object_formula_manual(self, replay):
        """Verify Eq. (2): f_o = 1 - 0.5*(m_o/c_o + r_o/p_o)."""
        result = replay.run(build_log_B(num_orders=10, violation_rate=0.5))
        for s in result.per_object:
            if s.consumed == 0 and s.produced == 0:
                continue
            expected = 1 - 0.5 * (
                (s.missing   / s.consumed  if s.consumed  > 0 else 0) +
                (s.remaining / s.produced  if s.produced  > 0 else 0)
            )
            assert s.fitness == pytest.approx(expected, abs=1e-9), \
                f"f_o mismatch for {s.obj_id}"

    def test_per_type_aggregates_correctly(self, replay):
        """Verify Eq. (3): f_tau aggregates over all objects of same type."""
        result = replay.run(build_log_B(num_orders=10, violation_rate=0.3))
        ft = result.fitness_by_type()

        for tau in ["order", "item"]:
            group = [s for s in result.per_object if s.obj_type == tau]
            p_sum = sum(s.produced  for s in group)
            c_sum = sum(s.consumed  for s in group)
            m_sum = sum(s.missing   for s in group)
            r_sum = sum(s.remaining for s in group)
            expected = 1 - 0.5 * (
                (m_sum / c_sum if c_sum > 0 else 0) +
                (r_sum / p_sum if p_sum > 0 else 0)
            )
            assert ft[tau] == pytest.approx(expected, abs=1e-9), \
                f"f_tau mismatch for type={tau}"

    def test_fitness_in_unit_interval(self, replay):
        """Fitness values must always be in [0, 1]."""
        for log in [build_log_A(5), build_log_B(5), build_log_C(5)]:
            result = replay.run(log)
            assert 0.0 <= result.fitness <= 1.0
            for s in result.per_object:
                assert 0.0 <= s.fitness <= 1.0, \
                    f"f_o out of range for {s.obj_id}: {s.fitness}"


# ─────────────────────────────────────────────────────────────
# 5. Edge cases
# ─────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_empty_log(self, replay):
        """Empty log should not crash and return f=1.0."""
        result = replay.run(OCELLog(events=[]))
        assert result.fitness == 1.0

    def test_unknown_activity_skipped(self, replay):
        """Events with activity not in model are skipped silently."""
        from src import make_synthetic_log
        log = make_synthetic_log([
            {"id": "e1", "activity": "UNKNOWN_ACTIVITY",
             "timestamp": 1, "objects": [["o1", "order"]]},
        ])
        result = replay.run(log)
        # No crash, and no missing tokens (event was skipped)
        assert result.M_g == 0

    def test_global_counters_sum_per_object(self, replay):
        """Global counters must equal the sum of per-object counters."""
        result = replay.run(build_log_B(num_orders=15))
        assert result.P_g == sum(s.produced  for s in result.per_object)
        assert result.C_g == sum(s.consumed  for s in result.per_object)
        assert result.M_g == sum(s.missing   for s in result.per_object)
        assert result.R_g == sum(s.remaining for s in result.per_object)
