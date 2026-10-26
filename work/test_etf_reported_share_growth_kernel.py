"""Synthetic-only tests. No project source/data files are loaded."""
from dataclasses import replace
from datetime import date, timedelta
import copy
import math
import unittest

from etf_reported_share_growth_kernel import (
    LAG, WINDOW, ShareObservation, SplitAction, UnitAdmission,
    reported_share_growth63,
)


def fixture(query_index=100):
    # Explicit irregular supplied sessions; no inference from calendar-day gaps.
    sessions = []
    current = date(2040, 1, 2)
    weekdays_seen = 0
    while len(sessions) < 160:
        if current.weekday() < 5:
            weekdays_seen += 1
            if weekdays_seen % 17:
                sessions.append(current.isoformat())
        current += timedelta(days=1)
    rows = [ShareObservation(day, 1000. + i, day + "T21:00:00Z",
                             "synthetic-v1", "synthetic-fund")
            for i, day in enumerate(sessions)]
    admission = UnitAdmission(
        identity="synthetic-fund", unit_basis="native",
        start_session=sessions[0], end_session=sessions[-1],
        known_at=sessions[0] + "T00:00:00Z", evidence_id="synthetic-unit-proof",
        units_verified=True, action_ledger_verified=True,
        reporting_clock_verified=True,
    )
    query = sessions[query_index]
    return sessions, rows, query, query + "T21:00:00Z", admission


class ReportedShareGrowthTests(unittest.TestCase):
    def setUp(self):
        self.axis, self.rows, self.query, self.cutoff, self.admission = fixture()

    def run_feature(self, rows=None, actions=(), admission=None, axis=None,
                    query=None, cutoff=None):
        return reported_share_growth63(
            self.axis if axis is None else axis,
            self.rows if rows is None else rows,
            self.query if query is None else query,
            self.cutoff if cutoff is None else cutoff,
            admission=self.admission if admission is None else admission,
            actions=actions,
        )

    def assert_neutral(self, result, reason=None):
        self.assertFalse(result["available"])
        self.assertEqual(result["candidate"], 0.)
        self.assertEqual(result["control"], 0.)
        self.assertIsNone(result["raw_candidate"])
        self.assertIsNone(result["raw_control"])
        self.assertFalse(result["supplied_clock_checks_passed"])
        self.assertFalse(result["strict_PIT_certified"])
        if reason:
            self.assertEqual(result["reason"], reason)

    def action(self, index=60, factor=2., legal_index=None, **changes):
        a = SplitAction(
            "synthetic-split", self.axis[index if legal_index is None else legal_index],
            self.axis[index], factor, self.axis[index - 2] + "T21:00:00Z",
            "synthetic-action-proof", True,
        )
        return replace(a, **changes)

    def constant_rows(self, value=100.):
        return [replace(r, shares=value) for r in self.rows]

    def test_exact_63_changes_and_five_exchange_session_lag(self):
        r = self.run_feature()
        self.assertTrue(r["available"])
        self.assertEqual((WINDOW, LAG), (63, 5))
        self.assertEqual(r["source_start"], self.axis[32])
        self.assertEqual(r["source_end"], self.axis[95])
        self.assertEqual(r["observations_used"], 64)
        expected = 63. / 1032.
        self.assertAlmostEqual(r["candidate"], expected, places=14)
        self.assertAlmostEqual(r["control"], expected, places=14)

    def test_first_permitted_query_is_68_not_67(self):
        for index in (0, 63, 67):
            self.assert_neutral(self.run_feature(query=self.axis[index],
                cutoff=self.axis[index] + "T21:00:00Z"), "insufficient_history")
        r = self.run_feature(query=self.axis[68], cutoff=self.axis[68] + "T21:00:00Z")
        self.assertTrue(r["available"])
        self.assertEqual(r["source_start"], self.axis[0])

    def test_only_required_window_is_needed(self):
        self.assertEqual(self.run_feature(rows=self.rows[32:96]), self.run_feature())

    def test_recent_five_and_future_share_changes_do_not_enter(self):
        rows = [replace(r, shares=float("nan"), known_at=None) if i >= 96 else r
                for i, r in enumerate(self.rows)]
        self.assertEqual(self.run_feature(rows=rows), self.run_feature())

    def test_every_physical_query_prefix_is_exact(self):
        for index in range(68, len(self.axis)):
            query, cutoff = self.axis[index], self.axis[index] + "T21:00:00Z"
            full = self.run_feature(query=query, cutoff=cutoff)
            prefix = self.run_feature(axis=self.axis[:index + 1],
                rows=self.rows[:index + 1], query=query, cutoff=cutoff)
            self.assertEqual(full, prefix)

    def test_constant_and_true_zero_are_available(self):
        r = self.run_feature(rows=self.constant_rows())
        self.assertTrue(r["available"])
        self.assertEqual((r["candidate"], r["control"]), (0., 0.))
        self.assertIsNotNone(r["raw_candidate"])

    def test_common_positive_scale_invariance(self):
        base = self.run_feature()
        for scale in (0.125, 8., 1000000.):
            r = self.run_feature(rows=[replace(x, shares=x.shares * scale) for x in self.rows])
            self.assertAlmostEqual(r["candidate"], base["candidate"], places=13)
            self.assertAlmostEqual(r["control"], base["control"], places=13)

    def test_forward_split_only_is_neutral(self):
        rows = [replace(r, shares=100. if i < 60 else 200.) for i, r in enumerate(self.rows)]
        r = self.run_feature(rows=rows, actions=[self.action()])
        self.assertTrue(r["available"])
        self.assertEqual((r["candidate"], r["control"]), (0., 0.))
        self.assertEqual(r["applied_split_ids"], ["synthetic-split"])

    def test_reverse_split_only_is_neutral(self):
        rows = [replace(r, shares=100. if i < 60 else 20.) for i, r in enumerate(self.rows)]
        r = self.run_feature(rows=rows, actions=[self.action(factor=.2)])
        self.assertTrue(r["available"])
        self.assertEqual((r["candidate"], r["control"]), (0., 0.))

    def test_multiple_splits_on_distinct_sessions(self):
        rows = [replace(r, shares=100. if i < 50 else (200. if i < 80 else 600.))
                for i, r in enumerate(self.rows)]
        actions = [self.action(50, 2., event_id="a"), self.action(80, 3., event_id="b")]
        r = self.run_feature(rows=rows, actions=actions)
        self.assertTrue(r["available"])
        self.assertEqual((r["candidate"], r["control"]), (0., 0.))

    def test_split_with_net_creation_has_expected_quantity_growth(self):
        rows = [replace(r, shares=100. if i < 60 else 220.) for i, r in enumerate(self.rows)]
        r = self.run_feature(rows=rows, actions=[self.action()])
        self.assertAlmostEqual(r["candidate"], .1)
        self.assertAlmostEqual(r["control"], .1)

    def test_pre_window_action_not_applied_again(self):
        r = self.run_feature(rows=self.constant_rows(200), actions=[self.action(32)])
        self.assertTrue(r["available"])
        self.assertEqual((r["candidate"], r["control"]), (0., 0.))
        self.assertEqual(r["applied_split_ids"], [])

    def test_action_on_last_source_session_applies(self):
        rows = [replace(r, shares=100. if i < 95 else 200.) for i, r in enumerate(self.rows)]
        r = self.run_feature(rows=rows, actions=[self.action(95)])
        self.assertTrue(r["available"])
        self.assertEqual((r["candidate"], r["control"]), (0., 0.))

    def test_verified_reported_unit_day_is_not_guessed_from_ex_date(self):
        rows = [replace(r, shares=100. if i < 60 else 200.) for i, r in enumerate(self.rows)]
        r = self.run_feature(rows=rows, actions=[self.action(60, legal_index=59)])
        self.assertTrue(r["available"])
        self.assertEqual((r["candidate"], r["control"]), (0., 0.))

    def test_reported_unit_before_legal_date_is_not_accepted(self):
        r = self.run_feature(actions=[self.action(60, legal_index=61)])
        self.assert_neutral(r, "source_unit_precedes_legal_effective")

    def test_common_unit_data_is_not_double_split_adjusted(self):
        r = self.run_feature(rows=self.constant_rows(200), actions=[self.action()],
            admission=replace(self.admission, unit_basis="common"))
        self.assertTrue(r["available"])
        self.assertEqual((r["candidate"], r["control"]), (0., 0.))
        self.assertEqual(r["applied_split_ids"], [])

    def test_announced_future_split_does_not_apply_early(self):
        future = self.action(120, known_at=self.axis[90] + "T21:00:00Z")
        self.assertEqual(self.run_feature(actions=[future]), self.run_feature())

    def test_future_announced_action_unknown_mapping_preserves_prefix(self):
        future = self.action(120, source_unit_session=None,
            known_at=self.axis[90] + "T21:00:00Z", verified=False)
        full = self.run_feature(actions=[future])
        prefix = self.run_feature(axis=self.axis[:101], rows=self.rows[:101],
                                  actions=[future])
        self.assertEqual(full, self.run_feature())
        self.assertEqual(prefix, full)
        # Even if legally effective inside the excluded recent-five interval,
        # it must not be pulled into an earlier reported-share source window.
        recent = self.action(98, source_unit_session="unknown", verified=False)
        self.assertEqual(self.run_feature(actions=[recent]), self.run_feature())

    def test_effective_action_unknown_reporting_mapping_is_neutral(self):
        for mapping in (None, "unknown"):
            action = self.action(60, source_unit_session=mapping)
            self.assert_neutral(self.run_feature(actions=[action]),
                                "action_source_unit_session_unknown")

    def test_effective_action_unknown_at_query_is_unavailable(self):
        late = self.action(60, known_at=self.axis[101] + "T21:00:00Z")
        self.assert_neutral(self.run_feature(actions=[late]), "action_not_known")
        self.assert_neutral(self.run_feature(actions=[self.action(60, known_at=None)]),
                            "action_clock_missing")

    def test_no_automatic_lag_substitution_for_publication_clock(self):
        for known_at, reason in [(None, "observation_clock_missing"),
                                (self.axis[101] + "T21:00:00Z", "observation_not_known"),
                                (self.axis[40] + "T21:00:00", "observation_clock_invalid")]:
            rows = list(self.rows)
            rows[40] = replace(rows[40], known_at=known_at)
            self.assert_neutral(self.run_feature(rows=rows), reason)

    def test_exact_publication_cutoff_is_allowed(self):
        rows = list(self.rows)
        rows[40] = replace(rows[40], known_at=self.cutoff)
        self.assertTrue(self.run_feature(rows=rows)["available"])
        rows[40] = replace(rows[40], known_at=self.query + "T21:00:00.000001Z")
        self.assert_neutral(self.run_feature(rows=rows), "observation_not_known")

    def test_unknown_or_unverified_units_neutralize_both_arms(self):
        for changes in [dict(unit_basis="unknown"), dict(units_verified=False),
                        dict(action_ledger_verified=False), dict(reporting_clock_verified=False)]:
            self.assert_neutral(self.run_feature(admission=replace(self.admission, **changes)))

    def test_missing_admission_and_late_or_short_unit_evidence(self):
        self.assert_neutral(reported_share_growth63(self.axis, self.rows, self.query,
            self.cutoff, admission=None), "unit_admission_missing")
        for changes in [dict(known_at=None),
                        dict(known_at=self.axis[101] + "T21:00:00Z"),
                        dict(end_session=self.axis[94]), dict(start_session=self.axis[33]),
                        dict(evidence_id="")]:
            self.assert_neutral(self.run_feature(admission=replace(self.admission, **changes)))

    def test_missing_interior_not_filled_even_with_endpoints(self):
        self.assert_neutral(self.run_feature(rows=self.rows[:50] + self.rows[51:]),
                            "missing_observation")

    def test_bad_numeric_values_neutral(self):
        for value in (None, True, 0., -1., float("nan"), float("inf"), "--"):
            rows = list(self.rows); rows[50] = replace(rows[50], shares=value)
            self.assert_neutral(self.run_feature(rows=rows), "invalid_share_balance")

    def test_duplicates_and_identity_or_version_mismatch_neutral(self):
        self.assert_neutral(self.run_feature(rows=self.rows + [self.rows[50]]),
                            "duplicate_observation")
        for changes, reason in [(dict(identity="another-fund"), "observation_identity_mismatch"),
                                (dict(version_id=""), "observation_version_missing")]:
            rows = list(self.rows); rows[50] = replace(rows[50], **changes)
            self.assert_neutral(self.run_feature(rows=rows), reason)

    def test_unverified_duplicate_and_invalid_action_neutral(self):
        for actions in [[self.action(verified=False)], [self.action(factor=0.)],
                        [self.action(), self.action()], [self.action(evidence_id="")]]:
            self.assert_neutral(self.run_feature(actions=actions))

    def test_absolute_daily_net_control_is_not_absolute_total_net(self):
        rows = self.constant_rows()
        rows[50] = replace(rows[50], shares=150.)
        r = self.run_feature(rows=rows)
        self.assertEqual(r["candidate"], 0.)
        self.assertEqual(r["control"], 1.)

    def test_overflow_and_underflow_neutralize_both(self):
        self.assert_neutral(self.run_feature(rows=self.constant_rows(1e308),
            actions=[self.action(factor=1e308)]), "nonfinite_or_underflow_common_unit")
        self.assert_neutral(self.run_feature(rows=self.constant_rows(1e-308),
            actions=[self.action(factor=1e-308)]), "nonfinite_or_underflow_common_unit")
        rows = self.constant_rows(1e308)
        rows[32] = replace(rows[32], shares=1e-308)
        self.assert_neutral(self.run_feature(rows=rows), "nonfinite_feature_arithmetic")

    def test_bad_protocol_dates_and_cutoff_fail(self):
        for axis in [self.axis + [self.axis[-1]], self.axis[::-1]]:
            with self.assertRaises(ValueError):
                self.run_feature(axis=axis)
        for cutoff in [self.query + "T21:00:00", self.axis[99] + "T21:00:00Z"]:
            with self.assertRaises(ValueError):
                self.run_feature(cutoff=cutoff)

    def test_inputs_unchanged_no_source_certification_claimed(self):
        before = copy.deepcopy((self.axis, self.rows, self.admission))
        result = self.run_feature()
        self.assertEqual(before, (self.axis, self.rows, self.admission))
        self.assertTrue(result["supplied_clock_checks_passed"])
        self.assertFalse(result["strict_PIT_certified"])


if __name__ == "__main__":
    unittest.main()
