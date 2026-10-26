import copy
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

_SPEC = importlib.util.spec_from_file_location("earnings_event_schema_under_test", Path(__file__).with_name("event_schema.py"))
model = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(model)


def record(at="2024-07-02T08:00:00-04:00", date=None, zone="America/New_York"):
    official = dict(actual_period_end="2024-04-28", verified=True,
                    source_url="https://example.com/issuer/earnings-release", publication_kind="release",
                    publication_timezone=zone)
    if at is not None:
        official["publication_at"] = at
    if date is not None:
        official["publication_date"] = date
    return dict(symbol="NVDA", asset_type="equity", vendor=dict(fiscalDateEnding="2024-04-30",
        reportedDate="2024-07-02", reportTime="pre-market"), official=official)


class NormalizeTests(unittest.TestCase):
    def test_vendor_period_difference_retained_not_event_rejected(self):
        result = model.normalize_event(record())
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["vendor_fiscal_date_ending"], "2024-04-30")
        self.assertEqual(result["official_actual_period_end"], "2024-04-28")
        self.assertTrue(result["fiscal_labels_differ"])
        self.assertEqual(result["publication_at"], "2024-07-02T12:00:00+00:00")
        self.assertEqual(result["known_by_at"], result["publication_at"])
        self.assertEqual(result["available_session"], "2024-07-02")

    def test_date_string_in_timestamp_never_becomes_midnight(self):
        result = model.normalize_event(record(at="2024-07-02"))
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["precision"], "date_only")
        self.assertIsNone(result["publication_at"])
        self.assertFalse(result["known_by_is_actual_publication"])
        self.assertEqual(result["date_only_upper_bound_at"], "2024-07-03T03:59:59.999999999+00:00")
        # July3 is an actual completed XNYS session despite its early close.
        self.assertEqual(result["known_by_at"], "2024-07-03T17:00:00+00:00")

    def test_date_only_weekend_holiday_and_foreign_timezone(self):
        cases = [("2024-07-03", "America/New_York", "2024-07-05T20:00:00+00:00"),
                 ("2024-07-06", "America/New_York", "2024-07-08T20:00:00+00:00"),
                 ("2024-07-03", "Asia/Taipei", "2024-07-05T20:00:00+00:00"),
                 ("2024-07-03", "Pacific/Auckland", "2024-07-05T20:00:00+00:00")]
        for date, zone, expected in cases:
            result = model.normalize_event(record(at=None, date=date, zone=zone))
            self.assertEqual(result["status"], "ready")
            self.assertEqual(result["known_by_at"], expected)
            self.assertGreater(pd.Timestamp(result["known_by_at"]), pd.Timestamp(result["date_only_upper_bound_at"]))
        taipei = model.normalize_event(record(at=None, date="2024-07-03", zone="Asia/Taipei"))
        self.assertEqual(taipei["date_only_upper_bound_at"], "2024-07-03T15:59:59.999999999+00:00")

    def test_exact_postclose_holiday_and_early_close_use_actual_calendar(self):
        for at, session in [("2024-07-02T17:00:00-04:00", "2024-07-03"),
                            ("2024-07-03T14:00:00-04:00", "2024-07-05"),
                            ("2024-07-04T09:00:00-04:00", "2024-07-05"),
                            ("2024-07-06T12:00:00-04:00", "2024-07-08")]:
            event = model.normalize_event(record(at=at))
            self.assertEqual(event["status"], "ready")
            self.assertEqual(event["available_session"], session)

    def test_dst_offsets_ambiguous_nonexistent_times(self):
        for at, utc in [("2024-03-08 08:00:00", "2024-03-08T13:00:00+00:00"),
                        ("2024-03-11 08:00:00", "2024-03-11T12:00:00+00:00")]:
            r = record(at=at);r["official"]["actual_period_end"] = "2024-01-31"
            e = model.normalize_event(r)
            self.assertEqual(e["status"], "ready")
            self.assertEqual(e["publication_at"], utc)
        for at in ["2024-11-03 01:30:00", "2024-03-10 02:30:00"]:
            r = record(at=at);r["official"]["actual_period_end"] = "2024-01-31"
            e = model.normalize_event(r)
            self.assertEqual(e["status"], "pending_review")
            self.assertIsNone(e["known_by_at"])
        e = model.normalize_event(record(at="2024-11-03T01:30:00-04:00"))
        self.assertEqual(e["publication_at"], "2024-11-03T05:30:00+00:00")
        r = record(at=None, date="2024-11-03")
        e = model.normalize_event(r)
        self.assertEqual(e["date_only_upper_bound_at"], "2024-11-04T04:59:59.999999999+00:00")
        self.assertEqual(e["known_by_at"], "2024-11-04T21:00:00+00:00")

    def test_unknown_zone_unverified_and_future_period_pending(self):
        rows = [record(at=None, date="2024-07-02", zone=None),
                record(at="2024-07-02 08:00:00", zone=None)]
        unverified = record();unverified["official"]["verified"] = False;rows.append(unverified)
        missing_period = record();missing_period["official"].pop("actual_period_end");rows.append(missing_period)
        future_period = record();future_period["official"]["actual_period_end"] = "2024-08-31";rows.append(future_period)
        for r in rows:
            e = model.normalize_event(r)
            self.assertEqual(e["status"], "pending_review")
            self.assertIsNone(e["known_by_at"])

    def test_vendor_hints_modified_and_call_timestamps_not_substitutes(self):
        r = record(at=None)
        r["official"].update(dateModified="2024-07-02T08:00:00-04:00",
                             conference_call_at="2024-07-02T08:00:00-04:00")
        e = model.normalize_event(r)
        self.assertEqual(e["status"], "pending_review")
        self.assertIsNone(e["publication_at"])
        for kind in ["conference_call", "dateModified", "sec_acceptance"]:
            r = record();r["official"]["publication_kind"] = kind
            self.assertEqual(model.normalize_event(r)["status"], "pending_review")

    def test_aware_foreign_timestamp_retains_source_date_without_zone_field(self):
        r = record(at="2024-07-03T01:30:00+08:00", date="2024-07-03", zone=None)
        e = model.normalize_event(r)
        self.assertEqual(e["status"], "ready")
        self.assertEqual(e["publication_date"], "2024-07-03")
        self.assertEqual(e["publication_at"], "2024-07-02T17:30:00+00:00")
        self.assertEqual(e["available_session"], "2024-07-02")

    def test_etf_not_applicable_and_equity_missing_are_distinct(self):
        etf = model.normalize_event(dict(symbol="QQQ", asset_type="etf"))
        missing = model.normalize_event(dict(symbol="AAPL", asset_type="equity"))
        self.assertEqual(etf["status"], "not_applicable")
        self.assertEqual(missing["status"], "pending_review")
        self.assertFalse(model.event_available(etf, "2024-07-05T21:00:00Z"))


class ReactionTests(unittest.TestCase):
    def setUp(self):
        self.event = model.normalize_event(record())
        self.bars = pd.DataFrame(dict(close=[100., 102., 105., 106.], volume=[5., 10., 20., 30.]),
                                 index=pd.to_datetime(["2024-07-02", "2024-07-03", "2024-07-05", "2024-07-08"]))
        self.benchmark = pd.DataFrame(dict(close=[200., 201., 203., 204.]), index=self.bars.index)

    def test_future_event_and_unclosed_window_do_not_read_future_bars(self):
        result = model.reaction_features(self.event, object(), "2024-07-02T11:59:59Z")
        self.assertEqual(result["status"], "not_yet_available")
        self.assertIsNone(result["post_known_close_to_close_return"])
        result = model.reaction_features(self.event, object(), "2024-07-03T16:59:59Z")
        self.assertEqual(result["status"], "waiting_for_closed_window")
        self.assertEqual(result["end_close_at"], "2024-07-03T17:00:00+00:00")
        self.assertIsNone(result["post_known_close_to_close_return"])

    def test_mature_early_close_market_adjustment_and_volume(self):
        r = model.reaction_features(self.event, self.bars, "2024-07-03T17:00:00Z", benchmark_bars=self.benchmark)
        self.assertEqual(r["status"], "ready")
        self.assertEqual(r["anchor_session"], "2024-07-02")
        self.assertEqual(r["end_session"], "2024-07-03")
        self.assertAlmostEqual(r["post_known_close_to_close_return"], .02)
        self.assertAlmostEqual(r["benchmark_return"], .005)
        self.assertAlmostEqual(r["market_adjusted_return"], .015)
        self.assertEqual(r["reaction_volume_sum"], 10.)
        self.assertLessEqual(pd.Timestamp(r["end_close_at"]), pd.Timestamp(r["signal_asof"]))
        self.assertGreaterEqual(pd.Timestamp(r["anchor_close_at"]), pd.Timestamp(r["known_by_at"]))

    def test_date_only_is_known_at_nextclose_and_reaction_waits_another_session(self):
        e = model.normalize_event(record(at=None, date="2024-07-02"))
        r = model.reaction_features(e, self.bars, "2024-07-03T17:00:00Z")
        self.assertEqual(r["status"], "waiting_for_closed_window")
        self.assertEqual(r["anchor_session"], "2024-07-03")
        self.assertEqual(r["end_session"], "2024-07-05")
        mature = model.reaction_features(e, self.bars, "2024-07-05T20:00:00Z")
        self.assertAlmostEqual(mature["post_known_close_to_close_return"], 105. / 102. - 1.)

    def test_missing_interior_price_or_volume_never_filled_with_zero(self):
        changed = self.bars.copy();changed.loc["2024-07-03", "close"] = np.nan
        r = model.reaction_features(self.event, changed, "2024-07-05T20:00:00Z", horizon=2)
        self.assertEqual(r["status"], "missing_price_data")
        self.assertIsNone(r["post_known_close_to_close_return"])
        changed = self.bars.copy();changed.loc["2024-07-03", "volume"] = np.nan
        r = model.reaction_features(self.event, changed, "2024-07-03T17:00:00Z")
        self.assertEqual(r["status"], "ready")
        self.assertIsNone(r["reaction_volume_sum"])
        self.assertIsNone(r["market_adjusted_return"])
        bad_benchmark = self.benchmark.copy();bad_benchmark.loc["2024-07-03", "close"] = np.nan
        r = model.reaction_features(self.event, self.bars, "2024-07-03T17:00:00Z", benchmark_bars=bad_benchmark)
        self.assertIsNone(r["benchmark_return"])
        self.assertIsNone(r["market_adjusted_return"])

    def test_future_bar_mutation_is_invariant_and_policy_mismatch_rejected(self):
        a = model.reaction_features(self.event, self.bars, "2024-07-03T17:00:00Z")
        changed = self.bars.copy();changed.loc[changed.index > "2024-07-03"] = 99999.
        b = model.reaction_features(self.event, changed, "2024-07-03T17:00:00Z")
        self.assertEqual(a, b)
        event = copy.deepcopy(self.event);event["policy_version"] = "unknown_policy"
        self.assertFalse(model.event_available(event, "2024-07-05T21:00:00Z"))
        self.assertEqual(model.reaction_features(event, self.bars, "2024-07-05T21:00:00Z")["status"], "pending_review")
        with self.assertRaises(ValueError): model.event_available(self.event, "2024-07-05 21:00:00")
        with self.assertRaises(ValueError): model.reaction_features(self.event, self.bars, "2024-07-05T21:00:00Z", horizon=0)


if __name__ == "__main__":
    unittest.main()
