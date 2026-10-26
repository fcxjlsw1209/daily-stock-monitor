"""Offline regression checks for temporal safety, exact mapping, and masks."""
import copy
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

_SPEC = importlib.util.spec_from_file_location("earnings_factor_features_under_test",
                                              Path(__file__).with_name("earnings_features.py"))
features = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(features)


class FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calendar = features._calendar()
        sessions = cls.calendar.sessions_in_range("2023-10-02", "2024-06-28")
        steps = np.arange(len(sessions), dtype=float)
        cls.full_px = pd.DataFrame({"AAPL": 100.0 + steps + steps ** 2 / 100,
                                   "research_equity_FB": 200.0 + steps / 3,
                                   "SPY": 200.0 + steps / 2}, index=sessions)

    def setUp(self):
        self.px = self.full_px.copy()
        self.meta = dict(dates=["2024-03-28", "2024-04-30"],
                         symbols=[["AAPL", "research_equity_FB", "SPY"] for _ in range(2)])
        self.document = dict(rows=[])
        for day in self.meta["dates"]:
            for symbol in self.meta["symbols"][0]:
                status = ("not_applicable_etf" if symbol == "SPY" else
                          "missing_event_history" if symbol == "research_equity_FB" else
                          "verified_event_available")
                row = dict(signal_date=day, signal_asof=self.calendar.session_close(day).isoformat(),
                           symbol=symbol, status=status, reactions={})
                if status == "verified_event_available":
                    row["event_id"] = "AAPL_2024-02-01"
                    row["reactions"] = {str(h): self.reaction(day, h) for h in (10, 21)}
                self.document["rows"].append(row)

    def reaction(self, day, horizon):
        anchor_i = self.calendar.sessions.get_loc(pd.Timestamp("2024-02-01"))
        anchor = self.calendar.sessions[anchor_i]
        end = self.calendar.sessions[anchor_i + horizon]
        stock = self.px.loc[end, "AAPL"] / self.px.loc[anchor, "AAPL"] - 1
        benchmark = self.px.loc[end, "SPY"] / self.px.loc[anchor, "SPY"] - 1
        return dict(status="ready", horizon=horizon, policy_version=features.POLICY_VERSION,
                    signal_asof=self.calendar.session_close(day).isoformat(),
                    known_by_at=self.calendar.session_close(anchor).isoformat(),
                    anchor_session=str(anchor.date()), end_session=str(end.date()),
                    anchor_close_at=self.calendar.session_close(anchor).isoformat(),
                    end_close_at=self.calendar.session_close(end).isoformat(),
                    post_known_close_to_close_return=float(stock), benchmark_return=float(benchmark),
                    market_adjusted_return=float(stock - benchmark), benchmark_status="ready")

    def run_features(self, document=None, meta=None, px=None):
        return features.build_features(document if document is not None else self.document,
                                       meta if meta is not None else self.meta,
                                       px if px is not None else self.px, {"SPY"})

    def test_exact_values_and_ml_features(self):
        arrays, audit = self.run_features()
        self.assertEqual(arrays["ml_features"].shape, (2, 3, 3))
        for i, day in enumerate(self.meta["dates"]):
            for h in (10, 21):
                original = self.document["rows"][i * 3]["reactions"][str(h)]["market_adjusted_return"]
                self.assertEqual(arrays["event" + str(h)][i, 0], original)
                j = self.px.index.get_loc(pd.Timestamp(day))
                expected = (self.px.iloc[j]["AAPL"] / self.px.iloc[j-h]["AAPL"] - 1
                            - (self.px.iloc[j]["SPY"] / self.px.iloc[j-h]["SPY"] - 1))
                self.assertAlmostEqual(arrays["recent" + str(h)][i, 0], expected, places=14)
                self.assertIs(arrays["recent" + str(h)], arrays["matched_recent" + str(h)])
            np.testing.assert_array_equal(arrays["ml_features"][i, 0],
                [arrays["event10"][i, 0], arrays["event21"][i, 0],
                 arrays["event21"][i, 0] - arrays["event10"][i, 0]])
        self.assertEqual(audit["candidate_rows"], 6)
        self.assertEqual(audit["valid10"], 2)
        self.assertEqual(audit["event_price_crosscheck"]["checked"], 4)
        self.assertEqual(audit["event_price_crosscheck"]["max_abs_diff"], 0)

    def test_missing_and_etf_candidates_remain_nan_and_keep_mapping(self):
        arrays, audit = self.run_features()
        for key in ("event10", "event21", "recent10", "recent21", "ml_features"):
            self.assertTrue(np.isnan(arrays[key][:, 1:]).all())
        for key in ("valid10", "valid21", "valid_ml"):
            self.assertFalse(arrays[key][:, 1:].any())
        self.assertEqual(audit["row_status_counts"]["not_applicable_etf"], 2)
        self.assertEqual(audit["row_status_counts"]["missing_event_history"], 2)

    def test_immature_latest_horizon_stays_missing_and_ml_requires_both(self):
        reaction = self.document["rows"][0]["reactions"]["21"]
        reaction.update(status="waiting_for_closed_window", end_close_at="2024-04-01T20:00:00+00:00")
        for key in features.RETURN_KEYS:
            reaction[key] = None
        arrays, _ = self.run_features()
        self.assertTrue(arrays["valid10"][0, 0])
        self.assertFalse(arrays["valid21"][0, 0])
        self.assertTrue(np.isnan(arrays["event21"][0, 0]))
        self.assertTrue(np.isnan(arrays["recent21"][0, 0]))
        self.assertFalse(arrays["valid_ml"][0, 0])
        self.assertTrue(np.isnan(arrays["ml_features"][0, 0]).all())

    def test_future_ready_window_rejected(self):
        reaction = self.document["rows"][0]["reactions"]["10"]
        reaction["end_close_at"] = "2024-04-01T20:00:00+00:00"
        with self.assertRaisesRegex(ValueError, "future or invalid"):
            self.run_features()

    def test_known_after_anchor_rejected(self):
        self.document["rows"][0]["reactions"]["10"]["known_by_at"] = "2024-02-02T21:00:00+00:00"
        with self.assertRaisesRegex(ValueError, "future or invalid"):
            self.run_features()

    def test_nonready_values_rejected_even_for_unused_horizon(self):
        for horizon in ("3", "10", "21"):
            doc = copy.deepcopy(self.document)
            doc["rows"][0]["reactions"][horizon] = dict(status="waiting_for_closed_window", market_adjusted_return=0.0)
            with self.subTest(horizon=horizon), self.assertRaisesRegex(ValueError, "non-ready reaction"):
                self.run_features(document=doc)

    def test_unverified_and_etf_reactions_rejected(self):
        for row_index in (1, 2):
            doc = copy.deepcopy(self.document)
            doc["rows"][row_index]["reactions"] = copy.deepcopy(doc["rows"][0]["reactions"])
            with self.subTest(row_index=row_index), self.assertRaisesRegex(ValueError, "ETF/unverified"):
                self.run_features(document=doc)

    def test_ready_without_market_adjustment_keeps_horizon_nan(self):
        self.document["rows"][0]["reactions"]["10"].update(market_adjusted_return=None,
                benchmark_return=None, benchmark_status="unavailable")
        arrays, _ = self.run_features()
        self.assertFalse(arrays["valid10"][0, 0])
        self.assertTrue(arrays["valid21"][0, 0])
        self.assertTrue(np.isnan(arrays["ml_features"][0, 0]).all())

    def test_duplicate_feature_candidate_rejected(self):
        self.document["rows"].append(copy.deepcopy(self.document["rows"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate feature"):
            self.run_features()

    def test_missing_feature_candidate_rejected(self):
        self.document["rows"].pop()
        with self.assertRaisesRegex(ValueError, "missing feature"):
            self.run_features()

    def test_extra_date_and_symbol_alias_never_silently_mapped(self):
        for field, value in (("signal_date", "2024-03-27"), ("symbol", "FB"), ("symbol", "aapl")):
            doc = copy.deepcopy(self.document)
            doc["rows"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, "unmatched feature"):
                self.run_features(document=doc)

    def test_metadata_duplicates_and_wrong_shape_rejected(self):
        cases = []
        meta = copy.deepcopy(self.meta); meta["symbols"][0][1] = "AAPL"; cases.append(meta)
        meta = copy.deepcopy(self.meta); meta["dates"][1] = meta["dates"][0]; cases.append(meta)
        meta = copy.deepcopy(self.meta); meta["symbols"][1].pop(); cases.append(meta)
        for meta in cases:
            with self.subTest(meta=meta), self.assertRaises(ValueError):
                self.run_features(meta=meta)

    def test_row_order_does_not_change_metadata_order(self):
        original, _ = self.run_features()
        self.document["rows"].reverse()
        reordered, _ = self.run_features()
        for key in original:
            np.testing.assert_array_equal(reordered[key], original[key])

    def test_interior_recent_price_and_missing_whole_session_rejected(self):
        day = self.calendar.sessions[self.calendar.sessions.get_loc(pd.Timestamp("2024-03-28")) - 3]
        for symbol in ("AAPL", "SPY"):
            changed = self.px.copy(); changed.loc[day, symbol] = np.nan
            with self.subTest(symbol=symbol), self.assertRaisesRegex(ValueError, "incomplete recent"):
                self.run_features(px=changed)
        with self.assertRaisesRegex(ValueError, "incomplete recent"):
            self.run_features(px=self.px.drop(index=day))

    def test_nonpositive_recent_prices_rejected(self):
        for value in (0, -1, np.inf):
            changed = self.px.copy(); changed.loc["2024-03-28", "AAPL"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "incomplete recent"):
                self.run_features(px=changed)

    def test_no_price_requirement_for_missing_event_and_etf(self):
        arrays, _ = self.run_features(px=self.px.drop(columns="research_equity_FB"))
        self.assertFalse(arrays["valid_ml"][:, 1:].any())

    def test_future_price_mutations_and_truncation_preserve_prefix(self):
        original, _ = self.run_features()
        meta = {"dates": self.meta["dates"][:1], "symbols": self.meta["symbols"][:1]}
        doc = {"rows": self.document["rows"][:3]}
        truncated, _ = self.run_features(document=doc, meta=meta, px=self.px.loc[:"2024-03-28"])
        changed = self.px.copy(); changed.loc[changed.index > "2024-03-28"] = 987654321
        mutated, _ = self.run_features(document=doc, meta=meta, px=changed)
        for key in original:
            np.testing.assert_array_equal(original[key][:1], truncated[key])
            np.testing.assert_array_equal(original[key][:1], mutated[key])

    def test_price_crosscheck_records_mismatch_without_replacing_source(self):
        original = self.document["rows"][0]["reactions"]["10"]["market_adjusted_return"]
        changed = self.px.copy(); changed.loc["2024-02-01", "AAPL"] *= 1.02
        arrays, audit = self.run_features(px=changed)
        self.assertEqual(arrays["event10"][0, 0], original)
        crosscheck = audit["event_price_crosscheck"]
        self.assertEqual(crosscheck["differences_over_5bp"], 4)
        self.assertGreater(crosscheck["max_abs_diff"], .0005)
        self.assertEqual(crosscheck["examples"][0]["symbol"], "AAPL")

    def test_incomplete_event_crosscheck_does_not_change_event_or_control_mask(self):
        changed = self.px.copy(); changed.loc["2024-02-01", "AAPL"] = np.nan
        arrays, audit = self.run_features(px=changed)
        self.assertTrue(arrays["valid_ml"][:, 0].all())
        self.assertEqual(audit["event_price_crosscheck"]["incomplete_paths"], 4)
        self.assertEqual(audit["event_price_crosscheck"]["checked"], 0)

    def test_signal_clock_and_horizon_identity_rejected(self):
        doc = copy.deepcopy(self.document); doc["rows"][0]["signal_asof"] = "2024-03-28T21:00:00+00:00"
        with self.assertRaisesRegex(ValueError, "canonical XNYS close"):
            self.run_features(document=doc)
        doc = copy.deepcopy(self.document); doc["rows"][0]["reactions"]["10"]["horizon"] = 21
        with self.assertRaisesRegex(ValueError, "horizon mismatch"):
            self.run_features(document=doc)

    def test_actual_zero_is_preserved_but_missing_is_not_zero(self):
        r = self.document["rows"][0]["reactions"]["10"]
        r["post_known_close_to_close_return"] = r["benchmark_return"]
        r["market_adjusted_return"] = 0.0
        arrays, _ = self.run_features()
        self.assertEqual(arrays["event10"][0, 0], 0.0)
        self.assertTrue(arrays["valid10"][0, 0])
        self.assertTrue(np.isnan(arrays["event10"][0, 1]))

    def test_nonfinite_source_return_is_rejected(self):
        self.document["rows"][0]["reactions"]["10"]["market_adjusted_return"] = np.nan
        with self.assertRaisesRegex(ValueError, "finite numeric"):
            self.run_features()

    def test_build_does_not_modify_inputs(self):
        before = copy.deepcopy(self.document)
        meta = copy.deepcopy(self.meta)
        px = self.px.copy()
        self.run_features()
        self.assertEqual(self.document, before)
        self.assertEqual(self.meta, meta)
        pd.testing.assert_frame_equal(px, self.px)


if __name__ == "__main__":
    unittest.main()
