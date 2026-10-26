"""Return-basis, missing-observation and causal-window checks."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "monthly_session_features_for_tests", Path(__file__).with_name("session_features.py"))
session = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(session)


def panel(rows=420):
    rng = np.random.default_rng(622)
    dates = pd.bdate_range("2018-01-01", periods=rows)
    r = rng.normal(.001, .012, size=(rows, 2))
    c = pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=dates, columns=["A", "B"])
    opening = c.shift(1) * np.exp(rng.normal(.0002, .004, size=c.shape))
    opening.iloc[0] = c.iloc[0]
    return {"adj": c.copy(), "close": c.copy(), "open": opening}


class TestSessionDecomposition(unittest.TestCase):
    def test_daily_and_rolling_identity(self):
        p = panel()
        daily, audit = session.decompose_sessions(p)
        expected = np.log(p["adj"]) - np.log(p["adj"].shift(1))
        np.testing.assert_allclose(daily["overnight_log"] + daily["intraday_log"], expected,
                                   equal_nan=True, atol=1e-12)
        self.assertLess(audit["decomposition_max_abs_error"], 1e-12)
        frames, audit = session.build_session_features(p)
        expected = np.log(p["adj"].shift(42) / p["adj"].shift(252))
        np.testing.assert_allclose(frames["overnight_log_sum"] + frames["intraday_log_sum"],
                                   expected, equal_nan=True, atol=1e-12)
        self.assertEqual(audit["window"], 210)
        self.assertFalse(audit["trading_eligibility_modified"])

    def test_split_only_day_has_no_artificial_overnight_loss(self):
        dates = pd.bdate_range("2020-01-01", periods=4)
        p = {"close": pd.DataFrame({"A": [100., 100., 50., 50.]}, index=dates),
             "open": pd.DataFrame({"A": [100., 100., 50., 50.]}, index=dates),
             "adj": pd.DataFrame({"A": [50., 50., 50., 50.]}, index=dates)}
        daily, audit = session.decompose_sessions(p)
        self.assertAlmostEqual(daily["overnight_log"].iloc[2, 0], 0.)
        self.assertAlmostEqual(daily["intraday_log"].iloc[2, 0], 0.)
        self.assertEqual(audit["adjustment_factor_changes"], 1)
        np.testing.assert_allclose(daily["adjusted_open"], p["adj"])

    def test_dividend_adjustment_uses_same_basis_on_both_legs(self):
        dates = pd.bdate_range("2020-01-01", periods=3)
        # A one-dollar distribution accompanies a 100-to-99 ex-date opening.
        p = {"close": pd.DataFrame({"A": [100., 100., 99.]}, index=dates),
             "open": pd.DataFrame({"A": [100., 100., 99.]}, index=dates),
             "adj": pd.DataFrame({"A": [99., 99., 99.]}, index=dates)}
        daily, _ = session.decompose_sessions(p)
        self.assertAlmostEqual(daily["overnight_log"].iloc[2, 0], 0.)
        self.assertAlmostEqual(daily["intraday_log"].iloc[2, 0], 0.)
        p["close"].iloc[2, 0] = 99.5
        p["adj"].iloc[2, 0] = 99.5
        daily, _ = session.decompose_sessions(p)
        self.assertAlmostEqual(daily["overnight_log"].iloc[2, 0], 0.)
        self.assertAlmostEqual(daily["intraday_log"].iloc[2, 0], np.log(99.5 / 99.))

    def test_missing_open_stays_missing_not_zero_and_inputs_unchanged(self):
        p = panel()
        day = p["adj"].index[-60]
        p["open"].loc[day, "A"] = np.nan
        saved = {k: v.copy(deep=True) for k, v in p.items()}
        daily, audit = session.decompose_sessions(p)
        self.assertTrue(np.isnan(daily["overnight_log"].at[day, "A"]))
        self.assertTrue(np.isnan(daily["intraday_log"].at[day, "A"]))
        self.assertTrue(np.isfinite(daily["total_log"].at[day, "A"]))
        frames, _ = session.build_session_features(p)
        for v in frames.values():
            self.assertTrue(np.isnan(v.iloc[-1, 0]))
            self.assertTrue(np.isfinite(v.iloc[-1, 1]))
        self.assertEqual(audit["invalid_inputs"]["open"]["missing"], 1)
        for key in p:
            pd.testing.assert_frame_equal(p[key], saved[key])

    def test_nonpositive_and_infinite_inputs_never_become_flat_observations(self):
        for key in ("adj", "open", "close"):
            for value in (0., -1., np.inf, -np.inf):
                with self.subTest(key=key, value=value):
                    p = panel()
                    p[key].iloc[-60, 0] = value
                    daily, _ = session.decompose_sessions(p)
                    self.assertTrue(np.isnan(daily["overnight_log"].iloc[-60, 0]))
                    self.assertTrue(np.isnan(daily["intraday_log"].iloc[-60, 0]))
                    frames, _ = session.build_session_features(p)
                    self.assertTrue(np.isnan(frames["session_log_spread"].iloc[-1, 0]))

    def test_persistence_is_signed_and_valid_flat_day_is_zero(self):
        dates = pd.bdate_range("2020-01-01", periods=4)
        close = pd.DataFrame({"A": [100., 100., 100., 100.]}, index=dates)
        opening = pd.DataFrame({"A": [100., 105., 100., 95.]}, index=dates)
        frames, _ = session.build_session_features({"adj": close, "close": close, "open": opening},
                                                   window=3, skip=0)
        self.assertAlmostEqual(frames["overnight_sign_mean"].iloc[-1, 0], 0.)
        self.assertAlmostEqual(frames["intraday_sign_mean"].iloc[-1, 0], 0.)
        self.assertTrue(frames["overnight_sign_mean"].iloc[:3].isna().all().all())

    def test_awkward_decimal_flat_prices_do_not_create_false_direction(self):
        dates = pd.bdate_range("2020-01-01", periods=300)
        close = pd.DataFrame(260.37, index=dates, columns=["A"])
        adjusted = pd.DataFrame(133.3594403782062, index=dates, columns=["A"])
        p = {"adj": adjusted, "open": close.copy(), "close": close}
        daily, _ = session.decompose_sessions(p)
        np.testing.assert_array_equal(daily["intraday_log"], np.zeros((300, 1)))
        np.testing.assert_array_equal(daily["overnight_log"].iloc[1:], np.zeros((299, 1)))
        frames, audit = session.build_session_features(p)
        for key in session.FEATURE_NAMES:
            self.assertEqual(frames[key].iloc[-1, 0], 0.)
        self.assertEqual(audit["sign_zero_tolerance"], 1e-12)

    def test_skip42_excludes_recent_returns_and_prefix_is_invariant(self):
        p = panel()
        original, _ = session.build_session_features(p)
        changed = {k: v.copy() for k, v in p.items()}
        for v in changed.values():
            v.iloc[-42:] *= 2.0
        other, _ = session.build_session_features(changed)
        for key in session.FEATURE_NAMES:
            np.testing.assert_allclose(original[key].iloc[-1], other[key].iloc[-1], atol=0.)
        date = p["adj"].index[350]
        truncated, _ = session.build_session_features({k: v.loc[:date] for k, v in p.items()})
        for key in session.FEATURE_NAMES:
            pd.testing.assert_frame_equal(original[key].loc[:date], truncated[key])
        daily, _ = session.decompose_sessions(p)
        t = len(p["adj"]) - 1
        expected = daily["overnight_log"].iloc[t - 251:t - 41].sum()
        np.testing.assert_allclose(original["overnight_log_sum"].iloc[-1], expected, atol=1e-12)

    def test_misaligned_or_duplicate_sessions_rejected(self):
        p = panel()
        p["open"] = p["open"].iloc[::-1]
        with self.assertRaises(ValueError):
            session.build_session_features(p)
        p = panel()
        for key in p:
            p[key].index = pd.DatetimeIndex([p[key].index[0]] * len(p[key]))
        with self.assertRaises(ValueError):
            session.build_session_features(p)

    def test_bad_window_parameters_rejected(self):
        for window, skip in [(0, 42), (True, 42), (210.5, 42), (210, -1), (210, True)]:
            with self.subTest(window=window, skip=skip), self.assertRaises(ValueError):
                session.build_session_features(panel(), window=window, skip=skip)


if __name__ == "__main__":
    unittest.main()
