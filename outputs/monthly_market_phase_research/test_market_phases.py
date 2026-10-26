"""Observed-state, exact-window and prefix-causality checks for market phases."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location("market_phases_for_tests", Path(__file__).with_name("market_phases.py"))
phases = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(phases)


def panel(values):
    dates = pd.bdate_range("2017-01-02", periods=len(values))
    return {"adj": pd.DataFrame({"SPY": np.asarray(values, float), "IGNORED": 10.}, index=dates)}


def calculate(p, t=-1, symbols=("A", "B")):
    return phases.build_states(p, [(p["adj"].index[t], symbols)])


class TestMarketPhases(unittest.TestCase):
    def test_known_observed_peak_trough_and_repair(self):
        prices = np.r_[np.full(200, 70.), 100., np.linspace(99., 80., 30), np.linspace(81., 90., 21)]
        p = panel(prices)
        state, audit = calculate(p)
        expected = dict(depth=.1, peak_age=51., trough_depth=.2, recovered=.5,
                        trough_age=21., pressure=.5, repair=.5 * (1 - 21 / 126), underwater=51 / 126)
        for key, value in expected.items():
            self.assertAlmostEqual(state[key][0], value, places=13)
        self.assertTrue(state["valid"][0])
        self.assertEqual(audit["observations"][0]["peak_date"], str(p["adj"].index[200].date()))
        self.assertEqual(audit["observations"][0]["observed_trough_date"], str(p["adj"].index[230].date()))
        # The earlier all-window low70 is deliberately not the post-peak trough.
        self.assertAlmostEqual(state["trough_depth"][0], .2)

    def test_prefix_and_future_new_low_do_not_rewrite_observed_state(self):
        values = np.r_[np.linspace(100., 200., 250), np.linspace(190., 160., 50), np.linspace(170., 190., 40)]
        p = panel(values)
        expected, _ = calculate(p, 310)
        prefix, _ = calculate({"adj": p["adj"].iloc[:311].copy()})
        p["adj"].iloc[311:, 0] = .01
        changed, _ = calculate(p, 310)
        for key in list(phases.STATE_NAMES) + ["valid"]:
            np.testing.assert_array_equal(expected[key], prefix[key])
            np.testing.assert_array_equal(expected[key], changed[key])

    def test_price_scale_invariance(self):
        p = panel(np.r_[np.linspace(100., 200., 200), np.linspace(199., 130., 40), np.linspace(140., 170., 60)])
        expected, _ = calculate(p)
        p["adj"]["SPY"] *= 37.191
        changed, _ = calculate(p)
        for key in phases.STATE_NAMES:
            np.testing.assert_allclose(expected[key], changed[key], atol=1e-13)

    def test_last_equal_peak_and_last_equal_postpeak_low(self):
        prices = np.full(252, 90.)
        prices[[100, 200]] = 100.
        prices[[230, 240]] = 80.
        state, _ = calculate(panel(prices))
        self.assertEqual(state["peak_age"][0], 51.)
        self.assertEqual(state["trough_age"][0], 11.)
        self.assertAlmostEqual(state["recovered"][0], .5)

    def test_flat_and_current_new_peak_are_zero_states(self):
        for prices in (np.full(252, 260.37), np.linspace(100., 200., 252)):
            state, _ = calculate(panel(prices))
            self.assertTrue(state["valid"][0])
            for key in phases.STATE_NAMES:
                self.assertEqual(state[key][0], 0.)

    def test_exact252_window_includes_signal_and_oldest_price(self):
        prices = np.full(301, 100.)
        prices[47], prices[48] = 1000., 200.
        p = panel(prices)
        state, _ = calculate(p, 299)
        self.assertAlmostEqual(state["depth"][0], .5)
        self.assertEqual(state["peak_age"][0], 251.)
        self.assertEqual(state["underwater"][0], 1.)
        # The old peak exits at the next session; the finite-window state resets.
        next_state, _ = calculate(p, 300)
        self.assertEqual(next_state["depth"][0], 0.)
        self.assertEqual(next_state["peak_age"][0], 0.)
        p["adj"].iloc[299, 0] = 150.
        current_changed, _ = calculate(p, 299)
        self.assertAlmostEqual(current_changed["depth"][0], .25)

    def test_missing_nonpositive_and_nonfinite_window_prices_invalid(self):
        for bad in (np.nan, 0., -1., np.inf, -np.inf):
            for location in (0, 130, 251):
                with self.subTest(bad=bad, location=location):
                    p = panel(np.linspace(100., 200., 252))
                    p["adj"].iloc[location, 0] = bad
                    state, audit = calculate(p)
                    self.assertFalse(state["valid"][0])
                    self.assertEqual(audit["invalid_events"], 1)
                    for key in phases.STATE_NAMES:
                        self.assertTrue(np.isnan(state[key][0]))

    def test_history_boundary_outside_nan_and_candidate_independence(self):
        p = panel(np.linspace(100., 200., 260))
        before, _ = calculate(p, 250)
        at, _ = calculate(p, 251)
        self.assertFalse(before["valid"][0])
        self.assertTrue(at["valid"][0])
        expected, _ = calculate(p)
        p["adj"].iloc[:8, 0] = np.nan
        p["adj"]["IGNORED"] = np.nan
        changed, _ = calculate(p, symbols=("OTHER",))
        for key in list(phases.STATE_NAMES) + ["valid"]:
            np.testing.assert_array_equal(expected[key], changed[key])

    def test_no_mutation_and_empty_events(self):
        p = panel(np.linspace(100., 200., 300))
        saved = p["adj"].copy(deep=True)
        calculate(p)
        pd.testing.assert_frame_equal(p["adj"], saved)
        state, audit = phases.build_states(p, [])
        self.assertEqual(audit["events"], 0)
        for value in state.values():
            self.assertEqual(value.shape, (0,))

    def test_missing_spy_or_bad_calendar_fail_loudly(self):
        p = panel(np.linspace(100., 200., 300))
        with self.assertRaisesRegex(KeyError, "SPY adjusted"):
            phases.build_states({"adj": p["adj"].drop(columns="SPY")}, [])
        with self.assertRaisesRegex(ValueError, "exact panel session"):
            phases.build_states(p, [(pd.Timestamp("2017-01-01"), [])])
        with self.assertRaisesRegex(ValueError, "unique and increasing"):
            calculate({"adj": p["adj"].iloc[::-1]})


if __name__ == "__main__":
    unittest.main()
