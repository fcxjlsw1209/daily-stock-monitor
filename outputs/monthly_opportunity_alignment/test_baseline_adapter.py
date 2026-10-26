"""Original-score parity and historical eligibility, without any return search."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
P = Path(__file__).resolve().parent
O = P.parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


adapter = module("alignment_adapter_test", P / "baseline_adapter.py")
inputs = module("alignment_inputs_test", O / "monthly_stability_monitor/inputs.py")


def choose(row, good, symbols):
    indices = sorted(np.flatnonzero(good), key=lambda j: (-row[j], symbols[j]))[:3]
    return [symbols[j] for j in indices] if len(indices) >= 2 else []


class BaselineParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        meta = json.loads((O / "monthly_architecture_restart/META.json").read_text())
        cls.m, cls.t = meta["monthly"], meta["training"]
        cls.etfs = cls.m["etfs"]
        cls.panel, _ = inputs.load_panel(cls.etfs)
        cls.frames = adapter._STRATEGY.factors(cls.panel)
        cls.events = list(zip(cls.m["dates"], cls.m["symbols"]))
        cls.training = list(zip(cls.t["dates"], cls.t["symbols"]))
        with patch.object(adapter._STRATEGY, "factors", return_value=cls.frames) as call:
            cls.event_data = adapter.prepare(cls.panel, cls.events, cls.etfs)
            if call.call_count != 1:
                raise AssertionError("Factor calculation must occur once per prepare")
            cls.train_data = adapter.prepare(cls.panel, cls.training, cls.etfs)

    def test_exact122_scores_core_eligibility_and_holdings(self):
        reference = np.load(O / "monthly_trend_evidence_research/DATA.npz")
        for key, target in [("score", "base_original65"), ("core", "core_original65")]:
            np.testing.assert_allclose(self.event_data[key], reference[target], rtol=0., atol=1e-12)
        np.testing.assert_array_equal(self.event_data["cw"],
                                      np.broadcast_to(reference["cw_original65"], (122, 49)))
        np.testing.assert_array_equal(self.event_data["eligible"], reference["eligible"])
        orders = json.loads((O / "monthly_daily_episodes/original65_ORDERS.json").read_text())
        actual = [choose(s, g, ss) for s, g, ss in zip(self.event_data["score"],
                  self.event_data["eligible"], self.m["symbols"])]
        self.assertEqual(actual, [row["symbols"] for row in orders])

    def test_training180_uses_signal_eligibility_not_labels(self):
        self.assertEqual(len(self.training), 180)
        self.assertEqual(self.train_data["score"].shape, (180, 49))
        expected = np.stack([self.frames[1].loc[day, ss].to_numpy(bool)
                             for day, ss in self.training])
        np.testing.assert_array_equal(self.train_data["eligible"], expected)
        labels = np.load(O / "monthly_architecture_restart/DATA.npz")["train_good"].reshape(180, 49)
        self.assertFalse((labels & ~expected).any())
        # A label mask is deliberately not accepted by prepare. Future missing
        # labels therefore cannot suppress an otherwise eligible candidate.
        self.assertTrue(np.isfinite(self.train_data["score"]).all())

    def test_original_selection_on_temporary_training_membership(self):
        member = pd.DataFrame(False, index=pd.to_datetime(self.t["dates"]),
                              columns=self.panel["adj"].columns)
        for day, ss in self.training:
            member.loc[day, [s for s in ss if s not in self.etfs]] = True
        temporary = dict(self.panel, member=member)
        with patch.object(adapter._STRATEGY, "factors", return_value=self.frames):
            for i, (day, ss) in enumerate(self.training):
                decision = adapter._STRATEGY.selection(temporary, day, self.etfs)
                self.assertEqual(choose(self.train_data["score"][i], self.train_data["eligible"][i], ss),
                                 decision["symbols"])
                for s, score in decision["scores"].items():
                    self.assertAlmostEqual(self.train_data["score"][i, ss.index(s)], score, places=12)
        self.assertIsNot(self.panel["member"], member)

    def test_given_symbol_order_is_preserved(self):
        day, symbols = self.events[15]
        reverse = list(reversed(symbols))
        with patch.object(adapter._STRATEGY, "factors", return_value=self.frames):
            data = adapter.prepare(self.panel, [(day, reverse)], self.etfs)
        for key in data:
            np.testing.assert_allclose(data[key][0], self.event_data[key][15, ::-1], atol=1e-12, rtol=0)

    def test_input_frames_unchanged_and_no_global_factor_cache(self):
        before = self.panel["member"].copy()
        with patch.object(adapter._STRATEGY, "factors", return_value=self.frames) as call:
            adapter.prepare(self.panel, self.events[:1], self.etfs)
            adapter.prepare(self.panel, self.events[:1], self.etfs)
            self.assertEqual(call.call_count, 2)
        pd.testing.assert_frame_equal(self.panel["member"], before)

    def test_empty_and_invalid_candidates(self):
        data = adapter.prepare(self.panel, [], self.etfs)
        self.assertTrue(all(value.shape == (0, 49) for value in data.values()))
        day, ss = self.events[0]
        with self.assertRaises(ValueError):
            adapter.prepare(self.panel, [(day, ss[:-1])], self.etfs)
        with self.assertRaises(ValueError):
            adapter.prepare(self.panel, [(day, ss[:-1] + [ss[0]])], self.etfs)


if __name__ == "__main__":
    unittest.main()
