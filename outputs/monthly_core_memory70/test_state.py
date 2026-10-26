"""Synthetic fixed-boundary, missing-data and strict-prefix state tests."""
from pathlib import Path
import importlib.util
import json
import unittest

import numpy as np

sp = importlib.util.spec_from_file_location('core_memory70_state_test', Path(__file__).with_name('state.py'))
s = importlib.util.module_from_spec(sp)
sp.loader.exec_module(s)


class StateTests(unittest.TestCase):
    def test_initialization_strict_point_ten(self):
        for value, expected in ((0., True), (.09, True), (.099999, True),
                                (.10, False), (.100001, False), (.11, False)):
            got, audit = s.hysteresis([value])
            np.testing.assert_array_equal(got, [expected])
            self.assertEqual((audit['observed'], audit['missing'], audit['transitions'],
                              audit['band_holds'], audit['missing_holds']), (1, 0, 0, 0, 0))
            self.assertEqual(audit['initial_state'], expected)
            self.assertEqual(audit['final_state'], expected)

    def test_exact_band_edges_and_memory(self):
        vol = [.10, .095, .09, .10, .109999, .11, .10, .090001, .09, .11]
        got, audit = s.hysteresis(vol)
        np.testing.assert_array_equal(got, [False, False, True, True, True, False, False, False, True, False])
        self.assertEqual(audit['transitions'], 4)
        self.assertEqual(audit['band_holds'], 5)
        self.assertEqual(audit['enter_threshold_observations'], 2)
        self.assertEqual(audit['exit_threshold_observations'], 2)

    def test_initial_and_subsequent_nan_hold_separately(self):
        values = [np.nan, np.nan, .095, .09, np.nan, .10, .11, np.nan]
        got, audit = s.hysteresis(values)
        np.testing.assert_array_equal(got, [False, False, False, True, True, True, False, False])
        self.assertEqual((audit['observed'], audit['missing']), (4, 4))
        self.assertTrue(audit['initial_missing'])
        self.assertFalse(audit['initial_state'])
        self.assertEqual(audit['missing_holds'], 3)
        self.assertEqual(audit['band_holds'], 2)
        self.assertEqual(audit['transitions'], 2)

    def test_all_missing_and_empty(self):
        got, audit = s.hysteresis([np.nan]*5)
        np.testing.assert_array_equal(got, [False]*5)
        self.assertEqual((audit['observed'], audit['missing'], audit['missing_holds']), (0, 5, 4))
        empty, info = s.hysteresis([])
        self.assertEqual(empty.dtype, np.dtype(bool))
        self.assertEqual(empty.shape, (0,))
        self.assertEqual(info['count'], 0)
        self.assertEqual(info['observed'], 0)
        self.assertEqual(info['missing'], 0)
        self.assertIsNone(info['initial_state'])
        self.assertIsNone(info['final_state'])
        self.assertFalse(info['initial_missing'])

    def test_all_prefixes_and_future_poison_do_not_reach_suffix(self):
        values = np.array([np.nan, .095, .09, .105, .11, np.nan, .10, .08, .12])
        full, _ = s.hysteresis(values)
        for n in range(len(values)+1):
            for container in (list(values), tuple(values), values.copy()):
                got, info = s.hysteresis(container, np.int64(n))
                np.testing.assert_array_equal(got, full[:n])
                expected, expected_info = s.hysteresis(values[:n])
                np.testing.assert_array_equal(got, expected)
                self.assertEqual(info, expected_info)
            for poison in (np.inf, -.1, 'not numeric', 1+2j, {'not': 'a scalar'}):
                future = list(values[:n])+[poison]*3
                got, info = s.hysteresis(future, n)
                np.testing.assert_array_equal(got, full[:n])
                self.assertEqual(info, s.hysteresis(values[:n])[1])

    def test_never_converts_or_indexes_original_sequence_beyond_prefix(self):
        class Poison:
            def __float__(self):raise AssertionError('future object converted')
            def __array__(self, *args, **kw):raise AssertionError('future object inspected')
        class PrefixOnly:
            def __init__(self, n):self.n=n
            def __len__(self):return 8
            def __getitem__(self, key):
                if not isinstance(key, slice) or key.start is not None or key.stop != self.n or key.step is not None:
                    raise AssertionError('not a single exact prefix read')
                return [.08, .10, .11, .09, .10][:self.n]
            def __array__(self, *args, **kw):raise AssertionError('whole input converted')
        for n in range(6):
            values = [.08, .10, .11, .09, .10]
            got, info = s.hysteresis(PrefixOnly(n), n)
            expected, expected_info = s.hysteresis(values[:n])
            np.testing.assert_array_equal(got, expected)
            self.assertEqual(info, expected_info)
            got, _ = s.hysteresis(values[:n]+[Poison()], n)
            np.testing.assert_array_equal(got, expected)

    def test_invalid_prefix_values_and_containers_fail(self):
        for bad in (np.inf, -np.inf, -.001, '0.10', None, True, np.bool_(False), 1+0j, object()):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):s.hysteresis([.10, bad])
                got, _ = s.hysteresis([.10, bad], 1)
                np.testing.assert_array_equal(got, [False])
        for bad in (None, .10, '0.10', {'a': .10}, {.10}, iter([.10]), np.zeros((2, 2)), [[.10], [.11]]):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):s.hysteresis(bad)

    def test_invalid_n_rejected_and_zero_does_not_validate_values(self):
        for n in (True, False, np.bool_(False), -1, 3, 1., '1', np.nan):
            with self.subTest(n=n):
                with self.assertRaises(ValueError):s.hysteresis([.1, .2], n)
        got, info = s.hysteresis([object(), 'bad', -1., np.inf], 0)
        self.assertEqual(len(got), 0)
        self.assertEqual(info['count'], 0)

    def test_no_input_mutation_and_json_audit_fixed_thresholds(self):
        values = np.array([.08, np.nan, .10, .12, .10, .08])
        saved = values.copy(); values.flags.writeable = False
        first, audit = s.hysteresis(values)
        second, again = s.hysteresis(values)
        np.testing.assert_array_equal(values, saved)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(audit, again)
        json.dumps(audit, allow_nan=False)
        self.assertTrue(audit['thresholds_fixed'])
        self.assertEqual((audit['initial_threshold'], audit['enter_calm_threshold'], audit['exit_calm_threshold']), (.10, .09, .11))
        with self.assertRaises(TypeError):s.hysteresis(values, enter_threshold=.095)
        self.assertEqual(audit['observed']+audit['missing'], audit['count'])
        self.assertEqual(audit['band_holds']+audit['missing_holds']+audit['enter_threshold_observations']+
                         audit['exit_threshold_observations'], audit['count']-1)


if __name__ == '__main__':
    unittest.main()
