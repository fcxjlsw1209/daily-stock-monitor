"""Only synthetic data: formation consistency, exact slot routing and accounts."""
import ast
import copy
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import monthly_trend_gate182_independent as audit


def price_path(returns):
    return np.exp(np.r_[0., np.cumsum(returns)])


def fixture(width=2, count=3):
    dates = pd.bdate_range('2014-01-02', periods=400)
    calendar = dates.strftime('%Y-%m-%d').tolist()
    names = ['S' + str(i) for i in range(width - 1)] + ['QQQ']
    time = np.arange(399)
    frame = pd.DataFrame({name: price_path(.001 + .002 * np.sin(time * (.07 + j * .01)))
                          for j, name in enumerate(names)}, index=dates)
    events = [calendar[i] for i in (252, 273, 294)[:count]]
    return {'adj': frame}, dict(dates=events, symbols=[names[:] for _ in events], etfs=['QQQ']), calendar


def routing_fixture(rows=7):
    base = np.tile([.20, .19, .18, np.nan, .17], (rows, 1))
    d = np.tile([-.01, .007, .003, 0., 0.], (rows, 1))
    pre = base.copy(); nonzero = d != 0.; pre[nonzero] += d[nonzero]
    available = np.ones(base.shape, bool); consistent = np.zeros(base.shape, bool)
    consistent[:, 0] = True; consistent[:, 2] = True
    if rows > 2: consistent[2] = False
    if rows > 3: consistent[3, :3] = True
    if rows > 4: available[4] = False; consistent[4] = False
    if rows > 5: available[5, 0] = False; consistent[5, 0] = False
    if rows > 6: consistent[6] = False; consistent[6, 4] = True
    return pre, base, d, consistent, available


class SourceTests(unittest.TestCase):
    def test_three_disjoint70_blocks_strict_positive(self):
        for block_returns, consistent in (([.001, .002, .003], True), ([.001, 0., .003], False),
                                           ([.003, -.001, .004], False), ([-.001] * 3, False)):
            prices = price_path(np.repeat(block_returns, 70))
            result = audit.measure_oracle(prices)
            self.assertTrue(result['available']); self.assertEqual(result['consistent'], consistent)
            np.testing.assert_allclose(result['block_log_returns'], np.array(block_returns) * 70, atol=1e-14)
            self.assertEqual(result['block_return_estimates'], 3)

    def test_interior_path_does_not_change_endpoint_gate_but_must_be_valid(self):
        p = price_path(np.repeat(.001, 210)); expected = audit.measure_oracle(p)
        changed = p.copy(); changed[1:70] *= 20.; changed[71:140] /= 3.
        self.assertEqual(audit.measure_oracle(changed), expected)
        for i in (0, 1, 69, 70, 71, 139, 140, 209, 210):
            for bad in (0., -1., np.nan, np.inf):
                q = p.copy(); q[i] = bad
                result = audit.measure_oracle(q)
                self.assertFalse(result['available']); self.assertFalse(result['consistent'])
                self.assertEqual(result['block_return_estimates'], 0)

    def test_constant_nonnumeric_shapes_and_log_overflow(self):
        result = audit.measure_oracle(np.ones(211))
        self.assertTrue(result['available']); self.assertFalse(result['consistent'])
        p = np.ones(211, object); p[10] = 'bad'
        self.assertEqual(audit.measure_oracle(p)['reason'], 'nonnumeric_adjusted_price')
        for p in (np.ones(210), np.ones(212), np.ones((211, 1))):
            with self.assertRaises(ValueError): audit.measure_oracle(p)
        p = np.ones(211); p[[0, 70, 140, 210]] = [5e-324, 1e308, 5e-324, 1e308]
        self.assertTrue(audit.measure_oracle(p)['available'])
        with patch.object(audit.math, 'log', return_value=math.inf), self.assertRaises(FloatingPointError):
            audit.measure_oracle(np.ones(211))

    def test_scale_and_input_immutability(self):
        p = price_path(np.repeat([.001, -.001, .003], 70)); before = p.copy()
        expected = audit.measure_oracle(p)
        for scale in (1e-150, 1e150, 100., .01): audit.record_check(audit.measure_oracle(p * scale), expected)
        np.testing.assert_array_equal(p, before)

    def test_official_windows_counts_and_full49(self):
        panel, meta, cal = fixture(width=49)
        before, original_meta = panel['adj'].copy(deep=True), copy.deepcopy(meta)
        values, record = audit.independent_features(panel, meta, cal)
        receipt = audit.verify_features(panel, meta, cal, values, record)
        self.assertEqual(receipt['block_return_estimates'], 3 * 3 * 49)
        self.assertEqual(values['available'].shape, (3, 49))
        for t, event in zip((252, 273, 294), record['event_history']):
            self.assertEqual((event['price_start'], event['price_end']), (cal[t - 252], cal[t - 42]))
            self.assertEqual(event['windows']['blocks'], [dict(start_index=k, end_index=k+70,
                             start_date=cal[t-252+k], end_date=cal[t-252+k+70]) for k in (0, 70, 140)])
        pd.testing.assert_frame_equal(panel['adj'], before); self.assertEqual(meta, original_meta)
        json.dumps(record, allow_nan=False); json.dumps(receipt, allow_nan=False)

    def test_prefixes_recent42_and_future_not_converted(self):
        panel, meta, cal = fixture(); full, record = audit.independent_features(panel, meta, cal)
        for n in range(4):
            small, rec = audit.independent_features(panel, meta, cal, limit=n)
            for key in full: np.testing.assert_array_equal(small[key], full[key][:n])
            self.assertEqual(rec['event_history'], record['event_history'][:n])
        class Poison:
            def __float__(self): raise AssertionError('Future read')
        panel['adj'] = panel['adj'].astype(object)
        panel['adj'].loc[panel['adj'].index > pd.Timestamp(cal[210]), :] = Poison()
        small, _ = audit.independent_features(panel, meta, cal, limit=1)
        for key in full: np.testing.assert_array_equal(small[key], full[key][:1])
        panel['adj'] = panel['adj'].loc[:cal[210]]
        small, _ = audit.independent_features(panel, meta, cal, limit=1)
        for key in full: np.testing.assert_array_equal(small[key], full[key][:1])

    def test_missing_precedence_and_holiday_session_numbering(self):
        panel, meta, cal = fixture(); panel['adj'] = panel['adj'].drop(index=cal[10], columns='S0')
        _, audit_record = audit.independent_features(panel, meta, cal)
        self.assertEqual([c['reason'] for c in audit_record['event_history'][0]['cells']],
                         ['missing_symbol', 'missing_official_session'])
        meta['dates'][0] = cal[251]
        _, early = audit.independent_features(panel, meta, cal)
        self.assertIsNone(early['event_history'][0]['windows'])
        self.assertTrue(all(c['reason'] == 'insufficient_history' for c in early['event_history'][0]['cells']))
        panel, meta, cal = fixture(); holiday = cal.pop(100); panel['adj'] = panel['adj'].drop(index=holiday)
        meta['dates'][0] = cal[252]
        _, rec = audit.independent_features(panel, meta, cal)
        self.assertEqual(rec['event_history'][0]['price_end'], cal[210])
        self.assertEqual(rec['event_history'][0]['missing_official_sessions'], [])

    def test_audit_tampering_boolean_and_schema_failures(self):
        panel, meta, cal = fixture(); values, rec = audit.independent_features(panel, meta, cal)
        bad = copy.deepcopy(rec); bad['event_history'][0]['cells'][0]['block_log_returns'][0] += .1
        with self.assertRaises(ValueError): audit.verify_features(panel, meta, cal, values, bad)
        for field in ('block_return_estimates', 'block_sessions', 'consistent_cells'):
            bad = copy.deepcopy(rec); bad[field] += 1
            with self.assertRaises(ValueError): audit.verify_features(panel, meta, cal, values, bad)
        bad = copy.deepcopy(values); bad['consistent'][0, 0] = not bad['consistent'][0, 0]
        with self.assertRaises(ValueError): audit.verify_features(panel, meta, cal, bad, rec)
        bad = copy.deepcopy(values); bad['available'][0, 0] = False; bad['consistent'][0, 0] = True
        with self.assertRaises(ValueError): audit.arrays(bad)
        bad = copy.deepcopy(values); bad['consistent'] = bad['consistent'].astype(int)
        with self.assertRaises(ValueError): audit.arrays(bad)
        for n in (-1, 4, True, 1.1):
            with self.assertRaises(ValueError): audit.independent_features(panel, meta, cal, limit=n)
        del meta['etfs']
        with self.assertRaises(ValueError): audit.independent_features(panel, meta, cal)


class RoutingTests(unittest.TestCase):
    def test_gate_and_l1_weighted_control_exact_manual(self):
        args = routing_fixture(); scores, record = audit.independent_routing(*args)
        pre, base, d, consistent, available = args
        b = .013 / .020
        self.assertAlmostEqual(record['uniform_removed_fraction'][1], b)
        np.testing.assert_array_equal(scores[audit.ARMS[1]][1, [0, 2]], base[1, [0, 2]])
        self.assertEqual(scores[audit.ARMS[1]][1, 1], pre[1, 1])
        np.testing.assert_allclose(scores[audit.ARMS[2]][1, :3], base[1, :3] + (1-b)*d[1, :3], atol=1e-16)
        self.assertAlmostEqual(record['candidate_removed_abs_increment'][1], .013)
        self.assertAlmostEqual(record['control_removed_abs_increment'][1], .013)
        receipt = audit.verify_routing(*args, scores, record)
        self.assertTrue(receipt['matched_absolute_removal_checked'])
        self.assertEqual(receipt['episode_applications'], 0)
        json.dumps(record, allow_nan=False); json.dumps(receipt, allow_nan=False)

    def test_zero_full_initial_invalid_and_zero_increment_exact_copy(self):
        args = routing_fixture(); scores, rec = audit.independent_routing(*args)
        for arm in scores:
            for row in (0, 2, 4, 6): np.testing.assert_array_equal(scores[arm][row], args[0][row])
            np.testing.assert_array_equal(scores[arm][3], args[1][3])
            np.testing.assert_array_equal(scores[arm][:, 3:], args[0][:, 3:])
            self.assertEqual(scores[arm][5, 0], args[0][5, 0])
        self.assertEqual(rec['uniform_removed_fraction'][0], 0.)
        self.assertGreater(rec['event_history'][0]['consistent_abs_increment'], 0.)
        self.assertEqual(rec['candidate_removed_abs_increment'][0], 0.)
        self.assertEqual(rec['uniform_removed_fraction'][3], 1.)

    def test_no_increment_all_states_neutral_and_no_averaging_by_member_count(self):
        args = list(routing_fixture()); args[2][:] = 0.; args[0] = args[1].copy()
        scores, rec = audit.independent_routing(*args)
        self.assertTrue(all(b == 0. for b in rec['uniform_removed_fraction']))
        for arm in scores: np.testing.assert_array_equal(scores[arm], args[0])
        args = routing_fixture(); scores, rec = audit.independent_routing(*args)
        self.assertNotEqual(rec['uniform_removed_fraction'][1], float(args[3][1].mean()))

    def test_prefixes_inputs_preserved_and_nan_unchanged(self):
        args = routing_fixture(); before = [x.copy() for x in args]
        scores, rec = audit.independent_routing(*args)
        for n in range(len(args[0])+1):
            small, sub = audit.independent_routing(*(x[:n] for x in args))
            for arm in scores: np.testing.assert_array_equal(small[arm], scores[arm][:n])
            self.assertEqual(sub['event_history'], rec['event_history'][:n])
        for got, old in zip(args, before): np.testing.assert_array_equal(got, old)

    def test_increment_identity_bound_missing_and_shapes_fail(self):
        for kind in ('identity', 'infinite_pre', 'infinite_base', 'infinite_d', 'nan_d', 'bound', 'nan_mismatch', 'nonzero_on_nan', 'shape'):
            args = list(routing_fixture())
            if kind == 'identity': args[0][1, 0] += 1e-13
            elif kind == 'infinite_pre': args[0][1, 0] = math.inf
            elif kind == 'infinite_base': args[1][1, 0] = math.inf
            elif kind == 'infinite_d': args[2][1, 0] = math.inf
            elif kind == 'nan_d': args[2][1, 0] = math.nan
            elif kind == 'bound': args[2][1, 0] = .02; args[0][1, 0] = args[1][1, 0] + .02
            elif kind == 'nan_mismatch': args[0][1, 3] = .1
            elif kind == 'nonzero_on_nan': args[2][1, 3] = .001
            else: args[2] = args[2][:, :4]
            with self.subTest(kind=kind), self.assertRaises(ValueError): audit.independent_routing(*args)

    def test_tiny_valid_increment_has_no_added_budget_floor(self):
        base = np.zeros((2, 2)); d = np.full((2, 2), 1e-300)
        args = (d.copy(), base, d, np.tile([True, False], (2, 1)), np.ones((2, 2), bool))
        scores, rec = audit.independent_routing(*args)
        self.assertEqual(rec['uniform_removed_fraction'][1], .5)
        self.assertEqual(scores[audit.ARMS[2]][1, 0], 5e-301)

    def test_score_and_budget_tamper_and_even_tiny_copy_change_rejected(self):
        args = routing_fixture(); scores, rec = audit.independent_routing(*args)
        bad = copy.deepcopy(scores); bad[audit.ARMS[1]][1, 0] += .0001
        with self.assertRaises(ValueError): audit.verify_routing(*args, bad, rec)
        for row, col in ((0, 0), (2, 1), (4, 0), (1, 4), (1, 0)):
            bad = copy.deepcopy(scores)
            bad[audit.ARMS[1]][row, col] = np.nextafter(bad[audit.ARMS[1]][row, col], math.inf)
            with self.assertRaises(ValueError): audit.verify_routing(*args, bad, rec)
        bad = copy.deepcopy(rec); bad['uniform_removed_fraction'][1] += .01
        with self.assertRaises(ValueError): audit.verify_routing(*args, scores, bad)
        bad = copy.deepcopy(rec); bad['event_history'][1]['candidate_removed_abs_increment'] += .01
        with self.assertRaises(ValueError): audit.verify_routing(*args, scores, bad)

    def test_source_contains_no_production_import_or_execution_entry(self):
        text = Path(audit.__file__).read_text(); tree = ast.parse(text)
        imports = [n.module or '' for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        imports += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        self.assertFalse(any('kernel' in n or 'host' in n for n in imports)); self.assertNotIn('__main__', text)


class AccountingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Reuse only a frozen synthetic constant-price fixture, not a calculation oracle.
        import test_monthly_upside_retention181_independent as previous_tests
        with patch.object(previous_tests, 'audit', audit): cls.args = previous_tests.account_fixture()

    def test_nine_paths45_slices(self):
        result = audit.audit_outcomes(*self.args)
        self.assertEqual((result['checked_paths'], result['checked_slices']), (9, 45))
        self.assertEqual(result['outcome_helper_sha256'], audit.OUTCOME_SHA)
        metric = self.args[4][audit.ARMS[0]]['same_close_zero']['slices']['first88']
        audit.accounting().check_metrics(metric, metric.copy(), 'synthetic training')

    def test_hash_fills_and_slice_failures(self):
        with patch.object(audit, 'OUTCOME_SHA', '0'*64), self.assertRaises(ValueError): audit.audit_outcomes(*self.args)
        args = list(self.args); args[3] = copy.deepcopy(args[3])
        args[3][audit.ARMS[1]]['same_close_zero']['TRADES'][0]['adjusted_units'] += 1
        with self.assertRaises(ValueError): audit.audit_outcomes(*args)
        args = list(self.args); args[4] = copy.deepcopy(args[4])
        args[4][audit.ARMS[2]]['next_open_cost']['slices']['later34']['cagr'] += .1
        with self.assertRaises(ValueError): audit.audit_outcomes(*args)


if __name__ == '__main__':
    unittest.main()
