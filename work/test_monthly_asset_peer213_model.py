"""Pure synthetic tests for same-asset residual transfer; no market/cache reads."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

P = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('tested_asset213', P/'monthly_asset_peer213_model.py')
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)
ETFS = ('C', 'D')


def fixture():
    days = pd.date_range('2013-12-31', '2022-12-31', freq='ME')
    dates = np.repeat(days.values.astype('datetime64[D]'), 4)
    ends = np.repeat((days+pd.offsets.MonthEnd(1)).values.astype('datetime64[D]'), 4)
    symbols = np.tile(['A', 'B', 'C', 'D'], len(days))
    rng = np.random.default_rng(213)
    x = rng.uniform(-.5, .5, (len(dates), 10))
    y = .08*x[:, 0]+.19*x[:, 2]**2+np.tile([-.04, .06, -.03, .01], len(days))
    y = (y.reshape(-1, 4)-y.reshape(-1, 4).mean(axis=1, keepdims=True)).reshape(-1)
    targets = {key: y*(i+1) for i, key in enumerate(m.TARGETS)}
    events = np.array(['2016-08-17', '2018-01-31', '2019-01-31',
                       '2019-11-29', '2021-02-26', '2021-12-31'], dtype='datetime64[D]')
    return [dates, ends, symbols, x, targets, np.ones(len(dates), bool), events,
            np.array(['A', 'D', 'B', 'C', 'A', 'D']),
            rng.uniform(-.5, .5, (len(events), 10)), np.ones(len(events), bool)]


def cached(args, minimum=18, own_minimum=18):
    old = m.reference.annual_forecasts(*args, min_months=minimum, min_own_months=own_minimum)
    out = {'raw_'+key: value.copy() for key, value in old['raw'].items()}
    out.update({key: old[key].copy() for key in ('available', 'local_available')})
    return out


def run(args, etfs=ETFS, cache=None, minimum=18, own_minimum=18):
    if cache is None:
        cache = cached(args, minimum, own_minimum)
    return m.annual_forecasts(*args, etfs=etfs, cached_reference=cache,
                             min_months=minimum, min_own_months=own_minimum)


def prefix(args, n, physical=True):
    a = copy.deepcopy(args)
    for i in (6, 7, 8, 9):
        a[i] = a[i][:n]
    if physical:
        cutoff = np.datetime64('%04d-12-31' % (int(a[6].max().astype('datetime64[Y]').astype(int))+1969)) if n else None
        use = a[1] <= cutoff if n else np.zeros(len(a[0]), bool)
        for i in (0, 1, 2, 3, 5):
            a[i] = a[i][use]
        a[4] = {key: value[use] for key, value in a[4].items()}
    return a


class ModelTests(unittest.TestCase):
    def test_cache_snapshots_and_all_peer_control_are_exact(self):
        a = fixture(); c = cached(a); before = copy.deepcopy(c); z = run(a, cache=c)
        self.assertEqual(set(z), {'raw', 'available', 'local_available', 'audit'})
        self.assertEqual(tuple(z['raw']), m.RAW_KEYS)
        for key in m.RAW_KEYS[:3]:
            np.testing.assert_array_equal(z['raw'][key], c['raw_'+key])
        np.testing.assert_array_equal(z['raw']['control'], c['raw_candidate'])
        for key in ('available', 'local_available'):
            np.testing.assert_array_equal(z[key], c[key])
        for key in c:
            self.assertEqual(c[key].tobytes(), before[key].tobytes())
        self.assertTrue(np.any(z['raw']['candidate'] != z['raw']['control']))

    def test_only_one_log_regression_no_source_or_local_linear_refit(self):
        a = fixture(); c = cached(a)
        old_fit = m.numeric._fit_ridge
        with patch.object(m.reference, 'annual_forecasts', side_effect=AssertionError('206 refit forbidden')), \
             patch.object(m.local, 'annual_forecasts', side_effect=AssertionError('4target/linear refit forbidden')), \
             patch.object(m.numeric, '_fit_ridge', wraps=old_fit) as fit:
            z = run(a, cache=c)
        expected = sum(r['prediction_rows'] > 0 for r in z['audit']['annual'])
        self.assertEqual(fit.call_count, expected)
        self.assertEqual(z['audit']['counts']['actual_regression_solves'], expected)
        self.assertEqual(z['audit']['source_four_target_refits'], 0)
        self.assertEqual(z['audit']['local_linear_refits'], 0)
        self.assertNotIn('new_model_fits', z['audit'])

    def test_same_type_only_changes_peer_domain_not_own_mass(self):
        a = fixture(); z = run(a)
        for row in z['audit']['local']:
            self.assertEqual(row['status'], 'same_type_supported')
            self.assertEqual(row['correction_denominator'], row['weight_mass']+12.)
            self.assertEqual(len(row['peer_months']), row['own_months'])
            self.assertEqual(row['asset_type'], 'etf' if row['symbol'] in ETFS else 'stock')
            for entry, own_id, weight in zip(row['peer_months'], row['own_row_ids'], row['own_weights']):
                self.assertEqual(entry['own_row_id'], own_id)
                self.assertEqual(entry['own_weight'], weight)
                self.assertEqual(entry['peer_count'], 3)
                self.assertEqual(entry['same_peer_count'], 1)
                self.assertNotIn(own_id, entry['peer_row_ids'])
                self.assertTrue(set(entry['same_peer_row_ids']) <= set(entry['peer_row_ids']))
                for p in entry['peer_row_ids']:
                    self.assertNotEqual(a[2][p], row['symbol'])
                    self.assertEqual(str(a[0][p].astype('datetime64[M]')), entry['month'])
                for p in entry['same_peer_row_ids']:
                    self.assertEqual(a[2][p] in ETFS, row['symbol'] in ETFS)
            num = float(np.einsum('n,n->', np.array(row['own_weights']),
                np.array([e['same_peer_mean'] for e in row['peer_months']]), optimize=False))
            self.assertEqual(row['same_peer_numerator'], num)
            self.assertEqual(row['same_peer_correction'], num/row['correction_denominator'])
            self.assertEqual(z['raw']['candidate'][row['event_row']],
                             z['raw']['source_log'][row['event_row']]+row['same_peer_correction'])

    def test_all_one_type_exactly_recovers_all_peer(self):
        a = fixture(); c = cached(a)
        for types in ([], ['A', 'B', 'C', 'D']):
            z = run(a, types, c)
            np.testing.assert_array_equal(z['raw']['candidate'], z['raw']['control'])
            for r in z['audit']['local']:
                for e in r['peer_months']:
                    self.assertEqual(e['same_peer_row_ids'], e['peer_row_ids'])
                    self.assertEqual(e['same_peer_mean'], e['peer_mean'])

    def test_singleton_type_falls_back_whole_query_without_partial_estimates(self):
        a = fixture(); z = run(a, ['D'])
        for row in z['audit']['local']:
            if row['symbol'] == 'D':
                self.assertEqual(row['status'], 'same_type_missing_month')
                self.assertEqual(len(row['missing_same_type_months']), row['own_months'])
                self.assertIsNone(row['same_peer_numerator'])
                self.assertIsNone(row['same_peer_correction'])
                self.assertEqual(z['raw']['candidate'][row['event_row']], z['raw']['control'][row['event_row']])
                for entry in row['peer_months']:
                    self.assertEqual(entry['same_peer_count'], 0)
                    self.assertIsNone(entry['same_peer_mean'])

    def test_one_missing_month_keeps_all_months_and_no_partial_candidate(self):
        a = prefix(fixture(), 1, physical=False)
        b = np.flatnonzero((a[2] == 'B') & (a[1] <= np.datetime64('2015-12-31')))
        a[5][b[-1]] = False
        z = run(a); row = z['audit']['local'][0]
        self.assertEqual(row['status'], 'same_type_missing_month')
        self.assertEqual(row['missing_same_type_months'], [str(a[0][b[-1]].astype('datetime64[M]'))])
        self.assertEqual(len(row['peer_months']), row['own_months'])
        self.assertTrue(any(e['same_peer_count'] for e in row['peer_months']))
        self.assertTrue(all(e['same_peer_mean'] is None for e in row['peer_months']))
        self.assertEqual(z['audit']['counts']['same_type_month_evaluations'], 0)
        np.testing.assert_array_equal(z['raw']['candidate'], z['raw']['control'])

    def test_own17_both_pooled_and_own18_retains_correction_support(self):
        a = prefix(fixture(), 1, physical=False)
        own = np.flatnonzero((a[2] == 'A') & (a[1] <= np.datetime64('2015-12-31')))
        a[5][own[:-18]] = False
        z = run(a); self.assertTrue(z['local_available'][0])
        self.assertEqual(z['audit']['local'][0]['own_months'], 18)
        a[5][own[-18]] = False
        z = run(a); self.assertFalse(z['local_available'][0])
        row = z['audit']['local'][0]
        self.assertEqual(row['status'], 'insufficient_own_history')
        self.assertEqual(row['peer_months'], [])
        for key in ('candidate', 'control'):
            np.testing.assert_array_equal(z['raw'][key], z['raw']['source_log'])
        self.assertEqual(z['audit']['counts']['same_type_fallback_queries'], 0)

    def test_actual_counts_aggregate_full_estimates_not_repeated_fits(self):
        z = run(fixture()); a = z['audit']; c = a['counts']; rows = a['local']
        supported = [r for r in rows if r['status'] == 'same_type_supported']
        fitted = [r for r in rows if r['peer_months']]
        self.assertEqual(set(c), set(m.COUNT_FIELDS))
        self.assertEqual(len(c), 11)
        self.assertEqual(c['cached_forecast_rows'], int(z['available'].sum()))
        self.assertEqual(c['annual_pooled_log_refits'], sum(x['prediction_rows'] > 0 for x in a['annual']))
        self.assertEqual(c['actual_regression_solves'], c['annual_pooled_log_refits'])
        self.assertEqual(c['own_weight_evaluations'], len(fitted))
        self.assertEqual(c['all_peer_query_evaluations'], len(fitted))
        self.assertEqual(c['same_type_supported_queries']+c['same_type_fallback_queries'], len(fitted))
        self.assertEqual(c['all_peer_month_evaluations'], sum(len(r['peer_months']) for r in fitted))
        self.assertEqual(c['all_peer_row_observations'], sum(e['peer_count'] for r in fitted for e in r['peer_months']))
        self.assertEqual(c['same_type_month_evaluations'], sum(len(r['peer_months']) for r in supported))
        self.assertEqual(c['same_type_row_observations'], sum(e['same_peer_count'] for r in supported for e in r['peer_months']))

    def test_exact_annual_maturity_exclusive_five_year_and_minimum_two(self):
        a = prefix(fixture(), 1, physical=False); a[6][0] = np.datetime64('2021-01-04')
        a[5][:3] = False
        z = run(a); annual = z['audit']['annual'][0]
        self.assertEqual(annual['lookback_exclusive_start'], '2015-12-31')
        self.assertEqual(annual['first_training_signal'], '2016-01-31')
        self.assertEqual(annual['latest_label_end'], '2020-12-31')
        for i in annual['training_row_ids']:
            self.assertGreater(a[0][i], np.datetime64(annual['lookback_exclusive_start']))
            self.assertLessEqual(a[1][i], np.datetime64(annual['cutoff']))
        b = prefix(fixture(), 1, physical=False); b[5][:3] = False
        z = run(b)
        self.assertTrue(all(i not in z['audit']['annual'][0]['training_row_ids'] for i in range(4)))

    def test_shared_four_targets_and_unknown_query_has_no_future_filter(self):
        a = fixture(); a[4]['loss_utility'][0] = np.nan
        z = run(a)
        self.assertTrue(all(0 not in row['training_row_ids'] for row in z['audit']['annual']))
        a[7] = a[7].astype(object); a[7][-1] = 'UNKNOWN'
        z = run(a)
        self.assertTrue(z['available'][-1]); self.assertFalse(z['local_available'][-1])
        self.assertEqual(z['raw']['candidate'][-1], z['raw']['source_log'][-1])

    def test_physical_prefixes_and_all_input_bytes_unchanged(self):
        a = fixture(); before = copy.deepcopy(a); c = cached(a); full = run(a, cache=c)
        for n in range(len(a[6])+1):
            z = run(prefix(a, n), cache={k: v[:n] for k, v in c.items()})
            for key in m.RAW_KEYS:
                np.testing.assert_array_equal(z['raw'][key], full['raw'][key][:n])
            for key in ('available', 'local_available'):
                np.testing.assert_array_equal(z[key], full[key][:n])
        for i, value in enumerate(a):
            if isinstance(value, dict):
                for k in value: self.assertEqual(value[k].tobytes(), before[i][k].tobytes())
            else: self.assertEqual(value.tobytes(), before[i].tobytes())

    def test_future_poison_not_consumed_and_empty_prefix(self):
        a = prefix(fixture(), 2, physical=False); c = cached(a); expected = run(a, cache=c)
        future = a[1] > np.datetime64('2017-12-31')
        a[3] = a[3].astype(object); a[3][future] = 'future poison'
        a[5] = a[5].astype(object); a[5][future] = 'future poison'
        for key in a[4]:
            a[4][key] = a[4][key].astype(object); a[4][key][future] = 'future poison'
        z = run(a, cache=c)
        self.assertEqual(z['audit'], expected['audit'])
        for key in m.RAW_KEYS:
            np.testing.assert_array_equal(z['raw'][key], expected['raw'][key])
        z = run(prefix(a, 0, physical=False), cache={k: v[:0] for k, v in c.items()})
        self.assertEqual(z['audit']['annual'], []); self.assertEqual(z['audit']['local'], [])
        self.assertTrue(all(x == 0 for x in z['audit']['counts'].values()))

    def test_three_exact_cache_parities_reject_independent_mutations(self):
        a = fixture(); original = cached(a)
        for key in ('raw_source_log', 'raw_local_constant', 'raw_candidate'):
            c = copy.deepcopy(original); c[key][0] += .01
            if key == 'raw_local_constant': c['raw_control'][0] = c[key][0]
            with self.subTest(key=key), self.assertRaises(AssertionError): run(a, cache=c)
        c = copy.deepcopy(original); c['raw_local_linear'] += .01
        z = run(a, cache=c)
        np.testing.assert_array_equal(z['raw']['local_linear'], c['raw_local_linear'])
        self.assertEqual(z['audit']['local_linear_refits'], 0)

    def test_cached_masks_schema_dtype_and_nonfinite_rejected(self):
        a = fixture(); original = cached(a)
        mutations = ('missing', 'extra', 'float32', 'inf', 'nan', 'badbool', 'available', 'local')
        for case in mutations:
            c = copy.deepcopy(original)
            if case == 'missing': del c['raw_control']
            if case == 'extra': c['extra'] = 1
            if case == 'float32': c['raw_candidate'] = c['raw_candidate'].astype('float32')
            if case == 'inf': c['raw_candidate'][0] = np.inf
            if case == 'nan': c['raw_candidate'][0] = np.nan
            if case == 'badbool': c['available'] = c['available'].astype(int)
            if case == 'available': c['available'][0] = False
            if case == 'local': c['local_available'][0] = False
            with self.subTest(case=case), self.assertRaises((ValueError, AssertionError)):
                run(a, cache=c)

    def test_explicit_asset_identity_not_ticker_heuristics(self):
        a = fixture(); c = cached(a)
        for etfs in (None, 'D', ['D', 'D'], [''], [' D']):
            with self.subTest(etfs=etfs), self.assertRaises(ValueError): run(a, etfs, c)
        z = run(a, ['A', 'OUTSIDE_POOL'], c)
        self.assertEqual(z['audit']['etfs'], ['A', 'OUTSIDE_POOL'])
        for row in z['audit']['local']:
            self.assertEqual(row['asset_type'], 'etf' if row['symbol'] == 'A' else 'stock')

    def test_cold_start_inactive_query_and_zero_residual(self):
        a = fixture(); a[9][:] = False; z = run(a)
        self.assertFalse(z['available'].any())
        self.assertTrue(all(v == 0 for v in z['audit']['counts'].values()))
        a = prefix(fixture(), 1); a[5][:] = False; z = run(a)
        self.assertEqual(z['audit']['annual'][0]['status'], 'cold_start')
        self.assertEqual(z['audit']['local'], [])
        a = fixture()
        for v in a[4].values(): v[:] = 0.
        z = run(a)
        for key in m.RAW_KEYS:
            np.testing.assert_array_equal(z['raw'][key], np.zeros(len(a[6])))
        self.assertGreater(z['audit']['counts']['same_type_month_evaluations'], 0)

    def test_protocol_bad_shapes_booleans_duplicate_and_minimum(self):
        a = fixture(); c = cached(a)
        for case in ('shape', 'boolean', 'duplicate', 'minimum'):
            b = copy.deepcopy(a)
            if case == 'shape': b[3] = b[3][:, :9]
            if case == 'boolean': b[5] = b[5].astype(int)
            if case == 'duplicate': b[2][1] = b[2][0]
            with self.subTest(case=case), self.assertRaises(ValueError):
                run(b, cache=c, minimum=True if case == 'minimum' else 18)

    def test_nonfinite_derived_arithmetic_never_falls_back(self):
        a = fixture(); c = cached(a)
        with patch.object(m.reference, 'peer_month_average', side_effect=FloatingPointError('finite overflow')):
            with self.assertRaises(FloatingPointError): run(a, ['A'], c)
        with self.assertRaises(FloatingPointError):
            m._add(1e308, 1e308)


if __name__ == '__main__':
    unittest.main(verbosity=2)
