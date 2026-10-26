"""Artificial-data-only tests for the preparatory paired target interface."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import sys
import unittest

import numpy as np
import pandas as pd

import after142_paired_target_labels as subject


def calendar(start='2019-12-01', end='2022-03-31'):
    # Explicit synthetic exchange schedule; no market data/provider is used.
    return pd.bdate_range(start, end).difference(pd.to_datetime(['2020-12-25', '2021-01-01']))


def fixture(dates=('2020-11-30',), symbols=('A', 'B', 'C'), sessions=None):
    sessions = calendar() if sessions is None else sessions
    close = pd.DataFrame(100., index=sessions, columns=['A', 'B', 'C', 'SPY'])
    opening = close.copy()
    flat_dates, flat_symbols = [], []
    for date in dates:
        s = pd.Timestamp(date)
        period = sessions.to_period('M')
        e = sessions[period == s.to_period('M') + 1][-1]
        a = sessions[sessions > s][0]
        b = sessions[sessions > e][0]
        close.loc[e] = [120., 110., 90., 105.]
        opening.loc[a] = [100., 100., 100., 100.]
        opening.loc[b] = [118., 104., 92., 104.]
        flat_dates.extend([date] * len(symbols)); flat_symbols.extend(symbols)
    n = len(flat_dates)
    return dict(adjusted_close=close, adjusted_open=opening, official_calendar=sessions,
                dates=np.array(flat_dates, dtype='datetime64[D]'), symbols=np.array(flat_symbols),
                original_common_good=np.ones(n, bool), original_eligible=np.ones(n, bool),
                original_features_finite=np.ones(n, bool))


def build(args):
    return subject.build_paired_targets(**args)


def assert_labels_equal(test, left, right):
    for name in subject.ARMS:
        np.testing.assert_array_equal(left['targets'][name], right['targets'][name])
        np.testing.assert_array_equal(left['raw_targets'][name], right['raw_targets'][name])
    np.testing.assert_array_equal(left['good'], right['good'])


def maturity(result, sessions, cutoff):
    return subject.mature_training_mask(sessions, result['dates'], result['calendar_ends'],
        result['entry_sessions'], result['exit_sessions'], result['available_sessions'],
        result['good'], cutoff)


class PairedTargetTests(unittest.TestCase):
    def test_december_january_endpoints(self):
        z = build(fixture())
        for key, day in [('calendar_ends','2020-12-31'), ('entry_sessions','2020-12-01'),
                         ('exit_sessions','2021-01-04'), ('available_sessions','2021-01-04')]:
            np.testing.assert_array_equal(z[key], np.array([day] * 3, dtype='datetime64[D]'))
        self.assertTrue(z['good'].all())

    def test_shared_maturity_uses_january_not_december_close(self):
        f = fixture(); z = build(f)
        self.assertFalse(maturity(z, f['official_calendar'], '2020-12-31').any())
        self.assertFalse(maturity(z, f['official_calendar'], '2021-01-03').any())
        self.assertTrue(maturity(z, f['official_calendar'], '2021-01-04').all())

    def test_exclusive_five_year_signal_boundary(self):
        sessions = calendar('2015-01-01', '2021-03-31')
        f = fixture(('2015-11-30', '2015-12-31'), sessions=sessions); z = build(f)
        np.testing.assert_array_equal(maturity(z, sessions, '2020-11-30'), [False]*3 + [True]*3)

    def test_calendar_object_sessions(self):
        f = fixture(); old = build(f); f['official_calendar'] = SimpleNamespace(sessions=f['official_calendar'])
        assert_labels_equal(self, old, build(f))

    def test_truncated_calendar_is_error(self):
        f = fixture(); f['official_calendar'] = f['official_calendar'][f['official_calendar'] <= '2020-12-31']
        with self.assertRaisesRegex(ValueError, 'extend'): build(f)

    def test_entire_target_month_absent_is_error(self):
        f = fixture(); x = f['official_calendar']; f['official_calendar'] = x[x.to_period('M') != pd.Period('2020-12')]
        with self.assertRaisesRegex(ValueError, 'target month'): build(f)

    def test_missing_exit_month_cannot_jump_an_extra_month(self):
        f = fixture(); x = f['official_calendar']; f['official_calendar'] = x[x.to_period('M') != pd.Period('2021-01')]
        with self.assertRaisesRegex(ValueError, 'monthly boundary'): build(f)

    def test_non_month_end_training_signal_rejected(self):
        f = fixture(); f['dates'][:] = np.datetime64('2020-11-27')
        with self.assertRaisesRegex(ValueError, 'month-end'): build(f)

    def test_signal_not_an_official_session_rejected(self):
        f = fixture(); f['dates'][:] = np.datetime64('2020-11-29')
        with self.assertRaisesRegex(ValueError, 'month-end'): build(f)

    def test_calendar_duplicate_unsorted_timezone_rejected(self):
        for mode in ('duplicate', 'unsorted', 'timezone'):
            with self.subTest(mode=mode):
                f = fixture(); x = f['official_calendar']
                f['official_calendar'] = x.append(x[:1]) if mode=='duplicate' else x[::-1] if mode=='unsorted' else x.tz_localize('UTC')
                with self.assertRaises(ValueError): build(f)

    def test_invalid_calendar_type_rejected(self):
        f = fixture(); f['official_calendar'] = list(f['official_calendar'])
        with self.assertRaises(ValueError): build(f)

    def test_truncated_price_cache_keeps_true_future_endpoints(self):
        f = fixture()
        for key in ('adjusted_close', 'adjusted_open'): f[key] = f[key].loc[:'2020-12-31']
        z = build(f); self.assertFalse(z['good'].any())
        self.assertEqual(str(z['exit_sessions'][0]), '2021-01-04')
        self.assertEqual(z['audit']['months'][0]['reason'], 'missing_price_session')
        self.assertTrue(np.isnan(z['targets']['paired_close']).all())

    def test_missing_entry_price_session_is_not_interpolated(self):
        f = fixture()
        for key in ('adjusted_close', 'adjusted_open'): f[key] = f[key].drop(pd.Timestamp('2020-12-01'))
        z = build(f); self.assertFalse(z['good'].any())
        self.assertEqual(str(z['entry_sessions'][0]), '2020-12-01')

    def test_bad_single_asset_endpoint_removes_both_labels(self):
        for bad in (np.nan, np.inf, -np.inf, 0., -1., 'not a price'):
            with self.subTest(bad=bad):
                f = fixture(); f['adjusted_open'] = f['adjusted_open'].astype(object)
                f['adjusted_open'].loc['2021-01-04', 'A'] = bad
                z = build(f); np.testing.assert_array_equal(z['good'], [False, True, True])
                for key in subject.ARMS:
                    self.assertTrue(np.isnan(z['targets'][key][0]))
                    self.assertAlmostEqual(z['targets'][key][1:].sum(), 0.)

    def test_missing_benchmark_endpoint_removes_month(self):
        f = fixture(); f['adjusted_open'].loc['2021-01-04', 'SPY'] = np.nan
        z = build(f); self.assertFalse(z['good'].any())
        self.assertEqual(z['audit']['months'][0]['reason'], 'invalid_benchmark_endpoint')

    def test_missing_symbol_and_missing_spy_columns(self):
        for symbol in ('A', 'SPY'):
            f = fixture()
            for key in ('adjusted_close', 'adjusted_open'): f[key] = f[key].drop(columns=symbol)
            z = build(f)
            np.testing.assert_array_equal(z['good'], [False,True,True] if symbol=='A' else [False]*3)

    def test_month_with_only_one_paired_row_has_no_targets(self):
        f = fixture(); f['adjusted_open'].loc['2021-01-04', ['A','B']] = np.nan
        z = build(f); self.assertFalse(z['good'].any())
        self.assertEqual(z['endpoint_good'].sum(), 1)
        for key in subject.ARMS: self.assertTrue(np.isnan(z['targets'][key]).all())

    def test_original_masks_cannot_be_expanded(self):
        for key in ('original_common_good', 'original_eligible', 'original_features_finite'):
            f = fixture(); f[key][0] = False; z = build(f)
            np.testing.assert_array_equal(z['good'], [False,True,True])
            for arm in subject.ARMS: self.assertTrue(np.isnan(z['raw_targets'][arm][0]))

    def test_excluded_row_unparseable_values_not_read(self):
        f = fixture(); f['original_common_good'][0] = False
        for key in ('adjusted_close', 'adjusted_open'):
            f[key] = f[key].astype(object); f[key]['A'] = 'excluded row poison'
        z = build(f); np.testing.assert_array_equal(z['good'], [False,True,True])

    def test_zero_gap_labels_exactly_identical(self):
        f = fixture(); f['adjusted_open'].loc['2020-12-01'] = f['adjusted_close'].loc['2020-11-30']
        f['adjusted_open'].loc['2021-01-04'] = f['adjusted_close'].loc['2020-12-31']
        z = build(f)
        np.testing.assert_array_equal(z['targets']['paired_close'], z['targets']['paired_open'])
        np.testing.assert_array_equal(z['raw_targets']['paired_close'], z['raw_targets']['paired_open'])

    def test_opposing_overnight_gaps_reverse_relative_label_order(self):
        f = fixture(); f['adjusted_open'].loc['2020-12-01', ['A','B']] = [130.,90.]
        f['adjusted_open'].loc['2021-01-04', ['A','B']] = [120.,110.]
        z = build(f)
        self.assertGreater(z['targets']['paired_close'][0], z['targets']['paired_close'][1])
        self.assertLess(z['targets']['paired_open'][0], z['targets']['paired_open'][1])

    def test_actual_spy_open_endpoints_are_used(self):
        f = fixture(); f['adjusted_open'].loc['2020-12-01','SPY'] = 90.
        f['adjusted_open'].loc['2021-01-04','SPY'] = 130.
        z = build(f); expected = np.log(118.)-np.log(100.)-np.log(130.)+np.log(90.)
        self.assertAlmostEqual(z['raw_targets']['paired_open'][0], expected)

    def test_clip_then_common_month_center_not_center_then_clip(self):
        f = fixture(); f['adjusted_close'].loc['2020-12-31'] = [10000.,1.,1.,100.]
        z = build(f); np.testing.assert_allclose(z['targets']['paired_close'], [4./3.,-2./3.,-2./3.])
        self.assertEqual(z['audit']['months'][0]['clip_counts']['paired_close'], 3)
        self.assertGreater(z['targets']['paired_close'][0], 1.)

    def test_fixed_positive_per_security_scale_invariance(self):
        f = fixture(); z = build(f); scale = pd.Series([.00001,100.,2.3,.002], index=f['adjusted_close'].columns)
        for key in ('adjusted_close','adjusted_open'): f[key] = f[key].mul(scale, axis='columns')
        other = build(f)
        for arm in subject.ARMS:
            np.testing.assert_allclose(z['raw_targets'][arm], other['raw_targets'][arm], rtol=0., atol=3e-15)
            np.testing.assert_allclose(z['targets'][arm], other['targets'][arm], rtol=0., atol=3e-15)

    def test_positive_finite_extremes_use_stable_log_differences(self):
        f = fixture(); f['adjusted_open'].loc['2020-12-01','A'] = 1e-300
        f['adjusted_open'].loc['2021-01-04','A'] = 1e300
        z = build(f); self.assertTrue(np.isfinite(z['raw_targets']['paired_open']).all())
        self.assertTrue(z['good'].all()); self.assertGreater(z['raw_targets']['paired_open'][0], 1000.)

    def test_unused_prices_and_future_values_not_converted(self):
        f = fixture(); z = build(f)
        used = {'adjusted_close':pd.to_datetime(['2020-11-30','2020-12-31']),
                'adjusted_open':pd.to_datetime(['2020-12-01','2021-01-04'])}
        for key in used:
            f[key] = f[key].astype(object); f[key].loc[~f[key].index.isin(used[key])] = 'unused future/history poison'
        assert_labels_equal(self, z, build(f))

    def test_no_input_mutation_or_output_mask_alias(self):
        f = fixture(); frames={k:f[k].copy(deep=True) for k in ('adjusted_close','adjusted_open')}
        arrays={k:f[k].copy() for k in ('dates','symbols','original_common_good','original_eligible','original_features_finite')}
        z = build(f); z['good'][:] = False; z['base_good'][:] = False; z['dates'][:] = np.datetime64('1999-01-01')
        z['symbols'][0] = 'Z'
        for key in frames: pd.testing.assert_frame_equal(f[key], frames[key])
        for key in arrays: np.testing.assert_array_equal(f[key], arrays[key])

    def test_months_center_independently(self):
        f = fixture(('2020-10-30','2020-11-30')); z = build(f)
        for arm in subject.ARMS:
            self.assertAlmostEqual(z['targets'][arm][:3].sum(),0.)
            self.assertAlmostEqual(z['targets'][arm][3:].sum(),0.)

    def test_later_training_row_poison_does_not_change_earlier_labels(self):
        f = fixture(('2020-10-30','2020-11-30')); z = build(f)
        f['adjusted_open'].loc['2021-01-04'] = np.nan; changed = build(f)
        for arm in subject.ARMS: np.testing.assert_array_equal(z['targets'][arm][:3], changed['targets'][arm][:3])
        self.assertFalse(changed['good'][3:].any())

    def test_nonboolean_masks_rejected(self):
        for key in ('original_common_good','original_eligible','original_features_finite'):
            for bad in (np.ones(3,int), np.ones(3,float), np.ones(3,object), np.ones(2,bool)):
                f = fixture(); f[key] = bad
                with self.assertRaises(ValueError): build(f)

    def test_duplicate_identity_rejected(self):
        f = fixture(); f['symbols'][1] = 'A'
        with self.assertRaisesRegex(ValueError, 'duplicate security'): build(f)

    def test_bad_symbol_and_date_shape_rejected(self):
        for change in ('space','number','length','numeric_date','timezone','nat'):
            f = fixture()
            if change=='space': f['symbols'] = np.array(['A ','B','C'])
            elif change=='number': f['symbols'] = np.array([1,2,3])
            elif change=='length': f['symbols'] = np.array(['A','B'])
            elif change=='numeric_date': f['dates'] = np.array([20201130]*3)
            elif change=='timezone': f['dates'] = pd.DatetimeIndex(['2020-11-30']*3,tz='UTC')
            else: f['dates'][0] = np.datetime64('NaT')
            with self.subTest(change=change), self.assertRaises(ValueError): build(f)

    def test_misaligned_or_duplicate_price_axes_rejected(self):
        for change in ('columns','rows','duplicate','unsorted','intraday'):
            f = fixture()
            if change=='columns': f['adjusted_open'] = f['adjusted_open'][['SPY','A','B','C']]
            elif change=='rows': f['adjusted_open'] = f['adjusted_open'].iloc[1:]
            elif change=='duplicate': f['adjusted_open'] = pd.concat([f['adjusted_open'],f['adjusted_open'].iloc[:1]])
            elif change=='unsorted': f['adjusted_open'] = f['adjusted_open'].iloc[::-1]
            else: f['adjusted_open'].index = f['adjusted_open'].index + pd.Timedelta(hours=1)
            with self.subTest(change=change), self.assertRaises(ValueError): build(f)

    def test_maturity_dates_cannot_be_forged(self):
        f = fixture(); z = build(f)
        original=[z['calendar_ends'],z['entry_sessions'],z['exit_sessions'],z['available_sessions']]
        for i in range(4):
            args=[v.copy() for v in original]; args[i][0]=np.datetime64('2020-12-31') if i else np.datetime64('2020-12-30')
            with self.subTest(i=i), self.assertRaises(ValueError): subject.validate_maturity_interface(f['official_calendar'],z['dates'],*args)

    def test_maturity_bad_mask_cutoff_and_lengths_rejected(self):
        f = fixture(); z = build(f)
        with self.assertRaises(ValueError): maturity(z, f['official_calendar'], True)
        z['good'] = np.ones(3,int)
        with self.assertRaises(ValueError): maturity(z, f['official_calendar'], '2021-01-04')
        z['good'] = np.ones(3,bool); z['available_sessions'] = z['available_sessions'][:2]
        with self.assertRaises(ValueError): maturity(z, f['official_calendar'], '2021-01-04')

    def test_empty_explicit_date_input(self):
        f = fixture(); f['dates'] = np.array([],dtype='datetime64[D]'); f['symbols']=np.array([],dtype='U1')
        for key in ('original_common_good','original_eligible','original_features_finite'): f[key]=np.array([],bool)
        z=build(f); self.assertEqual(z['good'].shape,(0,)); self.assertEqual(z['audit']['months'],[])


if __name__ == '__main__':
    write_receipt = sys.argv[1:] == ['--receipt']
    if sys.argv[1:] and not write_receipt:
        raise SystemExit('Only --receipt is supported')
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PairedTargetTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if write_receipt:
        root = Path(__file__).resolve().parents[1]
        files = [Path(__file__).resolve(), Path(subject.__file__).resolve(), subject._VALIDATOR_PATH,
                 root/'run_local.sh', root/'local_run.py']
        receipt = dict(stage='preparatory_paired_target_synthetic_only', registered=False,
            passed=result.wasSuccessful(), tests_run=result.testsRun, failures=len(result.failures),
            errors=len(result.errors), skipped=len(result.skipped), real_labels_constructed=0,
            real_model_fits=0, real_factor_rank_or_portfolio_return_calculations=0,
            fixtures='Only artificial constant/hand-set prices and explicitly synthetic session schedules.',
            command='./run_local.sh work/after142_paired_target_labels_tests.py --receipt',
            limitations=['Exact prior industry-target novelty remains unresolved.',
                'No actual source closure or endpoint-price coverage was checked.',
                'No annual model or portfolio selector is implemented or run.',
                'Explicit calendar provenance is the caller responsibility; an index cannot prove missing interior sessions.',
                'Passing pure tests is not backtest admission.'],
            python=sys.version, numpy=np.__version__, pandas=pd.__version__,
            source_hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
        (root/'work/after142_paired_target_synthetic.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
