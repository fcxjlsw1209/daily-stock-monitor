"""Synthetic-only directional-impact arithmetic, axes and historical-window tests."""
from pathlib import Path
import copy
import importlib.util
import json
import math
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location('directional_impact70_synthetic', Path(__file__).with_name('features.py'))
f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f)
ETFS = ['ETF%02d' % i for i in range(17)]
SYMBOLS = ['STOCK%02d' % i for i in range(32)]+ETFS
KEYS = {'raw_dollar', 'raw_price', 'rank_dollar', 'rank_price',
        'impact_available', 'n_positive', 'n_negative'}


def fixture(rows=345):
    rng = np.random.default_rng(72461)
    days = pd.bdate_range('2000-01-03', periods=rows)
    simple = rng.normal(.0001, .025, (rows, 49))
    adj = 50.*np.cumprod(1.+simple, axis=0)
    close = adj*np.exp(.2*np.sin(np.arange(rows)[:, None]/13.+np.arange(49)[None, :]))
    volume = 1e6*np.exp(rng.normal(0., .5, (rows, 49)))
    panel = {key: pd.DataFrame(value, index=days, columns=SYMBOLS)
             for key, value in [('adj', adj), ('close', close), ('volume', volume)]}
    return panel, [(days[294], SYMBOLS.copy()), (days[320], SYMBOLS.copy())]


def scalar_oracle(adjusted, close, volume):
    """Unscaled scalar formula: no production helpers or numpy reductions."""
    r = [float(adjusted[k+1])/float(adjusted[k])-1. for k in range(252)]
    positive, negative = [k for k, x in enumerate(r) if x > 0.], [k for k, x in enumerate(r) if x < 0.]
    u = math.fsum(r[k]/(float(close[k])*float(volume[k])) for k in positive)/len(positive)
    l = math.fsum(-r[k]/(float(close[k])*float(volume[k])) for k in negative)/len(negative)
    u0 = math.fsum(r[k] for k in positive)/len(positive)
    l0 = math.fsum(-r[k] for k in negative)/len(negative)
    return (u-l)/(u+l), (u0-l0)/(u0+l0), len(positive), len(negative)


def set_count_path(panel, column, positive, negative):
    r = np.r_[np.ones(positive), np.full(negative, -.5), np.zeros(252-positive-negative)]
    panel['adj'].iloc[:253, column] = np.r_[1., np.cumprod(1.+r)]


class FeatureTests(unittest.TestCase):
    def test_independent_scalar_formula_counts_and_full49_ranks(self):
        panel, events = fixture(); got, audit = f.prepare(panel, events, ETFS)
        wrong_adjusted_dollars, wrong_logs = [], []
        for i, (day, symbols) in enumerate(events):
            t = panel['adj'].index.get_loc(day)
            for j, symbol in enumerate(symbols):
                a = panel['adj'][symbol].iloc[t-294:t-41].to_numpy()
                c = panel['close'][symbol].iloc[t-293:t-41].to_numpy()
                v = panel['volume'][symbol].iloc[t-293:t-41].to_numpy()
                expected = scalar_oracle(a, c, v)
                self.assertAlmostEqual(got['raw_dollar'][i, j], expected[0], places=14)
                self.assertAlmostEqual(got['raw_price'][i, j], expected[1], places=14)
                self.assertEqual(got['n_positive'][i, j], expected[2])
                self.assertEqual(got['n_negative'][i, j], expected[3])
                wrong_adjusted_dollars.append(abs(expected[0]-scalar_oracle(a, a[1:], v)[0]))
                r = np.log(a[1:]/a[:-1]); u, l = r[r > 0.].mean(), -r[r < 0.].mean()
                wrong_logs.append(abs(expected[1]-(u-l)/(u+l)))
            for raw, ranked in [('raw_dollar', 'rank_dollar'), ('raw_price', 'rank_price')]:
                # Independent average ordinal ranks, including exact ties.
                row = got[raw][i]
                expected_rank = [(sum(row < x)+(sum(row == x)+1)/2)/49-.5 for x in row]
                np.testing.assert_array_equal(got[ranked][i], expected_rank)
        self.assertEqual(set(got), KEYS)
        self.assertTrue(got['impact_available'].all())
        self.assertGreater(max(wrong_adjusted_dollars), .001)
        self.assertGreater(max(wrong_logs), .001)
        for key, value in got.items():
            self.assertEqual(value.shape, (2, 49))
            expected_dtype = bool if key == 'impact_available' else np.int64 if key.startswith('n_') else np.float64
            self.assertEqual(value.dtype, np.dtype(expected_dtype))
        self.assertEqual(audit['available_cells'], 98)
        self.assertEqual((audit['price_count'], audit['return_count'], audit['raw_count']), (253, 252, 252))
        for key in f.FLOAT_KEYS:
            np.testing.assert_array_equal(np.isfinite(got[key]), got['impact_available'])
            self.assertLessEqual(np.abs(got[key]).max(), .5 if key.startswith('rank_') else 1.)
        json.dumps(audit, allow_nan=False)

    def test_sign_conditioned_means_are_not_sums_or_separately_rescaled(self):
        r = np.r_[np.full(40, .5), np.full(80, -.25), np.zeros(132)]
        dollar, price = f._directional_pair(r, np.ones(252))
        self.assertAlmostEqual(dollar, 1/3, places=15)
        self.assertEqual(dollar, price)
        self.assertNotEqual(dollar, 0.)  # Equal total positive/negative amplitude is irrelevant.
        d = np.r_[np.full(40, 4.), np.full(80, 1.), np.ones(132)]
        dollar, price = f._directional_pair(r, d)
        self.assertAlmostEqual(dollar, -1/3, places=15)
        self.assertAlmostEqual(price, 1/3, places=15)

    def test_40_vs39_each_sign_and_zero_returns_keep_counts(self):
        panel, events = fixture()
        for col, counts in enumerate([(40, 40), (39, 40), (40, 39), (0, 0), (252, 0), (0, 252)]):
            set_count_path(panel, col, *counts)
        # Smallest upward ratio distinguishable from1 remains a strict sign; no epsilon.
        panel['adj'].iloc[:253, 6] = np.r_[np.tile([1., np.nextafter(1., 2.)], 40), np.ones(173)]
        got, audit = f.prepare(panel, events[:1], ETFS)
        np.testing.assert_array_equal(got['n_positive'][0, :6], [40, 39, 40, 0, 252, 0])
        np.testing.assert_array_equal(got['n_negative'][0, :6], [40, 40, 39, 0, 0, 252])
        self.assertTrue(got['impact_available'][0, 0]); self.assertFalse(got['impact_available'][0, 1:6].any())
        for key in f.FLOAT_KEYS:self.assertTrue(np.isnan(got[key][0, 1:6]).all())
        self.assertEqual(audit['insufficient_sign_counts_cells'], 5)
        self.assertTrue((got['n_positive']+got['n_negative'] <= 252).all())
        self.assertEqual((got['n_positive'][0, 6], got['n_negative'][0, 6]), (40, 40))
        self.assertTrue(got['impact_available'][0, 6])

    def test_valid_zero_and_exact_ties_are_not_missing(self):
        panel, events = fixture(); r = np.tile([1., -.5], 126)
        panel['adj'].iloc[:253, :] = np.r_[1., np.cumprod(1.+r)][:, None]
        panel['close'].iloc[:, :] = 1.; panel['volume'].iloc[:, :] = 1.
        panel['close'].iloc[1:253, :] = np.tile([2., 1.], 126)[:, None]
        got, _ = f.prepare(panel, events[:1], ETFS)
        self.assertTrue(got['impact_available'].all())
        np.testing.assert_array_equal(got['raw_dollar'], np.zeros((1, 49)))
        np.testing.assert_array_equal(got['raw_price'], np.full((1, 49), 1/3))
        for key in ('rank_dollar', 'rank_price'):
            np.testing.assert_array_equal(got[key], np.full((1, 49), 25/49-.5))
        self.assertEqual(f._directional_pair(np.tile([.25, -.25], 126), np.ones(252)), (0., 0.))

    def test_constant_dollar_volume_matches_price_control_exactly(self):
        panel, events = fixture()
        close = np.exp2(np.arange(len(panel['adj'])) % 20)
        panel['close'].iloc[:, :] = close[:, None]
        panel['volume'].iloc[:, :] = (1./close)[:, None]
        got, _ = f.prepare(panel, events, ETFS)
        self.assertEqual(got['raw_dollar'].tobytes(), got['raw_price'].tobytes())
        self.assertEqual(got['rank_dollar'].tobytes(), got['rank_price'].tobytes())

    def test_missing_or_nonpositive_path_has_common_mask_and_zero_counts(self):
        panel, events = fixture()
        for j, value in enumerate((np.nan, 0., -1., np.inf, -np.inf)):
            panel['adj'].iloc[100, j] = value
            panel['close'].iloc[100, j+5] = value
            panel['volume'].iloc[100, j+10] = value
        got, audit = f.prepare(panel, events[:1], ETFS)
        self.assertFalse(got['impact_available'][0, :15].any()); self.assertTrue(got['impact_available'][0, 15:].all())
        for key in f.FLOAT_KEYS:
            np.testing.assert_array_equal(np.isfinite(got[key]), got['impact_available'])
            self.assertTrue(np.isnan(got[key][0, :15]).all())
        for key in ('n_positive', 'n_negative'):np.testing.assert_array_equal(got[key][0, :15], np.zeros(15, np.int64))
        for raw, rank in [('raw_dollar', 'rank_dollar'), ('raw_price', 'rank_price')]:
            expected = pd.Series(got[raw][0, 15:]).rank(pct=True)-.5
            np.testing.assert_array_equal(got[rank][0, 15:], expected.to_numpy())
        self.assertEqual(audit['invalid_price_volume_cells'], 15)

    def test_exact253_252_endpoints_and_first_complete_history(self):
        panel, _ = fixture(); days = panel['adj'].index
        events = [(days[293], SYMBOLS), (days[294], SYMBOLS)]
        got, audit = f.prepare(panel, events, ETFS)
        self.assertFalse(got['impact_available'][0].any()); self.assertTrue(got['impact_available'][1].all())
        self.assertFalse(got['n_positive'][0].any()); self.assertFalse(got['n_negative'][0].any())
        row = audit['months'][1]
        for key, index in [('first_price_date', 0), ('last_price_date', 252),
                           ('first_return_date', 1), ('last_return_date', 252)]:
            self.assertEqual(row[key], days[index].strftime('%Y-%m-%d'))
        self.assertEqual(audit['insufficient_history_cells'], 49)
        panel['close'].iloc[0, :] = np.nan; panel['volume'].iloc[0, :] = np.nan
        same, _ = f.prepare(panel, events[1:], ETFS)
        for key in got:np.testing.assert_array_equal(same[key], got[key][1:])
        for col, (key, index) in enumerate([('adj', 0), ('adj', 252), ('close', 1),
                                            ('close', 252), ('volume', 1), ('volume', 252)]):
            panel[key].iloc[index, col] = np.nan
        bad, audit = f.prepare(panel, events[1:], ETFS)
        self.assertFalse(bad['impact_available'][0, :6].any())
        self.assertEqual(audit['invalid_price_volume_cells'], 6)

    def test_latest42_future_unused_columns_and_raw_first_row_never_converted(self):
        class Poison:
            def __float__(self):raise AssertionError('unused value converted')
            def __array__(self, *args, **kwargs):raise AssertionError('unused value inspected')
        panel, events = fixture(); events = events[:1]
        before, audit_before = f.prepare(panel, events, ETFS)
        poisoned = {key: frame.astype(object) for key, frame in panel.items()}
        for key, frame in poisoned.items():
            frame['UNUSED'] = [Poison() for _ in range(len(frame))]
            for row in range(253, len(frame)):
                for col in range(49):frame.iat[row, col] = Poison()
            if key != 'adj':
                for col in range(49):frame.iat[0, col] = Poison()
            self.assertIsInstance(frame.iat[253, 0], Poison)
        after, audit_after = f.prepare(poisoned, events, ETFS)
        for key in before:np.testing.assert_array_equal(after[key], before[key])
        self.assertEqual(audit_before, audit_after)
        truncated = {key: frame.iloc[:295] for key, frame in poisoned.items()}
        after, _ = f.prepare(truncated, events, ETFS)
        for key in before:np.testing.assert_array_equal(after[key], before[key])

        # Earlier unused history must also be sliced away for a later event.
        panel, events = fixture(); events = events[1:]
        before, _ = f.prepare(panel, events, ETFS)
        for key, frame in panel.items():
            frame = frame.astype(object)
            for row in range(26 if key == 'adj' else 27):
                for col in range(49):frame.iat[row, col] = Poison()
            panel[key] = frame
        after, _ = f.prepare(panel, events, ETFS)
        for key in before:np.testing.assert_array_equal(after[key], before[key])

    def test_insufficient_history_reads_no_numeric_values(self):
        panel, _ = fixture(); days = panel['adj'].index
        for key in panel:panel[key] = panel[key].astype(object).map(lambda _: 'never numeric')
        got, audit = f.prepare(panel, [(days[293], SYMBOLS)], ETFS)
        self.assertFalse(got['impact_available'].any())
        self.assertEqual(audit['insufficient_history_cells'], 49)

    def test_unit_scaling_and_reciprocal_time_varying_conversion(self):
        panel, events = fixture(); before, _ = f.prepare(panel, events, ETFS)
        scaled = {key: frame*np.exp2(np.linspace(-400, 400, 49).astype(int))
                  for key, frame in panel.items()}
        after, _ = f.prepare(scaled, events, ETFS)
        for key in before:np.testing.assert_array_equal(after[key], before[key])
        scaled = {key: frame.copy() for key, frame in panel.items()}
        units = np.exp2((np.arange(len(panel['adj'])) % 21)-10)
        scaled['close'] = scaled['close'].mul(units, axis=0)
        scaled['volume'] = scaled['volume'].div(units, axis=0)
        after, _ = f.prepare(scaled, events, ETFS)
        for key in before:np.testing.assert_array_equal(after[key], before[key])

    def test_large_and_small_products_use_common_scale_without_product_overflow(self):
        panel, events = fixture(); before, _ = f.prepare(panel, events, ETFS)
        for exponent in (900, -900):
            altered = {key: frame.copy() for key, frame in panel.items()}
            altered['close'] *= math.ldexp(1., exponent)
            altered['volume'] *= math.ldexp(1., exponent)
            after, _ = f.prepare(altered, events, ETFS)
            for key in before:np.testing.assert_array_equal(after[key], before[key])
        np.testing.assert_array_equal(f._scaled_dollars(np.full(252, 1e308), np.full(252, 1e308)),
                                      np.full(252, f._scaled_dollars(np.full(252, 1e308), np.full(252, 1e308))[0]))

    def test_arithmetic_overflow_and_scaling_loss_raise_instead_of_masking(self):
        panel, events = fixture(); panel['adj'].iloc[99, 0] = np.finfo(float).tiny
        panel['adj'].iloc[100, 0] = np.finfo(float).max
        with self.assertRaises(FloatingPointError):f.prepare(panel, events[:1], ETFS)
        for data in [np.r_[1e-308, np.full(251, 1e308)], np.r_[1e308, np.full(251, 1e-308)]]:
            with self.assertRaises(FloatingPointError):f._scaled_dollars(data, data)
        r = np.r_[np.full(126, 1e308), np.full(126, -1e-308)]
        with self.assertRaises(FloatingPointError):f._directional_pair(r, np.ones(252))
        r = np.tile([1., -np.finfo(float).tiny], 126)
        d = np.tile([np.finfo(float).tiny, 1.], 126)
        with self.assertRaises(FloatingPointError):f._directional_pair(r, d)

    def test_undefined_denominator_is_shared_and_other_errors_propagate(self):
        self.assertTrue(np.isnan(f._asymmetry(0., 0.)))
        self.assertTrue(all(np.isnan(x) for x in f._directional_pair(np.zeros(252), np.ones(252))))
        panel, events = fixture()
        with patch.object(f, '_directional_pair', return_value=(np.nan, np.nan)):
            got, audit = f.prepare(panel, events[:1], ETFS)
        self.assertFalse(got['impact_available'].any())
        self.assertEqual(audit['undefined_denominator_cells'], 49)
        self.assertTrue((got['n_positive'] > 0).all())
        with patch.object(f, '_directional_pair', side_effect=FloatingPointError('explicit computation failure')):
            with self.assertRaises(FloatingPointError):f.prepare(panel, events[:1], ETFS)

    def test_full49_includes_etfs_and_never_reads_eligibility_or_open(self):
        panel, events = fixture(); before, _ = f.prepare(panel, events, ETFS)
        class RequiredOnly(dict):
            def __getitem__(self, key):
                if key not in ('adj', 'close', 'volume'):raise AssertionError('irrelevant panel input read')
                return super().__getitem__(key)
        after, audit = f.prepare(RequiredOnly(**panel, eligible=object(), open=object()), events, ETFS)
        for key in before:np.testing.assert_array_equal(after[key], before[key])
        for key in ('rank_dollar', 'rank_price'):
            np.testing.assert_array_equal(np.sort(after[key][0]), np.arange(1, 50)/49-.5)
        self.assertFalse(audit['eligibility_read'])

    def test_candidate_permutation_prefix_and_source_unchanged(self):
        panel, events = fixture(); saved, saved_events = copy.deepcopy(panel), copy.deepcopy(events)
        etfs = ETFS.copy(); before, _ = f.prepare(panel, events, etfs)
        permutation = np.random.default_rng(9142).permutation(49)
        after, _ = f.prepare(panel, [(day, [symbols[j] for j in permutation]) for day, symbols in events], etfs)
        for key in before:np.testing.assert_array_equal(after[key], before[key][:, permutation])
        for n in range(3):
            prefix, _ = f.prepare(panel, events[:n], etfs)
            for key in before:np.testing.assert_array_equal(prefix[key], before[key][:n])
        for key in panel:pd.testing.assert_frame_equal(panel[key], saved[key])
        self.assertEqual(events, saved_events); self.assertEqual(etfs, ETFS)

    def test_duplicate_missing_and_wrong_universe_ids_fail(self):
        panel, events = fixture(); day = events[0][0]
        variants = [SYMBOLS[:-1], SYMBOLS[:-1]+[SYMBOLS[0]], SYMBOLS[:-1]+['OTHER'],
                    [' MISSING']+SYMBOLS[1:], [None]+SYMBOLS[1:]]
        for symbols in variants:
            with self.subTest(symbols=symbols[:1]):
                with self.assertRaises(ValueError):f.prepare(panel, [(day, symbols)], ETFS)
        for etfs in [ETFS[:-1], ETFS[:-1]+[ETFS[0]], 'ETF00']:
            with self.assertRaises(ValueError):f.prepare(panel, events, etfs)
        bad = {key: frame.copy() for key, frame in panel.items()}
        for key in bad:bad[key].columns = [SYMBOLS[1]]+SYMBOLS[1:]
        with self.assertRaises(ValueError):f.prepare(bad, events, ETFS)

    def test_misaligned_axes_and_invalid_daily_sessions_fail(self):
        panel, events = fixture()
        for key in ('close', 'volume'):
            for axis in ('index', 'columns'):
                bad = {k: v.copy() for k, v in panel.items()}
                if axis == 'index':bad[key] = bad[key].iloc[::-1]
                else:bad[key] = bad[key].iloc[:, ::-1]
                with self.assertRaises(ValueError):f.prepare(bad, events, ETFS)
        for mode in ('duplicate', 'reverse', 'timezone', 'intraday', 'nat', 'not_datetime'):
            bad = {k: v.copy() for k, v in panel.items()}
            index = panel['adj'].index
            if mode == 'duplicate':index = pd.DatetimeIndex([index[1]]+list(index[1:]))
            elif mode == 'reverse':index = index[::-1]
            elif mode == 'timezone':index = index.tz_localize('UTC')
            elif mode == 'intraday':index = index+pd.Timedelta(hours=1)
            elif mode == 'nat':index = pd.DatetimeIndex([pd.NaT]+list(index[1:]))
            else:index = pd.RangeIndex(len(index))
            for frame in bad.values():frame.index = index
            with self.subTest(mode=mode):
                with self.assertRaises(ValueError):f.prepare(bad, events, ETFS)

    def test_invalid_events_and_required_shapes_fail(self):
        panel, events = fixture(); day = events[0][0]
        for invalid in [None, ['bad'], [(day,)], [events[0], events[0]], events[::-1],
                        [(day+pd.Timedelta(hours=1), SYMBOLS)], [(pd.NaT, SYMBOLS)],
                        [(day.tz_localize('UTC'), SYMBOLS)], [('1900-01-01', SYMBOLS)]]:
            with self.assertRaises(ValueError):f.prepare(panel, invalid, ETFS)
        with self.assertRaises(ValueError):f.prepare({'adj': panel['adj']}, events, ETFS)
        with self.assertRaises(ValueError):f._window_feature(np.ones((252, 2)), np.ones((252, 2)), np.ones((252, 2)))
        with self.assertRaises(ValueError):f._scaled_dollars(np.ones(251), np.ones(251))
        with self.assertRaises(ValueError):f._scaled_dollars(np.zeros(252), np.ones(252))
        with self.assertRaises(ValueError):f._directional_pair(np.ones(251), np.ones(251))
        with self.assertRaises(FloatingPointError):f._directional_pair(np.full(252, np.inf), np.ones(252))
        with self.assertRaises(FloatingPointError):f._asymmetry(-1., 1.)


if __name__ == '__main__':
    unittest.main()
