"""Synthetic fixtures only. No saved forecasts, prices or real returns read."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

import local_year170_independent as audit


def producer():
    path = Path(__file__).with_name('local_year170_model.py')
    spec = importlib.util.spec_from_file_location('year170_synthetic_producer', path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def fixture():
    month = pd.date_range('2015-01-31', '2021-12-31', freq='ME')
    dates = np.repeat(month.values.astype('datetime64[D]'), 3)
    ends = np.repeat((month + pd.offsets.MonthEnd(1)).values.astype('datetime64[D]'), 3)
    rng = np.random.default_rng(170071)
    X = rng.uniform(-.5, .5, (len(dates), 10))
    y = .15 * X[:, 0] + .3 * X[:, 4] ** 2 + np.tile([.07, -.08, .01], len(month))
    y = (y.reshape(-1, 3) - y.reshape(-1, 3).mean(axis=1)[:, None]).ravel()
    event = np.array(['2016-06-30', '2019-01-31', '2021-02-26'], dtype='datetime64[D]')
    args = [dates, ends, np.tile(['A', 'B', 'C'], len(month)), X,
            {k: (i + 1.) * y for i, k in enumerate(audit.TARGETS)}, np.ones(len(dates), bool),
            np.repeat(event, 3), np.tile(['A', 'B', 'Z'], len(event)),
            rng.uniform(-.5, .5, (len(event) * 3, 10)), np.ones(len(event) * 3, bool)]
    return args


def packed(args, forecast):
    train = dict(dates=args[0], ends=args[1], train_plain=args[3], common_good=args[5],
                 event_dates=args[6], event_plain=args[8], **{'target_' + k: v for k, v in args[4].items()})
    meta = dict(monthly=dict(dates=args[6][::3].astype(str).tolist(), symbols=args[7].reshape(-1, 3).tolist()),
                training=dict(dates=args[0][::3].astype(str).tolist(), ends=args[1][::3].astype(str).tolist(),
                              symbols=args[2].reshape(-1, 3).tolist()))
    a = dict(eligible=args[9].reshape(-1, 3), model_available=forecast['available'].reshape(-1, 3),
             local_available=forecast['local_available'].reshape(-1, 3),
             raw_source_log=forecast['raw']['source_log'].reshape(-1, 3),
             raw_local_linear=forecast['raw']['local_linear'].reshape(-1, 3))
    return train, meta, a


def rank_fixture():
    source = np.tile([.1, .2, .3, .4], (4, 1))
    a = dict(eligible=np.ones((4, 4), bool), forecast_available=np.ones((4, 4), bool),
             rank_log_excess=np.tile([-.25, 0., .25, .5], (4, 1)),
             raw_source_log=source, raw_local_linear=source.copy(), raw_local_constant=source.copy(),
             model_available=np.ones((4, 4), bool), state_weak_narrow=np.array([True, True, False, False]),
             state_calm10=np.array([False, False, True, False]), extension_available=np.ones((4, 4), bool))
    changed = np.tile([.1, .4, .3, .2], (4, 1))
    fc = dict(available=a['model_available'].ravel(),
              raw=dict(source_log=source.ravel(), local_linear=source.ravel(),
                       local_year_stability=changed.ravel(), local_half=changed.ravel()))
    return a, fc


class IndependentPreparationTests(unittest.TestCase):
    def test_scalar_weighted_agreement_and_zero_priority(self):
        result, signs = audit.shrink(2., [8., 1., 1.], [1., -1., 0.])
        self.assertEqual(signs, [True, False, False])
        self.assertAlmostEqual(result['p'], .8)
        self.assertAlmostEqual(result['shrink'], .6)
        self.assertEqual(audit.shrink(1., [4., 0.], [1., 1.])[0]['shrink'], 0.)
        self.assertEqual(audit.shrink(0., [0.], [-1.])[0]['shrink'], 1.)
        self.assertIsNone(audit.shrink(1., [0.], [-1.])[0]['p'])

    def test_local_zero_design_closed_form_and_no_fold_minimum(self):
        residual = np.array([2., -1., 3.]); weights = np.array([.2, .7, .3])
        _, correction, info = audit.local_fit(np.zeros((3, 10)), np.zeros(10), np.ones(10), residual, weights)
        self.assertAlmostEqual(correction, np.dot(residual, weights) / (12 + sum(weights)), places=15)
        self.assertEqual(info['status'], 'fit')
        _, wrong, _ = audit.local_fit(np.zeros((3, 10)), np.zeros(10), np.ones(10), residual, weights / sum(weights))
        self.assertNotAlmostEqual(correction, wrong)
        self.assertEqual(audit.local_fit(np.empty((0, 10)), np.zeros(10), np.ones(10), np.array([]), np.array([]))[2]['status'], 'zero_weight')

    def test_independent_full_numerical_audit_matches_synthetic_producer(self):
        args = fixture(); before = copy.deepcopy(args)
        actual = producer().annual_forecasts(*args)
        expected = audit.reconstruct(*args)
        for key in audit.RAW_KEYS:
            audit.close(actual['raw'][key], expected['raw'][key], key)
        audit.same(actual['available'], expected['available'], 'mask')
        audit.same(actual['local_available'], expected['local_available'], 'own mask')
        audit.record_check(actual['audit'], expected['audit'], 'audit')
        receipt = audit.check_prepare(*packed(args, actual), actual)
        self.assertTrue(receipt['passed']); json.dumps(receipt, allow_nan=False)
        for actual_value, original in zip(args, before):
            if isinstance(original, dict):
                for key in original:
                    np.testing.assert_array_equal(actual_value[key], original[key])
            else:
                np.testing.assert_array_equal(actual_value, original)

    def test_prepare_rejects_tampered_forecast_audit_mask_and_dates(self):
        args = fixture(); actual = producer().annual_forecasts(*args)
        for mode in ('raw', 'fold', 'weight', 'availability', 'date'):
            bad = copy.deepcopy(actual); bundle = packed(args, actual)
            if mode == 'raw': bad['raw']['local_half'][3] += .01
            if mode == 'fold': bad['audit']['local'][0]['folds'][0]['correction'] += .01
            if mode == 'weight': bad['audit']['local'][0]['own_weights'][0] *= 2
            if mode == 'availability': bad['local_available'][3] = False
            if mode == 'date': bundle[1]['training']['dates'][0] = '2014-01-31'
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                audit.check_prepare(*bundle, bad)

    def test_pooled_cutoff_future_poison_and_no_history_source_copy(self):
        args = fixture(); result = audit.reconstruct(*args)
        future = args[1] > np.datetime64('2018-12-31')
        args[3][future] = np.inf
        for value in args[4].values(): value[future] = np.nan
        changed = audit.reconstruct(*args)
        for key in audit.RAW_KEYS:
            np.testing.assert_array_equal(result['raw'][key][:6], changed['raw'][key][:6])
        self.assertFalse(result['available'][:3].any())
        self.assertTrue(result['available'][5]); self.assertFalse(result['local_available'][5])
        for key in audit.RAW_KEYS:
            self.assertEqual(result['raw'][key][5], result['raw']['source_log'][5])

    def test_own18_folds_less18_not_dropped(self):
        args = fixture()
        own = np.flatnonzero((args[2] == 'A') & (args[1] <= np.datetime64('2018-12-31')))
        args[5][own[:-18]] = False
        result = audit.reconstruct(*args)
        detail = next(d for d in result['audit']['local'] if d['event_row'] == 3)
        self.assertEqual(detail['own_months'], 18)
        self.assertTrue(result['local_available'][3])
        self.assertTrue(all(f['retained_rows'] < 18 and f['status'] == 'fit' for f in detail['folds']))
        args[5][own[-18]] = False
        self.assertFalse(audit.reconstruct(*args)['local_available'][3])

    def test_zero_target_exact_source_and_zero_full_priority(self):
        args = fixture()
        for target in args[4].values(): target[:] = 0.
        result = audit.reconstruct(*args)
        self.assertTrue(all(d['shrink'] == 1. for d in result['audit']['local']))
        for key in audit.RAW_KEYS:
            np.testing.assert_array_equal(result['raw'][key], result['raw']['source_log'])


class IndependentRankTests(unittest.TestCase):
    def test_full_u_ranks_include_etf_and_initial_delta_neutral(self):
        a, fc = rank_fixture(); ranks, delta, ready = audit.rank_oracle(a, fc)
        np.testing.assert_array_equal(ranks['local_year_stability'][1], [-.25, .5, .25, 0.])
        self.assertAlmostEqual(delta['local_year_stability'][1, 3], -.00625)
        self.assertAlmostEqual(delta['local_year_stability'][2, 3], -.00125)
        self.assertTrue(np.all(delta['local_year_stability'][0] == 0))
        self.assertTrue(np.all(delta['local_year_stability'][3] == 0))
        self.assertTrue(ready['local_year_stability'].all())
        self.assertTrue(audit.verify_rank_slot(a, fc, ranks, delta)['passed'])
        wrong = copy.deepcopy(delta); wrong['local_year_stability'][1, 3] = 0.
        with self.assertRaises(ValueError): audit.verify_rank_slot(a, fc, ranks, wrong)

    def test_ties_source_fallback_ineligible_and_unready(self):
        np.testing.assert_allclose(audit.centered_rank([2., 1., 2.]), [1/3, -1/6, 1/3], rtol=0., atol=1e-15)
        a, fc = rank_fixture(); a['eligible'][1, 3] = False
        for key in fc['raw']: fc['raw'][key] = a['raw_source_log'].ravel().copy()
        ranks, delta, _ = audit.rank_oracle(a, fc)
        for key in ranks:
            np.testing.assert_array_equal(ranks[key], a['rank_log_excess'])
            self.assertFalse(np.any(delta[key]))
        a['model_available'][2, 1] = False
        fc['available'] = a['model_available'].ravel()
        _, delta, ready = audit.rank_oracle(a, fc)
        self.assertFalse(ready['local_year_stability'][2]); self.assertFalse(np.any(delta['local_year_stability'][2]))

    def test_wrong_subset_and_original_readiness_change_rejected(self):
        a, fc = rank_fixture(); ranks, delta, _ = audit.rank_oracle(a, fc)
        wrong = copy.deepcopy(ranks); wrong['local_half_control'][1] = [0., .5, .25, -.25]
        with self.assertRaises(ValueError): audit.verify_rank_slot(a, fc, wrong, delta)
        fc['available'] = fc['available'].copy(); fc['available'][4] = False
        with self.assertRaisesRegex(ValueError, 'readiness'): audit.rank_oracle(a, fc)


class IndependentOutcomeTests(unittest.TestCase):
    def test_pinned_accounting_synthetic_price_and_fee_audit(self):
        helper = audit.accounting()
        for _, mode, cost in audit.PATHS:
            args = helper.Tests().fixture(mode, cost)
            result = helper.audit_path(*args)
            self.assertEqual(result['trade_rows'], 2)
            self.assertAlmostEqual(result['fees_paid'], 100000 / (1 + cost) * cost)
            args[4][0]['adjusted_price'] *= 1.01
            with self.assertRaises(ValueError): helper.audit_path(*args)

    def test_three_arm_five_slice_wrapper_on_generated_calendar(self):
        helper = audit.accounting()
        sessions = pd.bdate_range('2016-08-17', '2026-09-24')
        monthends = sessions.to_series().groupby(sessions.to_period('M')).last().tolist()[:-1]
        dates = ['2016-08-17'] + [str(x.date()) for x in monthends]
        meta = dict(dates=dates, ends=dates[1:] + ['2026-09-24'])
        self.assertEqual(len(dates), 122)
        px = pd.DataFrame({'A': 10 + np.arange(len(sessions)) / 1000., 'B': 20 + np.arange(len(sessions)) / 500.}, index=sessions)
        opening = px * .999
        orders = {arm: [dict(date=d, signal_date=d, end=e, symbols=['A', 'B'], scores=[.5, .4]) for d, e in zip(meta['dates'], meta['ends'])] for arm in audit.ARMS}
        exports, expected = {}, {}
        for arm in audit.ARMS:
            exports[arm], expected[arm] = {}, {}
            for name, mode, cost in audit.PATHS:
                start = sessions[0 if mode == 'close' else 1]
                prices = px if mode == 'close' else opening
                post = 100000 / (1 + cost)
                units = {s: post / 2 / prices.at[start, s] for s in px.columns}
                fills = [dict(date=str(start.date()), symbol=s, side='buy', adjusted_units=units[s], adjusted_price=float(prices.at[start, s]), dollars=post/2, cost=post/2*cost) for s in units]
                daily, positions = [], []
                for day in sessions:
                    active = day >= start
                    cash = 0. if active else 100000.
                    market = {s: units[s] * px.at[day, s] for s in units} if active else {}
                    nav = cash + sum(market.values()); ds = str(day.date())
                    daily.append(dict(date=ds, equity=nav))
                    positions.append(dict(date=ds, cash=cash, equity=nav, adjusted_units=units if active else {}, market_values=market))
                periods = [dict(execution_date=(o['date'] if mode == 'close' else str(sessions[sessions.get_loc(o['date'])+1].date())), signal_date=o['date'], symbols=o['symbols'], scores=o['scores'], fee=(post*cost if i == 0 else 0.)) for i, o in enumerate(orders[arm])]
                exports[arm][name] = dict(DAILY=daily, TRADES=fills, HOLDINGS=periods, DAILY_POSITIONS=positions)
                days, values = [v['date'] for v in daily], [v['equity'] for v in daily]
                slices = helper.statistics_by_split(days, values, meta)
                annual = []
                for year in sorted({d[:4] for d in days}):
                    ids = [i for i, d in enumerate(days) if d.startswith(year)]
                    prior = values[ids[0]-1] if ids[0] else 100000.
                    annual.append(dict(year=int(year), return_=values[ids[-1]]/prior-1., partial=int(year) in (2016, 2026)))
                met = dict(slices['full122'], fees_paid=post*cost, dollar_turnover=post, trade_rows=2, annual_returns=annual)
                expected[arm][name] = dict(metrics=met, slices=slices, accounting=dict(exported_fill_ledger_matches=True, terminal_liquidation=False))
        result = audit.audit_outcomes(px, opening, orders, exports, expected, meta)
        self.assertTrue(result['passed']); self.assertEqual(result['checked_slices'], 45)
        expected[audit.ARMS[1]]['next_open_cost']['slices']['later34']['cagr'] += .1
        with self.assertRaises(ValueError): audit.audit_outcomes(px, opening, orders, exports, expected, meta)


if __name__ == '__main__':
    unittest.main()
