import unittest
import numpy as np
import pandas as pd
import strict_study as s


class StrictRelativeTests(unittest.TestCase):
    def prices(self):
        t = np.arange(400)
        return pd.DataFrame({'A': 100 * np.exp(.002 * t), 'XLK': 100 * np.exp(.001 * t),
                             'XLE': 100 * np.exp(.003 * t)}, index=pd.bdate_range('2020-01-01', periods=400))

    def test_compare_own_stock_to_sector_not_other_sector(self):
        px = self.prices()
        mom = px.shift(42) / px.shift(252) - 1
        date = px.index[-1]
        ans = s.relative_row(mom, date, ['A'], {'A': 'XLK'})
        expected = (px.A.iloc[-43] / px.A.iloc[-253]) / (px.XLK.iloc[-43] / px.XLK.iloc[-253]) - 1
        self.assertAlmostEqual(ans.A, expected)
        changed = mom.copy()
        changed.loc[date, 'XLE'] = 999
        pd.testing.assert_series_equal(ans, s.relative_row(changed, date, ['A'], {'A': 'XLK'}))
        changed.loc[date, 'A'] += .1
        self.assertNotEqual(ans.A, s.relative_row(changed, date, ['A'], {'A': 'XLK'}).A)

    def test_latest42_prices_do_not_enter_momentum_or_relative(self):
        px = self.prices()
        changed = px.copy()
        changed.iloc[-42:] *= np.linspace(1., 5., 42)[:, None]
        for n in [126, 252, 294]:
            a = px.shift(42) / px.shift(n) - 1
            b = changed.shift(42) / changed.shift(n) - 1
            pd.testing.assert_series_equal(a.iloc[-1], b.iloc[-1])
            pd.testing.assert_series_equal(s.relative_row(a, px.index[-1], ['A'], {'A': 'XLK'}),
                                           s.relative_row(b, px.index[-1], ['A'], {'A': 'XLK'}))
        for n in [12, 14]:
            # Separate calendar convention: recent two whole calendar months.
            cut = px.index[-1]
            endpoint_month = cut.to_period('M') - 2
            changed = px.copy()
            changed.loc[px.index.to_period('M') > endpoint_month] *= 3
            a = s.e.previous.calendar_momentum(px, n)
            b = s.e.previous.calendar_momentum(changed, n)
            pd.testing.assert_series_equal(a.iloc[-1], b.iloc[-1])

    def test_asof_classification_and_etf_neutral(self):
        dates = pd.to_datetime(['2020-12-31', '2021-01-29'])
        sectors = pd.DataFrame({'A': ['Technology', 'Energy']}, index=dates)
        self.assertEqual(s.benchmark_map(dates[0], ['A', 'XLK', 'unknown'], sectors, {'XLK'}),
                         {'A': 'XLK', 'XLK': None, 'unknown': None})

    def test_same_close_does_not_earn_new_holdings_prior_month_return(self):
        dates = pd.to_datetime(['2020-12-31', '2021-01-29', '2021-02-26'])
        px = pd.DataFrame({'A': [100., 200., 50.], 'B': [100., 80., 120.]}, index=dates)
        orders = [dict(date=str(dates[i].date()), signal_date=str(dates[i].date()), symbols=[name], scores=[1.])
                  for i, name in enumerate(['A', 'B'])]
        met, eq, trades, holds = s.e.engine.replay(px, px, orders, cost=0, execution='close', start=dates[0], end=dates[-1])
        np.testing.assert_allclose(eq.to_numpy(), [100000., 200000., 300000.], rtol=1e-12)
        buy = next(t for t in trades if t['symbol'] == 'B' and t['side'] == 'buy')
        self.assertEqual(buy['date'], '2021-01-29')
        self.assertEqual(buy['adjusted_price'], 80.)

    def test_fixed_shares_between_rebalances(self):
        dates = pd.to_datetime(['2020-12-31', '2021-01-04', '2021-01-29'])
        px = pd.DataFrame({'A': [100., 200., 200.], 'B': [100., 100., 200.]}, index=dates)
        orders = [dict(date='2020-12-31', signal_date='2020-12-31', symbols=['A', 'B'], scores=[1., .9])]
        met, eq, trades, holds = s.e.engine.replay(px, px, orders, cost=0, execution='close', start=dates[0], end=dates[-1])
        np.testing.assert_allclose(eq.to_numpy(), [100000., 150000., 200000.], rtol=1e-12)
        self.assertEqual(len(trades), 2)

    def test_saved_features_have_no_old_industry_strength_or_recent_return(self):
        if not (s.P / 'INPUTS.json').exists():
            self.skipTest('Run prepare first')
        m = s.read(s.P / 'INPUTS.json')
        for norm in ['percentile', 'magnitude']:
            d = s.e.pack(norm, m)
            self.assertFalse(any(k in d['names'] for k in ['industry_relative60', 'industry_strength_recent', 'sector_strength_long', 'anti_recent42', 'ordinary_ridge']))
            for i, (ss, coverage) in enumerate(zip(d['symbols'], d['coverage'])):
                for symbol in coverage['neutral_etfs']:
                    for name in ['stock_sector_skip42_126', 'stock_sector_skip42_252']:
                        self.assertEqual(d['X'][i, ss.index(symbol), d['names'].index(name)], 0.)

    def test_all_fit_budgets_and_later_labels_excluded(self):
        if not (s.P / 'TRIALS.json').exists():
            self.skipTest('Run train first')
        for row in s.read(s.P / 'TRIALS.json'):
            A, b, prior = s.e.limits(row['spec'])
            w = np.array(row['weights'])
            self.assertAlmostEqual(w.sum(), 1, places=7)
            self.assertTrue((np.einsum('ij,j->i', A, w, optimize=False) <= b + 1e-7).all())
        m = s.read(s.P / 'INPUTS.json')
        d = s.e.pack('percentile', m)
        X, spec = s.design(d, m['families'][0])
        dev = np.array(m['development'])
        changed = dict(d, order=d['order'].copy())
        changed['order'][~dev] = -1
        for a, b in zip(s.e.events(X, d, dev), s.e.events(X, changed, dev)):
            np.testing.assert_array_equal(a, b)


if __name__ == '__main__':
    unittest.main()
