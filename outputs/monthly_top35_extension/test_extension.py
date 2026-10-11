import unittest,json
from pathlib import Path
import numpy as np
import pandas as pd
from run import snapshot_cap, corrected_published, formula, pick, independent

P=Path(__file__).resolve().parent
class ExtensionTests(unittest.TestCase):
    def test_cap_observation_never_uses_future_snapshot_or_price(self):
        z=pd.Series([100.,1e12],index=pd.to_datetime(['2021-01-15','2021-02-15']))
        p=pd.Series([10.,12.,1e6],index=pd.to_datetime(['2021-01-15','2021-01-29','2021-02-15']))
        a=snapshot_cap(z,pd.Timestamp('2021-01-29'),p)
        b=snapshot_cap(z.iloc[:1],pd.Timestamp('2021-01-29'),p.iloc[:2])
        self.assertEqual(a,b);self.assertAlmostEqual(a['market_cap_usd'],120.)
        self.assertIsNone(snapshot_cap(z,pd.Timestamp('2021-01-01'),p))

    def test_extended_pool_is_every_rank_not_only_author_names(self):
        pools=json.loads((P/'POOLS.json').read_text())
        self.assertEqual(len(pools['32']),69)
        for date,a in pools['32'].items():
            b=pools['35'][date]
            self.assertEqual(a,b[:32]);self.assertEqual(len(set(b)),35)
        events=json.loads((P.parent/'us_listed_top32_yahoo/PUBLISHED_MONTHLY_RANKINGS.json').read_text())['periods']
        for e in events:
            self.assertEqual(pools['35'][e['date']],[r['symbol'] for r in corrected_published(e,35)])

    def test_earlier_snapshots_observed_before_decision(self):
        for e in json.loads((P/'EARLIER_RANKS.json').read_text()):
            self.assertLess(e['cap_asof'],e['date'])
            for r in e['ranks'][:35]:
                self.assertLessEqual(r['observation_date'],e['cap_asof'])
                if 'shares_known_date' in r:self.assertLessEqual(r['shares_known_date'],e['cap_asof'])

    def test_scoring_and_selection_unchanged_by_future_data(self):
        rng=np.random.default_rng(913);dates=pd.bdate_range('2019-01-01',periods=450)
        px=pd.DataFrame(np.exp(np.cumsum(rng.normal(.003,.018,(450,5)),axis=0))*100,index=dates,columns=list('ABCDE'))
        date=dates[350];f=formula(px);truncated=px.loc[:date];ft=formula(truncated)
        a=pick(px,f,date,list('ABCDE'));b=pick(truncated,ft,date,list('ABCDE'))
        self.assertEqual(a,b)
        expected=px.shift(42)/px.shift(252)-1
        pd.testing.assert_frame_equal(f['mom252_skip42'],expected)

    def test_fixed_units_and_cash_period(self):
        dates=pd.date_range('2021-01-01',periods=4)
        px=pd.DataFrame({'A':[100.,200.,100.,300.],'B':[100.,100.,100.,200.]},index=dates)
        orders=[dict(date='2021-01-01',end='2021-01-03',symbols=['A','B']),dict(date='2021-01-03',end='2021-01-04',symbols=[])]
        np.testing.assert_allclose(independent(px,orders),[100000.,150000.,100000.,100000.])

    def test_early_historical_candidates_not_lost_to_current_rank(self):
        records=json.loads((P/'CAP_COLLECTION.json').read_text())['records']
        self.assertTrue({'PYPL','NKE','HON'}.issubset({r['symbol'] for r in records}))
        ranks=json.loads((P/'EARLIER_RANKS.json').read_text())
        self.assertTrue(any('PYPL' in [r['symbol'] for r in e['ranks'][:32]] for e in ranks))

if __name__=='__main__':unittest.main()
