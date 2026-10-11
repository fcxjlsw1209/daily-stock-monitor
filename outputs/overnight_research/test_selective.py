import unittest,sqlite3
import numpy as np
import pandas as pd
from selective_backtest import selective_signals,ROOT
from backtest import DB
class SelectiveChecks(unittest.TestCase):
    def test_causality_and_sizing(self):
        with sqlite3.connect(f'file:{DB}?mode=ro',uri=True) as db:x=pd.read_sql_query('SELECT * FROM daily_prices',db)
        spy=pd.read_csv(ROOT/'spy_source.csv',index_col=0,parse_dates=True);x.trade_date=pd.to_datetime(x.trade_date)
        arrays=[x.pivot(index='trade_date',columns='symbol',values=k).reindex(spy.index) for k in ['close','open','dividends','volume']]
        models=selective_signals(*arrays,spy)[0]
        prefix=selective_signals(*[a.iloc[:1100] for a in arrays],spy.iloc[:1100])[0]
        for name,w in models.items():
            pd.testing.assert_frame_equal(w.iloc[:1100],prefix[name])
            self.assertTrue((w.sum(axis=1)<=1.0000001).all())
            self.assertTrue(((w>0).sum(axis=1).isin([0,3,4,5])).all())
    def test_original_unchanged(self):
        a=pd.read_csv(ROOT/'metrics.csv');b=pd.read_csv(ROOT/'selective'/'metrics.csv')
        a=a[a.strategy=='strong_pullback_market'].drop(columns='strategy').reset_index(drop=True)
        b=b[b.strategy=='original'].drop(columns='strategy').reset_index(drop=True)
        pd.testing.assert_frame_equal(a,b)
    def test_return_replay(self):
        t=pd.read_csv(ROOT/'selective'/'trades.csv.gz')
        d=pd.read_csv(ROOT/'selective'/'daily_portfolios.csv')
        t=t[t.strategy=='strong_and_deep'];d=d[d.strategy=='strong_and_deep'].set_index('entry_date')
        r=t.assign(pnl=lambda f:f.weight*f.net).groupby('entry_date').pnl.sum().reindex(d.index).fillna(0)
        np.testing.assert_allclose(r,d.net,atol=1e-12)
if __name__=='__main__':unittest.main()
