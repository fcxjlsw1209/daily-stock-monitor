import unittest,sqlite3
import numpy as np
import pandas as pd
from daily_component_backtest import make_signals,ROOT,DATA
from backtest import DB
class DailyComponentChecks(unittest.TestCase):
    def test_future_truncation(self):
        with sqlite3.connect(f'file:{DB}?mode=ro',uri=True) as db:x=pd.read_sql_query('SELECT * FROM daily_prices',db)
        spy=pd.read_csv(ROOT/'spy_source.csv',index_col=0,parse_dates=True);x.trade_date=pd.to_datetime(x.trade_date)
        a=[x.pivot(index='trade_date',columns='symbol',values=k).reindex(spy.index) for k in ['close','open','dividends','volume']]
        full=make_signals(*a,spy)[0];n=2800
        truncated=make_signals(*[q.iloc[:n] for q in a],spy.iloc[:n])[0]
        for key in full:pd.testing.assert_frame_equal(full[key].iloc[:n],truncated[key])
    def test_trade_replay(self):
        p=DATA/'daily_component';t=pd.read_csv(p/'trades.csv.gz');d=pd.read_csv(p/'daily_portfolios.csv')
        for name in ['daily_pullback','daily_market_linked']:
            tt=t[t.strategy==name];dd=d[d.strategy==name]
            calc=tt.assign(pnl=lambda z:z.weight*z.net).groupby('entry_date').pnl.sum()
            np.testing.assert_allclose(dd.net,calc.reindex(dd.entry_date).fillna(0),atol=1e-12)
if __name__=='__main__':unittest.main()
