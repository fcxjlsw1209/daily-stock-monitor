import unittest,sqlite3
import numpy as np
import pandas as pd
from backtest import pick,make_signals,P,DB

class BacktestChecks(unittest.TestCase):
    def test_slots_and_minimum_names(self):
        s=pd.DataFrame([[4,3,2,1],[4,3,np.nan,np.nan]])
        w=pick(s,s.notna())
        self.assertAlmostEqual(w.iloc[0].sum(),.8)
        self.assertEqual(w.iloc[1].sum(),0)
    def test_no_future_data_in_signals(self):
        with sqlite3.connect(f'file:{DB}?mode=ro',uri=True) as db:
            x=pd.read_sql_query('SELECT * FROM daily_prices',db)
        spy=pd.read_csv(P/'spy_source.csv',index_col=0,parse_dates=True)
        x.trade_date=pd.to_datetime(x.trade_date)
        arrays=[x.pivot(index='trade_date',columns='symbol',values=k).reindex(spy.index) for k in ['close','open','dividends','volume']]
        full=make_signals(*arrays,spy)[0]
        n=900
        prefix=make_signals(*[a.iloc[:n] for a in arrays],spy.iloc[:n])[0]
        for name in full:
            pd.testing.assert_frame_equal(full[name].iloc[:n],prefix[name])
            # Buy-session weights use the prior day's fully known selection.
            pd.testing.assert_series_equal(full[name].shift(1).iloc[n],full[name].iloc[n-1],check_names=False)
    def test_replay_prices_and_costs(self):
        t=pd.read_csv(P/'trades.csv.gz',parse_dates=['entry_date','exit_date'])
        with sqlite3.connect(f'file:{DB}?mode=ro',uri=True) as db:
            for row in t.sample(100,random_state=23).itertuples():
                close=db.execute('SELECT close FROM daily_prices WHERE symbol=? AND trade_date=?',(row.symbol,str(row.entry_date.date()))).fetchone()[0]
                opening,div=db.execute('SELECT open,dividends FROM daily_prices WHERE symbol=? AND trade_date=?',(row.symbol,str(row.exit_date.date()))).fetchone()
                self.assertAlmostEqual(row.gross,(opening+div)/close-1)
                self.assertAlmostEqual(row.net,row.gross-.001)
                self.assertLess(row.entry_date,row.exit_date)
        # Reconstruct daily full-capital portfolio returns from its fills.
        strategy='strong_pullback_market'
        replay=t[t.strategy==strategy].assign(pnl=lambda f:f.weight*f.net).groupby('entry_date').pnl.sum()
        daily=pd.read_csv(P/'daily_portfolios.csv',parse_dates=['entry_date'])
        daily=daily[daily.strategy==strategy].set_index('entry_date')
        np.testing.assert_allclose(daily.net,replay.reindex(daily.index).fillna(0),atol=1e-12)

if __name__=='__main__':unittest.main()
