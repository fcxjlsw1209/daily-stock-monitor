import unittest,sqlite3
import numpy as np
import pandas as pd
from validate_intraday import P,DB,read_bars,snapshot,choose
class IntradayValidationChecks(unittest.TestCase):
    def test_no_post1540_in_snapshot(self):
        h=read_bars(P/'raw'/'AAPL_5m.csv.gz');a=snapshot(h)
        altered=h.copy();later=altered.index.strftime('%H:%M')>='15:40'
        altered.loc[later,['Open','High','Low','Close','Volume']]*=100
        pd.testing.assert_frame_equal(a,snapshot(altered))
        # Dropping every post-cutoff bar must also leave each signal unchanged.
        pd.testing.assert_frame_equal(a,snapshot(h.loc[~later]))
    def test_missing_bars_not_filled(self):
        h=read_bars(P/'raw'/'AAPL_5m.csv.gz');day=str(h.index[0].date())
        remove=(h.index.strftime('%Y-%m-%d')==day)&(h.index.strftime('%H:%M')=='15:35')
        self.assertNotIn(day,snapshot(h.loc[~remove]).index)
    def test_selection_replay_and_caps(self):
        f=pd.read_csv(P/'signal_features.csv.gz');p=pd.read_csv(P/'daily_selection.csv').fillna({'selected':''})
        for r in p.itertuples():
            day=f[f.date==r.date];s=choose(day,r.strategy)
            self.assertEqual(','.join(s),r.selected);self.assertIn(len(s),[0,3,4,5])
            if s:self.assertLessEqual(day.set_index('symbol').loc[s].sector.value_counts().max(),2)
        self.assertTrue(f.last_signal_bar.str.contains('15:35:00').all())
    def test_execution_pnl_and_day_aggregation(self):
        t=pd.read_csv(P/'trades.csv');d=pd.read_csv(P/'daily_portfolios.csv')
        with sqlite3.connect(f'file:{DB}?mode=ro',uri=True) as db:
            for r in t.sample(min(80,len(t)),random_state=11).itertuples():
                entry=db.execute('SELECT close FROM daily_prices WHERE symbol=? AND trade_date=?',(r.symbol,r.entry_date)).fetchone()[0]
                self.assertAlmostEqual(r.entry,entry)
                if r.exit_mode=='open':sell=db.execute('SELECT open FROM daily_prices WHERE symbol=? AND trade_date=?',(r.symbol,r.exit_date)).fetchone()[0]
                else:
                    h=read_bars(P/'raw'/f'{r.symbol}_5m.csv.gz')
                    sell=h[(h.index.strftime('%Y-%m-%d')==r.exit_date)&(h.index.strftime('%H:%M')==r.exit_mode)].Open.iloc[0]
                self.assertAlmostEqual(r.exit,sell)
                self.assertAlmostEqual(r.gross,(sell+r.dividend)/entry-1)
                self.assertLess(r.entry_date,r.exit_date)
        for (strategy,mode,cost),g in d.groupby(['strategy','exit_mode','cost_bps']):
            tt=t[(t.strategy==strategy)&(t.exit_mode==mode)]
            calc=tt.assign(net=lambda x:x.weight*(x.gross-cost/1e4)).groupby('entry_date').net.sum()
            np.testing.assert_allclose(g.net,calc.reindex(g.entry_date).fillna(0),atol=1e-12)
if __name__=='__main__':unittest.main()
