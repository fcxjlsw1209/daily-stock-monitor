import unittest
import numpy as np
import pandas as pd
from backtest_dynamic import load,features,simulate,P
class DynamicTests(unittest.TestCase):
    def fixture(self):
        ix=pd.bdate_range('2020-01-01',periods=12)
        h=pd.DataFrame({'Open':100.,'High':101.,'Low':99.,'Close':100.,'Volume':1e7,'Dividends':0.},index=ix)
        f=pd.DataFrame({'rebound':False,'meanrev':False,'breakout':False,'legacy':False,'low5':90.,'atr':1.,'ma10':99.,'ma20':99.},index=ix)
        return h,f,ix
    def test_partial_is_one_episode_and_dividends_follow_shares(self):
        h,f,ix=self.fixture();f.loc[ix[1],'rebound']=True
        h.loc[ix[2],'Close']=101;h.loc[ix[3],'Open']=102;h.loc[ix[3],'Dividends']=1
        f.loc[ix[3],'ma10']=102;h.loc[ix[4],'Open']=103;h.loc[ix[4],'Dividends']=2
        d,t,o,u=simulate(h,f,'rebound','partial',0,start=str(ix[2].date()),end=str(ix[4].date()),initial=1000)
        self.assertEqual(len(t),1);self.assertEqual(len(o),3)
        self.assertEqual(t.net_pnl.iloc[0],45);self.assertEqual(t.dividends.iloc[0],20)
        self.assertTrue(t.partial_taken.iloc[0]);self.assertEqual(d.equity.iloc[-1],1045)
    def test_stop_uses_next_open_not_trigger(self):
        h,f,ix=self.fixture();f.loc[ix[1],'meanrev']=True;f.loc[ix[1],'low5']=95
        h.loc[ix[2],'Close']=90;h.loc[ix[3],'Open']=80
        d,t,o,u=simulate(h,f,'meanrev','meanrev',0,start=str(ix[2].date()),end=str(ix[3].date()),initial=1000)
        self.assertEqual(t.net_return.iloc[0],-.2);self.assertEqual(o.price.iloc[-1],80)
    def test_trailing_stop_never_widens(self):
        h,f,ix=self.fixture();f.loc[ix[1],'breakout']=True
        h.loc[ix[2],'Close']=120;h.loc[ix[3],'Close']=119;f.loc[ix[3],'atr']=20
        d,t,o,u=simulate(h,f,'breakout','trailing',0,start=str(ix[2].date()),end=str(ix[3].date()),initial=1000)
        self.assertEqual(d.stop.iloc[0],117);self.assertEqual(d.stop.iloc[1],117)
        self.assertEqual(len(t),0);self.assertEqual(len(u),1);self.assertEqual(u[0]['sessions'],1)
    def test_causal_features(self):
        data=load()
        for symbol,h in data.items():
            f=features(h,data['SPY']);short=features(h.iloc[:2000],data['SPY'].iloc[:2000])
            pd.testing.assert_frame_equal(f.iloc[:2000],short)
    def test_legacy_reproduces_previous_backtest(self):
        old=pd.read_csv(P.parent/'swing_research'/'etf_extended_equity.csv.gz')
        new=pd.read_csv(P/'daily_equity.csv.gz')
        for symbol in ['QQQ','SPY']:
            for cost in [5,10]:
                a=old[(old.strategy==symbol+'_pullback')&(old.exit_rule=='fixed5')&(old.cost_bps==cost)].set_index('date')
                b=new[(new.symbol==symbol)&(new.strategy=='legacy_fixed5')&(new.cost_bps==cost)].set_index('date')
                np.testing.assert_allclose(a.equity,b.equity,atol=1e-6)
    def test_entire_saved_cash_ledger(self):
        daily=pd.read_csv(P/'daily_equity.csv.gz');orders=pd.read_csv(P/'orders.csv')
        for key,g in orders.groupby(['symbol','strategy','cost_bps']):
            d=daily[(daily.symbol==key[0])&(daily.strategy==key[1])&(daily.cost_bps==key[2])].set_index('date')
            flow=np.where(g.side=='buy',-g.qty*g.price-g.fee,g.qty*g.price-g.fee)
            change=pd.Series(flow,index=g.date).groupby(level=0).sum().reindex(d.index,fill_value=0)
            np.testing.assert_allclose(d.cash,100000+change.cumsum()+d.dividends_cumulative,atol=1e-6)
            np.testing.assert_allclose(d.equity,d.cash+d.invested,atol=1e-6)
            self.assertTrue((d.cash>=-1e-6).all())
if __name__=='__main__':unittest.main()
