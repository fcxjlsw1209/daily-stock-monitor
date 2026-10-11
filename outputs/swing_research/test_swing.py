import unittest
import numpy as np
import pandas as pd
from backtest_swing import Inputs,simulate,signals,load,P

class SwingTests(unittest.TestCase):
    def fixture(self,n=12):
        ix=pd.bdate_range('2020-01-01',periods=n)
        c=pd.DataFrame({'X':100.},index=ix)
        data=Inputs(c,c.copy(),c*0,c*100,{'X':'test'},pd.DataFrame(index=ix))
        return data,pd.DataFrame(True,index=ix,columns=['X']),pd.DataFrame(1.,index=ix,columns=['X'])
    def test_fixed_hold_dividend_and_no_same_day_rebuy(self):
        data,mask,score=self.fixture();dates=data.close.index
        data.dividends.loc[dates[1],'X']=2 # entry day: buyer is not entitled
        data.dividends.loc[dates[2],'X']=1 # existing holder is entitled
        d,t,o,u=simulate(data,mask,score,'fixed3',0,start=str(dates[1].date()),end=str(dates[4].date()),initial=1000)
        self.assertEqual(len(t),1);self.assertEqual(t.sessions.iloc[0],3)
        self.assertEqual(t.dividends.iloc[0],2);self.assertEqual(d.equity.iloc[-1],1002)
        self.assertEqual(len(o),2);self.assertEqual(d.positions.iloc[-1],0)
    def test_ma5_next_open_and_mark_to_market(self):
        data,mask,score=self.fixture();dates=data.close.index
        data.close.loc[dates[5],'X']=103;data.opening.loc[dates[6],'X']=104
        d,t,o,u=simulate(data,mask,score,'ma5_max5',0,start=str(dates[5].date()),end=str(dates[6].date()),etf=True,initial=1000)
        self.assertEqual(t.sessions.iloc[0],1);self.assertEqual(t.exit_price.iloc[0],104)
        self.assertEqual(d.equity.iloc[0],1030);self.assertEqual(d.equity.iloc[1],1040)
    def test_costs_and_terminal_open_position(self):
        data,mask,score=self.fixture();dates=data.close.index
        d,t,o,u=simulate(data,mask,score,'fixed5',10,start=str(dates[1].date()),end=str(dates[2].date()),etf=True,initial=1000)
        self.assertEqual(len(t),0);self.assertEqual(len(u),1)
        self.assertAlmostEqual(d.equity.iloc[-1],1000/1.0005)
        self.assertTrue((d.cash>=-1e-8).all())
    def test_no_future_signal_data(self):
        data=load();full=signals(data);n=2800
        short=Inputs(*[x.iloc[:n] for x in [data.close,data.opening,data.dividends,data.volume]],data.sector,data.sector_prices.iloc[:n])
        limited=signals(short)
        for name,(mask,score) in full.items():
            pd.testing.assert_frame_equal(mask.iloc[:n],limited[name][0])
            pd.testing.assert_frame_equal(score.iloc[:n],limited[name][1])
    def test_saved_cash_ledger(self):
        daily=pd.read_csv(P/'daily_equity.csv.gz');orders=pd.read_csv(P/'orders.csv')
        for key,g in orders.groupby(['strategy','exit_rule','cost_bps']):
            dd=daily[(daily.strategy==key[0])&(daily.exit_rule==key[1])&(daily.cost_bps==key[2])].set_index('date')
            flow=np.where(g.side=='buy',-g.quantity*g.price-g.fee,g.quantity*g.price-g.fee)
            movements=pd.Series(flow,index=g.date).groupby(level=0).sum().reindex(dd.index,fill_value=0)
            expected=100000+movements.cumsum()+dd.dividends_cumulative
            np.testing.assert_allclose(dd.cash,expected,atol=1e-6)
            np.testing.assert_allclose(dd.equity,dd.cash+dd.invested,atol=1e-6)
            self.assertLessEqual(dd.positions.max(),5)
if __name__=='__main__':unittest.main()
