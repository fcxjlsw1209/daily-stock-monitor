import unittest
import numpy as np
import pandas as pd
from backtest import Spec, features, ma, prior_week_rsi, rsi, simulate, summarize


def fixture(n=400):
    rng=np.random.default_rng(31)
    c=100*np.exp(np.cumsum(rng.normal(0.0005,0.02,n)))
    return pd.DataFrame(dict(Open=c*.998,High=c*1.02,Low=c*.98,Close=c,Dividends=np.zeros(n)),
                        index=pd.bdate_range('2010-01-01',periods=n))


class Tests(unittest.TestCase):
    def test_signal_prefixes(self):
        d=fixture()
        for spec in [Spec('default'),Spec('weekly',secondary='weekly'),
                     Spec('hma',inner='HMA',fast=5,slow=8,outer='RMA',smooth=8,gap=4)]:
            full=features(d,spec)
            for n in [251,252,253,254,255,301,350]:
                pd.testing.assert_frame_equal(full.iloc[:n],features(d.iloc[:n],spec))

    def test_week_does_not_use_current_week(self):
        d=fixture()
        c=d.Close.copy()
        week=c.index.to_period('W-FRI')
        selected=week[-3]
        base=prior_week_rsi(c)
        c.loc[week==selected]*=10
        pd.testing.assert_series_equal(base.loc[week<=selected],prior_week_rsi(c).loc[week<=selected])

    def test_known_averages(self):
        s=pd.Series([1.,2.,3.,4.,5.])
        self.assertAlmostEqual(ma(s,3,'WMA').iloc[2],14/6)
        self.assertAlmostEqual(ma(s,3,'RMA').iloc[2],2.)
        self.assertAlmostEqual(ma(s,3,'RMA').iloc[3],8/3)
        self.assertEqual(rsi(pd.Series(np.arange(30,dtype=float))).iloc[-1],100.)

    def test_next_open_and_dividends_and_fees(self):
        d=fixture(5)
        d[['Open','High','Low','Close']]=[100.,110.,90.,100.]
        d.loc[d.index[2],'Open']=110.
        d.loc[d.index[1],'Dividends']=1. # New entry does NOT receive.
        d.loc[d.index[2],'Dividends']=2. # Ex-date seller receives.
        f=pd.DataFrame(dict(buy=[True,False,False,False,False],sell=[False,True,False,False,False]),index=d.index)
        eq,t,o,u=simulate(d,f,str(d.index[1].date()),5)
        q=100000/(100*1.0005)
        expected=q*(110*.9995+2)
        self.assertAlmostEqual(eq.equity.iloc[-1],expected)
        self.assertEqual(o[0]['date'],str(d.index[1].date()))
        self.assertEqual(o[1]['date'],str(d.index[2].date()))
        self.assertAlmostEqual(t[0]['dividends'],2*q)
        self.assertAlmostEqual(t[0]['pnl'],expected-100000)

    def test_future_price_cannot_change_past_orders(self):
        d=fixture()
        f=features(d,Spec('x'))
        _,_,o,_=simulate(d,f,'2011-01-01')
        d2=d.copy()
        d2.iloc[-20:,d2.columns.get_indexer(['Open','High','Low','Close'])]*=2
        _,_,p,_=simulate(d2,features(d2,Spec('x')),'2011-01-01')
        cut=str(d.index[-20].date())
        self.assertEqual([x for x in o if x['date']<cut],[x for x in p if x['date']<cut])

    def test_buyhold_and_open_reconciliation(self):
        d=fixture()
        f=features(d,Spec('x'))
        eq,t,o,u=simulate(d,f,'2011-01-01',hold=True)
        first=d.loc['2011-01-01':].iloc[0]
        q=100000/(first.Open*1.0005)
        self.assertAlmostEqual(eq.equity.iloc[-1],q*d.Close.iloc[-1])
        self.assertEqual(len(t),0)
        self.assertEqual(len(o),1)
        self.assertAlmostEqual(u['open_pnl'],eq.equity.iloc[-1]-100000)

    def test_fee_sensitivity(self):
        d=fixture()
        f=features(d,Spec('x'))
        a,*_=simulate(d,f,'2011-01-01',5)
        b,*_=simulate(d,f,'2011-01-01',10)
        self.assertLessEqual(b.equity.iloc[-1],a.equity.iloc[-1])

    def test_drawdown_includes_initial_capital(self):
        d=pd.DataFrame(dict(equity=[90.,95.],exposure=[1.,1.],**{'return':[-.1,95/90-1]}),index=pd.date_range('2020-01-01',periods=2))
        self.assertAlmostEqual(summarize(d,[],'2020-01-01','2020-01-02')['max_drawdown'],-.1)


if __name__=='__main__':
    unittest.main()
