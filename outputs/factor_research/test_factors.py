import unittest
from dataclasses import replace
import numpy as np
import pandas as pd
from backtest_factors import P,WEIGHTS,FACTORS,load_inputs,build,forward_returns

class FactorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data,cls.raw=load_inputs();cls.b=build(cls.data,cls.raw)

    def test_formulas_and_fixed_weights(self):
        a=self.raw['adj_close'];day=a.index[-1];s='AAPL'
        self.assertAlmostEqual(self.b['factors']['long_momentum'].at[day,s],a[s].iloc[-22]/a[s].iloc[-253]-1)
        self.assertAlmostEqual(self.b['r12'].at[day,s],a[s].iloc[-1]/a[s].iloc[-253]-1)
        expected=(self.data.close.at[day,s]-self.raw['low'].at[day,s])/(self.raw['high'].at[day,s]-self.raw['low'].at[day,s])
        self.assertAlmostEqual(self.b['factors']['stabilization'].at[day,s],expected)
        self.assertAlmostEqual(WEIGHTS.sum(),1.)
        total=sum(self.b['ranks'][k]*WEIGHTS[k] for k in FACTORS)
        pd.testing.assert_frame_equal(total,self.b['models']['soft_weighted'][1])
        z=np.array([-4,-1.25,-.1]);self.assertEqual(np.argmax(-abs(z+1.25)),1)

    def test_no_future_inputs(self):
        n=3100
        d=replace(self.data,**{k:getattr(self.data,k).iloc[:n] for k in ['close','opening','dividends','volume','sector_prices']})
        raw={k:v.iloc[:n] for k,v in self.raw.items()};b=build(d,raw)
        for name in FACTORS:pd.testing.assert_frame_equal(self.b['factors'][name].iloc[:n],b['factors'][name])
        for name,(mask,score) in self.b['models'].items():
            pd.testing.assert_frame_equal(mask.iloc[:n],b['models'][name][0])
            pd.testing.assert_frame_equal(score.iloc[:n],b['models'][name][1])

    def test_missing_history_excludes_without_reweighting(self):
        raw={k:v.copy() for k,v in self.raw.items()};raw['adj_close'].iloc[-100,raw['adj_close'].columns.get_loc('AAPL')]=np.nan
        b=build(self.data,raw)
        self.assertFalse(b['valid'].AAPL.iloc[-1])
        self.assertTrue(np.isnan(b['models']['soft_weighted'][1].AAPL.iloc[-1]))
        self.assertFalse(b['models']['soft_weighted'][0].AAPL.iloc[-1])

    def test_identical_candidate_masks_for_ranking_only(self):
        for name in ['legacy_equal','legacy_weighted']:
            pd.testing.assert_frame_equal(self.b['models']['legacy_common'][0],self.b['models'][name][0])
        for name in ['soft_equal','soft_weighted','soft_weighted_r12']:
            mask,score=self.b['models'][name]
            self.assertFalse((mask&~self.b['gate']).any().any())
            self.assertTrue((score.where(mask).stack()>=.65).all())

    def test_forward_label_timing_and_dividends(self):
        ix=pd.bdate_range('2020-01-01',periods=9)
        opening=pd.DataFrame({'X':np.arange(100,109,dtype=float)},index=ix)
        div=pd.DataFrame({'X':[0,50,1,2,3,4,5,6,7]},index=ix)
        d=replace(self.data,opening=opening,dividends=div)
        r=forward_returns(d,['X'],3,10)
        expected=(104*(1-.0005)+1+2+3)/(101*(1+.0005))-1
        self.assertAlmostEqual(r.X.iloc[0],expected)
        self.assertTrue(r.X.iloc[-4:].isna().all())

    def test_baseline_reproduces_saved_old_equity(self):
        old=pd.read_csv(P.parent/'swing_research'/'daily_equity.csv.gz')
        new=pd.read_csv(P/'daily_equity.csv.gz')
        for cost in [5,10]:
            a=old.query("strategy=='stock_base' and exit_rule=='fixed5' and cost_bps==@cost")
            b=new.query("strategy=='legacy' and cost_bps==@cost")
            self.assertEqual(a.date.tolist(),b.date.tolist())
            np.testing.assert_allclose(a.equity,b.equity,rtol=1e-12)

    def test_saved_accounts_and_prior_day_entry_eligibility(self):
        daily=pd.read_csv(P/'daily_equity.csv.gz');orders=pd.read_csv(P/'orders.csv.gz')
        for (name,cost),g in orders.groupby(['strategy','cost_bps']):
            dd=daily[(daily.strategy==name)&(daily.cost_bps==cost)].set_index('date')
            flow=np.where(g.side=='buy',-g.quantity*g.price-g.fee,g.quantity*g.price-g.fee)
            movements=pd.Series(flow,index=g.date).groupby(level=0).sum().reindex(dd.index,fill_value=0)
            np.testing.assert_allclose(dd.cash,100000+movements.cumsum()+dd.dividends_cumulative,atol=1e-6)
            np.testing.assert_allclose(dd.equity,dd.cash+dd.invested,atol=1e-6)
            self.assertTrue((dd.cash>=-1e-7).all());self.assertLessEqual(dd.positions.max(),5)
            live={}
            for row in g.itertuples():
                i=self.data.close.index.get_loc(pd.Timestamp(row.date))
                if row.side=='buy':
                    self.assertTrue(self.b['models'][name][0].at[self.data.close.index[i-1],row.symbol])
                    self.assertNotIn(row.symbol,live);live[row.symbol]=i
                else:self.assertEqual(i-live.pop(row.symbol),5)
                sectors=[self.data.sector[s] for s in live]
                self.assertLessEqual(len(live),5)
                for sec in set(sectors):self.assertLessEqual(sectors.count(sec),2)

if __name__=='__main__':unittest.main()
