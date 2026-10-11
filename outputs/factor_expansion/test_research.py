import unittest
from dataclasses import replace
import numpy as np
import pandas as pd
from research import P,NEW,load,prepare,new_factors,combine_score,residualize,correlate,bootstrap_indices,bootstrap_interval

class ExpansionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data,cls.raw=load();cls.b=prepare(cls.data,cls.raw)

    def test_factor_formulas_and_disjoint_windows(self):
        s='AAPL';a=self.raw['adj_close'][s];c=self.data.close[s];v=self.data.volume[s]
        f=self.b['factors'];day=a.index[-1]
        self.assertAlmostEqual(f['high_proximity'].at[day,s],a.iloc[-1]/a.iloc[-252:].max())
        self.assertAlmostEqual(f['quiet_volume'].at[day,s],-v.iloc[-3:].mean()/v.iloc[-23:-3].mean())
        r=a.pct_change(fill_method=None)
        self.assertAlmostEqual(f['low_downside'].at[day,s],-np.sqrt(np.minimum(r.iloc[-63:],0).pow(2).mean()))
        high=self.raw['high'][s];low=self.raw['low'][s];rng=(high-low)/c.shift(1)
        self.assertAlmostEqual(f['range_contraction'].at[day,s],-rng.iloc[-5:].mean()/rng.iloc[-25:-5].mean())
        mult=((2*c-high-low)/(high-low)).where(high!=low,0)
        self.assertAlmostEqual(f['accumulation'].at[day,s],(mult*v).iloc[-21:].sum()/v.iloc[-21:].sum())

    def test_beta_is_lagged_ols_and_residual_formula(self):
        a=self.raw['adj_close'].AAPL;m=self.raw['market_adjusted']
        lr=np.log(a/a.shift(1));mr=np.log(m/m.shift(1));x=mr.iloc[-127:-1].to_numpy();y=lr.iloc[-127:-1].to_numpy()
        beta=np.linalg.lstsq(np.column_stack([np.ones(len(x)),x]),y,rcond=None)[0][1]
        self.assertAlmostEqual(self.b['beta'].AAPL.iloc[-1],beta,places=10)
        self.assertAlmostEqual(self.b['residual'].AAPL.iloc[-1],lr.iloc[-1]-beta*mr.iloc[-1])
        e=self.b['residual'].AAPL.iloc[-60:]
        self.assertAlmostEqual(self.b['factors']['beta_adjusted_strength'].AAPL.iloc[-1],e.sum()/(e.std()*np.sqrt(60)))

    def test_weight_preserves_r12_minus_one_and_admission(self):
        self.assertAlmostEqual(combine_score(1.,1.,1.),1.)
        old,mom,new=.6,.4,.8;delta=.1
        difference=combine_score(old+.3*delta,mom+delta,new)-combine_score(old,mom,new)
        self.assertAlmostEqual(difference,.3*delta)
        for name,(mask,score) in self.b['models'].items():
            self.assertNotIn('r12',name)
            if name!='baseline_original':pd.testing.assert_frame_equal(mask,self.b['models']['baseline_common'][0])
        m=self.b['old']['ranks']['long_momentum'];old=self.b['models']['baseline_common'][1]
        for k in NEW:pd.testing.assert_frame_equal(self.b['models'][k][1],combine_score(old,m,self.b['ranks'][k]))

    def test_temporal_truncation(self):
        n=3050
        data=replace(self.data,**{k:getattr(self.data,k).iloc[:n] for k in ['close','opening','dividends','volume','sector_prices']})
        raw={k:v.iloc[:n] for k,v in self.raw.items()};b=prepare(data,raw)
        for k in NEW:pd.testing.assert_frame_equal(self.b['factors'][k].iloc[:n],b['factors'][k])
        pd.testing.assert_frame_equal(self.b['beta'].iloc[:n],b['beta'])
        for k,(mask,score) in b['models'].items():
            pd.testing.assert_frame_equal(self.b['models'][k][0].iloc[:n],mask)
            pd.testing.assert_frame_equal(self.b['models'][k][1].iloc[:n],score)

    def test_missing_new_input_cannot_enter(self):
        raw={k:v.copy() for k,v in self.raw.items()};raw['high'].iloc[-10,raw['high'].columns.get_loc('AAPL')]=np.nan
        b=prepare(self.data,raw)
        self.assertFalse(b['common'].AAPL.iloc[-1])
        self.assertFalse(b['models']['baseline_common'][0].AAPL.iloc[-1])

    def test_partial_regression_algebra(self):
        rng=np.random.default_rng(1);controls=rng.normal(size=(50,5));values=rng.normal(size=(50,6))
        residual=residualize(values,controls)
        x=np.column_stack([np.ones(50),controls])
        np.testing.assert_allclose(x.T@residual,0,atol=1e-12)
        self.assertTrue(np.isnan(correlate(residualize(controls[:,0],controls),values[:,0])))

    def test_block_sampling_reproducible(self):
        a=bootstrap_indices(100);b=bootstrap_indices(100)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(a.shape,(1000,100));self.assertGreaterEqual(a.min(),0);self.assertLess(a.max(),100)
        np.testing.assert_array_equal(np.diff(a[:,:20],axis=1),np.ones((1000,19),dtype=int))
        lo,hi=bootstrap_interval(np.full(100,.001));self.assertAlmostEqual(lo,.001);self.assertAlmostEqual(hi,.001)

    def test_original_baseline_reproduced(self):
        old=pd.read_csv(P.parent/'factor_research'/'daily_equity.csv.gz')
        new=pd.read_csv(P/'daily_equity.csv.gz')
        for cost in [5,10,20]:
            a=old[(old.strategy=='soft_weighted')&(old.cost_bps==cost)]
            b=new[(new.strategy=='baseline_original')&(new.cost_bps==cost)]
            self.assertEqual(a.date.tolist(),b.date.tolist());np.testing.assert_allclose(a.equity,b.equity,rtol=1e-12)

    def test_all_saved_accounts_and_execution(self):
        daily=pd.read_csv(P/'daily_equity.csv.gz');orders=pd.read_csv(P/'orders.csv.gz')
        for (name,cost),g in orders.groupby(['strategy','cost_bps']):
            dd=daily[(daily.strategy==name)&(daily.cost_bps==cost)].set_index('date')
            flow=np.where(g.side=='buy',-g.quantity*g.price-g.fee,g.quantity*g.price-g.fee)
            movement=pd.Series(flow,index=g.date).groupby(level=0).sum().reindex(dd.index,fill_value=0)
            np.testing.assert_allclose(dd.cash,100000+movement.cumsum()+dd.dividends_cumulative,atol=1e-6)
            np.testing.assert_allclose(dd.equity,dd.cash+dd.invested,atol=1e-6)
            self.assertTrue((dd.cash>=-1e-7).all());live={}
            for r in g.itertuples():
                i=self.data.close.index.get_loc(pd.Timestamp(r.date))
                if r.side=='buy':
                    self.assertNotIn(r.symbol,live);live[r.symbol]=i
                    self.assertTrue(self.b['models'][name][0].at[self.data.close.index[i-1],r.symbol])
                else:self.assertEqual(i-live.pop(r.symbol),5)
                self.assertLessEqual(len(live),5);sectors=[self.data.sector[s] for s in live]
                for sec in set(sectors):self.assertLessEqual(sectors.count(sec),2)

if __name__=='__main__':unittest.main()
