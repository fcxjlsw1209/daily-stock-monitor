import unittest
import numpy as np
import pandas as pd
import shape_study as s

class ShapeTests(unittest.TestCase):
    def test_shapes_and_industry_are_prefix_causal(self):
        px,raw,members,periods,sources,pools=s.load_prices()
        ss=['SPY','QQQ','NVDA','AMZN']+list(set(s.SECTOR.values()));ss=sorted(set(ss));px=px[ss];raw={k:v[ss] for k,v in raw.items()}
        sectors=pd.read_csv(s.O/'index_union/alias_resolved_inputs/sectors.csv.gz',index_col=0,parse_dates=True,low_memory=False)
        cut='2024-02-29';f=s.shape_features(px,raw,sectors,set())
        q=s.shape_features(px.loc[:cut],{k:v.loc[:cut] for k,v in raw.items()},sectors.loc[:cut],set())
        for k in f:pd.testing.assert_series_equal(f[k].loc[cut],q[k].loc[cut],rtol=1e-12,atol=1e-12)

    def test_corrected_sector_uses_same_momentum_endpoints(self):
        px,raw,members,periods,sources,pools=s.load_prices();ss=sorted(set(['SPY','NVDA']+list(s.SECTOR.values())));px=px[ss];raw={k:v[ss] for k,v in raw.items()}
        sectors=pd.DataFrame({'NVDA':'Technology'},index=px.index);f=s.shape_features(px,raw,sectors,set())
        m=px.shift(42)/px.shift(252)-1;expected=(1+m.NVDA)/(1+m.XLK)-1
        pd.testing.assert_series_equal(f['industry_long252'].NVDA,expected,check_names=False)

    def test_all_trial_budgets(self):
        for r in s.read(s.P/'TRIALS.json'):
            A,b,prior=s.e.limits(r['spec']);w=np.array(r['weights'])
            self.assertAlmostEqual(w.sum(),1,places=7)
            self.assertTrue((np.einsum('ij,j->i',A,w,optimize=False)<=b+1e-7).all())

if __name__=='__main__':unittest.main()
