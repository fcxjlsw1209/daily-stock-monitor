import unittest
import numpy as np
import pandas as pd
from audit import feature_panel,snapshot,pair_difference,ranked,transition

class AuditTests(unittest.TestCase):
    def sample(self):
        dates=pd.bdate_range('2017-01-02',periods=650)
        rng=np.random.default_rng(12)
        px=pd.DataFrame(50*np.exp(np.cumsum(rng.normal(.001,.01,(650,4)),axis=0)),index=dates,columns=['A','B','C','SPY'])
        raw={'close':px,'high':px*1.015,'low':px*.99,'volume':pd.DataFrame(100000.,index=dates,columns=px.columns),'sector60':px.pct_change(60,fill_method=None)*0}
        return px,raw

    def test_future_prices_do_not_change_signal(self):
        px,raw=self.sample();d=px.index[520]
        feat,state,warm=feature_panel(px,raw);a=snapshot(d,['A','B','C','SPY'],feat,state,warm)
        short=px.loc[:d];rr={k:v.loc[:d] for k,v in raw.items()}
        f2,s2,w2=feature_panel(short,rr);b=snapshot(d,['A','B','C','SPY'],f2,s2,w2)
        pd.testing.assert_frame_equal(a['tab'],b['tab'])
        pd.testing.assert_series_equal(a['score'],b['score'])
        self.assertEqual(a['chosen'],b['chosen'])

    def test_components_and_pair_gap_are_exact(self):
        px,raw=self.sample();f,s,w=feature_panel(px,raw);z=snapshot(px.index[-1],['A','B','C','SPY'],f,s,w)
        a,b=z['order'][-1],z['order'][0];pair=pair_difference(a,b,z)
        self.assertAlmostEqual(pair['gap_points'],sum(pair['components'].values()),places=10)
        np.testing.assert_allclose(z['score'],z['comp'].sum(axis=1),atol=1e-14)

    def test_rank_eligibility_prevents_selecting_high_score(self):
        s=pd.Series([1.,.5,.4],index=['A','B','C']);e=pd.Series([False,True,True],index=s.index)
        self.assertEqual(ranked(s,e),(['B','C'],['B','C']))

    def test_single_eligible_is_cash(self):
        s=pd.Series([1.,.5],index=['A','B']);e=pd.Series([True,False],index=s.index)
        self.assertEqual(ranked(s,e),(['A'],[]))

    def test_transition_is_unordered_and_removal_is_full_exit(self):
        t=transition({'A','B','C'},{'B','C','D'})
        self.assertEqual(t['removed'],['A']);self.assertEqual(t['added'],['D']);self.assertEqual(t['retained'],['B','C'])
        self.assertEqual(transition({'A','B','C'},set(['C','A','B']))['removed'],[])

    def test_ties_have_stable_order(self):
        s=pd.Series([1.,1.,1.,.5],index=['A','B','C','D']);e=pd.Series(True,index=s.index)
        self.assertEqual(ranked(s,e)[1],['A','B','C'])

if __name__=='__main__':unittest.main()
