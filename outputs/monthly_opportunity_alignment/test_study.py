import unittest
import numpy as np
import pandas as pd
import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('opportunity_driver_test',Path(__file__).with_name('study.py'))
study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)


class DriverTests(unittest.TestCase):
    def fixture(self,n=18):
        ss=['S%02d'%i for i in range(49)]
        dates=pd.bdate_range('2018-01-01','2020-01-31')
        months=pd.Series(dates,index=dates).groupby(dates.to_period('M')).last().iloc[:n+1]
        m=dict(dates=[str(d.date()) for d in months.iloc[:-1]],ends=[str(d.date()) for d in months.iloc[1:]],symbols=[ss]*n)
        sc=np.tile(np.r_[.4,.3,.2,.199,.19,np.arange(44)*-.01],(n,1))
        a=dict(base_original65=sc,eligible=np.ones_like(sc,bool),full=np.zeros(n*49),available=np.ones(n*49,bool))
        a['full'].reshape(n,49)[:,3]=.10
        px=pd.DataFrame(100.,index=dates,columns=ss)
        # Each complete month's challenger differs, creating strictly causal budget history.
        px.S03=100*np.exp(np.arange(len(px))*.001)
        return m,a,px

    def test_pair_missing_label_does_not_replace_candidate(self):
        m,a,px=self.fixture(2);x=np.arange(2*49*10).reshape(2,49,10)/1000.
        rows,features,targets,audit=study.pair_rows(m,a['base_original65'],a['eligible'],x,px)
        broken=px.copy();broken.loc[m['ends'][0],'S03']=np.nan
        rr,ff,tt,aa=study.pair_rows(m,a['base_original65'],a['eligible'],x,broken)
        self.assertEqual(rows,rr);np.testing.assert_array_equal(features,ff)
        self.assertTrue(np.isnan(tt['portfolio'][0]));self.assertEqual(aa['potential_pairs'],audit['potential_pairs'])

    def test_budget_changes_only_third_and_near_boundary(self):
        m,a,px=self.fixture(1)
        row=dict(base='original65',family='budget',key='fixed',budget=.05)
        ch,_=study.choose(row,a,m,1)
        self.assertEqual(ch[0],[0,1,3])
        zero,_=study.choose(row,a,m,1,np.zeros(1));self.assertEqual(zero[0],[0,1,2])
        a['base_original65'][0,3]=.10
        ch,_=study.choose(row,a,m,1);self.assertEqual(ch[0],[0,1,2])

    def test_dynamic_future_price_change_preserves_prior_decisions(self):
        m,a,px=self.fixture();row=dict(base='original65',family='budget',key='dynamic',budget=.05)
        ch,sc,ga=study.decisions(row,a,m,len(m['dates']),px)
        i=15;alt=px.copy();alt.loc[alt.index>pd.Timestamp(m['dates'][i]),'S03']*=20
        cc,ss,gg=study.decisions(row,a,m,len(m['dates']),alt)
        self.assertEqual(ch[:i+1],cc[:i+1]);np.testing.assert_array_equal(sc[:i+1],ss[:i+1])
        self.assertEqual(ga['months'][:i+1],gg['months'][:i+1])
        self.assertTrue(all(z['budget']==0 for z in ga['months'][:13]))

    def test_unavailable_pair_forecast_is_exact_baseline(self):
        m,a,px=self.fixture(1);a['original65_portfolio']=np.ones((1,49));a['original65_valid_portfolio']=np.zeros((1,49),bool)
        row=dict(base='original65',family='pair',key='portfolio',budget=.05)
        ch,sc=study.choose(row,a,m,1);self.assertEqual(ch[0],[0,1,2]);np.testing.assert_array_equal(sc,a['base_original65'])

    def test_truncated_month_is_missing_gate_evidence(self):
        m,a,px=self.fixture();row=dict(base='original65',family='budget',key='dynamic',budget=.05)
        px=px.drop(pd.Timestamp(m['ends'][12]))
        _,_,ga=study.decisions(row,a,m,len(m['dates']),px)
        self.assertIsNone(ga['virtual_log_differences'][12])
        self.assertEqual(ga['months'][14]['budget'],0.)


if __name__=='__main__':unittest.main()
