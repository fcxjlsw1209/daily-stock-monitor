"""Only synthetic source windows and pure numerical checks."""
from copy import deepcopy
from decimal import Decimal, localcontext
import json
import unittest
import numpy as np
import pandas as pd
import monthly_sector_asymmetry138_factor as f


def vectors():
    x=np.r_[np.linspace(.001,.04,126),-np.linspace(.001,.04,126)]
    noise=np.sin(np.arange(252))*.001
    y=np.where(x>0,2*x+.1,.5*x-.05)+noise
    return y,x


def fixture():
    dates=pd.bdate_range('2015-01-01',periods=340)
    names=['S'+str(i) for i in range(32)]+['E'+str(i) for i in range(17)]
    m=dict(dates=[str(dates[300].date()),str(dates[320].date())],symbols=[names[:],names[:]],etfs=names[32:])
    x=.01*np.sin(np.arange(339)*.53)+.002*np.cos(np.arange(339)*.17)
    values={'SECTOR':100*np.exp(np.r_[0,np.cumsum(x)])}
    for i,s in enumerate(names):values[s]=100*np.exp(np.r_[0,np.cumsum((1+i*.01)*x+.003*np.sin(np.arange(339)*(i+1)*.13))])
    panel=dict(adj=pd.DataFrame(values,index=dates),sectors=pd.DataFrame(index=dates))
    mapper=lambda day,ns,sectors,etfs:{s:None if s in etfs else 'SECTOR' for s in ns}
    return panel,m,[str(d.date()) for d in dates],mapper


class MomentTests(unittest.TestCase):
    def test_exact_separate_intercepts(self):
        _,x=vectors();y=np.where(x>0,2*x+10.,.5*x-20.)
        d=f.conditional_gaps(y,x);self.assertTrue(d['available'])
        self.assertAlmostEqual(d['raw_beta_gap'],1.5,places=11)
        self.assertAlmostEqual(d['raw_correlation_gap'],0.,places=12)
    def test_decimal_moment_oracle(self):
        y,x=vectors();d=f.conditional_gaps(y,x)
        with localcontext() as c:
            c.prec=90
            for key,mask in (('up',x>0),('down',x<0)):
                xx=[Decimal.from_float(float(v)) for v in x[mask]];yy=[Decimal.from_float(float(v)) for v in y[mask]]
                mx=sum(xx)/len(xx);my=sum(yy)/len(yy)
                sx=sum((v-mx)**2 for v in xx);sy=sum((v-my)**2 for v in yy);cross=sum((a-mx)*(b-my) for a,b in zip(xx,yy))
                self.assertAlmostEqual(d[key]['beta'],float(cross/sx),places=12)
                self.assertAlmostEqual(d[key]['rho'],float(cross/(sx*sy).sqrt()),places=12)
    def test_scale_changes_beta_not_correlation(self):
        y,x=vectors();a=f.conditional_gaps(y,x);b=f.conditional_gaps(y*3,x*2)
        self.assertAlmostEqual(b['raw_beta_gap'],a['raw_beta_gap']*1.5,places=12)
        self.assertAlmostEqual(a['raw_correlation_gap'],b['raw_correlation_gap'],places=12)
    def test_negative_stock_relation_and_sign(self):
        y,x=vectors();a=f.conditional_gaps(y,x);b=f.conditional_gaps(-y,x)
        self.assertAlmostEqual(a['raw_beta_gap'],-b['raw_beta_gap'],places=12)
        self.assertAlmostEqual(a['raw_correlation_gap'],-b['raw_correlation_gap'],places=12)
    def test_forty_boundary_and_zero_exclusion(self):
        x=np.r_[np.linspace(.01,.04,40),-np.linspace(.01,.04,40),np.zeros(172)];y=x+np.sin(np.arange(252))*.001
        d=f.conditional_gaps(y,x);self.assertTrue(d['available']);self.assertEqual(d['zero_count'],172)
        x[0]=0;self.assertEqual(f.conditional_gaps(y,x)['reason'],'insufficient_group')
    def test_constant_stock_or_sector_shared_fallback(self):
        y,x=vectors();y[:126]=3.;d=f.conditional_gaps(y,x)
        self.assertEqual(d['reason'],'zero_stock_variance');self.assertIsNone(d['raw_beta_gap'])
        y,x=vectors();x[:126]=.01;self.assertEqual(f.conditional_gaps(y,x)['reason'],'zero_sector_variance')
    def test_small_but_nonzero_variance_not_thresholded(self):
        x=np.r_[np.linspace(1e-90,2e-90,126),-np.linspace(1e-90,2e-90,126)];y=3*x
        self.assertTrue(f.conditional_gaps(y,x)['available'])
    def test_bad_return_shape_values_bool(self):
        y,x=vectors()
        for bad in (x[:-1],np.full(252,np.nan),np.full(252,True),np.full(252,'0.1')):
            with self.assertRaises(ValueError):f.conditional_gaps(y,bad)
    def test_log_exact_round60_formula_extreme_prices(self):
        p=np.geomspace(1e-250,1e250,253);m,e=np.frexp(p)
        np.testing.assert_array_equal(f.log_returns(p),np.log(m[1:]/m[:-1])+np.diff(e)*np.log(2.))
    def test_log_rejects_source_missing(self):
        for value in (0.,-1.,np.inf,np.nan):
            p=np.ones(253);p[120]=value
            with self.assertRaises(ValueError):f.log_returns(p)
    def test_derived_beta_overflow_fatal(self):
        x=np.r_[np.linspace(1e-300,2e-300,126),-np.linspace(1e-300,2e-300,126)];y=x*1e300*1e100
        with self.assertRaises(FloatingPointError):f.conditional_gaps(y,x)


class BuilderTests(unittest.TestCase):
    def test_calendar_window_ranking_and_immutability(self):
        p,m,s,mapper=fixture();old=p['adj'].copy();d=f.build_features(p,m,s,mapper)
        self.assertEqual(d['arrays']['common_available'].sum(),64);self.assertEqual(len(d['audit']['cells']),98)
        cell=d['audit']['cells'][0];self.assertEqual((cell['window_start'],cell['window_end']),(s[6],s[258]))
        for raw,rank in zip(f.EXTRA[:2],f.EXTRA[2:4]):
            expected=pd.Series(d['arrays'][raw][0,:32]).rank(method='average',pct=True)-.5
            np.testing.assert_array_equal(d['arrays'][rank][0,:32],expected)
            self.assertTrue(np.isnan(d['arrays'][rank][:,32:]).all())
        pd.testing.assert_frame_equal(p['adj'],old);json.dumps(d['audit'],allow_nan=False)
    def test_all_stock_ranks_ties_no_eligibility(self):
        p,m,s,mapper=fixture();p['adj']['S1']=p['adj']['S0'];m['eligible']=[[False]*49]*2
        d=f.build_features(p,m,s,mapper);a=d['arrays'];self.assertEqual(a['rank_beta_gap'][0,0],a['rank_beta_gap'][0,1])
        self.assertTrue(a['common_available'][:,:32].all())
    def test_missing_interior_not_compressed(self):
        p,m,s,mapper=fixture();p['adj']=p['adj'].drop(pd.Timestamp(s[100]));d=f.build_features(p,m,s,mapper)
        self.assertFalse(d['arrays']['common_available'].any());self.assertEqual(d['audit']['cells'][0]['reason'],'invalid_sector_path')
    def test_missing_stock_only_one_common_mask(self):
        p,m,s,mapper=fixture();p['adj'].loc[p['adj'].index[100],'S0']=np.nan;d=f.build_features(p,m,s,mapper)
        self.assertEqual(d['audit']['cells'][0]['reason'],'invalid_stock_path')
        for key in f.EXTRA[:4]:self.assertTrue(np.isnan(d['arrays'][key][0,0]))
        self.assertEqual(d['arrays']['common_available'].sum(),62)
    def test_mapper_none_and_etf_first(self):
        p,m,s,_=fixture();mapper=lambda day,ns,sectors,etfs:{name:None if name=='S0' else 'SECTOR' for name in ns}
        d=f.build_features(p,m,s,mapper);self.assertEqual(d['audit']['cells'][0]['reason'],'missing_sector_mapping')
        self.assertEqual(d['audit']['cells'][32]['reason'],'etf_not_applicable');self.assertIsNone(d['audit']['cells'][32]['benchmark'])
    def test_future_and_recent_poison_prefix(self):
        p,m,s,mapper=fixture();short=dict(m,dates=m['dates'][:1],symbols=m['symbols'][:1])
        d=f.build_features(p,short,s,mapper);p['adj'].iloc[259:]=np.nan
        got=f.build_features(p,short,s,mapper);self.assertEqual(d['audit'],got['audit'])
        for k in f.EXTRA:np.testing.assert_array_equal(d['arrays'][k],got['arrays'][k])
    def test_insufficient_history(self):
        p,m,s,mapper=fixture();m['dates']=[s[100]];m['symbols']=m['symbols'][:1]
        d=f.build_features(p,m,s,mapper);self.assertEqual(d['audit']['cells'][0]['reason'],'insufficient_history')
        self.assertIsNone(d['audit']['cells'][0]['window_end'])
    def test_bad_axes_and_mapping(self):
        p,m,s,mapper=fixture()
        with self.assertRaises(ValueError):f.build_features(p,m,s[::-1],mapper)
        with self.assertRaises(ValueError):f.build_features(p,m,s,lambda *args:{})
        m['dates'][0]='2015-01-03'
        with self.assertRaises(ValueError):f.build_features(p,m,s,mapper)

if __name__=='__main__':unittest.main()
