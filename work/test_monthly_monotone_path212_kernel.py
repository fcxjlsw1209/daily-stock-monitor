"""Synthetic path projection, matched clocks, degenerate support and rank tests."""
import copy
import json
import math
import unittest

import numpy as np
import pandas as pd
import monthly_monotone_path212_kernel as k


def fixture():
    dates=pd.bdate_range('2014-01-01','2017-01-04');x=np.arange(len(dates),dtype=float)
    px=pd.DataFrame({'A':100.+.06*x+3*np.sin(.13*x),'B':200.+.02*x+5*np.cos(.23*x),
        'FUND':80.+.02*x+4*np.sin(.37*x)},index=dates)
    return {'adj':px},dict(dates=['2016-08-17','2016-08-31','2016-09-30'],
        symbols=[['A','B','FUND'] for _ in range(3)],etfs=['FUND']),dates.strftime('%Y-%m-%d').tolist()


def prices(shape='curve'):
    if shape=='up':formation=[100.*math.exp(.002*i) for i in range(253)]
    elif shape=='down':formation=[200.*math.exp(-.002*i) for i in range(253)]
    elif shape=='step':formation=[100. if i<126 else 150. for i in range(253)]
    elif shape=='flat':formation=[100.]*253
    else:formation=[100.*math.exp(.3*(i/252.)**2+.015*math.sin(i*.2)) for i in range(253)]
    out=list(formation)
    for j in range(42):out.append(out[-1]*(1.01 if j%2 else .985))
    return np.array(out)


class ProjectionTests(unittest.TestCase):
    def test_known_pava_merges_equal_maximal_blocks(self):
        fit,blocks=k.pava([3.,1.,2.,5.,4.])
        self.assertEqual(fit,[2.,2.,2.,4.5,4.5])
        self.assertEqual(blocks,[dict(start_index=0,end_index=2,count=3,mean=2.),dict(start_index=3,end_index=4,count=2,mean=4.5)])
        self.assertEqual(k.pava([1.,1.,2.,2.])[1],[dict(start_index=0,end_index=1,count=2,mean=1.),dict(start_index=2,end_index=3,count=2,mean=2.)])
        self.assertEqual(k.pava([5.,4.,3.,2.,1.])[0],[3.]*5)
        for bad in ([],[True,1.],[1.,np.inf],[[1.,2.]]):
            with self.assertRaises(ValueError):k.pava(bad)

    def test_pava_residual_kkt_and_block_means(self):
        z=[2.,0.,3.,1.,4.,2.,5.];fit,blocks=k.pava(z)
        self.assertTrue(all(a<=b for a,b in zip(fit,fit[1:])))
        for b in blocks:
            v=z[b['start_index']:b['end_index']+1]
            self.assertEqual(b['mean'],math.fsum(v)/len(v))
            residual=[x-b['mean'] for x in v]
            self.assertAlmostEqual(math.fsum(residual),0.)
            self.assertTrue(all(math.fsum(residual[:j])>=-1e-14 for j in range(1,len(v)+1)))

    def test_exact_quality_branches_and_roundoff_limit(self):
        self.assertEqual(k.quality(0.,1.),(1.,1.,'zero_sse'))
        self.assertEqual(k.quality(1.+1e-14,1.,'single_block'),(0.,0.,'single_block'))
        q,unclipped,branch=k.quality(1.+5e-13,1.);self.assertEqual(q,0.);self.assertLess(unclipped,0.)
        for args in ((1.+2e-12,1.),(float('inf'),1.),(1.,1e-320)):
            with self.assertRaises(FloatingPointError):k.quality(*args)
        for args in ((-1.,1.),(0.,0.),(True,1.)):
            with self.assertRaises(ValueError):k.quality(*args)
        z=[float(i-126) for i in range(253)];sst=math.fsum(x*x for x in z);a=k.path_fits(z,0.,sst)
        self.assertEqual(a['q_iso'],1.);self.assertEqual(a['q_line'],1.)
        self.assertEqual(a['ols_slope'],1.);self.assertEqual(a['sse_line'],0.)

    def test_up_step_curve_iso_dominates_line_without_magnitude_claim(self):
        for shape in ('up','step','curve'):
            a=k.measure(prices(shape));self.assertTrue(a['available']);self.assertTrue(a['attenuation_applied'])
            self.assertGreaterEqual(a['q_iso']+1e-12,a['q_line'])
            self.assertEqual(a['raw_candidate'],a['normalized_momentum']*a['q_iso'])
            self.assertEqual(a['raw_control'],a['normalized_momentum']*a['q_line'])
        step=k.measure(prices('step'));self.assertEqual(step['q_iso'],1.)
        self.assertLess(step['q_line'],1.);self.assertEqual(len(step['blocks']),2)
        self.assertGreater(step['raw_candidate'],step['raw_control'])

    def test_down_and_nonpositive_numerator_never_lifted_by_quality(self):
        a=k.measure(prices('down'));self.assertTrue(a['available']);self.assertFalse(a['attenuation_applied'])
        self.assertLess(a['normalized_momentum'],0.);self.assertEqual(a['q_iso'],0.);self.assertEqual(a['q_line'],0.)
        self.assertEqual(a['iso_quality_branch'],'single_block');self.assertEqual(a['line_quality_branch'],'zero_slope')
        self.assertEqual(a['raw_candidate'],a['normalized_momentum']);self.assertEqual(a['raw_control'],a['normalized_momentum'])
        p=prices();p[0]=p[252];a=k.measure(p)
        self.assertTrue(a['available']);self.assertEqual(a['numerator'],0.);self.assertEqual(a['raw_candidate'],0.)
        self.assertEqual(a['raw_control'],0.);self.assertFalse(a['attenuation_applied'])

    def test_zero_volatility_or_formation_variance_joint_missing_no_fits(self):
        for p,reason in ((np.ones(295)*100.,'zero_volatility'),(prices('flat'),'zero_formation_variance')):
            a=k.measure(p);self.assertTrue(a['source_available']);self.assertFalse(a['available'])
            self.assertEqual(a['reason'],reason);self.assertEqual(a['descriptive_pava_fits'],0)
            self.assertEqual(a['descriptive_affine_fits'],0);self.assertEqual(a['blocks'],[])
            self.assertEqual(len(a['simple_returns']),63);self.assertEqual(len(a['formation_logs']),253)
            self.assertIsNone(a['q_iso']);self.assertTrue(np.isnan(a['raw_candidate']))
        p=prices();p[231:]=100.;a=k.measure(p);self.assertGreater(a['sst'],0.)
        self.assertEqual(a['volatility'],0.);self.assertEqual(a['reason'],'zero_volatility')

    def test_exact_risk_numerator_and_double_centering(self):
        p=prices();a=k.measure(p);r=[float(p[i])/float(p[i-1])-1. for i in range(232,295)]
        mean=math.fsum(r)/63.;vol=math.sqrt(math.fsum((v-mean)**2 for v in r)/62.)
        logs=[math.log(float(v)) for v in p[:253]];mu=math.fsum(logs)/253.;z=[v-mu for v in logs];zm=math.fsum(z)/253.
        self.assertEqual(a['simple_returns'],r);self.assertEqual(a['return_mean'],mean);self.assertEqual(a['volatility'],vol)
        self.assertEqual(a['formation_logs'],logs);self.assertEqual(a['centered_logs'],z);self.assertEqual(a['centered_mean'],zm)
        self.assertEqual(a['sst'],math.fsum((v-zm)**2 for v in z))
        self.assertEqual(a['numerator'],float(p[252])/float(p[0])-1.)
        self.assertEqual(a['normalized_momentum'],a['numerator']/vol)

    def test_full295_price_support_and_protocol(self):
        for bad,reason in ((None,'nonnumeric_adjusted_price'),(True,'nonnumeric_adjusted_price'),('1','nonnumeric_adjusted_price'),
            (1j,'nonnumeric_adjusted_price'),(0.,'invalid_adjusted_price'),(-1.,'invalid_adjusted_price'),
            (np.nan,'invalid_adjusted_price'),(np.inf,'invalid_adjusted_price')):
            p=list(prices());p[140]=bad;a=k.measure(p)
            self.assertEqual(a['reason'],reason);self.assertFalse(a['source_available']);self.assertEqual(a['formation_logs'],[])
        for p in (np.ones(294),np.ones((295,1)),1.):
            with self.assertRaises(ValueError):k.measure(p)

    def test_finite_source_nonfinite_arithmetic_is_hard_error(self):
        for mode in ('numerator','return','variance','normalized'):
            p=prices()
            if mode=='numerator':p[0]=1e-308;p[252]=1e308
            elif mode=='return':p[231]=1e-308;p[232]=1e308
            elif mode=='variance':p[231]=1.;p[232]=1e200
            else:
                p[:]=1.;p[0]=1e-300
                for i in range(232,295):p[i]=p[i-1]*(1.+(1e-14 if i%2 else -2e-14))
            with self.subTest(mode=mode),self.assertRaises(FloatingPointError):k.measure(p)

    def test_tiny_positive_risk_and_source_scaling_no_floor(self):
        p=prices();p[231:]=100.
        for i in range(232,295):p[i]=p[i-1]*(1.+(1e-14 if i%2 else -2e-14))
        a=k.measure(p);self.assertTrue(a['available']);self.assertGreater(a['volatility'],0.);self.assertLess(a['volatility'],1e-12)
        p=prices();before=p.copy();a=k.measure(p)
        for scale in (2.**-400,2.**400):
            b=k.measure(p*scale)
            self.assertEqual(a['numerator'],b['numerator']);self.assertEqual(a['volatility'],b['volatility'])
            for field in ('q_iso','q_line','raw_candidate','raw_control'):
                self.assertAlmostEqual(a[field],b[field],delta=1e-10)
        np.testing.assert_array_equal(p,before)


class BuilderTests(unittest.TestCase):
    def test_schema_fit_counts_clocks_and_json_safe_audit(self):
        p,m,c=fixture();v,a=k.build_features(p,m,c)
        self.assertEqual(set(v),set(k.VALUE_KEYS));self.assertEqual(a['source_available_by_event'],[3]*3)
        self.assertEqual(a['available_by_event'],[3]*3)
        expected=dict(valid_price_observations=295*9,formation_log_observations=253*9,simple_return_observations=63*9,
            numerator_evaluations=9,risk_evaluations=9,descriptive_pava_fits=9,descriptive_affine_fits=9,ranked_observations=18)
        for name,value in expected.items():self.assertEqual(a[name],value)
        for name in ('new_model_fits','predictive_model_fits','new_predictive_model_fits'):self.assertEqual(a[name],0)
        self.assertTrue(np.isfinite(v['candidate'][0]).all());json.dumps(a,allow_nan=False)
        for row in a['event_history']:
            t=row['session_position'];self.assertEqual([row[x] for x in ('price_start','price_end','formation_price_start',
                'formation_price_end','risk_price_start','risk_price_end')],[c[t-294],c[t],c[t-294],c[t-42],c[t-63],c[t]])

    def test_original_average_percentile_ties_singleton_and_empty(self):
        np.testing.assert_array_equal(k.percentile_rank(np.array([1.,2.,2.,4.,np.nan]),np.array([True]*4+[False])),[-.25,.125,.125,.5,np.nan])
        np.testing.assert_array_equal(k.percentile_rank(np.ones(3),np.ones(3,bool)),[2/3-.5]*3)
        np.testing.assert_array_equal(k.percentile_rank(np.array([1.,np.nan]),np.array([True,False])),[.5,np.nan])
        p,m,c=fixture();p['adj'].loc[:,['B','FUND']]=np.nan;v,a=k.build_features(p,m,c)
        self.assertEqual(a['available_by_event'],[1]*3);np.testing.assert_array_equal(v['candidate'][:,0],[.5]*3)
        p['adj'].loc[:,:]=100.;v,a=k.build_features(p,m,c)
        self.assertEqual(a['source_available_cells'],9);self.assertEqual(a['available_cells'],0)
        self.assertEqual(a['risk_evaluations'],9);self.assertEqual(a['descriptive_pava_fits'],0)
        self.assertTrue(all(np.isnan(x).all() for key,x in v.items() if key!='available'));json.dumps(a,allow_nan=False)

    def test_missing_symbol_history_sessions_and_intermediate_price(self):
        p,m,c=fixture();t=c.index(m['dates'][0]);p['adj']=p['adj'].drop(index=c[t-100],columns='B')
        _,a=k.build_features(p,m,c,1)
        self.assertEqual([x['reason'] for x in a['event_history'][0]['cells']],['missing_official_session','missing_symbol','missing_official_session'])
        _,a=k.build_features(p,m,c[t-100:],1);self.assertEqual(a['event_history'][0]['price_end'],None)
        self.assertEqual(a['reason_counts'],{'insufficient_history':3})
        p,m,c=fixture();p['adj'].loc[c[t-270],'A']=0.;v,a=k.build_features(p,m,c,1)
        self.assertFalse(v['available'][0,0]);self.assertEqual(a['source_available_cells'],2)

    def test_physical_prefix_future_poison_and_no_mutation(self):
        p,m,c=fixture();original=p['adj'].copy(deep=True);mm=copy.deepcopy(m);v,a=k.build_features(p,m,c)
        for n in range(4):
            pp={'adj':p['adj'].loc[:m['dates'][n-1]].copy() if n else p['adj'].iloc[:0].copy()}
            vv,aa=k.build_features(pp,m,c,n)
            for key in v:np.testing.assert_array_equal(vv[key],v[key][:n])
            self.assertEqual(aa['event_history'],a['event_history'][:n])
        pp={'adj':p['adj'].astype(object)};pp['adj'].loc[pp['adj'].index>m['dates'][0]]='FUTURE';pp['adj']['UNUSED']='POISON'
        vv,aa=k.build_features(pp,m,c,1)
        for key in v:np.testing.assert_array_equal(vv[key],v[key][:1])
        self.assertEqual(aa['event_history'],a['event_history'][:1]);pd.testing.assert_frame_equal(original,p['adj']);self.assertEqual(m,mm)

    def test_cross_section_permutation_and_etfs_unchanged_domain(self):
        p,m,c=fixture();v,_=k.build_features(p,m,c);perm=[2,0,1]
        other=dict(m,symbols=[[x[j] for j in perm] for x in m['symbols']],etfs=[])
        vv,_=k.build_features(p,other,c)
        for key in v:np.testing.assert_array_equal(vv[key],v[key][:,perm])

    def test_source_metadata_and_calendar_protocol(self):
        p,m,c=fixture()
        for n in (-1,4,True,1.):
            with self.assertRaises(ValueError):k.build_features(p,m,c,n)
        for key,value in [('dates',['2016-08-17','2016-08-30','2016-09-30']),('symbols',[['A'],['A','B'],['A']]),('etfs',['FUND','FUND'])]:
            bad=dict(m);bad[key]=value
            with self.assertRaises(ValueError):k.build_features(p,bad,c)
        with self.assertRaises(ValueError):k.build_features(p,m,c[:c.index(m['dates'][-1])+1])
        with self.assertRaises(ValueError):k.build_features({'adj':p['adj'],'volume':p['adj']},m,c)
        with self.assertRaises(ValueError):k.build_features({'adj':p['adj'].iloc[::-1]},m,c)
        for raw,mask in ((np.zeros(2),np.ones(2,int)),(np.zeros(2),np.ones(1,bool)),(np.array([np.inf]),np.ones(1,bool))):
            with self.assertRaises(ValueError):k.percentile_rank(raw,mask)


if __name__=='__main__':unittest.main(verbosity=2)
