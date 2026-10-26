"""Synthetic-only207 tests; no stored market arrays, orders or outcomes."""
import copy
import json
import math
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import monthly_down_volume207_kernel as kernel


def fixture():
    days=pd.bdate_range('2016-01-01','2016-11-02')
    x=np.arange(len(days),dtype=float)
    adj=pd.DataFrame({'A':100.+3.*np.sin(x*.7),
                      'B':80.+5.*np.cos(x*.39),
                      'FUND':200.+4.*np.sin(x*.23)},index=days)
    volume=pd.DataFrame({'A':100.+(x%11)*13.,
                         'B':80.+(x%7)**2,
                         'FUND':200.+(x%17)*7.},index=days)
    meta=dict(dates=['2016-08-17','2016-08-31','2016-09-30'],
              symbols=[['A','B','FUND'] for _ in range(3)],etfs=['FUND'])
    return {'adj':adj,'volume':volume},meta,days.strftime('%Y-%m-%d').tolist()


def two_down():
    p=np.full(64,100.);p[1]=99.;p[3]=98.
    v=np.full(63,2.);v[0]=1.;v[2]=3.
    return p,v


class MeasureTests(unittest.TestCase):
    def test_two_down_hand_value_and_alignment(self):
        p,v=two_down();r=kernel.measure(p,v)
        self.assertEqual(r['down_indices'],[0,2])
        self.assertEqual(r['down_volumes'],[1.,3.])
        self.assertAlmostEqual(r['candidate'],.25,places=15)
        self.assertAlmostEqual(r['control'],1/126,places=15)
        self.assertEqual(r['candidate_mean'],2/3)
        self.assertEqual(r['dispersion_estimates'],2)
        self.assertTrue(r['source_available'] and r['available'])

    def test_strict_down_flats_and_no_down_support(self):
        for p,n in ((np.ones(64),0),(np.arange(1.,65.),0),(np.r_[2.,np.ones(63)],1)):
            r=kernel.measure(p,np.ones(63))
            self.assertTrue(r['source_available']);self.assertFalse(r['available'])
            self.assertEqual(r['down_count'],n)
            self.assertEqual(r['reason'],'insufficient_down_days')
            self.assertEqual(r['dispersion_estimates'],0)
            self.assertEqual(r['candidate_scaled'],[])
            self.assertIsNone(r['control_mean'])
            self.assertTrue(math.isnan(r['candidate']) and math.isnan(r['control']))

    def test_constant_positive_zero_and_control_is_not_down_fraction(self):
        p,v=two_down();r=kernel.measure(p,np.full(63,7.))
        self.assertEqual((r['candidate'],r['control']),(0.,0.))
        self.assertTrue(r['candidate_constant'] and r['control_constant'])
        self.assertEqual(r['candidate_scaled'],[1.,1.])
        v[:]=1.;v[1]=9.
        r=kernel.measure(p,v)
        self.assertEqual(r['candidate'],0.);self.assertGreater(r['control'],0.)
        self.assertEqual(r['down_count'],2)

    def test_all_down_same_raws_and_endpoint_included(self):
        p=np.arange(64.,0.,-1.);v=np.arange(1.,64.)
        r=kernel.measure(p,v)
        self.assertEqual(r['down_indices'],list(range(63)))
        self.assertEqual(r['candidate'],r['control'])
        self.assertEqual(r['candidate_scaled'],r['control_scaled'])
        p=np.ones(64)*100;p[1]=99;p[-1]=98
        r=kernel.measure(p,v)
        self.assertEqual(r['down_indices'],[0,62])
        self.assertEqual(r['down_volumes'],[1.,63.])

    def test_power_two_invariance_and_extreme_finite(self):
        p,v=two_down();base=kernel.measure(p,v)
        for scale in (2.**-900,2.**900):
            r=kernel.measure(p*scale,v*scale)
            for key in ('down_flags','down_indices','candidate','control',
                        'candidate_scaled','control_scaled'):
                self.assertEqual(r[key],base[key])
        p=np.full(64,np.finfo(float).max);p[1]=np.finfo(float).tiny;p[3]=np.nextafter(0.,1.)
        v=np.full(63,np.nextafter(0.,1.));v[0]=np.finfo(float).max
        r=kernel.measure(p,v)
        self.assertTrue(r['available'])
        self.assertTrue(math.isfinite(r['candidate']) and math.isfinite(r['control']))
        self.assertEqual(r['candidate'],1.)

    def test_bad_source_common_missing_and_priority(self):
        invalid=[(0.,'invalid'),(-1.,'invalid'),(np.nan,'invalid'),(np.inf,'invalid'),
                 (True,'nonnumeric'),(None,'nonnumeric'),('2','nonnumeric'),(1+0j,'nonnumeric')]
        for item,tag in invalid:
            for which,label,index in ((0,'adjusted_price',20),(1,'volume',20)):
                p,v=two_down();args=[p.tolist(),v.tolist()];args[which][index]=item
                r=kernel.measure(*args)
                self.assertFalse(r['source_available']);self.assertFalse(r['available'])
                self.assertEqual(r['reason'],tag+'_'+label)
                self.assertEqual(r['prices'],[]);self.assertEqual(r['down_flags'],[])
        p,v=two_down();p[2]=0;v[2]=np.nan
        self.assertEqual(kernel.measure(p,v)['reason'],'invalid_adjusted_price')

    def test_shape_hard_fail_and_arithmetic_nonfinite_hard_fail(self):
        p,v=two_down()
        for a,b in ((p[:-1],v),(p,v[:-1]),(p[:,None],v),(p,v[:,None]),(1.,v)):
            with self.assertRaises(ValueError):kernel.measure(a,b)
        with patch.object(kernel.math,'fsum',return_value=float('inf')):
            with self.assertRaises(FloatingPointError):kernel.measure(p,v)

    def test_source_inputs_immutable_and_selection_alignment(self):
        p,v=two_down();bp,bv=p.copy(),v.copy()
        baseline=kernel.measure(p,v)
        np.testing.assert_array_equal(p,bp);np.testing.assert_array_equal(v,bv)
        v[0],v[1]=v[1],v[0]
        r=kernel.measure(p,v)
        self.assertNotEqual(r['candidate'],baseline['candidate'])


class RankTests(unittest.TestCase):
    def test_average_ties_missing_and_singleton(self):
        np.testing.assert_array_equal(kernel.centered_rank(
            np.array([1.,2.,2.,4.,np.nan]),np.array([1,1,1,1,0],bool)),[-.5,0.,0.,.5,0.])
        for mask in ([False,False],[True,False],[True,True]):
            np.testing.assert_array_equal(kernel.centered_rank(np.ones(2),np.array(mask)),[0.,0.])
        with self.assertRaises(ValueError):
            kernel.centered_rank(np.array([np.nan]),np.array([True]))

    def test_permutation_and_rank_domain(self):
        p,m,c=fixture();v,_=kernel.build_features(p,m,c)
        perm=[2,0,1];m['symbols']=[[row[j] for j in perm] for row in m['symbols']]
        a,_=kernel.build_features(p,m,c)
        for key in v:np.testing.assert_array_equal(a[key],v[key][:,perm])


class BuildTests(unittest.TestCase):
    def test_schema_current_windows_counts_initial_etf(self):
        p,m,c=fixture();v,a=kernel.build_features(p,m,c)
        self.assertEqual(set(v),{'candidate','control','available','candidate_rank','control_rank'})
        self.assertTrue(v['available'].all())
        self.assertEqual(a['source_available_by_event'],[3]*3)
        self.assertEqual(a['available_by_event'],[3]*3)
        for key,factor in zip(kernel.COUNT_FIELDS,(64,63,63,2,2,2)):
            self.assertEqual(a[key],9*factor)
        self.assertEqual(a['predictive_model_fits'],0);self.assertEqual(a['new_model_fits'],0)
        t=c.index(m['dates'][0]);row=a['event_history'][0]
        self.assertEqual((row['price_start'],row['price_end']),(c[t-63],c[t]))
        self.assertEqual((row['volume_start'],row['volume_end']),(c[t-62],c[t]))
        self.assertEqual(row['price_dates'][1:],row['volume_dates'])
        self.assertTrue(row['cells'][2]['available'])
        self.assertEqual(row['cells'][2]['symbol'],'FUND')
        json.dumps(a,allow_nan=False)

    def test_complete_source_with_insufficient_down_counts(self):
        p,m,c=fixture();p['adj'].loc[:,'A']=100.
        v,a=kernel.build_features(p,m,c,limit=1)
        self.assertEqual(a['source_available_cells'],3);self.assertEqual(a['available_cells'],2)
        self.assertEqual(a['down_comparisons'],189);self.assertEqual(a['dispersion_estimates'],4)
        self.assertEqual(a['event_history'][0]['cells'][0]['reason'],'insufficient_down_days')
        self.assertTrue(np.isnan(v['candidate'][0,0]));self.assertEqual(v['candidate_rank'][0,0],0.)

    def test_all_prefixes_and_physical_source_truncation(self):
        p,m,c=fixture();full,fa=kernel.build_features(p,m,c)
        for n in range(4):
            q={key:frame.loc[:m['dates'][n-1]].copy() if n else frame.iloc[:0].copy() for key,frame in p.items()}
            values,a=kernel.build_features(q,m,c,limit=n)
            for key in full:np.testing.assert_array_equal(values[key],full[key][:n])
            self.assertEqual(a['event_history'],fa['event_history'][:n])
        self.assertEqual(values['candidate'].shape,(3,3))
        empty,a=kernel.build_features(p,m,c,limit=0)
        self.assertEqual(empty['candidate'].shape,(0,3));self.assertEqual(a['valid_price_observations'],0)

    def test_future_numeric_and_future_symbol_poison(self):
        p,m,c=fixture();base,ba=kernel.build_features(p,m,c,limit=1)
        bad={key:frame.astype(object) for key,frame in p.items()}
        for frame in bad.values():
            frame.loc[frame.index>m['dates'][0],:]=object()
            frame['FUTURE']='invalid future symbol'
        m['symbols'][1:]=[['A','B','FUTURE']]*2
        current,a=kernel.build_features(bad,m,c,limit=1)
        for key in base:np.testing.assert_array_equal(current[key],base[key])
        self.assertEqual(a,ba)
        kernel.build_features(bad,m,c,limit=0)

    def test_missing_dates_and_symbols_do_not_compress_window(self):
        for field,reason in (('adj','missing_official_price_session'),('volume','missing_official_volume_session')):
            p,m,c=fixture();day=c[c.index(m['dates'][0])-10]
            p[field]=p[field].drop(pd.Timestamp(day))
            v,a=kernel.build_features(p,m,c,limit=1)
            self.assertFalse(v['available'].any())
            self.assertEqual(a['reason_counts'],{reason:3})
            self.assertEqual(a['source_available_cells'],0)
        for field,reason in (('adj','missing_price_symbol'),('volume','missing_volume_symbol')):
            p,m,c=fixture();p[field]=p[field].drop(columns='B')
            v,a=kernel.build_features(p,m,c,limit=1)
            self.assertEqual(a['reason_counts'],{'available':2,reason:1})
            self.assertEqual(v['candidate_rank'][0,1],0.)

    def test_exact_volume_start_no_extra_price_or_volume_reads(self):
        p,m,c=fixture();t=c.index(m['dates'][0])
        p['volume']=p['volume'].astype(object);p['adj']=p['adj'].astype(object)
        p['volume'].loc[c[t-63],:]='not consumed'
        p['adj'].loc[c[t-64],:]='not consumed'
        v,_=kernel.build_features(p,m,c,limit=1);self.assertTrue(v['available'].all())
        p['volume'].loc[c[t-62],'A']='consumed bad'
        v,a=kernel.build_features(p,m,c,limit=1)
        self.assertFalse(v['available'][0,0]);self.assertEqual(a['reason_counts']['nonnumeric_volume'],1)

    def test_missing_source_value_and_single_available(self):
        p,m,c=fixture();p['volume'].loc[:,'B']=0.;p['adj'].loc[:,'FUND']=100.
        v,a=kernel.build_features(p,m,c)
        self.assertEqual(a['source_available_by_event'],[2]*3)
        self.assertEqual(a['available_by_event'],[1]*3)
        self.assertTrue(v['available'][:,0].all())
        self.assertTrue((v['candidate_rank']==0.).all() and (v['control_rank']==0.).all())
        json.dumps(a,allow_nan=False)

    def test_insufficient_calendar_history(self):
        p,m,c=fixture();m=dict(dates=['2016-01-29'],symbols=[['A']],etfs=[])
        v,a=kernel.build_features(p,m,c)
        self.assertEqual(a['reason_counts'],{'insufficient_history':1})
        self.assertFalse(v['available'].any());self.assertEqual(a['event_history'][0]['price_dates'],[])

    def test_no_input_mutation(self):
        p,m,c=fixture();before=copy.deepcopy((p,m,c));kernel.build_features(p,m,c)
        for key in p:pd.testing.assert_frame_equal(p[key],before[0][key])
        self.assertEqual(m,before[1]);self.assertEqual(c,before[2])

    def test_invalid_protocol_metadata_calendar_and_limit(self):
        p,m,c=fixture()
        for limit in (-1,4,True,1.):
            with self.assertRaises(ValueError):kernel.build_features(p,m,c,limit=limit)
        with self.assertRaises(ValueError):kernel.build_features(dict(p,close=p['adj']),m,c)
        bad=copy.deepcopy(m);bad['dates'][1]='2016-08-30'
        with self.assertRaises(ValueError):kernel.build_features(p,bad,c)
        with self.assertRaises(ValueError):
            kernel.build_features(p,m,c[:c.index(m['dates'][-1])+1])
        for key,value in (('etfs',['FUND','FUND']),('symbols',[['A','A']]*3),
                          ('symbols',[['A'],['A','B'],['A']])):
            bad=copy.deepcopy(m);bad[key]=value
            with self.assertRaises(ValueError):kernel.build_features(p,bad,c)
        p,m,c=fixture();p['adj']=p['adj'].iloc[::-1]
        with self.assertRaises(ValueError):kernel.build_features(p,m,c)
        p,m,c=fixture();p['volume'].index=p['volume'].index.tz_localize('UTC')
        with self.assertRaises(ValueError):kernel.build_features(p,m,c)


if __name__=='__main__':
    unittest.main(verbosity=2)

