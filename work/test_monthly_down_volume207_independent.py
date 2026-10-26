"""Synthetic-only round207 oracle checks; no actual source or portfolio inputs."""
import copy
import hashlib
import inspect
import math
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import monthly_down_volume207_independent as independent


def price_path(down_indices):
    prices=[128.]
    for k in range(63):prices.append(prices[-1] + (-1. if k in down_indices else 0.))
    return np.asarray(prices)


def fixture(count=3,width=4):
    index=pd.bdate_range('2015-01-01',periods=600);x=np.arange(len(index))
    prices=pd.DataFrame({'A':128.+x%9, 'B':256.+(x*7%13), 'FUND':64.+(x//3%7),
                         'GROW':100.+x},index=index)
    volumes=pd.DataFrame({'A':1.+(x*11%41), 'B':10.+(x//5%3)*15,
                          'FUND':np.full(len(x),8.), 'GROW':1.+x%5},index=index)
    names=['A','B','FUND','GROW'][:width]
    dates=['2016-08-17','2016-08-31','2016-09-30'][:count]
    return {'adj':prices,'volume':volumes},dict(dates=dates,symbols=[names[:] for _ in dates],etfs=['FUND']),index.strftime('%Y-%m-%d').tolist()


class ScalarTests(unittest.TestCase):
    def test_handworked_two_down_cv_squared_and_matched_control(self):
        p=price_path([0,62]);v=np.ones(63);v[-1]=3.
        r=independent.measure_oracle(p,v)
        self.assertTrue(r['available']);self.assertEqual(r['down_count'],2)
        self.assertEqual(r['down_indices'],[0,62]);self.assertEqual(r['down_volumes'],[1.,3.])
        self.assertAlmostEqual(r['candidate'],.25,places=15)
        self.assertAlmostEqual(r['control'],63*(62+9)/65**2-1,places=14)
        self.assertEqual(r['dispersion_estimates'],2)
        self.assertGreater(r['candidate'],r['control'])

    def test_same_all_volume_dispersion_different_down_assignment(self):
        p=price_path([0,1]);v=np.ones(63);v[1]=9.
        a=independent.measure_oracle(p,v);v[[1,2]]=v[[2,1]]
        b=independent.measure_oracle(p,v)
        self.assertEqual(a['control'],b['control']);self.assertGreater(a['candidate'],b['candidate'])
        self.assertEqual(b['candidate'],0.)
        self.assertEqual(a['down_count'],b['down_count'])

    def test_no_or_one_down_is_missing_not_constant_zero(self):
        for flags in ([],[0]):
            r=independent.measure_oracle(price_path(flags),np.ones(63))
            self.assertTrue(r['source_available']);self.assertFalse(r['available'])
            self.assertEqual(r['reason'],'insufficient_down_days');self.assertIsNone(r['candidate'])
            self.assertEqual(r['dispersion_estimates'],0);self.assertEqual(r['down_count'],len(flags))
        z=independent.measure_oracle(price_path([1,2]),np.ones(63))
        self.assertTrue(z['available']);self.assertEqual(z['candidate'],0.);self.assertEqual(z['control'],0.)
        self.assertTrue(z['candidate_constant'] and z['control_constant'])

    def test_strict_negative_flats_and_direct_extreme_price_comparison(self):
        p=np.ones(64);p[1]=np.nextafter(1.,0.);p[2]=1.;p[63]=np.nextafter(1.,0.)
        z=independent.measure_oracle(p,np.ones(63))
        self.assertEqual(z['down_indices'],[0,62])
        p=np.array([np.finfo(float).max if k%2 else np.nextafter(0.,1.) for k in range(64)])
        z=independent.measure_oracle(p,np.ones(63));self.assertTrue(z['available'])
        self.assertEqual(z['down_count'],31) # comparisons require no overflowing price ratios

    def test_all_down_is_exact_same_statistic_and_order_insensitivity(self):
        p=price_path(range(63));v=np.arange(1.,64.)
        z=independent.measure_oracle(p,v)
        self.assertEqual(z['candidate'],z['control'])
        p=price_path([0,1,2,10]);a=independent.measure_oracle(p,v)
        v[[0,10]]=v[[10,0]];b=independent.measure_oracle(p,v)
        self.assertEqual(a['candidate'],b['candidate']);self.assertEqual(a['control'],b['control'])

    def test_constant_unit_invariance_and_within_window_jump_caveat(self):
        p=price_path([0,2,4,8]);v=np.arange(1.,64.);before=v.copy()
        a=independent.measure_oracle(p,v)
        for power in (-500,500):
            b=independent.measure_oracle(p*2.**power,v*2.**power)
            for key in ('candidate','control','candidate_scaled','control_scaled','candidate_mean','control_mean','down_flags'):
                self.assertEqual(a[key],b[key])
        np.testing.assert_array_equal(v,before)
        flat=np.ones(63);jump=flat.copy();jump[2:]*=100.
        self.assertEqual(independent.measure_oracle(p,flat)['candidate'],0.)
        self.assertGreater(independent.measure_oracle(p,jump)['candidate'],0.)

    def test_extreme_finite_volumes_stable_and_pairwise_identity(self):
        for v in ([1.,3.],[1.,1.,2.],[1e308,np.nextafter(0.,1.)]*31+[1e308],
                  [1.,np.nextafter(1.,2.)]*31+[1.], [1e-300]*63):
            result,detail=independent.cv_squared(v)
            z=detail['scaled'];pair=math.fsum((a-b)**2 for i,a in enumerate(z) for b in z[:i])/(math.fsum(z)**2)
            self.assertAlmostEqual(result,pair,places=12);self.assertTrue(math.isfinite(result));self.assertGreaterEqual(result,0.)
        r=independent.measure_oracle(price_path(range(63)),np.array([1e308]+[1e-300]*62))
        self.assertTrue(r['available']);self.assertAlmostEqual(r['candidate'],62.,places=12)

    def test_source_invalid_types_whole_windows_and_shape_priority(self):
        for index in (0,31,63):
            for value in (0.,-1.,np.nan,np.inf):
                p=price_path([1,2]);p[index]=value
                r=independent.measure_oracle(p,np.ones(63));self.assertFalse(r['source_available'])
                self.assertEqual(r['reason'],'invalid_adjusted_price');self.assertEqual(r['prices'],[])
        for value in (True,None,'1',1+0j):
            p=price_path([1,2]).astype(object);p[10]=value
            self.assertEqual(independent.measure_oracle(p,np.ones(63))['reason'],'nonnumeric_adjusted_price')
            v=np.ones(63,object);v[20]=value
            self.assertEqual(independent.measure_oracle(price_path([1,2]),v)['reason'],'nonnumeric_volume')
        for value in (0.,-1.,np.nan,np.inf):
            v=np.ones(63);v[30]=value
            self.assertEqual(independent.measure_oracle(price_path([1,2]),v)['reason'],'invalid_volume')
        with self.assertRaises(ValueError):independent.measure_oracle(np.zeros(64),np.ones(62))


class FeatureTests(unittest.TestCase):
    def test_current_clocks_source_counts_etfs_and_immutable_inputs(self):
        p,m,c=fixture();before=copy.deepcopy(p);v,a=independent.independent_features(p,m,c)
        receipt=independent.verify_features(p,m,c,v,a)
        self.assertTrue(all(receipt[k] for k in independent.FEATURE_FLAGS))
        self.assertEqual(receipt['source_available_cells'],12);self.assertEqual(receipt['available_cells'],9)
        self.assertEqual([receipt[k] for k in independent.COUNTERS],[768,756,756,18,18,18])
        self.assertTrue(v['available'][:,2].all());self.assertFalse(v['available'][:,3].any())
        for row in a['event_history']:
            t=c.index(row['date'])
            self.assertEqual(row['price_dates'],c[t-63:t+1]);self.assertEqual(row['volume_dates'],c[t-62:t+1])
        for key in p:pd.testing.assert_frame_equal(p[key],before[key])

    def test_all_prefixes_physical_source_truncation_and_future_object_poison(self):
        p,m,c=fixture();v,a=independent.independent_features(p,m,c)
        for n in range(4):
            cutoff=pd.Timestamp(m['dates'][n-1]) if n else pd.Timestamp.min
            q={k:f.astype(object) for k,f in p.items()}
            for f in q.values():f.loc[f.index>cutoff,:]='future poison'
            vv,aa=independent.independent_features(q,m,c,n)
            independent.verify_features(q,m,c,vv,aa,n)
            for key in v:np.testing.assert_array_equal(vv[key],v[key][:n])
            self.assertEqual(aa['event_history'],a['event_history'][:n])
            q={k:f.loc[:cutoff] for k,f in q.items()};pv,pa=independent.independent_features(q,m,c,n)
            for key in v:np.testing.assert_array_equal(pv[key],vv[key])
            self.assertEqual(pa,aa)
        self.assertEqual(independent.independent_features(p,m,c,0)[0]['available'].shape,(0,4))

    def test_missing_price_volume_sessions_columns_and_history_priority(self):
        p,m,c=fixture(count=1);t=c.index(m['dates'][0])
        for frame,day,reason in (('adj',c[t-63],'missing_official_price_session'),
                                 ('volume',c[t-62],'missing_official_volume_session')):
            q=copy.deepcopy(p);q[frame]=q[frame].drop(pd.Timestamp(day));v,a=independent.independent_features(q,m,c)
            self.assertFalse(v['available'].any());self.assertEqual(a['reason_counts'],{reason:4})
        q=copy.deepcopy(p);q['adj']=q['adj'].drop(columns='A');q['volume']=q['volume'].drop(columns='B')
        _,a=independent.independent_features(q,m,c)
        self.assertEqual([x['reason'] for x in a['event_history'][0]['cells'][:2]],['missing_price_symbol','missing_volume_symbol'])
        short=[day for day in c if day>='2016-07-01'];_,a=independent.independent_features(q,m,short)
        self.assertEqual(a['reason_counts'],{'insufficient_history':4})
        self.assertEqual(a['event_history'][0]['price_dates'],[])

    def test_monthend_shape_and_unconsumed_column_rules(self):
        p,m,c=fixture()
        for limit in (-1,4,True,1.5):
            with self.assertRaises(ValueError):independent.independent_features(p,m,c,limit)
        mm=copy.deepcopy(m);mm['dates'][1]='2016-08-30'
        with self.assertRaises(ValueError):independent.independent_features(p,mm,c)
        with self.assertRaises(ValueError):independent.independent_features(p,m,c[:c.index(m['dates'][-1])+1])
        for key in p:p[key]['UNUSED']='not consumed'
        independent.independent_features(p,m,c)
        bad=copy.deepcopy(p);bad['adj']=bad['adj'].iloc[::-1]
        with self.assertRaises(ValueError):independent.independent_features(bad,m,c)
        mm=copy.deepcopy(m);mm.pop('etfs')
        with self.assertRaises(ValueError):independent.independent_features(p,mm,c)

    def test_corrupt_source_stats_masks_counts_and_exact_rank_rejected(self):
        p,m,c=fixture();v,a=independent.independent_features(p,m,c)
        for change in ('price','flag','count','scale','cv','rank','mask','counter','date'):
            vv,aa=copy.deepcopy(v),copy.deepcopy(a);cell=aa['event_history'][1]['cells'][0]
            if change=='price':cell['prices'][10]+=1.
            if change=='flag':cell['down_flags'][0]=not cell['down_flags'][0]
            if change=='count':cell['down_count']+=1
            if change=='scale':cell['candidate_scale']*=2
            if change=='cv':vv['candidate'][1,0]+=1e-15
            if change=='rank':vv['candidate_rank'][1,0]+=1e-15
            if change=='mask':vv['available'][1,0]=False
            if change=='counter':aa['dispersion_estimates']+=2
            if change=='date':aa['event_history'][1]['volume_start']=m['dates'][0]
            with self.subTest(change=change),self.assertRaises(ValueError):independent.verify_features(p,m,c,vv,aa)

    def test_both_producer_fixtures_full_audit_all_prefixes(self):
        import monthly_down_volume207_kernel as producer
        import test_monthly_down_volume207_kernel as tests
        for make in (fixture,tests.fixture):
            p,m,c=make();full,_=producer.build_features(p,m,c)
            for n in range(len(m['dates'])+1):
                stop=pd.Timestamp(m['dates'][n-1]) if n else pd.Timestamp.min
                truncated={k:f.loc[:stop] for k,f in p.items()}
                v,a=producer.build_features(truncated,m,c,limit=n)
                independent.verify_features(truncated,m,c,v,a,limit=n)
                for key in full:np.testing.assert_array_equal(v[key],full[key][:n])
            for frame,bad in (('adj',0.),('volume',0.),('volume','bad')):
                q=copy.deepcopy(p);q[frame]=q[frame].astype(object)
                q[frame].at[pd.Timestamp(m['dates'][0]),m['symbols'][0][0]]=bad
                v,a=producer.build_features(q,m,c);independent.verify_features(q,m,c,v,a)


class DeltaTests(unittest.TestCase):
    def test_original_symmetric_rank_and_full_support_no_eligibility_filter(self):
        raw=np.array([[1.,1.,3.,np.nan],[2.,np.nan,np.nan,np.nan],[1.,1.,1.,np.nan]])
        mask=np.isfinite(raw);r=independent.centered_ranks(raw,mask)
        np.testing.assert_array_equal(r,[[-.25,-.25,.5,0.],[0.,0.,0.,0.],[0.,0.,0.,0.]])
        v=dict(candidate=raw,control=raw.copy(),available=mask,candidate_rank=r,control_rank=r.copy())
        d=independent.independent_deltas(v);report=independent.verify_deltas(v,d)
        self.assertEqual(report['dose'],.025);self.assertEqual(report['max_absolute_delta'],.0125)
        self.assertTrue(report['no_eligibility_or_model_filter']);self.assertTrue(all(not x.any() for x in d.values()))

    def test_direct_linear_initial_zero_and_corruption(self):
        p,m,c=fixture();v,_=independent.independent_features(p,m,c);d=independent.independent_deltas(v)
        for arm,raw in independent.RAW.items():
            np.testing.assert_array_equal(d[arm][0],0.);np.testing.assert_array_equal(d[arm][1:],.025*v[raw+'_rank'][1:])
            self.assertTrue((abs(d[arm])<=.0125).all())
        independent.verify_deltas(v,d)
        bad=copy.deepcopy(d);bad[independent.ARMS[1]][1,0]+=1e-15
        with self.assertRaises(ValueError):independent.verify_deltas(v,bad)
        with self.assertRaises(ValueError):independent.independent_deltas(v,.01)
        with self.assertRaises(ValueError):independent.independent_deltas(v,True)

    def test_accounting_pins_and_no_producer_import(self):
        self.assertEqual(hashlib.sha256(independent.FROZEN_HELPER.read_bytes()).hexdigest(),independent.FROZEN_HELPER_SHA)
        source=inspect.getsource(independent)
        self.assertNotIn('import monthly_down_volume207_kernel',source)
        helper=independent.accounting();self.assertTrue(callable(helper.audit_path))
        baseline=independent.frozen_helper().ARMS
        def fake(*args):return dict(checked_paths=9,checked_slices=45,paths={a:{} for a in independent.ARMS})
        isolated=independent.frozen_helper()
        with patch.object(independent,'frozen_helper',return_value=isolated),patch.object(isolated,'audit_outcomes',side_effect=fake):
            report=independent.audit_outcomes(None,None,None,None,None,None)
        self.assertEqual(report['outcome_adapter_sha256'],independent.FROZEN_HELPER_SHA)
        self.assertEqual(independent.frozen_helper().ARMS,baseline)


if __name__=='__main__':unittest.main(verbosity=2)
