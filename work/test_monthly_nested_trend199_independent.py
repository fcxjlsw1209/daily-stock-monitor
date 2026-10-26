"""Synthetic-only independent nested-trend and half-core artifact verification."""
import copy
import hashlib
import inspect
import math
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import monthly_nested_trend199_independent as independent


def fixture(count=3, width=4):
    index = pd.bdate_range('2014-01-01', periods=1000)
    x = np.arange(len(index), dtype=float)
    frame = pd.DataFrame({'A': np.exp(.0003*x+.08*np.sin(x/37.)),
        'B': 3000.-x, 'FUND': 2.+x/1000.+.2*np.sin(x/71.), 'D': np.full(len(index),8.)}, index=index)
    names = ['A','B','FUND','D'][:width]
    dates = ['2016-08-17','2016-08-31','2016-09-30'][:count]
    return {'adj':frame},dict(dates=dates,symbols=[names[:] for _ in dates],etfs=['FUND']),index.strftime('%Y-%m-%d').tolist()


def example_prices():
    p = np.ones(253)*10.
    p[0],p[63],p[126],p[252] = 4.,8.,16.,32.
    return p


def original_fixture(rows=3,width=4,shape='column'):
    calm=np.array([i%2==0 for i in range(rows)],bool)
    weights=np.where(calm,.5,.52)
    rank=(np.arange(width)+1)/max(width,1)-.5
    core=np.array([np.roll(rank,i) for i in range(rows)])*weights[:,None] if rows else np.empty((0,width))
    if shape=='matrix': weights=np.broadcast_to(weights[:,None],(rows,width)).copy()
    elif shape=='column': weights=weights[:,None]
    return {'core':core,'core_weight':weights,'state_calm10':calm,
            'eligible':np.zeros((rows,width),bool),'untouched':object()}


def targets(values,full=None):
    result={}
    n,width=values['available'].shape
    for arm,key in independent.RAW.items():
        value=np.full((n if full is None else full,width),np.nan)
        value[:n]=values[key]
        if n: value[0]=np.nan
        result[arm]=value
    return result


class ScalarTests(unittest.TestCase):
    def test_three_actual_intervals_shared_endpoint_handworked(self):
        p=example_prices();a=independent.measure_oracle(p)
        self.assertTrue(a['available'])
        self.assertEqual(a['start_prices'],{'126':16.,'189':8.,'252':4.})
        self.assertEqual(a['end_price'],32.)
        self.assertEqual([a['gain'+str(h)] for h in independent.HORIZONS],[1.,3.,7.])
        self.assertNotEqual(a['gain126'],math.log(2.))
        p[252]=16.;b=independent.measure_oracle(p)
        self.assertEqual([b['gain'+str(h)] for h in independent.HORIZONS],[0.,1.,3.])

    def test_full253_not_endpoints_only_invalid_and_nonnumeric(self):
        for index in (0,1,63,126,200,252):
            for value in (0.,-1.,np.inf,np.nan):
                p=example_prices();p[index]=value
                a=independent.measure_oracle(p)
                self.assertEqual(a['reason'],'invalid_adjusted_price')
                self.assertFalse(a['source_available']);self.assertIsNone(a['start_prices'])
        for value in (True,None,'3',complex(2,0)):
            p=example_prices().astype(object);p[200]=value
            self.assertEqual(independent.measure_oracle(p)['reason'],'nonnumeric_adjusted_price')
        with self.assertRaises(ValueError):independent.measure_oracle(np.ones(252))

    def test_binary64_simple_gain_extremes_unit_invariance_and_no_mutation(self):
        p=example_prices();before=p.copy();a=independent.measure_oracle(p)
        for scale in (2.**-500,2.**500):
            b=independent.measure_oracle(p*scale)
            for h in independent.HORIZONS:self.assertEqual(a['gain'+str(h)],b['gain'+str(h)])
        np.testing.assert_array_equal(p,before)
        p[0]=5e-324;p[252]=1e308
        with self.assertRaises(FloatingPointError):independent.measure_oracle(p)
        p=np.full(253,1e308);p[252]=5e-324
        a=independent.measure_oracle(p)
        self.assertEqual([a['gain'+str(h)] for h in independent.HORIZONS],[-1.,-1.,-1.])

    def test_original_average_pct_ties_singleton_and_empty(self):
        x=np.array([[1.,1.,3.,np.nan]])
        r=independent.original_pct_ranks(x,np.isfinite(x))
        np.testing.assert_array_equal(r,[[0.,0.,.5,np.nan]])
        x=np.array([[1.,1.,1.]])
        r=independent.original_pct_ranks(x,np.isfinite(x))
        self.assertTrue(np.all(r==2./3.-.5))
        np.testing.assert_array_equal(independent.original_pct_ranks(np.array([[2.]]),np.array([[True]])),[[.5]])
        np.testing.assert_array_equal(independent.original_pct_ranks(np.array([[np.nan]]),np.array([[False]])),[[np.nan]])
        for row in ([3.,1.,1.,5.],[0.,0.,0.],[-1.],[-1.,0.,1.]):
            values=np.array([row]);mask=np.ones_like(values,bool)
            np.testing.assert_array_equal(independent.original_pct_ranks(values,mask)[0],pd.Series(row).rank(pct=True).to_numpy()-.5)

    def test_min_mean_same_horizon_domain_no_second_rank(self):
        q,c,e=independent.aggregate_ranks([.5,-.25,.125])
        self.assertEqual(q,-.25);self.assertEqual(c,math.fsum([.5,-.25,.125])/3);self.assertFalse(e)
        value=2./3.-.5
        q,c,e=independent.aggregate_ranks([value]*3)
        self.assertTrue(e);self.assertEqual(np.float64(q).tobytes(),np.float64(c).tobytes())
        self.assertEqual(c,value)
        with self.assertRaises(ValueError):independent.aggregate_ranks([.5,np.nan,.5])


class BuildTests(unittest.TestCase):
    def test_full_audit_horizon_dates_counters_and_etf_domain(self):
        p,m,c=fixture();before=p['adj'].copy();v,a=independent.independent_features(p,m,c)
        r=independent.verify_features(p,m,c,v,a)
        self.assertEqual(r['source_available_by_event'],[4,4,4]);self.assertEqual(r['available_by_event'],[4,4,4])
        self.assertEqual((r['valid_price_observations'],r['horizon_returns'],r['ranked_horizon_observations'],r['aggregated_scores']),(253*12,36,36,24))
        self.assertTrue(v['available'][:,2].all())
        for event in a['event_history']:
            t=c.index(event['date'])
            self.assertEqual((event['price_start'],event['price_end']),(c[t-294],c[t-42]))
            self.assertEqual(event['horizon_start_dates'],{'126':c[t-168],'189':c[t-231],'252':c[t-294]})
        pd.testing.assert_frame_equal(p['adj'],before)

    def test_common_mask_drops_entire_price_path_before_every_rank(self):
        p,m,c=fixture(1);t=c.index(m['dates'][0]);p['adj'].loc[c[t-200],'A']=np.nan
        v,a=independent.independent_features(p,m,c)
        self.assertEqual(a['available_by_event'],[3])
        for key in set(v)-{'available'}:self.assertTrue(np.isnan(v[key][0,0]))
        for h in independent.HORIZONS:
            raw=v['gain'+str(h)][0,1:]
            np.testing.assert_array_equal(v['rank'+str(h)][0,1:],pd.Series(raw).rank(pct=True).to_numpy()-.5)
        independent.verify_features(p,m,c,v,a)

    def test_exact_all_ties_singleton_and_empty_domain(self):
        p,m,c=fixture(1,3);p['adj'][:]=10.
        v,a=independent.independent_features(p,m,c)
        self.assertTrue(np.all(v['candidate']==2./3.-.5))
        np.testing.assert_array_equal(v['candidate'],v['control'])
        self.assertTrue(all(x['exact_equal_ranks'] for x in a['event_history'][0]['cells']))
        p,m,c=fixture(1,1);v,a=independent.independent_features(p,m,c)
        self.assertEqual((v['candidate'][0,0],v['control'][0,0]),(.5,.5))
        p['adj'][:]=np.nan;v,a=independent.independent_features(p,m,c)
        self.assertEqual(a['available_cells'],0);self.assertTrue(np.isnan(v['candidate']).all())

    def test_all_prefixes_actual_cutoff_future_poison_and_zero_prefix(self):
        p,m,c=fixture();v,a=independent.independent_features(p,m,c)
        for n in range(4):
            stop=pd.Timestamp.min if not n else pd.Timestamp(c[c.index(m['dates'][n-1])-42])
            q={'adj':p['adj'].astype(object).copy()};q['adj'].loc[q['adj'].index>stop,:]='future poison'
            vv,aa=independent.independent_features(q,m,c,n)
            independent.verify_features(q,m,c,vv,aa,n)
            for key in v:np.testing.assert_array_equal(vv[key],v[key][:n])
            self.assertEqual(aa['event_history'],a['event_history'][:n])
            q['adj']=q['adj'].loc[:stop];vvv,aaa=independent.independent_features(q,m,c,n)
            self.assertEqual(aa,aaa)
            for key in vv:np.testing.assert_array_equal(vv[key],vvv[key])
        self.assertEqual(independent.independent_features(p,m,c,0)[0]['candidate'].shape,(0,4))

    def test_structural_priority_monthend_calendar_and_metadata(self):
        p,m,c=fixture(1);t=c.index(m['dates'][0]);p['adj']=p['adj'].drop(columns='A').drop(pd.Timestamp(c[t-200]))
        _,a=independent.independent_features(p,m,c)
        self.assertEqual(a['reason_counts'],{'missing_symbol':1,'missing_official_session':3})
        _,a=independent.independent_features(p,m,[x for x in c if x>='2016-01-01'])
        self.assertEqual(a['reason_counts'],{'insufficient_history':4})
        p,m,c=fixture()
        for limit in (-1,4,True,1.5):
            with self.assertRaises(ValueError):independent.independent_features(p,m,c,limit)
        bad=copy.deepcopy(m);bad['dates'][1]='2016-08-30'
        with self.assertRaises(ValueError):independent.independent_features(p,bad,c)
        with self.assertRaises(ValueError):independent.independent_features(p,m,c[:c.index(m['dates'][-1])+1])
        bad=dict(m);bad.pop('etfs')
        with self.assertRaises(ValueError):independent.independent_features(p,bad,c)
        with self.assertRaises(ValueError):independent.independent_features({'close':p['adj']},m,c)

    def test_wrong_gains_ranks_aggregate_mask_audit_rejected_exactly(self):
        p,m,c=fixture();v,a=independent.independent_features(p,m,c)
        for kind in ('gain','rank','candidate','control','mask','date','start','count','type','identity'):
            vv,aa=copy.deepcopy(v),copy.deepcopy(a)
            if kind=='gain':vv['gain126'][0,0]=np.nextafter(vv['gain126'][0,0],math.inf)
            elif kind=='rank':vv['rank126'][0,0]+=1e-15
            elif kind=='candidate':vv['candidate'][0,0]+=1e-15
            elif kind=='control':vv['control'][0,0]+=1e-15
            elif kind=='mask':vv['available'][0,0]=False
            elif kind=='date':aa['event_history'][0]['horizon_start_dates']['126']=aa['event_history'][0]['price_start']
            elif kind=='start':aa['event_history'][0]['cells'][0]['start_prices']['126']*=2
            elif kind=='count':aa['aggregated_scores']+=1
            elif kind=='type':aa['horizon_returns']=float(aa['horizon_returns'])
            else:aa['event_history'][0]['cells'][0]['exact_equal_ranks']=not aa['event_history'][0]['cells'][0]['exact_equal_ranks']
            with self.assertRaises(ValueError,msg=kind):independent.verify_features(p,m,c,vv,aa)


class IntegrationTests(unittest.TestCase):
    def test_half_core_formula_missing_keeps_core_firstpartial_and_no_eligibility_filter(self):
        p,m,c=fixture();t=c.index(m['dates'][1]);p['adj'].loc[c[t-200],'A']=np.nan
        v,_=independent.independent_features(p,m,c);o=original_fixture();before=copy.deepcopy(o)
        ts=targets(v);d=independent.independent_deltas(v,o,ts);r=independent.verify_deltas(v,o,ts,d)
        self.assertEqual(r['max_absolute_delta'],.26)
        for arm in independent.RAW:
            np.testing.assert_array_equal(d[arm][0],np.zeros(4))
            self.assertEqual(d[arm][1,0],0.)
            mask=np.isfinite(ts[arm]);cw=np.broadcast_to(o['core_weight'],o['core'].shape)
            np.testing.assert_array_equal(d[arm][mask],.5*(cw[mask]*ts[arm][mask]-o['core'][mask]))
            np.testing.assert_array_equal(d[arm][~mask],np.zeros(np.count_nonzero(~mask)))
        self.assertTrue(any(np.any(x!=0) for x in d.values()))
        for key in ('core','core_weight','state_calm10','eligible'):np.testing.assert_array_equal(o[key],before[key])

    def test_weight_shapes_state_binding_prefix_and_future_array_poison(self):
        p,m,c=fixture();v,_=independent.independent_features(p,m,c);ref=None
        for shape in ('vector','column','matrix'):
            o=original_fixture(shape=shape);d=independent.independent_deltas(v,o,targets(v))
            if ref is not None:
                for arm in d:np.testing.assert_array_equal(d[arm],ref[arm])
            ref=d
            for n in range(4):
                vv={k:x[:n] for k,x in v.items()};oo=copy.deepcopy(o)
                for key in ('core','core_weight','state_calm10'):
                    oo[key]=oo[key].astype(object);oo[key][n:]='future unread'
                ts=targets(vv,3);dd=independent.independent_deltas(vv,oo,ts)
                independent.verify_deltas(vv,oo,ts,dd)
                for arm in dd:np.testing.assert_array_equal(dd[arm],ref[arm][:n])

    def test_target_neutral_zero_differs_missing_and_no_second_rank(self):
        p,m,c=fixture();v,_=independent.independent_features(p,m,c);o=original_fixture();ts=targets(v)
        for arm in ts:
            bad=copy.deepcopy(ts);bad[arm][0]=0.
            with self.assertRaises(ValueError):independent.independent_deltas(v,o,bad)
            bad=copy.deepcopy(ts)
            bad[arm][1]=pd.Series(bad[arm][1]).rank(pct=True).to_numpy()-.5
            if not np.array_equal(bad[arm][1],ts[arm][1],equal_nan=True):
                with self.assertRaises(ValueError):independent.independent_deltas(v,o,bad)
        o['core_weight'][1,0]=.5
        with self.assertRaises(ValueError):independent.independent_deltas(v,o,ts)

    def test_invalid_delta_weights_core_shapes_and_target_suffix(self):
        p,m,c=fixture();v,_=independent.independent_features(p,m,c);o=original_fixture();ts=targets(v)
        d=independent.independent_deltas(v,o,ts);bad=copy.deepcopy(d);bad['nested_min_core'][1,1]+=1e-15
        with self.assertRaises(ValueError):independent.verify_deltas(v,o,ts,bad)
        for key,value in [('core',np.inf),('core_weight',np.nan)]:
            bad=copy.deepcopy(o);bad[key][1,0]=value
            with self.assertRaises(ValueError):independent.independent_deltas(v,bad,ts)
        for blend in (0.,1.,True):
            with self.assertRaises(ValueError):independent.independent_deltas(v,o,ts,blend)
        bad=copy.deepcopy(o);bad['state_calm10']=np.ones(3,int)
        with self.assertRaises(ValueError):independent.independent_deltas(v,bad,ts)
        bad=copy.deepcopy(o);bad['core_weight']=np.ones((3,2))*.5
        with self.assertRaises(ValueError):independent.independent_deltas(v,bad,ts)
        bad=copy.deepcopy(ts);bad['nested_min_core'][1,0]=np.inf
        with self.assertRaises(ValueError):independent.independent_deltas(v,o,bad)
        vv={k:x[:1] for k,x in v.items()};tt=targets(vv,3);tt['nested_min_core'][2,0]=0.
        with self.assertRaises(ValueError):independent.independent_deltas(vv,o,tt)

    def test_frozen_accounting_pin_and_no_producer_import(self):
        self.assertEqual(hashlib.sha256(independent.FROZEN_HELPER.read_bytes()).hexdigest(),independent.FROZEN_HELPER_SHA)
        self.assertTrue(callable(independent.accounting().audit_path))
        self.assertNotIn('import monthly_nested_trend199_kernel',inspect.getsource(independent))
        with patch.object(independent,'FROZEN_HELPER_SHA','0'*64):
            with self.assertRaises(ValueError):independent.accounting()

    def test_unique_both_producer_fixtures_full_audit_all_prefix_and_edge_support(self):
        import monthly_nested_trend199_kernel as producer
        import test_monthly_nested_trend199_kernel as producer_tests
        for p,m,c in (fixture(),producer_tests.fixture()):
            full_values,full_audit=producer.build_features(p,m,c)
            for n in range(len(m['dates'])+1):
                stop=pd.Timestamp.min if not n else pd.Timestamp(m['dates'][n-1])
                q={'adj':p['adj'].loc[:stop]}
                v,a=producer.build_features(q,m,c,limit=n)
                independent.verify_features(q,m,c,v,a,limit=n)
                for key in v:np.testing.assert_array_equal(v[key],full_values[key][:n])
                self.assertEqual(a['event_history'],full_audit['event_history'][:n])
                o=original_fixture(len(m['dates']),len(m['symbols'][0]));ts=targets(v,len(m['dates']))
                independent.verify_deltas(v,o,ts,independent.independent_deltas(v,o,ts))
        p,m,c=fixture(1);t=c.index(m['dates'][0]);days=c[t-294:t-41]
        for path in (example_prices(),np.ones(253),np.arange(1.,254.),np.arange(253.,0.,-1.)):
            q=copy.deepcopy(p)
            for name in q['adj']:q['adj'].loc[days,name]=path
            v,a=producer.build_features(q,m,c);independent.verify_features(q,m,c,v,a)
        for bad in (0.,np.nan,'bad'):
            q={'adj':p['adj'].astype(object).copy()};q['adj'].loc[days[100],'A']=bad
            v,a=producer.build_features(q,m,c);independent.verify_features(q,m,c,v,a)


if __name__=='__main__':
    unittest.main(verbosity=2)
