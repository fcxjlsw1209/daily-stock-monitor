"""Synthetic-only independent cumulative-return and tail-moment verification."""
import copy
import hashlib
import inspect
import math
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import monthly_tail_moment221_independent as independent


def fixture(count=4, width=5):
    index = pd.bdate_range('2014-01-01', periods=1100)
    x = np.arange(len(index), dtype=float)
    trend = np.exp(.0003*x+.08*np.sin(x/37.))
    frame = pd.DataFrame({'A':trend, 'B':3000.-x,
        'FUND':2.+x/1000.+.2*np.sin(x/71.), 'D':np.full(len(index),8.), 'E':trend.copy()}, index=index)
    names = ['A','B','FUND','D','E'][:width]
    dates = ['2016-08-17','2016-08-31','2016-09-30','2016-10-31'][:count]
    return {'adj':frame},dict(dates=dates,symbols=[names[:] for _ in dates],etfs=['FUND']),index.strftime('%Y-%m-%d').tolist()


def example_prices():
    p = np.full(253,2.)
    p[0:4] = [1.,2.,1.,2.]
    return p


def original_fixture(rows=4,width=5,shape='column'):
    calm=np.array([i%2==0 for i in range(rows)],bool)
    weights=np.where(calm,.5,.52)
    rank=(np.arange(width)+1)/max(width,1)-.5
    core=np.array([np.roll(rank,i) for i in range(rows)])*weights[:,None] if rows else np.empty((0,width))
    if shape=='matrix': weights=np.broadcast_to(weights[:,None],(rows,width)).copy()
    elif shape=='column': weights=weights[:,None]
    return {'core':core,'core_weight':weights,'state_calm10':calm,
            'eligible':np.zeros((rows,width),bool),'untouched':object()}


def targets(values,original,full=None):
    result={}
    n,width=values['available'].shape
    calm=np.asarray(original['state_calm10'][:n],bool)
    for arm,key in independent.RAW.items():
        value=np.full((n if full is None else full,width),np.nan)
        value[:n]=values[key]
        value[:n][calm]=np.nan
        if n: value[0]=np.nan
        result[arm]=value
    return result


def physical_prefix(args,n):
    p,m,c=args
    last = pd.Timestamp.min if not n else pd.Timestamp(m['dates'][n-1])
    mm=copy.deepcopy(m)
    for key in ('dates','ends','symbols'):
        if key in mm: mm[key]=mm[key][:n]
    return {'adj':p['adj'].loc[:last].copy()},mm,c


class ScalarTests(unittest.TestCase):
    def test_handworked_momentum_not_mean_and_all252_day_denominator(self):
        p=example_prices();a=independent.measure_oracle(p)
        d2=math.sqrt(.25/252.);d3=float(np.cbrt(.125/252.))
        self.assertTrue(a['source_available']);self.assertTrue(a['available'])
        self.assertEqual(a['momentum'],1.)
        self.assertEqual(a['negative_day_count'],1)
        self.assertEqual(a['downside_square_sum'],.25)
        self.assertEqual(a['downside_cube_sum'],.125)
        self.assertEqual(a['quadratic_risk'],d2);self.assertEqual(a['cubic_risk'],d3)
        self.assertEqual(a['raw_candidate'],1./d3);self.assertEqual(a['raw_control'],1./d2)
        self.assertNotEqual(a['raw_candidate'],(1.5/252.)/d3)
        self.assertNotEqual(a['quadratic_risk'],.5)
        self.assertNotEqual(a['cubic_risk'],.5)
        self.assertEqual((a['start_price'],a['end_price']),(1.,2.))

    def test_no_negative_returns_is_valid_source_not_common_available(self):
        for p,momentum in [(np.ones(253),0.), (np.exp2(np.arange(253,dtype=float)-126.),2.**252-1.)]:
            a=independent.measure_oracle(p)
            self.assertTrue(a['source_available']);self.assertFalse(a['available'])
            self.assertEqual(a['reason'],'no_negative_returns')
            self.assertEqual(a['momentum'],momentum)
            self.assertEqual(a['negative_day_count'],0)
            for key in ('downside_square_sum','downside_cube_sum','cubic_risk','quadratic_risk'):
                self.assertEqual(a[key],0.)
            self.assertIsNone(a['raw_candidate']);self.assertIsNone(a['raw_control'])

    def test_constant_negative_returns_are_available_even_without_centered_variance(self):
        a=independent.measure_oracle(np.exp2(-np.arange(253,dtype=float)))
        self.assertTrue(a['available']);self.assertEqual(a['negative_day_count'],252)
        self.assertEqual(a['downside_square_sum'],63.)
        self.assertEqual(a['downside_cube_sum'],31.5)
        self.assertEqual(a['quadratic_risk'],.5);self.assertEqual(a['cubic_risk'],float(np.cbrt(.125)))
        self.assertEqual(a['raw_candidate'],a['momentum']/a['cubic_risk'])
        self.assertEqual(a['raw_control'],a['momentum']/a['quadratic_risk'])

    def test_signed_and_zero_momentum_are_not_filtered_or_clipped(self):
        negative=np.ones(253);negative[1:]=.5
        a=independent.measure_oracle(negative)
        self.assertEqual(a['momentum'],-.5);self.assertTrue(a['available'])
        self.assertGreater(a['raw_candidate'],a['raw_control'])
        zero=np.ones(253);zero[1]=.5
        a=independent.measure_oracle(zero)
        self.assertEqual(a['momentum'],0.);self.assertTrue(a['available'])
        self.assertEqual(a['raw_candidate'],0.);self.assertEqual(a['raw_control'],0.)
        self.assertFalse(np.signbit(a['raw_candidate']))

    def test_cubic_weights_large_losses_without_conditional_day_denominator(self):
        paths=[]
        for losses in ([.5]+[0.]*251,[.25,.25]+[0.]*250):
            prices=[100.]
            for loss in losses:prices.append(prices[-1]*(1-loss))
            paths.append(independent.measure_oracle(np.array(prices)))
        for a in paths:
            self.assertGreaterEqual(a['cubic_risk']+1e-15,a['quadratic_risk'])
        self.assertGreater(paths[0]['cubic_risk']/paths[0]['quadratic_risk'],
                           paths[1]['cubic_risk']/paths[1]['quadratic_risk'])

    def test_full253_support_invalid_and_nonnumeric_not_endpoint_only(self):
        for index in (0,1,42,126,200,252):
            for value in (0.,-1.,np.inf,np.nan):
                p=example_prices();p[index]=value;a=independent.measure_oracle(p)
                self.assertEqual(a['reason'],'invalid_adjusted_price')
                self.assertFalse(a['source_available'])
                self.assertTrue(all(a[k] is None for k in ('momentum','cubic_risk','quadratic_risk','raw_candidate','negative_day_count')))
        for value in (True,None,'3',complex(2,0)):
            p=example_prices().astype(object);p[200]=value
            self.assertEqual(independent.measure_oracle(p)['reason'],'nonnumeric_adjusted_price')
        with self.assertRaises(ValueError):independent.measure_oracle(np.ones(252))

    def test_unit_invariance_no_mutation_and_nonfinite_valid_arithmetic_hard_fails(self):
        p=example_prices();before=p.copy();a=independent.measure_oracle(p)
        for scale in (2.**-500,2.**500):
            b=independent.measure_oracle(p*scale)
            for key in ('momentum','cubic_risk','quadratic_risk','raw_candidate','raw_control'):
                self.assertEqual(a[key],b[key])
        np.testing.assert_array_equal(p,before)
        p[0]=5e-324;p[-1]=1e308
        with self.assertRaises(FloatingPointError):independent.measure_oracle(p)
        p=np.ones(253);p[1]=5e-324
        with self.assertRaises(FloatingPointError):independent.measure_oracle(p)
        # A finite huge upside return is not squared/cubed by downside moments.
        p=np.ones(253);p[1]=1e308
        huge=independent.measure_oracle(p)
        self.assertTrue(huge['available']);self.assertEqual(huge['momentum'],0.)
        self.assertEqual(huge['downside_square_sum'],1.)
        self.assertEqual(huge['downside_cube_sum'],1.)

    def test_smallest_representable_negative_step_preserved_without_epsilon(self):
        p=np.ones(253);p[1:]=np.nextafter(1.,0.)
        a=independent.measure_oracle(p);loss=1.-np.nextafter(1.,0.)
        self.assertTrue(a['available']);self.assertEqual(a['negative_day_count'],1)
        self.assertEqual(a['downside_square_sum'],loss*loss)
        self.assertEqual(a['downside_cube_sum'],(loss*loss)*loss)
        self.assertGreater(a['cubic_risk'],0.);self.assertGreater(a['quadratic_risk'],0.)

    def test_pair_counter_original_pct_ties_singletons_and_empty(self):
        for row in ([3.,1.,1.,5.],[0.,0.,0.],[-1.],[-1.,0.,1.]):
            x=np.array([row]);mask=np.ones_like(x,bool)
            np.testing.assert_array_equal(independent.original_pct_ranks(x,mask)[0],pd.Series(row).rank(pct=True).to_numpy()-.5)
        x=np.array([[1.,1.,3.,np.nan]])
        np.testing.assert_array_equal(independent.original_pct_ranks(x,np.isfinite(x)),[[0.,0.,.5,np.nan]])
        np.testing.assert_array_equal(independent.original_pct_ranks(np.array([[2.]]),np.array([[True]])),[[.5]])
        np.testing.assert_array_equal(independent.original_pct_ranks(np.array([[np.nan]]),np.array([[False]])),[[np.nan]])


class BuildTests(unittest.TestCase):
    def test_full_audit_dates_counters_and_no_negative_source_mask(self):
        p,m,c=fixture();before=p['adj'].copy();v,a=independent.independent_features(p,m,c)
        r=independent.verify_features(p,m,c,v,a)
        self.assertEqual(r['source_available_by_event'],[5]*4)
        self.assertEqual(r['available_by_event'],[4]*4)
        self.assertEqual((r['valid_price_observations'],r['simple_returns'],r['downside_moment_evaluations'],r['normalized_momentum_scores'],r['ranked_observations']),
                         (253*20,252*20,2*20,2*16,2*16))
        for event in a['event_history']:
            t=c.index(event['date'])
            self.assertEqual((event['price_start'],event['first_return_date'],event['price_end']),
                             (c[t-294],c[t-293],c[t-42]))
            self.assertEqual(event['cells'][3]['reason'],'no_negative_returns')
        for key in independent.SOURCE_KEYS:self.assertTrue(np.isfinite(v[key][:,3]).all())
        for key in independent.TARGET_KEYS:self.assertTrue(np.isnan(v[key][:,3]).all())
        self.assertTrue(v['available'][:,2].all())
        self.assertTrue(all(r[x] for x in independent.FEATURE_FLAGS))
        self.assertEqual(r['numerical_tolerance'],0.)
        pd.testing.assert_frame_equal(p['adj'],before)

    def test_invalid_interior_removes_both_rank_domains_without_eligibility_filter(self):
        p,m,c=fixture(1);t=c.index(m['dates'][0]);p['adj'].loc[c[t-200],'A']=np.nan
        v,a=independent.independent_features(p,m,c)
        self.assertEqual((a['source_available_cells'],a['available_cells']),(4,3))
        for key in set(v)-{'available'}:self.assertTrue(np.isnan(v[key][0,0]))
        for raw,key in [('raw_candidate','candidate'),('raw_control','control')]:
            mask=v['available'][0]
            np.testing.assert_array_equal(v[key][0,mask],pd.Series(v[raw][0,mask]).rank(pct=True).to_numpy()-.5)
        independent.verify_features(p,m,c,v,a)

    def test_all_ties_nonzero_risk_singleton_zero_and_invalid_domains(self):
        p,m,c=fixture(1,3)
        for name in p['adj']:p['adj'][name]=p['adj']['A'].to_numpy()
        v,a=independent.independent_features(p,m,c)
        self.assertTrue(np.all(v['candidate']==2./3.-.5))
        np.testing.assert_array_equal(v['candidate'],v['control'])
        p,m,c=fixture(1,1);v,a=independent.independent_features(p,m,c)
        self.assertEqual((v['candidate'][0,0],v['control'][0,0]),(.5,.5))
        p['adj'][:]=7.;v,a=independent.independent_features(p,m,c)
        self.assertEqual((a['source_available_cells'],a['available_cells']),(1,0))
        p['adj'][:]=np.nan;v,a=independent.independent_features(p,m,c)
        self.assertEqual((a['source_available_cells'],a['available_cells']),(0,0))
        self.assertTrue(np.isnan(v['candidate']).all())

    def test_all_logical_and_physical_prefixes_recent42_and_future_poison(self):
        args=fixture();p,m,c=args;v,a=independent.independent_features(p,m,c)
        for n in range(len(m['dates'])+1):
            stop=pd.Timestamp.min if not n else pd.Timestamp(c[c.index(m['dates'][n-1])-42])
            q={'adj':p['adj'].astype(object).copy()};q['adj'].loc[q['adj'].index>stop,:]='future poison'
            vv,aa=independent.independent_features(q,m,c,n)
            independent.verify_features(q,m,c,vv,aa,n)
            for key in v:np.testing.assert_array_equal(vv[key],v[key][:n])
            self.assertEqual(aa['event_history'],a['event_history'][:n])
            qq,mm,cc=physical_prefix(args,n)
            vvv,aaa=independent.independent_features(qq,mm,cc)
            independent.verify_features(qq,mm,cc,vvv,aaa)
            for key in vvv:
                if n:np.testing.assert_array_equal(vvv[key],v[key][:n])
                else:self.assertEqual(vvv[key].shape,(0,0))
            self.assertEqual(aaa['event_history'],a['event_history'][:n])
        self.assertEqual(independent.independent_features(p,m,c,0)[0]['candidate'].shape,(0,5))

    def test_metadata_missing_source_priority_and_official_calendar(self):
        p,m,c=fixture(1);t=c.index(m['dates'][0]);p['adj']=p['adj'].drop(columns='A').drop(pd.Timestamp(c[t-200]))
        _,a=independent.independent_features(p,m,c)
        self.assertEqual(a['reason_counts'],{'missing_symbol':1,'missing_official_session':4})
        _,a=independent.independent_features(p,m,[x for x in c if x>='2016-01-01'])
        self.assertEqual(a['reason_counts'],{'insufficient_history':5})
        p,m,c=fixture()
        for limit in (-1,5,True,1.5):
            with self.assertRaises(ValueError):independent.independent_features(p,m,c,limit)
        bad=copy.deepcopy(m);bad['dates'][1]='2016-08-30'
        with self.assertRaises(ValueError):independent.independent_features(p,bad,c)
        with self.assertRaises(ValueError):independent.independent_features(p,m,c[:c.index(m['dates'][-1])+1])
        bad=dict(m);bad.pop('etfs')
        with self.assertRaises(ValueError):independent.independent_features(p,bad,c)
        with self.assertRaises(ValueError):independent.independent_features({'close':p['adj']},m,c)
        bad=copy.deepcopy(m);bad['symbols'][1][1]=bad['symbols'][1][0]
        with self.assertRaises(ValueError):independent.independent_features(p,bad,c)
        with self.assertRaises(ValueError):independent.independent_features(p,m,c+[c[-1]])

    def test_schema_rejects_bad_frame_calendar_and_string_price_is_missing(self):
        p,m,c=fixture(1)
        q={'adj':p['adj'].iloc[::-1]}
        with self.assertRaises(ValueError):independent.independent_features(q,m,c)
        q={'adj':p['adj'].copy()};q['adj'].index=q['adj'].index.tz_localize('UTC')
        with self.assertRaises(ValueError):independent.independent_features(q,m,c)
        with self.assertRaises(ValueError):independent.independent_features(p,m,['2014-01-04']+c)
        t=c.index(m['dates'][0]);q={'adj':p['adj'].astype(object).copy()};q['adj'].loc[c[t-200],'A']='1.0'
        _,a=independent.independent_features(q,m,c)
        self.assertEqual(a['reason_counts']['nonnumeric_adjusted_price'],1)

    def test_exact_value_rank_aggregate_mask_and_audit_mutations_rejected(self):
        p,m,c=fixture();v,a=independent.independent_features(p,m,c)
        for kind in ('momentum','cubic_risk','quadratic_risk','raw_candidate','raw_control',
                     'candidate','control','mask','date','start','count','type','negative_count'):
            vv,aa=copy.deepcopy(v),copy.deepcopy(a)
            if kind in independent.VALUE_KEYS and kind!='available':
                vv[kind][0,0]=np.nextafter(vv[kind][0,0],math.inf)
            elif kind=='mask':vv['available'][0,0]=False
            elif kind=='date':aa['event_history'][0]['first_return_date']=aa['event_history'][0]['price_start']
            elif kind=='start':aa['event_history'][0]['cells'][0]['start_price']*=2
            elif kind=='count':aa['downside_moment_evaluations']+=1
            elif kind=='type':aa['simple_returns']=float(aa['simple_returns'])
            else:aa['event_history'][0]['cells'][0]['negative_day_count']+=1
            with self.assertRaises(ValueError,msg=kind):independent.verify_features(p,m,c,vv,aa)
        vv=copy.deepcopy(v);vv['momentum'][0,3]=np.nan
        with self.assertRaises(ValueError):independent.feature_arrays(vv)
        vv=copy.deepcopy(v);vv['raw_candidate'][0,3]=.5
        with self.assertRaises(ValueError):independent.feature_arrays(vv)


class IntegrationTests(unittest.TestCase):
    def test_half_core_formula_missing_calm_initial_and_no_eligibility_filter(self):
        p,m,c=fixture();t=c.index(m['dates'][1]);p['adj'].loc[c[t-200],'A']=np.nan
        v,_=independent.independent_features(p,m,c);o=original_fixture();before=copy.deepcopy(o)
        ts=targets(v,o);d=independent.independent_deltas(v,o,ts);r=independent.verify_deltas(v,o,ts,d)
        self.assertEqual(r['max_absolute_delta'],.26)
        for arm in independent.RAW:
            np.testing.assert_array_equal(d[arm][o['state_calm10']],0.)
            np.testing.assert_array_equal(d[arm][0],0.)
            self.assertEqual(d[arm][1,0],0.)
            mask=np.isfinite(ts[arm]);cw=np.broadcast_to(o['core_weight'],o['core'].shape)
            np.testing.assert_array_equal(d[arm][mask],.5*(cw[mask]*ts[arm][mask]-o['core'][mask]))
            np.testing.assert_array_equal(d[arm][~mask],np.zeros(np.count_nonzero(~mask)))
        self.assertTrue(any(np.any(x!=0) for x in d.values()))
        for key in ('core','core_weight','state_calm10','eligible'):np.testing.assert_array_equal(o[key],before[key])

    def test_weight_shapes_prefix_and_future_array_poison(self):
        p,m,c=fixture();v,_=independent.independent_features(p,m,c);ref=None
        for shape in ('vector','column','matrix'):
            o=original_fixture(shape=shape);d=independent.independent_deltas(v,o,targets(v,o))
            if ref is not None:
                for arm in d:np.testing.assert_array_equal(d[arm],ref[arm])
            ref=d
            for n in range(5):
                vv={k:x[:n] for k,x in v.items()};oo=copy.deepcopy(o)
                for key in ('core','core_weight','state_calm10'):
                    oo[key]=oo[key].astype(object);oo[key][n:]='future unread'
                ts=targets(vv,oo,4);dd=independent.independent_deltas(vv,oo,ts)
                independent.verify_deltas(vv,oo,ts,dd)
                for arm in dd:np.testing.assert_array_equal(dd[arm],ref[arm][:n])

    def test_target_zero_differs_missing_no_second_rank_and_bad_delta(self):
        p,m,c=fixture();v,_=independent.independent_features(p,m,c);o=original_fixture();ts=targets(v,o)
        for arm in ts:
            for row in (0,2):
                bad=copy.deepcopy(ts);bad[arm][row]=0.
                with self.assertRaises(ValueError):independent.independent_deltas(v,o,bad)
            bad=copy.deepcopy(ts);bad[arm][1]=pd.Series(bad[arm][1]).rank(pct=True).to_numpy()-.5
            if not np.array_equal(bad[arm][1],ts[arm][1],equal_nan=True):
                with self.assertRaises(ValueError):independent.independent_deltas(v,o,bad)
        d=independent.independent_deltas(v,o,ts);d[independent.ARMS[1]][1,1]+=1e-15
        with self.assertRaises(ValueError):independent.verify_deltas(v,o,ts,d)
        o['core_weight'][1,0]=.5
        with self.assertRaises(ValueError):independent.independent_deltas(v,o,ts)

    def test_invalid_core_risk_state_shapes_blend_and_target_suffix(self):
        p,m,c=fixture();v,_=independent.independent_features(p,m,c);o=original_fixture();ts=targets(v,o)
        for key,value in [('core',np.inf),('core_weight',np.nan),('core',True),('core_weight','0.52')]:
            bad=copy.deepcopy(o);bad[key]=bad[key].astype(object);bad[key][1,0]=value
            with self.assertRaises(ValueError):independent.independent_deltas(v,bad,ts)
        for blend in (0.,1.,True):
            with self.assertRaises(ValueError):independent.independent_deltas(v,o,ts,blend)
        bad=copy.deepcopy(o);bad['state_calm10']=np.ones(4,int)
        with self.assertRaises(ValueError):independent.independent_deltas(v,bad,ts)
        bad=copy.deepcopy(o);bad['core_weight']=np.ones((4,2))*.5
        with self.assertRaises(ValueError):independent.independent_deltas(v,bad,ts)
        bad=copy.deepcopy(ts);bad[independent.ARMS[1]][1,0]=np.inf
        with self.assertRaises(ValueError):independent.independent_deltas(v,o,bad)
        vv={k:x[:1] for k,x in v.items()};tt=targets(vv,o,4);tt[independent.ARMS[1]][2,0]=0.
        with self.assertRaises(ValueError):independent.independent_deltas(vv,o,tt)

    def test_frozen_accounting_pin_no_producer_import_and_flags(self):
        self.assertEqual(hashlib.sha256(independent.FROZEN_HELPER.read_bytes()).hexdigest(),independent.FROZEN_HELPER_SHA)
        self.assertTrue(callable(independent.accounting().audit_path))
        self.assertNotIn('import monthly_tail_moment221_kernel',inspect.getsource(independent))
        self.assertEqual(len(independent.FEATURE_FLAGS),18)
        self.assertEqual(len(independent.DELTA_FLAGS),9)
        self.assertEqual(len(independent.COUNT_FIELDS),9)
        with patch.object(independent,'FROZEN_HELPER_SHA','0'*64):
            with self.assertRaises(ValueError):independent.accounting()

    def test_unique_both_fixtures_full_audit_physical_prefix_cross(self):
        import monthly_tail_moment221_kernel as producer
        import test_monthly_tail_moment221_kernel as producer_tests
        for args in (fixture(),producer_tests.fixture()):
            p,m,c=args
            full_values,full_audit=producer.build_features(p,m,c)
            for n in range(len(m['dates'])+1):
                q,mm,cc=physical_prefix(args,n)
                v,a=producer.build_features(q,mm,cc)
                independent.verify_features(q,mm,cc,v,a)
                for key in v:
                    if n:np.testing.assert_array_equal(v[key],full_values[key][:n])
                    else:self.assertEqual(v[key].shape,(0,0))
                self.assertEqual(a['event_history'],full_audit['event_history'][:n])
                # Full identity metadata plus a logical limit preserves original column width.
                vv,aa=producer.build_features(q,m,c,limit=n)
                independent.verify_features(q,m,c,vv,aa,limit=n)
                o=original_fixture(len(m['dates']),len(m['symbols'][0]))
                ts=targets(vv,o,len(m['dates']))
                independent.verify_deltas(vv,o,ts,independent.independent_deltas(vv,o,ts))
        p,m,c=fixture(1);t=c.index(m['dates'][0]);days=c[t-294:t-41]
        for path in (example_prices(),np.ones(253),np.exp2(np.arange(253.)-126.),np.arange(253.,0.,-1.)):
            q=copy.deepcopy(p)
            for name in q['adj']:q['adj'].loc[days,name]=path
            v,a=producer.build_features(q,m,c);independent.verify_features(q,m,c,v,a)
        for bad in (0.,np.nan,'bad'):
            q={'adj':p['adj'].astype(object).copy()};q['adj'].loc[days[100],'A']=bad
            v,a=producer.build_features(q,m,c);independent.verify_features(q,m,c,v,a)


if __name__=='__main__':
    unittest.main(verbosity=2)
