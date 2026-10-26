"""Offline invented paths and frozen-code integration only; no price loaders."""
from copy import deepcopy
import importlib.util
import json
import math
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import monthly_block_sharpe242_kernel as kernel


def fixture():
    dates=pd.bdate_range('2014-01-01','2017-01-05');x=np.arange(len(dates),dtype=float)
    names=['A','B','C','D','E','FUND']
    frame=pd.DataFrame({s:100.+(.01+.004*j)*x+(2.+j)*np.sin(x*(.07+.031*j)+j) for j,s in enumerate(names)},index=dates)
    meta=dict(dates=['2016-08-17','2016-08-31','2016-09-30','2016-10-31'],
        ends=['2016-08-31','2016-09-30','2016-10-31','2016-11-30'],symbols=[names.copy() for _ in range(4)],etfs=['FUND'])
    return {'adj':frame},meta,dates.strftime('%Y-%m-%d').tolist()


def physical(panel,meta,calendar,n):
    m=deepcopy(meta)
    for key in ('dates','ends','symbols'):m[key]=m[key][:n]
    if n:
        t=calendar.index(m['dates'][-1]);cal=calendar[:t+2];frame=panel['adj'].loc[:m['dates'][-1]].copy()
    else:cal=[];frame=panel['adj'].iloc[:0].copy()
    return {'adj':frame},m,cal


def study():
    path=kernel.ROOT/'outputs/monthly_local_linear70/study.py'
    spec=importlib.util.spec_from_file_location('block242_test_study',path)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj


def selection_fixture():
    p,m,cal=fixture();values,_=kernel.build_features(p,m,cal);s=study()
    shape=(4,6);z=np.zeros(shape);f=np.zeros(shape,bool)
    calm=np.array([True,False,True,False]);weak=np.array([False,False,False,True])
    core=np.tile([-.18,-.08,.02,.06,.12,.20],(4,1))
    a=dict(baseline=core.copy(),core=core.copy(),core_weight=np.where(calm,.5,.52)[:,None],
        eligible=np.ones(shape,bool),effects=np.tile([0.,0.,0.,.5,.3,.5],(4,1)),state_calm10=calm,state_weak_narrow=weak,
        pre_score=np.tile([.22,.20,.18,.175,.17,.16],(4,1)),old_ml4=z.copy(),rank_log_excess=z.copy(),
        extension_available=f.copy(),forecast_available=f.copy(),local_available=f.copy(),model_available=f.copy(),
        raw_local_constant=z.copy(),raw_local_linear=z.copy(),raw_source_log=z.copy())
    return s,a,m,values


def source_dates(meta,calendar,row=0):
    t=calendar.index(meta['dates'][row]);return pd.to_datetime(calendar[t-294:t-41])


def same(case,a,b):
    case.assertEqual(tuple(a),tuple(b))
    for k in a:
        case.assertEqual(a[k].shape,b[k].shape,k);case.assertEqual(a[k].dtype,b[k].dtype,k)
        np.testing.assert_array_equal(a[k],b[k],err_msg=k)


class MeasureTests(unittest.TestCase):
    def test_exact_daily_and_disjoint_block_moments_no_annualization(self):
        p=100.+np.arange(253.)+7*np.sin(np.arange(253.)*.11);got=kernel.measure(p)
        self.assertTrue(got['available'])
        for prefix,r in [('daily',[float(p[j+1])/float(p[j])-1. for j in range(252)]),
                         ('block',[float(p[j+21])/float(p[j])-1. for j in range(0,252,21)])]:
            total=math.fsum(r);mean=total/len(r);sq=math.fsum((q-mean)*(q-mean) for q in r);sd=math.sqrt(sq/(len(r)-1))
            self.assertEqual(got[prefix+'_return_sum'],total);self.assertEqual(got[prefix+'_mean'],mean)
            self.assertEqual(got[prefix+'_centered_square_sum'],sq);self.assertEqual(got[prefix+'_sample_vol'],sd)
            self.assertEqual(got[prefix+'_sharpe'],mean/sd)
            self.assertNotEqual(sd,math.sqrt(sq/len(r)))
        self.assertEqual(tuple(got),kernel.MEASURE_KEYS);self.assertNotEqual(got['daily_sharpe'],got['block_sharpe'])

    def test_interior_path_changes_daily_but_not_block_ratios(self):
        p=100.+np.arange(253.)+np.sin(np.arange(253.));before=kernel.measure(p)
        p[100]*=1.2;after=kernel.measure(p)
        for k in ('block_return_sum','block_mean','block_sample_vol','block_sharpe'):self.assertEqual(before[k],after[k])
        self.assertNotEqual(before['daily_sharpe'],after['daily_sharpe'])

    def test_block_endpoints_are_all_thirteen_fixed_prices(self):
        p=100.+np.arange(253.);base=kernel.measure(p)
        for j in range(0,253,21):
            q=p.copy();q[j]*=1.1
            self.assertNotEqual(kernel.measure(q)['block_sample_vol'],base['block_sample_vol'])

    def test_flat_and_constant_positive_growth_common_missing(self):
        for p in (np.ones(253),np.exp2(np.arange(253.))):
            z=kernel.measure(p);self.assertTrue(z['source_available']);self.assertFalse(z['available'])
            self.assertEqual(z['reason'],'zero_both_sample_volatility')
            self.assertEqual(z['daily_sample_vol'],0.);self.assertEqual(z['block_sample_vol'],0.)
            self.assertTrue(math.isnan(z['block_sharpe']) and math.isnan(z['daily_sharpe']))

    def test_flat_block_endpoints_but_daily_variation_common_missing(self):
        p=np.full(253,100.);p[1::21]=110.
        z=kernel.measure(p);self.assertEqual(z['reason'],'zero_block_sample_volatility')
        self.assertGreater(z['daily_sample_vol'],0.);self.assertEqual(z['block_sample_vol'],0.)
        self.assertTrue(math.isnan(z['daily_sharpe']));self.assertFalse(z['available'])

    def test_negative_means_and_zero_mean_are_valid(self):
        p=500.-np.arange(253.)+np.sin(np.arange(253.));z=kernel.measure(p)
        self.assertTrue(z['available']);self.assertLess(z['block_sharpe'],0.);self.assertLess(z['daily_sharpe'],0.)
        total,mean,ss,sd=kernel._moments([-.5,.5]);self.assertEqual(mean,0.);self.assertEqual(kernel._sharpe(mean,sd),0.)

    def test_type_before_finite_source_validation_all253(self):
        for bad,reason in [(True,'nonnumeric_adjusted_price'),('100','nonnumeric_adjusted_price'),(None,'nonnumeric_adjusted_price'),
                           (1+0j,'nonnumeric_adjusted_price'),(10**1000,'nonnumeric_adjusted_price'),(np.nan,'invalid_adjusted_price'),(np.inf,'invalid_adjusted_price'),
                           (0.,'invalid_adjusted_price'),(-1.,'invalid_adjusted_price')]:
            for j in (0,1,21,126,251,252):
                p=[100.]*253;p[j]=bad;z=kernel.measure(p)
                self.assertFalse(z['source_available']);self.assertEqual(z['reason'],reason)
        p=[np.nan]*253;p[-1]=True;self.assertEqual(kernel.measure(p)['reason'],'nonnumeric_adjusted_price')

    def test_shape_rejected(self):
        for p in ([1.]*252,np.ones((253,1)),1.):
            with self.assertRaises(ValueError):kernel.measure(p)

    def test_derived_overflow_and_underflow_hardfail(self):
        for p in ([1e-308]+[1e308]*252,[1e308]+[1e-308]*252,[1.]*120+[1e200]+[1.]*132):
            with self.assertRaises(FloatingPointError):kernel.measure(p)
        with self.assertRaises(FloatingPointError):kernel._moments([0.,1e-200])
        with self.assertRaises(FloatingPointError):kernel._moments([0.,np.nextafter(0.,1.)])
        with self.assertRaises(FloatingPointError):kernel._sharpe(1e-300,1e300)

    def test_power_two_scale_invariance_no_mutation(self):
        p=100.+np.arange(253.)+7*np.sin(np.arange(253.));before=p.copy();base=kernel.measure(p)
        for scale in (2.**-500,2.**500):
            got=kernel.measure(p*scale)
            for k in kernel.MEASURE_KEYS:
                if k not in ('start_price','end_price'):self.assertEqual(got[k],base[k],k)
        np.testing.assert_array_equal(p,before)


class BuildTests(unittest.TestCase):
    def test_schema_source_dates_full_counts_and_json(self):
        p,m,c=fixture();v,a=kernel.build_features(p,m,c)
        self.assertEqual(tuple(v),kernel.VALUE_KEYS);self.assertEqual(tuple(a),('schema','counts','reason_counts','event_history'))
        self.assertEqual(tuple(a['counts']),kernel.COUNT_FIELDS)
        expected=dict(events=4,symbols_per_event=6,checked_cells=24,source_available_cells=24,available_cells=24,unavailable_cells=0,
            valid_price_observations=6072,daily_returns=6048,block_returns=288,ranked_observations=48,zero_daily_volatility_cells=0,zero_block_volatility_cells=0,new_model_fits=0,predictive_model_fits=0)
        self.assertEqual(a['counts'],expected);json.dumps(a,allow_nan=False)
        for event in a['event_history']:
            self.assertEqual(tuple(event),kernel.EVENT_KEYS);t=event['session_position']
            self.assertEqual(event['source_dates'],c[t-294:t-41]);self.assertEqual(len(event['block_endpoint_dates']),13)
            self.assertTrue(all(tuple(cell)==kernel.CELL_KEYS for cell in event['cells']))

    def test_rank_average_pct_ties_singleton_empty(self):
        np.testing.assert_array_equal(kernel.percentile_rank(np.array([1.,2.,2.,4.,np.nan]),np.array([True]*4+[False])),[-.25,.125,.125,.5,np.nan])
        np.testing.assert_array_equal(kernel.percentile_rank(np.ones(3),np.ones(3,bool)),np.full(3,2/3-.5))
        np.testing.assert_array_equal(kernel.percentile_rank(np.array([1.]),np.array([True])),[.5])
        self.assertTrue(np.isnan(kernel.percentile_rank(np.ones(2),np.zeros(2,bool))).all())
        with self.assertRaises(ValueError):kernel.percentile_rank(np.ones(2),np.ones(2))

    def test_common_zero_mask_raws_and_ranks_nan_statistics_retained(self):
        p,m,c=fixture();d=source_dates(m,c);p['adj'].loc[d,'A']=100.
        p['adj'].loc[d,'B']=100.;p['adj'].loc[d[1::21],'B']=110.
        v,a=kernel.build_features(p,m,c,1)
        self.assertEqual(a['counts']['zero_daily_volatility_cells'],1);self.assertEqual(a['counts']['zero_block_volatility_cells'],2)
        for k in kernel.VALUE_KEYS[1:]:self.assertTrue(np.isnan(v[k][0,:2]).all())
        self.assertGreater(a['event_history'][0]['cells'][1]['daily_sample_vol'],0.)
        self.assertIsNone(a['event_history'][0]['cells'][1]['daily_sharpe'])
        self.assertEqual(a['counts']['source_available_cells'],6)

    def test_invalid_interior_source_marks_both_missing(self):
        p,m,c=fixture();p['adj'].loc[source_dates(m,c)[107],'C']=np.nan
        v,a=kernel.build_features(p,m,c,1)
        self.assertFalse(v['available'][0,2]);self.assertEqual(a['event_history'][0]['cells'][2]['reason'],'invalid_adjusted_price')
        self.assertEqual(a['counts']['source_available_cells'],5)

    def test_no_E_ETF_or_calm_feature_filter(self):
        p,m,c=fixture();v,a=kernel.build_features(p,m,c)
        m['eligible']=[[False]*6]*4;m['state_calm10']=[True]*4;m['etfs']=['A','B','C','D','E','FUND']
        vv,aa=kernel.build_features(p,m,c);same(self,v,vv);self.assertEqual(a,aa)
        self.assertTrue(v['available'].all())

    def test_all_source_ties_and_no_second_rank(self):
        p,m,c=fixture()
        for name in p['adj']:p['adj'][name]=p['adj']['A']
        v,a=kernel.build_features(p,m,c)
        for k in ('candidate','control'):np.testing.assert_array_equal(v[k],np.full((4,6),3.5/6-.5))

    def test_latest42_and_older_source_poison_ignored(self):
        p,m,c=fixture();v,a=kernel.build_features(p,m,c,1);t=c.index(m['dates'][0]);px=p['adj'].copy()
        px.iloc[t-41:]=np.nan;px.iloc[:t-294]=np.inf
        vv,aa=kernel.build_features({'adj':px},m,c,1);same(self,v,vv);self.assertEqual(a,aa)

    def test_structural_missing_precedence_and_official_window(self):
        p,m,c=fixture();d=source_dates(m,c);p['adj']=p['adj'].drop(index=d[100],columns='B')
        v,a=kernel.build_features(p,m,c,1)
        reasons=[cell['reason'] for cell in a['event_history'][0]['cells']]
        self.assertEqual(reasons,['missing_official_session','missing_symbol']+['missing_official_session']*4)
        self.assertEqual(a['event_history'][0]['missing_official_sessions'],[str(d[100].date())]);self.assertFalse(v['available'].any())

    def test_short_calendar_history_both_unavailable(self):
        p,m,c=fixture();c=c[c.index(m['dates'][0])-100:]
        v,a=kernel.build_features(p,m,c,1)
        self.assertEqual(a['reason_counts'],{'insufficient_history':6});self.assertEqual(a['event_history'][0]['source_dates'],[])

    def test_all_logical_physical_prefixes_and_future_poison(self):
        p,m,c=fixture();v,a=kernel.build_features(p,m,c)
        for n in range(5):
            logical,la=kernel.build_features(p,m,c,n);same(self,logical,{k:x[:n] for k,x in v.items()})
            pp,mm,cc=physical(p,m,c,n);pv,pa=kernel.build_features(pp,mm,cc)
            if n:same(self,logical,pv);self.assertEqual(la,pa)
            else:self.assertEqual(pv['available'].shape,(0,0));self.assertEqual(logical['available'].shape,(0,6))
            self.assertEqual(la['event_history'],a['event_history'][:n])
            if n<4:
                mm=deepcopy(m);mm['dates'][n:]=['bad']*(4-n);mm['symbols'][n:]=[[True]]*(4-n)
                if n:vv,_=kernel.build_features(p,mm,c,n);same(self,logical,vv)

    def test_bad_calendar_frame_queries_and_limit(self):
        p,m,c=fixture()
        for bad in (c+c[-1:],list(reversed(c)),['2026-10-24'],[]):
            with self.assertRaises(ValueError):kernel.build_features(p,m,bad)
        for change in ('duplicate_date','duplicate_symbol','nonmonthend'):
            mm=deepcopy(m)
            if change=='duplicate_date':mm['dates'][1]=mm['dates'][0]
            elif change=='duplicate_symbol':mm['symbols'][0][1]='A'
            else:mm['dates'][1]='2016-08-30'
            with self.assertRaises(ValueError):kernel.build_features(p,mm,c)
        for n in (True,-1,5,1.0):
            with self.assertRaises(ValueError):kernel.build_features(p,m,c,n)
        with self.assertRaises(ValueError):kernel.build_features({'adj':pd.concat([p['adj'],p['adj'].iloc[-1:]])},m,c)


class SelectionTests(unittest.TestCase):
    def test_original_pre_once_and_episode_once_each_with_full_audit(self):
        s,a,m,v=selection_fixture()
        with patch.object(s,'modified_pre_score',wraps=s.modified_pre_score) as pre,patch.object(s.ep,'select_boundary',wraps=s.ep.select_boundary) as ep:
            o,sc,p,au=kernel.decisions(s,a,m,v,4);self.assertEqual(pre.call_count,1);self.assertEqual(ep.call_count,3)
        self.assertEqual(tuple(o),kernel.ARMS);self.assertEqual(tuple(au['counts']),kernel.DECISION_COUNT_FIELDS)
        self.assertEqual(au['counts']['original_episode_calls'],3);self.assertEqual(au['counts']['active_events'],2)
        self.assertEqual(au['counts']['available_active_cells'],12);self.assertEqual(au['counts']['new_model_fits'],0)
        self.assertTrue(all(tuple(e)==kernel.DECISION_EVENT_KEYS for e in au['events']));json.dumps(au,allow_nan=False)
        old,fs=s.choices(s.configs()[4],a,m,4,policy=dict(s.POLICY_FIELDS))
        self.assertEqual(o[kernel.ARMS[0]],s.rp.make_orders(old,fs,m));np.testing.assert_array_equal(sc[kernel.ARMS[0]],fs)

    def test_half_core_delta_and_preserve_original_episode_support(self):
        s,a,m,v=selection_fixture();o,sc,p,au=kernel.decisions(s,a,m,v,4);base=s.modified_pre_score(s.configs()[4],a,4)
        for i in (1,3):
            for arm,key,dkey in ((kernel.ARMS[1],'candidate','candidate_delta'),(kernel.ARMS[2],'control','control_delta')):
                delta=.5*(.52*v[key][i]-a['core'][i]);expected=base[i].copy();nz=delta!=0.;expected[nz]+=delta[nz]
                np.testing.assert_array_equal(p[arm][i],expected);self.assertEqual(au['events'][i][dkey],delta.tolist())
        for i in (0,2):
            for arm in kernel.ARMS[1:]:self.assertEqual(o[arm][i],o[kernel.ARMS[0]][i]);np.testing.assert_array_equal(sc[arm][i],sc[kernel.ARMS[0]][i])

    def test_zero_target_valid_funded_replacement_and_missing_exact_fallback(self):
        s,a,m,v=selection_fixture();v['candidate'][1]=0.;o,sc,p,au=kernel.decisions(s,a,m,v,4)
        np.testing.assert_array_equal(np.asarray(au['events'][1]['candidate_delta']),-.5*a['core'][1])
        for k in kernel.VALUE_KEYS[1:]:v[k][3]=np.nan
        v['available'][3]=False
        o,sc,p,au=kernel.decisions(s,a,m,v,4)
        for arm in kernel.ARMS[1:]:self.assertEqual(o[arm][3],o[kernel.ARMS[0]][3]);np.testing.assert_array_equal(sc[arm][3],sc[kernel.ARMS[0]][3])

    def test_eligibility_never_expands_and_zero_one_two_rule(self):
        for count in range(3):
            s,a,m,v=selection_fixture();a['eligible'][1]=False;a['eligible'][1,:count]=True
            o,_,_,_=kernel.decisions(s,a,m,v,4)
            for arm in kernel.ARMS:self.assertEqual(len(o[arm][1]['symbols']),2 if count==2 else 0)

    def test_decision_all_logical_and_physical_prefixes(self):
        s,a,m,v=selection_fixture();o,sc,p,au=kernel.decisions(s,a,m,v,4)
        for n in range(5):
            oo,ss,pp,aa=kernel.decisions(s,a,m,v,n)
            self.assertEqual(oo,{k:x[:n] for k,x in o.items()});same(self,ss,{k:x[:n] for k,x in sc.items()});same(self,pp,{k:x[:n] for k,x in p.items()});self.assertEqual(aa['events'],au['events'][:n])
            am={k:x[:n].copy() for k,x in a.items()};mm=deepcopy(m)
            for k in ('dates','ends','symbols'):mm[k]=mm[k][:n]
            vv={k:x[:n].copy() for k,x in v.items()};po,ps,ppr,pa=kernel.decisions(s,am,mm,vv,n)
            self.assertEqual(oo,po);same(self,ss,ps);same(self,pp,ppr);self.assertEqual(aa,pa)

    def test_selection_future_numeric_and_metadata_poison_ignored(self):
        s,a,m,v=selection_fixture();before=kernel.decisions(s,a,m,v,1)
        for x in a.values():
            if x.dtype.kind=='f':x[1:]=np.nan
        for k in kernel.VALUE_KEYS[1:]:v[k][1:]=np.nan
        m['dates'][1:]=['bad']*3;m['ends'][1:]=['bad']*3;m['symbols'][1:]=[[True]]*3
        after=kernel.decisions(s,a,m,v,1);self.assertEqual(before[0],after[0]);same(self,before[1],after[1]);same(self,before[2],after[2]);self.assertEqual(before[3],after[3])

    def test_no_mutations_and_zero_delta_does_not_add(self):
        s,a,m,v=selection_fixture();aa={k:x.copy() for k,x in a.items()};vv={k:x.copy() for k,x in v.items()};mm=deepcopy(m)
        for key in ('candidate','control'):v[key][1]=a['core'][1]/.52
        o,sc,p,au=kernel.decisions(s,a,m,v,4)
        self.assertEqual(au['events'][1]['candidate_delta'],[0.]*6);self.assertEqual(o[kernel.ARMS[0]][1],o[kernel.ARMS[1]][1])
        same(self,a,aa);self.assertEqual(m,mm)
        for key in ('available','block_sharpe','daily_sharpe'):np.testing.assert_array_equal(v[key],vv[key])

    def test_bad_feature_masks_nonfinite_and_bound_rejected(self):
        for which in ('maskdtype','missing','infinite','outofrange','boolnumeric'):
            s,a,m,v=selection_fixture()
            if which=='maskdtype':v['available']=v['available'].astype(float)
            elif which=='missing':v['candidate'][1,0]=np.nan
            elif which=='infinite':v['block_sharpe'][1,0]=np.inf
            elif which=='outofrange':v['candidate'][1,0]=.50001
            else:v['candidate']=v['candidate'].astype(object);v['candidate'][1,0]=True
            with self.assertRaises(ValueError):kernel.decisions(s,a,m,v,4)

    def test_bad_core_coefficient_E_and_nonfinite_pre_fail(self):
        for which in ('coreweight','E','pre'):
            s,a,m,v=selection_fixture()
            if which=='coreweight':a['core_weight'][1]=.5
            elif which=='E':a['eligible']=a['eligible'].astype(float)
            else:a['pre_score'][1,0]=np.inf
            with self.assertRaises(ValueError):kernel.decisions(s,a,m,v,4)

    def test_dependency_pins_and_input_mutation_detection(self):
        self.assertTrue(all(p.is_file() for p in kernel.dependency_files()))
        s,a,m,v=selection_fixture();original=s.modified_pre_score
        def mutate(spec,arrays,n):
            result=original(spec,arrays,n);arrays['effects'][0,0]=.123;return result
        with patch.object(s,'modified_pre_score',side_effect=mutate):
            with self.assertRaises(ValueError):kernel.decisions(s,a,m,v,4)


if __name__=='__main__':unittest.main(verbosity=2)
