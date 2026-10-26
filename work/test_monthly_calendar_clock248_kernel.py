"""Offline synthetic prices/cache only. No saved research arrays or outcomes."""
from copy import deepcopy
import importlib.util
import json
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import monthly_calendar_clock248_kernel as kernel


def study():
    spec=importlib.util.spec_from_file_location('calendar248_synthetic_local67',kernel.ROOT/'outputs/monthly_local_linear70/study.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def make_review(meta,calendar,first,bounds=None):
    bounds=deepcopy(bounds or []);lookup={d:i for i,d in enumerate(calendar)};last={}
    for d in calendar:last[d[:7]]=d
    cells=[]
    for i,(day,names) in enumerate(zip(meta['dates'],meta['symbols'])):
        t=lookup[day];y,m=map(int,day[:7].split('-'));k=y*12+m-1
        anchors={} if day==kernel.INITIAL_PARTIAL else {'M-'+str(z):last[f'{(k-z)//12:04d}-{(k-z)%12+1:02d}'] for z in (2,12,14)}
        old=dict(t_minus_42=calendar[t-42],t_minus_252=calendar[t-252],t_minus_294=calendar[t-294],risk63_price_start=calendar[t-63],risk63_price_end=day,risk63_prices=64,risk63_return_intervals=63)
        for j,s in enumerate(names):
            for b in bounds:
                if s==b['symbol']:cells.append(dict(query_index=i,column_index=j,date=day,symbol=s,issuer_key=b['issuer_key'],not_before=b['not_before'],available_from=b['available_from'],bound_visible=b['available_from']<=day,calendar_anchors=anchors,old_clock_on_official_axis=old))
    return dict(schema='calendar_clock248_metadata_source_review_v1',panel_first_session=first,listing_bounds=bounds,identity_cells=cells)


def refresh_core(args):
    _,a,m,p,_,_=args;px=p['adj'].astype(float);r=px.pct_change(fill_method=None);sd=r.rolling(63).std()
    raw210=px.shift(42)/px.shift(252)-1.;risk=(px.shift(42)/px.shift(294)-1.)/sd.replace(0,np.nan)
    for i,(day,names) in enumerate(zip(m['dates'],m['symbols'])):
        raw=(raw210 if a['state_calm10'][i] else risk).loc[day,names].replace([np.inf,-np.inf],np.nan)
        a['core'][i]=float(a['core_weight'][i,0])*(raw.rank(pct=True).fillna(.5).to_numpy()-.5)
    a['baseline']=a['core']+.07
    return args


def fixture():
    s=study();shape=(3,6);names=list('ABCDEF');z=np.zeros(shape);f=np.zeros(shape,bool)
    calm=np.array([True,False,True]);cw=np.where(calm,.5,.52)[:,None]
    a=dict(baseline=z.copy(),core=z.copy(),core_weight=cw,eligible=np.ones(shape,bool),effects=np.tile([0.,0.,0.,.5,-.5,0.],(3,1)),
      state_calm10=calm,state_weak_narrow=np.array([False,True,False]),pre_score=np.tile([.31,.305,.30,.295,.29,.285],(3,1)),
      old_ml4=z.copy(),rank_log_excess=z.copy(),extension_available=f.copy(),forecast_available=f.copy(),local_available=f.copy(),model_available=f.copy(),
      raw_local_constant=z.copy(),raw_local_linear=z.copy(),raw_source_log=z.copy())
    m=dict(dates=['2016-08-17','2016-08-31','2016-09-30'],ends=['2016-08-31','2016-09-30','2016-10-31'],symbols=[names.copy() for _ in range(3)],etfs=['F'])
    dates=pd.bdate_range('2014-01-01','2016-11-03');x=np.arange(len(dates),dtype=float)
    px=pd.DataFrame({v:90.+(j+1)*.03*x+2*np.sin(x/(11+j))+1.1*np.cos(x/(17+j)) for j,v in enumerate(names)},index=dates)
    c=dates.strftime('%Y-%m-%d').tolist();p={'adj':px};review=make_review(m,c,c[0])
    return refresh_core((s,a,m,p,c,review))


def physical(args,n):
    s,a,m,p,c,r=args;mm=deepcopy(m)
    for k in ('dates','ends','symbols'):mm[k]=mm[k][:n]
    aa={k:np.asarray(v)[:n].copy() for k,v in a.items()}
    if n:
        last=mm['dates'][-1];pp={'adj':p['adj'].loc[:last].copy()};cc=c[:c.index(last)+2]
    else:pp={'adj':p['adj'].iloc[:0].copy()};cc=[]
    return s,aa,mm,pp,cc,deepcopy(r)


def run(args,n=None):return kernel.prepare(*args,len(args[2]['dates']) if n is None else n)


def same_arrays(case,a,b):
    case.assertEqual(tuple(a),tuple(b))
    for k in a:np.testing.assert_array_equal(a[k],b[k],err_msg=k)


def anchor(args,row=1,lag=None):
    day=args[2]['dates'][row];lag=lag or (12 if args[1]['state_calm10'][row] else 14);y,m=map(int,day[:7].split('-'));k=y*12+m-1-lag
    return pd.Timestamp([d for d in args[4] if d[:7]==f'{k//12:04d}-{k%12+1:02d}'][-1])


def bound(symbol='A',visible=True):
    return dict(symbol=symbol,issuer_key='issuer:'+symbol,not_before='2016-01-01',available_from='2016-01-02' if visible else '2027-01-01',evidence_id='synthetic_listing',kind='new_issue_bound')


class KernelTests(unittest.TestCase):
    def test_schema_original_once_per_arm(self):
        args=fixture();s=args[0]
        with patch.object(s,'modified_pre_score',wraps=s.modified_pre_score) as pre,patch.object(s.ep,'select_boundary',wraps=s.ep.select_boundary) as ep:
            f,o,sc,p,au=run(args)
            self.assertEqual((pre.call_count,ep.call_count),(1,2))
        self.assertEqual(tuple(o),kernel.ARMS);self.assertEqual(tuple(au['counts']),kernel.COUNT_FIELDS)
        self.assertEqual(au['counts']['original_episode_calls'],2);self.assertEqual(au['counts']['original_checked_cells'],12)
        self.assertEqual(au['counts']['new_model_fits'],0)
        self.assertTrue(all(tuple(e)==kernel.EVENT_KEYS for e in f));self.assertTrue(all(tuple(c)==kernel.CELL_KEYS for e in f for c in e['cells']))
        self.assertTrue(all(tuple(e)==kernel.AUDIT_EVENT_KEYS for e in au['events']));json.dumps([f,au],allow_nan=False)

    def test_exact_original_local67_control(self):
        args=fixture();s,a,m,*_=args;_,o,sc,p,_=run(args)
        choices,post=s.choices(s.configs()[4],a,m,3,policy=dict(s.POLICY_FIELDS))
        self.assertEqual(o[kernel.ARMS[0]],s.rp.make_orders(choices,post,m));np.testing.assert_array_equal(sc[kernel.ARMS[0]],post)
        np.testing.assert_array_equal(p[kernel.ARMS[0]],s.modified_pre_score(s.configs()[4],a,3))

    def test_cached_decisions_no_prices_exact(self):
        args=fixture();f,o,s,p,a=run(args)
        with patch.object(kernel,'_setup',side_effect=AssertionError('No source replay')):
            oo,ss,pp,aa=kernel.decisions(args[0],args[1],args[2],f,3)
        self.assertEqual((o,a),(oo,aa));same_arrays(self,s,ss);same_arrays(self,p,pp)

    def test_partial_no_numeric_source_reads(self):
        args=fixture();args[3]['adj']=args[3]['adj'].astype(object);args[3]['adj'].loc[:,:]='bad'
        f,o,s,p,a=run(args,1);self.assertEqual(f[0]['cells'],[]);self.assertEqual(a['counts']['original_checked_cells'],0)
        self.assertEqual(o[kernel.ARMS[0]],o[kernel.ARMS[1]]);np.testing.assert_array_equal(s[kernel.ARMS[0]],s[kernel.ARMS[1]])

    def test_n0_empty_source_calendar_and_two_episodes(self):
        args=physical(fixture(),0)
        with patch.object(args[0].ep,'select_boundary',wraps=args[0].ep.select_boundary) as ep:
            f,o,s,p,a=run(args)
            self.assertEqual(ep.call_count,2)
        self.assertEqual(f,[]);self.assertEqual(o,{k:[] for k in kernel.ARMS});self.assertEqual(a['counts']['events'],0)

    def test_all_logical_physical_prefixes(self):
        args=fixture();full=run(args)
        for n in range(4):
            left=run(args,n);right=run(physical(args,n));self.assertEqual(left[0],full[0][:n]);self.assertEqual(left[0],right[0]);self.assertEqual(left[1],right[1]);self.assertEqual(left[4],right[4]);self.assertEqual(left[4]['events'],full[4]['events'][:n])
            same_arrays(self,left[2],right[2]);same_arrays(self,left[3],right[3])
            cached=kernel.decisions(args[0],args[1],args[2],full[0],n)
            self.assertEqual(left[1],cached[0]);self.assertEqual(left[4],cached[3])

    def test_future_numeric_metadata_poison(self):
        args=fixture();expected=run(args,2)
        args[2]['dates'][2]='bad';args[2]['symbols'][2]=['bad'];args[2]['ends'][2]='bad'
        args[3]['adj']=args[3]['adj'].astype(object);args[3]['adj'].loc['2016-09-01':,:]='future_bad'
        for key in ('core','baseline','effects','core_weight','raw_local_linear'):
            args[1][key]=args[1][key].astype(object);args[1][key][2]='future_bad'
        actual=run(args,2);self.assertEqual(expected[0],actual[0]);self.assertEqual(expected[1],actual[1]);self.assertEqual(expected[4],actual[4])

    def test_full_history_origin_required(self):
        args=fixture();args[3]['adj']=args[3]['adj'].iloc[1:]
        with self.assertRaisesRegex(ValueError,'history origin'):run(args,2)

    def test_original_consumed_clock_missing_row_hardfail(self):
        args=fixture();t=args[4].index(args[2]['dates'][1]);args[3]['adj']=args[3]['adj'].drop(index=pd.Timestamp(args[4][t-100]))
        with self.assertRaisesRegex(ValueError,'clock mismatch'):run(args,2)

    def test_earlier_calendar_row_missing_whole_fallback(self):
        args=fixture();d=anchor(args);t=args[4].index(args[2]['dates'][1]);self.assertLess(str(d.date()),args[4][t-294])
        args[3]['adj']=args[3]['adj'].drop(index=d);f,o,_,_,au=run(args,2)
        self.assertEqual(f[1]['branch'],'missing_calendar');self.assertTrue(all(c['calendar_reason']=='missing_row' for c in f[1]['cells']));self.assertEqual(o[kernel.ARMS[0]],o[kernel.ARMS[1]])

    def test_missing_E_false_member_falls_back_full_query(self):
        args=fixture();args[1]['eligible'][1,-1]=False;args[3]['adj'].at[anchor(args),'F']=np.nan
        f,o,s,p,au=run(args,2);self.assertEqual(f[1]['u0_indices'],list(range(6)));self.assertEqual(f[1]['calendar_missing_indices'],[5]);self.assertEqual(f[1]['branch'],'missing_calendar')
        self.assertIsNone(f[1]['cells'][5]['calendar_start_price']);self.assertIsNotNone(f[1]['cells'][5]['calendar_end_price'])
        self.assertTrue(all(c['calendar_rank'] is None and c['delta']==0 for c in f[1]['cells']));self.assertEqual(o[kernel.ARMS[0]],o[kernel.ARMS[1]])

    def test_missing_does_not_hide_invalid_later_name(self):
        args=fixture();d=anchor(args);args[3]['adj'].at[d,'A']=np.nan;args[3]['adj'].at[d,'F']=-1.
        with self.assertRaisesRegex(ValueError,'Positive calendar'):run(args,2)

    def test_missing_start_does_not_hide_invalid_end(self):
        args=fixture();args[1]['state_calm10'][1]=True;args[1]['core_weight'][1]=.5;args[3]['adj'].at[anchor(args),'A']=np.nan;args[3]['adj'].at[anchor(args,lag=2),'A']=np.inf
        refresh_core(args)
        with self.assertRaisesRegex(ValueError,'finite'):run(args,2)

    def test_outside_U0_no_calendar_expansion_or_positive_screen(self):
        args=fixture();t=args[4].index(args[2]['dates'][1]);args[3]['adj'].at[pd.Timestamp(args[4][t-294]),'F']=np.nan;args[3]['adj'].at[anchor(args),'F']=-1.;refresh_core(args)
        f,*_=run(args,2);c=f[1]['cells'][-1];self.assertFalse(c['original_finite']);self.assertEqual(c['calendar_reason'],'outside_U0');self.assertIsNone(c['calendar_start_price']);self.assertEqual(c['delta'],0.)

    def test_empty_domain_copies_incumbent(self):
        args=fixture();t=args[4].index(args[2]['dates'][1]);args[3]['adj'].loc[args[4][t-294],:]=np.nan;refresh_core(args)
        f,o,_,_,a=run(args,2);self.assertEqual(f[1]['branch'],'empty_domain');self.assertEqual(o[kernel.ARMS[0]],o[kernel.ARMS[1]])

    def test_flat_noncalm_zero_sigma_but_calm_ties_valid(self):
        args=fixture();args[3]['adj'].loc[:,:]=100.;refresh_core(args);f,*_=run(args)
        self.assertEqual(f[1]['branch'],'empty_domain');self.assertEqual(f[2]['branch'],'available')
        self.assertTrue(all(c['original_rank']==(3.5/6-.5) and c['calendar_rank']==(3.5/6-.5) for c in f[2]['cells']))

    def test_single_finite_member_rank_half(self):
        args=fixture();t=args[4].index(args[2]['dates'][1]);args[3]['adj'].loc[args[4][t-294],list('BCDEF')]=np.nan;refresh_core(args)
        f,*_=run(args,2);self.assertEqual(f[1]['u0_indices'],[0]);self.assertEqual(f[1]['cells'][0]['original_rank'],.5);self.assertEqual(f[1]['cells'][0]['calendar_rank'],.5)

    def test_finite_zero_rank_differs_from_missing_neutral(self):
        args=fixture();t=args[4].index(args[2]['dates'][1]);args[3]['adj'].loc[args[4][t-294],list('EF')]=np.nan;refresh_core(args)
        f,*_=run(args,2);cells=f[1]['cells'];self.assertTrue(any(c['original_finite'] and c['original_rank']==0 for c in cells))
        self.assertTrue(all(not c['original_finite'] and c['original_rank']==0 for c in cells[4:]))

    def test_raw_Inf_sanitized_not_new_positive_filter(self):
        args=fixture();t=args[4].index(args[2]['dates'][1]);args[3]['adj'].at[pd.Timestamp(args[4][t-294]),'A']=0.;args[3]['adj'].at[pd.Timestamp(args[4][t-294]),'B']=-100.;refresh_core(args)
        f,*_=run(args,2);self.assertFalse(f[1]['cells'][0]['original_finite']);self.assertTrue(f[1]['cells'][1]['original_finite'])

    def test_required_bool_nonreal_and_infinity_rejected(self):
        for bad in (True,'100',float('inf')):
            args=fixture();args[3]['adj']=args[3]['adj'].astype(object);args[3]['adj'].at[anchor(args),'A']=bad
            with self.assertRaises(ValueError):run(args,2)

    def test_derived_calendar_overflow_hardfail(self):
        args=fixture();args[1]['state_calm10'][1]=True;args[1]['core_weight'][1]=.5;d=anchor(args);args[3]['adj'].at[d,'A']=np.finfo(float).tiny;args[3]['adj'].at[anchor(args,lag=2),'A']=1e300;refresh_core(args)
        with self.assertRaises(ArithmeticError):run(args,2)

    def test_C_tolerance_source_only_literal_unsnapped_delta(self):
        args=fixture();args[3]['adj'].loc[:,:]=100.;args[1]['state_calm10'][:]=True;args[1]['core_weight'][:]=.5;refresh_core(args)
        args[1]['core'][1,0]+=5e-13;args[1]['baseline'][1,0]+=5e-13
        f,o,s,p,a=run(args,2);c=f[1]['cells'][0]
        self.assertGreater(c['core_abs_error'],0);self.assertLess(c['core_abs_error'],1e-12);self.assertNotEqual(c['delta'],0.)
        self.assertEqual(c['delta'],.5*c['calendar_rank']-args[1]['core'][1,0]);self.assertNotEqual(p[kernel.ARMS[0]][1,0],p[kernel.ARMS[1]][1,0])

    def test_C_mismatch_blocks_calendar_evaluation(self):
        args=fixture();args[1]['core'][2,0]+=.001
        with patch.object(kernel,'_calendar_values',side_effect=AssertionError('Must not reach calendar')) as q:
            with self.assertRaisesRegex(ValueError,'core source mismatch'):run(args)
            self.assertEqual(q.call_count,0)

    def test_exact_ties_and_nextafter_rank(self):
        x=[1.,1.,np.nextafter(1.,2.),np.nan];r=kernel._rank(x)
        np.testing.assert_array_equal(r,np.array([1.5/3-.5,1.5/3-.5,1.-.5,0.]))

    def test_original_full_pandas_sigma_retained(self):
        args=fixture();f,*_=run(args);sd=args[3]['adj'].pct_change(fill_method=None).rolling(63).std()
        for e in f[1:]:
            for c in e['cells']:self.assertEqual(c['sigma63'],float(sd.at[pd.Timestamp(e['date']),c['symbol']]))

    def test_state_specific_calendar_dates_and_variable_lag(self):
        args=fixture();f,*_=run(args);self.assertEqual(f[1]['calendar_start'],'2015-06-30');self.assertEqual(f[2]['calendar_start'],'2015-09-30')
        for e in f[1:]:self.assertEqual(e['calendar_lag_sessions'],args[4].index(e['date'])-args[4].index(e['calendar_end']))

    def test_visible_original_listing_contradiction_precedes_missing(self):
        args=fixture();args[5].update(make_review(args[2],args[4],args[4][0],[bound()]));args[3]['adj'].at[anchor(args),'F']=np.nan
        with self.assertRaisesRegex(ValueError,'pre-inception core'):run(args,2)

    def test_future_listing_evidence_not_used_early(self):
        args=fixture();args[5].update(make_review(args[2],args[4],args[4][0],[bound(visible=False)]));f,*_=run(args,2);self.assertEqual(f[1]['branch'],'available')

    def test_listing_identity_missing_duplicate_clock_tamper(self):
        for kind in ('missing','duplicate','clock'):
            args=fixture();args[5].update(make_review(args[2],args[4],args[4][0],[bound(visible=False)]))
            if kind=='missing':args[5]['identity_cells'].pop(1)
            elif kind=='duplicate':args[5]['identity_cells'].append(deepcopy(args[5]['identity_cells'][1]))
            else:args[5]['identity_cells'][1]['calendar_anchors']['M-2']='2016-06-29'
            with self.assertRaises(ValueError):run(args,2)

    def test_future_identity_contents_ignored(self):
        args=fixture();args[5]['identity_cells'].append({'query_index':2,'anything':'bad'});run(args,2)

    def test_zero_one_two_eligible_preserve_cash_or_two_names(self):
        for n in (0,1,2):
            args=fixture();args[1]['eligible'][:,n:]=False;_,orders,*_=run(args)
            for arm in kernel.ARMS:
                self.assertTrue(all(set(x['symbols'])==(set('AB') if n==2 else set()) for x in orders[arm]))
                self.assertTrue(all(len(x['scores'])==(2 if n==2 else 0) for x in orders[arm]))

    def test_original_support_and_calm_extension_preserved(self):
        args=fixture();args[1]['extension_available'][2]=True;args[1]['forecast_available'][2]=True;f,o,s,p,a=run(args)
        self.assertEqual(a['events'][2]['extension_active_indices'],list(range(6)))
        self.assertEqual(a['events'][2]['original_remaining_support'],[(1.-.5)-(.025+.01)-.05]*6)

    def test_cached_tamper_rejected(self):
        args=fixture();full=run(args)[0]
        for field in ('calendar_rank','delta','original_rank','frozen_core'):
            f=deepcopy(full);f[1]['cells'][0][field]+=.01
            with self.assertRaises(ValueError):kernel.decisions(args[0],args[1],args[2],f,3)
        f=deepcopy(full);f[1]['branch']='missing_calendar'
        with self.assertRaises(ValueError):kernel.decisions(args[0],args[1],args[2],f,3)

    def test_axes_boolean_shape_monthend_reject(self):
        for kind in ('duplicate','bool','shape','notmonthend','missingcolumn'):
            args=fixture()
            if kind=='duplicate':args[3]['adj'].index=pd.DatetimeIndex([args[3]['adj'].index[0]]+list(args[3]['adj'].index[:-1]))
            elif kind=='bool':args[1]['eligible']=args[1]['eligible'].astype(object)
            elif kind=='shape':args[1]['core']=args[1]['core'][:,:-1]
            elif kind=='notmonthend':args[2]['dates'][1]='2016-08-30'
            else:args[3]['adj']=args[3]['adj'].drop(columns='A')
            with self.assertRaises((ValueError,TypeError)):run(args,2)

    def test_source_gate_failure_precedes_all_episode_calls(self):
        args=fixture()
        for result in (None,{'passed':False}):
            with patch.object(args[0].ep,'select_boundary',wraps=args[0].ep.select_boundary) as ep:
                with self.assertRaisesRegex(ValueError,'source gate failed'):
                    kernel.prepare(*args,3,source_validator=lambda features:result)
                self.assertEqual(ep.call_count,0)

    def test_incumbent_gate_failure_precedes_candidate_episode(self):
        args=fixture()
        with patch.object(args[0].ep,'select_boundary',wraps=args[0].ep.select_boundary) as ep:
            with self.assertRaisesRegex(ValueError,'incumbent gate failed'):
                kernel.prepare(*args,3,source_validator=lambda f:{'passed':True},incumbent_validator=lambda *x:{'passed':False})
            self.assertEqual(ep.call_count,1)

    def test_frozen_incumbent_mismatch_precedes_candidate_and_callback(self):
        args=fixture();expected=run(args)[1][kernel.ARMS[0]];expected[1]['symbols']=[]
        with patch.object(args[0].ep,'select_boundary',wraps=args[0].ep.select_boundary) as ep,patch.object(kernel,'_calendar_values',wraps=kernel._calendar_values):
            callback=lambda *x:(_ for _ in ()).throw(AssertionError('Frozen mismatch must precede independent gate'))
            with self.assertRaisesRegex(ValueError,'Frozen incumbent'):
                kernel.prepare(*args,3,expected_incumbent_orders=expected,incumbent_validator=callback)
            self.assertEqual(ep.call_count,1)

    def test_successful_gates_order_and_cached_gate(self):
        args=fixture();full=run(args);sequence=[]
        def source(f):
            self.assertEqual(f,full[0]);sequence.append('source');f.clear();return {'passed':True}
        def incumbent(o,s,p):
            self.assertEqual(o,full[1][kernel.ARMS[0]]);np.testing.assert_array_equal(s,full[2][kernel.ARMS[0]]);np.testing.assert_array_equal(p,full[3][kernel.ARMS[0]])
            sequence.append('incumbent');o.clear();s[:]=999.;p[:]=999.;return {'passed':True}
        original=args[0].ep.select_boundary
        def episode(*x,**kw):sequence.append('episode');return original(*x,**kw)
        with patch.object(args[0].ep,'select_boundary',side_effect=episode):
            gated=kernel.prepare(*args,3,source_validator=source,incumbent_validator=incumbent,expected_incumbent_orders=full[1][kernel.ARMS[0]])
        self.assertEqual(sequence,['source','episode','incumbent','episode']);self.assertEqual(gated[0],full[0]);self.assertEqual(gated[1],full[1]);self.assertEqual(gated[4],full[4])
        same_arrays(self,gated[2],full[2]);same_arrays(self,gated[3],full[3])
        with patch.object(args[0].ep,'select_boundary',wraps=original) as ep:
            with self.assertRaisesRegex(ValueError,'incumbent gate failed'):
                kernel.decisions(args[0],args[1],args[2],full[0],3,incumbent_validator=lambda *x:{'passed':False},expected_incumbent_orders=full[1][kernel.ARMS[0]])
            self.assertEqual(ep.call_count,1)

    def test_no_mutation(self):
        args=fixture();before=deepcopy(args[1:]);run(args)
        same_arrays(self,args[1],before[0]);self.assertEqual(args[2],before[1]);pd.testing.assert_frame_equal(args[3]['adj'],before[2]['adj']);self.assertEqual(args[4],before[3]);self.assertEqual(args[5],before[4])


if __name__=='__main__':unittest.main(verbosity=2)
