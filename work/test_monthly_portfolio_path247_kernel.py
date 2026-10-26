"""Synthetic prices/cache only; no original source arrays or saved orders."""
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import monthly_portfolio_path247_kernel as kernel


def study():
    path=kernel.ROOT/'outputs/monthly_local_linear70/study.py'
    spec=importlib.util.spec_from_file_location('portfolio247_synthetic_local67',path)
    s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s);return s


def make_review(meta,calendar,bounds=None):
    sessions=pd.to_datetime(calendar);windows=[];cells=[];bounds=deepcopy(bounds or [])
    for i,(day,names) in enumerate(zip(meta['dates'],meta['symbols'])):
        t=calendar.index(day);ds=calendar[t-294:t-41]
        windows.append(dict(query_index=i,date=day,initial_partial=day==kernel.INITIAL_PARTIAL,query_session_position=t,
            price_start=ds[0],first_return_date=ds[1],price_end=ds[-1],source_start_position=t-294,source_end_position=t-42,
            prices=253,return_intervals=252,next_official_session=calendar[t+1],
            source_dates_sha256=hashlib.sha256(json.dumps(ds,separators=(',',':')).encode()).hexdigest()))
        for j,symbol in enumerate(names):
            for b in bounds:
                if symbol==b['symbol']:
                    cells.append(dict(query_index=i,column_index=j,date=day,symbol=symbol,issuer_key=b['issuer_key'],
                        new_price_start=ds[0],new_price_end=ds[-1],bound_visible=b['available_from']<=day,
                        start_precedes_bound=ds[0]<b['not_before']))
    return dict(calendar=dict(windows=windows),listing_bounds=bounds,listing_identity_cells=cells)


def fixture():
    s=study();shape=(3,6);names=list('ABCDEF');z=np.zeros(shape);f=np.zeros(shape,bool)
    calm=np.array([True,False,True]);weak=np.array([False,False,False]);cw=np.where(calm,.5,.52)[:,None]
    core=np.full(shape,.15)
    arrays=dict(baseline=core.copy(),core=core.copy(),core_weight=cw,eligible=np.ones(shape,bool),
        effects=np.tile([0.,0.,0.,.5,.5,.5],(3,1)),state_calm10=calm,state_weak_narrow=weak,
        pre_score=np.tile([.31,.305,.30,.295,.29,.285],(3,1)),old_ml4=z.copy(),rank_log_excess=z.copy(),
        extension_available=f.copy(),forecast_available=f.copy(),local_available=f.copy(),model_available=f.copy(),
        raw_local_constant=z.copy(),raw_local_linear=z.copy(),raw_source_log=z.copy())
    meta=dict(dates=['2016-08-17','2016-08-31','2016-09-30'],ends=['2016-08-31','2016-09-30','2016-10-31'],
              symbols=[names.copy() for _ in range(3)],etfs=['F'])
    dates=pd.bdate_range('2014-01-01','2016-11-02');x=np.arange(len(dates),dtype=float)
    frame=pd.DataFrame({symbol:100.+.1*x+.7*np.sin((k+1)*x/17.)+.3*np.cos(x/(11+k)) for k,symbol in enumerate(names)},index=dates)
    calendar=dates.strftime('%Y-%m-%d').tolist();panel={'adj':frame}
    return s,arrays,meta,panel,calendar,make_review(meta,calendar)


def physical(args,n):
    s,a,m,p,c,r=args;aa={k:np.asarray(v)[:n].copy() for k,v in a.items()};mm=deepcopy(m)
    for k in ('dates','ends','symbols'):mm[k]=mm[k][:n]
    if n:
        last=mm['dates'][-1];cc=c[:c.index(last)+2];pp={'adj':p['adj'].loc[:last].copy()}
    else:cc=[];pp={'adj':p['adj'].iloc[:0].copy()}
    return s,aa,mm,pp,cc,deepcopy(r)


def run(args,n=None):
    s,a,m,p,c,r=args;return kernel.prepare(s,a,m,p,c,r,len(m['dates']) if n is None else n)


def same_arrays(case,left,right):
    case.assertEqual(tuple(left),tuple(right))
    for key in left:np.testing.assert_array_equal(left[key],right[key],err_msg=key)


def source_dates(args,row=1):
    c=args[4];t=c.index(args[2]['dates'][row]);return pd.to_datetime(c[t-294:t-41])


def identical_paths(args):
    frame=args[3]['adj'];shared=frame.iloc[:,0].copy()
    for name in frame.columns:frame[name]=shared


class IntegrationTests(unittest.TestCase):
    def test_schema_once_bridge_all_arm_original_scores(self):
        args=fixture();s,a,m,p,c,r=args
        with patch.object(s,'modified_pre_score',wraps=s.modified_pre_score) as pre,patch.object(s.ep,'select_boundary',wraps=s.ep.select_boundary) as ep:
            features,orders,scores,pres,audit=run(args)
            self.assertEqual((pre.call_count,ep.call_count),(1,1))
        self.assertEqual(tuple(orders),kernel.ARMS);self.assertEqual(tuple(audit['counts']),kernel.COUNT_FIELDS)
        self.assertEqual(audit['counts']['original_episode_calls'],1)
        self.assertEqual(audit['counts']['new_model_fits'],0)
        self.assertTrue(all(tuple(e)==kernel.EVENT_KEYS for e in features))
        for arm in kernel.ARMS:
            np.testing.assert_array_equal(scores[arm],scores[kernel.ARMS[0]])
            np.testing.assert_array_equal(pres[arm],pres[kernel.ARMS[0]])
        json.dumps(features,allow_nan=False);json.dumps(audit,allow_nan=False)
        self.assertEqual(audit['counts']['source_price_observations'],2*5*253)
        self.assertEqual(audit['counts']['enumerated_trios'],20)

    def test_cached_decisions_exact_and_one_episode(self):
        args=fixture();s,a,m,*_=args;features,o,sc,p,au=run(args)
        with patch.object(s.ep,'select_boundary',wraps=s.ep.select_boundary) as ep:
            oo,ss,pp,aa=kernel.decisions(s,a,m,features,3)
            self.assertEqual(ep.call_count,1)
        self.assertEqual(o,oo);same_arrays(self,sc,ss);same_arrays(self,p,pp);self.assertEqual(au,aa)

    def test_incumbent_reproduces_original_study(self):
        args=fixture();s,a,m,*_=args;_,o,sc,p,_=run(args)
        chosen,post=s.choices(s.configs()[4],a,m,3,policy=dict(s.POLICY_FIELDS))
        self.assertEqual(o[kernel.ARMS[0]],s.rp.make_orders(chosen,post,m))
        np.testing.assert_array_equal(sc[kernel.ARMS[0]],post)

    def test_partial_no_source_or_trio_numeric_reads(self):
        args=fixture();args[3]['adj']=args[3]['adj'].astype(object);args[3]['adj'].loc[:,:]='bad'
        f,o,_,_,a=run(args,1)
        self.assertEqual(f[0]['branch'],'initial_partial');self.assertEqual(f[0]['source_cells'],[])
        self.assertEqual(f[0]['trios'],[]);self.assertEqual(a['counts']['source_price_observations'],0)
        for arm in kernel.ARMS:self.assertEqual(o[arm],o[kernel.ARMS[0]])

    def test_zero_one_two_cash_half_rules_no_source_reads(self):
        for count in (0,1,2):
            args=fixture();args[1]['eligible'][:,count:]=False;args[3]['adj']=args[3]['adj'].astype(object);args[3]['adj'].loc[:,:]='bad'
            f,o,_,_,au=run(args)
            for arm in kernel.ARMS:
                self.assertTrue(all(v['symbols']==(list('AB') if count==2 else []) for v in o[arm]))
            self.assertEqual(au['counts']['source_price_observations'],0)
            self.assertTrue(all(not e['trios'] for e in f))

    def test_three_four_five_enumeration_incumbent_feasible(self):
        for count,want in ((3,1),(4,4),(5,10),(6,10)):
            args=fixture();args[1]['eligible'][:,count:]=False;f,*_=run(args,2)
            e=f[1];self.assertEqual(len(e['trios']),want)
            inc=[t for t in e['trios'] if t['indices']==e['incumbent_trio']]
            self.assertEqual(len(inc),1);self.assertTrue(inc[0]['feasible'])
            self.assertEqual(inc[0]['raw_core_regret'],0.);self.assertEqual(inc[0]['final_regret'],0.)
            self.assertNotIn(5,e['shortlist_indices'])

    def test_all_states_active_and_no_new_core_or_E(self):
        args=fixture();before=deepcopy(args[1]);f,o,sc,p,au=run(args)
        self.assertEqual([e['branch'] for e in f[1:]],['available','available'])
        self.assertEqual([e['core_weight'] for e in f[1:]],[.52,.5])
        same_arrays(self,args[1],before)
        for e in au['events']:
            self.assertGreaterEqual(e['core_weight'],.5)
            self.assertEqual(e['noncore_scale'],1.-.025/(1.-e['core_weight']))

    def test_structural_selector_can_replace_anchor(self):
        f,o,_,_,_=run(fixture())
        self.assertEqual(set(o[kernel.ARMS[0]][1]['symbols']),set('ABD'))
        self.assertEqual(set(o[kernel.ARMS[3]][1]['symbols']),set('ADE'))
        self.assertTrue(set(f[1]['incumbent_choices'])<=set(f[1]['shortlist_indices']))

    def test_missing_symbol_whole_group_fallback_no_rank6(self):
        args=fixture();args[3]['adj']=args[3]['adj'].drop(columns='E');f,o,_,_,au=run(args,2)
        self.assertEqual(f[1]['branch'],'missing_source');self.assertEqual(f[1]['shortlist_indices'],[0,1,2,3,4])
        self.assertEqual(f[1]['source_cells'][-1]['reason'],'missing_symbol')
        self.assertTrue(all(t['statistics'] is None for t in f[1]['trios']))
        for arm in kernel.ARMS:self.assertEqual(o[arm],o[kernel.ARMS[0]])
        self.assertEqual(au['counts']['source_price_observations'],4*253)

    def test_missing_date_and_nan_reason_priority(self):
        args=fixture();ds=source_dates(args);args[3]['adj'].loc[ds[1],'A']=np.nan
        args[3]['adj']=args[3]['adj'].drop(index=ds[0]);f,*_=run(args,2)
        self.assertTrue(all(c['reason']=='missing_official_session' for c in f[1]['source_cells']))
        self.assertTrue(all(c['observations_checked']==252 for c in f[1]['source_cells']))

    def test_nan_unavailable_scans_remaining_shortlist(self):
        args=fixture();ds=source_dates(args);args[3]['adj'].loc[ds[1],'A']=np.nan;f,o,_,_,_=run(args,2)
        self.assertEqual(f[1]['source_cells'][0]['reason'],'missing_nan')
        self.assertTrue(all(c['source_available'] for c in f[1]['source_cells'][1:]))
        for arm in kernel.ARMS:self.assertEqual(o[arm],o[kernel.ARMS[0]])

    def test_invalid_not_hidden_by_missing_symbol_or_row(self):
        for invalid in (True,'7',None,float('inf'),-1.,0.):
            args=fixture();ds=source_dates(args);args[3]['adj']=args[3]['adj'].astype(object).drop(columns='A')
            args[3]['adj'].loc[ds[1],'E']=invalid
            with self.subTest(invalid=repr(invalid)),self.assertRaises(ValueError):run(args,2)
        args=fixture();ds=source_dates(args);args[3]['adj']=args[3]['adj'].drop(index=ds[0]);args[3]['adj'].loc[ds[1],'E']=np.inf
        with self.assertRaises(ValueError):run(args,2)

    def test_unshortlisted_invalid_source_ignored(self):
        args=fixture();args[3]['adj']['F']='bad';f,*_=run(args,2)
        self.assertEqual(f[1]['branch'],'available')

    def test_any_feasible_zero_variance_all_experimental_fallback(self):
        args=fixture()
        for name in 'ABC':args[3]['adj'][name]=100.
        f,o,_,_,au=run(args,2)
        self.assertEqual(f[1]['branch'],'zero_variance');self.assertGreater(au['counts']['zero_variance_trios'],0)
        self.assertEqual(au['counts']['trio_statistics'],len(f[1]['feasible_trio_indices']))
        for arm in kernel.ARMS:self.assertEqual(o[arm],o[kernel.ARMS[0]])
        self.assertTrue(all(t['joint_q'] is None for t in au['events'][1]['trios']))

    def test_zero_in_infeasible_trio_not_evaluated(self):
        args=fixture();a=args[1];a['effects'][:]=0.;a['core'][:,:3]=.24;a['core'][:,3:]=-.2;a['baseline']=a['core'].copy()
        for name in 'CDE':args[3]['adj'][name]=100.
        with patch.object(kernel.stats,'measure',wraps=kernel.stats.measure) as measured:
            f,*_=run(args,2)
        self.assertEqual(f[1]['feasible_trio_indices'],[0]);self.assertEqual(measured.call_count,1)
        self.assertEqual(f[1]['branch'],'available');self.assertTrue(all(t['statistics'] is None for t in f[1]['trios'][1:]))

    def test_constant_Q_can_change_structure_but_matches_risk_arms(self):
        args=fixture();a=args[1];a['pre_score'][:]=[.3,.299,.298,.297,.296,.1]
        a['effects'][:]=[0.,0.,0.,.06,.08,0.];a['core'][:,4]=.12;a['baseline']=a['core'].copy();identical_paths(args)
        f,o,_,_,au=run(args,2)
        self.assertEqual(set(o[kernel.ARMS[3]][1]['symbols']),set('ADE'))
        self.assertEqual(set(o[kernel.ARMS[1]][1]['symbols']),set('ABD'))
        self.assertEqual(o[kernel.ARMS[1]],o[kernel.ARMS[2]])
        qs=[t['joint_q'] for t in au['events'][1]['trios'] if t['feasible']]
        self.assertEqual(len(set(qs)),1);self.assertGreater(qs[0],0.)

    def test_exact_utility_ties_prefer_nonlex_incumbent(self):
        args=fixture();a=args[1];a['pre_score'][:]=[.234375,.234375,.234375,.25,.25,.2]
        a['effects'][:]=[.3125,.3125,.3125,0.,0.,0.];a['core'][:]=.125;a['baseline']=a['core'].copy();identical_paths(args)
        f,o,_,_,_=run(args,2)
        self.assertEqual(o[kernel.ARMS[0]][1]['symbols'],list('ADE'))
        self.assertEqual(f[1]['trios'][0]['symbols'],list('ABC'))
        for arm in kernel.ARMS:self.assertEqual(o[arm],o[kernel.ARMS[0]])

    def test_canonical_means_are_independent_per_member_operations(self):
        args=fixture();a=args[1];a['core'][1,:]=[.19,.17,.13,.11,.07,.03];a['baseline']=a['core'].copy()
        f,*_=run(args,2)
        for t in f[1]['trios']:
            ids=t['indices'];cw=.52
            self.assertEqual(t['raw_core_mean'],math.fsum(float(a['core'][1,j])/cw for j in ids)/3.)
            self.assertEqual(t['weighted_core_mean'],math.fsum(float(a['core'][1,j]) for j in ids)/3.)
            self.assertEqual(t['auxiliary_mean'],math.fsum(f[1]['final_scores'][j]-float(a['core'][1,j]) for j in ids)/3.)
            self.assertEqual(t['feasible'],t['raw_core_mean']>=f[1]['reference_raw_core_mean']-.02 and t['final_mean']>=f[1]['reference_final_mean']-.02)

    def test_known_listing_contradiction_not_masked_by_another_missing(self):
        args=fixture();bound=dict(symbol='E',issuer_key='synthetic:E',not_before='2016-08-30',available_from='2016-08-31',evidence_id='synthetic',kind='synthetic')
        args=list(args);args[5]=make_review(args[2],args[4],[bound]);args[3]['adj']=args[3]['adj'].drop(columns='A')
        with self.assertRaisesRegex(ValueError,'listing identity'):run(tuple(args),2)

    def test_known_listing_missing_source_fallback_and_future_evidence_not_applied(self):
        for future in (False,True):
            args=list(fixture());bound=dict(symbol='E',issuer_key='synthetic:E',not_before='2016-08-30',available_from='2017-01-01' if future else '2016-08-31',evidence_id='synthetic',kind='synthetic')
            args[5]=make_review(args[2],args[4],[bound])
            if not future:args[3]['adj']['E']=np.nan
            f,*_=run(tuple(args),2)
            self.assertEqual(f[1]['branch'],'available' if future else 'missing_source')
            self.assertEqual(f[1]['source_cells'][-1]['listing']['visible'],not future)

    def test_metadata_or_listing_identity_mismatch_hardfails(self):
        args=list(fixture());args[5]['calendar']['windows'][1]['price_start']='2010-01-01'
        with self.assertRaises(ValueError):run(tuple(args),2)
        args=list(fixture());bound=dict(symbol='E',issuer_key='synthetic:E',not_before='2016-08-30',available_from='2017-01-01',evidence_id='synthetic',kind='synthetic')
        args[5]=make_review(args[2],args[4],[bound]);args[5]['listing_identity_cells'][1]['issuer_key']='wrong'
        with self.assertRaises(ValueError):run(tuple(args),2)

    def test_derived_arithmetic_failure_propagates(self):
        args=fixture();ds=source_dates(args);args[3]['adj'].loc[ds,'A']=1.;args[3]['adj'].loc[ds[1],'A']=1e200
        with self.assertRaises(ArithmeticError):run(args,2)

    def test_cache_domain_scores_dates_and_zero_gates_rechecked(self):
        args=fixture();s,a,m,*_=args;features,*_=run(args,2)
        variants=[]
        f=deepcopy(features);f[1]['pre_scores'][0]+=.1;variants.append(f)
        f=deepcopy(features);f[1]['eligible_indices']=[];variants.append(f)
        f=deepcopy(features);f[1]['shortlist_indices'].reverse();variants.append(f)
        f=deepcopy(features);f[1]['date']='2016-08-30';variants.append(f)
        f=deepcopy(features);f[1]['trios'][0]['feasible']=not f[1]['trios'][0]['feasible'];variants.append(f)
        f=deepcopy(features);k=f[1]['feasible_trio_indices'][0];f[1]['trios'][k]['statistics']['joint_variance']=0.;variants.append(f)
        for f in variants:
            with self.assertRaises(ValueError):kernel.decisions(s,a,m,f,2)

    def test_all_logical_physical_and_cached_prefixes_exact(self):
        args=fixture();features,o,sc,p,au=run(args);s,a,m,*_=args
        for n in range(4):
            ff,oo,ss,pp,aa=run(args,n);pf,po,ps,pr,pa=run(physical(args,n))
            co,cs,cp,ca=kernel.decisions(s,a,m,features,n)
            self.assertEqual(ff,features[:n]);self.assertEqual(ff,pf);self.assertEqual(oo,po);self.assertEqual(oo,co)
            same_arrays(self,ss,ps);same_arrays(self,ss,cs);same_arrays(self,pp,pr);same_arrays(self,pp,cp)
            self.assertEqual(aa,pa);self.assertEqual(aa,ca);self.assertEqual(aa['events'],au['events'][:n])
            self.assertEqual(aa['counts']['original_episode_calls'],1)

    def test_future_values_metadata_and_review_records_ignored(self):
        args=list(fixture());expected=run(tuple(args),2)
        for key in ('core_weight','baseline','core','effects','pre_score'):
            args[1][key]=args[1][key].astype(object);args[1][key][2:]='future poison'
        args[2]['dates'][2]='bad';args[2]['ends'][2]='bad';args[2]['symbols'][2]=['bad']
        args[5]['calendar']['windows'][2]={'query_index':2,'poison':object()}
        args[5]['listing_identity_cells'].append({'query_index':2,'poison':object()})
        args[3]['adj']=args[3]['adj'].astype(object);args[3]['adj'].loc['2016-09-01':]='future poison'
        actual=run(tuple(args),2)
        self.assertEqual(actual[0],expected[0]);self.assertEqual(actual[1],expected[1]);self.assertEqual(actual[4],expected[4])

    def test_latest42_prices_do_not_change_new_features(self):
        args=fixture();expected=run(args,2);t=args[4].index(args[2]['dates'][1]);after=args[4][t-41:t+1]
        args[3]['adj']=args[3]['adj'].astype(object);args[3]['adj'].loc[pd.to_datetime(after)]='excluded latest42'
        actual=run(args,2);self.assertEqual(actual[0],expected[0]);self.assertEqual(actual[4],expected[4])

    def test_core_and_final_cap_equality_nextafter(self):
        for which in ('core','final'):
            middle=-.03 if which=='core' else -.06
            # One input ULP can round back onto the cap after division by three.
            below=math.nextafter(math.nextafter(middle,-math.inf),-math.inf)
            for value,want in ((below,False),(middle,True),(math.nextafter(middle,math.inf),True)):
                args=fixture();a=args[1];a['effects'][:]=0.;a['state_calm10'][:]=True;a['core_weight'][:]=.5;a['core'][:]=0.
                if which=='core':
                    a['core'][:,3]=value;a['pre_score'][:]=[.12,.115,.11,.105,.10,.09]
                else:a['pre_score'][:]=[0.,0.,0.,value,-.07,-.08]
                a['baseline']=a['core'].copy();f,*_=run(args,2)
                t=next(t for t in f[1]['trios'] if t['indices']==[0,1,3])
                with self.subTest(which=which,value=value):
                    self.assertEqual(t['feasible'],want)
                    mean=t['raw_core_mean'] if which=='core' else t['final_mean']
                    if value==middle:self.assertEqual(mean,-.02)
                    if value==below:self.assertEqual(mean,math.nextafter(-.02,-math.inf))

    def test_nonincumbent_exact_ties_choose_lexicographic_trio(self):
        args=fixture();a=args[1];a['pre_score'][:]=[.234375,.234375,.234375,.25,.24609375,.2]
        a['effects'][:]=[.3125,.3125,.3125,0.,0.,0.];a['core'][:]=.125;a['baseline']=a['core'].copy();identical_paths(args)
        _,o,_,_,_=run(args,2)
        self.assertEqual(o[kernel.ARMS[0]][1]['symbols'],list('ADE'))
        for arm in kernel.ARMS[1:]:self.assertEqual(o[arm][1]['symbols'],list('ABC'))

    def test_P_shortlist_keeps_original_when_F_rank_six_rises(self):
        args=fixture();a=args[1];a['pre_score'][:]=[.3,.299,.298,.297,.296,.295]
        a['effects'][:]=[0.,0.,-.5,-.5,-.5,0.];args[3]['adj']['F']='not shortlisted'
        f,_,sc,_,_=run(args,2);event=f[1]
        self.assertEqual(event['shortlist_indices'],[0,1,2,3,4])
        self.assertGreater(sc[kernel.ARMS[0]][1,5],sc[kernel.ARMS[0]][1,2])
        self.assertTrue(set(event['incumbent_choices'])<=set(event['shortlist_indices']))
        self.assertEqual(event['branch'],'available')

    def test_inputs_and_featurecache_not_mutated(self):
        args=fixture();a=deepcopy(args[1]);m=deepcopy(args[2]);p=args[3]['adj'].copy();r=deepcopy(args[5]);f,*_=run(args)
        ff=deepcopy(f);kernel.decisions(args[0],args[1],args[2],f,3)
        same_arrays(self,a,args[1]);self.assertEqual(m,args[2]);pd.testing.assert_frame_equal(p,args[3]['adj']);self.assertEqual(r,args[5]);self.assertEqual(ff,f)

    def test_wrong_boolean_axes_and_duplicate_calendar_hardfail(self):
        args=fixture();args[1]['eligible']=args[1]['eligible'].astype(object)
        with self.assertRaises(ValueError):run(args,2)
        args=list(fixture());args[4]=args[4][:10]+[args[4][9]]+args[4][10:]
        with self.assertRaises(ValueError):run(tuple(args),2)


if __name__=='__main__':unittest.main(verbosity=2)
