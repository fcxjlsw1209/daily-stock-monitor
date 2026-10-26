"""258 synthetic-only source, arithmetic, selection and boundary tests."""
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import monthly_downside_basket258_kernel as kernel


def study():
    path=kernel.ROOT/'outputs/monthly_local_linear70/study.py'
    spec=importlib.util.spec_from_file_location('downside258_synthetic_local67',path)
    s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s);return s


def make_review(meta,calendar,bounds=None):
    windows=[];cells=[];bounds=deepcopy(bounds or [])
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
    s=study();shape=(3,6);names=list('ABCDEF');z=np.zeros(shape);b=np.zeros(shape,bool)
    calm=np.array([True,False,True]);cw=np.where(calm,.5,.52)[:,None];core=np.full(shape,.15)
    arrays=dict(baseline=core.copy(),core=core.copy(),core_weight=cw,eligible=np.ones(shape,bool),
        effects=np.tile([0.,0.,0.,.5,.5,.5],(3,1)),state_calm10=calm,state_weak_narrow=np.zeros(3,bool),
        pre_score=np.tile([.31,.305,.30,.295,.29,.285],(3,1)),old_ml4=z.copy(),rank_log_excess=z.copy(),
        extension_available=b.copy(),forecast_available=b.copy(),local_available=b.copy(),model_available=b.copy(),
        raw_local_constant=z.copy(),raw_local_linear=z.copy(),raw_source_log=z.copy())
    meta=dict(dates=['2016-08-17','2016-08-31','2016-09-30'],ends=['2016-08-31','2016-09-30','2016-10-31'],
              symbols=[names.copy() for _ in range(3)],etfs=['F'])
    dates=pd.bdate_range('2014-01-01','2016-11-02');x=np.arange(len(dates),dtype=float)
    frame=pd.DataFrame({name:100.+.1*x+.7*np.sin((k+1)*x/17.)+.3*np.cos(x/(11+k)) for k,name in enumerate(names)},index=dates)
    calendar=dates.strftime('%Y-%m-%d').tolist()
    return s,arrays,meta,{'adj':frame},calendar,make_review(meta,calendar)


def physical(args,n):
    s,a,m,p,c,r=args;aa={k:np.asarray(v)[:n].copy() for k,v in a.items()}
    mm={k:deepcopy(m[k][:n]) for k in ('dates','ends','symbols')};mm['etfs']=m['etfs'].copy()
    if n:
        last=mm['dates'][-1];cc=c[:c.index(last)+2];pp={'adj':p['adj'].loc[:last].copy()}
    else:cc=[];pp={'adj':p['adj'].iloc[:0].copy()}
    return s,aa,mm,pp,cc,deepcopy(r)


def run(args,n=None,**kw):
    s,a,m,p,c,r=args;return kernel.prepare(s,a,m,p,c,r,len(m['dates']) if n is None else n,**kw)


def source_dates(args,row=1):
    c=args[4];t=c.index(args[2]['dates'][row]);return pd.to_datetime(c[t-294:t-41])


def same_arrays(test,left,right):
    test.assertEqual(tuple(left),tuple(right))
    for key in left:np.testing.assert_array_equal(left[key],right[key],err_msg=key)


def bound(symbol='A',visible='2016-08-01'):
    return dict(symbol=symbol,issuer_key='new-'+symbol,not_before='2016-07-01',available_from=visible,evidence_id='synthetic',kind='new_listing')


class Poison:
    def __deepcopy__(self,memo):raise AssertionError('Future deepcopy')
    def __eq__(self,other):raise AssertionError('Future equality')
    def __float__(self):raise AssertionError('Future float')


class MomentTests(unittest.TestCase):
    def test_constant_prices_legal_zero_complete_evidence(self):
        z=kernel.measure([[1.]*253,[2.]*253,[3.]*253])
        self.assertEqual(tuple(z),kernel.STAT_KEYS);self.assertTrue(z['zero_downside']);self.assertTrue(z['zero_symmetric'])
        self.assertEqual(z['zero_returns'],252);self.assertEqual(z['downside_moment'],0.);self.assertEqual(z['symmetric_moment'],0.)
        self.assertTrue(kernel.validate_statistics(z));json.dumps(z,allow_nan=False)

    def test_positive_constant_returns_not_zero_symmetric(self):
        p=[1.01**k for k in range(253)];z=kernel.measure([p,p,p])
        self.assertEqual(z['downside_moment'],0.);self.assertGreater(z['symmetric_moment'],0.)
        self.assertEqual(z['positive_returns'],252);self.assertEqual(z['upside_moment'],z['symmetric_moment'])

    def test_all_negative_returns_downside_equals_symmetric(self):
        p=[.99**k for k in range(253)];z=kernel.measure([p,p,p])
        self.assertEqual(z['negative_returns'],252);self.assertEqual(z['upside_moment'],0.)
        self.assertEqual(z['downside_moment'],z['symmetric_moment'])

    def test_fixed_units_not_daily_rebalanced_returns(self):
        p=[[1.,2.,2.]+[2.]*250,[1.,1.,.5]+[.5]*250,[1.]*253];z=kernel.measure(p)
        self.assertNotEqual(z['portfolio_returns'][1],(-.5)/3.)
        self.assertEqual(z['portfolio_returns'][1],-.5/4.)
        for a,b in zip(z['portfolio_returns'],z['direct_returns']):self.assertAlmostEqual(a,b,places=14)

    def test_mixed_sign_squares_exact_canonical(self):
        p=[1.+.01*math.sin(k/3.) for k in range(253)];z=kernel.measure([p,p,p]);r=z['portfolio_returns']
        self.assertEqual(z['downside_square_sum'],math.fsum(min(x,0.)*min(x,0.) for x in r))
        self.assertEqual(z['symmetric_square_sum'],math.fsum(x*x for x in r))
        self.assertEqual(z['downside_moment'],z['downside_square_sum']/252.)
        self.assertAlmostEqual(z['symmetric_moment'],z['downside_moment']+z['upside_moment'],places=18)

    def test_source_shape_types_values_fail_closed(self):
        for bad in (True,np.bool_(False),None,'1',0.,-1.,math.inf,math.nan,10**1000):
            p=[[1.]*253 for _ in range(3)];p[2][-1]=bad
            with self.subTest(bad=type(bad).__name__),self.assertRaises(ValueError):kernel.measure(p)
        for p in ([[1.]*253]*2,[[1.]*252]*3):
            with self.assertRaises(ValueError):kernel.measure(p)

    def test_underflow_square_and_final_division_hardfail(self):
        with self.assertRaises(ArithmeticError):kernel.f._square(1e-200,'test')
        with self.assertRaises(ArithmeticError):kernel.f._divide(math.ulp(0.),252.,'test')
        p=[[1.]+[1e-160]*251+[1.0000000001e-160],[1.]*253,[1.]*253]
        with self.assertRaises(ArithmeticError):kernel.measure(p)

    def test_overflow_normalization_hardfail(self):
        p=[[1e-308]+[1e308]*252,[1.]*253,[1.]*253]
        with self.assertRaises(ArithmeticError):kernel.measure(p)

    def test_cached_exact_oneulp_and_boolean_tamper_rejected(self):
        p=[1.01**k for k in range(253)];z=kernel.measure([p,p,p])
        for field in ('symmetric_moment','symmetric_square_sum'):
            changed=deepcopy(z);changed[field]=math.nextafter(changed[field],math.inf)
            with self.assertRaises(ValueError):kernel.validate_statistics(changed)
        changed=deepcopy(z);changed['price_count']=True
        with self.assertRaises(ValueError):kernel.validate_statistics(changed)

    def test_exact_tie_incumbent_then_symbol_and_oneulp(self):
        ts=[dict(indices=[0,1,2],symbols=list('ABC'),statistics=dict(downside_moment=.1)),
            dict(indices=[0,1,3],symbols=list('ABD'),statistics=dict(downside_moment=.1)),
            dict(indices=[0,2,3],symbols=list('ACD'),statistics=dict(downside_moment=.1))]
        e=dict(trios=ts,feasible_trio_indices=[0,1,2],incumbent_trio=[0,1,3])
        self.assertEqual(kernel.choose_trio(e,'downside_moment'),(1,[0,1,2]))
        ts[0]['statistics']['downside_moment']=math.nextafter(.1,-math.inf)
        self.assertEqual(kernel.choose_trio(e,'downside_moment'),(0,[0]))
        ts[0]['statistics']['downside_moment']=.1;ts[1]['statistics']['downside_moment']=math.nextafter(.1,math.inf)
        self.assertEqual(kernel.choose_trio(e,'downside_moment'),(0,[0,2]))


class IntegrationTests(unittest.TestCase):
    def test_full_schema_one_episode_preserved_scores(self):
        args=fixture();s=args[0]
        with patch.object(s.ep,'select_boundary',wraps=s.ep.select_boundary) as ep:
            features,o,sc,p,a=run(args);self.assertEqual(ep.call_count,1)
        self.assertEqual(tuple(a['counts']),kernel.COUNT_FIELDS)
        self.assertEqual(a['counts']['source_price_observations'],2*5*253)
        self.assertEqual(a['counts']['trio_statistics'],20)
        self.assertEqual(a['counts']['formation_return_observations'],20*252)
        self.assertTrue(all(tuple(e)==kernel.EVENT_KEYS for e in features))
        self.assertTrue(all(tuple(e)==kernel.DECISION_EVENT_KEYS for e in a['events']))
        for arm in kernel.ARMS:
            np.testing.assert_array_equal(sc[arm],sc[kernel.ARMS[0]]);np.testing.assert_array_equal(p[arm],p[kernel.ARMS[0]])
        self.assertEqual((a['counts']['new_model_fits'],a['counts']['predictive_model_fits']),(0,0))
        json.dumps(features,allow_nan=False);json.dumps(a,allow_nan=False)

    def test_actual_descriptive_path_calls_match_host_accounting(self):
        args=fixture()
        for n in (0,1,2,3):
            with patch.object(kernel,'measure',wraps=kernel.measure) as measured:
                features,*rest=run(args,n)
                count=rest[-1]['counts']['trio_statistics']
                self.assertEqual(measured.call_count,count)
            with patch.object(kernel,'measure',wraps=kernel.measure) as measured:
                cached=kernel.decisions(args[0],args[1],args[2],features,n)
                self.assertEqual(measured.call_count,count)
                self.assertEqual(cached[-1]['counts']['formation_return_observations'],252*count)
        # A common missing source means no path production or cache recomputation.
        args=fixture();args[3]['adj']=args[3]['adj'].drop(columns='A')
        with patch.object(kernel,'measure',wraps=kernel.measure) as measured:
            features,*_=run(args)
            kernel.decisions(args[0],args[1],args[2],features,3)
            self.assertEqual(measured.call_count,0)

    def test_original_incumbent_matches_frozen_study(self):
        args=fixture();s,a,m,*_=args;_,o,sc,_,_=run(args)
        choices,post=s.choices(s.configs()[4],a,m,3,policy=dict(s.POLICY_FIELDS))
        self.assertEqual(o[kernel.ARMS[0]],s.rp.make_orders(choices,post,m));np.testing.assert_array_equal(post,sc[kernel.ARMS[0]])

    def test_cached_full_artifact_exact(self):
        args=fixture();s,a,m,*_=args;fe,o,sc,p,au=run(args);oo,ss,pp,aa=kernel.decisions(s,a,m,fe,3)
        self.assertEqual(o,oo);same_arrays(self,sc,ss);same_arrays(self,p,pp);self.assertEqual(au,aa)

    def test_incumbent_gate_precedes_source_and_statistics(self):
        args=fixture();calls=[]
        def reject(*unused):calls.append('inc');return {'passed':False}
        with patch.object(kernel.old,'_source_setup',side_effect=AssertionError('Source touched')):
            with self.assertRaises(ValueError):run(args,incumbent_validator=reject)
        self.assertEqual(calls,['inc'])

    def test_source_gate_precedes_selection_and_receipts_preserved(self):
        args=fixture();seen=[]
        def inc(*unused):seen.append('inc');return {'passed':True,'scope':'old'}
        def src(features):seen.append('source');self.assertEqual(len(features),3);return {'passed':True,'scope':'full'}
        result=run(args,incumbent_validator=inc,source_validator=src)
        self.assertEqual(seen,['inc','source']);self.assertEqual(result[-1]['source_gate'],{'passed':True,'scope':'full'})
        with patch.object(kernel,'choose_trio',side_effect=AssertionError('Selection touched')):
            with self.assertRaises(ValueError):run(args,source_validator=lambda f:{'passed':False})

    def test_expected_original_orders_failure_before_stats(self):
        args=fixture();fe,o,*_=run(args);bad=deepcopy(o[kernel.ARMS[0]]);bad[0]['symbols']=[]
        with patch.object(kernel,'measure',side_effect=AssertionError('Stats touched')):
            with self.assertRaises(ValueError):run(args,expected_incumbent_orders=bad)
        run(args,expected_incumbent_orders=o[kernel.ARMS[0]])

    def test_partial_and_fewer_than_three_never_read_prices(self):
        args=fixture();args[3]['adj']=args[3]['adj'].astype(object);args[3]['adj'].iloc[:,:]='bad'
        fe,o,*_=run(args,1);self.assertEqual(fe[0]['source_cells'],[])
        for count in (0,1,2):
            aa=fixture();aa[1]['eligible'][:,count:]=False;aa[3]['adj']=args[3]['adj'].copy()
            fe,o,_,_,audit=run(aa)
            self.assertEqual(audit['counts']['source_price_observations'],0)
            for arm in kernel.ARMS:self.assertTrue(all(x['symbols']==(list('AB') if count==2 else []) for x in o[arm]))

    def test_shortlist_enumeration_no_sixth_replacement(self):
        for count,total in ((3,1),(4,4),(5,10),(6,10)):
            args=fixture();args[1]['eligible'][:,count:]=False;fe,*_=run(args,2)
            self.assertEqual(len(fe[1]['trios']),total);self.assertNotIn(5,fe[1]['shortlist_indices'])
        args=fixture();args[3]['adj']=args[3]['adj'].drop(columns='A');fe,*_=run(args,2)
        self.assertEqual(fe[1]['branch'],'missing_source');self.assertEqual(fe[1]['shortlist_indices'],[0,1,2,3,4])

    def test_missing_symbol_row_nan_shared_fallback(self):
        for reason in ('missing_symbol','missing_official_session','missing_nan'):
            args=fixture();dates=source_dates(args)
            if reason=='missing_symbol':args[3]['adj']=args[3]['adj'].drop(columns='A')
            elif reason=='missing_official_session':args[3]['adj']=args[3]['adj'].drop(index=dates[5])
            else:args[3]['adj'].loc[dates[5],'A']=np.nan
            fe,o,_,_,_=run(args,2);self.assertEqual(fe[1]['branch'],'missing_source');self.assertEqual(fe[1]['source_cells'][0]['reason'],reason)
            self.assertTrue(all(t['statistics'] is None for t in fe[1]['trios']))
            for arm in kernel.ARMS:self.assertEqual(o[arm],o[kernel.ARMS[0]])

    def test_missing_does_not_mask_later_invalid(self):
        for bad in (True,'7',None,math.inf,-1.,0.):
            args=fixture();dates=source_dates(args);args[3]['adj']=args[3]['adj'].astype(object)
            args[3]['adj'].loc[dates[0],'A']=np.nan;args[3]['adj'].loc[dates[-1],'E']=bad
            with self.subTest(bad=bad),self.assertRaises(ValueError):run(args,2)

    def test_unshortlisted_invalid_is_not_new_consumption(self):
        args=fixture();args[3]['adj']['F']='bad';fe,*_=run(args,2);self.assertEqual(fe[1]['branch'],'available')

    def test_zero_moments_are_available_not_variance_fallback(self):
        args=fixture();args[3]['adj'].iloc[:,:]=100.;fe,o,_,_,audit=run(args)
        self.assertEqual(fe[1]['branch'],'available');self.assertEqual(audit['counts']['zero_symmetric_trios'],20)
        for arm in kernel.ARMS:self.assertEqual(o[arm],o[kernel.ARMS[0]])
        args=fixture();x=np.arange(len(args[3]['adj']));p=1.0001**x
        for name in args[3]['adj']:args[3]['adj'][name]=p
        fe,_,_,_,audit=run(args,2);self.assertEqual(audit['counts']['zero_downside_trios'],10)
        self.assertEqual(audit['counts']['zero_symmetric_trios'],0);self.assertEqual(fe[1]['branch'],'available')

    def test_exact_cap_equality_and_oneulp_below(self):
        for which in ('core','final'):
            middle=-.03 if which=='core' else -.06;below=math.nextafter(math.nextafter(middle,-math.inf),-math.inf)
            for value,want in ((below,False),(middle,True)):
                args=fixture();a=args[1];a['effects'][:]=0.;a['state_calm10'][:]=True;a['core_weight'][:]=.5;a['core'][:]=0.
                if which=='core':a['core'][:,3]=value;a['pre_score'][:]=[.12,.115,.11,.105,.10,.09]
                else:a['pre_score'][:]=[0.,0.,0.,value,-.07,-.08]
                a['baseline']=a['core'].copy();fe,*_=run(args,2);t=next(t for t in fe[1]['trios'] if t['indices']==[0,1,3])
                self.assertEqual(t['feasible'],want)
                m=t['raw_core_mean'] if which=='core' else t['final_mean'];self.assertEqual(m,-.02 if want else math.nextafter(-.02,-math.inf))

    def test_complete_prelisting_conflict_even_other_missing(self):
        args=list(fixture());args[5]=make_review(args[2],args[4],[bound('E')]);args[3]['adj']=args[3]['adj'].drop(columns='A')
        with self.assertRaises(ValueError):run(args,2)

    def test_future_listing_bound_not_used_early(self):
        args=list(fixture());args[5]=make_review(args[2],args[4],[bound('A','2017-01-01')]);fe,*_=run(args,2)
        self.assertEqual(fe[1]['branch'],'available');self.assertFalse(fe[1]['source_cells'][0]['listing']['visible'])

    def test_source_metadata_identity_and_clock_tamper(self):
        args=fixture();args[5]['calendar']['windows'][1]['price_end']='2000-01-01'
        with self.assertRaises(ValueError):run(args,2)
        args=list(fixture());args[5]=make_review(args[2],args[4],[bound('A','2017-01-01')]);args[5]['listing_identity_cells'][1]['issuer_key']='wrong'
        with self.assertRaises(ValueError):run(args,2)

    def test_cached_source_domain_stat_tamper_rejected(self):
        args=fixture();fe,*_=run(args);s,a,m,*_=args
        for kind in ('domain','source','stats','branch'):
            changed=deepcopy(fe)
            if kind=='domain':changed[1]['pre_scores'][0]+=1.
            if kind=='source':changed[1]['source_cells'][0]['end_price']*=2.
            if kind=='stats':changed[1]['trios'][0]['statistics']['symmetric_moment']=math.nextafter(changed[1]['trios'][0]['statistics']['symmetric_moment'],math.inf)
            if kind=='branch':changed[1]['branch']='zero_variance'
            with self.subTest(kind=kind),self.assertRaises(ValueError):kernel.decisions(s,a,m,changed,3)

    def test_all_logical_physical_prefixes(self):
        args=fixture();features,*_=run(args)
        for n in range(4):
            actual=run(args,n);physical_result=run(physical(args,n));cached=kernel.decisions(args[0],args[1],args[2],features,n)
            self.assertEqual(actual[0],physical_result[0]);self.assertEqual(actual[1],physical_result[1]);self.assertEqual(actual[-1],physical_result[-1])
            self.assertEqual(actual[1],cached[0]);self.assertEqual(actual[-1],cached[-1]);same_arrays(self,actual[2],cached[1]);same_arrays(self,actual[3],cached[2])
            self.assertEqual(actual[-1]['counts']['original_episode_calls'],1)

    def test_n0_and_shortprefix_future_metadata_feature_poison(self):
        for n in (0,2):
            args=fixture();features,*_=run(args);expected=run(args,n)
            for key in ('dates','ends','symbols'):
                for i in range(n,3):args[2][key][i]=Poison()
            for i in range(n,3):features[i]=Poison()
            for key in args[1]:
                if args[1][key].dtype.kind!='b':args[1][key]=args[1][key].astype(object);args[1][key][n:]=Poison()
            actual=run(args,n);cached=kernel.decisions(args[0],args[1],args[2],features,n)
            self.assertEqual(actual[0],expected[0]);self.assertEqual(actual[-1],expected[-1]);self.assertEqual(cached[-1],expected[-1])

    def test_recent42_and_future_source_numeric_poison_ignored(self):
        args=fixture();expected=run(args,2);t=args[4].index(args[2]['dates'][1]);args[3]['adj']=args[3]['adj'].astype(object)
        args[3]['adj'].loc[pd.to_datetime(args[4][t-41:])]='excluded future/recent42'
        actual=run(args,2);self.assertEqual(actual[0],expected[0]);self.assertEqual(actual[-1],expected[-1])

    def test_input_and_cached_features_not_mutated(self):
        args=fixture();a=deepcopy(args[1]);m=deepcopy(args[2]);p=args[3]['adj'].copy();r=deepcopy(args[5]);features,*_=run(args);saved=deepcopy(features)
        kernel.decisions(args[0],args[1],args[2],features,3)
        same_arrays(self,a,args[1]);self.assertEqual(m,args[2]);pd.testing.assert_frame_equal(p,args[3]['adj']);self.assertEqual(r,args[5]);self.assertEqual(saved,features)

    def test_callbacks_cannot_mutate_owned_inputs(self):
        args=fixture();baseline=run(args)
        def inc(o,s,p):o.clear();s[:]=999.;p[:]=999.;return {'passed':True}
        def src(fe):fe.clear();return {'passed':True}
        actual=run(args,incumbent_validator=inc,source_validator=src)
        self.assertEqual(actual[0],baseline[0]);self.assertEqual(actual[1],baseline[1]);same_arrays(self,actual[2],baseline[2])

    def test_bool_eligibility_and_duplicate_calendar_hardfail(self):
        args=fixture();args[1]['eligible']=args[1]['eligible'].astype(object)
        with self.assertRaises(ValueError):run(args,2)
        args=list(fixture());args[4]=args[4][:10]+[args[4][9]]+args[4][10:]
        with self.assertRaises(ValueError):run(args,2)

    def test_all_states_active_and_changed_sets_use_original_F_order(self):
        args=fixture();args[1]['state_weak_narrow'][1]=True;fe,o,sc,_,au=run(args)
        self.assertTrue(fe[1]['weak']);self.assertTrue(fe[2]['calm'])
        for arm in kernel.ARMS[1:]:
            for i,e in enumerate(au['events']):
                chosen=e['choices'][arm]
                if set(chosen)==set(e['incumbent_choices']):self.assertEqual(chosen,e['incumbent_choices'])
                else:self.assertEqual(chosen,sorted(chosen,key=lambda j:(-sc[arm][i,j],args[2]['symbols'][i][j])))


if __name__=='__main__':unittest.main(verbosity=2)
