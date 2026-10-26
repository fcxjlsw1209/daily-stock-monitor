"""Invented273-price exact aggregation, source clocks and funded-score tests."""
from copy import deepcopy
import importlib.util
from fractions import Fraction
import json
import math
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import monthly_endpoint_median245_kernel as kernel


def fixture():
    dates=pd.bdate_range('2014-01-01','2017-01-05');x=np.arange(len(dates),dtype=float)
    market=100.+.01*x+2*np.sin(x*.17)
    frame=pd.DataFrame(dict(A=100.+.02*x+3*np.sin(x*.19),B=130.-.01*x+4*np.cos(x*.11),
        C=np.full(len(x),100.),D=market*2,SPY=market,FUND=90.+.03*x+5*np.sin(x*.23)),index=dates)
    names=list(frame.columns)
    meta=dict(dates=['2016-08-17','2016-08-31','2016-09-30','2016-10-31'],
        ends=['2016-08-31','2016-09-30','2016-10-31','2016-11-30'],symbols=[names.copy() for _ in range(4)],etfs=['SPY','FUND'])
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
    spec=importlib.util.spec_from_file_location('endpoint245_test_study',path);obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj


def selection_fixture():
    panel,meta,calendar=fixture();values,_=kernel.build_features(panel,meta,calendar);s=study()
    shape=(4,6);z=np.zeros(shape);f=np.zeros(shape,bool)
    calm=np.array([True,False,True,False]);weak=np.array([False,False,False,True]);cw=np.where(calm,.5,.52)[:,None]
    core=np.tile([.24,.2,.10,.06,.02,-.10],(4,1));support=np.tile([.125,-.125,.25,0.,-.25,.0625],(4,1))
    arrays=dict(baseline=core+(1.-cw)*support,core=core,core_weight=cw,eligible=np.ones(shape,bool),
        effects=np.tile([0.,0.,0.,.5,.3,.5],(4,1)),state_calm10=calm,state_weak_narrow=weak,
        pre_score=np.tile([.3,.21,.2,.195,.19,.189],(4,1)),old_ml4=z.copy(),rank_log_excess=z.copy(),
        extension_available=f.copy(),forecast_available=f.copy(),local_available=f.copy(),model_available=f.copy(),
        raw_local_constant=z.copy(),raw_local_linear=z.copy(),raw_source_log=z.copy())
    return s,arrays,meta,values


def source_dates(meta,calendar,row=0):
    t=calendar.index(meta['dates'][row]);return pd.to_datetime(calendar[t-314:t-41])


def same(case,a,b):
    case.assertEqual(tuple(a),tuple(b))
    for key in a:
        case.assertEqual(a[key].shape,b[key].shape);case.assertEqual(a[key].dtype,b[key].dtype)
        np.testing.assert_array_equal(a[key],b[key],err_msg=key)


def run(args,n=None):
    s,a,m,v=args;return kernel.decisions(s,a,m,v,len(m['dates']) if n is None else n)


def window(gains):
    p=np.ones(273);p[252:]=[1.+x for x in gains];return p


def unpack(value):return Fraction(int(value[0]),int(value[1]))


class MeasureTests(unittest.TestCase):
    def test_known_median_and_exact_arithmetic_mean(self):
        z=kernel.measure(window([-.5]*10+[.25]*11))
        self.assertEqual(tuple(z),kernel.MEASURE_KEYS);self.assertTrue(z['available'])
        self.assertEqual(z['raw_median'],['1','4']);self.assertEqual(z['raw_mean'],['-3','28'])

    def test_outlier_distinguishes_median_from_mean_no_trimming(self):
        z=kernel.measure(window([0.]*20+[1000.]))
        self.assertEqual(z['raw_median'],['0','1']);self.assertEqual(z['raw_mean'],['1000','21'])

    def test_flat_zero_and_negative_returns_are_valid(self):
        for g in (0.,-.5,.25):
            z=kernel.measure(window([g]*21));self.assertTrue(z['available'])
            self.assertEqual(unpack(z['raw_median']),Fraction.from_float(g));self.assertEqual(z['raw_median'],z['raw_mean'])

    def test_all21_pairs_used_including_first_and_last(self):
        p=window([0.]*21);p[0]=2.;p[-1]=2.;z=kernel.measure(p)
        self.assertEqual(z['raw_median'],['0','1']);self.assertEqual(z['raw_mean'],['1','42'])
        p=np.arange(1.,274.);z=kernel.measure(p)
        expected=[Fraction(253+k,1+k)-1 for k in range(21)]
        self.assertEqual(unpack(z['raw_median']),sorted(expected)[10]);self.assertEqual(unpack(z['raw_mean']),sum(expected)/21)

    def test_power_two_unit_rescaling_exact_invariance(self):
        p=window([-.5]*10+[.25]*11);z=kernel.measure(p)
        for scale in (2.**500,2.**-500):
            other=kernel.measure(p*scale)
            for key in kernel.MEASURE_KEYS:
                if key not in ('start_price','end_price'):self.assertEqual(other[key],z[key])

    def test_exact_order_survives_float_division_tie(self):
        a=1.;b=np.nextafter(a,np.inf);c=b;d=np.nextafter(c,np.inf)
        self.assertEqual(b/a,d/c)
        p=np.ones(273);p[:21]=a;p[252:]=b
        q=np.ones(273);q[:21]=c;q[252:]=d
        one=kernel.measure(p);two=kernel.measure(q)
        self.assertGreater(unpack(one['raw_median']),unpack(two['raw_median']))
        self.assertEqual(float(unpack(one['raw_median'])+1),float(unpack(two['raw_median'])+1))
        self.assertEqual(kernel._rank([unpack(one['raw_median']),unpack(two['raw_median'])]),[.5,0.])

    def test_extreme_finite_ratio_does_not_overflow_or_erase_ties(self):
        tiny=np.nextafter(0.,1.);large=np.finfo(float).max
        p=np.ones(273);p[:21]=tiny;p[252:]=large;z=kernel.measure(p)
        self.assertTrue(z['available']);self.assertEqual(z['raw_median'],z['raw_mean'])
        self.assertGreater(unpack(z['raw_median']),Fraction(10**500))
        p[:21]=large;p[252:]=tiny;z=kernel.measure(p)
        self.assertGreater(unpack(z['raw_median']),-1);self.assertLess(unpack(z['raw_median']),0)
        json.dumps(z,allow_nan=False)

    def test_all273_types_finite_positive_and_priority(self):
        for bad,reason in ((True,'nonnumeric_adjusted_price'),('1','nonnumeric_adjusted_price'),(None,'nonnumeric_adjusted_price'),
            (1+0j,'nonnumeric_adjusted_price'),(10**1000,'nonnumeric_adjusted_price'),(0.,'invalid_adjusted_price'),
            (-1.,'invalid_adjusted_price'),(np.inf,'invalid_adjusted_price'),(np.nan,'invalid_adjusted_price')):
            for j in (0,20,21,126,251,252,272):
                p=[1.]*273;p[j]=bad;z=kernel.measure(p)
                self.assertFalse(z['source_available']);self.assertEqual(z['reason'],reason);self.assertIsNone(z['raw_median']);self.assertIsNone(z['raw_mean'])
        p=[np.nan]*273;p[-1]=True;self.assertEqual(kernel.measure(p)['reason'],'nonnumeric_adjusted_price')

    def test_bad_shapes_fail(self):
        for bad in ([1.]*272,[1.]*274,np.ones((273,1)),1.):
            with self.assertRaises(ValueError):kernel.measure(bad)

    def test_exact_average_percentile_ties_and_missing(self):
        r=kernel._rank([Fraction(2),Fraction(1),Fraction(1),None,Fraction(4)])
        self.assertEqual(r[:3],[.25,-.125,-.125]);self.assertTrue(math.isnan(r[3]));self.assertEqual(r[4],.5)
        self.assertEqual(kernel._rank([Fraction(0)]),[.5])
        self.assertEqual(kernel._rank([Fraction(0)]*6),[3.5/6.-.5]*6)
        self.assertTrue(all(math.isnan(x) for x in kernel._rank([None,None])))


class BuildTests(unittest.TestCase):
    def test_full_schema_counts_clocks_and_serialization(self):
        p,m,c=fixture();v,a=kernel.build_features(p,m,c)
        self.assertEqual(tuple(v),kernel.VALUE_KEYS);self.assertEqual(tuple(a),('schema','events','counts','reason_counts','event_history'))
        self.assertEqual(tuple(a['counts']),kernel.COUNT_FIELDS);count=a['counts'];self.assertEqual(count['events'],4)
        self.assertEqual(count['source_available_cells'],24);self.assertEqual(count['available_cells'],24);self.assertEqual(count['unavailable_cells'],0)
        self.assertEqual(count['security_price_observations'],273*24);self.assertEqual(count['endpoint_pairs'],21*24)
        self.assertEqual(count['median_estimates'],24);self.assertEqual(count['mean_estimates'],24);self.assertEqual(count['ranked_cells'],48)
        self.assertEqual(count['new_model_fits'],0);self.assertEqual(count['predictive_model_fits'],0)
        for e in a['event_history']:
            self.assertEqual(tuple(e),kernel.EVENT_KEYS);self.assertEqual(e['source_dates'],c[e['session_position']-314:e['session_position']-41])
            self.assertEqual(len(e['source_dates']),273);self.assertEqual(len(e['pair_dates']),21)
            self.assertEqual(e['pair_dates'][0],dict(start=e['source_dates'][0],end=e['source_dates'][252]))
            self.assertEqual(e['pair_dates'][-1],dict(start=e['source_dates'][20],end=e['source_dates'][272]))
            self.assertTrue(all(tuple(cell)==kernel.CELL_KEYS for cell in e['cells']))
            self.assertEqual(e['cells'][3]['raw_median'],e['cells'][4]['raw_median'])
            self.assertEqual(e['cells'][3]['candidate'],e['cells'][4]['candidate'])
        json.dumps(a,allow_nan=False)

    def test_no_spy_common_mask_and_missing_own_only(self):
        for kind in ('drop','invalid'):
            p,m,c=fixture()
            if kind=='drop':p['adj']=p['adj'].drop(columns='SPY')
            else:p['adj']['SPY']=np.nan
            v,a=kernel.build_features(p,m,c)
            self.assertFalse(v['available'][:,4].any());self.assertTrue(v['available'][:,[0,1,2,3,5]].all())
            self.assertEqual(a['counts']['source_available_cells'],20);self.assertEqual(a['counts']['security_price_observations'],273*20)
            self.assertTrue(np.isnan(v['candidate'][~v['available']]).all());self.assertTrue(np.isnan(v['control'][~v['available']]).all())

    def test_all_flat_ties_are_valid_and_not_forced_to_zero(self):
        p,m,c=fixture();p['adj'][:]=100.;v,a=kernel.build_features(p,m,c)
        self.assertTrue(v['available'].all());np.testing.assert_array_equal(v['candidate'],np.full((4,6),3.5/6.-.5))
        np.testing.assert_array_equal(v['candidate'],v['control'])
        self.assertTrue(all(cell['raw_median']==['0','1'] for e in a['event_history'] for cell in e['cells']))

    def test_median_mean_rank_domains_separate_and_common(self):
        p,m,c=fixture();m['dates']=m['dates'][:1];m['symbols']=[['A','B','C']];dates=source_dates(m,c)
        p['adj'].loc[dates,'A']=window([0.]*20+[1000.]);p['adj'].loc[dates,'B']=window([1.]*21);p['adj'].loc[dates,'C']=window([-.5]*21)
        v,a=kernel.build_features(p,m,c)
        np.testing.assert_array_equal(v['candidate'],[[2./3.-.5,.5,1./3.-.5]])
        np.testing.assert_array_equal(v['control'],[[.5,2./3.-.5,1./3.-.5]])
        self.assertEqual(a['event_history'][0]['available_indices'],[0,1,2])

    def test_interior_values_not_used_in_gains_but_required_for_support(self):
        p,m,c=fixture();v,a=kernel.build_features(p,m,c,1);dates=source_dates(m,c)
        p['adj'].loc[dates[100],'A']=123456.;vv,aa=kernel.build_features(p,m,c,1);same(self,v,vv);self.assertEqual(a,aa)
        p['adj'].loc[dates[100],'A']=np.nan;vv,aa=kernel.build_features(p,m,c,1)
        self.assertFalse(vv['available'][0,0]);self.assertTrue(vv['available'][0,1:].all())

    def test_missing_official_interior_session_no_compression(self):
        p,m,c=fixture();date=source_dates(m,c)[100];p['adj']=p['adj'].drop(index=date)
        v,a=kernel.build_features(p,m,c,1);self.assertFalse(v['available'].any())
        self.assertEqual(a['event_history'][0]['missing_official_sessions'],[str(date.date())]);self.assertEqual(a['counts']['source_available_cells'],0)

    def test_recent42_and_before_window_poison_ignored(self):
        p,m,c=fixture();v,a=kernel.build_features(p,m,c,1);dates=source_dates(m,c)
        p['adj'].loc[p['adj'].index<dates[0],:]=np.nan;p['adj'].loc[p['adj'].index>dates[-1],:]=np.nan
        vv,aa=kernel.build_features(p,m,c,1);same(self,v,vv);self.assertEqual(a,aa)

    def test_inventory_permutation_and_etf_labels_do_not_filter(self):
        p,m,c=fixture();v,a=kernel.build_features(p,m,c);m['etfs']=[];vv,aa=kernel.build_features(p,m,c);same(self,v,vv);self.assertEqual(a,aa)
        perm=[5,3,0,4,2,1];m['symbols']=[[row[j] for j in perm] for row in m['symbols']];vv,aa=kernel.build_features(p,m,c)
        for key in v:np.testing.assert_array_equal(vv[key],v[key][:,perm])

    def test_all_logical_physical_prefixes_and_future_metadata_poison(self):
        p,m,c=fixture();v,a=kernel.build_features(p,m,c)
        for n in range(5):
            vv,aa=kernel.build_features(p,m,c,n);pv,pa=kernel.build_features(*physical(p,m,c,n))
            self.assertEqual(aa['event_history'],a['event_history'][:n]);self.assertEqual(aa['event_history'],pa['event_history'])
            if n:same(self,vv,pv);self.assertEqual(aa,pa)
            else:self.assertEqual(pv['available'].shape,(0,0));self.assertEqual(vv['available'].shape,(0,6))
            for key in v:np.testing.assert_array_equal(vv[key],v[key][:n])
        expected=kernel.build_features(p,m,c,1);m['dates'][1:]=['bad']*3;m['symbols'][1:]=[['bad']]*3
        actual=kernel.build_features(p,m,c,1);same(self,actual[0],expected[0]);self.assertEqual(actual[1],expected[1])

    def test_short_history_and_monthend_witness(self):
        p,m,c=fixture();m['dates']=['2014-01-31'];m['symbols']=[m['symbols'][0]];v,a=kernel.build_features(p,m,c)
        self.assertFalse(v['available'].any());self.assertIsNone(a['event_history'][0]['price_start']);self.assertEqual(a['event_history'][0]['pair_dates'],[])
        self.assertEqual(a['event_history'][0]['cells'][0]['reason'],'insufficient_history')
        p,m,c=fixture();m['dates']=['2016-08-16'];m['symbols']=[m['symbols'][0]]
        with self.assertRaises(ValueError):kernel.build_features(p,m,c)

    def test_bad_frame_calendar_and_symbol_axes(self):
        for kind in ('duplicate_date','duplicate_column','calendar_duplicate','weekend','duplicate_symbol','bad_width'):
            p,m,c=fixture()
            if kind=='duplicate_date':p['adj']=pd.concat([p['adj'].iloc[:1],p['adj']])
            elif kind=='duplicate_column':p['adj'].columns=['A','A','C','D','SPY','FUND']
            elif kind=='calendar_duplicate':c[1]=c[0]
            elif kind=='weekend':c[3]='2014-01-04'
            elif kind=='duplicate_symbol':m['symbols'][0][1]='A'
            else:m['symbols'][1]=m['symbols'][1][:-1]
            with self.subTest(kind=kind),self.assertRaises((TypeError,ValueError)):kernel.build_features(p,m,c)

    def test_build_does_not_mutate_inputs(self):
        p,m,c=fixture();frame=p['adj'].copy(deep=True);mm=deepcopy(m);cc=list(c);kernel.build_features(p,m,c)
        pd.testing.assert_frame_equal(p['adj'],frame);self.assertEqual(m,mm);self.assertEqual(c,cc)


class DecisionTests(unittest.TestCase):
    def test_original_incumbent_and_three_episodes(self):
        args=selection_fixture();s,a,m,v=args
        with patch.object(s.ep,'select_boundary',wraps=s.ep.select_boundary) as episode:
            o,sc,p,au=run(args);self.assertEqual(episode.call_count,3)
        old,post=s.choices(s.configs()[4],a,m,4,policy=dict(s.POLICY_FIELDS))
        self.assertEqual(o[kernel.ARMS[0]],s.rp.make_orders(old,post,m));np.testing.assert_array_equal(sc[kernel.ARMS[0]],post)
        self.assertEqual(tuple(au['counts']),kernel.DECISION_COUNT_FIELDS);self.assertTrue(all(tuple(e)==kernel.DECISION_EVENT_KEYS for e in au['events']))
        self.assertEqual(au['counts']['original_episode_calls'],3);self.assertEqual(au['counts']['active_events'],3);json.dumps(au,allow_nan=False)

    def test_zero_is_funded_and_missing_is_exact_noop_allstates(self):
        s,a,m,v=selection_fixture();v['candidate'][:]=0.;v['control'][:]=0.;v['available'][:,1]=False;v['candidate'][:,1]=v['control'][:,1]=np.nan
        o,sc,p,au=run((s,a,m,v));support=(a['baseline']-a['core'])/(1.-a['core_weight'])
        for i in range(1,4):
            self.assertEqual(au['events'][i]['candidate_delta'][0],-.025*support[i,0]);self.assertNotEqual(p[kernel.ARMS[1]][i,0],p[kernel.ARMS[0]][i,0])
            self.assertEqual(p[kernel.ARMS[1]][i,1],p[kernel.ARMS[0]][i,1]);self.assertTrue(au['events'][i]['active'])
        self.assertTrue(au['events'][2]['calm'])
        for arm in kernel.ARMS:self.assertEqual(o[arm][0],o[kernel.ARMS[0]][0]);np.testing.assert_array_equal(sc[arm][0],sc[kernel.ARMS[0]][0])

    def test_extension_funding_and_minimum_support(self):
        s,a,m,v=selection_fixture();a['extension_available'][:]=True;a['forecast_available'][:]=True
        a['eligible'][2,1]=False;a['forecast_available'][2,2]=False;o,sc,p,au=run((s,a,m,v))
        for i,e in enumerate(au['events']):
            for j in range(6):
                active=bool(a['state_calm10'][i] and not a['state_weak_narrow'][i] and a['extension_available'][i,j] and a['forecast_available'][i,j] and a['eligible'][i,j])
                self.assertEqual(j in e['extension_active_indices'],active)
                episode=min(.05,max(0.,.10-(.025*a['state_calm10'][i]+.05*a['state_weak_narrow'][i])))
                r0=(1.-float(a['core_weight'][i,0]))-(.025*a['state_calm10'][i]+.05*a['state_weak_narrow'][i]+.01*active)-episode
                expected=r0-.025*(i!=0 and v['available'][i,j])
                self.assertEqual(e['original_remaining_support'][j],r0);self.assertEqual(e['candidate_remaining_support'][j],expected)
                self.assertGreaterEqual(expected,.355-1e-12)
        self.assertAlmostEqual(min(au['events'][3]['candidate_remaining_support']),.355)

    def test_eligibility_not_feature_or_funding_mask(self):
        s,a,m,v=selection_fixture();a['eligible'][:,3:]=False;o,sc,p,au=run((s,a,m,v))
        self.assertEqual(au['counts']['available_active_cells'],18)
        self.assertNotEqual(au['events'][1]['candidate_delta'][4],0.)
        for arm in o.values():self.assertTrue(all(set(row['symbols'])<=set('ABC') for row in arm))

    def test_funded_pre_uses_original_support_and_final_original_boundary(self):
        s,a,m,v=selection_fixture();o,sc,p,au=run((s,a,m,v));support=(a['baseline']-a['core'])/(1.-a['core_weight'])
        for arm,key in ((kernel.ARMS[1],'candidate'),(kernel.ARMS[2],'control')):
            for i in range(1,4):
                expected=p[kernel.ARMS[0]][i]+.025*(v[key][i]-support[i]);np.testing.assert_array_equal(p[arm][i],expected)
                names=m['symbols'][i];rank=sorted(range(6),key=lambda j:(-p[arm][i,j],names[j]));j3=rank[2]
                boundary=[j for j in rank[2:5] if p[arm][i,j3]-p[arm][i,j]<=.02];winner=min(boundary,key=lambda j:(-sc[arm][i,j],names[j]))
                chosen=sorted(rank[:2]+[winner],key=lambda j:(-sc[arm][i,j],names[j]))
                self.assertEqual(o[arm][i]['symbols'],[names[j] for j in chosen])

    def test_zero_one_two_eligible_cash_or_halves(self):
        for count in (0,1,2):
            s,a,m,v=selection_fixture();a['eligible'][:,count:]=False;o,_,_,_=run((s,a,m,v))
            for arm in o.values():
                for row in arm:self.assertEqual(set(row['symbols']),{'A','B'} if count==2 else set())

    def test_original_episode_inputs_unchanged_and_no_input_mutation(self):
        s,a,m,v=selection_fixture();aa=deepcopy(a);mm=deepcopy(m);vv=deepcopy(v);original=s.ep.select_boundary;calls=[]
        def checked(pre,good,names,effect,source,policy):
            np.testing.assert_array_equal(source['baseline'],a['baseline']);np.testing.assert_array_equal(source['core'],a['core'])
            np.testing.assert_array_equal(effect,a['effects']);np.testing.assert_array_equal(good,a['eligible']);calls.append(True)
            return original(pre,good,names,effect,source,policy)
        with patch.object(s.ep,'select_boundary',side_effect=checked):run((s,a,m,v))
        self.assertEqual(len(calls),3);same(self,a,aa);same(self,v,vv);self.assertEqual(m,mm)

    def test_all_decision_prefixes_including_n0_and_future_poison(self):
        s,a,m,v=selection_fixture();o,sc,p,au=run((s,a,m,v))
        for n in range(5):
            args=(s,{k:x[:n].copy() for k,x in a.items()},dict(m,dates=m['dates'][:n],ends=m['ends'][:n],symbols=m['symbols'][:n]),{k:x[:n].copy() for k,x in v.items()})
            logical=run((s,a,m,v),n);physical_result=run(args)
            self.assertEqual(logical[0],physical_result[0]);same(self,logical[1],physical_result[1]);same(self,logical[2],physical_result[2]);self.assertEqual(logical[3],physical_result[3])
            self.assertEqual(logical[3]['events'],au['events'][:n]);self.assertEqual(logical[3]['counts']['original_episode_calls'],3)
        expected=run((s,a,m,v),1)
        for key in ('baseline','core','effects','pre_score','rank_log_excess','raw_local_constant','raw_local_linear','raw_source_log'):a[key][1:]=np.nan
        a['core_weight']=a['core_weight'].astype(object);a['core_weight'][1:]=None
        v['candidate'][1:]=v['control'][1:]=np.nan;m['dates'][1:]=['bad']*3;m['symbols'][1:]=[['bad']]*3;m['ends'][1:]=['bad']*3
        actual=run((s,a,m,v),1);self.assertEqual(actual[0],expected[0]);same(self,actual[1],expected[1]);same(self,actual[2],expected[2]);self.assertEqual(actual[3],expected[3])

    def test_regular_first_row_not_bypassed(self):
        s,a,m,v=selection_fixture();aa={k:x[1:].copy() for k,x in a.items()};vv={k:x[1:].copy() for k,x in v.items()}
        mm=dict(m,dates=m['dates'][1:],ends=m['ends'][1:],symbols=m['symbols'][1:]);o,sc,p,au=run((s,aa,mm,vv))
        self.assertTrue(au['events'][0]['active']);self.assertTrue(any(x!=0. for x in au['events'][0]['candidate_delta']))

    def test_invalid_features_core_weight_and_source_types_fail(self):
        for kind in ('missing','mask','finite_missing','nan_available','outofbounds','boolfeature','coreweight','boolcore'):
            s,a,m,v=selection_fixture()
            if kind=='missing':del v['control']
            elif kind=='mask':v['available']=v['available'].astype(int)
            elif kind=='finite_missing':v['available'][0,0]=False
            elif kind=='nan_available':v['candidate'][0,0]=np.nan
            elif kind=='outofbounds':v['candidate'][0,0]=.51
            elif kind=='boolfeature':v['candidate']=v['candidate'].astype(object);v['candidate'][0,0]=True
            elif kind=='coreweight':a['core_weight'][0,0]=.51
            else:a['core']=a['core'].astype(object);a['core'][0,0]=True
            with self.subTest(kind=kind),self.assertRaises(ValueError):run((s,a,m,v))

    def test_mutating_pre_or_nonfinite_post_rejected(self):
        s,a,m,v=selection_fixture();original=s.modified_pre_score
        def mutate(spec,arrays,n):
            pre=original(spec,arrays,n);arrays['effects'][0,0]+=.01;return pre
        with patch.object(s,'modified_pre_score',side_effect=mutate),self.assertRaises(ValueError):run((s,a,m,v))
        s,a,m,v=selection_fixture();original=s.ep.select_boundary
        def bad(*args):
            choices,score=original(*args);score[0,0]=np.inf;return choices,score
        with patch.object(s.ep,'select_boundary',side_effect=bad),self.assertRaises(ValueError):run((s,a,m,v))

    def test_frozen_dependencies(self):
        self.assertEqual(len(kernel.dependency_files()),6)
        self.assertEqual(kernel.hashlib.sha256(kernel.AXES_PATH.read_bytes()).hexdigest(),kernel.AXES_SHA)
        self.assertEqual(kernel.hashlib.sha256(kernel.ADAPTER_PATH.read_bytes()).hexdigest(),kernel.ADAPTER_SHA)


if __name__=='__main__':unittest.main(verbosity=2)
