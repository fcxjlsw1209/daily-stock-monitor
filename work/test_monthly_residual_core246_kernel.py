"""Invented-source rolling QR, exact-domain and original-episode checks."""
from copy import deepcopy
from functools import lru_cache
import importlib.util
import json
import math
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import monthly_residual_core246_kernel as kernel


def prices(returns,start=100.):
    out=[float(start)]
    for r in returns:out.append(out[-1]*(1.+float(r)))
    return np.array(out)


def scalar_fixture():
    x=np.arange(504,dtype=float)
    m=.001+.006*np.sin(.173*x)+.003*np.cos(.071*x)
    s=.0006+.004*np.cos(.291*x)-.002*np.sin(.113*x)
    y=.0008+.7*m+.4*s+.001*np.sin(.419*x)
    return prices(y),prices(m),prices(s)


def fixture():
    dates=pd.bdate_range('2013-01-01','2017-01-05');x=np.arange(len(dates)-1,dtype=float)
    m=.0004+.004*np.sin(.173*x)+.002*np.cos(.071*x)
    s=.0002+.003*np.cos(.291*x)-.001*np.sin(.113*x)
    frame=pd.DataFrame(dict(A=prices(.0006+.7*m+.4*s+.0005*np.sin(.419*x)),
        B=prices(-.0002+.3*m-.6*s+.001*np.cos(.221*x)),C=prices(.0001+.9*m+.2*s+.0012*np.sin(.331*x)),
        D=prices(.0004-.2*m+.7*s+.0009*np.cos(.371*x)),SPY=prices(m),FUND=prices(.0001+.2*m),XLK=prices(s)),index=dates)
    names=['A','B','C','D','SPY','FUND']
    meta=dict(dates=['2016-08-17','2016-08-31','2016-09-30','2016-10-31'],ends=['2016-08-31','2016-09-30','2016-10-31','2016-11-30'],
        symbols=[names.copy() for _ in range(4)],etfs=['SPY','FUND'])
    panel={'adj':frame,'vox':pd.DataFrame({'VOX':prices(s)},index=dates)}
    return panel,meta,dates.strftime('%Y-%m-%d').tolist(),make_review(meta)


def make_review(meta):
    records=[]
    for i,(date,names) in enumerate(zip(meta['dates'],meta['symbols'])):
        for j,symbol in enumerate(names):
            if symbol in meta['etfs']:continue
            records.append(dict(query_index=i,column_index=j,date=date,symbol=symbol,issuer_key='synthetic:'+symbol,
                original_sector='XLK',sector='XLK',mapping_origin='synthetic_query_mapping',mapping_repair='unchanged',mapping_evidence=None))
    return dict(mapping_records=records)


def physical(panel,meta,calendar,review,n):
    m=deepcopy(meta)
    for key in ('dates','ends','symbols'):m[key]=m[key][:n]
    if n:
        t=calendar.index(m['dates'][-1]);cal=calendar[:t+2];p={key:frame.loc[:m['dates'][-1]].copy() for key,frame in panel.items()}
    else:cal=[];p={key:frame.iloc[:0].copy() for key,frame in panel.items()}
    return p,m,cal,deepcopy(review)


def source_dates(meta,calendar,row=0):
    t=calendar.index(meta['dates'][row]);return pd.to_datetime(calendar[t-546:t-41])


def study():
    path=kernel.ROOT/'outputs/monthly_local_linear70/study.py'
    spec=importlib.util.spec_from_file_location('residual246_test_study',path);obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj


@lru_cache(None)
def _selection_fixture():
    p,m,c,r=fixture();v,_=kernel.build_features(p,m,c,r);s=study();shape=(4,6);z=np.zeros(shape);f=np.zeros(shape,bool)
    calm=np.array([True,False,True,False]);weak=np.array([False,False,False,True]);cw=np.where(calm,.5,.52)[:,None]
    core=np.tile([.24,.20,.10,.06,.02,-.10],(4,1));support=np.tile([.125,-.125,.25,0.,-.25,.0625],(4,1))
    a=dict(baseline=core+(1.-cw)*support,core=core,core_weight=cw,eligible=np.ones(shape,bool),effects=np.tile([0.,0.,0.,.5,.3,.5],(4,1)),
        state_calm10=calm,state_weak_narrow=weak,pre_score=np.tile([.3,.21,.2,.195,.19,.189],(4,1)),old_ml4=z.copy(),rank_log_excess=z.copy(),
        extension_available=f.copy(),forecast_available=f.copy(),local_available=f.copy(),model_available=f.copy(),
        raw_local_constant=z.copy(),raw_local_linear=z.copy(),raw_source_log=z.copy())
    return s,a,m,v


def selection_fixture():
    s,a,m,v=_selection_fixture();return s,deepcopy(a),deepcopy(m),deepcopy(v)


def run(args,n=None):
    s,a,m,v=args;return kernel.decisions(s,a,m,v,len(m['dates']) if n is None else n)


def same(case,a,b):
    case.assertEqual(tuple(a),tuple(b))
    for key in a:np.testing.assert_array_equal(a[key],b[key],err_msg=key)


class MeasureTests(unittest.TestCase):
    def test_complete_lagged_fit_schema(self):
        z=kernel.measure(*scalar_fixture());self.assertEqual(tuple(z),kernel.MEASURE_KEYS);self.assertTrue(z['available'])
        self.assertEqual(tuple(z['fit_arrays']),kernel.FIT_KEYS)
        self.assertTrue(all(len(a)==252 for a in z['fit_arrays'].values()));self.assertTrue(all(z['fit_arrays']['fit_available']))
        self.assertEqual(z['raw_candidate'],z['joint_adjusted_mean']/z['stock_sample_vol']);self.assertEqual(z['raw_control'],z['market_adjusted_mean']/z['stock_sample_vol'])
        json.dumps(z,allow_nan=False)

    def test_exact_model_retains_alpha_and_common_total_risk(self):
        y,m,s=scalar_fixture();mr=kernel._returns(m);sr=kernel._returns(s);alpha=.0008
        y=prices([alpha+.7*a+.4*b for a,b in zip(mr,sr)]);z=kernel.measure(y,m,s)
        np.testing.assert_allclose(z['fit_arrays']['beta_market_joint'],.7,rtol=0,atol=1e-12)
        np.testing.assert_allclose(z['fit_arrays']['beta_sector_joint'],.4,rtol=0,atol=1e-12)
        self.assertAlmostEqual(z['joint_adjusted_mean'],alpha,places=14)
        self.assertGreater(abs(z['joint_adjusted_mean']),1e-5)
        self.assertNotEqual(z['joint_adjusted_mean'],z['market_adjusted_mean'])
        ret=kernel._returns(y)[252:];mean=math.fsum(ret)/252.;sd=math.sqrt(math.fsum((a-mean)**2 for a in ret)/251.)
        self.assertEqual(z['stock_sample_vol'],sd)

    def test_application_day_and_future_excluded_from_its_fit(self):
        y,m,s=scalar_fixture();z=kernel.measure(y,m,s)
        for day in (253,400,504):
            altered=y.copy();altered[day:]*=1.1;q=kernel.measure(altered,m,s);idx=day-253
            for field in ('beta_market_only','beta_market_joint','beta_sector_joint','conditioning'):
                self.assertEqual(q['fit_arrays'][field][:idx+1],z['fit_arrays'][field][:idx+1])
            self.assertNotEqual(q['fit_arrays']['adjusted_joint'][idx],z['fit_arrays']['adjusted_joint'][idx])

    def test_one_bad_training_window_invalidates_whole_cell(self):
        y,m,s=scalar_fixture();s[:253]=m[:253];z=kernel.measure(y,m,s)
        self.assertFalse(z['available']);self.assertFalse(z['fit_arrays']['fit_available'][0]);self.assertTrue(any(z['fit_arrays']['fit_available'][1:]))
        self.assertEqual(z['reason'],'collinear_factors');self.assertIsNone(z['raw_candidate']);self.assertIsNone(z['market_adjusted_sum'])

    def test_constant_market_sector_and_zero_stock_risk(self):
        y,m,s=scalar_fixture()
        for inputs,reason in (((y,np.ones(505),s),'zero_market_variance'),((y,m,np.ones(505)),'zero_sector_variance'),((y,m,m),'collinear_factors')):
            z=kernel.measure(*inputs);self.assertFalse(z['available']);self.assertEqual(z['reason'],reason);self.assertEqual(sum(z['fit_arrays']['fit_available']),0)
        z=kernel.measure(np.ones(505),m,s);self.assertFalse(z['available']);self.assertEqual(z['reason'],'zero_stock_sample_volatility')
        self.assertTrue(all(z['fit_arrays']['fit_available']));self.assertEqual(z['fit_arrays']['beta_market_joint'],[0.]*252);self.assertEqual(z['joint_adjusted_mean'],0.)

    def test_gate_strict_threshold_and_negative_correlation(self):
        x=np.arange(252,dtype=float);m=np.sin(x*.17);s=np.cos(x*.29);y=np.sin(x*.11)
        fit=kernel._fit(y,m,s)
        with patch.object(kernel,'CONDITION_MIN',fit['conditioning']):self.assertFalse(kernel._fit(y,m,s)['fit_available'])
        with patch.object(kernel,'CONDITION_MIN',np.nextafter(fit['conditioning'],-np.inf)):self.assertTrue(kernel._fit(y,m,s)['fit_available'])
        self.assertFalse(kernel._fit(y,m,-m)['fit_available'])
        for eps,expected in ((1e-3,True),(1e-5,False)):
            self.assertEqual(kernel._fit(y,m,m+eps*s)['fit_available'],expected)

    def test_scaled_design_and_constant_training_response(self):
        x=np.arange(252,dtype=float);m=np.sin(x*.17);s=np.cos(x*.29);y=.7*m+.4*s+.01
        reference=kernel._fit(y,m,s);scaled=kernel._fit(y,m*1e8,s*1e-8)
        self.assertAlmostEqual(scaled['beta_market_joint']*1e8,reference['beta_market_joint'],places=11)
        self.assertAlmostEqual(scaled['beta_sector_joint']*1e-8,reference['beta_sector_joint'],places=11)
        z=kernel._fit(np.zeros(252),m,s);self.assertTrue(z['fit_available']);self.assertEqual(z['beta_market_only'],0.);self.assertEqual(z['beta_sector_joint'],0.)

    def test_zero_negative_drift_and_negative_beyond_one_statistics_valid(self):
        y,m,s=scalar_fixture();mr=kernel._returns(m);sr=kernel._returns(s)
        z=kernel.measure(prices([-.0008+.7*a+.4*b for a,b in zip(mr,sr)]),m,s)
        self.assertTrue(z['available']);self.assertLess(z['raw_candidate'],0.)
        a=kernel._fit(np.zeros(252),mr[:252],sr[:252]);self.assertEqual(a['beta_market_joint'],0.)
        yr=[.0001+2.*a+.4*b for a,b in zip(mr,sr)];mr[-1]=.9;yr[-1]=.01
        extreme=kernel.measure(prices(yr),prices(mr),prices(sr));self.assertTrue(extreme['available'])
        self.assertLess(extreme['fit_arrays']['adjusted_joint'][-1],-1.)

    def test_source_validation_priority_and_missing_numeric(self):
        y,m,s=scalar_fixture()
        for bad,reason in ((True,'nonnumeric_adjusted_price'),('1','nonnumeric_adjusted_price'),(None,'nonnumeric_adjusted_price'),(10**1000,'nonnumeric_adjusted_price'),(0.,'invalid_adjusted_price'),(-1.,'invalid_adjusted_price'),(np.inf,'invalid_adjusted_price'),(np.nan,'invalid_adjusted_price')):
            q=y.astype(object);q[250]=bad;z=kernel.measure(q,m,s);self.assertFalse(z['stock_source_available']);self.assertTrue(z['market_source_available']);self.assertEqual(z['reason'],'stock_'+reason)
        q=y.astype(object);q[0]=np.nan;q[-1]=True;self.assertEqual(kernel.measure(q,m,s)['stock_source_reason'],'nonnumeric_adjusted_price')
        for q in (y[:-1],np.ones((505,1)),1.):
            with self.assertRaises(ValueError):kernel.measure(q,m,s)

    def test_arithmetic_overflow_underflow_abort_not_missing(self):
        y,m,s=scalar_fixture()
        for a,b in ((np.nextafter(0.,1.),np.finfo(float).max),(np.finfo(float).max,np.nextafter(0.,1.))):
            q=y.copy();q[0]=a;q[1]=b
            with self.assertRaises(FloatingPointError):kernel.measure(q,m,s)
        with self.assertRaises(FloatingPointError):kernel._squares([1e-200])
        with self.assertRaises(FloatingPointError):kernel._divide(np.nextafter(0.,1.),2.)

    def test_price_units_and_exact_rank_ties(self):
        y,m,s=scalar_fixture();z=kernel.measure(y,m,s);q=kernel.measure(y*8,m*.5,s*4)
        for key in ('raw_candidate','raw_control','fit_arrays','stock_sample_vol'):self.assertEqual(z[key],q[key])
        np.testing.assert_array_equal(kernel._rank([2.,1.,1.,None],np.array([True,True,True,False])),[.5,0.,0.,np.nan])
        r=kernel._rank([1.,np.nextafter(1.,2.)],np.array([True,True]));self.assertEqual(r.tolist(),[0.,.5])


class BuildTests(unittest.TestCase):
    def test_schema_counts_clocks_source_and_no_etf_targets(self):
        p,m,c,r=fixture();v,a=kernel.build_features(p,m,c,r,1);counts=a['counts'];e=a['event_history'][0]
        self.assertEqual(tuple(v),kernel.VALUE_KEYS);self.assertEqual(tuple(counts),kernel.COUNT_FIELDS);self.assertEqual(tuple(e),kernel.EVENT_KEYS)
        self.assertEqual(counts['stock_cells'],4);self.assertEqual(counts['etf_cells'],2);self.assertEqual(counts['stock_price_observations'],2020)
        self.assertEqual(counts['rolling_windows_checked'],1008);self.assertEqual(counts['descriptive_joint_estimates'],1008);self.assertEqual(counts['new_model_fits'],0)
        self.assertEqual(e['source_dates'],c[e['session_position']-546:e['session_position']-41]);self.assertEqual(len(e['fit_dates']),252)
        for j,d in enumerate(range(253,505)):
            self.assertEqual(e['fit_dates'][j],dict(training_price_start=e['source_dates'][d-253],training_return_start=e['source_dates'][d-252],training_return_end=e['source_dates'][d-1],application_date=e['source_dates'][d]))
        self.assertTrue(v['available'][0,:4].all());self.assertFalse(v['available'][0,4:].any());self.assertTrue(np.isnan(v['candidate'][0,4:]).all())
        self.assertTrue(all(tuple(x)==kernel.CELL_KEYS for x in e['cells']));json.dumps(a,allow_nan=False)

    def test_unresolved_mapping_still_checks_own_stock_independently(self):
        p,m,c,r=fixture();r['mapping_records'][0]['sector']=None;r['mapping_records'][0]['mapping_repair']='unresolved'
        v,a=kernel.build_features(p,m,c,r,1);cell=a['event_history'][0]['cells'][0]
        self.assertTrue(cell['stock_source_available']);self.assertTrue(cell['market_source_available']);self.assertFalse(cell['sector_source_available']);self.assertEqual(cell['reason'],'unresolved_mapping')
        p['adj'].loc[source_dates(m,c)[50],'A']=np.nan;v,a=kernel.build_features(p,m,c,r,1)
        self.assertFalse(a['event_history'][0]['cells'][0]['stock_source_available'])

    def test_dedicated_vox_missing_does_not_fall_back_to_adj(self):
        p,m,c,r=fixture();r['mapping_records'][0]['sector']='VOX';p['adj']['VOX']=p['adj']['XLK'];p['vox']=p['vox'].iloc[1000:]
        v,a=kernel.build_features(p,m,c,r,1);self.assertFalse(v['available'][0,0]);self.assertTrue(v['available'][0,1:4].all())
        self.assertEqual(a['event_history'][0]['cells'][0]['sector_source_reason'],'missing_official_session')

    def test_missing_full_history_rows_and_columns_no_fill(self):
        for kind in ('column','row','market','short'):
            p,m,c,r=fixture()
            if kind=='column':p['adj']=p['adj'].drop(columns='A')
            elif kind=='market':p['adj']=p['adj'].drop(columns='SPY')
            elif kind=='row':p['adj']=p['adj'].drop(index=source_dates(m,c)[101])
            else:c=c[-550:]
            v,a=kernel.build_features(p,m,c,r,1);self.assertFalse(v['available'][0,0]);self.assertTrue(np.isnan(v['candidate'][~v['available']]).all())

    def test_earlier_and_latest42_future_poison_ignored(self):
        p,m,c,r=fixture();v,a=kernel.build_features(p,m,c,r,1);ds=source_dates(m,c)
        for frame in p.values():frame.loc[(frame.index<ds[0])|(frame.index>ds[-1]),:]=np.nan
        vv,aa=kernel.build_features(p,m,c,r,1);same(self,v,vv);self.assertEqual(a,aa)

    def test_exact_mapping_inventory_identity_duplicates_and_fields(self):
        for kind in ('missing','duplicate','etf','issuer','date','extra','wrongindex'):
            p,m,c,r=fixture()
            if kind=='missing':r['mapping_records'].pop(0)
            elif kind=='duplicate':r['mapping_records'].append(deepcopy(r['mapping_records'][0]))
            elif kind=='etf':r['mapping_records'][0].update(column_index=4,symbol='SPY')
            elif kind=='issuer':r['mapping_records'][1]['issuer_key']=r['mapping_records'][0]['issuer_key']
            elif kind=='date':r['mapping_records'][0]['date']='2016-08-31'
            elif kind=='extra':r['mapping_records'][0]['eligible']=True
            else:r['mapping_records'][0]['column_index']=True
            with self.subTest(kind=kind),self.assertRaises(ValueError):kernel.build_features(p,m,c,r,1)

    def test_all_logical_physical_prefixes_and_future_mapping_poison(self):
        p,m,c,r=fixture();v,a=kernel.build_features(p,m,c,r)
        for n in range(5):
            vv,aa=kernel.build_features(p,m,c,r,n);pv,pa=kernel.build_features(*physical(p,m,c,r,n))
            for key in v:np.testing.assert_array_equal(vv[key],v[key][:n]);np.testing.assert_array_equal(pv[key],v[key][:n,:] if n else np.empty((0,0),dtype=v[key].dtype))
            self.assertEqual(aa['event_history'],a['event_history'][:n]);self.assertEqual(pa['event_history'],aa['event_history'])
        expected=kernel.build_features(p,m,c,r,1)
        m['dates'][1:]=['bad']*3;m['symbols'][1:]=[['bad']]*3
        r['mapping_records']=[x if x['query_index']==0 else dict(query_index=x['query_index'],bad=object()) for x in r['mapping_records']]
        actual=kernel.build_features(p,m,c,r,1);same(self,actual[0],expected[0]);self.assertEqual(actual[1],expected[1])

    def test_no_input_mutation_and_bad_calendar_axes(self):
        p,m,c,r=fixture();pp={k:v.copy(deep=True) for k,v in p.items()};mm=deepcopy(m);rr=deepcopy(r);kernel.build_features(p,m,c,r,1)
        for k in p:pd.testing.assert_frame_equal(p[k],pp[k])
        self.assertEqual(m,mm);self.assertEqual(r,rr)
        with self.assertRaises(ValueError):kernel.build_features(p,m,c+[c[-1]],r,1)
        p['adj']=pd.concat([p['adj'],p['adj'].iloc[-1:]])
        with self.assertRaises(ValueError):kernel.build_features(p,m,c,r,1)


class DecisionTests(unittest.TestCase):
    def test_incumbent_exact_and_three_original_episodes(self):
        s,a,m,v=selection_fixture()
        with patch.object(s.ep,'select_boundary',wraps=s.ep.select_boundary) as ep:o,sc,p,au=run((s,a,m,v));self.assertEqual(ep.call_count,3)
        old,post=s.choices(s.configs()[4],a,m,4,policy=dict(s.POLICY_FIELDS));self.assertEqual(o[kernel.ARMS[0]],s.rp.make_orders(old,post,m));np.testing.assert_array_equal(sc[kernel.ARMS[0]],post)
        self.assertEqual(tuple(au['counts']),kernel.DECISION_COUNT_FIELDS);self.assertTrue(all(tuple(e)==kernel.DECISION_EVENT_KEYS for e in au['events']));self.assertEqual(au['counts']['active_events'],3)

    def test_halfcore_all_states_zero_valid_etf_missing_noop(self):
        s,a,m,v=selection_fixture();v['candidate'][v['available']]=0.;v['control'][v['available']]=0.
        v['available'][:,1]=False;v['candidate'][:,1]=v['control'][:,1]=np.nan;o,sc,p,au=run((s,a,m,v))
        for i in range(1,4):
            self.assertEqual(au['events'][i]['candidate_delta'][0],-.5*a['core'][i,0]);self.assertTrue(au['events'][i]['active'])
            for j in (1,4,5):self.assertEqual(p[kernel.ARMS[1]][i,j],p[kernel.ARMS[0]][i,j])
        for arm in kernel.ARMS:np.testing.assert_array_equal(sc[arm][0],sc[kernel.ARMS[0]][0]);self.assertEqual(o[arm][0],o[kernel.ARMS[0]][0])

    def test_original_support_budget_unchanged_including_calm_extension(self):
        s,a,m,v=selection_fixture();a['extension_available'][:]=True;a['forecast_available'][:]=True;o,sc,p,au=run((s,a,m,v))
        for event in au['events']:
            self.assertEqual(event['original_remaining_support'],event['candidate_remaining_support']);self.assertEqual(event['original_remaining_support'],event['control_remaining_support'])
        self.assertEqual(au['events'][2]['extension_active_indices'],list(range(6)))
        self.assertAlmostEqual(min(au['events'][3]['original_remaining_support']),.38)

    def test_Efalse_does_not_remove_core_domain_or_change_E(self):
        s,a,m,v=selection_fixture();a['eligible'][:,3:]=False;o,sc,p,au=run((s,a,m,v))
        self.assertNotEqual(au['events'][1]['candidate_delta'][3],0.);self.assertEqual(au['counts']['available_active_cells'],12)
        for arm in o.values():self.assertTrue(all(set(row['symbols'])<=set('ABC') for row in arm))

    def test_halfcore_pre_and_single_original_boundary_not_global_rerank(self):
        s,a,m,v=selection_fixture();o,sc,p,au=run((s,a,m,v))
        for arm,key in ((kernel.ARMS[1],'candidate'),(kernel.ARMS[2],'control')):
            for i in range(1,4):
                delta=np.zeros(6);mask=v['available'][i];delta[mask]=.5*(a['core_weight'][i,0]*v[key][i,mask]-a['core'][i,mask]);expected=p[kernel.ARMS[0]][i].copy();expected[delta!=0.]+=delta[delta!=0.]
                np.testing.assert_array_equal(p[arm][i],expected);names=m['symbols'][i];order=sorted(range(6),key=lambda j:(-p[arm][i,j],names[j]));r3=order[2]
                boundary=[j for j in order[2:5] if p[arm][i,r3]-p[arm][i,j]<=.02];winner=min(boundary,key=lambda j:(-sc[arm][i,j],names[j]));chosen=sorted(order[:2]+[winner],key=lambda j:(-sc[arm][i,j],names[j]))
                self.assertEqual(o[arm][i]['symbols'],[names[j] for j in chosen])

    def test_zero_one_two_E_cash_or_halves(self):
        for count in (0,1,2):
            s,a,m,v=selection_fixture();a['eligible'][:,count:]=False;o,_,_,_=run((s,a,m,v))
            for arm in o.values():
                for row in arm:self.assertEqual(set(row['symbols']),{'A','B'} if count==2 else set())

    def test_original_core_effects_support_unmutated(self):
        s,a,m,v=selection_fixture();aa=deepcopy(a);mm=deepcopy(m);vv=deepcopy(v);original=s.ep.select_boundary;calls=[]
        def checked(pre,good,names,effects,source,policy):
            np.testing.assert_array_equal(source['baseline'],a['baseline']);np.testing.assert_array_equal(source['core'],a['core']);np.testing.assert_array_equal(effects,a['effects']);np.testing.assert_array_equal(good,a['eligible']);calls.append(True)
            return original(pre,good,names,effects,source,policy)
        with patch.object(s.ep,'select_boundary',side_effect=checked):run((s,a,m,v))
        self.assertEqual(len(calls),3);same(self,a,aa);same(self,v,vv);self.assertEqual(m,mm)

    def test_all_decision_prefixes_and_n0_actual_adapter(self):
        s,a,m,v=selection_fixture();full=run((s,a,m,v))
        for n in range(5):
            logical=run((s,a,m,v),n);physical_args=(s,{k:x[:n].copy() for k,x in a.items()},dict(m,dates=m['dates'][:n],ends=m['ends'][:n],symbols=m['symbols'][:n]),{k:x[:n].copy() for k,x in v.items()})
            actual=run(physical_args);self.assertEqual(logical[0],actual[0]);same(self,logical[1],actual[1]);same(self,logical[2],actual[2]);self.assertEqual(logical[3],actual[3]);self.assertEqual(logical[3]['events'],full[3]['events'][:n]);self.assertEqual(logical[3]['counts']['original_episode_calls'],3)

    def test_decision_future_poison_and_regular_firstrow(self):
        s,a,m,v=selection_fixture();expected=run((s,a,m,v),1)
        for key in ('baseline','core','effects','pre_score','rank_log_excess','raw_local_constant','raw_local_linear','raw_source_log'):a[key][1:]=np.nan
        a['core_weight']=a['core_weight'].astype(object);a['core_weight'][1:]=None;v['candidate'][1:]=v['control'][1:]=np.nan
        m['dates'][1:]=['bad']*3;m['ends'][1:]=['bad']*3;m['symbols'][1:]=[['bad']]*3;actual=run((s,a,m,v),1)
        self.assertEqual(actual[0],expected[0]);same(self,actual[1],expected[1]);self.assertEqual(actual[3],expected[3])
        s,a,m,v=selection_fixture();a={k:x[1:] for k,x in a.items()};v={k:x[1:] for k,x in v.items()};m=dict(m,dates=m['dates'][1:],ends=m['ends'][1:],symbols=m['symbols'][1:]);actual=run((s,a,m,v));self.assertTrue(actual[3]['events'][0]['active'])

    def test_invalid_target_mask_etf_and_core_types_fail(self):
        for kind in ('etf','mask','missing','nan','bound','core','weight'):
            s,a,m,v=selection_fixture()
            if kind=='etf':v['available'][0,4]=True;v['candidate'][0,4]=v['control'][0,4]=0.
            elif kind=='mask':v['available']=v['available'].astype(int)
            elif kind=='missing':v['available'][0,0]=False
            elif kind=='nan':v['candidate'][0,0]=np.nan
            elif kind=='bound':v['candidate'][0,0]=.51
            elif kind=='core':a['core']=a['core'].astype(object);a['core'][0,0]=True
            else:a['core_weight'][0,0]=.51
            with self.subTest(kind=kind),self.assertRaises(ValueError):run((s,a,m,v))

    def test_mutating_or_nonfinite_callbacks_hardfail(self):
        s,a,m,v=selection_fixture();old=s.modified_pre_score
        def mutation(spec,arrays,n):
            out=old(spec,arrays,n);arrays['effects'][0,0]+=.01;return out
        with patch.object(s,'modified_pre_score',side_effect=mutation),self.assertRaises(ValueError):run((s,a,m,v))
        s,a,m,v=selection_fixture();old=s.ep.select_boundary
        def invalid(*args):
            choices,score=old(*args);score[0,0]=np.inf;return choices,score
        with patch.object(s.ep,'select_boundary',side_effect=invalid),self.assertRaises(ValueError):run((s,a,m,v))

    def test_frozen_dependency_hashes(self):
        self.assertEqual(len(kernel.dependency_files()),7)
        self.assertEqual(kernel.hashlib.sha256(kernel.AXES_PATH.read_bytes()).hexdigest(),kernel.AXES_SHA)
        self.assertEqual(kernel.hashlib.sha256(kernel.ADAPTER_PATH.read_bytes()).hexdigest(),kernel.ADAPTER_SHA)


if __name__=='__main__':unittest.main(verbosity=2)
