"""Synthetic four-control, same-fit rank/slot, identity and freeze tests only."""
import ast
import copy
import importlib.util
from contextlib import ExitStack
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

P=Path(__file__).resolve().parent
SPEC=importlib.util.spec_from_file_location('density70_study_tests',P/'study.py')
s=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(s)
POLICY=dict(s.POLICY_FIELDS)


def fixture():
    shape=(4,6);cw=np.array([.52,.5,.52,.5])[:,None]
    support=np.tile([.1,-.1,.05,0.,-.05,.1],(4,1))
    core=cw*np.tile([.45,.4,.3,.29,.28,.27],(4,1))
    a=dict(baseline=core+(1-cw)*support,core=core,core_weight=cw,
        eligible=np.ones(shape,bool),state_calm10=np.array([False,True,False,True]),
        state_weak_narrow=np.array([False,False,True,True]),
        pre_score=np.tile([.30,.25,.20,.19,.181,.1805],(4,1)),
        effects=np.tile([0.,0.,-.10,.05,.15,.50],(4,1)),
        forecast_available=np.ones(shape,bool),extension_available=np.ones(shape,bool),
        old_ml4=np.zeros(shape),rank_log_excess=np.tile([-.4,-.3,-.2,.1,.3,.5],(4,1)),
        model_available=np.ones(shape,bool),local_available=np.ones(shape,bool),
        raw_source_log=np.tile([-.06,-.05,-.04,.02,.04,.06],(4,1)),
        raw_local_constant=np.tile([5.,3.,1.,-2.,-3.,-4.],(4,1)),
        raw_local_linear=np.tile([-4.,-3.,-2.,1.,3.,5.],(4,1)),
        density_source_raw=np.tile([-.06,-.05,-.04,.02,.04,.06],(4,1)),
        density_raw_month21=np.tile([5.,3.,1.,-2.,-3.,-4.],(4,1)),
        density_available=np.ones(shape,bool),density_raw_union21=np.tile([-4.,-3.,-2.,1.,3.,5.],(4,1)))
    return a,dict(symbols=[list('ABCDEF') for _ in range(4)],
        dates=['2016-08-17','2016-08-31','2016-09-30','2016-10-31'],
        ends=['2016-08-31','2016-09-30','2016-10-31','2016-11-30'])


class StudyTests(unittest.TestCase):
    def setUp(self):self.a,self.m=fixture()

    def choose(self,ident,n=4,**kw):return s.choices(s.configs()[ident],self.a,self.m,n,policy=POLICY,**kw)

    def ranks(self,n=4):
        return s.ranked_forecasts({key:self.a['density_raw_'+key] for key in s.ARMS},self.a['density_available'],self.a['density_source_raw'],self.a,n)

    def rows(self):
        def metric(v):return dict(cagr=v,sharpe=1.6,max_drawdown=-.3)
        return [dict(spec=z,training=metric(.5 if z['id']==3 else .49),early=metric(.4),recent=metric(.6)) for z in s.configs()]

    def test_fixed_six_configs_and_fullslot_matched_pair(self):
        self.assertEqual(s.design()['grid'],s.configs());self.assertEqual(len(s.configs()),6)
        self.assertEqual([z['mode'] for z in s.configs()[4:]],['month21','union21'])
        self.assertEqual(s.SOURCE_SCOPE_SPEC,s.scope.configs()[3]);self.assertEqual(s.LOCAL_SPEC,s.local.configs()[4])
        self.assertEqual([s.diagnostic_name(z) for z in s.configs()[4:]],list(s.ARMS))
        self.assertTrue(all('fraction' not in z for z in s.configs()))

    def test_both_fullslot_coefficients_start_scope_not_local_pre(self):
        ranks,ready=self.ranks();self.assertTrue(ready.all())
        original=s.scope.modified_pre_score(s.SOURCE_SCOPE_SPEC,self.a,4)
        local=s.local.modified_pre_score(s.LOCAL_SPEC,self.a,4);self.assertFalse(np.array_equal(original,local))
        for ident in (4,5):
            spec=s.configs()[ident];coef=np.array([0.,.01,.05,.05])[:,None]
            want=coef*(ranks[spec['mode']]-self.a['rank_log_excess'])
            np.testing.assert_array_equal(s.score_delta(spec,self.a,4),want)
            expected=original.copy();expected[want!=0.]+=want[want!=0.]
            np.testing.assert_array_equal(s.modified_pre_score(spec,self.a,4),expected)
            self.assertFalse(np.array_equal(expected,local+want));self.assertLessEqual(np.abs(want).max(),.05)
        for spec in s.configs()[:4]:np.testing.assert_array_equal(s.score_delta(spec,self.a,4),0.)

    def test_one_missing_U_prediction_falls_back_both_whole_months(self):
        for key in s.ARMS:
            a=copy.deepcopy(self.a);a['density_raw_'+key][1,4]=np.nan
            ranks,ready=s.ranked_forecasts({k:a['density_raw_'+k] for k in s.ARMS},a['density_available'],a['density_source_raw'],a,4)
            self.assertEqual(ready.tolist(),[True,False,True,True])
            for k in s.ARMS:np.testing.assert_array_equal(ranks[k][1],a['rank_log_excess'][1])
            for spec in s.configs()[4:]:np.testing.assert_array_equal(s.score_delta(spec,a,4)[1],0.)
        self.a['density_available'][2,1]=False
        for ident in (4,5):np.testing.assert_array_equal(s.score_delta(s.configs()[ident],self.a,4)[2],0.)

    def test_U_not_filtered_by_slot_or_extension_or_training_history(self):
        self.a['density_raw_month21'][0,0]=np.nan
        self.a['extension_available'][1,0]=False;self.a['density_raw_union21'][1,0]=np.nan
        self.assertEqual(self.ranks()[1].tolist(),[False,False,True,True])
        self.a['forecast_available'][2,1]=False;self.a['eligible'][3,2]=False
        for ident in (4,5):
            delta=s.score_delta(s.configs()[ident],self.a,4)
            for row,col in ((2,1),(3,2),(1,0)):self.assertEqual(delta[row,col],0.)

    def test_missing_outside_U_preserves_current_population(self):
        self.a['eligible'][:,0]=False;self.a['forecast_available'][:,1]=False
        self.a['density_available'][:,:2]=False
        for key in s.ARMS:self.a['density_raw_'+key][:,:2]=np.nan
        ranks,ready=self.ranks();self.assertTrue(ready.all())
        for key in s.ARMS:
            np.testing.assert_array_equal(ranks[key][:,:2],self.a['rank_log_excess'][:,:2])
            for i in range(4):np.testing.assert_array_equal(ranks[key][i,2:],s.ep.old.prior.ranks(pd.Series(self.a['density_raw_'+key][i,2:])))

    def test_whole_missing_fallback_to_scope_not_local(self):
        expected=self.choose(2);self.assertFalse(np.array_equal(expected[1],self.choose(3)[1]))
        self.a['density_available'][:]=False
        for key in s.ARMS:self.a['density_raw_'+key][:]=np.nan
        for ident in (4,5):
            got=self.choose(ident);self.assertEqual(got[0],expected[0]);self.assertEqual(got[1].tobytes(),expected[1].tobytes())
            self.assertEqual(s.rp.make_orders(*got,self.m),s.rp.make_orders(*expected,self.m))

    def test_source_identity_cached_ranks_exact_and_equal_arms_equal_orders(self):
        self.a['rank_log_excess'][0,0]=-0.
        for key in s.ARMS:self.a['density_raw_'+key]=self.a['density_source_raw'].copy()
        ranks,ready=self.ranks();self.assertTrue(ready.all())
        for key in s.ARMS:self.assertEqual(ranks[key].tobytes(),self.a['rank_log_excess'].tobytes())
        for ident in (4,5):
            got,want=self.choose(ident),self.choose(2);self.assertEqual(got[0],want[0]);self.assertEqual(got[1].tobytes(),want[1].tobytes())
        self.a['density_raw_month21'][:]=1.;self.a['density_raw_union21'][:]=1.
        ranks,ready=self.ranks();want=s.ep.old.prior.ranks(pd.Series(np.ones(6)))
        for key in s.ARMS:np.testing.assert_array_equal(ranks[key],np.tile(want,(4,1)))
        left,right=self.choose(4),self.choose(5);self.assertEqual(left[0],right[0]);np.testing.assert_array_equal(left[1],right[1])
        self.a['forecast_available'][2]=False;self.assertFalse(self.ranks()[1][2])

    def test_candidate_schema_nonfinite_failure_and_no_available_expansion(self):
        available=np.array([True,True,False]);candidate=np.array([True,False,False])
        good=dict(raw=dict(source_log=np.array([0.,.1,np.nan]),month21=np.array([0.,np.nan,np.nan]),union21=np.array([.1,np.nan,np.nan])),
                  available=available,candidate_available=candidate)
        s.validate_forecasts(good,3)
        for key in good['raw']:
            for value in (np.nan,np.inf,-np.inf):
                bad=copy.deepcopy(good);bad['raw'][key][0]=value
                with self.assertRaises(FloatingPointError):s.validate_forecasts(bad,3)
        bad=copy.deepcopy(good);bad['candidate_available'][2]=True
        with self.assertRaises(ValueError):s.validate_forecasts(bad,3)
        bad=copy.deepcopy(good);bad['raw']['union21'][1]=0.
        with self.assertRaises(FloatingPointError):s.validate_forecasts(bad,3)
        bad=copy.deepcopy(good);bad['available']=available.astype(int)
        with self.assertRaises(ValueError):s.validate_forecasts(bad,3)

    def test_ranking_schema_fails_numerical_errors_not_missing(self):
        for n in (True,-1,5,1.5):
            with self.assertRaises(ValueError):self.choose(4,n=n)
        with self.assertRaises(ValueError):s.choices(dict(s.configs()[4],mode='other'),self.a,self.m,4,policy=POLICY)
        for key,value in (('density_raw_month21',np.inf),('density_raw_union21',np.inf),('density_source_raw',np.nan),('rank_log_excess',np.nan)):
            a=copy.deepcopy(self.a);a[key][0,0]=value
            with self.assertRaises((ValueError,FloatingPointError)):s.choices(s.configs()[4],a,self.m,4,policy=POLICY)
        a=copy.deepcopy(self.a);a['density_available']=a['density_available'].astype(int)
        with self.assertRaises(ValueError):s.choices(s.configs()[4],a,self.m,4,policy=POLICY)

    def test_four_controls_bypass_new_arrays_and_exact_orders(self):
        for key in tuple(self.a):
            if key.startswith('density_'):del self.a[key]
        with patch.object(s,'ranked_forecasts',side_effect=AssertionError('control bypass')):
            for ident,source in ((0,0),(1,1),(2,3),(3,4)):
                owner=s.scope if ident<3 else s.local
                want=owner.choices(owner.configs()[source],self.a,self.m,4,policy=POLICY)
                got=self.choose(ident);self.assertEqual(got[0],want[0]);self.assertEqual(got[1].tobytes(),want[1].tobytes())
                self.assertEqual(s.rp.make_orders(*got,self.m),s.rp.make_orders(*want,self.m))

    def test_candidate_uses_original_selector_payload_newpre_top2(self):
        for ident in (4,5):
            captured={}
            def capture(pre,eligible,symbols,effects,original,policy):
                captured.update(pre=pre.copy(),eligible=eligible.copy(),effects=effects.copy(),original=original,policy=policy)
                return s.ep.select_boundary(pre,eligible,symbols,effects,original,policy)
            got=self.choose(ident,boundary_selector=capture);pre=s.modified_pre_score(s.configs()[ident],self.a,4)
            np.testing.assert_array_equal(captured['pre'],pre)
            np.testing.assert_array_equal(captured['effects'],self.a['effects'])
            np.testing.assert_array_equal(captured['eligible'],self.a['eligible'])
            for key in ('baseline','core','state_calm10','state_weak_narrow'):
                np.testing.assert_array_equal(captured['original'][key],self.a[key])
            np.testing.assert_array_equal(captured['original']['core_weight'],np.broadcast_to(self.a['core_weight'],pre.shape))
            self.assertEqual(captured['policy'],POLICY)
            for i,row in enumerate(pre):
                ranked=sorted(np.flatnonzero(self.a['eligible'][i]),key=lambda j:(-row[j],self.m['symbols'][i][j]))
                self.assertTrue(set(ranked[:2]).issubset(got[0][i]))
                allowed={j for j in ranked[2:5] if row[ranked[2]]-row[j]<=.02}
                self.assertTrue(set(got[0][i])-set(ranked[:2])<=allowed)
                np.testing.assert_array_equal(got[1][i,ranked[:2]],pre[i,ranked[:2]])

    def test_all_prefixes_future_poison_independent_all_six(self):
        for ident in range(6):
            full=self.choose(ident)
            for n in range(5):
                got=self.choose(ident,n=n)
                self.assertEqual(got[0],full[0][:n]);np.testing.assert_array_equal(got[1],full[1][:n])
                poison=copy.deepcopy(self.a)
                for value in poison.values():value[n:]=False if value.dtype.kind=='b' else np.nan
                meta=copy.deepcopy(self.m);meta['symbols'][n:]=[[None]]*(4-n)
                other=s.choices(s.configs()[ident],poison,meta,n,policy=POLICY)
                self.assertEqual(other[0],got[0]);np.testing.assert_array_equal(other[1],got[1])

    def test_identity_exact_past_pool_source_and_offsets(self):
        import pandas as pd
        tm=dict(dates=['2018-01-31','2018-02-28'],ends=['2018-02-28','2018-03-29'],symbols=[['A','B'],['ETF','A']])
        monthly=dict(dates=['2020-01-31'],symbols=[['A','ETF']])
        meta=dict(training=tm,monthly=monthly)
        a=dict(dates=np.repeat(np.array(tm['dates'],dtype='datetime64[D]'),2),
            ends=np.repeat(np.array(tm['ends'],dtype='datetime64[D]'),2),
            event_dates=np.array(['2020-01-31']*2,dtype='datetime64[D]'),offsets=np.array([0,2,4]),
            train_plain=np.arange(40).reshape(4,10),event_plain=np.ones((2,10)),
            train_eligible=np.array([True,True,False,True]),train_good=np.array([False,True,False,True]),
            eligible=np.array([[True,True]]))
        original=copy.deepcopy(a)
        audit=s.validate_identity(meta,a,original,copy.deepcopy(meta))
        self.assertEqual(audit['eligible_rows'],3)
        self.assertTrue(audit['explicit_symbols_offsets_dates_ends'])
        for key in s.target.CACHED_KEYS:
            bad=copy.deepcopy(a)
            bad[key].flat[0]=not bool(bad[key].flat[0]) if bad[key].dtype.kind=='b' else bad[key].flat[0]+1
            with self.subTest(key=key),self.assertRaises((AssertionError,ValueError)):
                s.validate_identity(meta,bad,original,meta)
        wrong=copy.deepcopy(meta);wrong['training']['symbols'][0]=['B','A']
        with self.assertRaises(AssertionError):s.validate_identity(wrong,a,original,meta)

    def test_panel_parity_fixed_scale_missing_and_opening(self):
        index=pd.bdate_range('2020-01-01',periods=5)
        adj=pd.DataFrame({'A':[1.,2.,np.nan,3.,4.],'B':[np.nan]*5},index=index)
        close=adj*3
        panel=dict(adj=adj,close=close,open=close*.99)
        px=adj*7
        op=px*.99
        audit=s.verify_panel_parity(panel,px,op)
        self.assertEqual(audit['scales'],{'A':7.})
        self.assertEqual(audit['all_missing_symbols'],['B'])
        bad=px.copy();bad.iloc[-1,0]*=1.01
        with self.assertRaises(AssertionError): s.verify_panel_parity(panel,bad,op)
        bad=px.ffill()
        with self.assertRaises(AssertionError): s.verify_panel_parity(panel,bad,op)
        with self.assertRaises(AssertionError): s.verify_panel_parity(panel,px,op*1.01)
        for value in (0.,-1.):
            bad_panel={key:frame.copy() for key,frame in panel.items()}
            bad_panel['adj'].iloc[0,0]=value
            with self.assertRaises(AssertionError): s.verify_panel_parity(bad_panel,px,op)
        bad_panel={key:frame.copy() for key,frame in panel.items()}
        bad_panel['adj'].iloc[0,0]=0.
        bad=px.copy();bad.iloc[0,0]=0.
        with self.assertRaises(ValueError): s.verify_panel_parity(bad_panel,bad,op)

    def test_raw64_original_source_manifest_requires_all_exact(self):
        loader=str(s.O/'monthly_stability_monitor/inputs.py')
        raw={str(i):'hash' for i in range(63)};raw[loader]=s.sha(loader)
        with patch.object(s,'read',return_value=dict(raw)):
            self.assertEqual(s.raw_source_hashes(dict(source_hashes=raw)),raw)
            for bad in (dict(raw,extra='x'),{k:v for k,v in raw.items() if k!=loader},dict(raw,**{loader:'bad'})):
                with self.assertRaises(AssertionError):s.raw_source_hashes(dict(source_hashes=bad))
        with patch.object(s,'read',return_value={}):
            with self.assertRaises(AssertionError):s.raw_source_hashes(dict(source_hashes=raw))

    def test_source_raw_ranker_independent_full_U(self):
        self.a['eligible'][0,0]=False;raw=self.a['density_raw_month21'].copy();raw[0,0]=np.nan
        got=s.source_ranked_forecast(raw,self.a)
        self.assertEqual(got[0,0],self.a['rank_log_excess'][0,0])
        for i in range(4):
            use=self.a['eligible'][i]&self.a['forecast_available'][i]
            want=s.ep.old.prior.ranks(pd.Series(raw[i]).where(use))
            np.testing.assert_array_equal(got[i,use],want[use])
        raw[1,0]=np.nan
        with self.assertRaises(ValueError):s.source_ranked_forecast(raw,self.a)

    def test_only_local_benchmark_and_two_candidates_can_win(self):
        rows=self.rows()
        for row in rows[:3]:row['training']['cagr']=10.
        selected=s.select(rows);self.assertEqual(selected['primary']['spec']['id'],3)
        self.assertEqual(len(selected),7)
        for row in rows[4:]:
            self.assertEqual(selected[s.diagnostic_name(row['spec'])]['spec'],row['spec']);self.assertFalse(row['admitted'])

    def test_guards_and_tolerance_against_local_not_scope(self):
        rows=self.rows();rows[4]['training']['cagr']=.5+5e-13
        self.assertEqual(s.select(rows)['primary']['spec']['id'],3)
        rows[4]['training']['cagr']=.5+2e-12;rows[5]['training']['cagr']=.5+2.5e-12
        self.assertEqual(s.select(rows)['primary']['spec']['id'],4)
        for section,key,value in (('training','cagr',.499),('training','sharpe',1.54),
                ('training','max_drawdown',-.331),('early','cagr',.374),('recent','cagr',.574)):
            rows=self.rows();rows[4]['training']['cagr']=.99;rows[4][section][key]=value;rows[4]['later_cagr']=1000.
            self.assertEqual(s.select(rows)['primary']['spec']['id'],3)

    def test_calendar_official_provenance_and_common_hashes_exact(self):
        old=dict(provenance={'clock':'XNYS'},source_hashes={'calendar':'h','oldhelper':'a'})
        new=dict(provenance={'clock':'XNYS'},source_hashes={'calendar':'h','newhelper':'b'})
        s.validate_calendar(new,old)
        for bad in (dict(new,provenance={'clock':'fake'}),dict(new,source_hashes={'calendar':'changed'}),dict(new,source_hashes={'isolated':'a'})):
            with self.assertRaises(AssertionError):s.validate_calendar(bad,old)

    def test_original_arrays_dtype_values_and_missing_rejected(self):
        source={'x':np.array([0.,np.nan]),'b':np.array([True,False])}
        other=copy.deepcopy(source);other['new']=np.array([1.]);s.assert_original_arrays(source,other)
        for mutation in ('missing','dtype','value'):
            bad=copy.deepcopy(other)
            if mutation=='missing':del bad['x']
            elif mutation=='dtype':bad['x']=bad['x'].astype(np.float32)
            else:bad['b'][0]=False
            with self.assertRaises(AssertionError):s.assert_original_arrays(source,bad)

    def test_source_bundle_requires_exact_arrays_metadata_prices_policy(self):
        px=pd.DataFrame(np.ones((2,2)),columns=['A','B']);op=px*.99
        original={'x':np.array([0.,np.nan])};component=dict(original,new=np.array([1.]))
        meta={'dates':['2016-08-17']};protected={str(i):'hash' for i in range(21436)}
        clock=dict(provenance={'clock':'official'},source_hashes={'calendar':'same','new':'new'})
        oldclock=dict(provenance={'clock':'official'},source_hashes={'calendar':'same','old':'old'})
        files={'INCUMBENT_POLICY.json':POLICY,'PROTECTED.json':protected,'PLAN.json':{'calendar':oldclock}}
        with ExitStack() as stack:
            stack.enter_context(patch.object(s.scope,'load',return_value=(meta,original,px,op)))
            load=stack.enter_context(patch.object(s.local,'load',return_value=(copy.deepcopy(meta),component,px.copy(),op.copy())))
            reader=stack.enter_context(patch.object(s,'read',side_effect=lambda p:files[Path(p).name]))
            stack.enter_context(patch.object(s,'calendar_provenance',return_value=clock))
            stack.enter_context(patch.object(s.local,'calendar_provenance',return_value=oldclock))
            got=s.source_bundle();self.assertIs(got[1],component)
            for wrong in (({'dates':['bad']},component,px,op),(meta,{'x':np.ones(2)},px,op),(meta,component,px*2,op),(meta,component,px,op*2)):
                load.return_value=wrong
                with self.assertRaises(AssertionError):s.source_bundle()
            load.return_value=(meta,component,px,op)
            reader.side_effect=lambda p:dict(POLICY,x=1) if Path(p)==s.local.P/'INCUMBENT_POLICY.json' else files[Path(p).name]
            with self.assertRaises(AssertionError):s.source_bundle()

    def test_source_completion_guard_and_hashes(self):
        completion=dict(original_controls_exact=True,independent_fixed_unit_replay=True,training_freeze_parity=True,
            source_model_parity=True,protected_files=21436,protected_changes=[],observer_accounts_changed=False,fixed_configs=5,slot_replacement_fraction=.25)
        s.validate_local_completion(completion)
        for key in completion:
            bad=dict(completion);bad[key]=None
            with self.assertRaises(AssertionError):s.validate_local_completion(bad)
        payload={'FROZEN.json':dict(hashes={'f':'h'},selected={'local_linear':{'spec':s.LOCAL_SPEC}}),
                 'INPUT_HASHES.json':{'i':'h'},'CACHE_HASHES.json':{'c':'h'},'VALIDATION.json':completion}
        calls=[]
        with patch.object(s.local,'registered') as registered,patch.object(s.local,'verify_source') as verify, \
                patch.object(s,'sha',return_value='hash'),patch.object(Path,'read_text',return_value='hash\n'), \
                patch.object(s,'read',side_effect=lambda f:payload[Path(f).name]), \
                patch.object(s.old,'assert_hashes',side_effect=lambda x,root=None:calls.append((x,root))):
            s.verify_source()
        registered.assert_called_once_with();verify.assert_called_once_with()
        self.assertEqual(calls,[({'f':'h'},s.local.P),({'i':'h'},None),({'c':'h'},s.local.P)])

    def test_top2_changes_before_boundary_and_no_global_post_sort_or_mutation(self):
        before=copy.deepcopy(self.a)
        for ident in range(6):self.choose(ident)
        for key in before:self.assertEqual(self.a[key].tobytes(),before[key].tobytes())
        self.a['state_weak_narrow'][:]=True;self.a['state_calm10'][:]=False
        self.a['pre_score'][:]=[.20,.1999,.1998,.1997,.1996,.1995];self.a['old_ml4']=self.a['rank_log_excess'].copy()
        for key in s.ARMS:self.a['density_raw_'+key][:]=[-9.,-8.,7.,8.,9.,10.]
        for ident in (4,5):
            pre=s.modified_pre_score(s.configs()[ident],self.a,4)
            anchors=sorted(range(6),key=lambda j:(-pre[0,j],self.m['symbols'][0][j]))[:2]
            self.assertNotEqual(set(anchors),{0,1});self.assertTrue(set(anchors)<=set(self.choose(ident)[0][0]))
        self.a['pre_score'][:]=[.30,.25,.20,.19,.181,.1805]
        self.a['baseline']=self.a['pre_score'].copy();self.a['core']=self.a['pre_score'].copy()
        self.a['state_calm10'][:]=False;self.a['state_weak_narrow'][:]=False
        self.a['effects'][:]=[0.,0.,-.5,-.5,-.5,0.]
        self.a['density_available'][:]=False
        for key in s.ARMS:self.a['density_raw_'+key][:]=np.nan
        for ident in (4,5):
            chosen,score=self.choose(ident);self.assertTrue((score[:,5]>score[:,2]).all());self.assertEqual(chosen,[[0,1,2]]*4)

    def test_cash_twohalf_original_eligible_not_new_label_eligible(self):
        for count in (0,1,2):
            self.a['eligible'][:]=False;self.a['eligible'][:,:count]=True
            for ident in (4,5):
                chosen,score=self.choose(ident)
                self.assertEqual([len(row) for row in chosen],[count if count==2 else 0]*4)
                np.testing.assert_array_equal(score,s.modified_pre_score(s.configs()[ident],self.a,4)[:4])

    def density_input(self):
        X=np.arange(60,dtype=float).reshape(6,10)/100
        days=np.repeat(np.array(['2013-01-25','2013-01-31','2013-02-28'],dtype='datetime64[D]'),2)
        arrays=dict(density_X=X,density_eligible=np.array([True,False,True,True,False,True]),density_dates=days,
            density_symbols=np.array(list('ABBACA')),density_is_monthly=np.array([False,False,True,True,True,True]),monthly_row_index=np.array([2,3,4,5]))
        meta=dict(dates=['2013-01-25','2013-01-31','2013-02-28'],symbols=[list('AB'),list('BA'),list('CA')],
            offsets=[0,2,4,6],pool_origins=['past','frozen','frozen'],is_monthly=[False,True,True])
        cached=dict(dates=days[2:].copy(),train_plain=X[2:].copy(),train_eligible=arrays['density_eligible'][2:].copy())
        tm=dict(dates=meta['dates'][1:],symbols=meta['symbols'][1:])
        return dict(arrays=arrays,meta=meta,audit={}),cached,tm

    def test_density_input_flat_identity_original_A_and_dtype_exact(self):
        data,cached,tm=self.density_input();audit=s.validate_feature_inputs(data,cached,tm)
        self.assertEqual(audit['monthly_rows'],4);self.assertEqual(audit['union_rows'],6)
        for key,value in (('monthly_row_index',np.array([0,1,4,5])),('density_symbols',np.array(list('ABABCA'))),
                          ('density_X',data['arrays']['density_X'].astype(np.float32)),('density_eligible',np.ones(6,bool))):
            bad=copy.deepcopy(data);bad['arrays'][key]=value
            with self.assertRaises((AssertionError,ValueError)):s.validate_feature_inputs(bad,cached,tm)
        bad=copy.deepcopy(data);bad['meta']['offsets'][-1]=7
        with self.assertRaises(AssertionError):s.validate_feature_inputs(bad,cached,tm)
        with self.assertRaises(ValueError):s.validate_feature_inputs(dict(data,arrays=dict(data['arrays'],extra=0)),cached,tm)

    def test_independent_monthly_label_parity_end_good_and_nan(self):
        data,_,_=self.density_input();days=data['arrays']['density_dates'];idx=data['arrays']['monthly_row_index']
        union=dict(density_y=np.array([.1,-.1,0.,0.,np.nan,np.nan]),density_ends=days+31,density_good=np.array([True,True,True,True,False,False]))
        monthly={k:v[idx].copy() for k,v in union.items()}
        s.validate_labels(union,6,days);audit=s.validate_monthly_labels(union,monthly,idx);self.assertTrue(audit['independent_monthly_rebuild'])
        for key in union:
            bad={k:v.copy() for k,v in monthly.items()}
            bad[key][0]=not bad[key][0] if key=='density_good' else bad[key][0]+1
            with self.assertRaises(AssertionError):s.validate_monthly_labels(union,bad,idx)
        for key,value in (('density_y',np.ones(6)),('density_ends',days),('density_good',np.ones(6,int))):
            with self.assertRaises((ValueError,FloatingPointError)):s.validate_labels(dict(union,**{key:value}),6,days)

    def test_dependency_pins_include_labels_legacy_and_all_source(self):
        files=set(s.implementations());sources=set(s.source_files())
        for path in (P/'study.py',P/'test_study.py',P/'model.py',P/'test_model.py',P/'features.py',P/'test_features.py',P/'labels.py',P/'test_labels.py',
                     Path(s.adapter.__file__),Path(s.ep.__file__),Path(s.scope.__file__),Path(s.local.__file__),*s.model.dependency_files(),*s.features.dependency_files(),*s.labels.dependency_files()):
            self.assertIn(path,files)
        for path in (s.features.LEGACY_PATH,s.target.P/'TRAIN_INPUTS.npz',s.target.old.P/'META.json',s.target.old.P/'DATA.npz',s.arch.P/'META.json',
                     s.scope.P/'primary_ORDERS.json',s.local.P/'local_linear_ORDERS.json',s.local.P/'VALIDATION.json',s.local.P/'DATA.npz'):
            self.assertIn(path,sources)
        self.assertFalse({'RESULTS.json','TRAIN_TRIALS.json','TRAIN_RETURNS.npz'}&{p.name for p in sources})

    def test_phase_source_independence_label_rebuild_freeze_and_protection(self):
        source=(P/'study.py').read_text();tree=ast.parse(source)
        fs={node.name:ast.get_source_segment(source,node) for node in tree.body if isinstance(node,ast.FunctionDef)}
        protection=[]
        for node in tree.body:
            if isinstance(node,ast.FunctionDef):
                for item in ast.walk(node):
                    if isinstance(item,ast.Call) and isinstance(item.func,ast.Name) and item.func.id=='assert_protected':protection.append(node.name)
        self.assertEqual(protection,['evaluate'])
        for text in ('rp.paths(px,m,TRAIN)','choices(spec,a,m,TRAIN)','both_diagnostics_frozen=True','trials=6',"returns[3][bounds[n]:bounds[end]]", "assert not (P/'FROZEN.json').exists()"):
            self.assertIn(text,fs['train'])
        self.assertIn("frozen['selected'].items()",fs['evaluate']);self.assertNotIn('select(',fs['evaluate'])
        for text in ("event_good=a['eligible'].reshape(-1)&np.isfinite(cached['event_plain']).all(axis=1)",
                     "np.testing.assert_array_equal(source_available,a['forecast_available'])",
                     "np.testing.assert_allclose(source_raw,a['raw_log_excess'],rtol=0.,atol=PARITY_ATOL,equal_nan=True)",
                     "source_ranked_forecast(source_raw,a),a['rank_log_excess']",
                     "cached['dates'],cached['ends'],cached['train_plain'],source_targets",
                     "cached['common_good'],cached['event_dates'],cached['event_plain'],event_good",
                     "labels.build_labels(panel['adj'],meta['training'],cached['train_plain'],cached['train_eligible'])",
                     "validate_monthly_labels(union_labels['arrays'],monthly_labels['arrays'],density['monthly_row_index'])"):
            self.assertIn(text,fs['prepare'])
        self.assertIn("[('union21','month21')]",fs['evaluate'])
        self.assertIn("('next_open_zero',0.),('next_open_cost',.0005)",fs['evaluate'])
        self.assertNotIn("a['model_available']=",fs['prepare']);self.assertNotIn("a['raw_source_log']=",fs['prepare'])

    def test_synthetic_actual_features_independent_labels_model_and_selector(self):
        class Calendar:
            def sessions_in_range(self,start,end):return pd.bdate_range(start,end)
        calendar=Calendar();index=pd.bdate_range('2010-01-01','2017-01-31');names=['A','B','C','D','SPY','QQQ']
        rng=np.random.default_rng(734);r=rng.normal(.0015,.006,(len(index),6))
        adj=pd.DataFrame(100*np.exp(np.cumsum(r,axis=0)),index=index,columns=names)
        panel=dict(adj=adj,close=adj.copy(),open=adj*.999,high=adj*1.02,low=adj*.98,
            volume=pd.DataFrame(rng.uniform(10,1000,adj.shape),index=index,columns=names),
            sectors=pd.DataFrame(None,index=index,columns=names),
            member=pd.DataFrame([[True,True,True,True]],index=pd.to_datetime(['2013-01-15']),columns=names[:4]))
        legacy=pd.DataFrame(True,index=index,columns=names[:4])
        days=pd.date_range('2013-01-31','2014-12-31',freq=pd.offsets.BMonthEnd()).to_numpy(dtype='datetime64[D]')
        ends=pd.date_range('2013-02-28','2015-01-30',freq=pd.offsets.BMonthEnd()).to_numpy(dtype='datetime64[D]')
        tm=dict(dates=[str(d) for d in days],ends=[str(d) for d in ends],symbols=[names[:] for _ in days])
        self.m['symbols']=[names[:] for _ in self.m['dates']];self.m['etfs']=['SPY','QQQ']
        X,g=s.features.snapshots(panel,tm['dates'],tm['symbols'],self.m['etfs'])
        Q,qg=s.features.snapshots(panel,self.m['dates'],self.m['symbols'],self.m['etfs'])
        events=np.repeat(np.array(self.m['dates'],dtype='datetime64[D]'),6)
        cached=dict(dates=np.repeat(days,6),ends=np.repeat(ends,6),train_plain=X,train_eligible=g)
        data=s.features.build_inputs(panel,tm,self.m,X,Q,g,qg.reshape(4,6),legacy_member=legacy,calendar=calendar)
        s.validate_feature_inputs(data,cached,tm);a=data['arrays']
        u=s.labels.build_labels(adj,data['meta'],a['density_X'],a['density_eligible'],calendar=calendar)['arrays']
        monthly=s.labels.build_labels(adj,tm,X,g,calendar=calendar)['arrays']
        s.validate_labels(u,len(a['density_dates']),a['density_dates']);s.validate_monthly_labels(u,monthly,a['monthly_row_index'])
        self.assertGreater(len(a['density_dates']),len(cached['dates']))
        targets={key:monthly['density_y'].copy() for key in s.target.TARGETS};common=monthly['density_good'].copy()
        forecast=s.model.annual_forecasts(cached['dates'],cached['ends'],X,targets,common,events,Q,qg,
            a['density_dates'],u['density_ends'],a['density_X'],u['density_y'],u['density_good'],a['density_is_monthly'])
        s.validate_forecasts(forecast,24);self.assertTrue(forecast['candidate_available'].all())
        reference=s.target.model.annual_forecasts(cached['dates'],cached['ends'],X,targets,common,events,Q,qg)
        np.testing.assert_array_equal(forecast['raw']['source_log'],reference['log_excess']);np.testing.assert_array_equal(forecast['available'],reference['available'])
        for rec in forecast['audit']['annual']:
            self.assertEqual(rec['monthly_support'],rec['union_support'])
            for mode in s.ARMS:self.assertAlmostEqual(rec['arms'][mode]['weight_sum'],rec['monthly_rows'],places=10)
        self.a['eligible']=qg.reshape(4,6);self.a['forecast_available']=forecast['available'].reshape(4,6)
        self.a['density_source_raw']=forecast['raw']['source_log'].reshape(4,6)
        self.a['rank_log_excess']=s.source_ranked_forecast(self.a['density_source_raw'],self.a)
        self.a['density_available']=forecast['candidate_available'].reshape(4,6)
        for mode in s.ARMS:self.a['density_raw_'+mode]=forecast['raw'][mode].reshape(4,6)
        ranks,ready=self.ranks();self.assertTrue(ready.all())
        for mode in s.ARMS:
            for i in range(4):np.testing.assert_array_equal(ranks[mode][i],s.ep.old.prior.ranks(pd.Series(self.a['density_raw_'+mode][i])))
        for ident in (4,5):
            full=self.choose(ident);self.assertTrue(all(len(row)==3 for row in full[0]))
            for n in range(5):
                got=self.choose(ident,n=n);self.assertEqual(got[0],full[0][:n]);np.testing.assert_array_equal(got[1],full[1][:n])
        # Rebuilding union labels never changes A, even if only weekly rows are poisoned.
        poison={k:v.copy() for k,v in u.items()};poison['density_y'][~a['density_is_monthly']]=np.nan
        np.testing.assert_array_equal(poison['density_y'][a['monthly_row_index']],monthly['density_y'])


if __name__=='__main__':unittest.main()
