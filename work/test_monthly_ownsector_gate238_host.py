"""Synthetic-only238 final-selector eligibility integration and lifecycle guards."""
from pathlib import Path
from types import SimpleNamespace
import ast,copy,importlib.util,json,tempfile,unittest
from unittest.mock import Mock,patch
import numpy as np
import pandas as pd
import monthly_ownsector_gate238_kernel as kernel
import monthly_ownsector_gate238_independent as independent
from test_monthly_ownsector_gate238_kernel import make_report

spec=importlib.util.spec_from_file_location('tested_ownsector_gate238_host',Path(__file__).with_name('monthly_ownsector_gate238_host.py'))
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)


def fixture(full=False):
    n,w=(122,49) if full else (4,5)
    end='2026-09-25' if full else '2016-11-02'
    dates=pd.bdate_range('2014-01-01',end);t=np.arange(len(dates),dtype=float)
    names=['S%02d'%j for j in range(w)]
    p={'adj':pd.DataFrame({s:40.*np.exp((j+1)*.00002*t) for j,s in enumerate(names)},index=dates)}
    p['adj']['SPY']=50.*np.exp((w//2+1)*.00002*t)
    p['adj']['XLK']=p['adj']['SPY'].copy()
    p['vox']=pd.DataFrame({'VOX':p['adj']['SPY'].copy()},index=dates)
    signal=['2016-08-17']+[str(d.date()) for d in pd.date_range('2016-08-31','2026-08-31' if full else '2016-10-31',freq='BME')]
    ends=signal[1:]+['2026-09-24' if full else '2016-11-30']
    m=dict(dates=signal,ends=ends,symbols=[names.copy() for _ in range(n)],etfs=names[-17:] if full else [])
    calm=np.arange(n)%3==0;cw=np.where(calm,.5,.52)[:,None]
    support=np.tile(np.linspace(-.1,.1,w),(n,1));core=cw*np.tile(np.linspace(.45,.2,w),(n,1))
    zero=np.zeros((n,w));yes=np.ones((n,w),bool)
    a=dict(baseline=core+(1-cw)*support,core=core,core_weight=cw,eligible=yes.copy(),
        state_calm10=calm,state_weak_narrow=np.arange(n)%3!=2,pre_score=np.tile(np.linspace(.205,.201,w),(n,1)),
        effects=np.tile(np.resize([0.,0.,-.5,.5,0.],w),(n,1)),forecast_available=yes.copy(),extension_available=yes.copy(),
        old_ml4=zero.copy(),rank_log_excess=zero.copy(),model_available=yes.copy(),local_available=np.zeros((n,w),bool),
        raw_source_log=zero.copy(),raw_local_constant=zero.copy(),raw_local_linear=zero.copy())
    for name in ('simple_excess','log_excess','vol_scaled','loss_utility'):
        a['raw_'+name]=zero.copy();a.setdefault('rank_'+name,zero.copy())
    a['rank_local_constant']=zero.copy();a['rank_local_linear']=zero.copy();a['paired_month_ready']=np.ones(n,bool)
    assert len(a)==27
    return p,m,[str(d.date()) for d in dates],a


def build(p,m,c,a,limit=None):
    return kernel.build_features(p,m,c,a,make_report(p,m,c,a),limit=limit)


def check(p,m,c,a,v,au,limit=None):
    return independent.verify_features(p,m,c,a,make_report(p,m,c,a),v,au,limit=limit)


def selection_fixture():
    metrics={arm:{path:{e:dict(cagr=.5,sharpe=1.5,max_drawdown=-.3) for e in ('training','early','recent')}
        for path in ('same_close_zero','next_open_zero')} for arm in h.NAMES}
    orders={arm:[dict(symbols=['A','B','C']) for _ in range(122)] for arm in h.NAMES}
    for path in metrics[h.NAMES[1]]:metrics[h.NAMES[1]][path]['training']['cagr']+=.01
    for i in (1,2,53):orders[h.NAMES[1]][i]['symbols']=['A','B','D']
    return metrics,orders


class SelectionTests(unittest.TestCase):
    def test_two_zero_paths_choose_candidate(self):
        metrics,orders=selection_fixture();result=h.select_first88(metrics,orders)
        self.assertTrue(result['admitted']);self.assertEqual(result['primary'],h.NAMES[1])
        self.assertEqual(result['changed_months'],[1,2,53]);self.assertEqual(len(result['guards']),3)
    def test_every_return_risk_and_era_guard_rejects(self):
        for path,era,key,value in [('same_close_zero','training','cagr',.504),('next_open_zero','training','cagr',.49),
            ('same_close_zero','training','sharpe',1.449),('next_open_zero','training','sharpe',1.449),
            ('same_close_zero','training','max_drawdown',-.331),('next_open_zero','training','max_drawdown',-.331),
            ('same_close_zero','early','cagr',.474),('same_close_zero','recent','cagr',.474),
            ('next_open_zero','early','cagr',.474),('next_open_zero','recent','cagr',.474)]:
            metrics,orders=selection_fixture();metrics[h.NAMES[1]][path][era][key]=value
            with self.subTest(path=path,era=era,key=key):self.assertFalse(h.select_first88(metrics,orders)['admitted'])
    def test_later_cost_and_order_permutations_do_not_choose(self):
        metrics,orders=selection_fixture();expected=h.select_first88(metrics,orders)
        for arm in h.NAMES:
            metrics[arm]['next_open_cost']={'training':{'cagr':999. if arm==h.NAMES[0] else -999.}}
            metrics[arm]['full122']={'cagr':999.}
            for i in range(88,122):orders[arm][i]['symbols']=[arm]
        self.assertEqual(h.select_first88(metrics,orders),expected)
        for i in (1,2,53):orders[h.NAMES[1]][i]['symbols']=['C','B','A']
        self.assertFalse(h.select_first88(metrics,orders)['admitted'])
    def test_three_changes_and_each_era_required(self):
        for change in ([1,2],[1,2,3],[53,54,55]):
            metrics,orders=selection_fixture();orders[h.NAMES[1]]=copy.deepcopy(orders[h.NAMES[0]])
            for i in change:orders[h.NAMES[1]][i]['symbols']=['D']
            self.assertFalse(h.select_first88(metrics,orders)['admitted'])
    def test_extra_arm_rejected(self):
        metrics,orders=selection_fixture();metrics['third']=copy.deepcopy(metrics[h.NAMES[0]])
        with self.assertRaises(ValueError):h.select_first88(metrics,orders)


class DecisionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.study=h.local()
    def test_real_frozen_pre_single_episode_only_new_domain(self):
        p,m,c,a=fixture();v,au=build(p,m,c,a);before=h.array_snapshot(a)
        ep=self.study.ep.select_boundary
        with patch.object(self.study.ep,'select_boundary',wraps=ep) as boundary:
            orders,eligible,scores,pres=h.decisions(self.study,a,m,v,4)
        self.assertEqual(boundary.call_count,2);self.assertEqual(before,h.array_snapshot(a))
        receipt=independent.verify_selection(a,m,v,orders,scores=scores,pre_scores=pres,n=4)
        h.require_selection_audit(receipt,4)
        self.assertEqual(pres[h.NAMES[0]].tobytes(),pres[h.NAMES[1]].tobytes())
        for arm,call in zip(h.NAMES,boundary.call_args_list):
            base,good,names,effects,original,policy=call.args
            np.testing.assert_array_equal(base,pres[arm]);np.testing.assert_array_equal(good,eligible[arm])
            np.testing.assert_array_equal(original['baseline'],a['baseline'])
            np.testing.assert_array_equal(effects,a['effects']);self.assertEqual(policy,h.read(h.LOCAL/'INCUMBENT_POLICY.json'))
        expected,score=self.study.choices(self.study.configs()[4],a,m,4,policy=h.read(h.LOCAL/'INCUMBENT_POLICY.json'))
        self.assertEqual(orders[h.NAMES[0]],self.study.rp.make_orders(expected,score,m))
        self.assertEqual(orders[h.NAMES[1]][0],orders[h.NAMES[0]][0])
    def test_final_scores_may_change_but_never_original_pre(self):
        p,m,c,a=fixture();v,_=build(p,m,c,a)
        # Candidate removes early score leaders before the episode builds its boundary.
        o,e,s,pre=h.decisions(self.study,a,m,v,4)
        np.testing.assert_array_equal(pre[h.NAMES[0]],pre[h.NAMES[1]])
        self.assertFalse(np.array_equal(e[h.NAMES[0]][1],e[h.NAMES[1]][1]))
        self.assertFalse(np.array_equal(s[h.NAMES[0]][1],s[h.NAMES[1]][1]))
        independent.verify_selection(a,m,v,o,scores=s,pre_scores=pre)
    def test_original_model_eligibility_not_candidate_mask(self):
        p,m,c,a=fixture();a['rank_log_excess'][1]=np.linspace(-.4,.4,5)
        v,_=build(p,m,c,a);pre=self.study.modified_pre_score(self.study.configs()[4],a,4)
        o,e,s,pres=h.decisions(self.study,a,m,v,4)
        np.testing.assert_array_equal(pres[h.NAMES[1]],pre)
        wrong=copy.deepcopy(a);wrong['eligible']=v['candidate_eligible'].copy()
        altered=self.study.modified_pre_score(self.study.configs()[4],wrong,4)
        self.assertFalse(np.array_equal(pre[1],altered[1]))
        independent.verify_selection(a,m,v,o,scores=s,pre_scores=pres)
    def test_two_names_and_one_name_cash_are_preserved(self):
        for count in (0,1,2,3):
            p,m,c,a=fixture();a['eligible'][1]=False;a['eligible'][1,-count:]=True if count else False
            p['adj']['SPY']=50.;p['adj']['XLK']=50. # positive original trends pass zero benchmark threshold
            v,_=build(p,m,c,a);o,e,s,pr=h.decisions(self.study,a,m,v,4)
            self.assertEqual(len(o[h.NAMES[1]][1]['symbols']),count if count>=2 else 0)
            independent.verify_selection(a,m,v,o,scores=s,pre_scores=pr)
    def test_all_logical_and_physical_prefixes_and_future_poison(self):
        p,m,c,a=fixture();v,_=build(p,m,c,a);full=h.decisions(self.study,a,m,v,4)
        for n in range(5):
            aa=h.cache_prefix(a,n);mm=h.meta_prefix(m,n);pp=h.feature_prefix_panel(p,m,n);cc=h.feature_prefix_calendar(c,m,n)
            vv,au=build(pp,mm,cc,aa);ck=check(pp,mm,cc,aa,vv,au);h.require_feature_audit(ck,vv,n)
            out=h.decisions(self.study,aa,mm,vv,n)
            h.require_selection_audit(independent.verify_selection(aa,mm,vv,out[0],scores=out[2],pre_scores=out[3],n=n),n)
            for arm in h.NAMES:
                self.assertEqual(out[0][arm],full[0][arm][:n]);np.testing.assert_array_equal(out[2][arm],full[2][arm][:n])
            poison=copy.deepcopy(v);poison['momentum'][n:]=np.inf
            oo=h.decisions(self.study,a,m,poison,n)
            for arm in h.NAMES:self.assertEqual(oo[0][arm],full[0][arm][:n])
    def test_host_rejects_bad_mask_nonboolean_source_and_mutation(self):
        p,m,c,a=fixture();v,_=build(p,m,c,a)
        for kind in ('mask','add','dtype','source','inf','shape'):
            aa=copy.deepcopy(a);vv=copy.deepcopy(v)
            if kind=='mask':vv['required'][1,0]=False
            if kind=='add':aa['eligible'][1,0]=False;vv['original_eligible'][1,0]=False;vv['candidate_eligible'][1,0]=True
            if kind=='dtype':vv['candidate_eligible']=vv['candidate_eligible'].astype(int)
            if kind=='source':aa['forecast_available']=aa['forecast_available'].astype(int)
            if kind=='inf':vv['momentum'][1,0]=np.inf
            if kind=='shape':vv['threshold']=vv['threshold'][:,None]
            with self.subTest(kind=kind),self.assertRaises(ValueError):h.decisions(self.study,aa,m,vv,4)
        for n in (True,-1,5,.5):
            with self.assertRaises(ValueError):h.decisions(self.study,a,m,v,n)
    def test_all_regular_rows_active_and_weak_flag_is_descriptive_only(self):
        p,m,c,a=fixture();v,au=build(p,m,c,a)
        self.assertEqual(au['active_events'],3)
        np.testing.assert_array_equal(v['required'][1:],a['eligible'][1:])
        self.assertFalse(v['required'][0].any())
        self.assertTrue(np.isfinite(v['benchmark_momentum'][1:]).all())
        self.assertTrue((v['candidate_eligible'][2]!=a['eligible'][2]).any())
        flipped=copy.deepcopy(a);flipped['state_weak_narrow']=~a['state_weak_narrow']
        vv,aa=build(p,m,c,flipped)
        for key in h.VALUE_KEYS:np.testing.assert_array_equal(v[key],vv[key])
        self.assertEqual(aa['active_events'],au['active_events'])
        self.assertNotEqual(aa['weak_events'],au['weak_events'])
        check(p,m,c,flipped,vv,aa)
        # Original state is still a pre-score input, so this checks source/gate
        # invariance only, not a false claim of state-invariant final scores.
    def test_empty_eligibility_consumes_no_numeric_spy_and_partial_bypasses(self):
        p,m,c,a=fixture();a['eligible'][2]=False
        t=c.index(m['dates'][2]);p['adj'].loc[pd.Timestamp(c[t-42]),'SPY']=np.nan
        vv,au=build(p,m,c,a)
        self.assertFalse(vv['required'][2].any())
        self.assertTrue(np.isnan(vv['benchmark_momentum'][2]).all())
        pp=h.feature_prefix_panel(p,m,1);pp['adj']=pp['adj'].iloc[:0];pp['vox']=pp['vox'].iloc[:0]
        vv,au=build(pp,h.meta_prefix(m,1),h.feature_prefix_calendar(c,m,1),h.cache_prefix(a,1))
        self.assertEqual(au['active_events'],0)
        np.testing.assert_array_equal(vv['candidate_eligible'],a['eligible'][:1])
    def test_old_weak_only_required_mask_rejected_by_independent_selection(self):
        p,m,c,a=fixture();v,_=build(p,m,c,a)
        bad=copy.deepcopy(v);bad['required'][2]=False;bad['momentum'][2]=np.nan
        bad['benchmark_momentum'][2]=np.nan;bad['threshold'][2]=np.nan
        bad['candidate_eligible'][2]=a['eligible'][2]
        orders,_,scores,pres=h.decisions(self.study,a,m,bad,4)
        with self.assertRaises(ValueError):
            independent.verify_selection(a,m,bad,orders,scores=scores,pre_scores=pres,n=4)
    def test_double_episode_rejected(self):
        p,m,c,a=fixture();v,_=build(p,m,c,a)
        adapter=self.study.adapter.score_and_choose
        def twice(*args,**kw):adapter(*args,**kw);return adapter(*args,**kw)
        with patch.object(self.study.adapter,'score_and_choose',side_effect=twice),self.assertRaisesRegex(ValueError,'more than once'):
            h.decisions(self.study,a,m,v,4)


class EvidenceTests(unittest.TestCase):
    def test_real_feature_flags_exact_counts_and_scopes(self):
        p,m,c,a=fixture();v,au=build(p,m,c,a);ck=check(p,m,c,a,v,au)
        h.require_feature_audit(ck,v,4);self.assertEqual(h.estimate_counts(ck),h.estimate_counts(au))
        for key in h.FEATURE_FLAGS:
            bad=copy.deepcopy(ck);bad[key]=1
            with self.subTest(key=key),self.assertRaises(ValueError):h.require_feature_audit(bad,v,4)
        for key in h.COUNT_FIELDS:
            bad=copy.deepcopy(ck);bad[key]=True
            with self.subTest(key=key),self.assertRaises(ValueError):h.estimate_counts(bad)
        for key,value in [('source_price_observations',1),('spy_price_observations',1),('rejected_cells',999),('new_model_fits',1),('candidate_eligible_cells',999)]:
            bad=copy.deepcopy(ck);bad[key]=value
            with self.assertRaises(ValueError):h.estimate_counts(bad)
    def test_selection_receipt_requires_both_pre_and_full_scores(self):
        p,m,c,a=fixture();v,_=build(p,m,c,a);o,e,s,pr=h.decisions(h.local(),a,m,v,4)
        ck=independent.verify_selection(a,m,v,o,scores=s,pre_scores=pr);h.require_selection_audit(ck,4)
        for key in h.SELECTION_FLAGS:
            with self.subTest(key=key),self.assertRaises(ValueError):h.require_selection_audit(dict(ck,**{key:1}),4)
        for key,value in [('full_pre_scores_supplied',False),('full_scores_supplied',False),('ordered_choices_checked',7),
            ('numerical_tolerance',1e-12),('computes_returns',True),('producer_modules_imported',True)]:
            with self.subTest(key=key),self.assertRaises(ValueError):h.require_selection_audit(dict(ck,**{key:value}),4)
    def test_full_prepared_receipt_from_synthetic122x49_and_five_prefixes(self):
        p,m,cal,a=fixture(full=True);v,au=build(p,m,cal,a);fc=check(p,m,cal,a,v,au)
        study=h.local();o,e,s,pr=h.decisions(study,a,m,v,122)
        sc=independent.verify_selection(a,m,v,o,scores=s,pre_scores=pr);counts=h.estimate_counts(au)
        prefix=[]
        for n in h.FEATURE_PREFIXES:
            pp=h.feature_prefix_panel(p,m,n);cc=h.feature_prefix_calendar(cal,m,n);aa=h.cache_prefix(a,n);mm=h.meta_prefix(m,n)
            vv,ff=build(pp,mm,cc,aa);fck=check(pp,mm,cc,aa,vv,ff)
            oo,ee,ss,prs=h.decisions(study,aa,mm,vv,n);sck=independent.verify_selection(aa,mm,vv,oo,scores=ss,pre_scores=prs,n=n)
            prefix.append(dict(prefix=n,source_rows={k:len(f) for k,f in pp.items()},source_last_session={k:str(f.index[-1].date()) if len(f) else None for k,f in pp.items()},report_records=ff['required_cells'],
                calendar_rows=len(cc),calendar_last_session=cc[-1] if n else None,feature=fck,selection=sck,counts=h.estimate_counts(ff)))
        ev=dict(feature=fc,selection=sc,prefixes=123,feature_prefixes=list(h.FEATURE_PREFIXES),physical_prefix_evidence=prefix,
            original_array_count=27,model_counts=counts,new_predictive_model_fits=0,model_fit_accounting=h.MODEL_FIT_ACCOUNTING,**counts)
        for key in ('original_orders_exact','original_arrays_unchanged','original_pre_preserved','original_core_preserved',
            'original_E_for_pre','candidate_E_only_at_episode','single_original_episode','no_added_score_budget','feature_prefix_sources_truncated',
            'calendar_prefixes_truncated','physical_cache_meta_order_prefixes','selection_uses_zero_cost_paths_only'):ev[key]=True
        completion=dict(artifacts_sha256={x:'synthetic' for x in h.PREPARED_FILES},original_array_count=27,model_counts=counts,
            new_returns_computed=False,new_predictive_model_fits=0,model_fit_accounting=h.MODEL_FIT_ACCOUNTING,**counts)
        parity=dict(calendar_sha256=h.sha(h.CALENDAR),original_array_fingerprints=h.array_fingerprints(a),strict_pit=False,
            execution_prices={k:True for k in ('adjusted_levels_constant_scale','adjusted_opening_parity','missing_price_masks_exact','positive_price_masks_exact')})
        for key in ('original_arrays_unchanged','original_orders_exact','original_pre_preserved','original_E_for_pre','candidate_E_only_at_episode'):parity[key]=True
        with patch.object(h,'read',return_value={'sessions':cal}):
            h.verify_prepared_evidence(completion,v,ev,meta=m,parity=parity,feature_audit=au)
            for kind in ('manifest','prefix','calendar','pres','fit','source'):
                co,ee,pa=copy.deepcopy((completion,ev,parity))
                if kind=='manifest':co['artifacts_sha256']['TRANSFORMS_DELTAS.npz']='not-this-design'
                if kind=='prefix':ee['prefixes']=122
                if kind=='calendar':ee['physical_prefix_evidence'][2]['calendar_rows']+=1
                if kind=='pres':ee['original_pre_preserved']=False
                if kind=='fit':co['new_model_fits']=True
                if kind=='source':pa['candidate_E_only_at_episode']=False
                with self.subTest(kind=kind),self.assertRaises(ValueError):h.verify_prepared_evidence(co,v,ee,meta=m,parity=pa,feature_audit=au)
        for n in range(123):
            oo,_,ss,prs=h.decisions(study,h.cache_prefix(a,n),h.meta_prefix(m,n),h.value_prefix(v,n),n)
            for arm in h.NAMES:self.assertEqual(oo[arm],o[arm][:n]);np.testing.assert_array_equal(prs[arm],pr[arm][:n])
    def test_exact_six_path_thirty_slice_outcome_manifest(self):
        one=dict(passed=True,monthly_executions=122,slices={s:{} for s in ('full122','first88','early53','recent35','later34')})
        check=dict(passed=True,independent_accounts=True,independent_of_production_model_and_engine=True,checked_paths=6,checked_slices=30,
            terminal_liquidation=False,accounting_source_sha256=h.sha(h.ROOT/'work/ptr169_results_independent.py'),
            paths={arm:{p[0]:copy.deepcopy(one) for p in h.ev.PATHS} for arm in h.NAMES})
        h.require_outcome_audit(check)
        for kind in ('arm','path','slice','counts','terminal'):
            bad=copy.deepcopy(check)
            if kind=='arm':bad['paths'].pop(h.NAMES[1])
            if kind=='path':bad['paths'][h.NAMES[0]].pop('next_open_cost')
            if kind=='slice':bad['paths'][h.NAMES[1]]['same_close_zero']['slices'].pop('later34')
            if kind=='counts':bad['checked_paths']=9
            if kind=='terminal':bad['terminal_liquidation']=True
            with self.subTest(kind=kind),self.assertRaises(ValueError):h.require_outcome_audit(bad)
        manifest={x:'synthetic' for x in h.RESULT_FILES};self.assertEqual(len(manifest),26);h.require_result_manifest(manifest)
        for name in h.RESULT_FILES:
            bad=dict(manifest);bad.pop(name)
            with self.assertRaises(ValueError):h.require_result_manifest(bad)
        with self.assertRaises(ValueError):h.require_result_manifest(dict(manifest,**{'COMPLETION.json':'extra'}))


class LifecycleTests(unittest.TestCase):
    def test_bad_plan_cannot_load_prices_or_original_arrays(self):
        with patch.object(h,'gate',side_effect=ValueError('badplan')),patch.object(h,'local') as load,patch.object(h.ev,'prices') as prices:
            for name in ('prepare','select','diagnose','close'):
                with self.assertRaisesRegex(ValueError,'badplan'):getattr(h,name)('wrong')
            load.assert_not_called();prices.assert_not_called()
    def test_state_counters_and_predecessor(self):
        state=dict(completed=[dict(trials=1,directory='old')]*232+[dict(trials=981,directory=h.PRIOR.name)],
            best_training_selected_candidate=dict(directory='monthly_local_linear70'))
        h.state_check(state)
        for kind in ('trials','directory','incumbent'):
            bad=copy.deepcopy(state)
            if kind=='trials':bad['completed'][-1]['trials']+=1
            if kind=='directory':bad['completed'][-1]['directory']='other'
            if kind=='incumbent':bad['best_training_selected_candidate']['directory']='other'
            with self.assertRaises(ValueError):h.state_check(bad)
    def test_metadata_only_provenance_and_context(self):
        state=h.read(h.CENTRAL);h.state_check(state);self.assertEqual(state['registered_queue'],[])
        prior=h.predecessor_binding(state);self.assertEqual(prior['directory'],'monthly_allstate_gate237')
        binding=h.price_source_binding(h.read(h.LOCAL/'PLAN.json')['raw_source_hashes'])
        self.assertTrue(binding['metadata_only']);self.assertFalse(binding['source_npz_decoded']);self.assertFalse(binding['strict_pit'])
        self.assertEqual(len(binding['raw_source_hashes']),64);self.assertEqual(binding['events'],122)
        self.assertEqual(h.contract()['registered_trials'],2);self.assertEqual(h.contract()['execution_paths'],6)
        self.assertEqual(h.contract()['selection_paths'],['same_close_zero','next_open_zero'])
    def test_source_rejects_bad_coverage_and_raw_map_before_loading(self):
        coverage=h.read(h.COVERAGE)
        for key in ('actual_features_computed','actual_eligibility_computed','actual_orders_computed','actual_returns_computed','npz_decoded','observer_files_modified'):
            bad=copy.deepcopy(coverage);bad[key]=1
            with patch.object(h,'sha',side_effect=lambda p:h.CONTEXT_SHA if p==h.CONTEXT else h.COVERAGE_SHA),patch.object(h,'read',return_value=bad),self.assertRaises(ValueError):h.price_source_binding({})
        with self.assertRaises(ValueError):h.price_source_binding({})
    def test_bind_never_overwrites_existing_hash(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'source';p.write_text('old');pins={};h.bind(pins,p);old=dict(pins);p.write_text('new')
            with self.assertRaises(ValueError):h.bind(pins,p)
            self.assertEqual(pins,old)
    def test_external_runtime_allowlist_and_mutable_central(self):
        name,digest=next(iter(h.EXTERNAL_RUNTIME.items()))
        with patch.object(Path,'is_file',return_value=True),patch.object(h,'sha',return_value=digest):
            h.verify({name:digest})
            for pins in ({name:'wrong'},{name+'other':digest},{str(h.CENTRAL):digest},{'/private/tmp/escape':digest}):
                with self.assertRaises(ValueError):h.verify(pins)
    def test_formal_tests_are_write_once_before_loading_test_modules(self):
        with patch.object(Path,'exists',return_value=True),patch.object(h,'module') as load,self.assertRaises(ValueError):h.tests()
        load.assert_not_called()
    def test_gate_dependencies_and_registered_queue(self):
        p=dict(contract={'c':1},runtime={'r':1},input_sha256={'x':'y'},raw_source_hashes={'s':'h'},price_source_binding={'src':1},
            numerical_code_review={'cr':1},prior_central_sha256='prior',predecessor_binding={'pred':1})
        state=dict(registered_queue=[dict(directory=h.OUT.name,plan_sha256='plan')])
        def digest(path):return 'plan' if path==h.OUT/'PLAN.json' else 'prior'
        def rd(path):return p if path==h.OUT/'PLAN.json' else state
        with patch.object(h,'sha',side_effect=digest),patch.object(h,'read',side_effect=rd),patch.object(h,'contract',return_value={'c':1}),patch.object(h.ev,'runtime',return_value={'r':1}),patch.object(h,'verify'),patch.object(h,'price_source_binding',return_value={'src':1}),patch.object(h,'code_review_binding',return_value={'cr':1}),patch.object(h,'state_check'),patch.object(h,'predecessor_binding',return_value={'pred':1}):
            self.assertEqual(h.gate('plan'),p)
            state['registered_queue']=[]
            with self.assertRaises(ValueError):h.gate('plan')
    def test_central_compare_and_swap_cannot_replace_changed_file(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'CENTRAL.json';p.write_bytes(b'changed')
            with patch.object(h,'CENTRAL',p),patch.object(h.os,'replace') as replace,self.assertRaises(ValueError):h.commit_state(b'expected',{},'synthetic')
            replace.assert_not_called();self.assertEqual(p.read_bytes(),b'changed')
    def test_calendars_truncate_to_next_schedule_witness(self):
        p,m,c,a=fixture()
        for n in range(5):
            cc=h.feature_prefix_calendar(c,m,n);pp=h.feature_prefix_panel(p,m,n)
            if n:self.assertEqual(cc[-2],m['dates'][n-1]);self.assertGreater(cc[-1],m['dates'][n-1]);self.assertEqual(str(pp['adj'].index[-1].date()),m['dates'][n-1])
            else:self.assertEqual(cc,[]);self.assertEqual(len(pp['adj']),0)
    def test_report_physical_prefix_counts_and_irregular_vox_cutoff(self):
        p,m,c,a=fixture();report=make_report(p,m,c,a)
        for n in range(5):
            rr=h.source_report_prefix(report,n)
            self.assertEqual(len(rr['events']),n)
            self.assertEqual(rr['counts']['required_cells'],sum(len(e['required_indices']) for e in rr['events']))
            self.assertEqual(sum(rr['repair_counts'].values()),len(rr['records']))
            self.assertNotIn('independent',rr)
            pp=h.feature_prefix_panel(p,m,n);cc=h.feature_prefix_calendar(c,m,n)
            vv,au=kernel.build_features(pp,h.meta_prefix(m,n),cc,h.cache_prefix(a,n),rr)
            self.assertEqual(au['required_cells'],len(rr['records']))
        p['vox']=p['vox'].drop(pd.Timestamp(m['dates'][-1]))
        pp=h.feature_prefix_panel(p,m,4)
        self.assertLess(str(pp['vox'].index[-1].date()),m['dates'][-1])
    def test_source_admission_failure_blocks_before_new_price_parsing(self):
        real=h.read
        def rd(path):
            d=real(path)
            if path==h.SOURCE_OUT/'COMPLETION.json':d=dict(d,admitted=False)
            return d
        with patch.object(h,'read',side_effect=rd),patch.object(h,'local') as loader,self.assertRaises(ValueError):
            h.source_admission_binding(h.read(h.LOCAL/'PLAN.json')['raw_source_hashes'])
        loader.assert_not_called()
    def test_code_declares_no_actual_before_gate_and_exact_two_arm_coverage(self):
        source=Path(h.__file__).read_text();tree=ast.parse(source)
        parts={n.name:ast.get_source_segment(source,n) for n in tree.body if isinstance(n,ast.FunctionDef)}
        self.assertLess(parts['prepare'].index('gate(expected)'),parts['prepare'].index('study.load()'))
        self.assertLess(parts['register'].index('code_review_binding()'),parts['register'].index('OUT.mkdir()'))
        self.assertLess(parts['close'].index('require_result_manifest('),parts['close'].index('results = read('))
        self.assertEqual(parts['commit_state'].count('CENTRAL.read_bytes() == before'),2)
        self.assertIn('ev.PATHS[0], ev.PATHS[1]',parts['select'])
        self.assertEqual(len(h.PREPARED_FILES),6);self.assertNotIn('TRANSFORMS_DELTAS.npz',h.PREPARED_FILES)


if __name__=='__main__':unittest.main()
