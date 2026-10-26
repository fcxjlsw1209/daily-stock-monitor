"""Synthetic-only financial scoring, immutable source gates and lifecycle checks."""
from pathlib import Path
import ast
import copy
import importlib.util
import tempfile
import unittest
from unittest.mock import Mock, patch
import numpy as np
from test_monthly_roa_stability220_scoring import fixture, Study, POLICY

spec=importlib.util.spec_from_file_location('tested_roa_stability220_host',Path(__file__).with_name('monthly_roa_stability220_host.py'))
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)


def counts_fixture(a=3):
    return dict(events=122,symbols_per_event=49,checked_cells=5978,available_cells=a,
        unavailable_cells=5978-a,annual_ratio_evaluations=3*a,mean_estimates=a,
        sample_dispersion_estimates=a,bounded_scores=2*a)


def receipt_fixture():
    v=dict(available=np.zeros((122,49),bool),candidate=np.full((122,49),np.nan),control=np.full((122,49),np.nan))
    v['available'][0,:3]=True;v['candidate'][0,:3]=[-.3,0.,.3];v['control'][0,:3]=[-.2,0.,.4]
    feature=dict({k:True for k in h.FEATURE_FLAGS},**counts_fixture(),computes_returns=False,
        producer_modules_imported=False,predictive_model_fits=0,new_model_fits=0,numerical_tolerance=0.)
    deltas={arm:np.zeros((122,49)) for arm in h.NAMES[1:]}
    delta=dict({k:True for k in h.DELTA_FLAGS},dose=.01,max_absolute_delta=.01,changed_cells={k:0 for k in deltas})
    common=dict(new_model_fits=0,predictive_model_fits=0,new_predictive_model_fits=0,
        model_fit_accounting=h.MODEL_FIT_ACCOUNTING,original_array_count=27,**{k:feature[k] for k in h.ESTIMATE_FIELDS})
    audit=dict(common,feature=feature,delta=delta,prefixes=123,feature_prefixes=list(h.FEATURE_PREFIXES),
        feature_prefix_sources_truncated=True,feature_prefix_metadata_truncated=True,
        feature_prefix_independently_verified=True,financial_source_selection_exact=True,
        original_orders_exact=True,original_core_metadata_and_support_preserved=True,
        single_original_episode=True,original_arrays_unchanged=True)
    receipt=dict(common,new_returns_computed=False,artifacts_sha256={k:'fake' for k in h.PREPARED_FILES})
    return v,deltas,feature,delta,audit,receipt


class ScoreTests(unittest.TestCase):
    def test_host_uses_dedicated_scoring_unchanged(self):
        a,m,v=fixture();expected=h.scoring.scale_and_delta(v,a,4);actual=h.scale_and_delta(v,a,4)
        self.assertEqual(h.NAMES,h.scoring.NAMES);self.assertEqual(h.DOSE,h.scoring.DOSE)
        for l,r in zip(expected,actual):
            for key in l:np.testing.assert_array_equal(l[key],r[key])

    def test_real_frozen_adapter_one_episode_and_no_rank_range_error(self):
        a,m,v=fixture();before={k:x.tobytes() for k,x in a.items()};study=Study();adapter=h.core_adapter()
        with patch.object(h,'read',return_value=POLICY),patch.object(h,'core_adapter',return_value=adapter):
            orders,t,d=h.decisions(study,a,m,v,4)
        self.assertEqual(study.ep.select_boundary.call_count,3)
        for arm,call in zip(h.NAMES[1:],study.ep.select_boundary.call_args_list[1:]):
            pre=study.modified_pre_score(dict(id=4),a,4);changed=d[arm]!=0.;pre[changed]+=d[arm][changed]
            np.testing.assert_array_equal(call.args[0],pre)
            self.assertEqual(orders[arm][0]['symbols'],list('ABE'))
            for key in ('baseline','core'):np.testing.assert_array_equal(call.args[4][key],a[key])
            np.testing.assert_array_equal(call.args[1],a['eligible'])
            np.testing.assert_array_equal(call.args[3],a['effects'])
        self.assertEqual(before,{k:x.tobytes() for k,x in a.items()})
        self.assertGreater(t[h.NAMES[1]][0,0],.5)

    def test_all_order_prefixes_are_exact_including_zero(self):
        a,m,v=fixture()
        with patch.object(h,'read',return_value=POLICY):
            full,tt,dd=h.decisions(Study(),a,m,v,4)
            for n in range(5):
                orders,t,d=h.decisions(Study(),a,m,v,n)
                self.assertEqual(orders,{k:x[:n] for k,x in full.items()})
                for arm in h.NAMES[1:]:
                    np.testing.assert_array_equal(t[arm],tt[arm][:n]);np.testing.assert_array_equal(d[arm],dd[arm][:n])

    def test_zero_missing_and_ineligible_are_exact_neutral(self):
        for value in (0.,np.nan):
            a,m,v=fixture();v['candidate'][:,:4]=value;v['control'][:,:4]=value;v['available']=np.isfinite(v['candidate'])
            with patch.object(h,'read',return_value=POLICY):orders,_,d=h.decisions(Study(),a,m,v,4)
            for arm in h.NAMES[1:]:self.assertEqual(orders[arm],orders[h.NAMES[0]]);self.assertEqual(np.count_nonzero(d[arm]),0)
        a,m,v=fixture();_,d=h.scale_and_delta(v,a,4);self.assertEqual(d[h.NAMES[1]][2,0],0.)

    def test_financial_formula_not_suppressed_on_initial_or_calm(self):
        a,m,v=fixture();a['state_calm10'][:]=True;a['core_weight'][:]=.5
        _,d=h.scale_and_delta(v,a,4)
        self.assertNotEqual(d[h.NAMES[1]][0,0],0.)
        self.assertNotEqual(d[h.NAMES[1]][1,0],0.)

    def test_physical_metadata_and_source_prefix_zero(self):
        meta=dict(dates=['2020-01-31','2020-02-28'],ends=['2020-02-28','2020-03-31'],symbols=[['A'],['A']],etfs=[])
        kernel=Mock();dataset={'catalog':['immutable']}
        h.feature_prefix_sources(kernel,dataset,meta,0);kernel.truncate_sources.assert_called_with(dataset,None)
        h.feature_prefix_sources(kernel,dataset,meta,1);kernel.truncate_sources.assert_called_with(dataset,'2020-01-31')
        for n in (0,1,2):
            mm=h.feature_prefix_meta(meta,n)
            for k in ('dates','ends','symbols'):self.assertEqual(mm[k],meta[k][:n])
        for n in (-1,3,True):
            with self.assertRaises(ValueError):h.feature_prefix_meta(meta,n)


def selection_fixture():
    metrics={name:{path:{era:dict(cagr=.5,sharpe=1.5,max_drawdown=-.3) for era in ('training','early','recent')}
        for path in ('same_close_zero','next_open_cost')} for name in h.NAMES}
    orders={name:[dict(symbols=['A','B','C']) for _ in range(122)] for name in h.NAMES}
    for path in metrics[h.NAMES[1]]:metrics[h.NAMES[1]][path]['training']['cagr']+=.01
    for i in (1,2,53):orders[h.NAMES[1]][i]['symbols']=['A','B','D']
    return metrics,orders


class SelectionTests(unittest.TestCase):
    def test_later34_never_select_and_control_cannot_promote(self):
        metrics,orders=selection_fixture();expected=h.select_first88(metrics,orders);self.assertTrue(expected['admitted'])
        for name in h.NAMES:
            metrics[name]['full122']={'cagr':1000.}
            for i in range(88,122):orders[name][i]['symbols']=[name]
        self.assertEqual(h.select_first88(metrics,orders),expected)
        for path in ('same_close_zero','next_open_cost'):
            metrics,orders=selection_fixture();metrics[h.NAMES[2]][path]['training']['cagr']=1.
            result=h.select_first88(metrics,orders);self.assertFalse(result['admitted']);self.assertEqual(result['primary'],h.NAMES[0])

    def test_frozen_gain_era_sharpe_drawdown_guards(self):
        failures=[('same_close_zero','training','cagr',.504),('next_open_cost','training','cagr',.49)]
        failures += [(path,era,key,value) for path in ('same_close_zero','next_open_cost') for era,key,value in
            [('early','cagr',.474),('recent','cagr',.474),('training','sharpe',1.449),('training','max_drawdown',-.331)]]
        for path,era,key,value in failures:
            m,o=selection_fixture();m[h.NAMES[1]][path][era][key]=value
            with self.subTest(path=path,era=era,key=key):self.assertFalse(h.select_first88(m,o)['admitted'])

    def test_three_changed_sets_both_eras_against_each_reference(self):
        m,o=selection_fixture();o[h.NAMES[2]]=copy.deepcopy(o[h.NAMES[1]])
        self.assertFalse(h.select_first88(m,o)['admitted'])
        for i in (1,2,3):o[h.NAMES[2]][i]['symbols']=['A','E','F']
        self.assertFalse(h.select_first88(m,o)['guards']['changed_support_vs_control'])
        o[h.NAMES[2]][53]['symbols']=['A','E','F'];self.assertTrue(h.select_first88(m,o)['admitted'])
        m,o=selection_fixture()
        for i in (1,2,53):o[h.NAMES[1]][i]['symbols']=['C','B','A']
        self.assertEqual(h.select_first88(m,o)['changed_months'],[])


class EvidenceTests(unittest.TestCase):
    def test_financial_estimate_accounting_uses_common_cells(self):
        for a in (0,1,5978):
            c=counts_fixture(a);self.assertEqual(h.estimate_counts(c)['annual_ratio_evaluations'],3*a)
            for key in h.ESTIMATE_FIELDS:
                bad=dict(c);bad[key]+=1
                with self.assertRaises(ValueError):h.estimate_counts(bad)
        for k,val in (('available_cells',True),('unavailable_cells',None),('checked_cells',0),('events',122.)):
            c=counts_fixture();c[k]=val
            with self.assertRaises(ValueError):h.estimate_counts(c)

    def test_all_independent_feature_and_delta_flags_required(self):
        v,d,f,dd,_,_=receipt_fixture();h.require_feature_audit(f,v);h.require_delta_audit(dd,d)
        for key in h.FEATURE_FLAGS:
            bad=dict(f);bad.pop(key)
            with self.assertRaises(ValueError):h.require_feature_audit(bad,v)
        for key in h.DELTA_FLAGS:
            bad=dict(dd);bad.pop(key)
            with self.assertRaises(ValueError):h.require_delta_audit(bad,d)
        for k,val in (('dose',.025),('max_absolute_delta',.005),('changed_cells',{})):
            bad=dict(dd);bad[k]=val
            with self.assertRaises(ValueError):h.require_delta_audit(bad,d)

    def test_prepared_evidence_requires_all_source_prefixes_and_original_inputs(self):
        v,d,_,_,a,c=receipt_fixture();h.verify_prepared_evidence(c,v,d,a)
        for key in ('feature_prefix_sources_truncated','feature_prefix_metadata_truncated','feature_prefix_independently_verified',
            'financial_source_selection_exact','original_orders_exact','original_core_metadata_and_support_preserved','single_original_episode','original_arrays_unchanged'):
            bad=copy.deepcopy(a);bad[key]=False
            with self.assertRaises(ValueError):h.verify_prepared_evidence(c,v,d,bad)
        for key in ('new_model_fits','new_predictive_model_fits')+h.ESTIMATE_FIELDS:
            bad=copy.deepcopy(c);bad[key]=None
            with self.assertRaises(ValueError):h.verify_prepared_evidence(bad,v,d,a)
        bad=copy.deepcopy(a);bad['feature_prefixes']=[0,122]
        with self.assertRaises(ValueError):h.verify_prepared_evidence(c,v,d,bad)

    def test_source_audit_cell_identities_and_counts(self):
        m=dict(dates=['2020-01-31'],symbols=[['A','QQQ']],etfs=['QQQ'])
        v=dict(available=np.array([[True,False]]),candidate=np.array([[.1,np.nan]]),control=np.array([[.2,np.nan]]))
        cell=dict(symbol='A',date=m['dates'][0],available=True,reason='available',latest_event_id='A3',report_version_ids=['r1','r2','r3'],edge_version_ids=['e1','e2'],chain_version_id='c1',measurement={'candidate':.1,'control':.2})
        missing=dict(symbol='QQQ',date=m['dates'][0],available=False,reason='ETF',latest_event_id=None,report_version_ids=[],edge_version_ids=[],chain_version_id=None)
        a=dict(available_cells=1,event_history=[dict(date=m['dates'][0],symbols=m['symbols'][0],available_count=1,cells=[cell,missing])])
        self.assertTrue(h.require_source_support(a,m,v))
        for mode in ('identity','count','reports','measurement','unavailablemeasurement'):
            bad=copy.deepcopy(a);c=bad['event_history'][0]['cells'][0]
            if mode=='identity':c['symbol']='B'
            if mode=='count':bad['event_history'][0]['available_count']=2
            if mode=='reports':c['report_version_ids']=['r1','r1','r3']
            if mode=='measurement':c['measurement']['candidate']=.2
            if mode=='unavailablemeasurement':bad['event_history'][0]['cells'][1]['measurement']={}
            with self.subTest(mode=mode),self.assertRaises(ValueError):h.require_source_support(bad,m,v)

    def test_outcomes_all_nine_paths_and_fortyfive_slices(self):
        slices={k:{} for k in ('full122','first88','early53','recent35','later34')}
        c=dict(passed=True,independent_accounts=True,independent_of_production_model_and_engine=True,
            checked_paths=9,checked_slices=45,terminal_liquidation=False,accounting_source_sha256=h.sha(h.ROOT/'work/ptr169_results_independent.py'),
            paths={arm:{p[0]:dict(passed=True,monthly_executions=122,slices=copy.deepcopy(slices)) for p in h.ev.PATHS} for arm in h.NAMES})
        h.require_outcome_audit(c)
        for mode in ('arm','path','slice','terminal'):
            bad=copy.deepcopy(c)
            if mode=='arm':bad['paths'].pop(h.NAMES[2])
            if mode=='path':bad['paths'][h.NAMES[1]].pop(h.ev.PATHS[2][0])
            if mode=='slice':bad['paths'][h.NAMES[1]][h.ev.PATHS[0][0]]['slices'].pop('later34')
            if mode=='terminal':bad['terminal_liquidation']=True
            with self.assertRaises(ValueError):h.require_outcome_audit(bad)

    def test_exact_manifest_sets(self):
        self.assertEqual(len(h.PREPARED_FILES),7);self.assertEqual(len(h.RESULT_FILES),38)
        manifest={k:'synthetic' for k in h.RESULT_FILES};h.require_result_manifest(manifest)
        for missing in manifest:
            bad=dict(manifest);bad.pop(missing)
            with self.assertRaises(ValueError):h.require_result_manifest(bad)
        for extra in ('COMPLETION.json','EXTRA.json'):
            with self.assertRaises(ValueError):h.require_result_manifest(dict(manifest,**{extra:'fake'}))


class InterfaceTests(unittest.TestCase):
    def test_full_122x49_synthetic_feature_delta_and_five_physical_source_prefixes(self):
        import monthly_roa_stability220_kernel as kernel
        import monthly_roa_stability220_independent as independent
        from test_monthly_roa_stability220_kernel import fixture as financial_fixture
        dataset,calendar=financial_fixture()
        monthends=[day for i,day in enumerate(calendar[:-1]) if calendar[i+1][:7]!=day[:7] and day>='2016-08-31'][:121]
        dates=['2016-08-17']+monthends;symbols=['AAA']+['S'+str(i) for i in range(47)]+['ETF']
        meta=dict(dates=dates,ends=dates,symbols=[symbols.copy() for _ in dates],etfs=['ETF'])
        values,audit=kernel.build_features(dataset,meta,calendar)
        check=independent.verify_features(dataset,meta,calendar,values,audit)
        self.assertEqual(h.FEATURE_FLAGS,independent.FEATURE_FLAGS);self.assertEqual(h.DELTA_FLAGS,independent.DELTA_FLAGS)
        h.require_feature_audit(check,values);h.require_source_support(audit,meta,values)
        self.assertGreater(check['available_cells'],0);self.assertLess(check['available_cells'],5978)
        arrays=dict(baseline=np.zeros((122,49)),eligible=np.ones((122,49),bool));arrays['eligible'][20,0]=False
        targets,deltas=h.scale_and_delta(values,arrays,122)
        h.require_delta_audit(independent.verify_deltas(values,arrays,targets,deltas),deltas)
        for n in h.FEATURE_PREFIXES:
            partial=h.feature_prefix_sources(kernel,dataset,meta,n);mm=h.feature_prefix_meta(meta,n)
            small,small_audit=kernel.build_features(partial,mm,calendar)
            self.assertTrue(independent.verify_features(partial,mm,calendar,small,small_audit)['passed'])
            self.assertEqual(small_audit['event_history'],audit['event_history'][:n])
            for key in values:
                if n:np.testing.assert_array_equal(small[key],values[key][:n])
                else:self.assertEqual(small[key].shape,(0,0))
        for n in (0,1,53,88,122):
            tt,dd=h.scale_and_delta(values,arrays,n)
            h.require_delta_audit(independent.verify_deltas({k:v[:n] for k,v in values.items()},arrays,tt,dd),dd)
            for arm in h.NAMES[1:]:np.testing.assert_array_equal(dd[arm],deltas[arm][:n])

    def test_context_contract_and_complete_transitive_code_files(self):
        contract=h.contract()
        self.assertEqual(contract['dose'],.01);self.assertEqual(contract['max_absolute_delta'],.01)
        self.assertEqual(contract['arms'],list(h.NAMES));self.assertFalse(contract['strict_pit'])
        self.assertEqual(contract['verification']['feature_prefixes'],[0,1,53,88,122])
        for filename in ('monthly_roa_stability220_scoring.py','test_monthly_roa_stability220_scoring.py','roa_stability_preflight_math.py','test_roa_stability_preflight_math.py'):
            self.assertIn(h.ROOT/'work'/filename,h.code_files())
        self.assertNotEqual(h.CALENDAR,h.SOURCE_CALENDAR)


class LifecycleTests(unittest.TestCase):
    def test_unregistered_prepare_never_loads_cache_or_financial_source(self):
        with patch.object(h,'gate',side_effect=ValueError('unregistered')),patch.object(h,'local') as loader,patch.object(h,'load_source_dataset') as source:
            with self.assertRaisesRegex(ValueError,'unregistered'):h.prepare('bad')
        loader.assert_not_called();source.assert_not_called()

    def test_central_counter_contract(self):
        rows=[dict(trials=1,directory='old') for _ in range(217)];rows[0]['trials']=952;rows[-1]['directory']=h.PRIOR.name
        state=dict(completed=rows,best_training_selected_candidate={'directory':'monthly_local_linear70'})
        h.state_check(state);state['completed'][-1]['directory']='wrong'
        with self.assertRaises(ValueError):h.state_check(state)
        code=Path(h.__file__).read_text();self.assertIn("len(state['completed']) == 218",code);self.assertIn('== 1171',code)

    def test_pin_conflict_preserves_original_binding(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'input';p.write_text('old');pins={};h.bind(pins,p);before=dict(pins)
            p.write_text('new')
            with self.assertRaises(ValueError):h.bind(pins,p)
            self.assertEqual(pins,before)

    def test_only_exact_external_runtime_allowed(self):
        runtime,digest=next(iter(h.EXTERNAL_RUNTIME.items()))
        with patch.object(h,'sha',return_value=digest),patch.object(Path,'is_file',return_value=True):
            h.verify({runtime:digest})
            for p,d in ((runtime,'wrong'),('/Library/Developer/CommandLineTools/other.py',digest),('/tmp/other.py',digest),(str(h.CENTRAL),digest)):
                with self.subTest(path=p),self.assertRaises(ValueError):h.verify({p:d})

    def test_review_context_scope_files_and_mutable_central(self):
        files=[h.ROOT/'work'/('synthetic'+str(i)) for i in range(10)]
        r=dict(passed=True,unresolved_blockers=[],context_sha256=h.CONTEXT_SHA,code_sha256={str(p.relative_to(h.ROOT)):'digest' for p in files},input_sha256={'work/fake':'digest'})
        flags=('actual_features_computed','actual_model_fits_computed','actual_portfolio_returns_computed','production_files_modified','strategy_registered_by_reviewer')
        r.update({k:False for k in flags})
        with patch.object(h,'code_files',return_value=files),patch.object(h,'read',return_value=r),patch.object(h,'sha',return_value='digest'),patch.object(h,'bind'),patch.object(h,'verify'):
            h.code_review_binding()
            for key,value in (('context_sha256','wrong'),('actual_model_fits_computed',True),('input_sha256',{}),('input_sha256',{str(h.CENTRAL):'digest'})):
                old=r[key];r[key]=value
                with self.subTest(key=key),self.assertRaises(ValueError):h.code_review_binding()
                r[key]=old

    def test_writeonce_gate_order_and_two_cas_checks(self):
        code=Path(h.__file__).read_text();tree=ast.parse(code)
        f={n.name:ast.get_source_segment(code,n) for n in tree.body if isinstance(n,ast.FunctionDef)}
        for needle in ('code_review_binding()','len(protected) == 21444','Mutable central cannot enter registered input pins'):
            self.assertLess(f['register'].index(needle),f['register'].index('OUT.mkdir()'))
        self.assertEqual(f['commit_state'].count('CENTRAL.read_bytes() == before'),2)
        self.assertLess(f['prepare'].index('gate(expected)'),f['prepare'].index('study.load()'))
        self.assertLess(f['close'].index('require_result_manifest('),f['close'].index('require_outcome_audit('))
        self.assertIn('Selection escaped first88 cutoff',f['select'])
        self.assertIn('Code changed since tests',f['register'])


def source_documents():
    raw={'raw/'+str(i):'digest' for i in range(64)}
    review=dict(passed=True,context_sha256=h.CONTEXT_SHA,raw_source_hashes=raw,source_contract_checked=True,
        actual_new_features_computed=False,actual_new_orders_computed=False,actual_new_returns_computed=False,
        input_sha256={'work/immutable':'digest'},evidence_sha256={'work/review-only':'digest'})
    context=dict(evidence_sha256={'work/context-only':'digest'})
    checks={k:True for k in ('raw_digest_bindings','own_report_three_field_bindings','original_field_admissions_preserved','exact_identity_no_ticker_join','reported_NI_and_assets_checked_independently','admitted_edge_exact_scaled_source_equality','event_versions_not_deduplicated_by_raw_sha','complete_supplied_catalog_preserved','no_first_public_PIT_certification')}
    checks.update({k:0 for k in ('ROA_calculations','means','standard_deviations','factors','ranks','monthly_support_matches','orders','returns','network_requests','source_admission_updates_outside_new_directory','observer_changes')})
    docs={h.SOURCE_REVIEW:review,h.CONTEXT:context,
        h.SOURCE_DIR/'MANIFEST.json':dict(source_only=True,write_once=True,files={k:(h.SOURCE_DATASET_SHA if k=='DATASET.json' else 'digest') for k in h.SOURCE_FILES},input_sha256={'work/source':'digest'}),
        h.SOURCE_DIR/'VALIDATION.json':dict(passed=True,source_only=True,checks=checks),
        h.SOURCE_DIR/'POLICY.json':dict(no_strategy_registration=True,first_public_version_verified=False,complete_correction_history_verified=False,certified_daily_security_master=False),
        h.SOURCE_DIR/'CLOCK_REVIEW.json':dict(calendar_ref=str(h.SOURCE_CALENDAR.relative_to(h.ROOT)),calendar_sha256=h.SOURCE_CALENDAR_SHA,no_outside_calendar_clock_invented=True,actual_signal_queries=0)}
    return raw,docs


class FinancialSourceTests(unittest.TestCase):
    def test_final_source_receipt_metadata_hash_gate_without_numeric_load(self):
        raw=h.read(h.LOCAL/'PLAN.json')['raw_source_hashes']
        binding=h.financial_source_binding(raw)
        self.assertEqual(binding['sha256'],h.SOURCE_REVIEW_SHA)
        self.assertEqual(binding['manifest_sha256'],h.SOURCE_MANIFEST_SHA)
        self.assertEqual(binding['dataset_sha256'],h.SOURCE_DATASET_SHA)
        self.assertFalse(binding['monthly_support_computed']);self.assertFalse(binding['strict_pit'])
        self.assertEqual(binding['input_sha256'][str(h.SOURCE_CALENDAR)],h.SOURCE_CALENDAR_SHA)

    def call_binding(self,raw,docs):
        def digest(p):
            return h.SOURCE_REVIEW_SHA if p==h.SOURCE_REVIEW else h.SOURCE_MANIFEST_SHA if p==h.SOURCE_DIR/'MANIFEST.json' else 'digest'
        with patch.object(h,'SOURCE_FLAGS',('source_contract_checked',)),patch.object(h,'read',side_effect=lambda p:docs[p]),patch.object(h,'sha',side_effect=digest),patch.object(h,'bind'),patch.object(h,'verify'):
            return h.financial_source_binding(raw)

    def test_source_and_context_evidence_are_separate_nonidentical_maps(self):
        raw,docs=source_documents();binding=self.call_binding(raw,docs)
        self.assertNotEqual(binding['source_evidence_sha256'],binding['context_evidence_sha256'])
        self.assertFalse(binding['monthly_support_computed']);self.assertFalse(binding['strict_pit'])

    def test_source_requires_exact_manifest_and_no_preplan_strategy_work(self):
        for mode in ('missing','extra','boolzero','features','unpassed','PIT','calendar','inventedclock'):
            raw,docs=source_documents()
            if mode=='missing':docs[h.SOURCE_DIR/'MANIFEST.json']['files'].pop('POLICY.json')
            if mode=='extra':docs[h.SOURCE_DIR/'MANIFEST.json']['files']['EXTRA.json']='digest'
            if mode=='boolzero':docs[h.SOURCE_DIR/'VALIDATION.json']['checks']['monthly_support_matches']=False
            if mode=='features':docs[h.SOURCE_REVIEW]['actual_new_features_computed']=True
            if mode=='unpassed':docs[h.SOURCE_DIR/'VALIDATION.json']['passed']=False
            if mode=='PIT':docs[h.SOURCE_DIR/'POLICY.json']['first_public_version_verified']=True
            if mode=='calendar':docs[h.SOURCE_DIR/'CLOCK_REVIEW.json']['calendar_ref']=str(h.CALENDAR.relative_to(h.ROOT))
            if mode=='inventedclock':docs[h.SOURCE_DIR/'CLOCK_REVIEW.json']['no_outside_calendar_clock_invented']=False
            with self.subTest(mode=mode),self.assertRaises(ValueError):self.call_binding(raw,docs)

    def test_each_review_evidence_map_is_mandatory(self):
        for path,key in ((h.SOURCE_REVIEW,'input_sha256'),(h.SOURCE_REVIEW,'evidence_sha256'),(h.CONTEXT,'evidence_sha256')):
            raw,docs=source_documents();docs[path][key]={}
            with self.subTest(path=path,key=key),self.assertRaises(ValueError):self.call_binding(raw,docs)

    def test_pending_source_contract_never_passes(self):
        raw,docs=source_documents()
        with patch.object(h,'SOURCE_FLAGS',('PENDING_FINAL',)),patch.object(h,'read',side_effect=lambda p:docs[p]),patch.object(h,'sha',return_value=h.SOURCE_REVIEW_SHA):
            with self.assertRaisesRegex(ValueError,'not finalized'):h.financial_source_binding(raw)


if __name__=='__main__':unittest.main(verbosity=2)
