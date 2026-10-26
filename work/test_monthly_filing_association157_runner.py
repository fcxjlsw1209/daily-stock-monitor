"""Synthetic-only orchestration checks; no actual source or price cache reads."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import monthly_filing_association157_runner as r


def partition(t):
    labels,owners=['N']*210,[None]*210
    bundles=[]
    for k,a in enumerate((t-240,t-150)):
        for d in range(a,a+21):
            labels[d-(t-251)]='B' if d<a+2 else 'E'
            owners[d-(t-251)]=a
        bundles.append(dict(anchor=a,event_ids=['E'+str(k)],forms=['10-Q'],amendment=False,
                            owned_E_days=list(range(a+2,a+21)),full19=True))
    return dict(first_day=t-251,last_day=t-42,bundles=bundles,labels=''.join(labels),
                owner_anchor_indices=owners,counts=dict(E=38,B=4,N=168),full19_original_bundles=2)


def source_fixture(active=16, only_first=False):
    sessions=pd.bdate_range('2000-01-03',periods=1300).strftime('%Y-%m-%d').tolist()
    names=['S%d'%j for j in range(32)]+['ETF%d'%j for j in range(17)]
    dates=[sessions[600+i*4] for i in range(122)]
    meta=dict(dates=dates,ends=dates[1:]+[sessions[1090]],symbols=[names[:] for _ in dates],etfs=names[32:])
    rows=[]
    for i,day in enumerate(dates):
        for j,symbol in enumerate(names):
            usable=j<active and (not only_first or i==0)
            rows.append(dict(signal_index=i,candidate_index=j,symbol=symbol,decision_date=day,
                decision_cutoff=day+'T21:00:00Z',decision_cutoff_ns=int(pd.Timestamp(day+'T21:00:00Z').value),
                signal_session_index=600+i*4,research_eligible=usable,reasons=[] if usable else ['source_unavailable'],
                issuer_id='CIK'+str(j) if j<32 else None,partition=partition(600+i*4) if usable else None,
                event_ids=['E0','E1'] if usable else [],context_original_event_ids=[],source_refs=[dict(path='source.json',sha256='a'*64)]))
    return meta,rows,sessions


def prices(sessions):
    mr=.002*np.sin(np.arange(len(sessions)))+.0001
    return pd.DataFrame(dict(SPY=100*np.exp(np.cumsum(mr)),S0=20*np.exp(np.cumsum(2*mr+.0003))),index=pd.to_datetime(sessions))


class Adapter:
    def __init__(self):self.calls=0
    def score_and_choose(self,*args,**kwargs):
        self.calls+=1
        assert args[:2]==('incumbent66',0.) and kwargs['boundary_selector']=='original'
        score=kwargs['pre_score'][:args[5]].copy()
        return [sorted(range(49),key=lambda j:(-row[j],j))[:3] for row in score],score


def scoring_fixture():
    meta,_,_=source_fixture()
    arrays=dict(baseline=np.zeros((122,49)),eligible=np.ones((122,49),bool),effects=np.zeros((122,49)),common_available=np.zeros((122,49),bool))
    for key in r.HANDS:
        arrays[key]=np.full((122,49),np.nan);arrays[key][:,0]=.5
    arrays['common_available'][:,0]=True
    backend=SimpleNamespace(adapter=Adapter(),selector='original')
    backend.pre=lambda arrays,n:np.zeros((n,49))
    backend.baseline=lambda arrays,meta,n,policy:([[0,1,2] for _ in range(n)],np.zeros((n,49)))
    backend.make_orders=lambda choices,scores,meta:[dict(symbols=[meta['symbols'][i][j] for j in row]) for i,row in enumerate(choices)]
    return arrays,meta,backend


def measured(name,cagr=.4):
    return dict(name=name,training=dict(cagr=cagr,sharpe=1.5,max_drawdown=-.25),early=dict(cagr=.3),recent=dict(cagr=.5))


class MetadataTests(unittest.TestCase):
    def test_exact_half_coverage_and_fixed_denominators(self):
        meta,rows,_=source_fixture()
        gate=r.coverage_gate(meta,rows)
        self.assertTrue(gate['coverage_floor_passed'])
        self.assertEqual(gate['eras']['first88']['fraction'],.5)
        self.assertEqual(gate['eras']['first88']['stock_slots'],2816)
        self.assertEqual(gate['eras']['all122']['stock_slots'],3904)

    def test_duplicate_classes_do_not_inflate_issuer_breadth(self):
        meta,rows,_=source_fixture()
        for row in rows:
            if row['research_eligible']:row['issuer_id']='same_issuer'
        gate=r.coverage_gate(meta,rows)
        self.assertFalse(gate['coverage_floor_passed'])
        self.assertEqual(gate['eras']['first88']['distinct_issuers'],1)
        self.assertEqual(gate['eras']['first88']['months_with_at_least8_issuers'],0)

    def test_later34_does_not_decide_coverage(self):
        meta,rows,_=source_fixture()
        for row in rows:
            if row['signal_index']>=88:row.update(research_eligible=False,reasons=['late_missing'])
        self.assertTrue(r.coverage_gate(meta,rows)['coverage_floor_passed'])

    def test_post_price_mask_cannot_add_unavailable_sources(self):
        meta,rows,_=source_fixture()
        with self.assertRaises(ValueError):r.coverage_gate(meta,rows,np.ones((122,49),bool))

    def test_reordered_and_etf_admitted_rows_rejected(self):
        meta,rows,_=source_fixture()
        bad=deepcopy(rows);bad[0],bad[1]=bad[1],bad[0]
        with self.assertRaises(ValueError):r.validate_rows(meta,bad)
        bad=deepcopy(rows);bad[32].update(research_eligible=True,reasons=[],issuer_id='ETF')
        with self.assertRaises(ValueError):r.validate_rows(meta,bad)

    def test_partition_rebuilt_and_truncated_or_overlapping_bundle_rejected(self):
        _,rows,_=source_fixture(active=1,only_first=True)
        self.assertEqual(r.validate_partition(rows[0]).count('E'),38)
        for edit in ('label','owner','anchor','full','duplicate'):
            row=deepcopy(rows[0]);p=row['partition']
            if edit=='label':p['labels']='N'+p['labels'][1:].replace('E','N',1)
            if edit=='owner':p['owner_anchor_indices'][20]=123
            if edit=='anchor':p['bundles'][1]['anchor']=row['signal_session_index']-45
            if edit=='full':p['full19_original_bundles']=3
            if edit=='duplicate':p['bundles'][1]['event_ids']=['E0']
            with self.subTest(edit=edit),self.assertRaises(ValueError):r.validate_partition(row)

    def test_warmup_is_not_shortened(self):
        _,rows,_=source_fixture(active=1,only_first=True)
        rows[0]['signal_session_index']=503
        with self.assertRaises(ValueError):r.validate_partition(rows[0])


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.meta,self.rows,self.sessions=source_fixture(active=1,only_first=True)
        self.px=prices(self.sessions)

    def test_exact_window_one_estimate_and_common_mask(self):
        result=r.build_features(self.meta,self.rows,self.sessions,self.px)
        audit=result['audit'];self.assertEqual(r.counters(audit),dict(nuisance_beta_estimates_attempted=1,nuisance_beta_estimates_completed=1,new_predictive_model_fits=0))
        self.assertAlmostEqual(audit['cells'][0]['beta'],2.,places=10)
        self.assertEqual(audit['cells'][0]['price_window'],[self.sessions[96],self.sessions[558]])
        for key in r.HANDS:np.testing.assert_array_equal(np.isfinite(result['arrays'][key]),result['arrays']['common_available'])
        self.assertFalse(audit['common_coverage']['coverage_floor_passed'])

    def test_future_prices_never_enter_features(self):
        before=r.build_features(self.meta,self.rows,self.sessions,self.px)
        self.px.loc[pd.to_datetime(self.sessions[559:]),:]=np.nan
        after=r.build_features(self.meta,self.rows,self.sessions,self.px)
        self.assertEqual(before['audit'],after['audit'])
        for key in r.HANDS:np.testing.assert_array_equal(before['arrays'][key],after['arrays'][key])

    def test_missing_prices_are_common_neutral_and_not_estimated(self):
        self.px.loc[self.sessions[97],'S0']=np.nan
        result=r.build_features(self.meta,self.rows,self.sessions,self.px)
        self.assertEqual(result['audit']['usable_rows'],0)
        self.assertEqual(result['audit']['nuisance_beta_estimates_attempted'],0)
        self.assertFalse(result['arrays']['common_available'].any())

    def test_zero_market_variance_counts_attempt_but_not_completion(self):
        self.px['SPY']=100.
        audit=r.build_features(self.meta,self.rows,self.sessions,self.px)['audit']
        self.assertEqual(audit['nuisance_beta_estimates_attempted'],1)
        self.assertEqual(audit['nuisance_beta_estimates_completed'],0)
        self.assertEqual(audit['cells'][0]['reason'],'zero_preformation_market_variance')

    def test_missing_calendar_day_hard_fails_instead_of_compressing(self):
        self.px=self.px.drop(pd.Timestamp(self.sessions[97]))
        with self.assertRaises(ValueError):r.build_features(self.meta,self.rows,self.sessions,self.px)

    def test_source_unavailable_skips_numeric_measure(self):
        self.rows[0].update(research_eligible=False,reasons=['source_unavailable'])
        with patch.object(r.factor,'measure',side_effect=AssertionError('No measurement expected')):
            audit=r.build_features(self.meta,self.rows,self.sessions,self.px)['audit']
        self.assertEqual(audit['nuisance_beta_estimates_attempted'],0)

    def test_no_mutation_of_source_or_synthetic_prices(self):
        before=deepcopy(self.rows);px=self.px.copy()
        r.build_features(self.meta,self.rows,self.sessions,self.px)
        self.assertEqual(before,self.rows);pd.testing.assert_frame_equal(px,self.px)


class ScoreTests(unittest.TestCase):
    def test_reuses_frozen_scoring_with_original_eligibility_and_one_episode(self):
        self.assertIs(r.corrected,r.prior.corrected)
        arrays,meta,backend=scoring_fixture();arrays['eligible'][0,0]=False
        _,score=r.choose(r.NAMES[1],arrays,meta,2,backend,{})
        self.assertEqual(score[0,0],0.);self.assertEqual(score[1,0],.005)
        self.assertEqual(backend.adapter.calls,1)

    def test_all123_prefixes_and_null_fallback_preserve_input_bytes(self):
        arrays,meta,backend=scoring_fixture();before={k:v.tobytes() for k,v in arrays.items()}
        orders,scores=r.panels(arrays,meta,backend,{})
        self.assertEqual(list(orders),list(r.NAMES));self.assertEqual(set(scores),set(r.NAMES))
        self.assertTrue(all(len(v)==122 for v in orders.values()))
        self.assertEqual(backend.adapter.calls,3*124+6)
        self.assertEqual(before,{k:v.tobytes() for k,v in arrays.items()})

    def test_future_bad_hand_does_not_touch_earlier_prefix(self):
        arrays,meta,backend=scoring_fixture();arrays[r.HANDS[0]][121,0]=np.inf
        r.choose(r.NAMES[1],arrays,meta,88,backend,{})
        with self.assertRaises(ValueError):r.choose(r.NAMES[1],arrays,meta,122,backend,{})

    def test_inconsistent_common_mask_rejected_before_scoring(self):
        arrays,meta,backend=scoring_fixture();arrays[r.HANDS[0]][0,0]=np.nan
        with self.assertRaises(ValueError):r.panels(arrays,meta,backend,{})
        self.assertEqual(backend.adapter.calls,0)

    def test_only_primary_can_promote_and_all_controls_required(self):
        rows=[measured(n,.4+i*.01) for i,n in enumerate(r.NAMES)]
        self.assertEqual(r.select(rows,rows)['primary'],r.NAMES[3])
        for i in range(3):
            bad=deepcopy(rows);bad[i]['training']['cagr']=.9
            self.assertEqual(r.select(bad,rows)['primary'],r.NAMES[0])
        equal=[measured(n) for n in r.NAMES]
        self.assertFalse(r.select(equal,equal)['admitted'])

    def test_open_incumbent_noninferiority_and_nonfinite_rejection(self):
        rows=[measured(n,.4+i*.01) for i,n in enumerate(r.NAMES)]
        opened=deepcopy(rows);opened[0]['training']['cagr']=opened[3]['training']['cagr']
        self.assertTrue(r.select(rows,opened)['admitted'])
        rows[-1]['training']['cagr']=np.nan
        self.assertFalse(r.select(rows,opened)['admitted'])


class OrchestrationTests(unittest.TestCase):
    def test_proposal_binds_no_preregistered_nuisance_estimates(self):
        p=dict(round=157,arms=list(r.NAMES),primary_challenger=r.NAMES[-1],dose=.01,
            selection_contract=r.selection_contract(),measurement=r.measurement_contract(),status='FINAL_PROSPECTIVE_SPECIFICATION_BEFORE_REGISTRATION',
            actual_new_factors=0,actual_new_orders=0,actual_new_returns=0,actual_new_nuisance_estimates=0,
            actual_new_predictive_model_fits=0)
        r.validate_proposal(p)
        for key in ('actual_new_nuisance_estimates','actual_new_predictive_model_fits'):
            for bad in (1,False,None):
                changed=deepcopy(p);changed[key]=bad
                with self.assertRaises(ValueError):r.validate_proposal(changed)

    def test_selection_contract_separately_counts_predictive_and_nuisance_fits(self):
        declared=r.selection_contract();legacy=r.prior.selection_contract()
        self.assertNotIn('new_model_fits',declared)
        self.assertEqual(declared.pop('new_predictive_model_fits'),0)
        self.assertEqual(declared.pop('nuisance_estimates'),'preformation_beta_attempts_and_completions_counted_separately')
        legacy.pop('new_model_fits')
        self.assertEqual(declared,legacy)

    def test_registration_nuisance_count_checked_before_sources_or_backend(self):
        plan=dict(round=157,status='REGISTERED',arms=list(r.NAMES),primary_challenger=r.NAMES[-1],
            dose=.01,tolerances=r.TOLERANCES,runtime=dict(python=r.sys.version,numpy=np.__version__,pandas=pd.__version__),
            actual_feature_calculations_before_registration=0,actual_returns_before_registration=0,
            actual_model_fits_before_registration=0,new_market_downloads=0)
        for bad in (None,False,1):
            plan['actual_nuisance_estimates_before_registration']=bad
            with patch.object(r,'read',return_value=plan),patch.object(r,'sha',return_value='a'*64):
                with self.assertRaisesRegex(ValueError,'preregistration activity'):
                    r.Runner('/private/tmp','/private/tmp/PLAN.json','a'*64)

    def test_prepare_and_reload_gate_precedes_any_backend(self):
        for phase in ('prepare','loaded'):
            runner=object.__new__(r.Runner);runner.out=Path('/private/tmp/nonexistent_filing157_test')
            runner.backend_factory=lambda *args:(_ for _ in ()).throw(AssertionError('Backend must not run'))
            with patch.object(runner,'refresh'),patch.object(runner,'cache'),patch.object(runner,'require_source_grid',side_effect=ValueError('source failed')):
                with self.assertRaisesRegex(ValueError,'source failed'):getattr(runner,phase)()

    def test_post_price_floor_failure_saved_before_scoring(self):
        with tempfile.TemporaryDirectory() as temp:
            runner=object.__new__(r.Runner);runner.out=Path(temp);runner.root=Path(temp);runner.plan_sha='a'*64
            runner.inputs=dict(source_manifest=Path(temp)/'manifest',source_grid_review=Path(temp)/'review')
            runner.plan=dict(raw_source_hashes={})
            meta,rows,sessions=source_fixture(active=1,only_first=True)
            result=r.build_features(meta,rows,sessions,prices(sessions))
            backend=SimpleNamespace(load=lambda:(meta,dict.fromkeys(range(27)),None,None),price_parity=lambda *args:{})
            runner.backend_factory=lambda *args:backend
            with patch.object(runner,'refresh'),patch.object(runner,'require_source_grid',return_value=(meta,rows,sessions)),patch.object(r.base,'validate_meta'),patch.object(r,'sha',return_value='a'*64),patch.object(r,'build_features',return_value=result),patch.object(r,'panels',side_effect=AssertionError('No score after failed floor')):
                with self.assertRaisesRegex(ValueError,'Post-price common coverage'):runner.prepare()
            self.assertEqual({p.name for p in Path(temp).iterdir()},{'FEATURE_AUDIT.json','PREPARE_FAILURE.json'})
            self.assertFalse(json.loads((Path(temp)/'PREPARE_FAILURE.json').read_text())['strategy_returns_or_nav_computed'])

    def test_training_adapter_preserves_names_folds_and_first88_truncation(self):
        backend=object.__new__(r.Backend)
        orders={name:[dict(symbols=['S']) for _ in range(88)]+[None]*34 for name in r.NAMES}
        calls=[]
        rp=SimpleNamespace(paths=lambda px,meta,n:(None,None,list(range(123)),None),
            fast_returns=lambda selected,blocks,valid:(calls.append(len(selected)) or np.zeros(88)),
            stats_rr=lambda values,start,end:dict(count=len(values)))
        measure=lambda *args:{k:v for k,v in measured('unused').items() if k!='name'}
        backend.local=SimpleNamespace(rp=rp,old=SimpleNamespace(session=SimpleNamespace(measure=measure)))
        backend.open_training=lambda orders,meta,px,op,n,names:[measured(name) for name in names]
        meta=dict(symbols=[['S']]*122,dates=['d']*122,ends=['e']*122)
        rows,opened,walk=backend.training(orders,meta,None,None)
        self.assertEqual(calls,[88]*4);self.assertEqual([x['name'] for x in rows],list(r.NAMES))
        self.assertEqual([x['name'] for x in opened],list(r.NAMES));self.assertEqual([(x['n'],x['end']) for x in walk['folds']],[(36,60),(60,76),(76,88)])
        self.assertTrue(all(x['selection']['primary']==r.NAMES[0] for x in walk['folds']))

    def test_diagnostics_adapter_writes_twelve_new_named_paths(self):
        backend=object.__new__(r.Backend);written=[]
        orders={name:[] for name in r.NAMES}
        def frozen(self,renamed,meta,px,op,writer):
            self.assert_unused=None
            assert list(renamed)==list(r.previous.NAMES)
            for name in r.previous.NAMES:
                for path in ('same_close','next_open_zero','next_open_cost'):
                    writer(name+'_'+path+'_DAILY.json',[])
            return {name:{} for name in r.previous.NAMES}
        with patch.object(r.previous.Backend,'diagnostics',frozen):
            result=backend.diagnostics(orders,{},None,None,lambda name,value:written.append(name))
        self.assertEqual(set(result),set(r.NAMES));self.assertEqual(len(written),12)
        self.assertTrue(all(any(name.startswith(n+'_') for n in r.NAMES) for name in written))

    def test_counter_type_bounds_and_audit_consistency(self):
        good=dict(nuisance_beta_estimates_attempted=3,nuisance_beta_estimates_completed=2,new_predictive_model_fits=0)
        self.assertEqual(r.counters(good),good)
        for bad in (dict(good,new_predictive_model_fits=1),dict(good,nuisance_beta_estimates_attempted=True),dict(good,nuisance_beta_estimates_completed=4)):
            with self.assertRaises(ValueError):r.counters(bad)

    def test_source_package_review_precedes_parsing_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();out=root/r.SOURCE_OUT;out.mkdir(parents=True)
            (root/'stub.json').write_text('{}')
            for name in r.SOURCE_FILES:(out/name).write_text('INVALID_SOURCE_ROWS')
            manifest=dict(schema_version=1,source_only=True,strategy_registration=False,clock_policy_id=r.CLOCK,
                input_hashes={'stub.json':r.sha(root/'stub.json')},
                outputs={name:dict(sha256=r.sha(out/name),bytes=(out/name).stat().st_size) for name in r.SOURCE_FILES})
            (out/'MANIFEST.json').write_text(json.dumps(manifest))
            runner=object.__new__(r.Runner);runner.root=root
            runner.inputs=dict(source_manifest=out/'MANIFEST.json',source_grid_review=root/'review.json')
            closure={str(p):r.sha(p) for p in [root/'stub.json',out/'MANIFEST.json',*[out/name for name in r.SOURCE_FILES]]}
            review=dict(passed=True,research_eligibility_verified=True,actual_grid_verified=True,exact_acceptance_certified=False,
                        clock_policy_id=r.CLOCK,input_hashes=closure)
            runner.pins=dict(closure)
            for key,bad in (('passed',False),('research_eligibility_verified',False),('actual_grid_verified',False),('exact_acceptance_certified',True)):
                changed=dict(review,**{key:bad});runner.inputs['source_grid_review'].write_text(json.dumps(changed))
                with self.subTest(key=key),self.assertRaisesRegex(ValueError,'Independent reconstructed-source'):
                    runner.require_source_grid()

    def test_prepare_receipt_requires_exact_counts_and_nuisance_totals(self):
        runner=object.__new__(r.Runner);runner.out=Path('/private/tmp/synthetic_filing_receipt');runner.plan_sha='a'*64
        runner.plan={'independent_prepare_source_hashes':{'review.py':'b'*64}}
        totals=dict(nuisance_beta_estimates_attempted=90,nuisance_beta_estimates_completed=80,new_predictive_model_fits=0)
        audit=dict(common_coverage={'coverage_floor_passed':True},**totals)
        report=dict(passed=True,stage='prepare',producer_modules_imported=False,strategy_returns_or_nav_computed=False,
            orders_checked=488,candidate_cells_checked=5978,months_checked=122,prefixes_checked=123,filing_feature_matrices_verified=3,
            plan_sha256='a'*64,cache_sha256='a'*64,orders_sha256='a'*64,source_hashes={'review.py':'b'*64},
            artifact_hashes={str(runner.out/name):'a'*64 for name in ('PLAN.json','CACHE.json',*r.CACHE_FILES)},**totals)
        report.update(dict.fromkeys(r.CHECKS,True))
        def read(path):return audit if Path(path).name=='FEATURE_AUDIT.json' else report
        with patch.object(runner,'cache'),patch.object(runner,'verify',side_effect=lambda value:value),patch.object(r,'sha',return_value='a'*64),patch.object(r,'read',side_effect=read):
            runner.require_prepare_review()
            for key,bad in (('stage','synthetic'),('orders_checked',487),('candidate_cells_checked',5977),('prefixes_checked',122),
                            ('nuisance_beta_estimates_attempted',89),('nuisance_beta_estimates_completed',False),('new_predictive_model_fits',1)):
                old=report[key];report[key]=bad
                with self.subTest(key=key),self.assertRaises(ValueError):runner.require_prepare_review()
                report[key]=old

    def test_diagnostic_marker_binds_selection_before_backend_load(self):
        runner=object.__new__(r.Runner);runner.out=Path('/private/tmp/synthetic_filing_marker');runner.plan_sha='a'*64
        rows=[measured(name,.4+i*.01) for i,name in enumerate(r.NAMES)]
        frozen=dict(selection=r.select(rows,rows),hashes=dict.fromkeys(r.FROZEN_FILES,'a'*64))
        written={}
        def read(path):return frozen if Path(path).name=='FROZEN.json' else rows
        with patch.object(runner,'refresh'),patch.object(runner,'require_prepare_review'),patch.object(runner,'verify'), \
                patch.object(r,'read',side_effect=read),patch.object(r,'sha',return_value='f'*64), \
                patch.object(runner,'write',side_effect=lambda name,value:written.update({name:value})), \
                patch.object(runner,'loaded',side_effect=ValueError('stop_after_marker')):
            with self.assertRaisesRegex(ValueError,'stop_after_marker'):runner.diagnose_later34()
        self.assertEqual(written['DIAGNOSE_STARTED.json']['plan_sha256'],'a'*64)
        self.assertEqual(written['DIAGNOSE_STARTED.json']['frozen_sha256'],'f'*64)


if __name__=='__main__':unittest.main(verbosity=2)
