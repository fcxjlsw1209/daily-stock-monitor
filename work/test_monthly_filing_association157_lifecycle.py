"""Fabricated phase, concurrency, selection and source-binding regressions."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import patch
import monthly_filing_association157_lifecycle as c
from test_monthly_filing_association157_runner import source_fixture, r


def state():
    return dict(completed=[dict(directory='old'+str(i),trials=6) for i in range(155)]+[
        dict(directory='monthly_revenue_persistence156',trials=52,completion=dict(sha256=c.PREVIOUS_DONE_SHA))],
        registered_queue=[], unrelated={'preserve':[1,2]})


def rows():
    return [dict(name=n,training=dict(cagr=.5,sharpe=1.5,max_drawdown=-.3),
                 early=dict(cagr=.3),recent=dict(cagr=.6)) for n in c.NAMES]


class Selection(unittest.TestCase):
    def test_equal_keeps_incumbent(self):
        self.assertEqual(c.selection(rows(),rows())['primary'],c.NAMES[0])

    def test_primary_must_beat_every_control(self):
        a,b=rows(),rows();a[3]['training']['cagr']=b[3]['training']['cagr']=.6
        self.assertTrue(c.selection(a,b)['admitted'])
        for i in range(3):
            x=deepcopy(a);x[i]['training']['cagr']=.6
            self.assertFalse(c.selection(x,b)['admitted'])

    def test_only_open_incumbent_allows_tie(self):
        a,b=rows(),rows();a[3]['training']['cagr']=.6
        b[1]['training']['cagr']=b[2]['training']['cagr']=.4
        self.assertTrue(c.selection(a,b)['admitted'])
        b[1]['training']['cagr']=.5
        self.assertFalse(c.selection(a,b)['admitted'])

    def test_all_risk_and_subperiod_guards(self):
        for section,key,v in [('training','sharpe',1.44),('training','max_drawdown',-.331),
                              ('early','cagr',.274),('recent','cagr',.574)]:
            a,b=rows(),rows();a[3]['training']['cagr']=b[3]['training']['cagr']=.6
            a[3][section][key]=v
            self.assertFalse(c.selection(a,b)['admitted'])

    def test_nonfinite_boolean_and_arm_order(self):
        for v in [float('nan'),float('inf'),True,None]:
            a=rows();a[3]['training']['cagr']=v
            self.assertFalse(c.selection(a,rows())['admitted'])
        with self.assertRaises(ValueError):c.selection(rows()[::-1],rows())


class Files(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='synthetic-filing157-',dir='/private/tmp')
        self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.out=self.root/'outputs'/c.STUDY;self.out.mkdir(parents=True)
        self.put(c.STATE,state())

    def put(self,name,value):
        p=Path(name);p=p if p.is_absolute() else self.root/p
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes(value if isinstance(value,bytes) else c.old.encoded(value));return p

    def plan(self):
        pin=self.put('work/pin.py',b'synthetic source')
        return dict(round=157,status='REGISTERED',arms=list(c.NAMES),primary_challenger=c.NAMES[3],dose=.01,
            output_directory=str(self.out),registered_utc='2026-10-20T21:00:00+00:00',
            registration_precondition=dict(path=c.STATE,sha256=c.sha(self.root/c.STATE)),hashes={str(pin):c.sha(pin)})

    def test_exact_predecessor(self):
        c.predecessor(state())
        for action in [lambda x:x['completed'].pop(), lambda x:x['completed'][0].update(trials=True),
                       lambda x:x['completed'][0].update(trials=7),
                       lambda x:x['completed'][-1]['completion'].update(sha256='a'*64)]:
            d=state();action(d)
            with self.assertRaises(ValueError):c.predecessor(d)

    def test_exact_backup_and_unrelated_fields(self):
        path=self.root/c.STATE;raw=path.read_bytes();before=c.sha(path);v=state();v['current']='new'
        c.update_state(path,before,v)
        self.assertEqual((path.parent/('PRE157_'+before+'.json')).read_bytes(),raw)
        self.assertEqual(c.old.decode(path.read_bytes()),v)

    def test_state_snapshot_preserves_historical_paths_and_hashes_exact_bytes(self):
        # These are inert strings in a synthetic local file, never file reads.
        original='/old/CloudStorage/OneDrive/example/research'
        value=dict(path=original+'/outputs/x.json',nested={original+'/index':['unchanged',original]})
        p=self.put('state-paths.json',value);raw=p.read_bytes()
        with patch.object(c.json,'loads',side_effect=AssertionError('Relocating decoder must not run')):
            got,digest=c.state_snapshot(p)
        self.assertEqual(got,value);self.assertEqual(digest,c.sha(p));self.assertEqual(p.read_bytes(),raw)
        got['new_status']='updated';c.update_state(p,digest,got)
        restored=json.JSONDecoder().decode(p.read_text())
        self.assertEqual(restored['path'],value['path']);self.assertEqual(restored['nested'],value['nested'])

    def test_state_snapshot_rejects_duplicate_and_nonfinite_tokens(self):
        for raw in (b'{"x":1,"x":2}',b'{"x":NaN}',b'{"x":Infinity}'):
            p=self.put('bad-state.json',raw)
            with self.subTest(raw=raw),self.assertRaises(ValueError):c.state_snapshot(p)

    def test_concurrent_state_preserved(self):
        p=self.root/c.STATE;h=c.sha(p);self.put(p,{'concurrent':True});raw=p.read_bytes()
        with self.assertRaises(ValueError):c.update_state(p,h,state())
        self.assertEqual(p.read_bytes(),raw)

    def test_conflicting_backup(self):
        p=self.root/c.STATE;h=c.sha(p);self.put(p.parent/('PRE157_'+h+'.json'),b'wrong')
        with self.assertRaises(ValueError):c.update_state(p,h,state())

    def test_exclusive_and_failed_serialization(self):
        p=self.put('saved.json',{'original':True});raw=p.read_bytes()
        with self.assertRaises(FileExistsError):c.write(p,{'new':True})
        self.assertEqual(p.read_bytes(),raw)
        bad=self.root/'bad.json'
        with self.assertRaises(ValueError):c.write(bad,{'x':float('nan')})
        self.assertFalse(bad.exists());self.assertEqual(list(self.root.glob('.filing157-*')),[])

    def test_enqueue_idempotent(self):
        p=self.put(self.out/'PLAN.json',self.plan());h=c.sha(p)
        self.assertFalse(c.enqueue(self.root,h)['already_enqueued']);raw=(self.root/c.STATE).read_bytes()
        self.assertTrue(c.enqueue(self.root,h)['already_enqueued']);self.assertEqual((self.root/c.STATE).read_bytes(),raw)

    def test_changed_pin_and_wrong_hash(self):
        p=self.put(self.out/'PLAN.json',self.plan());h=c.sha(p)
        with self.assertRaises(ValueError):c.enqueue(self.root,'a'*64)
        self.put('work/pin.py',b'changed')
        with self.assertRaises(ValueError):c.enqueue(self.root,h)

    def test_existing_preparation_blocks_registration(self):
        p=self.put(self.out/'DATA.npz',b'already');raw=p.read_bytes()
        with patch.object(c,'runner',return_value=SimpleNamespace(CACHE_FILES=['DATA.npz'])):
            with self.assertRaisesRegex(ValueError,'Registration precedes'):c.build_plan(self.root,'unused','a'*64)
        self.assertEqual(p.read_bytes(),raw)

    def test_conflicting_queue(self):
        p=self.put(self.out/'PLAN.json',self.plan());h=c.sha(p);v=state();v['registered_queue']=[{'other':True}]
        self.put(c.STATE,v);raw=(self.root/c.STATE).read_bytes()
        with self.assertRaises(ValueError):c.enqueue(self.root,h)
        self.assertEqual((self.root/c.STATE).read_bytes(),raw)

    def source_fixture(self):
        raw=self.put('work/raw.json',{'synthetic':True})
        meta,source_rows,sessions=source_fixture()
        self.meta_path='outputs/monthly_local_linear70/META.json'
        self.put(self.meta_path,meta)
        self.put(c.DESIGN,{'design':'synthetic'});self.put(c.SOURCE_CALENDAR,{'sessions':sessions})
        inputs={str(self.root/name):c.sha(self.root/name) for name in
                ('work/raw.json',self.meta_path,c.DESIGN,c.SOURCE_CALENDAR)}
        outputs={};hashes=dict(inputs)
        for name,value in {'ROWS.json':source_rows, 'COVERAGE_GATE.json':r.coverage_gate(meta,source_rows)}.items():
            p=self.put(c.SOURCE_ROOT+'/'+name,value)
            outputs[name]=dict(sha256=c.sha(p),bytes=p.stat().st_size);hashes[str(p)]=c.sha(p)
        p=self.put(c.SOURCE_MANIFEST,dict(schema_version=1,source_only=True,strategy_registration=False,
            exact_acceptance_certified=False,
            clock_policy_id='second_XNYS_session_strictly_after_filingDate_v1',outputs=outputs,
            input_hashes=inputs));hashes[str(p)]=c.sha(p)
        p=self.put(c.SOURCE_REVIEW,dict(passed=True,actual_grid_verified=True,research_eligibility_verified=True,
            exact_acceptance_certified=False,input_hashes=hashes,candidate_cells=5978,stock_cells=3904,etf_cells=2074,
            checks=dict.fromkeys(c.SOURCE_CHECKS,True)))
        return dict(path=c.SOURCE_REVIEW,sha256=c.sha(p))

    def run_gate(self,ref):
        with patch.object(c,'DESIGN_SHA',c.sha(self.root/c.DESIGN)), \
             patch.object(c,'SOURCE_CALENDAR_SHA',c.sha(self.root/c.SOURCE_CALENDAR)), \
             patch.dict(r.FIXED_INPUTS,meta=(self.meta_path,c.sha(self.root/self.meta_path))):
            return c.source_gate(c.old.Reader(self.root),ref)

    def rebind(self,ref,output=None):
        """Rehash an intentional forged package so only semantic guards reject it."""
        manifest_path=self.root/c.SOURCE_MANIFEST
        manifest=c.old.decode(manifest_path.read_bytes())
        review_path=self.root/c.SOURCE_REVIEW;review=c.old.decode(review_path.read_bytes())
        if output is not None:
            path=self.root/c.SOURCE_ROOT/output
            manifest['outputs'][output]=dict(sha256=c.sha(path),bytes=path.stat().st_size)
            review['input_hashes'][str(path)]=c.sha(path)
        self.put(manifest_path,manifest)
        review['input_hashes'][str(manifest_path)]=c.sha(manifest_path)
        self.put(review_path,review);ref['sha256']=c.sha(review_path)

    def test_complete_conditional_source_gate(self):
        self.run_gate(self.source_fixture())

    def test_each_required_check_and_grid_count(self):
        ref=self.source_fixture();path=self.root/c.SOURCE_REVIEW;original=c.old.decode(path.read_bytes())
        for key in c.SOURCE_CHECKS:
            changed=deepcopy(original);changed['checks'].pop(key);self.put(path,changed);ref['sha256']=c.sha(path)
            with self.subTest(check=key),self.assertRaisesRegex(ValueError,'full-grid source checks'):self.run_gate(ref)
        for key in ('candidate_cells','stock_cells','etf_cells'):
            changed=deepcopy(original);changed[key]-=1;self.put(path,changed);ref['sha256']=c.sha(path)
            with self.subTest(count=key),self.assertRaisesRegex(ValueError,'Wrong exact count'):self.run_gate(ref)

    def test_empty_reordered_or_etf_promoted_rows_rejected_after_rehash(self):
        for mutation in ('empty','reordered','ETF'):
            ref=self.source_fixture();path=self.root/c.SOURCE_ROOT/'ROWS.json';source_rows=c.old.decode(path.read_bytes())
            if mutation=='empty':source_rows=[]
            elif mutation=='reordered':source_rows[0],source_rows[1]=source_rows[1],source_rows[0]
            else:source_rows[32].update(research_eligible=True,reasons=[],issuer_id='wrong')
            self.put(path,source_rows);self.rebind(ref,'ROWS.json')
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.run_gate(ref)

    def test_lying_coverage_rejected_after_rehash(self):
        ref=self.source_fixture();path=self.root/c.SOURCE_ROOT/'COVERAGE_GATE.json'
        gate=c.old.decode(path.read_bytes());gate['eras']['first88']['usable_slots']+=1
        self.put(path,gate);self.rebind(ref,'COVERAGE_GATE.json')
        with self.assertRaisesRegex(ValueError,'complete original metadata grid'):self.run_gate(ref)

    def test_each_missing_contract_closure_rejected(self):
        for name in (c.DESIGN,c.SOURCE_CALENDAR,'outputs/monthly_local_linear70/META.json'):
            ref=self.source_fixture();path=self.root/c.SOURCE_MANIFEST;manifest=c.old.decode(path.read_bytes())
            manifest['input_hashes'].pop(str(self.root/name));self.put(path,manifest);self.rebind(ref)
            with self.subTest(contract=name),self.assertRaisesRegex(ValueError,'contract absent'):self.run_gate(ref)

    def test_manifest_cannot_upgrade_exact_acceptance(self):
        ref=self.source_fixture();path=self.root/c.SOURCE_MANIFEST;manifest=c.old.decode(path.read_bytes())
        manifest['exact_acceptance_certified']=True;self.put(path,manifest);self.rebind(ref)
        with self.assertRaisesRegex(ValueError,'Exact source package'):self.run_gate(ref)

    def test_false_or_upgraded_review_rejected(self):
        for key,value in [('passed',False),('actual_grid_verified',False),
                          ('research_eligibility_verified',False),('exact_acceptance_certified',True)]:
            ref=self.source_fixture();p=self.root/c.SOURCE_REVIEW;v=c.old.decode(p.read_bytes());v[key]=value
            self.put(p,v);ref['sha256']=c.sha(p)
            with self.assertRaises(ValueError):self.run_gate(ref)

    def test_changed_source_bytes_rejected(self):
        ref=self.source_fixture();self.put(c.SOURCE_ROOT+'/ROWS.json',[{'changed':True}])
        with self.assertRaises(ValueError):self.run_gate(ref)

    def test_omitted_source_pin_rejected(self):
        ref=self.source_fixture();p=self.root/c.SOURCE_REVIEW;v=c.old.decode(p.read_bytes())
        del v['input_hashes'][str(self.root/'work/raw.json')];self.put(p,v);ref['sha256']=c.sha(p)
        with self.assertRaises(ValueError):self.run_gate(ref)

    def test_prereview_exact_independent_source_set(self):
        implementation={}
        for name in ('after156_association_measure.py','monthly_filing_association157_runner.py',
                     'monthly_filing_association157_lifecycle.py','prepare_filing_association157_proposal.py'):
            path=self.put('work/'+name,b'synthetic implementation');implementation[str(path)]=c.sha(path)
        inputs={}
        for name in (c.PROPOSAL,c.DESIGN,c.SOURCE_MANIFEST,c.SOURCE_REVIEW):
            path=self.put(name,{'synthetic':name});inputs[str(path)]=c.sha(path)
        a=self.put('work/review.py',b'independent');b=self.put('work/test_review.py',b'test')
        required={str(path):c.sha(path) for path in (a,b)}
        fake=SimpleNamespace(source_files=lambda root:[a,b])
        for stage in ('prepare','outcome'):
            doc=dict(passed=True,round=157,stage='preregistration_'+stage+'_review',synthetic_only=True,
                proposal_sha256='a'*64,implementation_hashes=implementation,input_hashes=inputs,
                actual_factors_calculated=0,actual_returns_calculated=0,actual_orders_calculated=0,
                actual_predictive_model_fits=0,actual_nuisance_estimates=0,source_hashes=required)
            for variant in ('exact','missing','extra'):
                modified=deepcopy(doc)
                if variant=='missing':modified['source_hashes'].pop(str(b))
                if variant=='extra':modified['source_hashes'][str(self.root/c.PROPOSAL)]=inputs[str(self.root/c.PROPOSAL)]
                self.put(c.PREREVIEWS[stage],modified)
                with patch.object(c.importlib,'import_module',return_value=fake):
                    if variant=='exact':
                        self.assertEqual(c.prereview(c.old.Reader(self.root),stage,'a'*64),required)
                    else:
                        with self.subTest(stage=stage,variant=variant),self.assertRaisesRegex(ValueError,'complete actual independent implementation'):
                            c.prereview(c.old.Reader(self.root),stage,'a'*64)


class CompleteInterfaceTests(unittest.TestCase):
    """In-memory completion prefix only; never publish a completion or state."""
    def fixture(self):
        root=Path('/private/tmp/fabricated-filing157-completion');out=root/'outputs'/c.STUDY
        h='a'*64;source={'/synthetic/reviewer.py':h}
        plan=dict(hashes={str(root/c.SELF):h},no_observer_writes=True,exact_acceptance_certified=False,
            inputs=dict(source_grid_review={}),independent_prepare_source_hashes=source,
            independent_outcome_source_hashes=source)
        for key in ('actual_feature_calculations_before_registration','actual_returns_before_registration',
                    'actual_model_fits_before_registration','actual_nuisance_estimates_before_registration','new_market_downloads'):
            plan[key]=0
        shared=dict(passed=True,plan_sha256=h,cache_sha256=h,orders_sha256=h,producer_modules_imported=False,
            source_hashes=source,artifact_hashes={'/deliberately/not/a/prepared/artifact':h},
            nuisance_beta_estimates_attempted=25,nuisance_beta_estimates_completed=24,new_predictive_model_fits=0,
            filing_feature_matrices_verified=3)
        prep=dict(shared,stage='prepare',strategy_returns_or_nav_computed=False,
            orders_checked=488,candidate_cells_checked=5978,months_checked=122,prefixes_checked=123,synthetic_prepare_check=True)
        outcome=dict(shared,stage='evaluate',execution_paths_verified=12,total_monthly_orders_verified=488,
            **dict.fromkeys(c.OUTCOME_CHECKS,True))
        counts=dict(nuisance_beta_estimates_attempted=25,nuisance_beta_estimates_completed=24,new_predictive_model_fits=0)
        documents={str(out/'PLAN.json'):plan,str(out/'CACHE.json'):{'DATA.npz':h},
            str(out/'INDEPENDENT_PREPARE.json'):prep,str(out/'INDEPENDENT_OUTCOME.json'):outcome,
            str(out/'FEATURE_AUDIT.json'):dict(counts),str(out/'VALIDATION.json'):dict(counts)}

        class Reader:
            def hash(self,path):return h
            def read(self,path):return documents[str(path)]
            def verify(self,mapping,base=None):
                return {str(Path(base)/key) if base else key:value for key,value in mapping.items()}
        fake=SimpleNamespace(CACHE_FILES=('DATA.npz',),CHECKS=('synthetic_prepare_check',),
            Runner=lambda *args:SimpleNamespace(require_prepare_review=lambda:None))
        return root,h,documents,Reader(),fake

    def run_prefix(self,edit=None,expected='Unaudited preparation artifact'):
        root,h,docs,reader,fake=self.fixture()
        if edit is not None:edit(docs,root/'outputs'/c.STUDY)
        with patch.object(c,'runner',return_value=fake),patch.object(c.old,'Reader',return_value=reader), \
             patch.object(c,'sha',return_value=h),patch.object(c,'source_gate',return_value=None), \
             patch.object(c,'write',side_effect=AssertionError('Must not publish')), \
             patch.object(c,'update_state',side_effect=AssertionError('Must not change state')):
            with self.assertRaisesRegex(ValueError,expected):c.complete(root,h)

    def test_valid_interface_reaches_artifact_closure_without_publication(self):
        self.run_prefix()

    def test_prepare_scope_and_predictive_counts_are_exact_integers(self):
        for key,bad in [('orders_checked',487),('candidate_cells_checked',5977),('months_checked',121),
                        ('prefixes_checked',122),('filing_feature_matrices_verified',2),('new_predictive_model_fits',True)]:
            with self.subTest(field=key):
                self.run_prefix(lambda d,o:d[str(o/'INDEPENDENT_PREPARE.json')].update({key:bad}), 'Wrong exact count')

    def test_outcome_scope_and_no_producer_import(self):
        for key,bad in [('execution_paths_verified',11),('total_monthly_orders_verified',487),('filing_feature_matrices_verified',2)]:
            with self.subTest(field=key):
                self.run_prefix(lambda d,o:d[str(o/'INDEPENDENT_OUTCOME.json')].update({key:bad}), 'Wrong exact count')
        self.run_prefix(lambda d,o:d[str(o/'INDEPENDENT_OUTCOME.json')].update(producer_modules_imported=True),
                        'Independent outcome audit incomplete')

    def test_nuisance_mismatch_zero_and_unqualified_zero_fits_fail(self):
        self.run_prefix(lambda d,o:d[str(o/'VALIDATION.json')].update(nuisance_beta_estimates_completed=23),'Wrong exact count')
        self.run_prefix(lambda d,o:d[str(o/'FEATURE_AUDIT.json')].update(nuisance_beta_estimates_attempted=0),'Actual nuisance estimate count')
        self.run_prefix(lambda d,o:d[str(o/'FEATURE_AUDIT.json')].update(new_model_fits=0),'Unqualified zero fit count')
        self.run_prefix(lambda d,o:d[str(o/'INDEPENDENT_PREPARE.json')].update(strategy_returns_or_nav_computed=True),
                        'Preparation must precede performance')

    def test_preregistration_nuisance_and_exact_clock_guards_fail(self):
        self.run_prefix(lambda d,o:d[str(o/'PLAN.json')].update(actual_nuisance_estimates_before_registration=1),'Wrong exact count')
        self.run_prefix(lambda d,o:d[str(o/'PLAN.json')].update(exact_acceptance_certified=True),'Observer and source interpretation')


if __name__=='__main__':unittest.main(verbosity=2)
