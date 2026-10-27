"""Saved272 verifier tests on synthetic arrays/receipts only."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
import ast
from pathlib import Path
from unittest.mock import patch
import numpy as np
import monthly_roa_increment272_saved_reader_v2 as r
import monthly_roa_increment272_runner as h
from test_monthly_roa_increment272_runner import prepared_fixture
from test_monthly_roa_increment272_factor import metrics,Poison


def rows():return {a:dict(same_close_zero={k:v for k,v in metrics(a,.6 if i==2 else .5).items() if k!='name'}) for i,a in enumerate(r.ARMS)}


class AdmissionTests(unittest.TestCase):
    def test_exact_independent_first88_selection(self):
        t=rows();self.assertEqual(r.selection(t),h.select_first88(t));self.assertTrue(r.selection(t)['admitted'])
        for field,limit in (('sharpe',.05),('max_drawdown',.03)):
            t=rows();base=t[r.ARMS[0]]['same_close_zero']['training'][field]
            t[r.ARMS[2]]['same_close_zero']['training'][field]=base-limit
            self.assertTrue(r.selection(t)['admitted'])
            t[r.ARMS[2]]['same_close_zero']['training'][field]-=1e-8
            self.assertFalse(r.selection(t)['admitted'])

    def test_strict_cagr_both_references_and_no_control_promotion(self):
        for role in r.ARMS[:2]:
            t=rows();t[role]['same_close_zero']['training']['cagr']=.6
            self.assertEqual(r.selection(t)['primary'],r.ARMS[0])
        t=rows();t[r.ARMS[2]]['same_close_zero']['training']['cagr']=.5+1e-12
        self.assertFalse(r.selection(t)['admitted'])

    def test_era_gates_and_missing_nan_bool(self):
        for segment in ('early','recent'):
            t=rows();base=t[r.ARMS[0]]['same_close_zero'][segment]['cagr'];t[r.ARMS[2]]['same_close_zero'][segment]['cagr']=base-.025
            self.assertTrue(r.selection(t)['admitted']);t[r.ARMS[2]]['same_close_zero'][segment]['cagr']-=1e-8;self.assertFalse(r.selection(t)['admitted'])
        for value in (None,float('nan'),float('inf'),True):
            t=rows();t[r.ARMS[2]]['same_close_zero']['training']['sharpe']=value;self.assertFalse(r.selection(t)['admitted'])

    def test_later_and_nextopen_never_consumed(self):
        t=rows();self.assertTrue(r.selection(t)['admitted'])  # existing rows contain Poison extras
        t[r.ARMS[0]]['next_open_zero']=Poison()
        with self.assertRaises(ValueError):r.selection(t)


class PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.fixture=prepared_fixture()

    def pieces(self):
        full,proof,source,args=self.fixture
        return deepcopy(full),deepcopy(proof),deepcopy(source),(None,deepcopy(args[1]),deepcopy(args[2]),None,deepcopy(args[4]),None)

    def check(self,full,proof,source,args):
        return r.preparation(proof,full[3],source,args[2],full[5],full[4],args[1]['eligible'],total=5,width=49,prefixes=(0,1,3,4,5),orders=full[0],rowwidth=sum(int(np.prod(v.shape[1:])) for v in args[1].values()))

    def test_exact_real_host_synthetic_fixture(self):
        full,proof,source,args=self.pieces();work=self.check(full,proof,source,args)
        self.assertEqual(work,proof['execution_accounting']);self.assertEqual(r.structured_digest(work),h.structured_digest(work))
        self.assertEqual(work['actual_prepare_entries'],11)

    def test_saved_math_uses_independent_full_values_once(self):
        full,p,_,args=self.pieces();fin=args[4]
        report=r.math_check(args[2],full[4],full[2],full[1],full[0],args[1],fin['catalog'],fin['records'],fin['sessions'],full[5],p['main'])
        self.assertEqual(report['saved_overlay_verifications'],1);self.assertEqual(report['saved_pre_score_verifications'],2);self.assertEqual(report['original_prefix_executions_repeated'],0)

    def test_saved_financial_value_mutation_rejected(self):
        full,p,_,args=self.pieces();fin=args[4];full[4]['candidate_hand'][0,1]+=.01
        with self.assertRaises(ValueError):r.math_check(args[2],full[4],full[2],full[1],full[0],args[1],fin['catalog'],fin['records'],fin['sessions'],full[5],p['main'])

    def test_saved_pre_score_mutation_rejected(self):
        full,p,_,args=self.pieces();fin=args[4];full[2][r.ARMS[2]][0,1]+=.01
        with self.assertRaises(ValueError):r.math_check(args[2],full[4],full[2],full[1],full[0],args[1],fin['catalog'],fin['records'],fin['sessions'],full[5],p['main'])

    def test_saved_order_and_display_scores_bound(self):
        for key in ('symbols','scores'):
            full,p,_,args=self.pieces();fin=args[4]
            full[0][r.ARMS[2]][0][key]=list(reversed(full[0][r.ARMS[2]][0][key]))
            with self.assertRaises(ValueError):r.math_check(args[2],full[4],full[2],full[1],full[0],args[1],fin['catalog'],fin['records'],fin['sessions'],full[5],p['main'])

    def test_balanced_cached_physical_incumbent_counter_forgery(self):
        for location,index in (('cached_prefixes',1),('physical_prefixes',1),('main',None)):
            full,p,s,args=self.pieces();e=p[location] if index is None else p[location][index]
            e['independent'][r.ARMS[0]]['selector']['work']['partial_events']=0
            p['execution_accounting']=h.execution_accounting(p)
            with self.assertRaises(ValueError):self.check(full,p,s,args)

    def test_balanced_overlay_counts_rejected(self):
        full,p,s,args=self.pieces();e=p['cached_prefixes'][1]
        for counts in (e['overlay_counts'],e['overlay_proof']['counts']):counts['ratio_evaluations']+=1
        p['execution_accounting']=h.execution_accounting(p)
        with self.assertRaises(ValueError):self.check(full,p,s,args)

    def test_balanced_pre_score_count_rejected(self):
        full,p,s,args=self.pieces();p['main']['independent'][r.ARMS[1]]['pre_score']['work']['eligible_delta_cells']+=1;p['execution_accounting']=h.execution_accounting(p)
        with self.assertRaises(ValueError):self.check(full,p,s,args)

    def test_alias_claim_source_flag_and_omitted_prefix_rejected(self):
        for mutation in ('alias','source','prefix'):
            full,p,s,args=self.pieces()
            if mutation=='alias':p['physical_prefixes'][-1]['reuses_main_verified_full']=False
            elif mutation=='source':p['main']['source_audit']['both_reference_gates_before_overlay']=False
            else:p['cached_prefixes'].pop(2)
            with self.assertRaises(ValueError):self.check(full,p,s,args)

    def test_zero_prefix_actual_empty_counts(self):
        full,p,s,args=self.pieces();e=p['cached_prefixes'][0]
        r.entry(e,0,49,args[2],full[3],full[5],full[4],args[1]['eligible'],cached=True,orders=full[0])
        self.assertEqual(e['overlay_counts']['catalog_headers_checked'],0);self.assertEqual(e['overlay_counts']['ratio_evaluations'],0)

    def test_scope_arrays_signedzero_nan_payload(self):
        a=np.array([-0.,np.nan]);r.array_exact(a,a.copy(),'copy')
        for position in (0,1):
            b=a.copy();b.view(np.uint64)[position]^=1
            with self.assertRaises(ValueError):r.array_exact(a,b,'bits')


class EvidenceTests(unittest.TestCase):
    def test_terminals_require_true_zero_and_bound_artifact(self):
        hashes=dict(prepare='a'*64,select='b'*64,diagnose='c'*64)
        record=dict(plan_sha256='p',phases={name:dict(session_id=1,exit_code=0,chunk_id='abc',artifact_path=str(r.OUT/path),artifact_sha256=hashes[name],owner='/root') for name,path in [('prepare','prepared/COMPLETION.json'),('select','selected/FROZEN.json'),('diagnose','results/COMPLETION.json')]})
        r.terminals(record,'p',hashes)
        for bad in (True,1):
            d=deepcopy(record);d['phases']['prepare']['exit_code']=bad
            with self.assertRaises(ValueError):r.terminals(d,'p',hashes)
        d=deepcopy(record);d['phases']['prepare']['owner']='elsewhere'
        with self.assertRaises(ValueError):r.terminals(d,'p',hashes)

    def test_accounts_explicit36_paths(self):
        calls=[];proxy=r.Accounts(SimpleNamespace(read=lambda p:calls.append(p)))
        self.assertEqual(len(proxy.paths),36)
        for p,target in proxy.paths.items():proxy.read(p);self.assertTrue(str(target).startswith(str(r.OUT/'results')))
        self.assertEqual(len(calls),36)
        with self.assertRaises(ValueError):proxy.read(r.OUT/'PLAN.json')

    def test_diagnostic_open_groups_added_only_for_accounting(self):
        t=rows();result=dict(arms={a:dict(next_open_zero=dict(slices={k:dict(cagr=.4) for k in ('first88','early53','recent35')})) for a in r.ARMS})
        ind=dict(paths={a:{} for a in r.ARMS});before=deepcopy(t)
        def check(proxy,res,proof,training):
            self.assertEqual(set(training),set(r.old.ARMS))
            self.assertTrue(all(set(v)=={'same_close_zero','next_open_zero'} for v in training.values()))
            return dict(checked_training_groups=18),{}
        with patch.object(r.old,'accounting',side_effect=check):report,_=r.accounting(None,result,ind,t)
        self.assertEqual(report['checked_training_groups'],18);self.assertEqual(r.selection(t),r.selection(before))
        self.assertTrue(all(set(v)=={'same_close_zero'} for v in t.values()))


class RuntimeBoundaryTests(unittest.TestCase):
    def plan(self):return {'input_sha256':{str(r.EXTERNAL_RUNTIME):r.EXTERNAL_RUNTIME_SHA256,'work/example.json':'a'*64}}

    def test_exact_runtime_registered_pin_finish(self):
        e=r.Evidence();p=self.plan()
        with patch.object(r,'digest',return_value=r.EXTERNAL_RUNTIME_SHA256) as h:
            self.assertEqual(e.registered(p,r.EXTERNAL_RUNTIME),r.EXTERNAL_RUNTIME_SHA256)
            self.assertEqual(e.pin(r.EXTERNAL_RUNTIME,r.EXTERNAL_RUNTIME_SHA256),r.EXTERNAL_RUNTIME_SHA256)
            self.assertEqual(e.pins,{str(r.EXTERNAL_RUNTIME):r.EXTERNAL_RUNTIME_SHA256})
            e.finish();self.assertEqual(h.call_count,2)

    def test_wrong_digest_missing_runtime_and_bad_sha_rejected(self):
        for bad in ('b'*64,None,True,0):
            p=self.plan();p['input_sha256'][str(r.EXTERNAL_RUNTIME)]=bad
            with self.assertRaises(ValueError):r.Evidence().registered(p,r.EXTERNAL_RUNTIME)
        p=self.plan();del p['input_sha256'][str(r.EXTERNAL_RUNTIME)]
        with self.assertRaises(ValueError):r.Evidence().registered(p,r.EXTERNAL_RUNTIME)
        with patch.object(r,'digest',return_value='b'*64):
            with self.assertRaises(ValueError):r.Evidence().pin(r.EXTERNAL_RUNTIME,r.EXTERNAL_RUNTIME_SHA256)
        with self.assertRaises(ValueError):r.Evidence().pin(r.EXTERNAL_RUNTIME,'b'*64)

    def test_other_external_cloud_and_runtime_aliases_rejected(self):
        denied=('/tmp/not-authorized-python','/usr/bin/python3',str(r.EXTERNAL_RUNTIME.parent/'python3'),'/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/./python3.9','/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/../bin/python3.9',str(r.ROOT/'CloudStorage/example.json'))
        for path in denied:
            with self.subTest(path=path):
                with patch.object(r,'digest') as h:
                    with self.assertRaises(ValueError):r.Evidence().pin(path,r.EXTERNAL_RUNTIME_SHA256)
                    with self.assertRaises(ValueError):r.Evidence().registered({'input_sha256':{path:r.EXTERNAL_RUNTIME_SHA256}},path)
                    h.assert_not_called()

    def test_extra_external_registration_rejected_on_internal_lookup(self):
        p=self.plan();p['input_sha256']['/tmp/other-file']='c'*64
        with self.assertRaises(ValueError):r.Evidence().registered(p,'work/example.json')

    def test_runtime_identity_and_root_local_alias_rejected(self):
        original=Path.resolve
        def changed(path,*a,**kw):
            if path==r.EXTERNAL_RUNTIME:return Path('/tmp/replacement-python')
            return original(path,*a,**kw)
        with patch.object(Path,'resolve',changed):
            with self.assertRaises(ValueError):r.Evidence().pin(r.EXTERNAL_RUNTIME,r.EXTERNAL_RUNTIME_SHA256)
        def alias(path,*a,**kw):
            if path==r.ROOT/'work/runtime-alias':return r.EXTERNAL_RUNTIME
            return original(path,*a,**kw)
        with patch.object(Path,'resolve',alias):
            with self.assertRaises(ValueError):r.Evidence().pin('work/runtime-alias',r.EXTERNAL_RUNTIME_SHA256)
        with patch.object(Path,'is_symlink',return_value=True):
            with self.assertRaises(ValueError):r.Evidence().registered(self.plan(),r.EXTERNAL_RUNTIME)

    def test_runtime_bytes_changed_at_finish_rejected(self):
        e=r.Evidence()
        with patch.object(r,'digest',side_effect=(r.EXTERNAL_RUNTIME_SHA256,'b'*64)):
            e.pin(r.EXTERNAL_RUNTIME,r.EXTERNAL_RUNTIME_SHA256)
            with self.assertRaises(ValueError):e.finish()

    def test_entire_registered_map_mutation_rejected(self):
        for change in ('unused','runtime','delete','extra'):
            p=self.plan();e=r.Evidence();e.registered(p,r.EXTERNAL_RUNTIME)
            if change=='unused':p['input_sha256']['work/example.json']='b'*64
            elif change=='runtime':p['input_sha256'][str(r.EXTERNAL_RUNTIME)]='b'*64
            elif change=='delete':del p['input_sha256']['work/example.json']
            else:p['input_sha256']['work/new.json']='c'*64
            with self.assertRaises(ValueError):e.finish()
            with self.assertRaises(ValueError):e.registered(p,r.EXTERNAL_RUNTIME)

    def test_conflicting_local_alias_and_mutable_central_rejected(self):
        p=self.plan();p['input_sha256'][str(r.ROOT/'work/example.json')]='b'*64
        with self.assertRaises(ValueError):r.Evidence().registered(p,'work/example.json')
        with self.assertRaises(ValueError):r.Evidence().pin(r.ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json')

    def test_local_pin_still_checked_and_rehashed(self):
        p=r.ROOT/'work/example.json';e=r.Evidence()
        with patch.object(r.old,'digest',return_value='a'*64):self.assertEqual(e.pin(p,'a'*64),'a'*64)
        self.assertEqual(e.pins,{'work/example.json':'a'*64})
        with patch.object(r,'digest',return_value='b'*64):
            with self.assertRaises(ValueError):e.finish()
        with patch.object(r.old,'digest',return_value='b'*64):
            with self.assertRaises(ValueError):e.pin(p,'a'*64)

    def test_runtime_binary_is_not_a_json_read_exception(self):
        with self.assertRaises(ValueError):r.Evidence().read(r.EXTERNAL_RUNTIME,r.EXTERNAL_RUNTIME_SHA256)

    def test_additive_pins_and_honest_scope(self):
        calls=[];r.pin_additive_evidence(SimpleNamespace(pin=lambda *args:calls.append(args)))
        self.assertEqual(calls,[(r.ROOT/p,) for p in r.ADDITIVE_FILES]+[(r.ROOT/r.FAILED_ATTEMPT,r.FAILED_ATTEMPT_SHA256)])
        scope=r.additive_review_scope()
        self.assertFalse(scope['additive_files_in_original_plan']);self.assertFalse(scope['financial_or_accounting_rules_changed'])
        self.assertFalse(scope['external_runtime']['other_external_paths_allowed'])
        self.assertEqual(r.SUPPLEMENTAL,('work/monthly_roa_increment272_saved_reader.py','work/test_monthly_roa_increment272_saved_reader.py'))

    def test_original_economic_and_run_logic_ast_unchanged(self):
        original=ast.parse((r.ROOT/'work/monthly_roa_increment272_saved_reader.py').read_text())
        current=ast.parse(Path(r.__file__).read_text())
        oldfunctions={n.name:n for n in original.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        newfunctions={n.name:n for n in current.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        for name,node in oldfunctions.items():
            actual=deepcopy(newfunctions[name])
            if name=='run':
                actual.body=[s for s in actual.body if not (isinstance(s,ast.Expr) and isinstance(s.value,ast.Call) and isinstance(s.value.func,ast.Name) and s.value.func.id=='pin_additive_evidence')]
                output=actual.body[-1].value
                output.keywords=[k for k in output.keywords if k.arg!='additive_validation']
                next(k for k in output.keywords if k.arg=='schema').value.value='roa_increment272_saved_reader_v1'
            self.assertEqual(ast.dump(node,include_attributes=False),ast.dump(actual,include_attributes=False),name)


if __name__=='__main__':unittest.main()
