"""Invented bytes and reports only; no actual financial source or census is read.

Fixture constants replace the two document byte identities and classification
identity in memory, solely to authenticate invented documents. No production
function or financial gate is patched to make the integration succeed.
"""
import copy
from contextlib import ExitStack, contextmanager
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import roa_field_scope270_host_v4 as h
import prepare_roa_field_scope270_policy_v4 as builder
import test_roa_field_scope270_host_v3 as prior


def template():
    return dict(policy=dict(binding_document=dict(accounting_review=copy.deepcopy(h.EXAMPLES[0][1]),
        producers={kind:dict(independent_reviews=[copy.deepcopy(h.EXAMPLES[i][1])])
                   for i,kind in ((1,'html'),(2,'pdf'))})))


def classification(union, binding, basis):
    return dict(schema='roa_field_scope270_schema_examples_v4',source_only=True,execution_authorized=False,
        financial_admission=False,declaring_document=union,examples=h.example_rows(),actual_binding=binding,basis=basis)


@contextmanager
def package(root):
    p,_,docs,ev=prior.package(root)
    refs=copy.deepcopy(p['inputs']);docs=copy.deepcopy(docs);pins=dict(ev.pins)
    design=copy.deepcopy(ev.read(p['design']))
    def put(path,obj):return prior.legacy.put(root,path,obj,pins)
    def save(role,obj):
        docs[role]=obj;refs[role]=put(refs[role]['path'],obj)
    producer_plan=put('metadata/PLAN.json',dict(synthetic=True))
    producer_manifest=put('metadata/MANIFEST.json',dict(synthetic=True))
    binding=dict(schema_version=1,stage='reviewed_source_union_inputs',source_union_authorized=True,
        admission_authorized=False,accounting_review=refs['accounting'],
        producers={kind:dict(directory='metadata',plan_sha256=producer_plan['sha256'],
            manifest_sha256=producer_manifest['sha256'],independent_reviews=[refs[role]])
            for kind,role in (('html','html_repair_review'),('pdf','pdf_review'))})
    bref=put('metadata/ACTUAL_BINDING.json',binding)
    up=copy.deepcopy(docs['union_plan']);up.update(template())
    up.update(binding_path=bref['path'],binding_sha256=bref['sha256'])
    active=[bref,producer_plan,producer_manifest,refs['accounting'],refs['html_repair_review'],refs['pdf_review']]
    up['source_hashes'].update({r['path']:r['sha256'] for r in active})
    save('union_plan',up)
    docs['union_data']['plan_sha256']=refs['union_plan']['sha256'];save('union_data',docs['union_data'])
    docs['union_manifest'].update(plan_sha256=refs['union_plan']['sha256'],output_hashes={'RAW_FACTS.json':refs['union_data']['sha256']})
    save('union_manifest',docs['union_manifest'])
    docs['union_review'].update(data_sha256=refs['union_data']['sha256'],plan_sha256=refs['union_plan']['sha256'],
        source_manifest_sha256=refs['union_manifest']['sha256'],
        input_hashes={refs[k]['path']:refs[k]['sha256'] for k in ('union_data','union_plan','union_manifest')})
    save('union_review',docs['union_review'])
    for key in ('union_data','union_plan','union_manifest'):docs['legacy_policy']['inputs'][key]=refs[key]
    docs['legacy_policy']['inputs']['union_independent']=refs['union_review'];save('legacy_policy',docs['legacy_policy'])
    docs['dataset']['policy_sha256']=refs['legacy_policy']['sha256'];save('dataset',docs['dataset'])
    docs['legacy_admission'].update(policy_sha256=refs['legacy_policy']['sha256'],data_sha256=refs['dataset']['sha256'],
        source_hashes={r['path']:r['sha256'] for r in docs['legacy_policy']['inputs'].values()})
    save('legacy_admission',docs['legacy_admission'])
    docs['legacy_review'].update(policy_sha256=refs['legacy_policy']['sha256'],data_sha256=refs['dataset']['sha256'])
    save('legacy_review',docs['legacy_review'])
    docs['three_review']['original_dataset']=dict(path=refs['dataset']['path'],byte_sha256=refs['dataset']['sha256'])
    docs['three_review']['input_sha256'][refs['dataset']['path']]=refs['dataset']['sha256']
    save('three_review',docs['three_review'])
    current={r['path']:r for r in refs.values()}
    for row in design['declared_document_inventory']:
        if row['document']['path'] in current:row['document']=current[row['document']['path']]
    for edge in design['historical_runtime']['edges']:
        if edge['declaring_document']['path']==refs['legacy_review']['path']:edge['declaring_document']=refs['legacy_review']
    design.update(role_refs=copy.deepcopy(refs),role_map={k:r['path'] for k,r in refs.items()})
    dref=put('new/DESIGN_V4_TEST.json',design)
    basis={k:put('new/'+k+'.json',dict(synthetic=True)) for k in ('schema_generator','failed_v3_attempt')}
    cref=put('new/SCHEMA_EXAMPLES.json',classification(refs['union_plan'],bref,basis))
    codes={name:put(name,(h.ROOT/name).read_bytes())['sha256'] for name in h.CODE_FILES}
    tr=put('new/TESTS_V4.json',dict(passed=True,synthetic_only=True,tests=1,failures=0,errors=0,
        terminal=dict(exit_code=0,chunk_id='synthetic-fixture'),code_sha256=codes))
    rr=put('new/REVIEW_V4.json',dict(passed=True,unresolved_blockers=[],code_sha256=codes))
    with ExitStack() as stack:
        for name,value in (('UNION_PLAN',refs['union_plan']),('ACTUAL_BINDING',bref),('CLASSIFICATION',cref)):
            stack.enter_context(patch.object(h,name,value))
        policy=builder.collect(root,dref,tr,rr)
        pref=put('new/POLICY_V4.json',policy)
        e=h.Evidence(root,{**policy['input_sha256'],pref['path']:pref['sha256']},policy['historical_dependency_edges'])
        authenticated=h.policy_check(policy,e)
        yield policy,pref,authenticated,e


class ExampleTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name).resolve();self.pins={}
        self.doc=template();self.ref=prior.legacy.put(self.root,'PLAN.json',self.doc,self.pins)
        self.patch=patch.object(h,'UNION_PLAN',self.ref);self.patch.start()
    def tearDown(self):self.patch.stop();self.tmp.cleanup()
    def reader(self):return h.Evidence(self.root,self.pins)
    def test_exact_three_are_not_files(self):
        ev=self.reader();ev.nested_refs(ev.read(self.ref));result=ev.complete_examples()
        self.assertEqual(result['examples'],h.example_rows());self.assertEqual(ev.verified,self.pins)
    def test_no_exemption_without_authenticated_read_or_after_mutation(self):
        ev=self.reader();doc=ev.read(self.ref)
        for obj in (self.doc,copy.deepcopy(doc)):
            with self.assertRaisesRegex(ValueError,'Authenticated'):ev.nested_refs(obj)
        doc['extra']=True
        with self.assertRaisesRegex(ValueError,'Authenticated'):ev.nested_refs(doc)
    def test_wrong_document_path_or_sha_is_not_exempt(self):
        for ref in (dict(self.ref,path='other.json'),dict(self.ref,sha256='f'*64)):
            for pointer,obj in h.EXAMPLES:self.assertFalse(h.is_schema_example(ref,pointer,obj))
        other=prior.legacy.put(self.root,'other.json',self.doc,self.pins)
        ev=self.reader()
        with self.assertRaises(ValueError):ev.nested_refs(ev.read(other))
    def test_full_object_and_pointer_are_exact(self):
        for pointer,obj in h.EXAMPLES:
            for changed in (dict(obj,extra=True),dict(obj,sha256=None),dict(obj,path='other.json')):
                self.assertFalse(h.is_schema_example(self.ref,pointer,changed))
            self.assertFalse(h.is_schema_example(self.ref,pointer+'/extra',obj))
    def test_duplicate_elsewhere_is_active_and_rejects(self):
        doc=copy.deepcopy(self.doc);doc['copied']=copy.deepcopy(h.EXAMPLES[0][1])
        ref=prior.legacy.put(self.root,'PLAN.json',doc,self.pins)
        with patch.object(h,'UNION_PLAN',ref):
            ev=self.reader()
            with self.assertRaises(ValueError):ev.nested_refs(ev.read(ref))
    def test_unknown_placeholder_null_or_malformed_digest_rejects(self):
        for sha in ('<64hex>',None,'','a'*63,'A'*64):
            doc=dict(ref=dict(path='real.bin',sha256=sha))
            ref=prior.legacy.put(self.root,'bad.json',doc,self.pins)
            ev=self.reader()
            with self.subTest(sha=sha),self.assertRaises(ValueError):ev.nested_refs(ev.read(ref))
    def test_null_digest_cannot_fall_back_to_existing_active_pin(self):
        actual=prior.legacy.put(self.root,'real.bin',b'real bytes',self.pins)
        ref=prior.legacy.put(self.root,'bad.json',dict(actual=dict(path=actual['path'],sha256=None)),self.pins)
        ev=self.reader()
        with self.assertRaisesRegex(ValueError,'Canonical'):ev.nested_refs(ev.read(ref))
    def test_omitted_example_rejects_completion(self):
        doc=copy.deepcopy(self.doc);del doc['policy']['binding_document']['accounting_review']
        ref=prior.legacy.put(self.root,'PLAN.json',doc,self.pins)
        with patch.object(h,'UNION_PLAN',ref):
            ev=self.reader();ev.nested_refs(ev.read(ref))
            with self.assertRaisesRegex(ValueError,'Complete three'):ev.complete_examples()
    def test_same_target_real_reference_remains_active(self):
        actual=prior.legacy.put(self.root,h.EXAMPLES[0][1]['path'],b'real accounting',self.pins)
        doc=copy.deepcopy(self.doc);doc['actual']=actual
        ref=prior.legacy.put(self.root,'PLAN.json',doc,self.pins)
        with patch.object(h,'UNION_PLAN',ref):
            ev=self.reader();ev.nested_refs(ev.read(ref));ev.complete_examples()
            self.assertEqual(ev.verified[actual['path']],actual['sha256'])
            (self.root/actual['path']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'SHA mismatch'):ev.nested_refs(ev.read(ref))
    def test_pointer_escaping_never_collapses_path(self):
        doc={'policy/binding_document':{'accounting_review':copy.deepcopy(h.EXAMPLES[0][1])}}
        ref=prior.legacy.put(self.root,'PLAN.json',doc,self.pins)
        with patch.object(h,'UNION_PLAN',ref):
            ev=self.reader()
            with self.assertRaises(ValueError):ev.nested_refs(ev.read(ref))
    def test_missing_active_pin_rejects(self):
        doc=dict(actual=dict(path='missing.bin',sha256='a'*64))
        ref=prior.legacy.put(self.root,'bad.json',doc,self.pins);ev=self.reader()
        with self.assertRaisesRegex(ValueError,'omitted'):ev.nested_refs(ev.read(ref))
    def test_changed_document_bytes_fail_before_classification(self):
        (self.root/self.ref['path']).write_bytes(b'changed');ev=self.reader()
        with self.assertRaisesRegex(ValueError,'SHA mismatch'):ev.read(self.ref)
    def test_collector_never_turns_unknown_placeholder_into_pin(self):
        ref=prior.legacy.put(self.root,'bad.json',dict(x=dict(path='missing.bin',sha256='<64hex>')),self.pins)
        ev=builder.Collector(self.root,{})
        with self.assertRaisesRegex(ValueError,'Canonical'):ev.nested_refs(ev.read(ref))
        self.assertNotIn('missing.bin',ev.pins)
    def test_classification_list_missing_duplicate_extra_or_changed_fails(self):
        for mode in ('missing','duplicate','extra','changed','scope'):
            d=classification(self.ref,h.ACTUAL_BINDING,{})
            if mode=='missing':d['examples'].pop()
            elif mode=='duplicate':d['examples'].append(copy.deepcopy(d['examples'][0]))
            elif mode=='extra':d['examples'][0]['complete_object']['extra']=True
            elif mode=='changed':d['examples'][0]['json_pointer']='/wrong'
            else:d['execution_authorized']=True
            ref=prior.legacy.put(self.root,'classification.json',d,self.pins)
            with patch.object(h,'CLASSIFICATION',ref):
                with self.subTest(mode=mode),self.assertRaises(ValueError):h.classification_check({'schema_examples':ref},self.reader())


class PackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name).resolve()
        cls.context=package(cls.root);cls.p,cls.ref,cls.docs,cls.ev=cls.context.__enter__()
    @classmethod
    def tearDownClass(cls):cls.context.__exit__(None,None,None);cls.tmp.cleanup()
    def test_full_v4_builder_policy_and_synthetic_census(self):
        self.assertEqual(len(self.p['code_sha256']),10)
        self.assertNotIn('<local-relative-review.json>',self.p['input_sha256'])
        self.assertIn(h.ACTUAL_BINDING['path'],self.p['input_sha256'])
        with patch.object(h.scope,'assess',wraps=h.scope.assess) as spy:r=h.run(self.root,self.ref['path'],self.ref['sha256'])
        self.assertEqual(spy.call_count,44);self.assertTrue(r['passed'])
        data=h.old.json.loads((self.root/h.OUT/'DATASET.json').read_bytes())
        for event,old in zip(data['events'],self.docs['dataset']['events']):
            self.assertEqual({k:v for k,v in event.items() if not k.startswith('roa_field_scope270_v3')},old)
        manifest=h.old.json.loads((self.root/h.OUT/'MANIFEST.json').read_bytes())
        self.assertEqual(len(manifest['schema_example_classification']['examples']),3)
        self.assertEqual(len(manifest['historical_dependency_edges']),574)
        with self.assertRaisesRegex(ValueError,'Write-once'):h.run(self.root,self.ref['path'],self.ref['sha256'])
    def test_actual_binding_and_real_references_must_match(self):
        for mode in ('binding','sourcehash','accounting','html','pdf'):
            docs=copy.deepcopy(self.docs);policy=copy.deepcopy(self.p)
            if mode=='binding':docs['union_plan']['binding_sha256']='f'*64
            elif mode=='sourcehash':del docs['union_plan']['source_hashes'][h.ACTUAL_BINDING['path']]
            else:
                role={'accounting':'accounting','html':'html_repair_review','pdf':'pdf_review'}[mode]
                policy['inputs'][role]['sha256']='f'*64
            with self.subTest(mode=mode),self.assertRaises(ValueError):h.actual_binding_check(policy,docs,self.ev)
    def test_real_binding_bytes_tamper_rejects_before_assess(self):
        path=self.root/h.ACTUAL_BINDING['path'];raw=path.read_bytes()
        try:
            path.write_bytes(b'changed')
            with patch.object(h.scope,'assess',side_effect=AssertionError('must not assess')) as spy:
                with self.assertRaisesRegex(ValueError,'SHA mismatch'):h.policy_check(self.p,h.Evidence(self.root,self.p['input_sha256'],self.p['historical_dependency_edges']))
            self.assertEqual(spy.call_count,0)
        finally:path.write_bytes(raw)
    def test_classification_byte_tamper_rejects_before_assess(self):
        path=self.root/h.CLASSIFICATION['path'];raw=path.read_bytes()
        try:
            path.write_bytes(b'changed')
            with patch.object(h.scope,'assess',side_effect=AssertionError('must not assess')) as spy:
                with self.assertRaisesRegex(ValueError,'SHA mismatch'):h.policy_check(self.p,h.Evidence(self.root,self.p['input_sha256'],self.p['historical_dependency_edges']))
            self.assertEqual(spy.call_count,0)
        finally:path.write_bytes(raw)
    def test_frozen_financial_functions_are_reused_verbatim(self):
        self.assertIs(h.pdf_package,h.v3.pdf_package);self.assertIs(h.assess_census,h.v3.assess_census)
        self.assertIs(h.design_check,h.v3.design_check);self.assertIs(h.scope,h.v3.scope)
    def test_changed_union_hash_map_still_requires_active_pin(self):
        d=copy.deepcopy(self.docs['union_plan']);d['source_hashes']['unknown-real.bin']='a'*64
        pins=dict(self.p['input_sha256']);ref=prior.legacy.put(self.root,'new/otherdecl.json',d,pins)
        ev=h.Evidence(self.root,pins,self.p['historical_dependency_edges'])
        with self.assertRaisesRegex(ValueError,'omitted'):ev.declared(ev.read(ref))


def load_tests(loader, tests, pattern):
    # The inherited financial routines are literally the same functions; retain
    # their frozen 28 edge/financial negative tests without rebinding globals.
    tests.addTests(loader.loadTestsFromModule(prior))
    return tests


if __name__=='__main__':unittest.main(verbosity=2)
