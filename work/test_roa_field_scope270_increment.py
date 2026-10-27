"""Fabricated source packages only; never load the actual financial census."""
import copy
from contextlib import contextmanager
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import roa_field_scope270_increment as h
from test_roa_field_scope270_host import put, source_event


@contextmanager
def package(root):
    pins={}
    def save(name,obj): return put(root,name,obj,pins)
    evidence=save('reviews/fields.json',dict(synthetic=True))
    events=[source_event(root,i,pins) for i in range(44)]
    for i,event in enumerate(events):
        if i<21:
            event['accounting_review_flags']=[dict(code=h.scope.REVIEWABLE_FLAG),
                dict(code='cashflow_accounting_change_context_unreviewed',matched_terms=['adoption'])]
        if 31<=i<34:
            event['event_id']=h.EVENT_IDS[i-31]
    original=dict(schema_version=1,method='synthetic-legacy',events=events,policy_sha256='a'*64)
    original_ref=save('original/DATASET.json',original)
    code={name:save(name,(h.ROOT/name).read_bytes())['sha256'] for name in h.CODE_FILES}
    oldcode={name:code[name] for name in h.saved.CODE_FILES}
    # New module/test are not falsely included in the old executed policy.
    sourcepins={k:v for k,v in pins.items() if k not in h.CODE_FILES[-2:]}
    p4=dict(schema=h.saved.SCHEMA,code_sha256=oldcode,input_sha256=sourcepins,
        inputs={'dataset':original_ref},historical_dependency_edges=[{'synthetic_historical_edge':True}])
    refs={'v4_policy':save('saved/POLICY.json',p4)}
    sessions=sorted({day for e in events for day in (e['release_date'],e['available_from'])})+['2026-12-31']
    ev=h.saved.Evidence(root,pins)
    rows=[]; results=[]; regressions=[]; actual=[]; design=[]
    for i,event in enumerate(events):
        esha=h.scope.canonical_sha256(event)
        hashes={s+'.'+f:h.scope.canonical_sha256(event['fields'][s][f]) for s,f in h.scope.REQUIRED}
        review=dict(schema=h.scope.REVIEW_SCHEMA,event_id=event['event_id'],event_sha256=esha,
            raw_source={k:event['source'][k] for k in ('path','sha256')},cell_sha256=hashes,
            review_receipts=[evidence],independently_reviewed=True) if i<34 else None
        decision=h.scope.assess(event,sessions,review)
        auth=h.old.verify_receipt(event['source'],ev)
        row=copy.deepcopy(event); row.update(roa_field_scope270_v3=decision,
            roa_field_scope270_v3_evidence=dict(field_review=review,authentication=auth,mapping={'synthetic':True}))
        rows.append(row)
        prior=event['cash_quality_admission']['admitted']
        results.append(dict(event_id=event['event_id'],prior_admitted=prior,admitted=decision['admitted'],
            reasons=decision['reasons'],review_available=review is not None,event_sha256=esha))
        if prior and not decision['admitted']: regressions.append(dict(event_id=event['event_id'],reasons=decision['reasons']))
        if 31<=i<34:
            ar={k:copy.deepcopy(event[k]) for k in ('event_id','source','identity','release_date','available_from')}
            ar.update(event_sha256=esha,current_period=copy.deepcopy(event['periods']['current']),
                opening_asset_date=event['periods']['comparison']['period_end'],unresolved_ni_asset_fact_blockers=[],cells=[])
            for key in ('independent_field_fact_review_passed','independent_source_clock_identity_binding_passed',
                'source_only_incremental_acceptance_supported','clock_rule_verified','source_receipt_verified',
                'source_scope_decision_observed_admitted','comparison_ni_cfo_duration_still_unrepaired'): ar[key]=True
            for key in ('certified_daily_security_master','first_public_version_certified','ratio_or_chain_admitted','source_period_start_newly_certified'): ar[key]=False
            for side,field in h.scope.REQUIRED:
                c=event['fields'][side][field]
                ar['cells'].append(dict(field_path=side+'.'+field,cell_sha256=hashes[side+'.'+field],
                    matched_saved_independent_fact=True,iso_currency_certified=False,value_decimal=c['value_decimal'],
                    scale=c['scale'],period_end=c['period']['period_end'],literal_currency=c['currency_literal']))
            actual.append(ar)
            design.append(dict(event_id=event['event_id'],actual_per_event_admitted=True,actual_reasons=[],
                prior_cash_quality_admitted=False,original_event_sha256=esha,required_cell_sha256=hashes,
                field_review_sha256=decision['field_review_sha256']))
    data=copy.deepcopy(original); data['events']=rows; data['roa_field_scope270_source_v4']={'synthetic':True}
    refs['v4_dataset']=save('saved/DATASET.json',data)
    examples={'synthetic_schema_examples':True}
    assessment=dict(schema='roa_field_scope270_source_assessment_v4',policy=refs['v4_policy'],
        passed=False,events=44,admitted_events=13,held_events=31,original_event_preservation=True,
        input_sha256=sourcepins,historical_dependency_edges=p4['historical_dependency_edges'],
        schema_example_classification=examples,source_only=True,old_runtime_verified=False,
        event_results=results,old_admitted_regressions=regressions)
    refs['v4_assessment']=save('saved/ASSESSMENT.json',assessment)
    manifest=dict(schema='roa_field_scope270_source_manifest_v4',policy=refs['v4_policy'],completed=True,
        assessment_passed=False,old_runtime_verified=False,source_only=True,input_sha256=sourcepins,
        historical_dependency_edges=p4['historical_dependency_edges'],schema_example_classification=examples,
        output_sha256={'DATASET.json':refs['v4_dataset']['sha256'],'ASSESSMENT.json':refs['v4_assessment']['sha256']})
    refs['v4_manifest']=save('saved/MANIFEST.json',manifest)
    terminal=lambda code:dict(session_id=123,chunk_id='synthetic',exit_code=code,owner_polled_to_terminal=True)
    attempt=dict(status='ACTUAL_SOURCE_CENSUS_FAILED_OLD_ADMISSION_REGRESSIONS',policy=refs['v4_policy'],
        financial_admission_accepted=False,events=44,admitted_events=13,held_events=31,
        policy_terminal=terminal(0),census_terminal=terminal(1),
        logs={'census':save('logs/failed.log',b'fabricated failed census')},
        outputs={refs[k]['path']:refs[k]['sha256'] for k in ('v4_manifest','v4_dataset','v4_assessment')},
        old_admitted_regressions=regressions)
    refs['failed_attempt']=save('saved/ATTEMPT.json',attempt)
    actual_review=dict(schema='roa_field_scope270_source_v4_saved_actual_independent_review_v1',
        passed=False,full_census_passed=False,audit_completed=True,structural_checks_passed=True,
        source_evidence_checks_passed=True,source_only=True,strict_pit=False,counts=copy.deepcopy(h.COUNTS),
        policy_sha256=refs['v4_policy']['sha256'],code_sha256=oldcode,
        closure_review=dict(exact_active_input_sha256=sourcepins,passed=True,old_runtime_verified=False),
        input_sha256={ref['path']:ref['sha256'] for ref in refs.values()},
        failure_analysis=dict(regressions=[dict(event_id=x['event_id'],strict_scope_reasons=x['reasons']) for x in regressions],
            newly_observed_contradiction_in_selected_ni_asset_facts=False),
        exact_three_incremental_review=dict(passed=True,predeclared_exact_event_ids=list(h.EVENT_IDS),events=actual))
    refs['actual_review']=save('reviews/actual.json',actual_review)
    scope_review=dict(actual_overall_passed=False,financial_admission_accepted=False,
        source_increment_data_acceptance_authorized=False,input_sha256={},
        actual_failure=dict(full_manifest=refs['v4_manifest'],regression_rows=regressions),
        selection_bias_check=dict(predeclared_event_ids=list(h.EVENT_IDS)),exact_three_saved_evidence=design)
    refs['scope_review']=save('reviews/scope.json',scope_review)
    proposal=dict(known_scope_candidates_from_source_metadata=list(h.EVENT_IDS),
        required_fields=['fields.'+x for x in h.scope.CELL_NAMES],input_sha256={})
    refs['proposal']=save('reviews/proposal.json',proposal)
    notes=dict(source_admission_authorized=False,input_sha256={})
    refs['design_notes']=save('reviews/notes.json',notes)
    docs=dict(v4_policy=p4,v4_manifest=manifest,v4_assessment=assessment,v4_dataset=data,
        failed_attempt=attempt,actual_review=actual_review,scope_review=scope_review,proposal=proposal,design_notes=notes)
    # Separate JSON files do not share dict identities. Preserve that boundary in mutations.
    docs=h.old.json.loads(h.encoded(docs))
    testlog=save('new/tests.log',b'fabricated synthetic tests only')
    tests=save('new/TESTS.json',dict(schema='roa_field_scope270_increment_synthetic_tests_v1',
        passed=True,synthetic_only=True,tests=1,failures=0,errors=0,terminal=terminal(0),log=testlog,
        code_sha256=code,input_sha256={testlog['path']:testlog['sha256']}))
    review=save('new/CODE_REVIEW.json',dict(schema='roa_field_scope270_increment_independent_code_review_v1',
        passed=True,code_review_passed=True,unresolved_blockers=[],tests=tests,code_sha256=code,
        input_sha256={tests['path']:tests['sha256']}))
    policy=dict(schema=h.SCHEMA,execution_authorized=True,source_only=True,strategy_registration_authorized=False,
        ratios_permitted=False,chain_admission_permitted=False,inputs=refs,event_ids=list(h.EVENT_IDS),output_directory=h.OUT,
        code_sha256=code,input_sha256=dict(pins),tests=tests,independent_code_review=review,runtime=dict(python=sys.version,
            executable_sha256=h.digest(Path(sys.executable).read_bytes())))
    pref=save('new/POLICY.json',policy)
    with patch.object(h,'FIXED_INPUTS',copy.deepcopy(refs)):
        yield dict(policy=policy,pref=pref,refs=refs,docs=docs,pins=pins,original=original,
            ev=h.saved.Evidence(root,pins),root=root)


def rewrite_policy(x, change):
    policy=copy.deepcopy(x['policy']); change(policy)
    policy['input_sha256']={k:v for k,v in x['pins'].items() if k!=x['pref']['path']}
    x['pref']=put(x['root'],x['pref']['path'],policy,x['pins']); x['policy']=policy


def rewrite_receipt(x, role, change):
    ref=x['policy'][role]; obj=h.old.json.loads((x['root']/ref['path']).read_bytes()); change(obj)
    ref=put(x['root'],ref['path'],obj,x['pins'])
    rewrite_policy(x,lambda p:p.update({role:ref}))


class SavedIncrementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name).resolve()
        cls.ctx=package(cls.root);cls.x=cls.ctx.__enter__()
    @classmethod
    def tearDownClass(cls):cls.ctx.__exit__(None,None,None);cls.tmp.cleanup()
    def verify(self,docs=None): return h.verify_saved(docs or self.x['docs'],self.x['refs'],self.x['ev'])
    def rejected(self,change,pattern=None):
        docs=copy.deepcopy(self.x['docs']);change(docs)
        with self.assertRaisesRegex(ValueError,pattern or '.'):
            self.verify(docs)
    def rejected_run(self,change,before_verify=False):
        with tempfile.TemporaryDirectory() as tmp, package(Path(tmp).resolve()) as x:
            change(x)
            with patch.object(h.scope,'assess',side_effect=AssertionError('no new assess')):
                if before_verify:
                    with patch.object(h,'verify_saved',side_effect=AssertionError('review gate must precede extraction')):
                        with self.assertRaises(ValueError):h.run(x['root'],x['pref']['path'],x['pref']['sha256'])
                else:
                    with self.assertRaises(ValueError):h.run(x['root'],x['pref']['path'],x['pref']['sha256'])
            self.assertFalse((x['root']/h.OUT).exists())
    def test_complete_saved_fail_and_three_extract_no_assess(self):
        before=h.scope.canonical_sha256(self.x['docs'])
        with patch.object(h.scope,'assess',side_effect=AssertionError('must not reassess')):
            original,accepted,regressions=self.verify()
        self.assertEqual(len(original['events']),44);self.assertEqual(len(accepted),3)
        self.assertEqual(len(regressions),21);self.assertEqual(before,h.scope.canonical_sha256(self.x['docs']))
        self.assertIn('cashflow_accounting_change_context_unreviewed',str(original['events'][0]['accounting_review_flags']))
    def test_failed_census_cannot_be_relabelled_pass(self):
        for role,key in (('v4_manifest','assessment_passed'),('v4_assessment','passed'),('actual_review','passed'),
                         ('actual_review','full_census_passed'),('scope_review','actual_overall_passed')):
            with self.subTest(role=role,key=key):self.rejected(lambda d:d[role].update({key:True}))
    def test_independent_receipt_requires_scope_not_only_passed(self):
        for key in ('audit_completed','structural_checks_passed','source_evidence_checks_passed'):
            with self.subTest(key=key):self.rejected(lambda d:d['actual_review'].update({key:False}))
    def test_unknown_regression_cannot_replace_generic_hold(self):
        self.rejected(lambda d:d['v4_dataset']['events'][0]['roa_field_scope270_v3'].update(reasons=['new_bad_value']))
    def test_regression_rows_cannot_be_dropped(self):self.rejected(lambda d:d['v4_assessment']['old_admitted_regressions'].pop())
    def test_failed_terminal_exit_and_owner_required(self):
        for key,value in (('exit_code',0),('owner_polled_to_terminal',False),('session_id',True),('chunk_id','')):
            with self.subTest(key=key):self.rejected(lambda d:d['failed_attempt']['census_terminal'].update({key:value}))
    def test_original_full_catalog_cannot_be_three_only(self):
        self.rejected(lambda d:d['v4_dataset'].update(events=d['v4_dataset']['events'][31:34]))
    def test_original_catalog_order_or_body_tamper(self):
        self.rejected(lambda d:d['v4_dataset']['events'].reverse())
        self.rejected(lambda d:d['v4_dataset']['events'][43].update(release_date='2015-01-01'))
    def test_fourth_increment_not_accepted(self):
        self.rejected(lambda d:d['v4_dataset']['events'][34]['roa_field_scope270_v3'].update(admitted=True,reasons=[]))
    def test_fixed_declaration_and_duplicate_ids(self):
        self.rejected(lambda d:d['proposal']['known_scope_candidates_from_source_metadata'].append('FOURTH'))
        self.rejected(lambda d:d['actual_review']['exact_three_incremental_review']['events'].append(
            copy.deepcopy(d['actual_review']['exact_three_incremental_review']['events'][0])))
    def test_unknown_new_fact_blocker_rejected(self):
        self.rejected(lambda d:d['actual_review']['exact_three_incremental_review']['events'][0].update(unresolved_ni_asset_fact_blockers=['unknown']))
        self.rejected(lambda d:d['actual_review']['failure_analysis'].update(newly_observed_contradiction_in_selected_ni_asset_facts=True))
    def test_event_and_nine_cell_digest_binding(self):
        for obj,key in (('event','event_sha256'),('cell','cell_sha256')):
            def change(d):
                event=d['actual_review']['exact_three_incremental_review']['events'][0]
                (event if obj=='event' else event['cells'][0])[key]='f'*64
            with self.subTest(obj=obj):self.rejected(change)
    def test_missing_or_reordered_independent_field(self):
        self.rejected(lambda d:d['actual_review']['exact_three_incremental_review']['events'][0]['cells'].pop())
        self.rejected(lambda d:d['actual_review']['exact_three_incremental_review']['events'][0]['cells'].reverse())
    def test_actual_amount_scale_endpoint_currency_identity_mismatch(self):
        for key,value in (('value_decimal','0'),('scale','1'),('period_end','2000-01-01'),('literal_currency','EUR')):
            with self.subTest(key=key):self.rejected(lambda d:d['actual_review']['exact_three_incremental_review']['events'][0]['cells'][0].update({key:value}))
        self.rejected(lambda d:d['actual_review']['exact_three_incremental_review']['events'][0]['identity'].update(issuer_id='other'))
    def test_cannot_promote_strictpit_or_chain(self):
        for key in ('strict_pit','chain_admitted','currency_iso_certified','source_bytes_verified'):
            with self.subTest(key=key):self.rejected(lambda d:d['v4_dataset']['events'][31]['roa_field_scope270_v3'].update({key:True}))
    def test_original_failure_refs_and_current_code_are_binding(self):
        self.rejected(lambda d:d['failed_attempt']['outputs'].update({'other.json':'f'*64}))
        self.rejected(lambda d:d['actual_review']['code_sha256'].update({h.saved.CODE_FILES[0]:'f'*64}))
    def test_active_pin_or_runtime_history_relabel_rejected(self):
        self.rejected(lambda d:d['v4_manifest']['input_sha256'].pop(next(iter(d['v4_manifest']['input_sha256']))))
        self.rejected(lambda d:d['v4_manifest'].update(old_runtime_verified=True))
        self.rejected(lambda d:d['v4_manifest']['historical_dependency_edges'].append({'unreviewed':True}))
    def test_saved_field_review_evidence_tamper(self):
        self.rejected(lambda d:d['v4_dataset']['events'][31]['roa_field_scope270_v3_evidence']['field_review']['cell_sha256'].update({'current.net_income':'f'*64}))
        self.rejected(lambda d:d['v4_dataset']['events'][31]['roa_field_scope270_v3_evidence']['authentication'].update(http_status=403))
    def test_new_tests_and_review_refs_are_required(self):
        for key in ('tests','independent_code_review'):
            with self.subTest(key=key):self.rejected_run(lambda x:rewrite_policy(x,lambda p:p.pop(key)),True)
    def test_tests_scope_count_exit_and_code_inventory(self):
        changes=[lambda t:t.update(passed=1),lambda t:t.update(synthetic_only=False),lambda t:t.update(tests=0),
            lambda t:t.update(tests=True),lambda t:t.update(failures=1),lambda t:t.update(errors=1),
            lambda t:t['terminal'].update(exit_code=1),lambda t:t['terminal'].update(owner_polled_to_terminal=False),
            lambda t:t['code_sha256'].pop(h.CODE_FILES[-1]),lambda t:t['code_sha256'].update({h.CODE_FILES[0]:'a'*64})]
        for i,change in enumerate(changes):
            with self.subTest(i=i):self.rejected_run(lambda x:rewrite_receipt(x,'tests',change),True)
    def test_code_review_scope_inventory_and_test_binding(self):
        changes=[lambda r:r.update(passed=1),lambda r:r.update(code_review_passed=False),
            lambda r:r.update(unresolved_blockers=['unresolved']),lambda r:r['code_sha256'].pop(h.CODE_FILES[-1]),
            lambda r:r['tests'].update(sha256='a'*64)]
        for i,change in enumerate(changes):
            with self.subTest(i=i):self.rejected_run(lambda x:rewrite_receipt(x,'independent_code_review',change),True)
    def test_receipt_omitted_from_policy_pins(self):
        def change(x):
            p=copy.deepcopy(x['policy']);p['input_sha256'].pop(p['tests']['path'])
            x['pref']=put(x['root'],x['pref']['path'],p,x['pins'])
        self.rejected_run(change,True)
    def test_late_raw_receipt_and_code_byte_drift_no_output(self):
        for key in ('raw','receipt','code','log'):
            def change(x):
                src=x['original']['events'][-1]['source']
                name=src['path'] if key=='raw' else src['receipt']['path'] if key=='receipt' else h.CODE_FILES[-1] if key=='code' else 'new/tests.log'
                path=x['root']/name;path.write_bytes(path.read_bytes()+b'changed')
            with self.subTest(key=key):self.rejected_run(change)
    def test_post_extraction_active_mutation_does_not_publish(self):
        with tempfile.TemporaryDirectory() as tmp, package(Path(tmp).resolve()) as x:
            original_verify=h.verify_saved
            def mutate(*args):
                result=original_verify(*args)
                raw=x['root']/x['original']['events'][-1]['source']['path'];raw.write_bytes(b'late mutation')
                return result
            with patch.object(h,'verify_saved',side_effect=mutate), patch.object(h.scope,'assess',side_effect=AssertionError('no assess')):
                with self.assertRaisesRegex(ValueError,'hash|changed|SHA|digest'):
                    h.run(x['root'],x['pref']['path'],x['pref']['sha256'])
            self.assertFalse((x['root']/h.OUT).exists())
    def test_authorization_before_any_failed_source_read(self):
        p=copy.deepcopy(self.x['policy']);p['execution_authorized']=False
        ref=put(self.root,'new/DISABLED.json',p,self.x['pins'])
        with patch.object(h,'verify_saved',side_effect=AssertionError('no source validation')):
            with self.assertRaisesRegex(ValueError,'authorization'):h.run(self.root,ref['path'],ref['sha256'])
    def test_full_run_write_once_retains_failed_evidence(self):
        with patch.object(h.scope,'assess',side_effect=AssertionError('no assess')):
            result=h.run(self.root,self.x['pref']['path'],self.x['pref']['sha256'])
        self.assertEqual(result['increment_admitted'],3);self.assertFalse(result['full_census_passed'])
        out=h.old.json.loads((self.root/h.OUT/'INCREMENT.json').read_bytes())
        self.assertEqual(out['full_catalog'],self.x['original']['events'])
        self.assertEqual(out['increment_admitted_event_ids'],list(h.EVENT_IDS))
        self.assertEqual(out['counts']['new_assess_calls'],0);self.assertFalse(out['old_admissions_recertified'])
        with self.assertRaisesRegex(ValueError,'Write-once'):h.run(self.root,self.x['pref']['path'],self.x['pref']['sha256'])


if __name__=='__main__':unittest.main(verbosity=2)
