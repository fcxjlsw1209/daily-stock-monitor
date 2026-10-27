"""Strict saved-evidence acceptance of three predeclared single-report records.

Does not rerun assess, extract/render reports, compute ratios, or accept the
failed 44-event policy. Original catalog and every old decision remain intact.
Only a separately byte-pinned, explicitly authorized policy permits publication.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import json
import os
import sys

import roa_field_scope270_host_v4 as saved

scope, old = saved.scope, saved.old
need, same, digest, encoded = saved.need, saved.same, saved.digest, saved.encoded
ROOT = saved.ROOT
OUT = 'outputs/monthly_roa_field_scope270_increment_v1'
SCHEMA = 'roa_field_scope270_increment_policy_v1'
EVENT_IDS = ('AAPL:FY2016:2016-10-25', 'AMD:FY2015:2016-01-19', 'MSFT:FY2015:2015-07-21')
CODE_FILES = saved.CODE_FILES + ('work/roa_field_scope270_increment.py', 'work/test_roa_field_scope270_increment.py')
FIXED_INPUTS = {
    'v4_policy': dict(path='work/roa_field_scope270_SOURCE_POLICY_v4.json', sha256='c3da7df08a6ba7bafc4967313789d32742d1cd43b3d4051a83b34295b287412e'),
    'v4_manifest': dict(path='outputs/monthly_roa_field_scope270_source_v4/MANIFEST.json', sha256='4f32aa502b636e7c2a3e904533b7985838d6c73d0905ceb1e78cf2136565ea39'),
    'v4_dataset': dict(path='outputs/monthly_roa_field_scope270_source_v4/DATASET.json', sha256='f9d06b04fd4b91944fc918221a6d4fea55d67bf68334302af2a023afe03f78da'),
    'v4_assessment': dict(path='outputs/monthly_roa_field_scope270_source_v4/ASSESSMENT.json', sha256='1057a841138e4ef653885d590e03b43818980f028b10bd0ab3b0fcf8a47c9761'),
    'failed_attempt': dict(path='work/roa_field_scope270_v4_ACTUAL_ATTEMPT1.json', sha256='88e112731a10bae820d7cbeb4d8efb58f9966dcb9b279cbb6b88d179ac6bf7ad'),
    'actual_review': dict(path='work/roa_field_scope270_source_v4_ACTUAL_REVIEW.json', sha256='a2aa66b0fbb04ec960011c95376a6d08231fd77755b07eea0cb311961caf15d5'),
    'scope_review': dict(path='work/roa_field_scope270_INCREMENT_SCOPE_REVIEW.json', sha256='0ac4b78474141e574ec1d61858a8c58291e80c78d9c8fa36a4017d775fe11077'),
    'proposal': dict(path='work/after269_roa_field_scope270_PROPOSAL.json', sha256='98a08744dff92b8d1ffa51c8bce43a9e0a1fc94bbec901678235061127a7c856'),
    'design_notes': dict(path='work/roa_field_scope270_INCREMENT_DESIGN_NOTES.json', sha256='f68b98075f5973045cd8c2ee21b1cafb80e8aa7e276adcebc26356bbacfc21a6'),
}
HOLD = 'accounting_source_hold:Preserved accounting review flag: cashflow_accounting_change_context_unreviewed'
COUNTS = dict(original_events=44, old_admitted=31, old_admitted_retained=10,
    old_admitted_regressions=21, prior_held=13, newly_admitted=3,
    prior_held_still_held=10, admitted_events=13, held_events=31)


def _pins(ev, values):
    need(type(values) is dict, 'Declared byte-pin map required')
    for path, sha in values.items(): ev.hash(path, sha)


def _terminal(value, code):
    need(type(value.get('exit_code')) is int and value['exit_code'] == code
         and type(value.get('session_id')) is int and bool(value.get('chunk_id'))
         and value.get('owner_polled_to_terminal') is True, 'Actual terminal evidence')


def _execution_review(policy, ev):
    refs = [policy.get(k) for k in ('tests', 'independent_code_review')]
    for ref in refs:
        need(type(ref) is dict and set(ref) == {'path', 'sha256'}, 'Final test/review reference required')
        need(policy['input_sha256'].get(ref['path']) == ref['sha256'], 'Test/review omitted from active pins')
    tests, review = [ev.read(ref) for ref in refs]
    need(tests.get('schema') == 'roa_field_scope270_increment_synthetic_tests_v1'
         and tests.get('passed') is True and tests.get('synthetic_only') is True
         and type(tests.get('tests')) is int and tests['tests'] > 0
         and type(tests.get('failures')) is int and tests['failures'] == 0
         and type(tests.get('errors')) is int and tests['errors'] == 0, 'Final synthetic test scope')
    _terminal(tests['terminal'], 0)
    need(review.get('schema') == 'roa_field_scope270_increment_independent_code_review_v1'
         and review.get('passed') is True and review.get('code_review_passed') is True
         and review.get('unresolved_blockers') == [], 'Final independent code review')
    same(review['tests'], refs[0], 'Code review binds final tests')
    for receipt in (tests, review):
        same(receipt['code_sha256'], policy['code_sha256'], 'Final receipt twelve-code binding')
        _pins(ev, receipt['input_sha256'])
    log = tests['log']; ev.hash(log['path'], log['sha256'])


def _new_record(event, saved_row, actual, design, ev):
    decision = saved_row['roa_field_scope270_v3']
    evidence = saved_row['roa_field_scope270_v3_evidence']; review = evidence['field_review']
    esha = scope.canonical_sha256(event)
    cells = {s+'.'+f:scope.canonical_sha256(event['fields'][s][f]) for s,f in scope.REQUIRED}
    # Binding helper only: deliberately no scope.assess or financial arithmetic.
    scope._review(event, review, esha, cells)
    need(set(decision) == set(scope.OUTPUT_KEYS) and decision['schema'] == scope.SCHEMA
         and decision['definition'] == scope.DEFINITION and decision['reasons'] == [], 'Exact saved same-scope decision')
    for key in ('admitted','evidence_binding_passed','same_report_literal_currency_assumption',
                'source_only','original_rejections_preserved','inputs_preserved'):
        need(decision[key] is True, 'Saved true scope flag: '+key)
    for key in ('currency_iso_certified','strict_pit','source_bytes_verified','review_receipt_bytes_verified',
                'first_public_version_verified','complete_correction_history_verified','ratios_computed',
                'chain_admitted','existing_admissions_modified'):
        need(decision[key] is False, 'Saved false scope flag: '+key)
    same(decision['event_sha256'], esha, 'Complete event digest')
    same(decision['required_cell_sha256'], cells, 'Nine required-cell digests')
    same(decision['field_review_sha256'], scope.canonical_sha256(review), 'Field review digest')
    same(decision['review_receipts'], review['review_receipts'], 'Same review receipt list')
    same(decision['raw_source'], review['raw_source'], 'Same raw source projection')
    same(decision['current_report_expires'], event['identity']['identity_valid_through'], 'Saved 550-day identity expiry')
    need(event['prior_field_admitted'] is False and event['cash_quality_admission']['admitted'] is False
         and event['metadata_gaps'] == [], 'New increment cannot recertify an old admission')
    need(event['inherited_rejection_codes'] and all(x in scope.INAPPLICABLE_CODES for x in event['inherited_rejection_codes']),
         'Only unused comparison-duration/start reasons')
    same(decision['scope_inapplicable_inherited_codes'], event['inherited_rejection_codes'], 'Original scope reasons preserved')
    same(event['accounting_review_flags'], [dict(code=scope.REVIEWABLE_FLAG)], 'No unknown increment accounting hold')
    same(decision['scope_satisfied_accounting_flags'], [scope.REVIEWABLE_FLAG], 'Independent review requirement')
    for ref in review['review_receipts']: ev.hash(ref['path'], ref['sha256'])
    same(old.verify_receipt(event['source'], ev), evidence['authentication'], 'Saved raw/retrieval authentication')
    for key in ('independent_field_fact_review_passed','independent_source_clock_identity_binding_passed',
                'source_only_incremental_acceptance_supported','clock_rule_verified','source_receipt_verified',
                'source_scope_decision_observed_admitted','comparison_ni_cfo_duration_still_unrepaired'):
        need(actual[key] is True, 'Independent actual event scope: '+key)
    for key in ('certified_daily_security_master','first_public_version_certified','ratio_or_chain_admitted','source_period_start_newly_certified'):
        need(actual[key] is False, 'Uncertified property remains false: '+key)
    need(actual['unresolved_ni_asset_fact_blockers'] == [], 'Unresolved three-field fact blocker')
    for key in ('event_id','source','identity','release_date','available_from'): same(actual[key],event[key],'Actual event '+key)
    same(actual['event_sha256'],esha,'Actual event SHA')
    same(actual['current_period'],event['periods']['current'],'Actual current annual period')
    same(actual['opening_asset_date'],event['periods']['comparison']['period_end'],'Actual opening instant')
    need([c['field_path'] for c in actual['cells']] == list(scope.CELL_NAMES), 'Exactly ordered three independently reviewed cells')
    for row,(side,name) in zip(actual['cells'],scope.REQUIRED):
        cell=event['fields'][side][name]
        need(row['matched_saved_independent_fact'] is True and row['iso_currency_certified'] is False, 'Independent cell scope')
        same(row['cell_sha256'],cells[side+'.'+name],'Actual cell digest')
        for key in ('value_decimal','scale'): same(row[key],cell[key],'Actual cell '+key)
        same(row['period_end'],cell['period']['period_end'],'Actual cell end')
        same(row['literal_currency'],cell['currency_literal'],'Actual literal currency')
        same(cell['source'],event['source'],'Same-report cell source')
    need(design['actual_per_event_admitted'] is True and design['actual_reasons'] == []
         and design['prior_cash_quality_admitted'] is False, 'Prior exact-three design scope')
    same(design['original_event_sha256'],esha,'Design event SHA')
    same(design['required_cell_sha256'],cells,'Design three fields')
    same(design['field_review_sha256'],decision['field_review_sha256'],'Design field review')
    return dict(event_id=event['event_id'], increment_admitted=True, event_sha256=esha,
        fields={s+'.'+f:deepcopy(event['fields'][s][f]) for s,f in scope.REQUIRED},
        saved_scope_decision=deepcopy(decision), saved_evidence=deepcopy(evidence),
        independent_actual_review=deepcopy(actual))


def verify_saved(docs, refs, ev):
    p,m,a,d,t,r,s = (docs[k] for k in ('v4_policy','v4_manifest','v4_assessment','v4_dataset','failed_attempt','actual_review','scope_review'))
    need(p['schema'] == saved.SCHEMA and set(p['code_sha256']) == set(saved.CODE_FILES), 'Original v4 ten-code policy')
    _pins(ev,p['input_sha256']); _pins(ev,p['code_sha256'])
    for obj in (m,a,t): same(obj['policy'],refs['v4_policy'],'Saved policy cross-binding')
    need(m['schema'] == 'roa_field_scope270_source_manifest_v4' and m['completed'] is True
         and m['assessment_passed'] is False and m['old_runtime_verified'] is False,
         'Complete output is not a passed population')
    need(a['schema'] == 'roa_field_scope270_source_assessment_v4' and a['passed'] is False
         and a['events'] == 44 and a['admitted_events'] == 13 and a['held_events'] == 31
         and a['original_event_preservation'] is True, 'Preserve failed 44-event result')
    same(m['output_sha256'],{name:refs[role]['sha256'] for name,role in
         (('DATASET.json','v4_dataset'),('ASSESSMENT.json','v4_assessment'))},'Exactly two saved output members')
    same(m['input_sha256'],a['input_sha256'],'Saved active source closure')
    same(m['input_sha256'],p['input_sha256'],'Policy active source closure')
    for obj in (m,a):
        same(obj['historical_dependency_edges'],p['historical_dependency_edges'],'Historical edges unchanged')
        need(obj['source_only'] is True and obj['old_runtime_verified'] is False, 'Source-only historical limits')
    same(m['schema_example_classification'],a['schema_example_classification'],'Schema examples unchanged')
    need(t['status'] == 'ACTUAL_SOURCE_CENSUS_FAILED_OLD_ADMISSION_REGRESSIONS'
         and t['financial_admission_accepted'] is False and t['events'] == 44
         and t['admitted_events'] == 13 and t['held_events'] == 31, 'Failed actual receipt retained')
    _terminal(t['policy_terminal'],0); _terminal(t['census_terminal'],1)
    for ref in t['logs'].values(): ev.hash(ref['path'],ref['sha256'])
    same(t['outputs'],{refs[k]['path']:refs[k]['sha256'] for k in ('v4_manifest','v4_dataset','v4_assessment')},'Attempt output pins')
    need(r['schema'] == 'roa_field_scope270_source_v4_saved_actual_independent_review_v1'
         and r['passed'] is False and r['full_census_passed'] is False
         and r['audit_completed'] is True and r['structural_checks_passed'] is True
         and r['source_evidence_checks_passed'] is True and r['source_only'] is True and r['strict_pit'] is False,
         'Independent actual evidence passed while population remains failed')
    same(r['counts'],COUNTS,'Actual review full census counts'); same(r['policy_sha256'],refs['v4_policy']['sha256'],'Actual reviewed policy')
    same(r['code_sha256'],p['code_sha256'],'Actual reviewed v4 code')
    same(r['closure_review']['exact_active_input_sha256'],p['input_sha256'],'Actual reviewed active inventory')
    need(r['closure_review']['passed'] is True and r['closure_review']['old_runtime_verified'] is False,
         'Actual closure review scope')
    for document in (r,s,docs['proposal'],docs['design_notes']): _pins(ev,document['input_sha256'])
    for role in ('v4_policy','v4_manifest','v4_dataset','v4_assessment','failed_attempt'):
        need(r['input_sha256'].get(refs[role]['path']) == refs[role]['sha256'], 'Actual review binds '+role)
    need(s['actual_overall_passed'] is False and s['financial_admission_accepted'] is False
         and s['source_increment_data_acceptance_authorized'] is False, 'Scope review is not retroactive full acceptance')
    same(s['actual_failure']['full_manifest'],refs['v4_manifest'],'Scope review failed manifest')
    same(s['selection_bias_check']['predeclared_event_ids'],list(EVENT_IDS),'Predetermined three events')
    same(docs['proposal']['known_scope_candidates_from_source_metadata'],list(EVENT_IDS),'Original pre-actual proposal')
    same(docs['proposal']['required_fields'],['fields.'+x for x in scope.CELL_NAMES],'Original three-field scope')
    need(docs['design_notes']['source_admission_authorized'] is False, 'Proposal is not authorization')
    original=ev.read(p['inputs']['dataset'])
    events=old.unique(original['events']); rows=old.unique(d['events'])
    need(len(events)==44 and list(events)==list(rows), 'Complete ordered original44 catalog')
    projected=deepcopy(d); projected.pop('roa_field_scope270_source_v4')
    projected['events']=[{k:v for k,v in row.items() if k not in
        ('roa_field_scope270_v3','roa_field_scope270_v3_evidence')} for row in d['events']]
    same(projected,original,'Original44 complete projection unchanged')
    results=[]; regressions=[]; added=[]
    for key,event in events.items():
        decision=rows[key]['roa_field_scope270_v3']; prior=event['cash_quality_admission']['admitted']
        need(type(prior) is bool and type(decision['admitted']) is bool,'Strict admission bool')
        need(decision['event_sha256']==scope.canonical_sha256(event),'All44 event digests')
        results.append(dict(event_id=key,prior_admitted=prior,admitted=decision['admitted'],reasons=decision['reasons'],
            review_available=rows[key]['roa_field_scope270_v3_evidence']['field_review'] is not None,
            event_sha256=decision['event_sha256']))
        if prior and not decision['admitted']:
            need(decision['reasons']==[HOLD],'Unknown old-admission regression')
            regressions.append(dict(event_id=key,reasons=decision['reasons']))
        if not prior and decision['admitted']: added.append(key)
    same(results,a['event_results'],'All44 saved per-event decisions')
    counts = dict(original_events=len(results), old_admitted=sum(x['prior_admitted'] for x in results),
        old_admitted_retained=sum(x['prior_admitted'] and x['admitted'] for x in results),
        old_admitted_regressions=len(regressions), prior_held=sum(not x['prior_admitted'] for x in results),
        newly_admitted=len(added), prior_held_still_held=sum(not x['prior_admitted'] and not x['admitted'] for x in results),
        admitted_events=sum(x['admitted'] for x in results), held_events=sum(not x['admitted'] for x in results))
    same(counts, COUNTS, 'Derived complete census counts')
    need(len(regressions)==21 and added==list(EVENT_IDS),'Exactly three additions and21 preserved regressions')
    for value in (a['old_admitted_regressions'],t['old_admitted_regressions'],s['actual_failure']['regression_rows']):
        same(regressions,value,'Do not resolve or omit old21 holds')
    same([dict(event_id=x['event_id'],reasons=x['strict_scope_reasons']) for x in r['failure_analysis']['regressions']],
         regressions,'Independent regression scope')
    need(r['failure_analysis']['newly_observed_contradiction_in_selected_ni_asset_facts'] is False,
         'New substantive financial contradiction requires review')
    exact=r['exact_three_incremental_review']
    need(exact['passed'] is True and exact['predeclared_exact_event_ids']==list(EVENT_IDS)
         and [x['event_id'] for x in exact['events']]==list(EVENT_IDS),'Independent exact-three review')
    designs=old.unique(s['exact_three_saved_evidence']); need(set(designs)==set(EVENT_IDS),'Exact-three design evidence')
    accepted=[_new_record(events[key],rows[key],actual,designs[key],ev) for key,actual in zip(EVENT_IDS,exact['events'])]
    return original, accepted, regressions


def run(root, policy_path, policy_sha):
    pref=dict(path=policy_path,sha256=policy_sha)
    policy=old.Evidence(root,{policy_path:policy_sha}).read(pref)
    need(policy.get('schema')==SCHEMA and policy.get('execution_authorized') is True
         and policy.get('source_only') is True and policy.get('strategy_registration_authorized') is False
         and policy.get('ratios_permitted') is False and policy.get('chain_admission_permitted') is False,
         'Separate explicit source-only increment authorization required')
    same(policy['inputs'],FIXED_INPUTS,'Exactly reviewed saved artifacts')
    same(policy['event_ids'],list(EVENT_IDS),'Fixed increment population')
    need(policy['output_directory']==OUT and set(policy['code_sha256'])==set(CODE_FILES),'Fixed output and12code inventory')
    need(policy['runtime']['python']==sys.version and policy['runtime']['executable_sha256']==digest(Path(sys.executable).read_bytes()),
         'Pinned current runtime')
    ev=saved.Evidence(root,{**policy['input_sha256'],policy_path:policy_sha})
    need(not (Path(root)/OUT).exists(),'Write-once increment output')
    _pins(ev,policy['input_sha256']); _pins(ev,policy['code_sha256'])
    _execution_review(policy, ev)
    docs={k:ev.read(ref) for k,ref in policy['inputs'].items()}
    same({k:policy['code_sha256'][k] for k in saved.CODE_FILES},docs['v4_policy']['code_sha256'],'Frozen v4 code unchanged')
    original,accepted,regressions=verify_saved(docs,policy['inputs'],ev)
    ev.stable()
    result=dict(schema='roa_field_scope270_increment_v1',policy=pref,source_only=True,strict_pit=False,
        increment_admitted_event_ids=list(EVENT_IDS),increment_records=accepted,
        full_catalog=deepcopy(original['events']),original_catalog=deepcopy(docs['v4_policy']['inputs']['dataset']),
        original_catalog_sha256=scope.canonical_sha256(original),failed_population_inputs=deepcopy(policy['inputs']),
        full_census_passed=False,old_admission_regressions=regressions,old_admissions_recertified=False,
        original_masks_modified=False,ratio_or_chain_admitted=False,strategy_registration_authorized=False,
        assumptions=dict(same_report_literal_currency=True,annual_start_inherited_contiguity=True,
            next_pinned_session_after_date_only_release=True,max_age_days=550,latest_known_before_validity_required=True,
            older_report_fallback_permitted=False,first_public_version_verified=False,complete_correction_history_verified=False,
            certified_daily_security_master=False,currency_iso_certified=False),
        counts=dict(accepted_events=3,accepted_fields=9,retained_catalog_events=44,new_assess_calls=0,
            ratios_computed=0,features_computed=0,returns_computed=0,NPZ_decodes=0,models_fitted=0))
    out=Path(root)/OUT; need(out.resolve()==out,'No output symlink'); out.mkdir(parents=True,exist_ok=False)
    raw=encoded(result)
    with (out/'INCREMENT.json').open('xb') as f: f.write(raw); f.flush(); os.fsync(f.fileno())
    manifest=dict(schema='roa_field_scope270_increment_manifest_v1',policy=pref,
        increment_accepted=True,full_census_accepted=False,output_sha256={'INCREMENT.json':digest(raw)},
        input_sha256=dict(ev.verified),code_sha256=deepcopy(policy['code_sha256']),
        old_runtime_verified=False,historical_dependency_edges=deepcopy(docs['v4_manifest']['historical_dependency_edges']))
    with (out/'MANIFEST.json').open('xb') as f: f.write(encoded(manifest)); f.flush(); os.fsync(f.fileno())
    return dict(increment_admitted=3,accepted_fields=9,full_catalog_events=44,full_census_passed=False,
        new_assess_calls=0,manifest_sha256=digest(encoded(manifest)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy',required=True); parser.add_argument('--policy-sha',required=True)
    args=parser.parse_args(); print(json.dumps(run(ROOT,args.policy,args.policy_sha),sort_keys=True))
