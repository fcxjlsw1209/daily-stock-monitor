"""Explicit code-only amendment for the registered157 zero-budget adapter call.

No producer imports occur during import, digest verification or audit binding.
The original PLAN, proposal, rules and implementation remain immutable.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import tempfile

ROOT = Path(__file__).resolve().parents[1]
OUT = 'outputs/monthly_filing_association157'
PLAN_SHA = '43adc171b3ee82cd952544b75db63fffaf5b87bd25b96a03e709fd2ff22356db'
FAILURE = OUT + '/PREPARE_INTERFACE_FAILURE_01.json'
PREREVIEW = OUT + '/REPAIR_PREREVIEW_01.json'
AMENDMENT = OUT + '/AMENDMENT_01.json'
FILES = tuple('work/' + name for name in (
    'monthly_filing_association157_repaired_runner.py',
    'test_monthly_filing_association157_repaired_runner.py',
    'monthly_filing_association157_repair.py',
    'test_monthly_filing_association157_repair.py',
    'monthly_filing_association157_repair_review.py',
    'test_monthly_filing_association157_repair_review.py'))
REAL_ARTIFACTS = ('DATA.npz', 'SCORES.npz', 'META.json', 'FEATURE_AUDIT.json',
    'PRETRAIN_ORDERS.json', 'PRICE_PARITY.json', 'CACHE.json', 'PREPARE_FAILURE.json',
    'INDEPENDENT_PREPARE.json', 'TRAIN_TRIALS.json', 'TRAIN_OPEN_TRIALS.json',
    'WALKFORWARD.json', 'FROZEN.json', 'DIAGNOSE_STARTED.json', 'RESULTS.json',
    'VALIDATION.json', 'INDEPENDENT_OUTCOME.json', 'COMPLETION.json')


def fail_closed(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def local(root, name):
    root = Path(root).resolve()
    p = Path(name)
    fail_closed(not p.is_absolute() and '..' not in p.parts, 'Relative local repair path required')
    target = (root / p).resolve()
    fail_closed(target.is_relative_to(root), 'Repair path escaped local root')
    return target


def digest(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def reference(root, name):
    return dict(path=name, sha256=sha(local(root, name)))


def verify_reference(root, ref, name):
    fail_closed(type(ref) is dict and set(ref) == {'path', 'sha256'} and ref['path'] == name
                and digest(ref['sha256']), 'Exact repair reference required: ' + name)
    fail_closed(sha(local(root, name)) == ref['sha256'], 'Repair evidence changed: ' + name)
    return read(local(root, name))


def implementation_hashes(root):
    return {name: sha(local(root, name)) for name in FILES}


def write_once(path, value):
    path = Path(path)
    raw = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.repair157-', delete=False) as f:
            temp = Path(f.name); f.write(raw); f.flush(); os.fsync(f.fileno())
        os.link(str(temp), str(path))
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def unprepared(root):
    found = [name for name in REAL_ARTIFACTS if (Path(root)/OUT/name).exists()]
    fail_closed(not found, 'Real artifacts already exist: ' + repr(found))


def verify_plan(root, plan_sha):
    fail_closed(plan_sha == PLAN_SHA and sha(Path(root)/OUT/'PLAN.json') == PLAN_SHA,
                'Exact original registered PLAN required')
    plan = read(Path(root)/OUT/'PLAN.json')
    fail_closed(plan['round'] == 157 and plan['status'] == 'REGISTERED' and plan['dose'] == .01,
                'Original registration contract')
    return plan


def validate_failure(doc):
    fail_closed(doc.get('original_plan_sha256') == PLAN_SHA and doc.get('exit_code') == 1,
                'Original failed attempt required')
    activity = doc.get('activity', {})
    fail_closed(activity.get('real_features_calculated_in_memory') is True
                and activity.get('real_nuisance_estimates_attempted') is True
                and activity.get('exact_estimate_counts_persisted') is False
                and activity.get('strategy_returns_calculated') == 0
                and activity.get('prepared_artifacts_published') == 0,
                'Do not erase actual failed-attempt activity')


def verify_amendment(root, plan_sha, amendment_sha):
    """Read-only closure check, safe in independent numerical audit processes."""
    root = Path(root).resolve(); plan = verify_plan(root, plan_sha)
    fail_closed(digest(amendment_sha) and sha(root/AMENDMENT) == amendment_sha, 'Amendment digest changed')
    doc = read(root/AMENDMENT)
    expected = dict(schema_version=1, round=157, kind='zero_budget_adapter_dummy_only',
        original_plan_sha256=PLAN_SHA, same_experiment=True, additional_strategy_trials=0,
        formula_changed=False, parameters_changed=False)
    fail_closed(all(type(doc.get(k)) is type(v) and doc[k] == v for k,v in expected.items()),
                'Code-only amendment contract changed')
    fail_closed(doc.get('implementation_hashes') == implementation_hashes(root), 'Amended implementation changed')
    failure = verify_reference(root, doc['failure_record'], FAILURE); validate_failure(failure)
    review = verify_reference(root, doc['independent_prereview'], PREREVIEW)
    validate_prereview(root, review)
    fail_closed(doc['implementation_hashes'] == review['implementation_hashes'], 'Unreviewed repair code')
    times = [datetime.fromisoformat(v.replace('Z','+00:00')) for v in
             (plan['registered_utc'], failure['recorded_utc'], review['reviewed_utc'], doc['registered_utc'])]
    fail_closed(all(t.tzinfo is not None for t in times) and times == sorted(times)
                and times[-1] <= datetime.now(timezone.utc), 'Repair registration chronology')
    return doc


def validate_prereview(root, review):
    expected = dict(passed=True, synthetic_only=True, semantics_preserved=True,
                    original_plan_sha256=PLAN_SHA, actual_returns_calculated=0)
    fail_closed(all(type(review.get(k)) is type(v) and review[k] == v for k,v in expected.items()),
                'Independent synthetic repair review required')
    fail_closed(review.get('implementation_hashes') == implementation_hashes(root), 'Repair prereview code changed')
    failure = verify_reference(root, review['failure_record'], FAILURE); validate_failure(failure)
    tests = review.get('tests')
    fail_closed(type(tests) is dict and set(tests) == {'runner', 'lifecycle', 'independent'},
                'All three repair test suites required')
    for name, row in tests.items():
        fail_closed(row.get('passed') is True and type(row.get('count')) is int and row['count'] > 0
                    and re.search(r'Ran '+str(row['count'])+r' tests? in ', row.get('log',''))
                    and row['log'].rstrip().endswith('OK'), 'Invalid repair test evidence: '+name)
        source = {'runner': FILES[1], 'lifecycle': FILES[3], 'independent': FILES[5]}[name]
        fail_closed(row.get('source_sha256') == sha(local(root,source)), 'Repair test source changed')


def verify_repair_report(root, report, plan_sha, amendment_sha):
    amendment = verify_amendment(root, plan_sha, amendment_sha)
    fail_closed(report.get('amendment_sha256') == amendment_sha
                and report.get('original_plan_sha256') == PLAN_SHA,
                'Numerical audit lacks exact amended-execution provenance')
    fail_closed(report.get('repair_source_hashes') == amendment['implementation_hashes']
                and report.get('repair_review_sha256') == sha(Path(root)/FILES[4]),
                'Numerical repair audit implementation mismatch')
    artifacts = report.get('artifact_hashes', {})
    for name in (AMENDMENT, FAILURE, PREREVIEW):
        fail_closed(artifacts.get(str(local(root,name))) == sha(local(root,name)),
                    'Missing repair evidence in audit: '+name)
    return amendment


def record_failure(root):
    root = Path(root).resolve(); plan = verify_plan(root, PLAN_SHA); unprepared(root)
    # Full original closure includes frozen code/runtime/source package; no replay.
    import monthly_filing_association157_lifecycle as old
    reader = old.old.Reader(root); reader.verify(plan['hashes'])
    value = dict(schema_version=1, original_plan_sha256=PLAN_SHA,
        recorded_utc=datetime.now(timezone.utc).isoformat(), exit_code=1,
        evidence_origin='Prior goal-turn exec tool transcript, session72221; this is a later observation, not a recreated process log.',
        command='./run_local.sh work/monthly_filing_association157_runner.py prepare --plan '+OUT+'/PLAN.json --plan-sha '+PLAN_SHA,
        terminal_error='ValueError: finite hand factors must be bounded ranks in [-.5, .5]',
        call_path=['monthly_filing_association157_runner.Runner.prepare:panels',
                   'monthly_revenue_persistence156_runner.panels:choose',
                   'monthly_stock_comp149_runner.corrected',
                   'incumbent_adapter.score_and_choose:_inputs'],
        activity=dict(real_features_calculated_in_memory=True, real_nuisance_estimates_attempted=True,
                      exact_estimate_counts_persisted=False, strategy_returns_calculated=0,
                      prepared_artifacts_published=0, orders_attempted=True, complete_order_count_known=False),
        unchanged_registration_artifacts={str(p.relative_to(root)):sha(p) for p in sorted((root/OUT).iterdir()) if p.is_file()},
        resolution='Preserve H and additive .01H; supply a bounded neutral dummy only to the zero-budget legacy adapter after the pre-score correction.')
    validate_failure(value); write_once(root/FAILURE, value)
    return reference(root, FAILURE)


def register(root):
    root = Path(root).resolve(); plan = verify_plan(root, PLAN_SHA); unprepared(root)
    import monthly_filing_association157_lifecycle as old
    with old.state_lock(root):
        reader=old.old.Reader(root); reader.verify(plan['hashes'])
        review=read(root/PREREVIEW); validate_prereview(root,review)
        state,before=old.state_snapshot(root/old.STATE); old.predecessor(state)
        fail_closed(state.get('registered_queue') == [old.queue_entry(plan,PLAN_SHA)], 'Original single157 queue required')
        doc=dict(schema_version=1, round=157, kind='zero_budget_adapter_dummy_only',
            registered_utc=datetime.now(timezone.utc).isoformat(), original_plan_sha256=PLAN_SHA,
            same_experiment=True, additional_strategy_trials=0, formula_changed=False, parameters_changed=False,
            implementation_hashes=implementation_hashes(root), failure_record=reference(root,FAILURE),
            independent_prereview=reference(root,PREREVIEW), continuation_before_sha256=before,
            scope='Bounded neutral dummy only after .01H already entered pre-score; all original gates, data, arms, periods and accounting retained.')
        write_once(root/AMENDMENT,doc); amended=sha(root/AMENDMENT)
        verify_amendment(root,PLAN_SHA,amended)
        state.update(current_status='registered_code_repair_before_reprepare',
            active_execution_amendment=reference(root,AMENDMENT), last_goal_turn_classification='progress',
            updated_utc=doc['registered_utc'])
        old.update_state(root/old.STATE,before,state)
    return dict(registered=True, amendment=reference(root,AMENDMENT), original_plan_sha256=PLAN_SHA)


def complete(root, amendment_sha):
    root=Path(root).resolve(); amendment=verify_amendment(root,PLAN_SHA,amendment_sha)
    for name in ('INDEPENDENT_PREPARE.json','INDEPENDENT_OUTCOME.json'):
        verify_repair_report(root,read(root/OUT/name),PLAN_SHA,amendment_sha)
    import monthly_filing_association157_lifecycle as old
    with old.state_lock(root):
        # Retain every frozen completion check, including full21444 archive scan.
        result=old.complete(root,PLAN_SHA)
        verify_amendment(root,PLAN_SHA,amendment_sha)
        value=dict(schema_version=1, round=157, same_experiment=True, additional_strategy_trials=0,
            completed_utc=datetime.now(timezone.utc).isoformat(), original_completion=reference(root,OUT+'/COMPLETION.json'),
            amendment=reference(root,AMENDMENT), failure_record=amendment['failure_record'],
            independent_prereview=amendment['independent_prereview'],
            independent_prepare=reference(root,OUT+'/INDEPENDENT_PREPARE.json'),
            independent_outcome=reference(root,OUT+'/INDEPENDENT_OUTCOME.json'),
            preserved_original_plan_sha256=PLAN_SHA, implementation_hashes=amendment['implementation_hashes'])
        name=OUT+'/REPAIR_COMPLETION_01.json'
        if (root/name).exists():
            prior=read(root/name)
            fail_closed({k:v for k,v in prior.items() if k!='completed_utc'} ==
                        {k:v for k,v in value.items() if k!='completed_utc'}, 'Conflicting repair completion')
        else: write_once(root/name,value)
        state,before=old.state_snapshot(root/old.STATE)
        state['latest_execution_repair_completion']=reference(root,name)
        old.update_state(root/old.STATE,before,state)
    return dict(result=result, repair_completion=reference(root,name))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('record_failure','register','complete'))
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--amendment-sha')
    a=p.parse_args()
    result=(record_failure(a.root) if a.phase=='record_failure' else register(a.root)
            if a.phase=='register' else complete(a.root,a.amendment_sha))
    print(json.dumps(result,ensure_ascii=False,sort_keys=True))


if __name__=='__main__': main()
