"""Exclusive metadata/hash registration; never computes new state or returns."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import re
import sys
import tempfile

import monthly_same_report144_runner as r

ROOT = Path(__file__).resolve().parents[1]
PROPOSAL_SHA = '508c8534ae0ad253845453360f518e69cce487bd827e46a7793fb56180411d0a'
BASE_PLAN_SHA = '2d73a3afb4745f5106f6cc870d0c25a775721181cd3c99ad78a7e1f56b58500b'
LATEST_COMPLETION_SHA = 'b0251e542d0c88431bb0097e823e127c8cae7094b8af95de4a65dc4acdcb2c24'
CALENDAR_SHA = '4b8f32ad8a555d6537d7cf3bd0f831e64fbb2de26d033c8daaf68590fcd4a6ea'

STATE_RELATIVE = 'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
EXPERIMENT = 'monthly_same_report144'


SUITE_SOURCES = {
    'factor': 'test_monthly_same_report144_factor.py',
    'runner': 'test_monthly_same_report144_runner.py',
    'prepare_review': 'test_monthly_same_report144_prepare_review.py',
    'outcome_review': 'test_monthly_same_report144_outcome_review.py',
    'registration': 'test_register_monthly_same_report144.py',
    'completion': 'test_complete_monthly_same_report144.py',
}


def validate_tests(tests, add):
    """Bind every reported source/log and require each new lifecycle suite."""
    r.require(tests.get('passed') is True and isinstance(tests.get('suites'), list),
              'Passed prospective synthetic validation required')
    found = {}
    for suite in tests['suites']:
        name = suite.get('name')
        r.require(isinstance(name, str) and name not in found and suite.get('passed') is True,
                  'Unique passed synthetic suite required')
        count = suite.get('count')
        r.require(type(count) is int and count > 0, 'Positive synthetic count required')
        source = add(suite['source']['path'], suite['source']['sha256'])
        log = add(suite['log']['path'], suite['log']['sha256'])
        if name in SUITE_SOURCES:
            r.require(source.name == SUITE_SOURCES[name], 'Wrong suite source identity')
        text = log.read_text()
        r.require(re.search(r'Ran '+str(count)+r' tests? in ', text) and text.rstrip().endswith('OK'),
                  'Synthetic log does not report success')
        found[name] = count
    r.require(set(SUITE_SOURCES) <= set(found), 'All new synthetic suites required')
    r.require(type(tests.get('test_count')) is int and tests['test_count'] == sum(found.values()),
              'Synthetic total count mismatch')
    return found


def continuation_snapshot(root):
    raw = (Path(root)/STATE_RELATIVE).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def require_predecessor(state):
    completed = state.get('completed')
    r.require(isinstance(completed, list) and len(completed) == 143 and
              all(isinstance(x, dict) and type(x.get('trials')) is int and
                  x['trials'] >= 0 for x in completed) and
              sum(x['trials'] for x in completed) == 929,
              'Unexpected continuation predecessor')
    r.require(completed[-1].get('directory') == 'monthly_paired_target143',
              'Round143 must be the immediate predecessor')
    r.require(not any(x.get('directory') == EXPERIMENT for x in completed),
              'Round144 already in predecessor history')


def enqueue_registered_plan(root, plan_path, plan_sha):
    """Commit the queue after an exclusive PLAN write, or resume that commit.

    The state precondition belongs outside long-lived input hashes because this
    function intentionally changes the state. All central writers must remain
    serialized; the final digest comparison detects intervening changes but is
    not a cross-process compare-and-swap primitive.
    """
    root, plan_path = Path(root).resolve(), Path(plan_path).resolve()
    r.require(plan_path == root/'outputs'/EXPERIMENT/'PLAN.json',
              'Unexpected registration PLAN path')
    raw = plan_path.read_bytes()
    r.require(hashlib.sha256(raw).hexdigest() == plan_sha, 'Changed registration PLAN')
    plan = json.loads(raw)
    r.require(plan['round'] == 144 and plan['status'] == 'REGISTERED' and
              plan['arms'] == list(r.NAMES) and plan['primary_challenger'] == r.NAMES[2] and
              Path(plan['output_directory']).resolve() == plan_path.parent,
              'Wrong registered experiment')
    precondition = plan['registration_precondition']
    r.require(precondition['path'] == STATE_RELATIVE, 'Wrong continuation precondition')
    r.require(bool(plan['hashes']), 'Missing registration input hashes')
    for name, expected in plan['hashes'].items():
        r.require(r.sha(root/name) == expected, 'Changed registered input: '+name)
    entry = dict(round=144, directory=EXPERIMENT,
                 plan=dict(path=str(plan_path.relative_to(root)), sha256=plan_sha),
                 registered_utc=plan['registered_utc'])
    state, state_sha = continuation_snapshot(root)
    require_predecessor(state)
    if state['registered_queue'] == [entry]:
        return False
    r.require(not state['registered_queue'], 'Conflicting registered queue')
    r.require(state_sha == precondition['sha256'],
              'Continuation changed since PLAN construction; do not overwrite it')
    state['registered_queue'] = [entry]
    state.update(current=EXPERIMENT, current_status='registered_before_actual_preparation',
                 updated_utc=plan['registered_utc'])
    destination = root/STATE_RELATIVE
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                prefix='.CONTINUATION_STATUS.', suffix='.tmp',
                dir=str(destination.parent), delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(state, stream, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        r.require(r.sha(destination) == state_sha, 'Continuation changed before queue commit')
        r.require(r.sha(plan_path) == plan_sha, 'PLAN changed before queue commit')
        os.replace(str(temporary), str(destination))
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return True


def admitted_source_closure(root, audit_path, dataset_path, add):
    root = Path(root).resolve()
    folder = root/'outputs/monthly_same_report144_admission'
    r.require(Path(audit_path).resolve() == folder/'AUDIT.json' and
              Path(dataset_path).resolve() == folder/'DATA.json', 'Exact admitted source paths required')
    audit = r.read(audit_path)
    plan = r.read(folder/'PLAN.json')
    manifest = r.read(folder/'MANIFEST.json')
    tests = r.read(folder/'TESTS.json')
    r.require(audit['source_hashes'] == plan['source_hashes'] and
              audit['source_code_hashes'] == plan['source_code_hashes'] == tests['source_code_hashes'],
              'Source admission closure mismatch')
    r.require(audit['plan_sha256'] == manifest['plan_sha256'] == r.sha(folder/'PLAN.json') and
              audit['policy_sha256'] == r.sha(folder/'POLICY.json'), 'Source policy/plan binding')
    r.require(manifest['output_hashes'] == {n:r.sha(folder/n) for n in ('DATA.json','AUDIT.json')},
              'Source manifest output binding')
    r.require(tests.get('passed') is True and tests.get('synthetic_only') is True and
              type(tests.get('tests_run')) is int and tests['tests_run'] > 0 and
              tests.get('actual_financial_calculations') == 0 and tests.get('errors') == 0 and
              tests.get('failures') == 0 and tests.get('log','').rstrip().endswith('OK') and
              ('Ran '+str(tests['tests_run'])+' tests in ') in tests['log'], 'Source synthetic tests')
    closure = {}
    for source in (audit['source_hashes'], audit['source_code_hashes']):
        r.require(isinstance(source,dict) and bool(source), 'Nonempty admitted source map required')
        for name, digest in source.items():
            path = add(name,digest)
            r.require(str(path) not in closure or closure[str(path)] == digest, 'Source digest conflict')
            closure[str(path)] = digest
    for name in ('PLAN.json','POLICY.json','DATA.json','AUDIT.json','MANIFEST.json','TESTS.json'):
        path = add(folder/name)
        closure[str(path)] = r.sha(path)
    return closure


def build(root, prepare_review, outcome_review, admitted_dataset=None, source_admission=None):
    root = Path(root).resolve()
    out = root/'outputs/monthly_same_report144'
    r.require(not (out/'PLAN.json').exists(), 'Registration already exists')
    r.require(re.fullmatch('[0-9a-f]{64}',PROPOSAL_SHA) is not None and r.PROPOSAL_SHA==PROPOSAL_SHA,
              'Final prospective proposal must be bound before registration')
    r.require(admitted_dataset is not None and source_admission is not None,
              'Explicit admitted dataset and source-admission receipt required')
    hashes = {}

    def add(path, expected=None):
        p = Path(path)
        p = (p if p.is_absolute() else root/p).resolve()
        h = r.sha(p)
        r.require(expected is None or h == expected, 'Changed input: '+str(p))
        r.require(str(p) not in hashes or hashes[str(p)] == h, 'Concurrent input change')
        hashes[str(p)] = h
        return p

    def mapping(m):
        r.require(isinstance(m, dict) and bool(m), 'Nonempty dependency map required')
        for path, digest in m.items():
            add(path, digest)

    state, state_sha = continuation_snapshot(root)
    require_predecessor(state)
    r.require(not state['registered_queue'], 'Unexpected continuation state')
    add('outputs/monthly_paired_target143/COMPLETION.json', LATEST_COMPLETION_SHA)
    base_path = add('outputs/monthly_paired_target143/PLAN.json', BASE_PLAN_SHA)
    old = r.read(base_path)
    mapping(old['hashes'])
    r.require(len(old['raw_source_hashes']) == 64, 'Original raw64 scope changed')
    mapping(old['raw_source_hashes'])
    mapping(r.source_files(root))
    add(Path(__file__))
    add('work/test_register_monthly_same_report144.py')
    # Bind the new closure implementation and its prospective validation before
    # any real feature or result is computed.
    add('work/complete_monthly_same_report144.py')
    add('work/test_complete_monthly_same_report144.py')
    tests_path = add('outputs/monthly_same_report144/TESTS.json')
    tests = r.read(tests_path)
    validate_tests(tests, add)
    calendar_path = add('outputs/monthly_forward_guidance_preflight_v1/CALENDAR_PLAN.json', CALENDAR_SHA)
    mapping(r.read(calendar_path)['bindings'])
    inputs = {key: Path(old['inputs'][key]['path']) for key in
              ('meta', 'policy', 'original_orders', 'prices', 'protected_manifest')}
    inputs['proposal'] = add('work/monthly_same_report144_proposal.json', PROPOSAL_SHA)
    inputs['sessions'] = add('outputs/monthly_forward_guidance_preflight_v1/SESSIONS.json')
    r.require(r.read(inputs['sessions'])['plan_sha256'] == CALENDAR_SHA, 'Session provenance mismatch')
    source_contract = r.read(inputs['proposal'])['source_contract']
    inputs['admitted_dataset']=add(admitted_dataset, source_contract['dataset_sha256'])
    inputs['source_admission']=add(source_admission, source_contract['audit_sha256'])
    admission=r.read(inputs['source_admission'])
    r.validate_source_admission(admission, dataset_sha=r.sha(inputs['admitted_dataset']))
    fundamental = admitted_source_closure(root, inputs['source_admission'], inputs['admitted_dataset'], add)
    r.require(fundamental.get(str(inputs['sessions'])) == r.sha(inputs['sessions']),
              'Source and strategy session calendars must match')
    r.require(set(inputs) == set(r.INPUTS), 'Runner/registrar input schema mismatch')
    refs = {key: dict(path=str(add(path)), sha256=r.sha(path)) for key, path in inputs.items()}
    independent = {}
    reviews = []
    for name, path in (('prepare', prepare_review), ('outcome', outcome_review)):
        path = add(path)
        review = r.read(path)
        r.require(review.get('passed') is True, 'Incomplete '+name+' prereview')
        r.require(review.get('round') == 144 and
                  review.get('stage') == 'preregistration_'+name+'_review' and
                  review.get('proposal_sha256') == PROPOSAL_SHA and
                  review.get('synthetic_only') is True and
                  review.get('real_144_factor_or_return_calculations') == 0,
                  'Wrong prereview stage, scope or proposal')
        mapping(review['source_hashes'])
        for section in ('input_hashes', 'implementation_hashes'):
            if review.get(section):
                mapping(review[section])
        expected = 'monthly_same_report144_'+name+'_review.py'
        r.require(any(Path(p).name == expected for p in review['source_hashes']), 'Reviewer implementation absent')
        independent[name] = review['source_hashes']
        reviews.append(str(path))
    return dict(round=144, status='REGISTERED', registered_utc=datetime.now(timezone.utc).isoformat(),
        registration_precondition=dict(path=STATE_RELATIVE, sha256=state_sha),
        arms=list(r.NAMES), primary_challenger=r.NAMES[2], output_directory=str(out),
        inputs=refs, hashes=hashes, raw_source_hashes=old['raw_source_hashes'],
        fundamental_source_hashes=fundamental,
        independent_prepare_source_hashes=independent['prepare'],
        independent_outcome_source_hashes=independent['outcome'], prerequisite_reviews=reviews,
        runtime=dict(python=sys.version, numpy=r.np.__version__, pandas=r.pd.__version__),
        tolerances=dict(r.TOLERANCES), actual_model_fits_before_registration=0,
        actual_feature_calculations_before_registration=0, actual_returns_before_registration=0,
        new_market_downloads=0, no_observer_writes=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare-review')
    p.add_argument('--outcome-review')
    p.add_argument('--admitted-dataset')
    p.add_argument('--source-admission')
    p.add_argument('--resume-plan-sha', help='Complete only the queue commit for this exact existing PLAN')
    a = p.parse_args()
    plan_path = ROOT/'outputs'/EXPERIMENT/'PLAN.json'
    if a.resume_plan_sha:
        if a.prepare_review or a.outcome_review or a.admitted_dataset or a.source_admission:
            p.error('--resume-plan-sha cannot be combined with prereviews or source-admission arguments')
        plan_sha = a.resume_plan_sha
    else:
        if not all((a.prepare_review,a.outcome_review,a.admitted_dataset,a.source_admission)):
            p.error('new registration requires both prereviews, admitted dataset and source-admission receipt')
        plan = build(ROOT,a.prepare_review,a.outcome_review,a.admitted_dataset,a.source_admission)
        plan_path.parent.mkdir(exist_ok=True, parents=True)
        r.dump(plan_path, plan)
        plan_sha = r.sha(plan_path)
    committed = enqueue_registered_plan(ROOT, plan_path, plan_sha)
    plan = r.read(plan_path)
    print(json.dumps(dict(path=str(plan_path), sha256=plan_sha,
                         dependencies=len(plan['hashes']), queue_committed=committed,
                         registered_queue_contains_round=144)))


if __name__ == '__main__':
    main()
