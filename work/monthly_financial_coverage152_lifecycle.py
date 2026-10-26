"""Write-once register/enqueue/close for the fixed five-arm coverage experiment.

Only metadata/hash validation is performed here. Source admission and numerical
preparation/outcome audits must already exist; this module never fills a gap.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import re
import sys
import tempfile

import monthly_financial_coverage152_runner as r
import complete_monthly_same_report145 as old

ROOT = Path(__file__).resolve().parents[1]
STUDY = 'monthly_financial_coverage152'
STATE = 'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
LEGACY_PLAN = '9b0095d8112cb4579e63f85e2f6b837fe8d7ebb5eef5e399fd926f41c682735d'
PREVIOUS_PLAN = '6f8aa84dadee4d2499e683ecf93a6e82b09ea4f54ec05df5047ef19b8bafcac4'
LEGACY_DONE = '0811e2607b350ce4df89c32616548e17c17e65a2c752c916e67580633343f7c6'
PREVIOUS_DONE = '7dd764a81ff112d2706e4f4216685832830e899ca19af65b77e5dbc5a6648cf5'
PROPOSAL = 'work/monthly_financial_coverage152_proposal.json'
PRE_REVIEW = 'work/monthly_financial_coverage152_prepare_review_SYNTHETIC.json'
OUT_REVIEW = 'work/monthly_financial_coverage152_outcome_prereview.json'
TEST_SUITES = {'factor', 'selection', 'runner', 'prepare_review', 'outcome_review', 'lifecycle'}
require = r.require


def write(path, value):
    with Path(path).open('xb') as stream:
        stream.write(old.encoded(value))


def state_snapshot(path):
    raw = Path(path).read_bytes()
    return old.decode(raw), hashlib.sha256(raw).hexdigest()


@contextmanager
def state_lock(root):
    with (Path(root)/'outputs/monthly_target70_research/.lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def predecessor(state):
    items = state['completed']
    require(len(items) == 151 and sum(item['trials'] for item in items) == 957,
            'Expected151 completed rounds/957 trials')
    require(items[-1]['directory'] == 'monthly_financial_coverage151' and
            items[-1]['completion']['sha256'] == PREVIOUS_DONE, 'Wrong preceding completion')
    require(not any(item['directory'] == STUDY for item in items), 'Already completed152')


def update_state(path, expected, value):
    require(r.sha(path) == expected, 'Concurrent continuation update')
    backup = path.parent/('PRE152_'+expected+'.json')
    if not backup.exists():
        with backup.open('xb') as stream:
            stream.write(path.read_bytes())
    else:
        require(r.sha(backup) == expected, 'Conflicting original-state backup')
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(old.encoded(value))
        stream.flush()
        os.fsync(stream.fileno())
    try:
        require(r.sha(path) == expected, 'Concurrent continuation commit')
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def preserve_completion(path, done):
    if Path(path).exists():
        saved = r.read(path)
        require({k:v for k,v in saved.items() if k != 'completed_utc'} ==
                {k:v for k,v in done.items() if k != 'completed_utc'},
                'Existing completion conflicts with revalidated evidence')
        return saved
    write(path, done)
    return done


def validate_tests(doc, reader):
    require(doc.get('passed') is True and doc.get('synthetic_only') is True,
            'Prospective synthetic tests required')
    names, count = set(), 0
    for suite in doc['suites']:
        name = suite['name']
        require(name not in names and type(suite['count']) is int and suite['count'] > 0,
                'Distinct positive synthetic suites required')
        names.add(name)
        for key in ('source', 'log'):
            ref = suite[key]
            require(reader.hash(ref['path']) == ref['sha256'], 'Changed synthetic '+key)
        log = reader.path(suite['log']['path']).read_text()
        require(re.search(r'Ran '+str(suite['count'])+r' tests? in ', log) and
                log.rstrip().endswith('OK'), 'Synthetic test failure')
        count += suite['count']
    require(names == TEST_SUITES and type(doc['test_count']) is int and doc['test_count'] == count,
            'Complete six-suite prospective scope required')


def source_pins(reader, value):
    if isinstance(value, dict):
        return reader.verify(value)
    require(isinstance(value, (list, tuple)) and value, 'Reviewer source list required')
    return {str(reader.path(path)): reader.hash(path) for path in value}


def refs(reader, supplied, expected):
    require(set(supplied) == set(expected), 'Exact source reference keys required')
    result = {}
    for key, name in expected.items():
        ref = supplied[key]
        require(set(ref) == {'path', 'sha256'} and reader.path(ref['path']) == reader.root/name,
                'Exact source reference path required')
        require(reader.hash(ref['path']) == ref['sha256'], 'Source reference hash mismatch')
        result[key] = reader.read(ref['path'])
    return result


def verify_legacy(reader, supplied):
    objects = refs(reader, supplied, {k:r.LEGACY_OUT+'/'+v for k,v in r.LEGACY_FILES.items()})
    p, cache, done, review = (objects[k] for k in ('plan', 'cache', 'completion', 'outcome'))
    require(supplied['plan']['sha256'] == LEGACY_PLAN and
            supplied['completion']['sha256'] == LEGACY_DONE, 'Exact finalized149 required')
    require(p.get('round') == done.get('round') == 149 and p.get('status') == 'REGISTERED',
            'Finalized predecessor metadata required')
    require(done.get('plan_sha256') == review.get('plan_sha256') == LEGACY_PLAN and
            done.get('independent_outcome_sha256') == supplied['outcome']['sha256'] and
            review.get('cache_sha256') == supplied['cache']['sha256'] and
            review.get('passed') is True and review.get('stage') == 'evaluate',
            'Finalized149 audit/cache chain required')
    reader.verify(p['hashes'])
    require(len(p['raw_source_hashes']) == 64, 'Original64 raw price sources required')
    reader.verify(p['raw_source_hashes'])
    require(set(cache) == set(r.CACHE_FILES), 'Exact149 cache scope required')
    cachepins = reader.verify(cache, reader.root/r.LEGACY_OUT)
    audited = reader.verify(review['artifact_hashes'])
    require(all(audited.get(path) == digest for path,digest in cachepins.items()),
            'Original cache not independently audited')
    require(review.get('orders_sha256') == cache['PRETRAIN_ORDERS.json'], 'Original orders binding mismatch')
    reader.verify(review['source_hashes'])
    return p


def same_references(reader, supplied, expected, message):
    require(set(supplied) == set(expected), message)
    for key in expected:
        require(reader.path(supplied[key]['path']) == reader.path(expected[key]['path']) and
                supplied[key]['sha256'] == expected[key]['sha256'], message)


def verify_prior(reader, supplied, legacy):
    objects = refs(reader, supplied, {k:r.PRIOR_OUT+'/'+v for k,v in r.LEGACY_FILES.items()})
    p, cache, done, review = (objects[k] for k in ('plan', 'cache', 'completion', 'outcome'))
    require(supplied['plan']['sha256'] == PREVIOUS_PLAN and
            supplied['completion']['sha256'] == PREVIOUS_DONE, 'Exact finalized151 required')
    require(p.get('round') == done.get('round') == 151 and p.get('status') == 'REGISTERED',
            'Finalized151 metadata required')
    require(done.get('plan_sha256') == review.get('plan_sha256') == PREVIOUS_PLAN and
            done.get('independent_outcome_sha256') == supplied['outcome']['sha256'] and
            review.get('cache_sha256') == supplied['cache']['sha256'] and
            review.get('passed') is True and review.get('stage') == 'evaluate',
            'Finalized151 audit/cache chain required')
    same_references(reader, p['legacy149'], legacy, 'Prior151 legacy149 source changed')
    reader.verify(p['hashes'])
    require(len(p['raw_source_hashes']) == 64, 'Original64 prior151 raw sources required')
    reader.verify(p['raw_source_hashes'])
    require(set(cache) == set(r.CACHE_FILES), 'Exact151 cache scope required')
    cachepins = reader.verify(cache, reader.root/r.PRIOR_OUT)
    audited = reader.verify(review['artifact_hashes'])
    require(all(audited.get(path) == digest for path,digest in cachepins.items()),
            'Prior151 cache not independently audited')
    require(review.get('orders_sha256') == cache['PRETRAIN_ORDERS.json'], 'Prior151 orders binding mismatch')
    reader.verify(review['source_hashes'])
    return p


def verify_coverage(reader, supplied, previous, batch):
    require(batch in r.COVERAGE_REFS, 'Unknown source batch')
    objects = refs(reader, supplied, r.COVERAGE_REFS[batch])
    data, admission, review = (objects[k] for k in ('dataset', 'admission', 'review'))
    inputs = reader.verify(admission['input_hashes'])
    reader.verify(objects['manifest']['input_hashes'])
    reviewed = reader.verify(review['input_hashes'])
    source_plan_path = reader.root/r.SOURCE_BATCH[batch]/'PLAN.json'
    extraction_path = reader.root/r.SOURCE_BATCH[batch]/'EXTRACTIONS.json'
    source_plan, extraction = reader.read(source_plan_path), reader.read(extraction_path)
    require(data['source_plan_sha256'] == reader.hash(source_plan_path) and
            review['extractions_sha256'] == reader.hash(extraction_path), 'Source extraction linkage mismatch')
    for path in (source_plan_path, extraction_path,
                 reader.path(supplied['policy']['path']), reader.path(supplied['review']['path'])):
        require(inputs.get(str(path)) == reader.hash(path), 'Admission omitted mandatory dependency')
    for path in (source_plan_path, extraction_path):
        require(reviewed.get(str(path)) == reader.hash(path), 'Physical review omitted extraction binding')
    reader.verify(source_plan['inputs'])
    reader.verify(source_plan['code'])
    census_path = reader.root/'outputs/monthly_financial_coverage_census_v1/CENSUS.json'
    require(inputs.get(str(census_path)) == reader.hash(census_path), 'Complete candidate census omitted')
    r.validate_catalog(data, reader.read(census_path), batch=batch)
    original = data['original_four_issuer_reference']
    for field, reference in (('path', 'sbc_dataset'), ('admission_path', 'sbc_admission')):
        digest_key = 'sha256' if field == 'path' else 'admission_sha256'
        require(reader.path(original[field]) == reader.path(previous[reference]['path']) and
                original[digest_key] == previous[reference]['sha256'] and
                reader.hash(original[field]) == original[digest_key], 'Original149 source evidence rewired')
    old_data = reader.read(original['path'])
    r.validate_source(data, admission, objects['policy'], review, objects['manifest'],
                      objects['coverage'], supplied, old_data, extraction, source_plan, batch=batch)
    for event in data['events']:
        path = reader.path(event['source']['path'])
        require(reader.hash(path) == event['source']['sha256'] and
                reviewed.get(str(path)) == event['source']['sha256'], 'New source omitted independent physical audit')
    return objects


def verify_coverage_sources(reader, supplied, previous, prior):
    require(set(supplied) == {'batch1', 'batch2', 'batch3'}, 'Exact three source batches required')
    require(set(prior['coverage_sources']) == {'batch1', 'batch2'}, 'Prior151 source scope changed')
    for batch in ('batch1', 'batch2'):
        same_references(reader, supplied[batch], prior['coverage_sources'][batch],
                        'Frozen prior151 source reference changed')
    objects = {batch:verify_coverage(reader, supplied[batch], previous, batch)
               for batch in ('batch1', 'batch2', 'batch3')}
    r.combine_sources(objects['batch1']['dataset'], objects['batch2']['dataset'],
                      objects['batch3']['dataset'], supplied)
    return objects


def prospective_review(reader, path, expected):
    review = reader.read(path)
    require(review.get('passed') is True and
            type(review.get('actual_factors_calculated')) is int and review['actual_factors_calculated'] == 0 and
            type(review.get('actual_returns_calculated')) is int and review['actual_returns_calculated'] == 0,
            'Prospective independent review required')
    require(reader.verify(review['source_hashes']) == expected, 'Reviewer source closure changed')
    contracts = reader.verify(review['reviewed_contract_hashes'])
    for name in ('work/monthly_financial_coverage152_runner.py',
                 'work/monthly_financial_coverage152_factor.py', PROPOSAL):
        require(contracts.get(str(reader.path(name))) == reader.hash(name),
                'Prospective reviewer omitted current contract')


def build_plan(root, tests, proposal_sha):
    require(type(proposal_sha) is str and re.fullmatch('[0-9a-f]{64}', proposal_sha),
            'Explicit final prospective proposal SHA required')
    import monthly_financial_coverage152_prepare_review as prep
    import monthly_financial_coverage152_outcome_review as outcome
    root = Path(root).resolve()
    reader = old.Reader(root)
    out = root/'outputs'/STUDY
    actual_files = ('PLAN.json', 'CACHE.json', 'RESULTS.json', 'FROZEN.json', *r.CACHE_FILES,
                    'TRAIN_TRIALS.json', 'TRAIN_OPEN_TRIALS.json', 'DIAGNOSE_STARTED.json')
    require(not any((out/name).exists() for name in actual_files), 'Registration must precede all actual work')
    state, state_sha = state_snapshot(root/STATE)
    predecessor(state)
    require(state['registered_queue'] == [], 'Pending experiment')
    require(reader.hash(PROPOSAL) == proposal_sha, 'Final proposal changed')
    proposal = reader.read(PROPOSAL)
    r.validate_proposal(proposal)
    concept = proposal['concept']
    require(reader.hash(concept['path']) == concept['sha256'], 'Prospective concept changed')
    previous = verify_legacy(reader, proposal['sources']['legacy149'])
    prior = verify_prior(reader, proposal['sources']['prior151'], proposal['sources']['legacy149'])
    verify_coverage_sources(reader, proposal['sources']['coverage_sources'], previous, prior)
    reader.verify(r.source_files(root))
    for path in (Path(__file__), root/'work/test_monthly_financial_coverage152_lifecycle.py',
                 root/'work/prepare_financial_coverage152_proposal.py'):
        reader.hash(path)
    independent = {name: source_pins(reader, module.source_files(root))
                   for name,module in (('prepare',prep),('outcome',outcome))}
    prospective_review(reader, PRE_REVIEW, independent['prepare'])
    prospective_review(reader, OUT_REVIEW, independent['outcome'])
    validate_tests(reader.read(tests), reader)
    inputs = {key: dict(previous['inputs'][key]) for key in r.INPUTS if key != 'proposal'}
    inputs['proposal'] = dict(path=str(root/PROPOSAL), sha256=proposal_sha)
    for ref in inputs.values():
        require(reader.hash(ref['path']) == ref['sha256'], 'Named input changed')
        ref['path'] = str(reader.path(ref['path']))
    reader.verify(dict(reader.seen))
    hashes = dict(reader.seen)
    hashes.pop(str(root/STATE), None)
    return dict(round=152,status='REGISTERED',registered_utc=datetime.now(timezone.utc).isoformat(),
                registration_precondition=dict(path=STATE,sha256=state_sha),
                arms=list(r.NAMES),primary_challenger=r.NAMES[4],dose=r.DOSE,
                output_directory=str(out),runtime=dict(python=sys.version,numpy=r.np.__version__,pandas=r.pd.__version__),
                tolerances=r.TOLERANCES,hashes=hashes,inputs=inputs,
                legacy149=proposal['sources']['legacy149'],prior151=proposal['sources']['prior151'],
                coverage_sources=proposal['sources']['coverage_sources'],
                raw_source_hashes=previous['raw_source_hashes'],
                independent_prepare_source_hashes=independent['prepare'],
                independent_outcome_source_hashes=independent['outcome'],
                actual_feature_calculations_before_registration=0,actual_returns_before_registration=0,
                actual_model_fits_before_registration=0,no_observer_writes=True,tests=tests)


def queue_entry(root, plan, plan_sha):
    return dict(round=152,directory=STUDY,
                plan=dict(path='outputs/'+STUDY+'/PLAN.json',sha256=plan_sha),
                registered_utc=plan['registered_utc'])


def enqueue(root, plan_sha):
    root=Path(root).resolve()
    path=root/'outputs'/STUDY/'PLAN.json'
    require(r.sha(path) == plan_sha, 'Registered PLAN changed')
    plan=r.read(path)
    require(plan.get('round') == 152 and plan.get('status') == 'REGISTERED', 'Registered152 PLAN required')
    reader=old.Reader(root)
    reader.verify(plan['hashes'])
    state,_=state_snapshot(root/STATE)
    predecessor(state)
    entry=queue_entry(root,plan,plan_sha)
    if state['registered_queue'] == [entry]:
        return dict(registered=True,plan_sha256=plan_sha,already_enqueued=True)
    require(not state['registered_queue'], 'Conflicting registration')
    state.update(registered_queue=[entry],current=STUDY,current_status='registered_before_actual_preparation',
                 updated_utc=plan['registered_utc'])
    update_state(root/STATE,plan['registration_precondition']['sha256'],state)
    return dict(registered=True,plan_sha256=plan_sha,already_enqueued=False)


def register(root, tests, proposal_sha):
    root=Path(root).resolve()
    plan=build_plan(root,tests,proposal_sha)
    out=root/'outputs'/STUDY
    out.mkdir(exist_ok=True)
    write(out/'PLAN.json',plan)
    plan_sha=r.sha(out/'PLAN.json')
    enqueue(root,plan_sha)
    return dict(registered=True,plan_sha256=plan_sha,hashes=len(plan['hashes']))


def validate_outcome_receipt(review, plan_sha):
    require(review.get('passed') is True and review.get('stage') == 'evaluate' and
            review.get('plan_sha256') == plan_sha, 'Complete independent outcome audit required')
    for name, expected in (('execution_paths_verified',15), ('total_monthly_orders_verified',610),
                           ('financial_coverage_feature_matrices_verified',8)):
        require(type(review.get(name)) is int and review[name] == expected, 'Independent outcome scope: '+name)
    for name in ('coverage_source_admission_verified', 'original149_earnings_cache_and_mask_verified',
                 'old44_source_records_and_holds_preserved', 'latest_catalog_blocking_independently_verified',
                 'prior150_batch1_earnings_cache_and_mask_verified', 'batch1_source_records_and_holds_preserved',
                 'prior151_batch2_earnings_cache_and_mask_verified', 'batch2_source_records_and_holds_preserved'):
        require(review.get(name) is True, 'Independent outcome guarantee: '+name)


def required_outcome_files():
    files=['PLAN.json','CACHE.json','INDEPENDENT_PREPARE.json',*r.CACHE_FILES,
           'RESULTS.json','FROZEN.json','TRAIN_TRIALS.json','TRAIN_OPEN_TRIALS.json',
           'WALKFORWARD.json','VALIDATION.json','DIAGNOSE_STARTED.json']
    files.extend(name+'_'+mode+'_'+kind+'.json' for name in r.NAMES
                 for mode in ('same_close','next_open_zero','next_open_cost')
                 for kind in ('DAILY','TRADES','HOLDINGS'))
    return files


def complete(root, plan_sha):
    root=Path(root).resolve()
    out=root/'outputs'/STUDY
    runner=r.Runner(root,out/'PLAN.json',plan_sha)
    runner.require_prepare_review()
    reader=old.Reader(root)
    plan=reader.read(out/'PLAN.json')
    review=reader.read(out/'INDEPENDENT_OUTCOME.json')
    validate_outcome_receipt(review,plan_sha)
    artifacts=reader.verify(review['artifact_hashes'])
    require(reader.verify(review['source_hashes']) == reader.verify(plan['independent_outcome_source_hashes']),
            'Outcome reviewer code changed')
    for filename in required_outcome_files():
        require(artifacts.get(str(out/filename)) == reader.hash(out/filename),
                'Unreviewed outcome: '+filename)
    require(review.get('cache_sha256') == reader.hash(out/'CACHE.json') and
            review.get('orders_sha256') == reader.hash(out/'PRETRAIN_ORDERS.json'), 'Independent cache/order link changed')
    frozen=reader.read(out/'FROZEN.json')
    require(set(frozen['hashes']) == set(r.FROZEN_FILES), 'Exact frozen selection dependencies required')
    reader.verify(frozen['hashes'], out)
    selection=r.select(reader.read(out/'TRAIN_TRIALS.json'),reader.read(out/'TRAIN_OPEN_TRIALS.json'))
    require(frozen['selection'] == selection and review['selected_name'] == selection['primary'], 'Selected primary mismatch')
    validation=reader.read(out/'VALIDATION.json')
    require(validation.get('selection') == selection and validation.get('no_observer_writes') is True and
            validation.get('no_intramonth_trades') is True and
            type(validation.get('new_model_fits')) is int and validation['new_model_fits'] == 0,
            'Final execution validation changed')
    require(reader.read(out/'DIAGNOSE_STARTED.json').get('plan_sha256') == plan_sha,
            'Diagnostic must follow this registration')
    results=reader.read(out/'RESULTS.json')
    manifest=reader.read(plan['inputs']['protected_manifest']['path'])
    require(len(manifest) == 21444, 'Protected21444 archive required')
    proof=dict(passed=True,checked=len(manifest),verified=len(manifest),
               manifest_sha256=plan['inputs']['protected_manifest']['sha256'],changed=[],missing=[],read_or_identity_errors=[])
    old.protected_scan(reader,plan,proof)
    if not (out/'PROTECTED_AFTER.json').exists():
        write(out/'PROTECTED_AFTER.json',proof)
    else:
        require(reader.read(out/'PROTECTED_AFTER.json') == proof, 'Protection proof changed')
    state,expected=state_snapshot(root/STATE)
    predecessor(state)
    require(state['registered_queue'] == [queue_entry(root,plan,plan_sha)], 'Registered queue mismatch')
    primary=selection['primary']
    metric=results[primary]['same_close']['metrics']
    require(abs(metric['cagr']-review['primary_cagr']) < 1e-12 and
            review['primary_target_reached'] == (metric['cagr'] >= .7), 'Independent target mismatch')
    done=dict(round=152,directory=STUDY,completed_utc=datetime.now(timezone.utc).isoformat(),
              plan_sha256=plan_sha,selection=selection,all_arm_results=results,primary=primary,
              primary_cagr=metric['cagr'],primary_sharpe=metric['sharpe'],target_reached=metric['cagr']>=.7,
              trials=5,independent_execution_paths=15,protected_files=len(manifest),source_admission_conditional=True,
              independent_outcome_sha256=reader.hash(out/'INDEPENDENT_OUTCOME.json'),
              protected_sha256=reader.hash(out/'PROTECTED_AFTER.json'))
    reader.verify(dict(reader.seen))
    done=preserve_completion(out/'COMPLETION.json',done)
    ref=dict(path=str((out/'COMPLETION.json').relative_to(root)),sha256=r.sha(out/'COMPLETION.json'))
    entry={key:done[key] for key in ('directory','completed_utc','primary','primary_cagr','primary_sharpe',
                                   'target_reached','trials','independent_execution_paths','protected_files')}
    entry.update(completion=ref,new_admitted=int(selection['admitted']))
    state['completed'].append(entry)
    require(len(state['completed']) == 152 and sum(item['trials'] for item in state['completed']) == 962,
            'Final152 rounds/962 trials accounting')
    state.update(current=STUDY,current_status='completed_independently_verified',registered_queue=[],
                 latest_round_receipt=ref,updated_utc=done['completed_utc'],last_goal_turn_classification='progress')
    if selection['admitted'] and metric['cagr'] > state['best_training_selected_candidate']['cagr']:
        state['best_training_selected_candidate']=dict(directory=STUDY,cagr=metric['cagr'],sharpe=metric['sharpe'],
            max_drawdown=metric['max_drawdown'],next_open_5bp_cagr=results[primary]['next_open_cost']['metrics']['cagr'],
            spec=dict(mode=primary),observer_promotion=False)
    update_state(root/STATE,expected,state)
    return dict(completion=ref,primary=primary,cagr=metric['cagr'],sharpe=metric['sharpe'],
                target_reached=done['target_reached'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('register','enqueue','complete'))
    parser.add_argument('--plan-sha')
    parser.add_argument('--tests')
    parser.add_argument('--proposal-sha')
    args=parser.parse_args()
    with state_lock(ROOT):
        if args.phase == 'register':
            require(args.tests and args.proposal_sha, 'Tests and final prospective proposal SHA required')
            result=register(ROOT,args.tests,args.proposal_sha)
        elif args.phase == 'enqueue':
            result=enqueue(ROOT,args.plan_sha)
        else:
            result=complete(ROOT,args.plan_sha)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
