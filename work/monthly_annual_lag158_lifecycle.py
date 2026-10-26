"""Prospective registration and evidence-only completion of fixed lag pilot158.

No real feature or portfolio calculation occurs here. Historical source aliases
are exact-byte archives, never relaxed digests. Observer accounts are untouched.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import importlib
import json
import math
import re
import sys

import complete_monthly_same_report145 as old
import monthly_filing_association157_lifecycle as prior

ROOT = Path(__file__).resolve().parents[1]
STUDY = 'monthly_annual_lag158'
OUT = 'outputs/' + STUDY
STATE = prior.STATE
DESIGN = 'work/after157_lag_pilot_design.json'
DESIGN_SHA = '215ef056f8ac8fb3ac19e52acc49bd983802bb13cddec0b103826c85640055d5'
DESIGN_REVIEW = 'work/after157_lag_pilot_design_review.json'
DESIGN_REVIEW_SHA = '07e9cd6ea16ca6967b111ed6f91d0ea7c19b6b2d175f12e75e7bb6137f39c651'
BASE_PLAN = 'outputs/monthly_paired_target143/PLAN.json'
BASE_SHA = '2d73a3afb4745f5106f6cc870d0c25a775721181cd3c99ad78a7e1f56b58500b'
PREVIOUS = 'outputs/monthly_filing_association157/COMPLETION.json'
PREVIOUS_SHA = 'f4de303fc011a336865519d82b48219833eb0953cc9a8dbed0849f3be2354549'
PREVIOUS_PLAN = 'outputs/monthly_filing_association157/PLAN.json'
PREVIOUS_PLAN_SHA = '43adc171b3ee82cd952544b75db63fffaf5b87bd25b96a03e709fd2ff22356db'
PREVIOUS_REPAIR = 'outputs/monthly_filing_association157/REPAIR_COMPLETION_01.json'
PREVIOUS_REPAIR_SHA = '0994337624d66435d5cea4f82da15efb9e79f6a55c282fde4ff287c7e19e18eb'
OVERLAY = 'outputs/after157_walmart_context_admission_v1/ADMISSION.json'
OVERLAY_SHA = '9e90889a1fb4f057438b6ac16a9e234f97cc9ede08c6cb28a6fe190389f3f9d6'
SOURCE_REVIEW = 'work/after157_walmart_context_independent.json'
SOURCE_REVIEW_SHA = '16a666e2d72426a9baa92e693fc003d8c107f9908032762d61232c151dc18814'
SOURCE_PRIOR = 'work/after157_annual_pair_source_feasibility.json'
SOURCE_PRIOR_SHA = '4fdeaad7de09625991a2b3603d33799bf1c62dd0b99f560b900ac8928abc9242'
HISTORICAL_SHA = 'f0f6571a3f4de4be06aeb957fe03fd13eed11f86b288abafc91094db38ac32b7'
ARCHIVE = 'outputs/monthly_target70_research/PRE157_' + HISTORICAL_SHA + '.json'
PROPOSAL = 'work/monthly_annual_lag158_proposal.json'
SELF = 'work/monthly_annual_lag158_lifecycle.py'
NAMES = ('incumbent_local67', 'negative_current_annual_lag', 'negative_annual_lag_change')
MODES = ('same_close', 'next_open_zero', 'next_open_cost')
SUITES = dict(model='test_monthly_annual_lag158_model.py', runner='test_monthly_annual_lag158_runner.py',
    prepare_review='test_annual_lag158_independent_prepare.py',
    outcome_review='test_annual_lag158_independent_outcome.py', lifecycle='test_monthly_annual_lag158_lifecycle.py')
require, sha, write = old.require, old.sha, prior.write


def now():
    return datetime.now(timezone.utc).isoformat()


def runner():
    return importlib.import_module('monthly_annual_lag158_runner')


def reviewer(stage):
    return importlib.import_module('annual_lag158_independent_' + stage)


def predecessor(state):
    rows = state.get('completed')
    require(type(rows) is list and len(rows) == 157 and
        all(type(row.get('trials')) is int and row['trials'] >= 0 for row in rows) and
        sum(row['trials'] for row in rows) == 986, 'Expected157 completed rounds /986 trials')
    require(rows[-1].get('directory') == 'monthly_filing_association157' and
        rows[-1].get('completion', {}).get('sha256') == PREVIOUS_SHA, 'Wrong finalized157 predecessor')


def source_pins(reader, mapping):
    require(type(mapping) is dict and mapping, 'Nonempty source pins required')
    resolved = {}
    for name, digest in mapping.items():
        path = reader.path(name)
        if path == reader.root / STATE:
            require(digest == HISTORICAL_SHA, 'Only exact historical central alias allowed')
            path = reader.root / ARCHIVE
        require(reader.hash(path) == digest, 'Source digest changed: ' + str(path))
        resolved[str(path)] = digest
    return resolved


def validate_source(reader):
    for path, digest in ((DESIGN, DESIGN_SHA), (DESIGN_REVIEW, DESIGN_REVIEW_SHA),
                         (OVERLAY, OVERLAY_SHA), (SOURCE_REVIEW, SOURCE_REVIEW_SHA),
                         (SOURCE_PRIOR, SOURCE_PRIOR_SHA), (PREVIOUS, PREVIOUS_SHA),
                         (PREVIOUS_PLAN, PREVIOUS_PLAN_SHA), (PREVIOUS_REPAIR, PREVIOUS_REPAIR_SHA)):
        require(reader.hash(path) == digest, 'Fixed evidence changed: ' + path)
    review, overlay = reader.read(SOURCE_REVIEW), reader.read(OVERLAY)
    require(review.get('passed') is True and review.get('source_admission_changed') is False,
            'Independent conditional-source review required')
    source_pins(reader, review['input_hashes']); source_pins(reader, overlay['input_hashes'])
    require(len(overlay['rows']) == 5978 and overlay['summary'] == review['summary'], 'Source grid/summary mismatch')
    for key, count in (('all122', 230), ('first88', 168), ('later34', 62)):
        require(overlay['summary'][key]['conditional_pair_slots'] == count, 'Fixed source coverage changed')
    require(all(row['original_row']['tier3'] is False and row['old_admission_flags_changed'] is False
                for row in overlay['rows']), 'Original admission flags changed')
    return overlay


def ref(reader, path):
    return dict(path=str(reader.path(path)), sha256=reader.hash(path))


def proposal(root):
    r = runner(); reader = old.Reader(root)
    state, _ = prior.state_snapshot(reader.root/STATE); predecessor(state)
    require(state['registered_queue'] == [], 'Pending experiment')
    validate_source(reader)
    require(reader.hash(BASE_PLAN) == BASE_SHA, 'Frozen143 source plan changed')
    base = reader.read(BASE_PLAN)
    inputs = {k: dict(base['inputs'][k]) for k in r.INPUTS if k in base['inputs'] and k != 'proposal'}
    for key, path in dict(design=DESIGN, calendar='outputs/monthly_paired_target143/CALENDAR.json',
                         source_overlay=OVERLAY, source_review=SOURCE_REVIEW, source_prior=SOURCE_PRIOR).items():
        inputs[key] = ref(reader, path)
    require(set(inputs) == set(r.INPUTS)-{'proposal'}, 'Named input schema')
    doc = dict(round=158, status='FINAL_PROSPECTIVE_SPECIFICATION_BEFORE_REGISTRATION', created_utc=now(),
        arms=list(NAMES), primary_challenger=NAMES[2], design=dict(path=DESIGN, sha256=DESIGN_SHA),
        previous_completion=dict(path=PREVIOUS, sha256=PREVIOUS_SHA), source_inputs=inputs,
        source_plan=dict(path=BASE_PLAN, sha256=BASE_SHA),
        measurement=r.measurement_contract(), selection_contract=r.selection_contract(),
        source_refs={k: inputs[k] for k in ('source_overlay', 'source_review', 'source_prior')},
        actual_new_labels=0, actual_new_model_fits=0, actual_new_features=0, actual_new_orders=0,
        actual_new_returns=0, actual_new_lag_values=0,
        fixed=dict(period=['2016-08-17','2026-09-24'], monthly_signals=122, width=49,
                   first_training_months=88, later_diagnostic_months=34, original_arrays=27,
                   ordered_decisions=366, execution_paths=9, prefix_checks=123,
                   observer_changes=False, downloads=0),
        limitations=reader.read(DESIGN)['limits'])
    r.validate_proposal(doc)
    write(reader.root/PROPOSAL, doc)
    return ref(reader, PROPOSAL)


def validate_tests(doc, reader):
    require(doc.get('passed') is True and doc.get('synthetic_only') is True, 'Synthetic tests required')
    found = {}
    for row in doc['suites']:
        name, count = row['name'], row['count']
        require(name in SUITES and name not in found and type(count) is int and count > 0 and
                row['passed'] is True, 'Distinct positive passed suites required')
        for field in ('source','log'):
            require(reader.hash(row[field]['path']) == row[field]['sha256'], 'Changed test source/log')
        require(reader.path(row['source']['path']) == reader.root/'work'/SUITES[name], 'Wrong test suite')
        log = reader.path(row['log']['path']).read_text()
        require(re.findall(r'Ran (\d+) tests? in ', log) == [str(count)] and log.rstrip().endswith('OK'),
                'Unsuccessful/ambiguous test log')
        found[name] = count
    require(set(found) == set(SUITES) and type(doc['test_count']) is int and
            doc['test_count'] == sum(found.values()), 'Complete five-suite accounting required')
    return found


def build_plan(root, tests, proposal_sha):
    r = runner(); reader = old.Reader(root); out = reader.root/OUT
    require(not any((out/f).exists() for f in ('PLAN.json','CACHE.json','FROZEN.json','RESULTS.json',*r.CACHE_FILES)),
            'Registration must precede actual work')
    state, state_sha = prior.state_snapshot(reader.root/STATE); predecessor(state)
    require(state['registered_queue'] == [], 'Another experiment pending')
    validate_source(reader)
    require(reader.hash(BASE_PLAN) == BASE_SHA, 'Frozen143 PLAN changed')
    base = reader.read(BASE_PLAN); reader.verify(base['hashes']); reader.verify(base['raw_source_hashes'])
    require(len(base['raw_source_hashes']) == 64, 'Original64 raw prices')
    require(old.digest(proposal_sha) and reader.hash(PROPOSAL) == proposal_sha, 'Exact prospective specification required')
    spec = reader.read(PROPOSAL); r.validate_proposal(spec)
    reader.verify(r.source_files(reader.root))
    independent = {}
    for stage in ('prepare','outcome'):
        module = reviewer(stage)
        paths = module.source_files(reader.root)
        independent[stage] = reader.verify(paths) if isinstance(paths, dict) else {
            str(reader.path(p)): reader.hash(p) for p in paths}
    for p in (SELF, 'work/test_monthly_annual_lag158_lifecycle.py', old.__file__, prior.__file__): reader.hash(p)
    review_path = 'work/annual_lag158_registration_review.json'
    review = reader.read(review_path)
    require(review.get('passed') is True and review.get('round') == 158 and
            review.get('actual_features_or_returns_computed') is False and
            review.get('proposal_sha256') == proposal_sha, 'Independent prospective implementation review required')
    reviewed = reader.verify(review['implementation_hashes'])
    needed = set(r.source_files(reader.root)) | {str(reader.root/SELF)}
    needed.update(independent['prepare']); needed.update(independent['outcome'])
    require(all(reviewed.get(p) == reader.hash(p) for p in needed), 'Implementation not prospectively reviewed')
    require(reader.path(tests) == out/'TESTS.json', 'Canonical TESTS path required')
    validate_tests(reader.read(tests), reader)
    inputs = {k:dict(v) for k,v in spec['source_inputs'].items()}
    inputs['proposal'] = ref(reader, PROPOSAL)
    require(set(inputs) == set(r.INPUTS), 'Input contract mismatch')
    for k,v in inputs.items():
        require(reader.hash(v['path']) == v['sha256'], 'Changed input '+k)
        v['path'] = str(reader.path(v['path']))
    reader.verify(base['calendar']['source_hashes'])
    reader.verify(dict(reader.seen))
    return dict(round=158, status='REGISTERED', registered_utc=now(),
        registration_precondition=dict(path=STATE, sha256=state_sha), arms=list(NAMES),
        primary_challenger=NAMES[2], output_directory=str(out), inputs=inputs,
        hashes=dict(reader.seen), raw_source_hashes=base['raw_source_hashes'], calendar=base['calendar'],
        independent_prepare_source_hashes=independent['prepare'], independent_outcome_source_hashes=independent['outcome'],
        historical_pin_aliases={STATE:dict(original_sha256=HISTORICAL_SHA, archive_path=ARCHIVE, archive_sha256=HISTORICAL_SHA)},
        tests=str(out/'TESTS.json'), runtime=dict(python=sys.version, numpy=r.np.__version__, pandas=r.pd.__version__),
        tolerances=dict(r.TOLERANCES), dose=.01, measurement=r.measurement_contract(),
        selection_contract=r.selection_contract(), actual_label_calculations_before_registration=0,
        actual_model_fits_before_registration=0, actual_feature_calculations_before_registration=0,
        actual_returns_before_registration=0, actual_lag_calculations_before_registration=0,
        actual_nuisance_estimates_before_registration=0,
        new_market_downloads=0, no_observer_writes=True)


def queue_entry(plan, digest):
    return dict(round=158, directory=STUDY, plan=dict(path=OUT+'/PLAN.json',sha256=digest),
                registered_utc=plan['registered_utc'])


def enqueue(root, plan_sha):
    reader = old.Reader(root); path = reader.root/OUT/'PLAN.json'
    require(reader.hash(path) == plan_sha, 'Exact PLAN required')
    plan = reader.read(path)
    require(plan['round'] == 158 and plan['arms'] == list(NAMES) and plan['status'] == 'REGISTERED', 'Wrong PLAN')
    reader.verify(plan['hashes'])
    state, current = prior.state_snapshot(reader.root/STATE); predecessor(state)
    entry = queue_entry(plan,plan_sha)
    if state['registered_queue'] == [entry]: return dict(already_enqueued=True, plan_sha256=plan_sha)
    require(not state['registered_queue'] and current == plan['registration_precondition']['sha256'], 'Stale registration state')
    state.update(registered_queue=[entry], current=STUDY, current_status='registered_before_actual_preparation',
        updated_utc=now(), last_goal_turn_classification='progress',
        next_step='Execute fixed158 prepare, independent prepare, first88 selection once, then locked later34 diagnostics and independent nine-path accounting; no tuning.')
    prior.update_state(reader.root/STATE,current,state)
    return dict(registered=True, plan_sha256=plan_sha)


def register(root, tests, proposal_sha):
    plan = build_plan(root,tests,proposal_sha); out = Path(root)/OUT
    out.mkdir(exist_ok=True); write(out/'PLAN.json',plan)
    return enqueue(root,sha(out/'PLAN.json'))


def validate_receipt(reader, report, stage, checks, plan, out, plan_sha):
    require(report.get('passed') is True and report.get('stage') == stage and
        report.get('producer_modules_imported') is False and all(report.get(k) is True for k in checks),
        'Incomplete independent '+stage+' evidence')
    for key,file in (('plan_sha256','PLAN.json'),('cache_sha256','CACHE.json'),('orders_sha256','PRETRAIN_ORDERS.json')):
        require(report.get(key) == reader.hash(out/file), 'Receipt binding mismatch '+key)
    kind = 'prepare' if stage == 'prepare' else 'outcome'
    require(reader.verify(report['source_hashes']) == reader.verify(plan['independent_'+kind+'_source_hashes']), 'Review implementation changed')
    return reader.verify(report['artifact_hashes'])


def publish_completion(reader, plan, plan_sha, proof, payload):
    """Recover an interrupted publication using identical verified evidence."""
    out = reader.root/OUT
    state, current = prior.state_snapshot(reader.root/STATE)
    already = len(state.get('completed', [])) == 158
    if already:
        previous = dict(state, completed=state['completed'][:-1])
        predecessor(previous)
        require(state.get('registered_queue') == [] and
                (out/'COMPLETION.json').is_file() and (out/'PROTECTED_AFTER.json').is_file(),
                'Completed state needs original receipts and an empty queue')
    else:
        predecessor(state)
        require(state['registered_queue'] == [queue_entry(plan, plan_sha)], 'Exact active158 queue required')
    if (out/'PROTECTED_AFTER.json').exists():
        require(old.encoded(reader.read(out/'PROTECTED_AFTER.json')) == old.encoded(proof),
                'Existing protection proof differs')
    else:
        require(not (out/'COMPLETION.json').exists(), 'Completion without original protection proof')
        reader.verify(dict(reader.seen))
        write(out/'PROTECTED_AFTER.json', proof)
    expected = dict(payload, protected_sha256=reader.hash(out/'PROTECTED_AFTER.json'))
    if (out/'COMPLETION.json').exists():
        done = reader.read(out/'COMPLETION.json')
        require(old.encoded({k:v for k,v in done.items() if k != 'completed_utc'}) == old.encoded(expected),
                'Existing completion differs from reverified evidence')
        timestamp = datetime.fromisoformat(done['completed_utc'])
        require(timestamp.tzinfo is not None and
                datetime.fromisoformat(plan['registered_utc']) <= timestamp <= datetime.now(timezone.utc),
                'Invalid original completion time')
    else:
        done = dict(expected, completed_utc=now())
        reader.verify(dict(reader.seen))
        write(out/'COMPLETION.json', done)
    reference = dict(path=OUT+'/COMPLETION.json', sha256=reader.hash(out/'COMPLETION.json'))
    entry = {k:done[k] for k in ('directory','completed_utc','primary','primary_cagr','primary_sharpe',
            'target_reached','trials','new_model_fits','independent_execution_paths','protected_files')}
    entry.update(completion=reference, new_admitted=int(done['selection']['admitted']))
    if already:
        require(old.encoded(state['completed'][-1]) == old.encoded(entry) and
                state.get('latest_round_receipt') == reference and state.get('current') == STUDY and
                state.get('current_status') == 'completed_independently_verified', 'Completed central binding differs')
    else:
        state['completed'].append(entry)
        state.update(current=STUDY, current_status='completed_independently_verified', registered_queue=[],
            latest_round_receipt=reference, updated_utc=done['completed_utc'], last_goal_turn_classification='progress',
            next_step='Fixed two-issuer lag pilot complete; retain actual null/positive evidence and stop this fixed scope. Any next hypothesis requires distinct preregistration, not dose or universe tuning on this result.')
        if done['selection']['admitted'] and done['primary_cagr'] > state['best_training_selected_candidate']['cagr']:
            metric = done['all_arm_results'][done['primary']]['same_close']['metrics']
            state['best_training_selected_candidate'] = dict(directory=STUDY, spec=done['primary'],
                cagr=metric['cagr'], sharpe=metric['sharpe'], max_drawdown=metric['max_drawdown'],
                next_open_5bp_cagr=done['primary_next_open_cost']['cagr'], observer_promotion=False)
    require(len(state['completed']) == 158 and sum(x['trials'] for x in state['completed']) == 989,
            '158/989 accounting')
    reader.verify(dict(reader.seen))
    if not already:
        prior.update_state(reader.root/STATE, current, state)
    else:
        require(sha(reader.root/STATE) == current, 'Concurrent completed state change')
    return dict(completion=reference, primary=done['primary'], cagr=done['primary_cagr'],
                target_reached=done['target_reached'], already_published=already)


def complete(root, plan_sha):
    r = runner(); outcome_module = reviewer('outcome'); reader=old.Reader(root); out=reader.root/OUT
    require(reader.hash(out/'PLAN.json') == plan_sha, 'Exact PLAN required')
    plan=reader.read(out/'PLAN.json'); reader.verify(plan['hashes'])
    r.Runner(root,out/'PLAN.json',plan_sha).require_prepare_review()
    prep=reader.read(out/'INDEPENDENT_PREPARE.json'); result=reader.read(out/'INDEPENDENT_OUTCOME.json')
    pa=validate_receipt(reader,prep,'prepare',r.CHECKS,plan,out,plan_sha)
    oa=validate_receipt(reader,result,'evaluate',outcome_module.OUTCOME_CHECKS,plan,out,plan_sha)
    require(prep.get('returns_or_nav_computed') is False, 'Preparation read outcomes')
    cache=reader.read(out/'CACHE.json'); require(set(cache)==set(r.CACHE_FILES),'Complete cache')
    for f in (*r.CACHE_FILES,'PLAN.json','CACHE.json'):
        require(pa.get(str(out/f))==reader.hash(out/f),'Unverified prepared artifact '+f)
    files=set((*r.CACHE_FILES,*r.FROZEN_FILES,'CACHE.json','FROZEN.json','DIAGNOSE_STARTED.json',
               'RESULTS.json','MONTH_COMPARISON.json','ATTRIBUTION.json','VALIDATION.json'))
    files.update(name+'_'+mode+'_'+kind+'.json' for name in NAMES for mode in MODES for kind in ('DAILY','TRADES','HOLDINGS'))
    for f in files: require(oa.get(str(out/f))==reader.hash(out/f),'Unverified outcome artifact '+f)
    frozen=reader.read(out/'FROZEN.json'); reader.verify(frozen['hashes'],out)
    require(set(frozen['hashes'])==set(r.FROZEN_FILES) and not (out/'WALKFORWARD.json').exists(), 'Exactly one first88 selection scope')
    chosen=r.select(reader.read(out/'TRAIN_TRIALS.json'),reader.read(out/'TRAIN_OPEN_TRIALS.json'))
    require(chosen==frozen['selection'] and result['selected_name']==chosen['primary'],'Selection differs')
    require(result['frozen_sha256']==reader.hash(out/'FROZEN.json'),'Frozen audit binding')
    marker=reader.read(out/'DIAGNOSE_STARTED.json')
    require(marker['plan_sha256']==plan_sha and marker['frozen_sha256']==reader.hash(out/'FROZEN.json') and
        result['diagnose_started_sha256']==reader.hash(out/'DIAGNOSE_STARTED.json'),'Diagnostic binding')
    times=[datetime.fromisoformat(t) for t in (plan['registered_utc'],marker['started_utc'],result['audit_started_utc'])]
    require(all(t.tzinfo is not None for t in times) and times[0]<times[1]<times[2]<=datetime.now(timezone.utc), 'Causal audit chronology')
    results=reader.read(out/'RESULTS.json')
    require(set(results)==set(NAMES) and all(set(results[n])==set(MODES) for n in NAMES),'Exactly nine paths')
    metric=results[chosen['primary']]['same_close']['metrics']
    require(all(old.number(metric[k]) for k in ('cagr','sharpe','max_drawdown')),'Finite selected metrics')
    require(abs(metric['cagr']-result['primary_cagr'])<1e-12 and
        result['primary_target_reached']==(metric['cagr']>=.7),'Independent selected target mismatch')
    for key,count in (('execution_paths_verified',9),('total_monthly_orders_verified',366),('models_fit',0)):
        require(type(result.get(key)) is int and result[key]==count, 'Outcome count '+key)
    orders=reader.read(out/'PRETRAIN_ORDERS.json')
    require(orders[NAMES[0]]==reader.read(plan['inputs']['original_orders']['path']),'Incumbent changed')
    validation=reader.read(out/'VALIDATION.json')
    require(validation['selection']==chosen and validation['no_observer_writes'] is True and
        validation['no_intramonth_trades'] is True and validation['selection_count']==1,'Frozen framework changed')
    for key in ('new_predictive_model_fits','nuisance_estimates'):
        require(type(validation.get(key)) is int and validation[key]==0,'Unexpected estimates: '+key)
    suites=validate_tests(reader.read(out/'TESTS.json'),reader)
    manifest=reader.read(plan['inputs']['protected_manifest']['path']); require(len(manifest)==21444,'Full protected archive')
    proof=dict(passed=True,checked=21444,verified=21444,manifest_sha256=plan['inputs']['protected_manifest']['sha256'],changed=[],missing=[],read_or_identity_errors=[])
    old.protected_scan(reader,plan,proof)
    done=dict(round=158,directory=STUDY,status='completed_independently_verified',
        plan_sha256=plan_sha,selection=chosen,all_arm_results=results,primary=chosen['primary'],
        primary_cagr=metric['cagr'],primary_sharpe=metric['sharpe'],
        primary_next_open_cost=results[chosen['primary']]['next_open_cost']['metrics'],target_reached=metric['cagr']>=.7,
        trials=3,new_model_fits=0,independent_execution_paths=9,protected_files=21444,synthetic_suites=suites,
        independent_outcome_sha256=reader.hash(out/'INDEPENDENT_OUTCOME.json'),limitations=reader.read(DESIGN)['limits'])
    return publish_completion(reader, plan, plan_sha, proof, done)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('proposal','register','enqueue','complete'))
    parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--tests')
    parser.add_argument('--proposal-sha');parser.add_argument('--plan-sha')
    args=parser.parse_args()
    with prior.state_lock(args.root):
        if args.phase=='proposal': result=proposal(args.root)
        elif args.phase=='register': result=register(args.root,args.tests,args.proposal_sha)
        elif args.phase=='enqueue': result=enqueue(args.root,args.plan_sha)
        else: result=complete(args.root,args.plan_sha)
    print(old.encoded(result).decode())


if __name__=='__main__': main()
