"""Metadata-only registration and closure for the prospectively bounded retest.

Numerical evidence is produced and independently checked by separate programs.
This writer serializes central-state changes and preserves every original byte.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import argparse
import fcntl
import hashlib
import math
import os
import re
import sys
import tempfile

import complete_monthly_same_report145 as old

ROOT = Path(__file__).resolve().parents[1]
STUDY = 'monthly_endpoint155'
STATE = 'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
DESIGN = 'work/after154_endpoint_retest_design_v2.json'
DESIGN_SHA = '91d8ed23f6aa53cde1ac62728bae2b08376f2f7ecc36076d6893d1bf37f2ba63'
BASE_PLAN = 'outputs/monthly_paired_target143/PLAN.json'
BASE_SHA = '2d73a3afb4745f5106f6cc870d0c25a775721181cd3c99ad78a7e1f56b58500b'
PREVIOUS = 'outputs/monthly_financial_coverage154/COMPLETION.json'
PREVIOUS_SHA = 'cbc61cf6bd4501dedb3142be5c672a1b2f196ade55f42f9a3b14df66190fb749'
PROPOSAL = 'work/monthly_endpoint155_proposal.json'
PRE_REVIEW = 'work/monthly_endpoint155_prepare_prereview.json'
OUT_REVIEW = 'work/monthly_endpoint155_outcome_prereview.json'
NAMES = ('incumbent_local67', 'matched_mean_return_ridge', 'matched_loss_indicator_ridge')
MODES = ('same_close', 'next_open_zero', 'next_open_cost')
PROTECTED_COUNT = 21444
SUITES = {name:'test_monthly_endpoint155_'+name+'.py' for name in
          ('model', 'runner', 'prepare_review', 'outcome_review', 'lifecycle')}
PREPARE_CHECKS = ('original27_arrays_byte_identical', 'training_identity_and_plain10_verified',
    'raw_prices_independently_verified', 'calendar_overlay_independently_verified',
    'endpoint_labels_independently_reconstructed', 'shared_maturity_rows_weights_scalers_verified',
    'annual_forecasts_independently_reconstructed', 'shared_query_mask_and_centered_ranks_verified',
    'all_three_score_matrices_and_ordered_choices_independently_verified',
    'all_signal_prefixes_verified', 'initial_partial_and_missing_exact_fallback_verified',
    'one_episode_and_top2_lock_verified')
OUTCOME_CHECKS = ('daily_nav_trades_holdings_metrics_slices', 'first88_and_three_fold_selection_verified',
    'annual_partial_flags_verified', 'monthly_difference_and_attribution_verified',
    'first88_causal_replay_verified', 'model_fit_counts_verified', 'all_changed_months_reported',
    'diagnostic_start_marker_verified', 'baseline_local67_full_orders_unchanged')
require = old.require
sha = old.sha


def runner():
    import monthly_endpoint155_runner
    return monthly_endpoint155_runner


def publish_bytes(path, raw):
    """Publish complete bytes exclusively, never a partially written final file."""
    path=Path(path); temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.endpoint155-',delete=False) as stream:
            temporary=Path(stream.name);stream.write(raw);stream.flush();os.fsync(stream.fileno())
        os.link(str(temporary),str(path))
        fd=os.open(str(path.parent),os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)


def write(path, value):
    publish_bytes(path,old.encoded(value))


def state_snapshot(path):
    raw = Path(path).read_bytes()
    return old.decode(raw), hashlib.sha256(raw).hexdigest()


@contextmanager
def state_lock(root):
    with (Path(root)/'outputs/monthly_target70_research/.lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def predecessor(state):
    items = state.get('completed')
    require(isinstance(items, list) and len(items) == 154 and
            all(isinstance(x, dict) and type(x.get('trials')) is int and x['trials'] >= 0 for x in items)
            and sum(x['trials'] for x in items) == 975, 'Expected154 rounds/975 trials')
    require(items[-1].get('directory') == 'monthly_financial_coverage154' and
            items[-1].get('completion', {}).get('sha256') == PREVIOUS_SHA, 'Wrong predecessor')
    require(not any(x.get('directory') == STUDY for x in items), 'Already completed155')


def update_state(path, expected, value):
    path = Path(path)
    require(sha(path) == expected, 'Concurrent continuation update')
    backup = path.parent/('PRE155_'+expected+'.json')
    if not backup.exists():
        publish_bytes(backup,path.read_bytes())
    require(sha(backup) == expected, 'Original-state backup mismatch')
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(old.encoded(value)); stream.flush(); os.fsync(stream.fileno())
    try:
        require(sha(path) == expected, 'Concurrent continuation commit')
        os.replace(str(temporary), str(path))
    finally:
        temporary.unlink(missing_ok=True)


def validate_tests(doc, reader):
    require(doc.get('passed') is True and doc.get('synthetic_only') is True, 'Synthetic-only tests required')
    found = {}
    for row in doc['suites']:
        name, count = row.get('name'), row.get('count')
        require(name in SUITES and name not in found and row.get('passed') is True and
                type(count) is int and count > 0, 'Unique passed positive suite required')
        for field in ('source', 'log'):
            ref = row[field]
            require(reader.hash(ref['path']) == ref['sha256'], 'Changed synthetic '+field)
        require(Path(row['source']['path']).name == SUITES[name], 'Wrong suite source')
        log = reader.path(row['log']['path']).read_text()
        require(re.search(r'Ran '+str(count)+r' tests? in ', log) and log.rstrip().endswith('OK'),
                'Test log does not establish success')
        found[name] = count
    require(set(found) == set(SUITES) and type(doc.get('test_count')) is int and
            doc['test_count'] == sum(found.values()), 'Complete five-suite accounting required')
    return found


def pin_sources(reader, values):
    if isinstance(values, dict):
        return reader.verify(values)
    require(isinstance(values, (list, tuple)) and bool(values), 'Nonempty source closure required')
    return {str(reader.path(p)):reader.hash(p) for p in values}


def prereview(reader, path, stage, proposal_sha):
    review = reader.read(path)
    require(review.get('passed') is True and review.get('round') == 155 and
            review.get('stage') == 'preregistration_'+stage+'_review' and
            review.get('proposal_sha256') == proposal_sha and review.get('synthetic_only') is True,
            'Wrong prospective review scope')
    require(type(review.get('real_155_label_fit_or_return_calculations')) is int and
            review['real_155_label_fit_or_return_calculations'] == 0, 'No real calculations before registration')
    sources = reader.verify(review['source_hashes'])
    required = reader.root/'work'/('monthly_endpoint155_'+stage+'_review.py')
    require(sources.get(str(required)) == reader.hash(required), 'Missing reviewer implementation')
    implementations = reader.verify(review['implementation_hashes'])
    for name in ('monthly_endpoint155_model.py', 'monthly_endpoint155_runner.py', 'monthly_endpoint155_lifecycle.py'):
        p = reader.root/'work'/name
        require(implementations.get(str(p)) == reader.hash(p), 'Unreviewed implementation '+name)
    inputs = reader.verify(review['input_hashes'])
    for p in (reader.root/PROPOSAL, reader.root/DESIGN):
        require(inputs.get(str(p)) == reader.hash(p), 'Unreviewed prospective input '+str(p))
    return sources


def build_plan(root, tests, proposal_sha):
    r = runner(); root = Path(root).resolve(); reader = old.Reader(root)
    out = root/'outputs'/STUDY
    require(re.fullmatch('[0-9a-f]{64}', proposal_sha or '') is not None, 'Final proposal SHA required')
    require(not any((out/p).exists() for p in ('PLAN.json','CACHE.json','RESULTS.json','FROZEN.json',*r.CACHE_FILES)),
            'Registration must precede actual preparation')
    state, state_sha = state_snapshot(root/STATE); predecessor(state)
    require(state.get('registered_queue') == [], 'Pending experiment')
    require(reader.hash(PREVIOUS) == PREVIOUS_SHA, 'Previous completion changed')
    require(reader.hash(BASE_PLAN) == BASE_SHA, 'Frozen143 PLAN changed')
    base = reader.read(BASE_PLAN)
    reader.verify(base['hashes'])
    require(len(base['raw_source_hashes']) == 64, 'Original64 raw prices required')
    reader.verify(base['raw_source_hashes'])
    require(reader.hash(DESIGN) == DESIGN_SHA, 'Prospective design changed')
    design = reader.read(DESIGN); reader.verify(design['input_hashes'])
    require(reader.hash(PROPOSAL) == proposal_sha, 'Final proposal changed')
    r.validate_proposal(reader.read(PROPOSAL))
    pin_sources(reader, r.source_files(root))
    for p in (Path(__file__), Path(old.__file__), root/'work/test_monthly_endpoint155_lifecycle.py',
              root/'work/prepare_monthly_endpoint155_proposal.py'):
        reader.hash(p)
    independent = {stage:prereview(reader, path, stage, proposal_sha) for stage,path in
                   (('prepare',PRE_REVIEW),('outcome',OUT_REVIEW))}
    require(reader.path(tests) == out/'TESTS.json', 'Canonical tests manifest required')
    validate_tests(reader.read(tests), reader)
    inputs = {key:dict(base['inputs'][key]) for key in r.INPUTS if key in base['inputs'] and key != 'proposal'}
    for key,p in (('proposal',PROPOSAL),('design',DESIGN),('calendar','outputs/monthly_paired_target143/CALENDAR.json')):
        inputs[key] = dict(path=str(root/p), sha256=reader.hash(p))
    require(set(inputs) == set(r.INPUTS), 'Named input schema mismatch')
    for ref in inputs.values():
        require(reader.hash(ref['path']) == ref['sha256'], 'Named input changed')
        ref['path'] = str(reader.path(ref['path']))
    reader.verify(base['calendar']['source_hashes'])
    reader.verify(dict(reader.seen))
    hashes = dict(reader.seen); hashes.pop(str(root/STATE),None)
    return dict(round=155,status='REGISTERED',registered_utc=datetime.now(timezone.utc).isoformat(),
        registration_precondition=dict(path=STATE,sha256=state_sha),
        arms=list(NAMES),primary_challenger=NAMES[2],output_directory=str(out),inputs=inputs,
        hashes=hashes,raw_source_hashes=base['raw_source_hashes'],calendar=base['calendar'],
        independent_prepare_source_hashes=independent['prepare'],independent_outcome_source_hashes=independent['outcome'],
        prerequisite_reviews=[PRE_REVIEW,OUT_REVIEW],tests=str(out/'TESTS.json'),
        runtime=dict(python=sys.version,numpy=r.np.__version__,pandas=r.pd.__version__),
        tolerances=dict(r.TOLERANCES),actual_label_calculations_before_registration=0,
        actual_model_fits_before_registration=0,actual_feature_calculations_before_registration=0,
        actual_returns_before_registration=0,new_market_downloads=0,no_observer_writes=True)


def queue_entry(plan, plan_sha):
    return dict(round=155,directory=STUDY,plan=dict(path='outputs/'+STUDY+'/PLAN.json',sha256=plan_sha),
                registered_utc=plan['registered_utc'])


def enqueue(root, plan_sha):
    root = Path(root).resolve(); reader = old.Reader(root); path = root/'outputs'/STUDY/'PLAN.json'
    require(reader.hash(path) == plan_sha, 'Registered PLAN changed')
    plan = reader.read(path)
    require(plan.get('round') == 155 and plan.get('status') == 'REGISTERED' and plan.get('arms') == list(NAMES)
            and plan.get('primary_challenger') == NAMES[2] and reader.path(plan['output_directory']) == path.parent,
            'Wrong registered experiment')
    require(plan['registration_precondition']['path'] == STATE, 'Wrong state precondition')
    reader.verify(plan['hashes'])
    state, current_sha = state_snapshot(root/STATE); predecessor(state)
    entry = queue_entry(plan,plan_sha)
    if state.get('registered_queue') == [entry]:
        return dict(registered=True,plan_sha256=plan_sha,already_enqueued=True)
    require(state.get('registered_queue') == [], 'Conflicting queue')
    require(current_sha == plan['registration_precondition']['sha256'], 'Stale registration state')
    state.update(registered_queue=[entry],current=STUDY,current_status='registered_before_actual_preparation',
                 updated_utc=plan['registered_utc'])
    update_state(root/STATE,current_sha,state)
    return dict(registered=True,plan_sha256=plan_sha,already_enqueued=False)


def register(root, tests, proposal_sha):
    plan = build_plan(root,tests,proposal_sha); out = Path(root)/'outputs'/STUDY
    out.mkdir(exist_ok=True); write(out/'PLAN.json',plan)
    return enqueue(root,sha(out/'PLAN.json'))


def finite(value):
    return type(value) in (int,float) and math.isfinite(value)


def selection(rows, open_rows):
    for group in (rows,open_rows):
        require(isinstance(group,list) and len(group)==3 and [r.get('name') for r in group]==list(NAMES),
                'Three ordered first88 arms required')
    fields=(('training','cagr'),('training','sharpe'),('training','max_drawdown'),('early','cagr'),('recent','cagr'))
    def compare(candidate,control,strict):
        a=[candidate[k].get(v) for k,v in fields]; b=[control[k].get(v) for k,v in fields]
        if not all(finite(x) for x in a+b):return False
        return (a[0]>b[0]+1e-12 if strict else a[0]>=b[0]-1e-12) and all(
            x>=y-t for x,y,t in zip(a[1:],b[1:],(.05,.03,.025,.025)))
    guards={name:compare(rows[2],rows[i],True) for i,name in enumerate(NAMES[:2])}
    opening={name:compare(open_rows[2],open_rows[i],i==1) for i,name in enumerate(NAMES[:2])}
    admitted=all(guards.values()) and all(opening.values())
    return dict(primary=NAMES[2] if admitted else NAMES[0],admitted=admitted,guards=admitted,
                guards_against=guards,next_open_guards_against=opening,mechanism_confirmed=all(opening.values()))


def count(obj,key,expected):
    require(type(obj.get(key)) is int and obj[key]==expected,'Wrong scope '+key)


def fitted_models(audit):
    require(isinstance(audit.get('annual'),list),'Annual model audit required')
    years=set(); n=0
    for row in audit['annual']:
        require(type(row.get('year')) is int and row['year'] not in years and
                row.get('status') in ('fit','cold_start','no_queries'),'Unique annual fit status required')
        years.add(row['year']); n+=2*(row['status']=='fit')
    count(audit,'new_pooled_model_fits',n)
    return n


def required_outcome_files(r):
    files=set((*r.CACHE_FILES,*r.FROZEN_FILES,'CACHE.json','FROZEN.json','DIAGNOSE_STARTED.json',
               'RESULTS.json','MONTH_COMPARISON.json','ATTRIBUTION.json','VALIDATION.json'))
    files.update(name+'_'+mode+'_'+kind+'.json' for name in NAMES for mode in MODES
                 for kind in ('DAILY','TRADES','HOLDINGS'))
    return files


def receipt(reader,report,stage,checks,plan,out,plan_sha):
    require(report.get('passed') is True and report.get('stage')==stage and
            report.get('producer_modules_imported') is False and
            all(report.get(key) is True for key in checks),'Incomplete independent '+stage+' proof')
    require(report.get('plan_sha256')==plan_sha and report.get('cache_sha256')==reader.hash(out/'CACHE.json')
            and report.get('orders_sha256')==reader.hash(out/'PRETRAIN_ORDERS.json'),'Independent artifact binding')
    kind='prepare' if stage=='prepare' else 'outcome'
    require(reader.verify(report['source_hashes'])==reader.verify(plan['independent_'+kind+'_source_hashes']),
            'Independent source closure changed')
    return reader.verify(report['artifact_hashes'])


def complete(root, plan_sha):
    r=runner(); root=Path(root).resolve(); out=root/'outputs'/STUDY; reader=old.Reader(root)
    require(reader.hash(out/'PLAN.json')==plan_sha,'Exact PLAN hash required')
    # This method only verifies pins/runtime/cache/receipts; it does not recalculate features.
    r.Runner(root,out/'PLAN.json',plan_sha).require_prepare_review()
    plan=reader.read(out/'PLAN.json'); pins=reader.verify(plan['hashes'])
    for p in (root/'work/monthly_endpoint155_lifecycle.py', root/'work/test_monthly_endpoint155_lifecycle.py',
              root/'work/complete_monthly_same_report145.py'):
        require(pins.get(str(p))==reader.hash(p),'Unbound lifecycle dependency '+str(p))
    require(reader.hash(root/'work/monthly_endpoint155_lifecycle.py')==sha(__file__),
            'Executing lifecycle differs from registered source')
    for key in ('actual_label_calculations_before_registration','actual_model_fits_before_registration',
                'actual_feature_calculations_before_registration','actual_returns_before_registration',
                'new_market_downloads'):
        count(plan,key,0)
    require(plan.get('no_observer_writes') is True,'Registered observer protection required')
    cache=reader.read(out/'CACHE.json'); require(set(cache)==set(r.CACHE_FILES),'Complete cache required')
    cached=reader.verify(cache,out)
    prep=reader.read(out/'INDEPENDENT_PREPARE.json'); outcome=reader.read(out/'INDEPENDENT_OUTCOME.json')
    pa=receipt(reader,prep,'prepare',PREPARE_CHECKS,plan,out,plan_sha)
    oa=receipt(reader,outcome,'evaluate',OUTCOME_CHECKS,plan,out,plan_sha)
    require(prep.get('returns_or_nav_computed') is False,'Preparation cannot compute returns')
    fits=fitted_models(reader.read(out/'MODEL_AUDIT.json'))
    for key,n in (('orders_checked',366),('candidate_cells_checked',5978),('training_rows_checked',8820),
                  ('months_checked',122),('prefixes_checked',123),('forecast_arrays_verified',5),('new_model_fits',fits)):
        count(prep,key,n)
    for key,n in (('execution_paths_verified',9),('total_monthly_orders_verified',366),
                  ('endpoint_forecast_arrays_verified',5),('models_fit',0),('producer_pooled_model_fits',fits)):
        count(outcome,key,n)
    for p,h in {**cached,**{str(out/f):reader.hash(out/f) for f in ('PLAN.json','CACHE.json')}}.items():
        require(pa.get(p)==h,'Missing independently prepared artifact '+p)
    for f in required_outcome_files(r):
        require(oa.get(str(out/f))==reader.hash(out/f),'Missing independently replayed artifact '+f)
    frozen=reader.read(out/'FROZEN.json')
    require(set(frozen['hashes'])==set(r.FROZEN_FILES),'Exact frozen selection scope')
    reader.verify(frozen['hashes'],out)
    require(frozen.get('pristine_oos') is False and frozen.get('all_diagnostic_orders_frozen') is True,
            'Frozen reused-sample disclosure required')
    chosen=selection(reader.read(out/'TRAIN_TRIALS.json'),reader.read(out/'TRAIN_OPEN_TRIALS.json'))
    require(frozen['selection']==chosen and outcome.get('selected_name')==chosen['primary'],
            'First88 selected primary mismatch')
    require(outcome.get('frozen_sha256')==reader.hash(out/'FROZEN.json'),'Frozen audit binding')
    marker=reader.read(out/'DIAGNOSE_STARTED.json')
    require(marker.get('plan_sha256')==plan_sha and marker.get('frozen_sha256')==reader.hash(out/'FROZEN.json')
            and outcome.get('diagnose_started_sha256')==reader.hash(out/'DIAGNOSE_STARTED.json'),'Diagnostic start binding')
    times=[datetime.fromisoformat(x) for x in (plan['registered_utc'],marker['started_utc'],outcome['audit_started_utc'])]
    require(all(x.tzinfo is not None for x in times) and times[0]<times[1]<times[2]<=datetime.now(timezone.utc),
            'Registration/diagnostic/independent audit chronology')
    results=reader.read(out/'RESULTS.json')
    require(set(results)==set(NAMES) and all(set(results[n])==set(MODES) for n in NAMES),'Nine execution paths required')
    primary=chosen['primary']; metric=results[primary]['same_close']['metrics']
    require(all(finite(metric.get(k)) for k in ('cagr','sharpe','max_drawdown')),'Finite selected metrics')
    require(finite(outcome.get('primary_cagr')) and abs(metric['cagr']-outcome['primary_cagr'])<1e-12
            and type(outcome.get('primary_target_reached')) is bool and
            outcome['primary_target_reached']==(metric['cagr']>=.7),'Independent selected target mismatch')
    orders=reader.read(out/'PRETRAIN_ORDERS.json')
    require(set(orders)==set(NAMES) and all(len(orders[n])==122 for n in NAMES),'366 ordered choices required')
    require(orders[NAMES[0]]==reader.read(plan['inputs']['original_orders']['path']),'Original orders changed')
    require(all(orders[n][0]==orders[NAMES[0]][0] for n in NAMES[1:]),'Initial partial query must fall back exactly')
    validation=reader.read(out/'VALIDATION.json')
    require(validation.get('selection')==chosen and validation.get('no_observer_writes') is True and
            validation.get('no_intramonth_trades') is True,'Final execution invariants')
    count(validation,'new_model_fits',fits)
    suites=validate_tests(reader.read(out/'TESTS.json'),reader)
    manifest=reader.read(plan['inputs']['protected_manifest']['path'])
    require(len(manifest)==PROTECTED_COUNT,'Full protected archive required')
    proof=dict(passed=True,checked=PROTECTED_COUNT,verified=PROTECTED_COUNT,manifest_sha256=plan['inputs']['protected_manifest']['sha256'],
               changed=[],missing=[],read_or_identity_errors=[])
    old.protected_scan(reader,plan,proof)
    if (out/'PROTECTED_AFTER.json').exists():
        require(reader.read(out/'PROTECTED_AFTER.json')==proof,'Existing protection receipt changed')
    else:write(out/'PROTECTED_AFTER.json',proof)
    state,state_sha=state_snapshot(root/STATE)
    already=bool(state.get('completed') and state['completed'][-1].get('directory')==STUDY)
    if already:
        prior=dict(state,completed=state['completed'][:-1]);predecessor(prior)
        require(len(state['completed'])==155 and sum(x['trials'] for x in state['completed'])==978
                and state.get('registered_queue')==[] and state.get('current')==STUDY
                and state.get('current_status')=='completed_independently_verified'
                and (out/'COMPLETION.json').is_file(),'Conflicting completed continuation state')
    else:
        predecessor(state)
        require(state['registered_queue']==[queue_entry(plan,plan_sha)],'Exact registered queue required')
    done=dict(round=155,directory=STUDY,status='completed_independently_verified',
        completed_utc=datetime.now(timezone.utc).isoformat(),plan_sha256=plan_sha,selection=chosen,
        all_arm_results=results,primary=primary,primary_cagr=metric['cagr'],primary_sharpe=metric['sharpe'],
        primary_next_open_cost=results[primary]['next_open_cost']['metrics'],target_reached=metric['cagr']>=.7,
        trials=3,new_model_fits=fits,independent_execution_paths=9,protected_files=PROTECTED_COUNT,synthetic_suites=suites,
        independent_outcome_sha256=reader.hash(out/'INDEPENDENT_OUTCOME.json'),
        protected_sha256=reader.hash(out/'PROTECTED_AFTER.json'),
        limitations=[
            'Bounded retest, not established novelty; round32/116 original model source remains unavailable.',
            'Both targets inherit historical common_good restrictions, pooled model and shared plain10 sample.',
            'The binary Ridge output is an uncalibrated ranking score, not a guaranteed loss probability.',
            'First88 selection and later34 diagnostics are repeatedly reused historical evidence, not pristine OOS.',
            'Universe proxies, historical source vintages and idealized same-close zero-cost fills remain limitations.',
            'Initial partial query uses incumbent fallback; final partial replay never shortens label maturity.',
            'Original frozen observation accounts and their live ledgers were not modified.'])
    reader.verify(dict(reader.seen))
    if (out/'COMPLETION.json').exists():
        existing=reader.read(out/'COMPLETION.json')
        require({k:v for k,v in existing.items() if k!='completed_utc'}==
                {k:v for k,v in done.items() if k!='completed_utc'},'Existing completion conflicts')
        done=existing
    else:write(out/'COMPLETION.json',done)
    ref=dict(path=str((out/'COMPLETION.json').relative_to(root)),sha256=sha(out/'COMPLETION.json'))
    entry={k:done[k] for k in ('directory','completed_utc','primary','primary_cagr','primary_sharpe','target_reached',
                               'trials','new_model_fits','independent_execution_paths','protected_files')}
    entry.update(completion=ref,new_admitted=int(chosen['admitted']))
    if already:
        require(state['completed'][-1]==entry and state.get('latest_round_receipt')==ref,
                'Existing completed state does not match reverified evidence')
        require(sha(root/STATE)==state_sha,'State changed during idempotent verification')
        return dict(completion=ref,primary=primary,cagr=metric['cagr'],sharpe=metric['sharpe'],
                    target_reached=done['target_reached'],already_completed=True)
    state['completed'].append(entry)
    require(len(state['completed'])==155 and sum(x['trials'] for x in state['completed'])==978,'Final155/978 accounting')
    state.update(current=STUDY,current_status='completed_independently_verified',registered_queue=[],
        latest_round_receipt=ref,updated_utc=done['completed_utc'],last_goal_turn_classification='progress')
    if chosen['admitted'] and metric['cagr']>state['best_training_selected_candidate']['cagr']:
        state['best_training_selected_candidate']=dict(directory=STUDY,spec=primary,cagr=metric['cagr'],
            sharpe=metric['sharpe'],max_drawdown=metric['max_drawdown'],
            next_open_5bp_cagr=results[primary]['next_open_cost']['metrics']['cagr'],observer_promotion=False)
    update_state(root/STATE,state_sha,state)
    return dict(completion=ref,primary=primary,cagr=metric['cagr'],sharpe=metric['sharpe'],target_reached=done['target_reached'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('register','enqueue','complete'))
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--plan-sha'); parser.add_argument('--tests'); parser.add_argument('--proposal-sha')
    args=parser.parse_args()
    with state_lock(args.root):
        if args.phase=='register': result=register(args.root,args.tests,args.proposal_sha)
        elif args.phase=='enqueue': result=enqueue(args.root,args.plan_sha)
        else: result=complete(args.root,args.plan_sha)
    print(old.encoded(result).decode())


if __name__=='__main__':main()
