"""Prospective registration and verified closure for the fixed revenue pilot.

Only registration/closure metadata is handled here. Actual preparation,
first88 selection, later34 diagnostics and independent audits are separate.
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
STUDY = 'monthly_revenue_persistence156'
STATE = 'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
PREVIOUS_PLAN = 'outputs/monthly_endpoint155/PLAN.json'
PREVIOUS_PLAN_SHA = 'f10890b99c5b27d176eaa9467152358a2c3bd5b845c0bb4c9aa6865512f29201'
PREVIOUS_DONE = 'outputs/monthly_endpoint155/COMPLETION.json'
PREVIOUS_DONE_SHA = 'a0ebeb12d7bca362b2995c377382cdd40449c13f4b2b4bbbb0a98b5a4717dc06'
DESIGN = 'work/after155_revenue_experiment_path.json'
DESIGN_SHA = 'a84440879cf3db80782190de8eb7b43c402c25bdb88fa07e1d9b3401bf092806'
SOURCE_PLAN = 'work/after155_revenue_pilot_source_plan.json'
SOURCE_PLAN_SHA = '5ad9628bba2ad422fd38e31e305eaa7dcd68e0e9195cd68e26fcdaa49cf5ce10'
SOURCE_CALENDAR = 'outputs/monthly_paired_target143/CALENDAR.json'
SOURCE_CALENDAR_SHA = '8ee4da69fba84db1edee37a7e0f5af5d3ab16c6257e5d37cddbd531f01404622'
SOURCE_ROOT = 'outputs/monthly_revenue_pilot_source156'
SOURCE_MANIFEST = SOURCE_ROOT+'/MANIFEST.json'
SOURCE_REVIEW = 'work/after155_revenue_pilot_output_independent.json'
PROPOSAL = 'work/monthly_revenue_persistence156_proposal.json'
SELF = 'work/monthly_revenue_persistence156_lifecycle.py'
PROPOSAL_BUILDER = 'work/prepare_revenue_persistence156_proposal.py'
SESSIONS = 'outputs/monthly_forward_guidance_preflight_v1/SESSIONS.json'
SESSIONS_SHA = 'c05c15f19bc1f7a3428e456a71199d64f05799ef3c432fa45f7bc485d0ff3cf4'
NAMES = ('incumbent_local67', 'matched_latest_revenue_growth',
         'matched_annualized_total_revenue_growth', 'weakest_annual_revenue_growth')
MODES = ('same_close', 'next_open_zero', 'next_open_cost')
PREREVIEWS = {kind: 'work/monthly_revenue_persistence156_'+kind+'_prereview.json'
             for kind in ('runner', 'prepare', 'outcome')}
FACTOR_REVIEW = 'work/after155_revenue_factor_independent.json'
SUITES = {kind: 'test_monthly_revenue_persistence156_'+kind+'.py'
          for kind in ('factor', 'runner', 'prepare_review', 'outcome_review', 'lifecycle', 'proposal')}
SUITES['factor'] = 'test_after155_revenue_pilot_factor.py'
OUTCOME_CHECKS = ('daily_nav_trades_holdings_metrics_slices', 'first88_and_three_fold_selection_verified',
    'first88_causal_replay_verified', 'zero_model_fit_count_verified', 'all_changed_months_reported',
    'diagnostic_start_marker_verified', 'baseline_local67_full_orders_unchanged',
    'revenue_source_grid_verified', 'all_three_revenue_features_common_mask', 'original_source_cutoffs_verified')
require, sha = old.require, old.sha


def runner():
    import monthly_revenue_persistence156_runner
    return monthly_revenue_persistence156_runner


def publish_bytes(path, raw):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.revenue156-', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.link(str(temporary), str(path))
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write(path, value):
    publish_bytes(path, old.encoded(value))


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
    rows = state.get('completed')
    require(isinstance(rows, list) and len(rows) == 155 and
            all(isinstance(r, dict) and type(r.get('trials')) is int and r['trials'] >= 0 for r in rows)
            and sum(r['trials'] for r in rows) == 978, 'Expected completed155 /978 trials')
    require(rows[-1].get('directory') == 'monthly_endpoint155' and
            rows[-1].get('completion', {}).get('sha256') == PREVIOUS_DONE_SHA, 'Wrong finalized predecessor')
    require(not any(r.get('directory') == STUDY for r in rows), 'Already completed156')


def update_state(path, expected, value):
    path = Path(path)
    require(sha(path) == expected, 'Concurrent continuation update')
    backup = path.parent/('PRE156_'+expected+'.json')
    if not backup.exists():
        publish_bytes(backup, path.read_bytes())
    require(sha(backup) == expected, 'Original state backup mismatch')
    raw = old.encoded(value)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    try:
        require(sha(path) == expected, 'Concurrent continuation commit')
        os.replace(str(temporary), str(path))
    finally:
        temporary.unlink(missing_ok=True)


def exact_count(doc, key, count):
    require(type(doc.get(key)) is int and doc[key] == count, 'Wrong exact count: '+key)


def validate_tests(doc, reader):
    require(doc.get('passed') is True and doc.get('synthetic_only') is True, 'Synthetic-only tests required')
    found = {}
    for row in doc['suites']:
        name, count = row.get('name'), row.get('count')
        require(name in SUITES and name not in found and row.get('passed') is True and
                type(count) is int and count > 0, 'Distinct successful positive synthetic suite required')
        for key in ('source', 'log'):
            ref = row[key]
            require(reader.hash(ref['path']) == ref['sha256'], 'Changed synthetic '+key)
        require(Path(row['source']['path']).name == SUITES[name], 'Wrong synthetic suite source')
        log = reader.path(row['log']['path']).read_text()
        require(re.search(r'Ran '+str(count)+r' tests? in ', log) and log.rstrip().endswith('OK'),
                'Synthetic log does not establish success')
        found[name] = count
    require(set(found) == set(SUITES), 'All six synthetic suites required')
    exact_count(doc, 'test_count', sum(found.values()))
    return found


def pin_sources(reader, values):
    if isinstance(values, dict):
        return reader.verify(values)
    require(isinstance(values, (list, tuple)) and values, 'Nonempty source closure required')
    return {str(reader.path(p)): reader.hash(p) for p in values}


def source_gate(reader, review_ref):
    """Check already reviewed metadata without evaluating factors or coverage."""
    require(reader.hash(review_ref['path']) == review_ref['sha256'], 'Source grid review changed')
    review = reader.read(review_ref['path'])
    require(review.get('passed') is True and review.get('actual_grid_verified') is True,
            'Independent actual source-grid review required')
    reviewed = reader.verify(review['input_hashes'])
    manifest = reader.read(SOURCE_MANIFEST)
    require(manifest.get('source_only') is True and manifest.get('strategy_registration') is False,
            'Source-only manifest required before strategy registration')
    source_inputs = reader.verify(manifest['input_hashes'])
    require(set(manifest['outputs']) == {'ROWS.json', 'REPORTS.json', 'EDGES.json', 'COVERAGE_GATE.json', 'SUMMARY.json'},
            'Complete source-only pilot package required')
    required = dict(source_inputs)
    required[str(reader.path(SOURCE_MANIFEST))] = reader.hash(SOURCE_MANIFEST)
    for name, ref in manifest['outputs'].items():
        p = reader.path(SOURCE_ROOT+'/'+name)
        require(reader.hash(p) == ref['sha256'] and p.stat().st_size == ref['bytes'], 'Source output changed')
        required[str(p)] = ref['sha256']
    require(all(reviewed.get(p) == h for p, h in required.items()), 'Source review omitted exact input/output closure')
    gate = reader.read(SOURCE_ROOT+'/COVERAGE_GATE.json')
    require(gate.get('source_coverage_floor_passed') is True and gate.get('rows_verified') == 5978 and
            gate.get('source_reviews_verified_by_preparation_driver') is True,
            'Predeclared first88 source coverage floor not passed')
    require(reader.hash(SOURCE_PLAN) == SOURCE_PLAN_SHA and reader.hash(SOURCE_CALENDAR) == SOURCE_CALENDAR_SHA,
            'Frozen source scope/calendar changed')
    return manifest


def prereview(reader, path, stage, proposal_sha):
    doc = reader.read(path)
    require(doc.get('passed') is True and doc.get('round') == 156 and
            doc.get('stage') == 'preregistration_'+stage+'_review' and doc.get('synthetic_only') is True and
            doc.get('proposal_sha256') == proposal_sha, 'Wrong prospective '+stage+' review scope')
    for key in ('actual_factors_calculated', 'actual_returns_calculated', 'actual_orders_calculated', 'actual_model_fits'):
        exact_count(doc, key, 0)
    sources = reader.verify(doc['source_hashes'])
    implementations = reader.verify(doc['implementation_hashes'])
    for name in ('after155_revenue_pilot_factor.py', 'monthly_revenue_persistence156_runner.py',
                 'monthly_revenue_persistence156_lifecycle.py', 'prepare_revenue_persistence156_proposal.py'):
        p = reader.path('work/'+name)
        require(implementations.get(str(p)) == reader.hash(p), 'Unreviewed prospective implementation: '+name)
    inputs = reader.verify(doc['input_hashes'])
    for p in (PROPOSAL, DESIGN, SOURCE_PLAN, SOURCE_MANIFEST):
        require(inputs.get(str(reader.path(p))) == reader.hash(p), 'Unreviewed prospective input: '+p)
    return sources


def factor_review(reader):
    doc = reader.read(FACTOR_REVIEW)
    require(doc.get('passed') is True and doc.get('synthetic_only') is True and
            doc.get('status') == 'PURE_REVENUE_FACTOR_REVIEW_PASSED', 'Independent pure factor review required')
    for key in ('actual_factors_computed', 'actual_financial_values_read', 'actual_orders_computed',
                'actual_returns_computed', 'frozen_files_changed', 'observer_changes', 'strategy_registrations'):
        exact_count(doc['activity_counts'], key, 0)
    inputs = reader.verify(doc['input_hashes'])
    for p in ('work/after155_revenue_pilot_factor.py', 'work/test_after155_revenue_pilot_factor.py', DESIGN):
        require(inputs.get(str(reader.path(p))) == reader.hash(p), 'Factor review omitted current contract')


def build_plan(root, tests, proposal_sha):
    r = runner(); root = Path(root).resolve(); reader = old.Reader(root)
    out = root/'outputs'/STUDY
    require(old.digest(proposal_sha), 'Exact final prospective proposal SHA required')
    require(not any((out/name).exists() for name in ('PLAN.json', 'CACHE.json', 'RESULTS.json', 'FROZEN.json', *r.CACHE_FILES)),
            'Registration must precede actual preparation')
    state, state_sha = state_snapshot(root/STATE)
    predecessor(state)
    require(state.get('registered_queue') == [], 'Empty registration queue required')
    require(reader.hash(PREVIOUS_DONE) == PREVIOUS_DONE_SHA and reader.hash(PREVIOUS_PLAN) == PREVIOUS_PLAN_SHA,
            'Finalized155 predecessor changed')
    previous = reader.read(PREVIOUS_PLAN)
    reader.verify(previous['hashes'])
    require(len(previous['raw_source_hashes']) == 64, 'Original64 raw-price sources required')
    reader.verify(previous['raw_source_hashes'])
    runtime = dict(python=sys.version, numpy=r.np.__version__, pandas=r.pd.__version__)
    require(runtime == previous['runtime'], 'Registered local runtime changed')
    require(reader.hash(PROPOSAL) == proposal_sha, 'Prospective proposal changed')
    proposal = reader.read(PROPOSAL)
    r.validate_proposal(proposal)
    require(reader.hash(DESIGN) == proposal['design']['sha256'] == DESIGN_SHA, 'Predeclared revenue design changed')
    source_refs = proposal.get('source_inputs', {})
    expected_source_paths = dict(source_manifest=SOURCE_MANIFEST, source_grid_review=SOURCE_REVIEW,
                                 source_plan=SOURCE_PLAN, source_calendar=SOURCE_CALENDAR)
    require(set(source_refs) == set(expected_source_paths), 'Exact source input names required')
    for key, path in expected_source_paths.items():
        require(reader.path(source_refs[key]['path']) == reader.path(path) and
                reader.hash(path) == source_refs[key]['sha256'], 'Canonical source input changed: '+key)
    source_gate(reader, source_refs['source_grid_review'])
    pin_sources(reader, r.source_files(root))
    for p in (SELF, PROPOSAL_BUILDER, 'work/test_monthly_revenue_persistence156_lifecycle.py',
              'work/test_monthly_revenue_persistence156_proposal.py', Path(old.__file__)):
        reader.hash(p)
    factor_review(reader)
    independent = {stage: prereview(reader, path, stage, proposal_sha) for stage, path in PREREVIEWS.items()}
    require(reader.path(tests) == out/'TESTS.json', 'Canonical tests manifest required')
    validate_tests(reader.read(tests), reader)
    inputs = {k: dict(previous['inputs'][k]) for k in
              ('meta', 'policy', 'original_orders', 'prices', 'protected_manifest')}
    inputs.update({key: dict(value) for key, value in proposal['source_inputs'].items()})
    inputs.update(proposal=dict(path=PROPOSAL, sha256=proposal_sha),
                  sessions=dict(path=SESSIONS, sha256=SESSIONS_SHA))
    require(set(inputs) == set(r.INPUTS), 'Registered named input schema mismatch')
    for key, ref in inputs.items():
        require(reader.hash(ref['path']) == ref['sha256'], 'Changed named input: '+key)
        ref['path'] = str(reader.path(ref['path']))
    reader.verify(dict(reader.seen))
    hashes = dict(reader.seen); hashes.pop(str(root/STATE), None)
    return dict(round=156, status='REGISTERED', registered_utc=datetime.now(timezone.utc).isoformat(),
        registration_precondition=dict(path=STATE, sha256=state_sha), arms=list(NAMES),
        primary_challenger=NAMES[3], dose=.01, tolerances=dict(r.TOLERANCES), runtime=runtime,
        output_directory=str(out), hashes=hashes, inputs=inputs, raw_source_hashes=previous['raw_source_hashes'],
        independent_prepare_source_hashes=independent['prepare'], independent_outcome_source_hashes=independent['outcome'],
        prerequisite_reviews=dict(PREREVIEWS), tests=str(out/'TESTS.json'),
        actual_feature_calculations_before_registration=0, actual_returns_before_registration=0,
        actual_model_fits_before_registration=0, new_market_downloads=0, no_observer_writes=True)


def queue_entry(plan, plan_sha):
    return dict(round=156, directory=STUDY, plan=dict(path='outputs/'+STUDY+'/PLAN.json', sha256=plan_sha),
                registered_utc=plan['registered_utc'])


def enqueue(root, plan_sha):
    root = Path(root).resolve(); reader = old.Reader(root); path = root/'outputs'/STUDY/'PLAN.json'
    require(reader.hash(path) == plan_sha, 'Registered PLAN changed')
    plan = reader.read(path)
    require(plan.get('round') == 156 and plan.get('status') == 'REGISTERED' and plan.get('arms') == list(NAMES)
            and plan.get('primary_challenger') == NAMES[3] and plan.get('dose') == .01 and
            reader.path(plan['output_directory']) == path.parent, 'Wrong registered revenue experiment')
    require(plan['registration_precondition']['path'] == STATE, 'Wrong state precondition')
    reader.verify(plan['hashes'])
    state, state_sha = state_snapshot(root/STATE); predecessor(state)
    entry = queue_entry(plan, plan_sha)
    if state.get('registered_queue') == [entry]:
        return dict(registered=True, plan_sha256=plan_sha, already_enqueued=True)
    require(state.get('registered_queue') == [] and state_sha == plan['registration_precondition']['sha256'],
            'Conflicting queue or stale registration state')
    state.update(registered_queue=[entry], current=STUDY, current_status='registered_before_actual_preparation',
                 updated_utc=plan['registered_utc'])
    update_state(root/STATE, state_sha, state)
    return dict(registered=True, plan_sha256=plan_sha, already_enqueued=False)


def register(root, tests, proposal_sha):
    plan = build_plan(root, tests, proposal_sha); out = Path(root)/'outputs'/STUDY
    out.mkdir(exist_ok=True)
    write(out/'PLAN.json', plan)
    return enqueue(root, sha(out/'PLAN.json'))


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def selection(rows, opening):
    for group in (rows, opening):
        require(isinstance(group, list) and [r.get('name') for r in group] == list(NAMES), 'Four ordered arms required')
    fields = (('training', 'cagr'), ('training', 'sharpe'), ('training', 'max_drawdown'), ('early', 'cagr'), ('recent', 'cagr'))
    def guard(a, b, strict):
        values = [r.get(k, {}).get(v) for r in (a, b) for k, v in fields]
        if not all(finite(v) for v in values):
            return False
        x, y = values[:5], values[5:]
        return bool((x[0] > y[0]+1e-12 if strict else x[0] >= y[0]-1e-12) and
                    all(u >= v-t for u, v, t in zip(x[1:], y[1:], (.05, .03, .025, .025))))
    close = {n: guard(rows[3], rows[i], True) for i, n in enumerate(NAMES[:3])}
    opened = {n: guard(opening[3], opening[i], i != 0) for i, n in enumerate(NAMES[:3])}
    admitted = all(close.values()) and all(opened.values())
    return dict(primary=NAMES[3] if admitted else NAMES[0], admitted=admitted,
                guards_against=close, next_open_guards_against=opened, mechanism_confirmed=all(opened.values()))


def receipt(reader, doc, stage, checks, plan, out, plan_sha):
    require(doc.get('passed') is True and doc.get('stage') == stage and doc.get('plan_sha256') == plan_sha
            and doc.get('producer_modules_imported') is False and all(doc.get(k) is True for k in checks),
            'Incomplete independent '+stage+' audit')
    require(doc.get('cache_sha256') == reader.hash(out/'CACHE.json') and
            doc.get('orders_sha256') == reader.hash(out/'PRETRAIN_ORDERS.json'), 'Independent cache/order binding')
    kind = 'prepare' if stage == 'prepare' else 'outcome'
    require(reader.verify(doc['source_hashes']) == reader.verify(plan['independent_'+kind+'_source_hashes']),
            'Independent reviewer implementation changed')
    return reader.verify(doc['artifact_hashes'])


def complete(root, plan_sha):
    r = runner(); root = Path(root).resolve(); out = root/'outputs'/STUDY; reader = old.Reader(root)
    require(reader.hash(out/'PLAN.json') == plan_sha, 'Exact registered PLAN required')
    r.Runner(root, out/'PLAN.json', plan_sha).require_prepare_review()
    plan = reader.read(out/'PLAN.json'); pins = reader.verify(plan['hashes'])
    require(pins.get(str(root/SELF)) == sha(__file__), 'Executing lifecycle differs from registered source')
    for key in ('actual_feature_calculations_before_registration', 'actual_returns_before_registration',
                'actual_model_fits_before_registration', 'new_market_downloads'):
        exact_count(plan, key, 0)
    require(plan.get('no_observer_writes') is True, 'Registered observer protection required')
    source_gate(reader, plan['inputs']['source_grid_review'])
    cache = reader.read(out/'CACHE.json'); require(set(cache) == set(r.CACHE_FILES), 'Complete cache required')
    cached = reader.verify(cache, out)
    prep = reader.read(out/'INDEPENDENT_PREPARE.json'); outcome = reader.read(out/'INDEPENDENT_OUTCOME.json')
    pa = receipt(reader, prep, 'prepare', r.CHECKS, plan, out, plan_sha)
    oa = receipt(reader, outcome, 'evaluate', OUTCOME_CHECKS, plan, out, plan_sha)
    require(prep.get('returns_or_nav_computed') is False, 'Preparation cannot compute returns')
    for key, count in [('orders_checked', 488), ('candidate_cells_checked', 5978), ('months_checked', 122),
                       ('prefixes_checked', 123), ('revenue_feature_matrices_verified', 3), ('new_model_fits', 0)]:
        exact_count(prep, key, count)
    for key, count in [('execution_paths_verified', 12), ('total_monthly_orders_verified', 488),
                       ('revenue_feature_matrices_verified', 3), ('models_fit', 0)]:
        exact_count(outcome, key, count)
    for p, h in {**cached, **{str(out/f): reader.hash(out/f) for f in ('PLAN.json', 'CACHE.json')}}.items():
        require(pa.get(p) == h, 'Unaudited preparation artifact: '+p)
    required = set((*r.CACHE_FILES, *r.FROZEN_FILES, 'CACHE.json', 'FROZEN.json', 'DIAGNOSE_STARTED.json',
                    'RESULTS.json', 'VALIDATION.json'))
    required.update(n+'_'+mode+'_'+kind+'.json' for n in NAMES for mode in MODES for kind in ('DAILY', 'TRADES', 'HOLDINGS'))
    for f in required:
        require(oa.get(str(out/f)) == reader.hash(out/f), 'Unaudited outcome artifact: '+f)
    frozen = reader.read(out/'FROZEN.json')
    require(set(frozen['hashes']) == set(r.FROZEN_FILES), 'Exact frozen first88 artifact scope required')
    reader.verify(frozen['hashes'], out)
    require(frozen.get('pristine_oos') is False and frozen.get('all_diagnostic_orders_frozen') is True,
            'Frozen reused-sample and diagnostic-order disclosure required')
    chosen = selection(reader.read(out/'TRAIN_TRIALS.json'), reader.read(out/'TRAIN_OPEN_TRIALS.json'))
    require(frozen['selection'] == chosen and outcome.get('selected_name') == chosen['primary'], 'First88 selection mismatch')
    marker = reader.read(out/'DIAGNOSE_STARTED.json')
    require(marker.get('plan_sha256') == plan_sha and marker.get('frozen_sha256') == reader.hash(out/'FROZEN.json')
            and outcome.get('frozen_sha256') == reader.hash(out/'FROZEN.json') and
            outcome.get('diagnose_started_sha256') == reader.hash(out/'DIAGNOSE_STARTED.json'), 'Diagnostic chronology binding')
    times = [old.utc(t) for t in (plan['registered_utc'], marker['started_utc'], outcome['audit_started_utc'])]
    require(times[0] < times[1] < times[2] <= datetime.now(timezone.utc), 'Registration/diagnostics/audit chronology')
    results = reader.read(out/'RESULTS.json')
    require(set(results) == set(NAMES) and all(set(results[n]) == set(MODES) for n in NAMES), 'Twelve paths required')
    primary = chosen['primary']; metric = results[primary]['same_close']['metrics']
    require(all(finite(metric.get(k)) for k in ('cagr', 'sharpe', 'max_drawdown')) and finite(outcome.get('primary_cagr'))
            and abs(metric['cagr']-outcome['primary_cagr']) < 1e-12 and
            type(outcome.get('primary_target_reached')) is bool and
            outcome.get('primary_target_reached') == (metric['cagr'] >= .7), 'Independent selected metrics mismatch')
    orders = reader.read(out/'PRETRAIN_ORDERS.json')
    require(set(orders) == set(NAMES) and all(len(orders[n]) == 122 for n in NAMES), '488 ordered choices required')
    require(orders[NAMES[0]] == reader.read(plan['inputs']['original_orders']['path']), 'Incumbent original orders changed')
    validation = reader.read(out/'VALIDATION.json')
    require(validation.get('selection') == chosen and validation.get('no_observer_writes') is True and
            validation.get('no_intramonth_trades') is True, 'Final execution invariants')
    exact_count(validation, 'new_model_fits', 0)
    suites = validate_tests(reader.read(out/'TESTS.json'), reader)
    manifest = reader.read(plan['inputs']['protected_manifest']['path'])
    require(len(manifest) == 21444, 'Full protected21444 inventory required')
    proof = dict(passed=True, checked=21444, verified=21444, manifest_sha256=plan['inputs']['protected_manifest']['sha256'],
                 changed=[], missing=[], read_or_identity_errors=[])
    old.protected_scan(reader, plan, proof)
    if (out/'PROTECTED_AFTER.json').exists():
        require(reader.read(out/'PROTECTED_AFTER.json') == proof, 'Existing protection receipt changed')
    else:
        write(out/'PROTECTED_AFTER.json', proof)
    state, state_sha = state_snapshot(root/STATE)
    already = bool(state.get('completed') and state['completed'][-1].get('directory') == STUDY)
    if already:
        predecessor(dict(state, completed=state['completed'][:-1]))
        require(len(state['completed']) == 156 and sum(x['trials'] for x in state['completed']) == 982 and
                state.get('registered_queue') == [] and state.get('current_status') == 'completed_independently_verified',
                'Conflicting completed state')
    else:
        predecessor(state)
        require(state.get('registered_queue') == [queue_entry(plan, plan_sha)], 'Exact registered queue required')
    done = dict(round=156, directory=STUDY, status='completed_independently_verified',
        completed_utc=datetime.now(timezone.utc).isoformat(), plan_sha256=plan_sha, selection=chosen,
        all_arm_results=results, primary=primary, primary_cagr=metric['cagr'], primary_sharpe=metric['sharpe'],
        target_reached=metric['cagr'] >= .7, trials=4, new_model_fits=0, independent_execution_paths=12,
        protected_files=21444, synthetic_suites=suites, source_admission_conditional=True,
        independent_outcome_sha256=reader.hash(out/'INDEPENDENT_OUTCOME.json'),
        protected_sha256=reader.hash(out/'PROTECTED_AFTER.json'),
        limitations=['Sparse fixed issuer pilot with explicit reported-dollar and archival-vintage assumptions.',
                     'Repeated historical first88 and later34 are not pristine out-of-sample evidence.',
                     'No certification of organic growth, accounting homogeneity or full amendment history.',
                     'No observation accounts or live ledgers changed.'])
    reader.verify(dict(reader.seen))
    if (out/'COMPLETION.json').exists():
        saved = reader.read(out/'COMPLETION.json')
        require({k: v for k, v in saved.items() if k != 'completed_utc'} ==
                {k: v for k, v in done.items() if k != 'completed_utc'}, 'Existing completion conflicts')
        done = saved
    else:
        require(not already, 'Completed state lacks completion receipt')
        write(out/'COMPLETION.json', done)
    ref = dict(path=str((out/'COMPLETION.json').relative_to(root)), sha256=sha(out/'COMPLETION.json'))
    entry = {k: done[k] for k in ('directory', 'completed_utc', 'primary', 'primary_cagr', 'primary_sharpe',
                                'target_reached', 'trials', 'new_model_fits', 'independent_execution_paths', 'protected_files')}
    entry.update(completion=ref, new_admitted=int(chosen['admitted']))
    if already:
        require(state['completed'][-1] == entry and state.get('latest_round_receipt') == ref and sha(root/STATE) == state_sha,
                'Completed state does not match reverified evidence')
        return dict(completion=ref, already_completed=True)
    state['completed'].append(entry)
    require(len(state['completed']) == 156 and sum(r['trials'] for r in state['completed']) == 982, 'Final156/982 accounting')
    state.update(current=STUDY, current_status='completed_independently_verified', registered_queue=[],
                 latest_round_receipt=ref, updated_utc=done['completed_utc'], last_goal_turn_classification='progress')
    if chosen['admitted'] and metric['cagr'] > state['best_training_selected_candidate']['cagr']:
        state['best_training_selected_candidate'] = dict(directory=STUDY, spec=primary, cagr=metric['cagr'],
            sharpe=metric['sharpe'], max_drawdown=metric['max_drawdown'],
            next_open_5bp_cagr=results[primary]['next_open_cost']['metrics']['cagr'], observer_promotion=False)
    update_state(root/STATE, state_sha, state)
    return dict(completion=ref, primary=primary, cagr=metric['cagr'], target_reached=done['target_reached'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'enqueue', 'complete'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--tests'); parser.add_argument('--proposal-sha'); parser.add_argument('--plan-sha')
    args = parser.parse_args()
    with state_lock(args.root):
        if args.phase == 'register':
            result = register(args.root, args.tests, args.proposal_sha)
        elif args.phase == 'enqueue':
            result = enqueue(args.root, args.plan_sha)
        else:
            result = complete(args.root, args.plan_sha)
    print(old.encoded(result).decode())


if __name__ == '__main__':
    main()
