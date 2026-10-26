"""Register/close a fixed filing-date association experiment, never tune it.

The source protocol deliberately remains reconstructed and conditional. This
module handles hash-bound metadata; real features and returns require the
separate registered runner and independently implemented audits.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import argparse
import fcntl
import hashlib
import importlib
import json
import math
import os
import re
import sys
import tempfile

import complete_monthly_same_report145 as old

ROOT = Path(__file__).resolve().parents[1]
STUDY = 'monthly_filing_association157'
STATE = 'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
PREVIOUS = 'outputs/monthly_revenue_persistence156/'
PREVIOUS_PLAN_SHA = 'e80b6692ae126fc42da37dc3fbb3c859ccc57b559a5aa4ec9618f2340a009bcc'
PREVIOUS_DONE_SHA = '97177e6413d2227cb727e9efcc027b0cff68084a8c8e8a899f97d02b573665e2'
DESIGN = 'work/after156_association_protocol.json'
DESIGN_SHA = '9906d41f2aad9f5981030b4264540cdca0f40f8fb5c723e1060ddc32b00fa88e'
SOURCE_ROOT = 'outputs/monthly_filing_association157_source'
SOURCE_MANIFEST = SOURCE_ROOT + '/MANIFEST.json'
SOURCE_REVIEW = 'work/after156_association_package_review.json'
SOURCE_CALENDAR = 'outputs/monthly_paired_target143/CALENDAR.json'
SOURCE_CALENDAR_SHA = '8ee4da69fba84db1edee37a7e0f5af5d3ab16c6257e5d37cddbd531f01404622'
PROPOSAL = 'work/monthly_filing_association157_proposal.json'
SELF = 'work/monthly_filing_association157_lifecycle.py'
PROPOSAL_BUILDER = 'work/prepare_filing_association157_proposal.py'
FACTOR_REVIEW = 'work/after156_association_measure_review.json'
NAMES = ('incumbent_local67', 'whole_formation_beta_residual_momentum',
         'filing_window_density', 'filing_vs_nonfiling_beta_residual_contrast')
MODES = ('same_close', 'next_open_zero', 'next_open_cost')
PREREVIEWS = {kind: 'work/monthly_filing_association157_' + kind + '_prereview.json'
             for kind in ('runner', 'prepare', 'outcome')}
SOURCE_CHECKS = ('package_matches_source_review', 'original_5978_grid_preserved',
    'all_output_and_input_hashes_verified', 'stock_3904_etf_2074_preserved',
    'all_rows_preserve_evidence_and_eligibility')
SUITES = {kind: 'test_monthly_filing_association157_' + kind + '.py'
          for kind in ('runner', 'prepare_review', 'outcome_review', 'lifecycle', 'proposal')}
SUITES['factor'] = 'test_after156_association_measure.py'
OUTCOME_CHECKS = ('daily_nav_trades_holdings_metrics_slices',
    'first88_and_three_fold_selection_verified', 'first88_causal_replay_verified',
    'nuisance_estimate_counts_verified', 'all_changed_months_reported',
    'diagnostic_start_marker_verified', 'baseline_local67_full_orders_unchanged',
    'filing_source_grid_verified', 'all_three_filing_features_common_mask',
    'original_source_cutoffs_verified')
require, sha = old.require, old.sha


def runner():
    import monthly_filing_association157_runner
    return monthly_filing_association157_runner


def write(path, value):
    """Exclusive, durable publication. Existing evidence is never replaced."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.filing157-', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(old.encoded(value)); stream.flush(); os.fsync(stream.fileno())
        os.link(str(temporary), str(path))
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def state_snapshot(path):
    raw = Path(path).read_bytes()
    # Mutable coordination state must retain every historical path string;
    # unlike execution inputs, it must not use the launcher's relocation hook.
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, 'Duplicate state key')
            value[key] = item
        return value
    def bad(value):
        raise ValueError('Nonfinite state token: ' + value)
    return json.JSONDecoder(object_pairs_hook=pairs, parse_constant=bad).decode(raw.decode('utf-8')), hashlib.sha256(raw).hexdigest()


@contextmanager
def state_lock(root):
    with (Path(root)/'outputs/monthly_target70_research/.lifecycle.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def predecessor(state):
    rows = state.get('completed')
    require(isinstance(rows, list) and len(rows) == 156 and
            all(isinstance(row, dict) and type(row.get('trials')) is int and row['trials'] >= 0
                for row in rows) and sum(row['trials'] for row in rows) == 982,
            'Expected156 completed rounds /982 trials')
    require(rows[-1].get('directory') == 'monthly_revenue_persistence156' and
            rows[-1].get('completion', {}).get('sha256') == PREVIOUS_DONE_SHA,
            'Wrong finalized predecessor')
    require(not any(row.get('directory') == STUDY for row in rows), 'Already completed157')


def update_state(path, expected, value):
    path = Path(path)
    require(sha(path) == expected, 'Concurrent continuation change')
    backup = path.parent/('PRE157_' + expected + '.json')
    if not backup.exists():
        # Keep exact historical strings/bytes rather than re-encoding the backup.
        with backup.open('xb') as stream:
            stream.write(path.read_bytes()); stream.flush(); os.fsync(stream.fileno())
    require(sha(backup) == expected, 'Original state backup mismatch')
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(old.encoded(value)); stream.flush(); os.fsync(stream.fileno())
    try:
        require(sha(path) == expected, 'Concurrent continuation commit')
        os.replace(str(temporary), str(path))
    finally:
        temporary.unlink(missing_ok=True)


def exact_count(doc, key, expected):
    require(type(doc.get(key)) is int and doc[key] == expected, 'Wrong exact count: ' + key)


def validate_tests(doc, reader):
    require(doc.get('passed') is True and doc.get('synthetic_only') is True, 'Synthetic tests required')
    found = {}
    for row in doc['suites']:
        name, count = row.get('name'), row.get('count')
        require(name in SUITES and name not in found and row.get('passed') is True and
                type(count) is int and count > 0, 'Distinct positive successful suite required')
        for key in ('source', 'log'):
            ref = row[key]
            require(reader.hash(ref['path']) == ref['sha256'], 'Changed test ' + key)
        require(Path(row['source']['path']).name == SUITES[name], 'Unexpected test source')
        log = reader.path(row['log']['path']).read_text()
        require(re.search(r'Ran ' + str(count) + r' tests? in ', log) and log.rstrip().endswith('OK'),
                'Test log does not establish success')
        found[name] = count
    require(set(found) == set(SUITES), 'All six synthetic suites required')
    exact_count(doc, 'test_count', sum(found.values()))
    return found


def source_gate(reader, reference):
    require(reader.path(reference['path']) == reader.path(SOURCE_REVIEW) and
            reader.hash(reference['path']) == reference['sha256'], 'Exact package review required')
    review = reader.read(reference['path'])
    require(review.get('passed') is True and review.get('actual_grid_verified') is True and
            review.get('research_eligibility_verified') is True and
            review.get('exact_acceptance_certified') is False,
            'Independent conditional source review required')
    require(all(review.get('checks', {}).get(k) is True for k in SOURCE_CHECKS),
            'Independent full-grid source checks required')
    for key, count in (('candidate_cells', 5978), ('stock_cells', 3904), ('etf_cells', 2074)):
        exact_count(review, key, count)
    reviewed = reader.verify(review['input_hashes'])
    manifest = reader.read(SOURCE_MANIFEST)
    require(manifest.get('schema_version') == 1 and manifest.get('source_only') is True and
            manifest.get('strategy_registration') is False and
            manifest.get('exact_acceptance_certified') is False and
            manifest.get('clock_policy_id') == 'second_XNYS_session_strictly_after_filingDate_v1' and
            set(manifest['outputs']) == {'ROWS.json', 'COVERAGE_GATE.json'}, 'Exact source package')
    needed = reader.verify(manifest['input_hashes'])
    needed[str(reader.path(SOURCE_MANIFEST))] = reader.hash(SOURCE_MANIFEST)
    for name, ref in manifest['outputs'].items():
        path = reader.path(SOURCE_ROOT + '/' + name)
        require(reader.hash(path) == ref['sha256'] and path.stat().st_size == ref['bytes'],
                'Source output changed')
        needed[str(path)] = ref['sha256']
    require(all(reviewed.get(path) == digest for path, digest in needed.items()),
            'Package review omitted source/output closure')
    require(reader.hash(DESIGN) == DESIGN_SHA and reader.hash(SOURCE_CALENDAR) == SOURCE_CALENDAR_SHA,
            'Design/calendar changed')
    r = runner()
    meta_path, meta_sha = r.FIXED_INPUTS['meta']
    for path, digest in ((DESIGN, DESIGN_SHA), (SOURCE_CALENDAR, SOURCE_CALENDAR_SHA), (meta_path, meta_sha)):
        require(needed.get(str(reader.path(path))) == digest and
                reviewed.get(str(reader.path(path))) == digest and reader.hash(path) == digest,
                'Source contract absent from complete input closure: ' + path)
    gate = reader.read(SOURCE_ROOT + '/COVERAGE_GATE.json')
    require(gate.get('coverage_floor_passed') is True and gate.get('original_candidate_rows') == 5978,
            'Full unchanged denominator and eight breadth floors required')
    require(gate == r.coverage_gate(reader.read(meta_path), reader.read(SOURCE_ROOT + '/ROWS.json')),
            'Source coverage must equal the complete original metadata grid')
    return manifest


def prereview(reader, stage, proposal_sha):
    doc = reader.read(PREREVIEWS[stage])
    require(doc.get('passed') is True and doc.get('round') == 157 and
            doc.get('stage') == 'preregistration_' + stage + '_review' and
            doc.get('synthetic_only') is True and doc.get('proposal_sha256') == proposal_sha,
            'Independent prospective ' + stage + ' review required')
    for key in ('actual_factors_calculated', 'actual_returns_calculated', 'actual_orders_calculated',
                'actual_predictive_model_fits', 'actual_nuisance_estimates'):
        exact_count(doc, key, 0)
    implementations = reader.verify(doc['implementation_hashes'])
    for name in ('after156_association_measure.py', 'monthly_filing_association157_runner.py',
                 'monthly_filing_association157_lifecycle.py', 'prepare_filing_association157_proposal.py'):
        path = reader.path('work/' + name)
        require(implementations.get(str(path)) == reader.hash(path), 'Unreviewed implementation ' + name)
    inputs = reader.verify(doc['input_hashes'])
    for path in (PROPOSAL, DESIGN, SOURCE_MANIFEST, SOURCE_REVIEW):
        require(inputs.get(str(reader.path(path))) == reader.hash(path), 'Unreviewed input ' + path)
    source_hashes = reader.verify(doc['source_hashes'])
    if stage in ('prepare', 'outcome'):
        module = importlib.import_module('monthly_filing_association157_' + stage + '_review')
        require(source_hashes == {str(reader.path(p)): reader.hash(p) for p in module.source_files(reader.root)},
                'Prereview must cover the complete actual independent implementation: ' + stage)
    return source_hashes


def build_plan(root, tests, proposal_sha):
    r = runner(); root = Path(root).resolve(); reader = old.Reader(root); out = root/'outputs'/STUDY
    require(old.digest(proposal_sha), 'Explicit final proposal SHA required')
    require(not any((out/name).exists() for name in ('PLAN.json', 'CACHE.json', 'RESULTS.json',
            'FROZEN.json', *r.CACHE_FILES)), 'Registration precedes all real preparation')
    state, state_sha = state_snapshot(root/STATE); predecessor(state)
    require(state.get('registered_queue') == [], 'Empty registration queue required')
    require(reader.hash(PREVIOUS + 'COMPLETION.json') == PREVIOUS_DONE_SHA and
            reader.hash(PREVIOUS + 'PLAN.json') == PREVIOUS_PLAN_SHA, 'Previous completion changed')
    previous = reader.read(PREVIOUS + 'PLAN.json')
    reader.verify(previous['hashes']); reader.verify(previous['raw_source_hashes'])
    require(len(previous['raw_source_hashes']) == 64, 'Original64 raw price dependencies required')
    runtime = dict(python=sys.version, numpy=r.np.__version__, pandas=r.pd.__version__)
    require(runtime == previous['runtime'], 'Exact pinned runtime required')
    require(reader.hash(PROPOSAL) == proposal_sha, 'Proposal changed')
    proposal = reader.read(PROPOSAL); r.validate_proposal(proposal)
    require(reader.hash(DESIGN) == proposal['design']['sha256'] == DESIGN_SHA, 'Design changed')
    expected = dict(source_manifest=SOURCE_MANIFEST, source_grid_review=SOURCE_REVIEW,
                    source_plan=DESIGN, source_calendar=SOURCE_CALENDAR)
    require(set(proposal['source_inputs']) == set(expected), 'Exact source input names required')
    for key, path in expected.items():
        ref = proposal['source_inputs'][key]
        require(reader.path(ref['path']) == reader.path(path) and reader.hash(path) == ref['sha256'],
                'Canonical source reference changed')
    source_gate(reader, proposal['source_inputs']['source_grid_review'])
    reader.verify(r.source_files(root))
    for path in (SELF, PROPOSAL_BUILDER, 'work/test_monthly_filing_association157_lifecycle.py',
                 'work/test_monthly_filing_association157_proposal.py', old.__file__):
        reader.hash(path)
    arithmetic = reader.read(FACTOR_REVIEW)
    require(arithmetic.get('passed') is True and arithmetic.get('stage') ==
            'INDEPENDENT_SYNTHETIC_ASSOCIATION_MEASURE_REVIEW', 'Independent arithmetic review required')
    require(all(type(v) is int and v == 0 for v in arithmetic['activity'].values()), 'No real pre-review work')
    arithmetic_pins = reader.verify(arithmetic['input_hashes'])
    for path in ('work/after156_association_measure.py', 'work/test_after156_association_measure.py', DESIGN):
        require(arithmetic_pins.get(str(reader.path(path))) == reader.hash(path),
                'Arithmetic review omitted current factor/test/design: ' + path)
    independent = {kind: prereview(reader, kind, proposal_sha) for kind in PREREVIEWS}
    require(reader.path(tests) == out/'TESTS.json', 'Canonical test manifest required')
    validate_tests(reader.read(tests), reader)
    inputs = {key: dict(previous['inputs'][key]) for key in
              ('meta', 'policy', 'original_orders', 'prices', 'protected_manifest')}
    inputs.update(proposal['source_inputs']); inputs['proposal'] = dict(path=PROPOSAL, sha256=proposal_sha)
    require(set(inputs) == set(r.INPUTS), 'Named input schema mismatch')
    for ref in inputs.values():
        require(reader.hash(ref['path']) == ref['sha256'], 'Named input changed')
        ref['path'] = str(reader.path(ref['path']))
    reader.verify(dict(reader.seen)); hashes = dict(reader.seen); hashes.pop(str(root/STATE), None)
    return dict(round=157, status='REGISTERED', registered_utc=datetime.now(timezone.utc).isoformat(),
        registration_precondition=dict(path=STATE, sha256=state_sha), arms=list(NAMES),
        primary_challenger=NAMES[3], dose=.01, tolerances=dict(r.TOLERANCES), runtime=runtime,
        output_directory=str(out), hashes=hashes, inputs=inputs, raw_source_hashes=previous['raw_source_hashes'],
        independent_prepare_source_hashes=independent['prepare'],
        independent_outcome_source_hashes=independent['outcome'], prerequisite_reviews=dict(PREREVIEWS),
        tests=str(out/'TESTS.json'), actual_feature_calculations_before_registration=0,
        actual_returns_before_registration=0, actual_model_fits_before_registration=0,
        actual_nuisance_estimates_before_registration=0, new_market_downloads=0,
        no_observer_writes=True, exact_acceptance_certified=False)


def queue_entry(plan, plan_sha):
    return dict(round=157, directory=STUDY, plan=dict(path='outputs/' + STUDY + '/PLAN.json', sha256=plan_sha),
                registered_utc=plan['registered_utc'])


def enqueue(root, plan_sha):
    root = Path(root).resolve(); reader = old.Reader(root); path = root/'outputs'/STUDY/'PLAN.json'
    require(reader.hash(path) == plan_sha, 'Registered PLAN changed'); plan = reader.read(path)
    require(plan.get('round') == 157 and plan.get('status') == 'REGISTERED' and plan['arms'] == list(NAMES)
            and plan['primary_challenger'] == NAMES[3] and plan['dose'] == .01 and
            reader.path(plan['output_directory']) == path.parent, 'Wrong registered experiment')
    require(plan['registration_precondition']['path'] == STATE, 'Wrong continuation precondition')
    reader.verify(plan['hashes']); state, before = state_snapshot(root/STATE); predecessor(state)
    entry = queue_entry(plan, plan_sha)
    if state.get('registered_queue') == [entry]:
        return dict(registered=True, plan_sha256=plan_sha, already_enqueued=True)
    require(state.get('registered_queue') == [] and before == plan['registration_precondition']['sha256'],
            'Stale registration state or conflicting queue')
    state.update(registered_queue=[entry], current=STUDY, current_status='registered_before_actual_preparation',
                 updated_utc=plan['registered_utc'])
    update_state(root/STATE, before, state)
    return dict(registered=True, plan_sha256=plan_sha, already_enqueued=False)


def register(root, tests, proposal_sha):
    plan = build_plan(root, tests, proposal_sha); out = Path(root)/'outputs'/STUDY
    out.mkdir(exist_ok=True); write(out/'PLAN.json', plan)
    return enqueue(root, sha(out/'PLAN.json'))


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def selection(rows, opening):
    for group in (rows, opening):
        require(isinstance(group, list) and [r.get('name') for r in group] == list(NAMES), 'Four ordered arms')
    fields = (('training', 'cagr'), ('training', 'sharpe'), ('training', 'max_drawdown'),
              ('early', 'cagr'), ('recent', 'cagr'))
    def guard(a, b, strict):
        values = [r.get(k, {}).get(v) for r in (a, b) for k, v in fields]
        if not all(finite(v) for v in values):
            return False
        x, y = values[:5], values[5:]
        return bool((x[0] > y[0] + 1e-12 if strict else x[0] >= y[0] - 1e-12) and
                    all(u >= v-t for u, v, t in zip(x[1:], y[1:], (.05, .03, .025, .025))))
    close = {n: guard(rows[3], rows[i], True) for i, n in enumerate(NAMES[:3])}
    opened = {n: guard(opening[3], opening[i], i != 0) for i, n in enumerate(NAMES[:3])}
    admitted = all(close.values()) and all(opened.values())
    return dict(primary=NAMES[3] if admitted else NAMES[0], admitted=admitted,
                guards_against=close, next_open_guards_against=opened, mechanism_confirmed=all(opened.values()))


def complete(root, plan_sha):
    r = runner(); root = Path(root).resolve(); out = root/'outputs'/STUDY; reader = old.Reader(root)
    require(reader.hash(out/'PLAN.json') == plan_sha, 'Exact registered PLAN required')
    r.Runner(root, out/'PLAN.json', plan_sha).require_prepare_review()
    plan = reader.read(out/'PLAN.json'); pins = reader.verify(plan['hashes'])
    require(pins.get(str(root/SELF)) == sha(__file__), 'Executing lifecycle differs from registered source')
    for key in ('actual_feature_calculations_before_registration', 'actual_returns_before_registration',
                'actual_model_fits_before_registration', 'actual_nuisance_estimates_before_registration',
                'new_market_downloads'):
        exact_count(plan, key, 0)
    require(plan.get('no_observer_writes') is True and plan.get('exact_acceptance_certified') is False,
            'Observer and source interpretation invariants')
    source_gate(reader, plan['inputs']['source_grid_review'])
    cache = reader.read(out/'CACHE.json'); require(set(cache) == set(r.CACHE_FILES), 'Complete cache required')
    cached = reader.verify(cache, out)
    prep = reader.read(out/'INDEPENDENT_PREPARE.json'); outcome = reader.read(out/'INDEPENDENT_OUTCOME.json')
    scopes = {}
    for kind, doc, stage, checks in (('prepare', prep, 'prepare', r.CHECKS),
                                    ('outcome', outcome, 'evaluate', OUTCOME_CHECKS)):
        require(doc.get('passed') is True and doc.get('stage') == stage and
                doc.get('plan_sha256') == plan_sha and doc.get('producer_modules_imported') is False and
                all(doc.get(key) is True for key in checks), 'Independent ' + kind + ' audit incomplete')
        require(doc.get('cache_sha256') == reader.hash(out/'CACHE.json') and
                doc.get('orders_sha256') == reader.hash(out/'PRETRAIN_ORDERS.json'), 'Audit artifact binding')
        require(reader.verify(doc['source_hashes']) ==
                reader.verify(plan['independent_' + kind + '_source_hashes']), 'Reviewer source changed')
        scopes[kind] = reader.verify(doc['artifact_hashes'])
    require(prep.get('strategy_returns_or_nav_computed') is False, 'Preparation must precede performance')
    for key, count in [('orders_checked', 488), ('candidate_cells_checked', 5978), ('months_checked', 122),
                       ('prefixes_checked', 123), ('filing_feature_matrices_verified', 3),
                       ('new_predictive_model_fits', 0)]:
        exact_count(prep, key, count)
    for key, count in [('execution_paths_verified', 12), ('total_monthly_orders_verified', 488),
                       ('filing_feature_matrices_verified', 3), ('new_predictive_model_fits', 0)]:
        exact_count(outcome, key, count)
    feature = reader.read(out/'FEATURE_AUDIT.json'); validation = reader.read(out/'VALIDATION.json')
    for key in ('nuisance_beta_estimates_attempted', 'nuisance_beta_estimates_completed'):
        value = feature.get(key)
        require(type(value) is int and value > 0, 'Actual nuisance estimate count required')
        exact_count(prep, key, value); exact_count(outcome, key, value); exact_count(validation, key, value)
    require(feature['nuisance_beta_estimates_completed'] <= feature['nuisance_beta_estimates_attempted'],
            'Completed estimates exceed attempts')
    exact_count(feature, 'new_predictive_model_fits', 0); exact_count(validation, 'new_predictive_model_fits', 0)
    require('new_model_fits' not in feature and 'new_model_fits' not in validation,
            'Unqualified zero fit count must not conceal nuisance estimation')
    for p, digest in {**cached, **{str(out/f): reader.hash(out/f) for f in ('PLAN.json', 'CACHE.json')}}.items():
        require(scopes['prepare'].get(p) == digest, 'Unaudited preparation artifact ' + p)
    required = set((*r.CACHE_FILES, *r.FROZEN_FILES, 'CACHE.json', 'FROZEN.json', 'DIAGNOSE_STARTED.json',
                    'RESULTS.json', 'VALIDATION.json'))
    required.update(n + '_' + mode + '_' + kind + '.json' for n in NAMES for mode in MODES
                    for kind in ('DAILY', 'TRADES', 'HOLDINGS'))
    for filename in required:
        require(scopes['outcome'].get(str(out/filename)) == reader.hash(out/filename),
                'Unaudited outcome artifact ' + filename)
    frozen = reader.read(out/'FROZEN.json')
    require(set(frozen['hashes']) == set(r.FROZEN_FILES), 'Frozen first88 artifact scope')
    reader.verify(frozen['hashes'], out)
    require(frozen.get('pristine_oos') is False and frozen.get('all_diagnostic_orders_frozen') is True,
            'Explicit repeated-history and frozen-order disclosure required')
    chosen = selection(reader.read(out/'TRAIN_TRIALS.json'), reader.read(out/'TRAIN_OPEN_TRIALS.json'))
    require(frozen['selection'] == chosen and outcome.get('selected_name') == chosen['primary'],
            'Independent first88 selection mismatch')
    marker = reader.read(out/'DIAGNOSE_STARTED.json')
    require(marker.get('plan_sha256') == plan_sha and
            marker.get('frozen_sha256') == reader.hash(out/'FROZEN.json') and
            outcome.get('frozen_sha256') == reader.hash(out/'FROZEN.json') and
            outcome.get('diagnose_started_sha256') == reader.hash(out/'DIAGNOSE_STARTED.json'),
            'Diagnostic chronology binding')
    times = [old.utc(t) for t in (plan['registered_utc'], marker['started_utc'], outcome['audit_started_utc'])]
    require(times[0] < times[1] < times[2] <= datetime.now(timezone.utc), 'Registration/diagnostic/audit chronology')
    results = reader.read(out/'RESULTS.json')
    require(set(results) == set(NAMES) and all(set(results[n]) == set(MODES) for n in NAMES), 'All12 paths required')
    primary = chosen['primary']; metric = results[primary]['same_close']['metrics']
    require(all(finite(metric.get(k)) for k in ('cagr', 'sharpe', 'max_drawdown')) and
            finite(outcome.get('primary_cagr')) and abs(metric['cagr'] - outcome['primary_cagr']) < 1e-12 and
            type(outcome.get('primary_target_reached')) is bool and
            outcome['primary_target_reached'] == (metric['cagr'] >= .7), 'Independent selected metrics mismatch')
    orders = reader.read(out/'PRETRAIN_ORDERS.json')
    require(set(orders) == set(NAMES) and all(len(orders[n]) == 122 for n in NAMES), 'Full488 ordered choices')
    require(orders[NAMES[0]] == reader.read(plan['inputs']['original_orders']['path']), 'Original incumbent changed')
    require(validation.get('selection') == chosen and validation.get('no_observer_writes') is True and
            validation.get('no_intramonth_trades') is True, 'Final execution invariants')
    suites = validate_tests(reader.read(out/'TESTS.json'), reader)
    manifest = reader.read(plan['inputs']['protected_manifest']['path'])
    require(len(manifest) == 21444, 'Full protected21444 inventory required')
    proof = dict(passed=True, checked=21444, verified=21444,
        manifest_sha256=plan['inputs']['protected_manifest']['sha256'], changed=[], missing=[], read_or_identity_errors=[])
    old.protected_scan(reader, plan, proof)
    if (out/'PROTECTED_AFTER.json').exists():
        require(reader.read(out/'PROTECTED_AFTER.json') == proof, 'Existing protection receipt changed')
    else:
        write(out/'PROTECTED_AFTER.json', proof)
    state, state_sha = state_snapshot(root/STATE)
    already = bool(state.get('completed') and state['completed'][-1].get('directory') == STUDY)
    if already:
        predecessor(dict(state, completed=state['completed'][:-1]))
        require(len(state['completed']) == 157 and sum(x['trials'] for x in state['completed']) == 986 and
                state.get('registered_queue') == [] and state.get('current_status') == 'completed_independently_verified',
                'Conflicting completed state')
    else:
        predecessor(state)
        require(state.get('registered_queue') == [queue_entry(plan, plan_sha)], 'Exact registered queue required')
    done = dict(round=157, directory=STUDY, status='completed_independently_verified',
        completed_utc=datetime.now(timezone.utc).isoformat(), plan_sha256=plan_sha, selection=chosen,
        all_arm_results=results, primary=primary, primary_cagr=metric['cagr'], primary_sharpe=metric['sharpe'],
        target_reached=metric['cagr'] >= .7, trials=4, new_predictive_model_fits=0,
        nuisance_beta_estimates_attempted=feature['nuisance_beta_estimates_attempted'],
        nuisance_beta_estimates_completed=feature['nuisance_beta_estimates_completed'],
        independent_execution_paths=12, protected_files=21444, synthetic_suites=suites,
        source_admission_conditional=True, exact_acceptance_certified=False,
        independent_outcome_sha256=reader.hash(out/'INDEPENDENT_OUTCOME.json'),
        protected_sha256=reader.hash(out/'PROTECTED_AFTER.json'), limitations=[
            'Reconstructed administrative filing dates and explicit issuer-continuity assumptions; not exact news timing.',
            'Later-retrieved source versions and incomplete historical correction history remain limitations.',
            'First88 and later34 have been repeatedly examined; neither is pristine out-of-sample evidence.',
            'All observation accounts and frozen historical artifacts remain unchanged.'])
    reader.verify(dict(reader.seen))
    if (out/'COMPLETION.json').exists():
        saved = reader.read(out/'COMPLETION.json')
        require({k: v for k, v in saved.items() if k != 'completed_utc'} ==
                {k: v for k, v in done.items() if k != 'completed_utc'}, 'Existing completion conflicts')
        done = saved
    else:
        require(not already, 'Completed state lacks completion receipt'); write(out/'COMPLETION.json', done)
    reference = dict(path=str((out/'COMPLETION.json').relative_to(root)), sha256=sha(out/'COMPLETION.json'))
    fields = ('directory', 'completed_utc', 'primary', 'primary_cagr', 'primary_sharpe', 'target_reached', 'trials',
              'new_predictive_model_fits', 'nuisance_beta_estimates_attempted', 'nuisance_beta_estimates_completed',
              'independent_execution_paths', 'protected_files')
    entry = {key: done[key] for key in fields}; entry.update(completion=reference, new_admitted=int(chosen['admitted']))
    if already:
        require(state['completed'][-1] == entry and state.get('latest_round_receipt') == reference and
                sha(root/STATE) == state_sha, 'Completed state differs from reverified receipt')
        return dict(completion=reference, already_completed=True)
    state['completed'].append(entry)
    require(len(state['completed']) == 157 and sum(r['trials'] for r in state['completed']) == 986, 'Final157/986 count')
    state.update(current=STUDY, current_status='completed_independently_verified', registered_queue=[],
                 latest_round_receipt=reference, updated_utc=done['completed_utc'], last_goal_turn_classification='progress')
    if chosen['admitted'] and metric['cagr'] > state['best_training_selected_candidate']['cagr']:
        state['best_training_selected_candidate'] = dict(directory=STUDY, spec=primary, cagr=metric['cagr'],
            sharpe=metric['sharpe'], max_drawdown=metric['max_drawdown'],
            next_open_5bp_cagr=results[primary]['next_open_cost']['metrics']['cagr'], observer_promotion=False)
    update_state(root/STATE, state_sha, state)
    return dict(completion=reference, primary=primary, cagr=metric['cagr'], target_reached=done['target_reached'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'enqueue', 'complete'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--tests'); parser.add_argument('--proposal-sha'); parser.add_argument('--plan-sha')
    args = parser.parse_args()
    with state_lock(args.root):
        result = (register(args.root, args.tests, args.proposal_sha) if args.phase == 'register' else
                  enqueue(args.root, args.plan_sha) if args.phase == 'enqueue' else complete(args.root, args.plan_sha))
    print(old.encoded(result).decode())


if __name__ == '__main__':
    main()
