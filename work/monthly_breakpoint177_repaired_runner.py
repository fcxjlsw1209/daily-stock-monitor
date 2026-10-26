"""Explicitly registered representation-only repair around frozen round177.

No strategy expression or old file changes. Only the independent delta auditor
receives a copied one-dimensional representation of its original core weights.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import argparse
import ast
import hashlib
import importlib
import io
import json
import os
import re
import tempfile
import unittest
import zipfile

import monthly_breakpoint177_host as original
import monthly_breakpoint177_independent as independent

ROOT = original.ROOT
OUT = original.OUT
PLAN_SHA = 'cef087df4824c24438d16852fc8bfb079295568ea347b14e126b4e2c8fe1b5fd'
AMENDMENT = OUT / 'AMENDMENT_01.json'
FAILURE = OUT / 'PREPARE_INTERFACE_FAILURE_01.json'
FAILURE_LOG = OUT / 'PREPARE_INTERFACE_FAILURE_01_EXCERPT.txt'
TESTS = OUT / 'REPAIR_TESTS_01.json'
PREREVIEW = OUT / 'REPAIR_PREREVIEW_01.json'
REPAIR_COMPLETION = OUT / 'REPAIR_COMPLETION_01.json'
PREVIOUS_AMENDMENT = dict(path='outputs/monthly_annual_lag158/AMENDMENT_01.json',
    sha256='a8ba1ea3047f20d70448e00fae6f967473c70dd6fe73a880aef87c5b396064c1')
FILES = tuple(ROOT / 'work' / name for name in (
    'monthly_breakpoint177_repaired_runner.py', 'test_monthly_breakpoint177_repaired_runner.py',
    'breakpoint177_weight_repair.py', 'test_breakpoint177_weight_repair.py'))
ORIGINAL_GATE = original.gate
ORIGINAL_REQUIRE_DELTA = original.require_delta_audit
ORIGINAL_VERIFY_DELTAS = independent.verify_deltas
CONTRACT = dict(round=177, kind='independent_core_weight_representation', original_plan_sha256=PLAN_SHA,
    same_experiment=True, additional_strategy_trials=0, formula_changed=False, parameters_changed=False,
    input_data_changed=False, original_arrays_changed=False, source_eligibility_changed=False,
    normalization='Only independent verify_deltas receives a shallow input dictionary copy with a new(n,)core_weight array; accept(n),(n,1),or(n,width)only when every cell equals its row prescribed .50 calm/.52 noncalm weight. No averaging, rounding, rank or delta change.')
sha, read, require = original.sha, original.read, original.require


def now():
    return datetime.now(timezone.utc).isoformat()


def reference(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT.resolve()), 'Repair reference outside local root')
    return dict(path=str(path.relative_to(ROOT)), sha256=sha(path))


def write_once(path, value):
    raw = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()
    publish_bytes(path, raw)


def publish_bytes(path, raw):
    path = Path(path); temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name); stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.link(str(temporary), str(path))
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)


def verify_reference(ref, path):
    require(type(ref) is dict and set(ref) == {'path', 'sha256'} and ref == reference(path),
            'Repair reference changed: '+str(path))
    return read(path)


def implementation_hashes():
    return {str(path.relative_to(ROOT)): sha(path) for path in FILES}


def unprepared():
    require(not any((OUT / name).exists() for name in ('prepared', 'selected', 'results', 'COMPLETION.json')),
            'Original preparation/selection/results already exist')


def check_original(plan_sha):
    require(plan_sha == PLAN_SHA, 'Exact original177 PLAN required')
    return ORIGINAL_GATE(plan_sha)


def npz_shape(path, member):
    # NPY header only: no actual source values are materialized.
    with zipfile.ZipFile(path) as archive, archive.open(member) as stream:
        prefix = stream.read(8)
        require(prefix[:6] == b'\x93NUMPY', 'Invalid NPY header')
        width = 2 if tuple(prefix[6:8]) == (1, 0) else 4
        length = int.from_bytes(stream.read(width), 'little')
        require(0 < length < 100000, 'Invalid NPY header length')
        header = ast.literal_eval(stream.read(length).decode('latin1').strip())
    return list(header['shape'])


def validate_failure(doc):
    require(doc.get('original_plan_sha256') == PLAN_SHA and type(doc.get('exit_code')) is int
            and doc['exit_code'] == 1, 'Original failure required')
    for key in ('strategy_returns_calculated', 'prepared_artifacts_published'):
        require(type(doc.get(key)) is int and doc[key] == 0, 'Failure performed unexpected work')
    for key in ('feature_build_attempted', 'feature_independent_audit_passed', 'original_order_parity_passed'):
        require(doc.get(key) is True, 'Failed attempt milestone missing: '+key)
    require(doc.get('original_core_weight_shape') == [122, 1], 'Unexpected original weight schema')
    require(doc.get('excerpt') == reference(FAILURE_LOG), 'Failure excerpt binding differs')
    require(doc.get('artifact_kind') == 'retained_execution_context_excerpt' and
            doc.get('source') == 'tool_output_excerpt_not_full_log' and doc.get('full_raw_log_available') is False,
            'Failure provenance must identify retained excerpt rather than full raw log')
    log = FAILURE_LOG.read_bytes()
    require(all(text in log for text in (b'Core weight shape/finite', b'independent_deltas',
            b'verify_deltas', b'monthly_breakpoint177_host.py')), 'Wrong retained failure excerpt')


def record_failure(log_path):
    check_original(PLAN_SHA); unprepared()
    source = Path(log_path).resolve()
    require('CloudStorage' not in source.parts and source.is_file(), 'Local retained failure excerpt required')
    raw = source.read_bytes()
    require(all(text in raw for text in (b'Core weight shape/finite', b'independent_deltas',
            b'verify_deltas', b'monthly_breakpoint177_host.py')), 'Unexpected failed attempt')
    shape = npz_shape(original.LOCAL / 'DATA.npz', 'core_weight.npy')
    require(shape == [122, 1], 'Unexpected frozen weight header')
    if FAILURE_LOG.exists(): require(FAILURE_LOG.read_bytes() == raw, 'Existing failure excerpt differs')
    else: publish_bytes(FAILURE_LOG, raw)
    doc = dict(original_plan_sha256=PLAN_SHA, recorded_utc=now(), exec_session=60718,
        exit_code=1, excerpt=reference(FAILURE_LOG), original_core_weight_shape=shape,
        artifact_kind='retained_execution_context_excerpt', source='tool_output_excerpt_not_full_log',
        full_raw_log_available=False,
        feature_build_attempted=True, feature_independent_audit_passed=True,
        original_order_parity_passed=True, strategy_returns_calculated=0, prepared_artifacts_published=0,
        note='Retained execution context reports the delta shape-check failure; the complete raw stream is unavailable. The frozen control flow places that call after feature audit and original order parity and before publication. This is a later-recorded excerpt, not a recreated full traceback. All preparation checks must rerun from original pinned inputs; no transient result is reused.')
    validate_failure(doc); write_once(FAILURE, doc)
    return reference(FAILURE)


def run_tests():
    check_original(PLAN_SHA); unprepared()
    suites = {}; code = implementation_hashes()
    for path in (FILES[1], FILES[3]):
        module = original.module('repair177_test_'+path.stem, path)
        suite = unittest.defaultTestLoader.loadTestsFromModule(module)
        stream = io.StringIO(); result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        print(stream.getvalue(), flush=True)
        require(result.wasSuccessful() and result.testsRun > 0, 'Repair synthetic tests failed')
        suites[path.name] = dict(passed=True, count=result.testsRun, source_sha256=sha(path), log=stream.getvalue())
    require(code == implementation_hashes(), 'Repair code changed during tests')
    receipt = dict(passed=True, synthetic_only=True, original_plan_sha256=PLAN_SHA, created_utc=now(),
        runtime=original.ev.runtime(), implementation_hashes=code, suites=suites)
    write_once(TESTS, receipt); return reference(TESTS)


def validate_tests(doc):
    require(doc.get('passed') is True and doc.get('synthetic_only') is True and
            doc.get('original_plan_sha256') == PLAN_SHA and doc.get('runtime') == original.ev.runtime(),
            'Pinned synthetic test receipt required')
    require(doc.get('implementation_hashes') == implementation_hashes(), 'Repair code changed since tests')
    require(set(doc.get('suites', {})) == {FILES[1].name, FILES[3].name}, 'Missing synthetic suite')
    for path in (FILES[1], FILES[3]):
        row = doc['suites'][path.name]
        require(row.get('passed') is True and type(row.get('count')) is int and row['count'] > 0 and
                row.get('source_sha256') == sha(path), 'Failed or changed synthetic suite')
        require(re.findall(r'Ran (\d+) tests? in ', row.get('log', '')) == [str(row['count'])] and row['log'].rstrip().endswith('OK'),
                'Synthetic test log incomplete')


def validate_prereview(doc):
    require(all(doc.get(key) is True for key in ('passed', 'synthetic_only', 'semantics_preserved',
            'independent_review', 'original_frozen_files_unchanged')), 'Independent repair prereview required')
    require(doc.get('original_plan_sha256') == PLAN_SHA and
            doc.get('implementation_hashes') == implementation_hashes(), 'Unreviewed repair code')
    require(type(doc.get('actual_returns_calculated')) is int and doc['actual_returns_calculated'] == 0,
            'Repair prereview must precede portfolio returns')
    validate_failure(verify_reference(doc.get('failure_record'), FAILURE))
    validate_tests(verify_reference(doc.get('tests'), TESTS))


def verify_amendment(plan_sha, amendment_sha, *, active=True, original_gate=True):
    require(plan_sha == PLAN_SHA and sha(OUT / 'PLAN.json') == PLAN_SHA, 'Original PLAN changed')
    if original_gate:
        check_original(plan_sha)
    else:
        # Closure removes the queue, but must not disable the original frozen-file checks.
        plan = read(OUT / 'PLAN.json')
        require(plan['contract'] == original.contract() and plan['runtime'] == original.ev.runtime(), 'Original contract/runtime changed')
        original.verify(plan['input_sha256'])
        require(sha(OUT / 'STATE_BEFORE.json') == plan['prior_central_sha256'], 'Original prior state changed')
    require(type(amendment_sha) is str and len(amendment_sha) == 64 and
            sha(AMENDMENT) == amendment_sha, 'Exact amendment SHA required')
    doc = read(AMENDMENT)
    require(all(type(doc.get(key)) is type(value) and doc[key] == value for key, value in CONTRACT.items()),
            'Repair contract changed')
    require(doc.get('implementation_hashes') == implementation_hashes(), 'Amended implementation changed')
    require(doc.get('previous_active_execution_amendment') == PREVIOUS_AMENDMENT, 'Previous amendment differs')
    verify_reference(PREVIOUS_AMENDMENT, ROOT / PREVIOUS_AMENDMENT['path'])
    failure = verify_reference(doc.get('failure_record'), FAILURE); validate_failure(failure)
    review = verify_reference(doc.get('independent_prereview'), PREREVIEW); validate_prereview(review)
    validate_tests(verify_reference(doc.get('tests'), TESTS))
    times = [datetime.fromisoformat(value) for value in (read(OUT / 'PLAN.json')['created_utc'],
        failure['recorded_utc'], read(TESTS)['created_utc'], review['reviewed_utc'], doc['registered_utc'])]
    require(all(value.tzinfo is not None for value in times) and times == sorted(times)
            and times[-1] <= datetime.now(timezone.utc), 'Repair chronology differs')
    if active:
        require(read(original.CENTRAL).get('active_execution_amendment') == reference(AMENDMENT),
                'Central amendment binding differs')
    return doc


def register_repair():
    check_original(PLAN_SHA); unprepared()
    review = read(PREREVIEW); validate_prereview(review)
    before = original.CENTRAL.read_bytes(); state = json.loads(before)
    original.state_check(state)
    require(state['registered_queue'] == [dict(directory=OUT.name, plan_sha256=PLAN_SHA)], 'Original queue changed')
    old = state.get('active_execution_amendment')
    if AMENDMENT.exists():
        digest = sha(AMENDMENT)
        doc = verify_amendment(PLAN_SHA, digest, active=False)
        if old == reference(AMENDMENT):
            return dict(registered=True, original_plan_sha256=PLAN_SHA, amendment=reference(AMENDMENT))
        require(old == PREVIOUS_AMENDMENT and doc['continuation_before_sha256'] == hashlib.sha256(before).hexdigest(),
                'Uncommitted amendment cannot attach to a changed central state')
    else:
        require(old == PREVIOUS_AMENDMENT, 'Unexpected previous active amendment')
        verify_reference(old, ROOT / old['path'])
        doc = dict(CONTRACT, registered_utc=now(), implementation_hashes=implementation_hashes(),
            failure_record=reference(FAILURE), tests=reference(TESTS), independent_prereview=reference(PREREVIEW),
            previous_active_execution_amendment=old, continuation_before_sha256=hashlib.sha256(before).hexdigest())
        write_once(AMENDMENT, doc)
        verify_amendment(PLAN_SHA, sha(AMENDMENT), active=False)
    state.update(active_execution_amendment=reference(AMENDMENT), current_status='registered_representation_repair_before_reprepare',
        updated_utc=doc['registered_utc'], last_goal_turn_classification='progress')
    original.commit_state(before, state, 'register_weight_repair')
    return dict(registered=True, original_plan_sha256=PLAN_SHA, amendment=reference(AMENDMENT))


def audit_provenance(amendment_sha):
    return dict(original_plan_sha256=PLAN_SHA, execution_amendment=reference(AMENDMENT),
        repair_source_hashes=implementation_hashes(), repair_prereview=reference(PREREVIEW))


def validate_repaired_delta(check, amendment_sha):
    expected = audit_provenance(amendment_sha)
    require(expected['execution_amendment']['sha256'] == amendment_sha, 'Delta amendment digest differs')
    require(all(check.get(key) == value for key, value in expected.items()), 'Prepared delta lacks repair provenance')
    evidence = check.get('weight_shape_adapter')
    require(type(evidence) is dict and evidence.get('normalized_shape') == [122] and
            evidence.get('original_shape') in ([122], [122, 1], [122, 49]) and
            type(evidence.get('rows')) is int and evidence['rows'] == 122 and
            type(evidence.get('width')) is int and evidence['width'] == 49 and
            type(evidence.get('validated_weight_cells')) is int and evidence['validated_weight_cells'] ==
                (5978 if evidence['original_shape'] == [122, 49] else 122) and
            all(evidence.get(key) is True for key in ('passed', 'exact_state_weights', 'original_dict_unchanged',
                'original_arrays_unchanged', 'other_arrays_same_objects', 'only_weight_replaced',
                'weight_copy_not_shared', 'applied_to_verifier_only')),
            'Core weight representation evidence incomplete')


@contextmanager
def execution_context(plan_sha, amendment_sha, phase):
    repair = importlib.import_module('breakpoint177_weight_repair')
    frozen_verify = independent.verify_deltas
    require(original.gate is ORIGINAL_GATE and original.require_delta_audit is ORIGINAL_REQUIRE_DELTA,
            'Nested or unexpected host patch')
    require(frozen_verify is ORIGINAL_VERIFY_DELTAS, 'Nested or unexpected independent patch')
    def gated(expected):
        require(expected == plan_sha, 'Phase changed original PLAN')
        verify_amendment(plan_sha, amendment_sha, original_gate=False)
        return ORIGINAL_GATE(expected)
    def audited(values, deltas, arrays, dose=.5):
        verify_amendment(plan_sha, amendment_sha, original_gate=False)
        check = repair.adapted_verify_deltas(values, deltas, arrays, frozen_verify, dose=dose)
        require(not any(key in check for key in audit_provenance(amendment_sha)), 'Duplicate repair provenance')
        return dict(check, **audit_provenance(amendment_sha))
    def required(check, deltas):
        ORIGINAL_REQUIRE_DELTA(check, deltas)
        validate_repaired_delta(check, amendment_sha)
    original.gate, original.require_delta_audit = gated, required
    if phase == 'prepare': independent.verify_deltas = audited
    try:
        yield
    finally:
        valid = original.gate is gated and original.require_delta_audit is required and independent.verify_deltas is (audited if phase == 'prepare' else frozen_verify)
        original.gate, original.require_delta_audit = ORIGINAL_GATE, ORIGINAL_REQUIRE_DELTA
        independent.verify_deltas = frozen_verify
        require(valid, 'Unexpected execution function replacement')


def complete_repair(plan_sha, amendment_sha):
    verify_amendment(plan_sha, amendment_sha, original_gate=False)
    closed = read(OUT / 'COMPLETION.json')
    require(closed['status'] == 'COMPLETED_INDEPENDENTLY_VERIFIED' and closed['plan_sha256'] == plan_sha and
            closed['result_completion_sha256'] == sha(OUT / 'results/COMPLETION.json') and
            closed['protected_files'] == 21444 and closed['observers_modified'] is False, 'Original closure incomplete')
    for directory in ('prepared', 'results'):
        completed = read(OUT / directory / 'COMPLETION.json')
        require(completed['status'] == 'COMPLETE' and completed['plan_sha256'] == plan_sha, 'Original artifact closure differs')
        for name, digest in completed['artifacts_sha256'].items():
            require(Path(name).name == name and sha(OUT / directory / name) == digest, 'Original artifact changed')
    validate_repaired_delta(read(OUT / 'prepared/INDEPENDENT_PREPARE.json')['delta'], amendment_sha)
    original.frozen_selection(plan_sha, read(OUT / 'prepared/ORDERS.json'))
    require(read(OUT / 'results/COMPLETION.json')['frozen_sha256'] == sha(OUT / 'selected/FROZEN.json'),
            'Result-to-selection binding differs')
    stable = dict(round=177, same_experiment=True, additional_strategy_trials=0,
        original_completion=reference(OUT / 'COMPLETION.json'), amendment=reference(AMENDMENT),
        failure_record=reference(FAILURE), independent_prereview=reference(PREREVIEW),
        independent_prepare=reference(OUT / 'prepared/INDEPENDENT_PREPARE.json'),
        independent_outcome=reference(OUT / 'results/INDEPENDENT_OUTCOME.json'))
    before = original.CENTRAL.read_bytes(); state = json.loads(before)
    require(state['registered_queue'] == [] and state['current'] == OUT.name and
            state['current_status'] == 'completed_independently_verified' and
            len(state['completed']) == 175 and sum(row['trials'] for row in state['completed']) == 1040 and
            state['completed'][-1]['completion']['sha256'] == sha(OUT / 'COMPLETION.json'), 'Final central state differs')
    if REPAIR_COMPLETION.exists():
        require({key: value for key, value in read(REPAIR_COMPLETION).items() if key != 'completed_utc'} == stable,
                'Existing repair completion changed')
    else: write_once(REPAIR_COMPLETION, dict(stable, completed_utc=now()))
    ref = reference(REPAIR_COMPLETION)
    if state.get('latest_execution_repair_receipt') != ref:
        state['latest_execution_repair_receipt'] = ref
        original.commit_state(before, state, 'complete_weight_repair')
    return dict(complete=True, repair_completion=ref)


def run_phase(phase, plan_sha, amendment_sha):
    require(phase in ('prepare', 'select', 'diagnose', 'close'), 'Unsupported repair phase')
    verify_amendment(plan_sha, amendment_sha)
    if phase == 'prepare': unprepared()
    with execution_context(plan_sha, amendment_sha, phase):
        getattr(original, phase)(plan_sha)
    verify_amendment(plan_sha, amendment_sha, original_gate=phase != 'close')
    if phase == 'close': return complete_repair(plan_sha, amendment_sha)
    return dict(phase=phase, original_plan_sha256=plan_sha, amendment=reference(AMENDMENT))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('record-failure', 'test', 'register-repair', 'prepare', 'select', 'diagnose', 'close', 'complete-repair'))
    parser.add_argument('--log')
    parser.add_argument('--plan-sha256')
    parser.add_argument('--amendment-sha256')
    args = parser.parse_args()
    if args.phase == 'record-failure': result = record_failure(args.log)
    elif args.phase == 'test': result = run_tests()
    elif args.phase == 'register-repair': result = register_repair()
    elif args.phase == 'complete-repair': result = complete_repair(args.plan_sha256, args.amendment_sha256)
    else: result = run_phase(args.phase, args.plan_sha256, args.amendment_sha256)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
