"""Close the completed fixed PTR169 data sensitivity without a strategy trial.

Use ./run_local.sh. 'check' performs no writes. 'apply' requires an explicitly
reviewed current central SHA; it preserves every unlisted central field.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/monthly_ptr_pool_source169/fixed_incumbent_rebuild'
CENTRAL = ROOT / 'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
STATUS = 'FIXED_INCUMBENT_DATA_SENSITIVITY_COMPLETED_INDEPENDENTLY_VERIFIED'
PINS = {
    'PLAN.json': 'dd6cc6d21c486882423d0172742443814015c38cfd5e79e4442b81c259d58e82',
    'evaluation/PLAN.json': '34bf25eec0d0c524becc003f51d7f7333159d32d2455c1648e408b1a74c81d28',
    'evaluation/results/COMPLETION.json': '60bb9afd9cc69c24b362e65f3e3870ae354df13f9792ba93b3e0d57addedc860',
    'independent_review/COMPLETION.json': 'c5eaa80dbc580ad8af8a824a0dc3575f1d3e3020471b78b96137f0c5feb4f1f5',
}
ALLOWED = {'data_evidence_completions', 'latest_pool_data_progress', 'next',
           'next_step', 'last_goal_turn_classification', 'updated_utc'}
PATHS = ('same_close_zero', 'next_open_zero', 'next_open_cost')
SPLITS = ('full122', 'early53', 'recent35', 'later34', 'first88')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def encode(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()


def write_once(path, data):
    with path.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def load(path):
    return json.loads(path.read_text())


def near(a, b, message):
    require(math.isfinite(a) and math.isfinite(b) and abs(a - b) <= 1e-10 + abs(b) * 1e-12, message)


def validate_registry(state):
    require(len(state['completed']) == 167 and sum(x['trials'] for x in state['completed']) == 1016,
            'Expected 167 completed strategy rounds and 1016 search trials')
    require(not state['registered_queue'], 'Another strategy experiment is registered')
    require(state['current'] == 'monthly_sector_diffusion167' and
            state['current_status'] == 'completed_independently_verified', 'Unexpected current strategy experiment')
    require(state['target_full_period_cagr'] == .7, 'Target changed')
    best = state['best_training_selected_candidate']
    require(best['directory'] == 'monthly_local_linear70' and
            best['spec'] == dict(id=4, base='incumbent_scope67', mode='local_linear'), 'Incumbent changed')
    near(best['cagr'], .6766617372041686, 'Incumbent return changed')
    require(best['cagr'] < .7 and best['observer_promotion'] is False, 'Unexpected goal/observer state')


def verify_sources():
    pins = {str(BASE / name): expected for name, expected in PINS.items()}
    for path, expected in pins.items():
        require(sha(path) == expected, 'Receipt digest changed: ' + path)
    rebuild = load(BASE / 'PLAN.json')
    plan = load(BASE / 'evaluation/PLAN.json')
    completion = load(BASE / 'evaluation/results/COMPLETION.json')
    independent = load(BASE / 'independent_review/COMPLETION.json')
    require(rebuild['fixed_candidate'] == dict(id=4, base='incumbent_scope67', mode='local_linear') and
            rebuild['new_strategy_search_trials'] == 0, 'Not the fixed incumbent sensitivity')
    require(completion['status'] == 'COMPLETE' and completion['paths'] == 6 and completion['arms'] == 2 and
            completion['exact_control_rechecked'] is True, 'Evaluation incomplete')
    require(completion['evaluation_plan_sha256'] == PINS['evaluation/PLAN.json'] and
            completion['rebuild_plan_sha256'] == PINS['PLAN.json'], 'Unbound evaluation')
    require(independent['status'] == 'PASS' and independent['checked_paths'] == 6 and independent['checked_slices'] == 30 and
            independent['evaluation_completion_sha256'] == PINS['evaluation/results/COMPLETION.json'], 'Unbound independent review')
    for folder, receipt in ((BASE / 'evaluation/results', completion), (BASE / 'independent_review', independent)):
        require(not (folder / 'FAILURE.json').exists(), 'Failure receipt present')
        for name, expected in receipt['artifacts_sha256'].items():
            require(Path(name).name == name, 'Invalid artifact path')
            path = folder / name
            require(sha(path) == expected, 'Changed artifact: ' + str(path))
            pins[str(path)] = expected
    results = load(BASE / 'evaluation/results/RESULTS.json')
    review = load(BASE / 'independent_review/REPORT.json')
    require(review['passed'] and review['checked_paths'] == 6 and review['checked_slices'] == 30,
            'Independent report incomplete')
    require(review['completion_sha256'] == PINS['evaluation/results/COMPLETION.json'] and
            review['evaluation_plan_sha256'] == PINS['evaluation/PLAN.json'], 'Independent report provenance mismatch')
    require(results['optimized'] is False and results['promoted'] is False and results['records_are_simulated'],
            'Unexpected optimization/promotion state')
    for arm in ('control', 'conditional'):
        require(set(results['arms'][arm]) == set(PATHS), 'Wrong execution paths')
        for path in PATHS:
            actual, checked = results['arms'][arm][path], review['paths'][arm][path]
            require(checked['passed'] and checked['monthly_executions'] == 122, 'Incomplete independent path')
            require(set(actual['slices']) == set(SPLITS), 'Wrong evaluation segments')
            for segment in SPLITS:
                a, b = actual['slices'][segment], checked['slices'][segment]
                require(a['start'] == b['start'] and a['end'] == b['end'], 'Different metric periods')
                for key in ('cagr', 'sharpe', 'max_drawdown', 'total_return'):
                    near(a[key], b[key], 'Independent metric discrepancy: ' + '/'.join((arm, path, segment, key)))
    near(results['arms']['control']['same_close_zero']['metrics']['cagr'], .6766617372041686, 'Original control did not reproduce')
    for arm in ('control', 'conditional'):
        require(results['arms'][arm]['same_close_zero']['metrics']['cagr'] < .7, 'This closer expects the target to remain unmet')
    return results, review, plan, pins


def build_receipt(state, expected_central, results, review, plan, pins, now):
    validate_registry(state)
    summary = {arm: {name: {key: results['arms'][arm][name]['metrics'][key]
                           for key in ('start', 'end', 'cagr', 'sharpe', 'max_drawdown', 'total_return')}
                     for name in PATHS} for arm in ('control', 'conditional')}
    return dict(schema='ptr169_fixed_evaluation_progress_v1', created_utc=now, status=STATUS,
                fixed_candidate=dict(id=4, base='incumbent_scope67', mode='local_linear'),
                strategy_rounds=167, strategy_search_trials=1016, new_strategy_search_trials=0,
                fixed_data_arms=2, execution_paths=6, independently_checked_slices=30,
                new_portfolio_returns_computed=True, models_rebuilt_under_fixed_policy=True,
                complete_original_control_reproduced=True, full_point_in_time_dataset_complete=False,
                best_preserved=state['best_training_selected_candidate'], target_reached=False,
                target_full_period_cagr=.7, observers_modified=False, strategy_promoted=False,
                metrics=summary, conditional_minus_control=results['conditional_minus_control'],
                limitations=plan['uncertainty'], independent_review_sha256=PINS['independent_review/COMPLETION.json'],
                verified_artifacts_sha256={str(Path(p).relative_to(ROOT)): h for p, h in pins.items()},
                prior_central_sha256=expected_central, script_sha256=sha(__file__),
                next_step='Review a separately justified prospective data or feature hypothesis before further testing; do not tune PTR assumptions after observing returns.',
                interpretation='Completed fixed-model data sensitivity, not a new optimized strategy or full-market point-in-time certification. Existing strategy best remains unchanged.')


def updated_state(state, receipt, receipt_sha):
    result = copy.deepcopy(state)
    reference = dict(path=str((BASE / 'EVALUATION_PROGRESS.json').relative_to(ROOT)), sha256=receipt_sha,
                     strategy_trials=0, status=STATUS, full_point_in_time_dataset_complete=False)
    require(all(x.get('path') != reference['path'] for x in result['data_evidence_completions']), 'Already recorded')
    result['data_evidence_completions'].append(reference)
    result['latest_pool_data_progress'] = reference
    result['next'] = ('PTR fixed-incumbent data sensitivity completed: 2 data arms, 6 execution paths, 30 independently verified segments. '
                      'Original control CAGR67.666%, conditional CAGR67.036%; 167 strategy rounds / 1016 trials and best retained; 70% target unmet.')
    result['next_step'] = receipt['next_step']
    result['last_goal_turn_classification'] = 'progress'
    result['updated_utc'] = receipt['created_utc']
    changed = {k for k in set(state) | set(result) if state.get(k) != result.get(k)}
    require(changed <= ALLOWED, 'Unexpected central changes: ' + repr(changed - ALLOWED))
    for key in set(state) - ALLOWED:
        require(result[key] == state[key], 'Unrelated central field changed: ' + key)
    validate_registry(result)
    return result, reference, sorted(changed)


def commit(central, output, expected, before_bytes, after_bytes, receipt_bytes):
    """Compare expected bytes twice under an advisory lock, then atomic replace.

    All cooperating writers must honor the lock; an uncooperative writer between
    final comparison and replace cannot be excluded by portable filesystem CAS.
    """
    require(digest(before_bytes) == expected, 'Wrong expected central bytes')
    with central.open('rb') as locked:
        fcntl.flock(locked.fileno(), fcntl.LOCK_EX)
        require(central.read_bytes() == before_bytes, 'Concurrent central change before write')
        outputs = [output / n for n in ('EVALUATION_PROGRESS.json', 'CENTRAL_BEFORE_EVALUATION_PROGRESS.json',
                                        'CENTRAL_EVALUATION_UPDATE.json', 'CENTRAL_EVALUATION_PENDING.json')]
        require(not any(p.exists() for p in outputs), 'Closure is write-once')
        write_once(outputs[1], before_bytes)
        write_once(outputs[0], receipt_bytes)
        write_once(outputs[3], after_bytes)
        require(central.read_bytes() == before_bytes, 'Concurrent central change before atomic replace')
        require(os.stat(central).st_ino == os.fstat(locked.fileno()).st_ino, 'Central inode replaced concurrently')
        os.replace(outputs[3], central)
        require(central.read_bytes() == after_bytes, 'Central verification after replace failed')
        write_once(outputs[2], encode(dict(status='APPLIED', before=expected, after=digest(after_bytes),
                                          progress_receipt_sha256=digest(receipt_bytes),
                                          strategy_registry_and_best_preserved=True)))


def run(phase, expected):
    before = CENTRAL.read_bytes()
    require(digest(before) == expected, 'Concurrent central change; inspect before retry')
    state = json.loads(before)
    results, review, plan, pins = verify_sources()
    now = datetime.now(timezone.utc).isoformat()
    receipt = build_receipt(state, expected, results, review, plan, pins, now)
    encoded = encode(receipt)
    after, reference, changed = updated_state(state, receipt, digest(encoded))
    require(CENTRAL.read_bytes() == before, 'Central changed during check')
    if phase == 'apply':
        commit(CENTRAL, BASE, expected, before, encode(after), encoded)
    print(json.dumps(dict(status='CHECK_ONLY_NO_WRITES' if phase == 'check' else STATUS,
                          expected_central_sha256=expected, changed_fields=changed,
                          strategy_rounds=167, strategy_search_trials=1016, target_reached=False,
                          metrics=receipt['metrics'], receipt=reference if phase == 'apply' else None)))


class Tests(unittest.TestCase):
    def fixture(self):
        return dict(completed=[dict(trials=6) for _ in range(166)] + [dict(trials=20)],
                    registered_queue=[], current='monthly_sector_diffusion167', current_status='completed_independently_verified',
                    target_full_period_cagr=.7, data_evidence_completions=[dict(path='existing', sha256='prior')],
                    best_training_selected_candidate=dict(directory='monthly_local_linear70',
                        spec=dict(id=4, base='incumbent_scope67', mode='local_linear'), cagr=.6766617372041686, observer_promotion=False),
                    other_nested=dict(preserve=['a', 'b']))

    def test_preserves_unrelated_fields_and_strategy_counts(self):
        before = self.fixture()
        original = copy.deepcopy(before)
        after, _, changed = updated_state(before, dict(created_utc='test', next_step='test'), 'a' * 64)
        self.assertEqual(before, original)
        self.assertEqual(after['completed'], before['completed'])
        self.assertEqual(after['best_training_selected_candidate'], before['best_training_selected_candidate'])
        self.assertEqual(after['other_nested'], before['other_nested'])
        self.assertNotIn('target_reached', after)
        self.assertTrue(set(changed) <= ALLOWED)

    def test_rejects_new_trials_and_duplicate_closure(self):
        state = self.fixture(); state['completed'][0]['trials'] += 1
        with self.assertRaises(ValueError): validate_registry(state)
        state = self.fixture()
        after, _, _ = updated_state(state, dict(created_utc='test', next_step='test'), 'a' * 64)
        with self.assertRaises(ValueError): updated_state(after, dict(created_utc='test', next_step='test'), 'a' * 64)

    def test_cas_and_write_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); central = root / 'central.json'; output = root / 'output'; output.mkdir()
            before, after, receipt = b'{"a":1}\n', b'{"a":2}\n', b'{"receipt":true}\n'
            central.write_bytes(before)
            commit(central, output, digest(before), before, after, receipt)
            self.assertEqual(central.read_bytes(), after)
            self.assertEqual((output / 'CENTRAL_BEFORE_EVALUATION_PROGRESS.json').read_bytes(), before)
            with self.assertRaises(ValueError): commit(central, output, digest(after), after, before, receipt)

    def test_concurrent_change_fails_before_any_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); central = root / 'central.json'; output = root / 'output'; output.mkdir()
            expected = b'old'; central.write_bytes(b'new')
            with self.assertRaises(ValueError): commit(central, output, digest(expected), expected, b'after', b'receipt')
            self.assertEqual(list(output.iterdir()), [])
            self.assertEqual(central.read_bytes(), b'new')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('test', 'check', 'apply'))
    parser.add_argument('--central-sha256')
    args = parser.parse_args()
    if args.phase == 'test':
        unittest.main(argv=[sys.argv[0]], verbosity=2)
    else:
        require(args.central_sha256 is not None and len(args.central_sha256) == 64, 'Explicit central SHA256 required')
        run(args.phase, args.central_sha256)
