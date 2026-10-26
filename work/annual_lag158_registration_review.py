"""Reproducible metadata/code-closure receipt for the prospectively fixed pilot.

This module imports implementations only for constants, contracts and code-file
inventories. It never invokes model.build, either numerical audit, Runner, a
backtest, lifecycle.build_plan, registration or completion. Explicit SHA inputs
identify the final versions inspected by the reviewer before writing a receipt.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SELF = 'work/annual_lag158_registration_review.py'
OUT = 'work/annual_lag158_registration_review.json'
PROPOSAL = 'work/monthly_annual_lag158_proposal.json'
PROPOSAL_SHA = 'f8fb04286bea9d4215d70ff12371964210ebcab08a8819041d8e0f6d88706da3'
FIXED_REVIEWER = {
    'work/annual_lag158_independent_prepare.py': 'c5d57db0d112f79be603cf2913ec1731e031126ba4ca62e2f079ebb9d9cf3359',
    'work/test_annual_lag158_independent_prepare.py': '2b98f0c1ba5451731a128bf58931227f86f39edcd1ff37cce690304a74b4a05a',
}
ZERO_FIELDS = ('actual_label_calculations_before_registration', 'actual_lag_calculations_before_registration',
    'actual_model_fits_before_registration', 'actual_feature_calculations_before_registration',
    'actual_returns_before_registration', 'actual_nuisance_estimates_before_registration')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def local_path(name):
    raw = Path(name)
    raw = raw if raw.is_absolute() else ROOT/raw
    require(not any(x in str(raw) for x in ('CloudStorage', 'OneDrive')), 'Forbidden source path')
    return raw.resolve()


def code_function(path, function, cls=None):
    tree = ast.parse(Path(path).read_text())
    nodes = tree.body
    if cls:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == cls).body
    return next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == function)


def calls(node, suffix):
    return sorted(n.lineno for n in ast.walk(node) if isinstance(n, ast.Call) and
                  ((isinstance(n.func, ast.Attribute) and n.func.attr == suffix) or
                   (isinstance(n.func, ast.Name) and n.func.id == suffix)))


def review(args):
    inputs, implementations = {}, {}
    def pin(path, expected=None, code=False):
        path = local_path(path)
        actual = sha(path)
        if expected is not None:
            require(re.fullmatch('[0-9a-f]{64}', expected) and actual == expected, 'Reviewed version changed: ' + str(path))
        bucket = implementations if code else inputs
        require(str(path) not in bucket or bucket[str(path)] == actual, 'Conflicting shared source pin')
        bucket[str(path)] = actual
        return actual
    for name, digest in FIXED_REVIEWER.items():
        pin(name, digest, True)
    for name, digest in ((SELF, args.self_sha),
        ('work/monthly_annual_lag158_runner.py', args.runner_sha),
        ('work/monthly_annual_lag158_lifecycle.py', args.lifecycle_sha),
        ('work/annual_lag158_independent_outcome.py', args.outcome_sha)):
        pin(name, digest, True)
    pin(PROPOSAL, PROPOSAL_SHA)
    proposal = json.loads((ROOT/PROPOSAL).read_text())
    # Only the following module functions are invoked: source_files and pure
    # literal contract validators. No registered-data entry point is called.
    runner = importlib.import_module('monthly_annual_lag158_runner')
    lifecycle = importlib.import_module('monthly_annual_lag158_lifecycle')
    prepare = importlib.import_module('annual_lag158_independent_prepare')
    outcome = importlib.import_module('annual_lag158_independent_outcome')
    for name in runner.source_files(ROOT):
        pin(name, code=True)
    for module in (prepare, outcome):
        for name in module.source_files(ROOT):
            pin(name, code=True)
    for name in ('work/test_monthly_annual_lag158_lifecycle.py', lifecycle.old.__file__,
                 lifecycle.prior.__file__, 'run_local.sh', 'local_run.py', 'local.sandbox.sb', 'AGENTS.md', 'LOCAL_SETUP.md'):
        pin(name, code=True)
    require(tuple(runner.NAMES) == tuple(prepare.NAMES) == tuple(outcome.NAMES) == tuple(lifecycle.NAMES), 'Three-arm agreement')
    require(tuple(runner.EXTRA) == tuple(prepare.EXTRA) == tuple(outcome.EXTRA), 'Five extra arrays agreement')
    require(tuple(runner.INPUTS) == tuple(prepare.INPUTS), 'Named input agreement')
    require(tuple(runner.CACHE_FILES) == tuple(prepare.CACHE_FILES) == tuple(outcome.CACHE_FILES), 'Cache contract agreement')
    require(tuple(runner.FROZEN_FILES) == tuple(outcome.FROZEN_FILES), 'Frozen first88 contract agreement')
    require(tuple(runner.CHECKS) == tuple(prepare.CHECKS), 'Prepare receipt agreement')
    require(runner.OUT == prepare.OUT == outcome.STUDY == lifecycle.OUT, 'Dedicated output agreement')
    require(runner.measurement_contract() == prepare.measurement_contract() == proposal['measurement'] and
            runner.selection_contract() == prepare.selection_contract() == proposal['selection_contract'], 'Prospective policy agreement')
    runner.validate_proposal(proposal)
    for name, (path, digest) in prepare.FIXED.items():
        require(runner.FIXED_INPUTS[name] == (path, digest), 'Fixed source contract agreement')
        pin(path, digest)
        require(proposal['source_inputs'][name]['sha256'] == digest and
                local_path(proposal['source_inputs'][name]['path']) == local_path(path), 'Proposal source reference agreement')
    for path, digest in ((lifecycle.DESIGN_REVIEW, lifecycle.DESIGN_REVIEW_SHA),
                         (lifecycle.PREVIOUS, lifecycle.PREVIOUS_SHA),
                         (lifecycle.PREVIOUS_PLAN, lifecycle.PREVIOUS_PLAN_SHA),
                         (lifecycle.PREVIOUS_REPAIR, lifecycle.PREVIOUS_REPAIR_SHA)):
        pin(path, digest)
    tests_path = ROOT/lifecycle.OUT/'TESTS.json'
    pin(tests_path, '5fed17866c0f360207010d0ecd4a1a8578f7def21dc2e18f5a44a9aa8841c555')
    tests = json.loads(tests_path.read_text())
    require(tests.get('passed') is True and tests.get('synthetic_only') is True,
            'Passed synthetic-only test receipt required')
    counts = {}
    for row in tests['suites']:
        name, count = row['name'], row['count']
        require(name in lifecycle.SUITES and name not in counts and type(count) is int and
                count > 0 and row.get('passed') is True, 'Exact distinct passed test suites')
        require(local_path(row['source']['path']) == ROOT/'work'/lifecycle.SUITES[name],
                'Test suite source path mismatch')
        pin(row['source']['path'], row['source']['sha256'], True)
        pin(row['log']['path'], row['log']['sha256'])
        log = local_path(row['log']['path']).read_text()
        require(re.findall(r'Ran (\d+) tests? in ', log) == [str(count)] and log.rstrip().endswith('OK'),
                'Unsuccessful or ambiguous synthetic test log')
        counts[name] = count
    require(set(counts) == set(lifecycle.SUITES) and type(tests.get('test_count')) is int and
            tests['test_count'] == sum(counts.values()) == 192, 'Five-suite192 test accounting')
    require(lifecycle.HISTORICAL_SHA == prepare.HISTORICAL_SHA == runner.HISTORICAL_SHA and
            lifecycle.ARCHIVE == prepare.HISTORICAL_ARCHIVE == runner.HISTORICAL_ARCHIVE, 'Unique archive alias agreement')
    pin(lifecycle.ARCHIVE, lifecycle.HISTORICAL_SHA)
    for key in ('source_overlay', 'source_review'):
        document = json.loads(local_path(prepare.FIXED[key][0]).read_text())
        for name, digest in document['input_hashes'].items():
            path = local_path(name)
            if path == local_path(prepare.HISTORICAL_STATE):
                require(digest == prepare.HISTORICAL_SHA, 'No other mutable-state alias')
                path = local_path(prepare.HISTORICAL_ARCHIVE)
            pin(path, digest)
    # Concrete ordering checks support, but do not replace, the recorded manual review.
    rpath = ROOT/'work/monthly_annual_lag158_runner.py'
    ppath = ROOT/'work/annual_lag158_independent_prepare.py'
    prepare_phase = code_function(rpath, 'prepare', 'Runner')
    require(calls(prepare_phase, 'require_sources')[0] < calls(prepare_phase, 'load')[0] <
            calls(prepare_phase, 'build_features')[0], 'Source preflight must precede backend load and numerical features')
    require(calls(code_function(rpath, 'write', 'Runner'), 'refresh')[0] <
            calls(code_function(rpath, 'write', 'Runner'), 'write')[0], 'Each artifact write must refresh pinned inputs')
    require(not calls(code_function(rpath, 'training', 'Backend'), 'select'), 'Training must not select prefix models')
    require(len(calls(code_function(rpath, 'select_first88', 'Runner'), 'select')) == 1, 'Only one first88 promotion selection')
    audit = code_function(ppath, 'audit')
    require(calls(audit, 'validate_plan')[0] < calls(audit, 'source_inputs')[0] < calls(audit, 'npz')[0], 'Independent audit source-first ordering')
    init_strings = {n.value for n in ast.walk(code_function(rpath, '__init__', 'Runner')) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    plan_keywords = {n.arg for n in ast.walk(code_function(ROOT/lifecycle.SELF, 'build_plan')) if isinstance(n, ast.keyword)}
    require(set(ZERO_FIELDS) <= init_strings and set(ZERO_FIELDS) <= plan_keywords, 'All zero-activity fields checked and registered')
    forbidden = set(runner.CACHE_FILES) | {'PLAN.json', 'CACHE.json', 'FROZEN.json', 'TRAIN_TRIALS.json',
        'TRAIN_OPEN_TRIALS.json', 'WALKFORWARD.json', 'DIAGNOSE_STARTED.json', 'RESULTS.json', 'COMPLETION.json'}
    require(not any((ROOT/runner.OUT/name).exists() for name in forbidden), 'Registration review must precede actual experiment')
    for name, digest in {**inputs, **implementations}.items():
        require(sha(name) == digest, 'Evidence changed while reviewing: ' + name)
    return dict(schema='annual_lag158_prospective_registration_review_v1', round=158, passed=True,
        created_utc=datetime.now(timezone.utc).isoformat(), proposal_sha256=PROPOSAL_SHA,
        actual_features_or_returns_computed=False, actual_lag_values_computed=False, actual_orders_computed=False,
        strategy_registered=False, synthetic_tests=counts, synthetic_test_count=sum(counts.values()),
        implementation_hashes=implementations, input_hashes=inputs,
        checks={key: True for key in (
            'three_fixed_arms_and_only_lag_change_promotable', 'contracts_and_artifact_schemas_agree',
            'fixed_source_design_and_historical_aliases_bound', 'full_source_review_closure_bound',
            'source_preflight_precedes_backend_load_and_numerical_features', 'admitted_contradictions_hard_fail',
            'original_eligibility_and27_arrays_preserved', 'two_raw_lags_common_mask_and_centered_rank_bound',
            'original_local67_pre_plus_once_only_correction_and_selector', 'initial_missing_neutral_exact_fallback',
            'all_zero_activity_fields_registered_and_checked', 'every_artifact_write_refreshes_pins',
            'one_first88_selection_no_fold_reselection', 'locked_later34_and9_path_independent_accounting',
            'completion_requires_independent_artifacts_and_protected_archive', 'five_synthetic_suites_and_logs_bound',
            'no_actual_experiment_artifacts_present',
            'implementation_and_evidence_bytes_unchanged_during_review')},
        scope=dict(method='Manual review of new model/runner/lifecycle/prepare/outcome code plus executable metadata/AST agreement checks.',
            code_inventory='Union of runner.source_files, both independent reviewer.source_files, lifecycle/test and wrapper dependencies.',
            roles='The reviewer authored independent_prepare. Runner/lifecycle/outcome were written by other agents. Independent date/rank arithmetic was cross-reviewed by the runner and outcome authors; this receipt does not claim a separate author reviewed every dependency.',
            frozen_dependencies='Frozen numerical selector, accounting and loader helpers are hash-bound reused dependencies, not re-derived in this registration review.'),
        limits=['This is a preregistration code/metadata review, not a passed actual preparation or outcome audit.',
                'The full five-suite TESTS receipt is independently required by lifecycle registration.',
                'KO/WMT limited coverage, repeated pairs, reconstructed availability and historical price-identity limitations remain.',
                'Price data, real lag arithmetic, scores, orders, returns and target achievement were not evaluated here.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('self', 'runner', 'lifecycle', 'outcome'):
        parser.add_argument('--'+name+'-sha', required=True)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    result = review(args)
    if args.write:
        path = ROOT/OUT
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'w') as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps(dict(passed=True, implementation_files=len(result['implementation_hashes']),
        metadata_files=len(result['input_hashes']), output=OUT if args.write else None), sort_keys=True))


if __name__ == '__main__':
    main()
