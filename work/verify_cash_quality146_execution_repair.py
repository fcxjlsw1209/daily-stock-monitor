"""Verify an execution-only amendment without running factors or replaying orders."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import re

from cashflow_source_union_v1_independent import Evidence, digest, encoded, need

ROOT = Path(__file__).resolve().parents[1]
OUT = 'outputs/monthly_cash_quality146'
SELF = 'work/verify_cash_quality146_execution_repair.py'
TEST = 'work/test_verify_cash_quality146_execution_repair.py'
HELPER = 'work/cashflow_source_union_v1_independent.py'
HELPER_SHA = 'a71c7e16a5cf97a020726511d657a4bf33eb40dbcc7f02190ee7413561497929'
PLAN_SHA = '23807fe79e5cb9bd6a21273ec7ade26d8a6cb36aab18715e2e052fe1437e81f2'
FAILURE_SHA = '13ae62f9f893a4152164a20ee3050c0f3ba30ce8b920a2297598b1d205ddbd53'
RUNTIME = '/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9'
RUNTIME_SHA = '6b24be8ba00c4abc8b3e7e2620bd710e2cf7a4312d00c147f8f05fbb6f3d297a'
ADAPTER = 'work/monthly_cash_quality146_prefix_adapter.py'
ADAPTER_TEST = 'work/test_monthly_cash_quality146_prefix_adapter.py'
ADAPTER_REVIEW = 'work/monthly_cash_quality146_prefix_adapter_review.json'
AMENDMENT = OUT+'/PREFIX_EXECUTION_AMENDMENT.json'
CACHE_FILES = ('DATA.npz', 'SCORES.npz', 'META.json', 'FEATURE_AUDIT.json',
               'PRETRAIN_ORDERS.json', 'PRICE_PARITY.json')
PREPARE_CHECKS = (
    'original27_arrays_byte_identical', 'all_features_independently_reconstructed',
    'all_four_score_matrices_and_ordered_choices_independently_verified',
    'one_episode_and_prefix_isolation_verified', 'zero_and_missing_exact_fallback',
    'source_admission_bound', 'source_union_preservation_verified',
    'raw_prices_independently_verified', 'calendar_overlay_independently_verified',
    'all_signal_prefixes_verified', 'exact_integer_closed_form_arithmetic')
OUTCOME_CHECKS = (
    'daily_nav_trades_holdings_metrics_slices', 'first88_and_three_fold_selection_verified',
    'first88_dual_execution_selection_verified', 'annual_partial_flags_verified',
    'first88_causal_replay_verified', 'zero_model_fit_count_verified',
    'source_admission_bound', 'common_availability_verified',
    'baseline_local67_full_orders_unchanged', 'diagnostic_start_marker_verified')

class PlanEvidence(Evidence):
    """Source verifier plus the original PLAN's one fixed external interpreter."""
    def path(self, name):
        if name == RUNTIME:
            path = Path(name)
            need(path.resolve() == path, 'Registered interpreter must not be a symlink')
            return path
        return super().path(name)

    def hash(self, name, expected=None):
        if name == RUNTIME:
            need(expected is None or expected == RUNTIME_SHA, 'Fixed registered interpreter digest')
            expected = RUNTIME_SHA
        return super().hash(name, expected)


def utc(value):
    need(type(value) is str and value.endswith(('Z', '+00:00')), 'Explicit UTC time required')
    result = datetime.fromisoformat(value[:-1]+'+00:00' if value.endswith('Z') else value)
    need(result.tzinfo is not None and result.utcoffset().total_seconds() == 0, 'UTC time required')
    return result


def exact_count(record, name, expected):
    need(type(record.get(name)) is int and record[name] == expected, 'Exact count: '+name)


def normalized(evidence, mapping):
    evidence.pins(mapping)
    result = {}
    for name, sha in mapping.items():
        key = str(evidence.path(name))
        need(key not in result or result[key] == sha, 'Conflicting normalized hashes')
        result[key] = sha
    return result


def require_pins(evidence, actual, expected):
    pins = normalized(evidence, actual)
    for name, sha in expected.items():
        need(pins.get(str(evidence.path(name))) == sha, 'Required evidence binding: '+name)


def validate(root, amendment_sha, phase, amendment_path=AMENDMENT, *,
             expected_plan=PLAN_SHA, expected_failure=FAILURE_SHA, now=None):
    """Testable read-only checks; CLI always uses the fixed real PLAN/failure pins."""
    need(phase in ('prepare', 'complete'), 'Declared verification phase required')
    evidence = PlanEvidence(root)
    plan = evidence.read(OUT+'/PLAN.json', expected_plan)
    need(plan['round'] == 146 and plan['status'] == 'REGISTERED', 'Original registered PLAN')
    frozen = normalized(evidence, plan['hashes'])
    amendment = evidence.read(amendment_path, amendment_sha)
    need(amendment['plan_sha256'] == expected_plan
         and amendment['repair'] == 'full_pre_score_to_requested_prefix'
         and amendment['selection_or_factor_changes'] is False, 'Exact execution-only amendment')
    required = (OUT+'/PLAN.json', OUT+'/PREPARE.log', ADAPTER, ADAPTER_TEST,
                ADAPTER_REVIEW, OUT+'/PREFIX_ADAPTER_TESTS.log')
    direct = {name: evidence.hash(name) for name in required}
    need(direct[OUT+'/PREPARE.log'] == expected_failure, 'Original failed attempt log changed')
    require_pins(evidence, amendment['bindings'], direct)
    # The newly bound adapter is deliberately outside the original frozen PLAN.
    # Every original frozen dependency nevertheless remains byte-identical.
    need(str(evidence.path(ADAPTER)) not in frozen, 'Adapter must be disclosed as the post-registration repair')
    failure = evidence.path(OUT+'/PREPARE.log').read_text()
    need('Traceback (most recent call last):' in failure
         and 'ValueError: Original pre-score shape' in failure, 'Expected original interface failure')
    review = evidence.read(ADAPTER_REVIEW)
    need(review['passed'] is True and review['plan_sha256'] == expected_plan
         and review['repair'] == amendment['repair']
         and review['selection_or_factor_changes'] is False, 'Passing independent adapter review')
    require_pins(evidence, review['input_hashes'], {
        name: direct[name] for name in (OUT+'/PLAN.json', OUT+'/PREPARE.log', ADAPTER, ADAPTER_TEST)})
    need(review['checks']['full_122_pre_score_bytes_unchanged'] is True
         and review['checks']['same_n_passed_to_original_backend'] is True
         and review['checks']['original_baseline_method_not_overridden'] is True,
         'Independent review of repair scope')
    test_log = evidence.path(OUT+'/PREFIX_ADAPTER_TESTS.log').read_text()
    match = re.search(r'Ran ([1-9][0-9]*) tests? in ', test_log)
    need(match and test_log.rstrip().endswith('OK'), 'Passing adapter synthetic log')
    started = evidence.read(OUT+'/PREFIX_ADAPTER_STARTED.json')
    need(started['plan_sha256'] == expected_plan and started['amendment_sha256'] == amendment_sha,
         'Actual adapter start links')
    started_at = utc(started['started_utc'])
    current_time = now or datetime.now(timezone.utc)
    need(utc(plan['registered_utc']) <= started_at <= current_time, 'Actual start after registration, not future')
    need(utc(plan['registered_utc']) <= utc(amendment.get('created_utc')) <= started_at,
         'Amendment predates execution')
    need(utc(review['reviewed_utc']) <= started_at, 'Adapter reviewed before execution')
    execution = evidence.read(OUT+'/PREFIX_ADAPTER_EXECUTION.json')
    cache_sha = evidence.hash(OUT+'/CACHE.json')
    need(execution['plan_sha256'] == expected_plan and execution['amendment_sha256'] == amendment_sha
         and execution['cache_sha256'] == cache_sha
         and execution['full_horizon_pre_score_unchanged'] is True
         and execution['selection_or_factor_changes'] is False, 'Completed adapter/cache binding')
    cache = evidence.read(OUT+'/CACHE.json')
    need(set(cache) == set(CACHE_FILES), 'Exact six cache artifacts')
    cache_pins = {OUT+'/'+name: sha for name, sha in cache.items()}
    normalized(evidence, cache_pins)
    prepare = evidence.read(OUT+'/INDEPENDENT_PREPARE.json')
    need(prepare['passed'] is True and prepare['stage'] == 'prepare'
         and prepare['plan_sha256'] == expected_plan and prepare['cache_sha256'] == cache_sha
         and prepare['orders_sha256'] == cache['PRETRAIN_ORDERS.json'], 'Passing exact-cache independent prepare')
    need(all(prepare.get(key) is True for key in PREPARE_CHECKS), 'All independent preparation checks')
    need(prepare['producer_modules_imported'] is False and prepare['returns_or_nav_computed'] is False,
         'Independent prepare isolation')
    for key, value in (('orders_checked', 488), ('source_cells_checked', 5978),
                       ('stock_cells_checked', 3904), ('new_model_fits', 0), ('monthly_label_models_fit', 0)):
        exact_count(prepare, key, value)
    need(normalized(evidence, prepare['source_hashes'])
         == normalized(evidence, plan['independent_prepare_source_hashes']), 'Frozen prepare reviewer source')
    require_pins(evidence, prepare['artifact_hashes'],
                 dict(cache_pins, **{OUT+'/PLAN.json': expected_plan, OUT+'/CACHE.json': cache_sha}))
    prepare_started = evidence.read(OUT+'/INDEPENDENT_PREPARE_STARTED.json')
    need(prepare_started['plan_sha256'] == expected_plan
         and prepare_started['amendment_sha256'] == amendment_sha, 'Independent prepare start links')
    prepare_at = utc(prepare_started['started_utc'])
    need(started_at < prepare_at <= current_time, 'Adapter start precedes actual independent preparation')
    checks = dict(original_PLAN_and_entire_frozen_closure_unchanged=True,
                  original_failure_and_independent_scope_review_bound=True,
                  repaired_execution_bound_to_exact_final_cache=True,
                  actual_start_precedes_independent_preparation=True,
                  independent_prepare_confirms_all_features_scores_orders_and_prefixes=True)
    if phase == 'complete':
        earlier = evidence.read(OUT+'/EXECUTION_REPAIR_PREPARE_REVIEW.json')
        need(earlier['passed'] is True and earlier['phase'] == 'prepare'
             and earlier['plan_sha256'] == expected_plan and earlier['amendment_sha256'] == amendment_sha
             and earlier['cache_sha256'] == cache_sha, 'Prior repair preparation review')
        require_pins(evidence, earlier['input_hashes'], {
            OUT+'/INDEPENDENT_PREPARE.json': evidence.hash(OUT+'/INDEPENDENT_PREPARE.json'),
            OUT+'/PREFIX_ADAPTER_EXECUTION.json': evidence.hash(OUT+'/PREFIX_ADAPTER_EXECUTION.json')})
        outcome = evidence.read(OUT+'/INDEPENDENT_OUTCOME.json')
        need(outcome['passed'] is True and outcome['stage'] == 'evaluate'
             and outcome['plan_sha256'] == expected_plan and outcome['cache_sha256'] == cache_sha,
             'Passing exact-cache independent outcome')
        need(all(outcome.get(key) is True for key in OUTCOME_CHECKS), 'All independent outcome checks')
        for key, value in (('execution_paths_verified', 12), ('total_monthly_orders_verified', 488),
                           ('cash_quality_feature_matrices_verified', 4), ('models_fit', 0), ('producer_model_fits', 0)):
            exact_count(outcome, key, value)
        need(outcome['producer_modules_imported'] is False, 'Independent outcome isolation')
        need(normalized(evidence, outcome['source_hashes'])
             == normalized(evidence, plan['independent_outcome_source_hashes']), 'Frozen outcome reviewer source')
        require_pins(evidence, outcome['artifact_hashes'], {
            **cache_pins, OUT+'/PLAN.json': expected_plan, OUT+'/CACHE.json': cache_sha,
            OUT+'/INDEPENDENT_PREPARE.json': evidence.hash(OUT+'/INDEPENDENT_PREPARE.json')})
        outcome_at = utc(outcome['audit_started_utc'])
        need(prepare_at < outcome_at <= current_time, 'Chronological independent outcome')
        completion = evidence.read(OUT+'/COMPLETION.json')
        need(completion['round'] == 146 and completion['directory'] == 'monthly_cash_quality146'
             and completion['plan_sha256'] == expected_plan
             and completion['independent_outcome_sha256'] == evidence.hash(OUT+'/INDEPENDENT_OUTCOME.json'),
             'Completion exact independent outcome binding')
        need(outcome_at <= utc(completion['completed_utc']) <= current_time, 'Completion after independent outcome')
        exact_count(completion, 'trials', 4)
        exact_count(completion, 'independent_execution_paths', 12)
        exact_count(completion, 'protected_files', 21444)
        need(evidence.hash(OUT+'/PROTECTED_AFTER.json') == completion['protected_sha256'],
             'Completion protected receipt binding')
        checks.update(prior_repair_preparation_review_verified=True,
                      independent_outcome_12_paths_488_orders_verified=True,
                      completion_bound_to_review_and_protected_receipt=True)
    evidence.stable()
    return dict(schema_version=1, stage='independent_execution_repair_verification', phase=phase, passed=True,
                plan_sha256=expected_plan, amendment_sha256=amendment_sha, cache_sha256=cache_sha,
                verified_utc=current_time.isoformat(), checks=checks, input_hashes=dict(evidence.hashes),
                verifier_performed_factor_calculations=0, verifier_performed_return_calculations=0,
                verifier_performed_model_fits=0, verifier_performed_strategy_registrations=0,
                original_attempt_note='Real features and partial temporary orders preceded the shape failure; no cache or returns were published by that failed attempt.',
                limitation='Verifies recorded evidence chronology and exact bytes; timestamps are local recorded UTC, not externally attested time.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--amendment-sha', required=True)
    parser.add_argument('--phase', choices=('prepare', 'complete'), required=True)
    parser.add_argument('--amendment', default=AMENDMENT)
    args = parser.parse_args()
    result = validate(ROOT, args.amendment_sha, args.phase, args.amendment)
    e = Evidence(ROOT)
    e.hash(HELPER, HELPER_SHA)
    result['input_hashes'].update({name: e.hash(name) for name in (SELF, TEST, HELPER)})
    name = 'EXECUTION_REPAIR_PREPARE_REVIEW.json' if args.phase == 'prepare' else 'EXECUTION_REPAIR_COMPLETION_REVIEW.json'
    output = ROOT/OUT/name
    with output.open('xb') as stream:
        stream.write(encoded(result)); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps(dict(passed=True, phase=args.phase, output=str(output),
                          sha256=digest(encoded(result))), sort_keys=True))


if __name__ == '__main__':
    main()
