"""Bind final source-only recovery165 registration and synthetic evidence.

No producer import, market-array decoding, feature/order/return calculation,
registration or central-state mutation. Only the write-once review is saved.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import recovery165_independent as independent


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / 'work/recovery165_registration_tests.json'
EXPECTED_RECEIPT = 'd719376157ca9208ac570624de658ab8c86da4ceb3065ba7b90a04358ff7fb60'
EXPECTED_STATE = 'fd9383210b17f17656544af64246f0b208ab98b8323a2440de9f5e105d705a29'
DESTINATION = ROOT / 'work/recovery165_registration_review.json'
SUITES = {'kernel': 16, 'runner': 32, 'host': 24, 'independent': 52}
REVIEWED = {
    'work/recovery165_host.py': 'a62d98781f66445351d133e526f15640ef743d8cafb8f7cf617cff52dc4e50f6',
    'work/recovery165_kernel.py': 'f44b4c48c4730763a82a6229ddef407565e922a845bcb020a5bf13372b73ef92',
    'work/recovery165_runner.py': '89f590f8bdd6e34e46bced617f18df24481102512820f20a9a9ac159cead04da',
    'work/recovery165_validate_code.py': 'a28353574f280ce1c9592608da6afd952844a9f7146749e33a82db1229e2bac2',
    'work/recovery165_proposal.json': 'ad58156a8d0d450c63d08d3d22f83cea60eb55619a7919998b19eb86436c5a4f',
    'work/recovery165_source_probe.py': '3ec3be919b46dbeb4f84da49c59444e688a5ac80dbe0a3a8450e7283d3b69171',
    'work/recovery165_design_review.json': '97fb7e744b04ed0e5dd41faebc285ec595ea2c07a89f767d13ad90cfd65025c2',
    'work/recovery165_novelty_review.json': '29b26e28fdd8b453bf8b544beda6f89b300dd1bbd26d31d2121047fb8f3e9d79',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def path(value):
    item = Path(value)
    item = (item if item.is_absolute() else ROOT / item).resolve()
    require(item.is_relative_to(ROOT), 'Review input must be inside local research root')
    require(not any(part in str(item) for part in ('OneDrive', 'CloudStorage', 'stock-observer-runtime')),
            'Forbidden input')
    return item


def digest(value):
    return hashlib.sha256(path(value).read_bytes()).hexdigest()


def read_pinned(value, expected):
    raw = path(value).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, 'Input changed: ' + str(value))
    return json.loads(raw)


def zero_activity(item):
    require(all(type(item.get(key)) is int and item[key] == 0 for key in (
        'actual_features_computed', 'actual_orders_computed', 'actual_returns_computed')),
        'Source-only zero-activity counters required')


def main():
    require(not DESTINATION.exists(), 'Write-once registration review already exists')
    require(not (ROOT / 'outputs/monthly_recovery165/PLAN.json').exists(), 'Review must precede registration')
    independent.prepare.independent_imports()
    require(all(digest(name) == sha for name, sha in REVIEWED.items()), 'Reviewed source bytes changed')
    expected_paths = sorted(set(independent.source_files(ROOT) + [ROOT / 'work' / name for name in (
        'recovery165_host.py', 'recovery165_kernel.py', 'recovery165_runner.py',
        'test_recovery165_kernel.py', 'test_recovery165_runner.py', 'test_recovery165_host.py',
        'recovery165_validate_code.py')]))
    implementation = {str(item): digest(item) for item in expected_paths}
    require(len(implementation) == 22, 'Twenty-two exact validation paths required')
    receipt = read_pinned(RECEIPT, EXPECTED_RECEIPT)
    require(receipt['passed'] is True and receipt['synthetic_only'] is True and
            receipt['real_features_orders_or_returns_computed'] is False, 'Synthetic-only aggregate required')
    require(receipt['implementation_hashes'] == implementation, 'Exact complete validation_paths closure required')
    require([row['suite'] for row in receipt['suites']] == list(SUITES), 'Four exact suites required')
    evidence = {str(RECEIPT): EXPECTED_RECEIPT}
    for row in receipt['suites']:
        require(row['tests_run'] == SUITES[row['suite']], 'Final suite count changed')
        item = read_pinned(row['receipt']['path'], row['receipt']['sha256'])
        require(item['suite'] == row['suite'] and item['tests_run'] == row['tests_run'], 'Suite identity/count mismatch')
        require(item['passed'] is True and item['synthetic_only'] is True and item['code_unchanged'] is True,
                'Suite must pass with unchanged implementation')
        require(item['failures'] == item['errors'] == item['skipped'] == 0, 'No failed or skipped tests')
        require(item['implementation_hashes'] == implementation, 'Suite used different code')
        require(item['log'] == row['log'] and digest(row['log']['path']) == row['log']['sha256'], 'Suite log binding')
        evidence[str(path(row['receipt']['path']))] = row['receipt']['sha256']
        evidence[str(path(row['log']['path']))] = row['log']['sha256']
    require(receipt['tests_run'] == sum(SUITES.values()) == 124, '124 final tests required')

    source_names = ('recovery165_source_probe.py', 'recovery165_proposal.json',
                    'recovery165_design_review.json', 'recovery165_novelty_review.json')
    source_pins = {str(ROOT/'work'/name): digest(ROOT/'work'/name) for name in source_names}
    design = read_pinned('work/recovery165_design_review.json', REVIEWED['work/recovery165_design_review.json'])
    novelty = read_pinned('work/recovery165_novelty_review.json', REVIEWED['work/recovery165_novelty_review.json'])
    proposal = read_pinned('work/recovery165_proposal.json', REVIEWED['work/recovery165_proposal.json'])
    zero_activity(design); zero_activity(novelty); zero_activity(proposal)
    require(proposal['round'] == 165 and proposal['arms'] ==
            ['incumbent_local67', 'recovery_race', 'impact_control'], 'Exact three-arm prospective proposal')
    require(design['proposal_sha256'] == REVIEWED['work/recovery165_proposal.json'],
            'Current final proposal must be bound by design review')
    require(design['input_hashes'] == design['source_hashes'], 'Source closure maps must agree')
    for name in ('recovery165_source_probe.py', 'recovery165_proposal.json', 'recovery165_novelty_review.json'):
        key = str(ROOT/'work'/name)
        require(design['input_hashes'].get(key) == source_pins[key], 'Current source review binding required')
    require(all(digest(name) == sha for name, sha in design['input_hashes'].items()), 'Source inputs changed')
    raw_plan_path = 'outputs/monthly_paired_target143/PLAN.json'
    raw_plan = read_pinned(raw_plan_path, design['input_hashes'][str(ROOT/raw_plan_path)])
    raw_sources = raw_plan['raw_source_hashes']
    require(len(raw_sources) == 64, 'Exact original64 raw sources required')
    require(all(design['input_hashes'].get(str(path(name))) == sha for name, sha in raw_sources.items()),
            'All original raw sources must be bound by metadata review')
    require(design['passed'] is True and design['conditional_research_sources_ready'] is True and
            design['strict_pit_vintage'] is False and design['strict_pit_membership'] is False and
            design['strict_pit_sector_classification'] is False and design['strategy_admitted'] is False,
            'Conditional source review with explicit uncertified sectors required')
    for name in ('outputs/index_union/alias_resolved_inputs/sectors.csv.gz',
                 'outputs/monthly_strict_relative/strict_study.py'):
        key = str(ROOT/name)
        require(design['input_hashes'].get(key) == implementation.get(key) == digest(name),
                'Final synthetic and metadata reviews must bind the same sector inputs')
    summary = design['summary']
    require(summary['event_months'] == 122 and summary['training_months'] == 180 and
            summary['all_overlap_sets_equal'] is True and summary['noninitial_dates_verified_monthend'] is True and
            summary['historical_names_absent_from_price_panel'] == [] and summary['market_proxy_present'] is True,
            'Metadata identity review incomplete')
    require(novelty['passed'] is True and novelty['status'] == 'BOUNDED_REVIEW_NO_EXACT_DUPLICATE_FOUND' and
            novelty['registration_authorized_by_review'] is False, 'Bounded novelty scope required')
    novelty_pins = {str(path(name)): sha for name, sha in novelty['checked_files_sha256'].items()}
    require(len(novelty_pins) == 12 and all(digest(name) == sha for name, sha in novelty_pins.items()),
            'Twelve-source novelty snapshot changed')
    state_path = 'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
    state = read_pinned(state_path, EXPECTED_STATE)
    require(len(state['completed']) == 164 and sum(row['trials'] for row in state['completed']) == 1007 and
            state['registered_queue'] == [] and state['completed'][-1]['directory'] == 'monthly_activity164',
            'Expected unregistered164/1007 central state')

    report = dict(
        passed=True, schema='recovery165_registration_review_v1', round=165,
        created_utc=datetime.now(timezone.utc).isoformat(),
        host_sha256=implementation[str(ROOT/'work/recovery165_host.py')],
        registration_tests_sha256=EXPECTED_RECEIPT, implementation_hashes=implementation,
        tests_run=124, suite_counts=SUITES, test_evidence_hashes=evidence, design_evidence_hashes=source_pins,
        source_input_hashes=design['input_hashes'], raw_source_count=64, novelty_snapshot_hashes=novelty_pins,
        preregistration_state=dict(path=str(ROOT/state_path), sha256=EXPECTED_STATE, completed=164, trials=1007,
                                  registered_queue_empty=True),
        reviewer_scope='Independent source-only lifecycle, fixed formula, source binding and final synthetic evidence review; reviewer authored the bounded novelty audit but not the producer or numerical oracle.',
        checks={
            'complete_current22_validation_paths_match_all_four_suites': True,
            'all124_synthetic_tests_pass_with_unchanged_code_and_bound_logs': True,
            'fixed211_prices_189_joint_down_starts_21_session_recovery_censor22': True,
            'integer_recovery_sum_preserves_discrete_ties_and_control_uses_fsum': True,
            'matched_initial_impact_control_same_events_shared_stock_support_min20': True,
            'positive04dose_max02delta_initial_partial_and_etfs_neutral': True,
            'frozen_signal_date_mapping_independently_reconstructed_from_ast_constants': True,
            'explicit_uncertified_vintage_membership_and_sector_flags': True,
            'registration_precedes_actual_features_orders_and_outcomes': True,
            'source_probe_hashes_market_bytes_but_decodes_only_dates_and_symbols': True,
            'all64_raw_sources_official_calendar_original_axes_and_novelty_bound': True,
            'novelty_source_snapshot_verified_before_central_registration': True,
            'first88_once_selection_precedes_later34_diagnostics': True,
            'candidate_must_pass_incumbent_and_control_guards_both_execution_paths': True,
            'impact_control_cannot_be_selected_or_promoted': True,
            'all5978_cells366_orders123_prefixes_required_by_prepare_gate': True,
            'original27_arrays_plus2_deltas_once_then_original_episode': True,
            'all9_paths_required_by_independent_outcome_gate': True,
            'full21444_protected_manifest_required_before_completion': True,
            'prior164_1007_history_preserved_and_completed_counts165_1010': True,
            'new_best_requires_training_admission_and_verified_full_reference_gain': True,
            'completion_and_central_best_share_verified_expected_object': True,
            'write_once_artifacts_and_locked_atomic_state_commit': True,
            'stale_prior_count_fixture_and_audit_schema_fixed_before_final_tests': True,
        },
        unresolved_blocking_findings=[],
        limitations=[
            'This approves registration mechanics and the fixed reviewed design, not numerical source coverage, profitability or70percent achievement.',
            'Bounded local novelty review found no exact duplicate; missing historical sector leadlag/residual originals prevent exhaustive nonduplication certification.',
            'Hashing market input bytes is not numeric decoding; this reviewer did not compute actual features, orders or returns.',
            'Actual preparation and outcomes still require registered independent verification; the21444-file archive audit is enforced later and was not repeated here.',
            'Historical price vintages, estimated historical pools, assumed ETF list and sector mappings remain conditional research inputs; later34 is reused development history.',
            'Joint-down conditioning selects dependent overlapping events; censor22 is not eventual recovery time, and the impact control does not identify a causal timing effect.',
            'The central best is a descriptive full-period reference among first88-admitted recovery candidates; it does not change the frozen first88 decision or observation accounts.',
            'Preregistration central-state bytes bind the registration precondition, not an immutable central state after registration.',
            'This receipt binds prospective design and synthetic evidence only; its generator performs no registration or historical research phase.',
        ],
        actual_features_computed=0, actual_orders_computed=0, actual_returns_computed=0,
        producer_modules_imported=False, strategy_admitted=False, observer_promotion=False,
        strict_pit_vintage=False, strict_pit_membership=False, strict_pit_sector_classification=False,
        review_generator=dict(path=str(Path(__file__).resolve()), sha256=digest(__file__)),
    )
    require(implementation == {str(item): digest(item) for item in expected_paths}, 'Code changed during binding review')
    require(digest(RECEIPT) == EXPECTED_RECEIPT, 'Aggregate changed during binding review')
    require(source_pins == {str(ROOT/'work'/name): digest(ROOT/'work'/name) for name in source_names},
            'Design changed during review')
    require(digest(state_path) == EXPECTED_STATE, 'Central state changed during review')
    independent.prepare.independent_imports()
    with DESTINATION.open('x') as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    print(json.dumps(dict(passed=True, implementation_files=len(implementation), tests_run=124,
                         raw_source_count=64, path=str(DESTINATION), sha256=digest(DESTINATION))))


if __name__ == '__main__':
    main()
