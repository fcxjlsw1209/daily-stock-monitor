"""Bind source-only activity164 registration review to final synthetic evidence.

No producer imports, market-array decoding, feature/return calculations, central
state writes or registration. Only the write-once review receipt is published.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import activity164_independent as independent


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / 'work/activity164_registration_tests.json'
EXPECTED_RECEIPT = '6151f551f6016f1befe8ed279f84cd98820d38220f7ec2ff067e7c8beef6ca60'
EXPECTED_STATE = '60e0b5e979a5464163b5f81f78ebba027473b1c589c4153996326dfd10f1675c'
DESTINATION = ROOT / 'work/activity164_registration_review.json'
REVIEWED = {
    'work/activity164_host.py': 'abdc96249de27003f451ee0b06f54d0b05fee915a686d0c7ab8ee26206a21d83',
    'work/activity164_kernel.py': '99cab04e5299481e98e48d72dcbf83e4e3b53f7e3ec869672dd1f08c674616e0',
    'work/activity164_runner.py': 'ec424db9f91cc1cf4d1a8d59d8cc13ad56a1c15b25523032d7272970d2672af5',
    'work/activity164_validate_code.py': '21910d6ea3418d06a5cb28fb290aec27f414ecbd8731a391835f8e0bf07cd4cc',
    'work/activity164_proposal.json': '640eb3d1d51a56fe0774a9b69e3bc1fb20d280e679ccdb87ad0f5f6c18e2dd94',
    'work/activity164_source_probe.py': '3657e07b6716de7301e1ae9324b08ec6bfb3541987207da7df7d275745153f08',
    'work/activity164_design_review.json': '03ad70cb65339a3e5a19e4991483e98538130021fb49b5ffb017815f9ccd9e72',
    'work/activity164_novelty_review.json': '4883d7643edb33c2c829a3ee3d8b7c7bb92b27034ff60bac0cad5ffc0cee131f',
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
    require(not (ROOT / 'outputs/monthly_activity164/PLAN.json').exists(), 'Review must precede registration')
    independent.prepare.independent_imports()
    require(all(digest(name) == sha for name, sha in REVIEWED.items()), 'Reviewed source bytes changed')
    expected_paths = sorted(set(independent.source_files(ROOT) + [ROOT / 'work' / name for name in (
        'activity164_host.py', 'activity164_kernel.py', 'activity164_runner.py',
        'test_activity164_kernel.py', 'test_activity164_runner.py', 'test_activity164_host.py',
        'activity164_validate_code.py')]))
    implementation = {str(item): digest(item) for item in expected_paths}
    require(len(implementation) == 20, 'Twenty exact validation paths required')
    receipt = read_pinned(RECEIPT, EXPECTED_RECEIPT)
    require(receipt['passed'] is True and receipt['synthetic_only'] is True and
            receipt['real_features_orders_or_returns_computed'] is False, 'Synthetic-only aggregate required')
    require(receipt['implementation_hashes'] == implementation, 'Exact complete validation_paths closure required')
    suites = {'kernel': 18, 'runner': 30, 'host': 24, 'independent': 47}
    require([row['suite'] for row in receipt['suites']] == list(suites), 'Four exact suites required')
    evidence = {str(RECEIPT): EXPECTED_RECEIPT}
    for row in receipt['suites']:
        require(row['tests_run'] == suites[row['suite']], 'Final suite count changed')
        item = read_pinned(row['receipt']['path'], row['receipt']['sha256'])
        require(item['suite'] == row['suite'] and item['tests_run'] == row['tests_run'], 'Suite identity/count mismatch')
        require(item['passed'] is True and item['synthetic_only'] is True and item['code_unchanged'] is True,
                'Suite must pass with unchanged implementation')
        require(item['failures'] == item['errors'] == item['skipped'] == 0, 'No failed or skipped tests')
        require(item['implementation_hashes'] == implementation, 'Suite used different code')
        require(item['log'] == row['log'] and digest(row['log']['path']) == row['log']['sha256'], 'Suite log binding')
        evidence[str(path(row['receipt']['path']))] = row['receipt']['sha256']
        evidence[str(path(row['log']['path']))] = row['log']['sha256']
    require(receipt['tests_run'] == sum(suites.values()) == 119, '119 final tests required')

    source_names = ('activity164_source_probe.py', 'activity164_proposal.json',
                    'activity164_design_review.json', 'activity164_novelty_review.json')
    source_pins = {str(ROOT/'work'/name): digest(ROOT/'work'/name) for name in source_names}
    design = read_pinned('work/activity164_design_review.json', REVIEWED['work/activity164_design_review.json'])
    novelty = read_pinned('work/activity164_novelty_review.json', REVIEWED['work/activity164_novelty_review.json'])
    proposal = read_pinned('work/activity164_proposal.json', REVIEWED['work/activity164_proposal.json'])
    zero_activity(design); zero_activity(novelty); zero_activity(proposal)
    require(proposal['round'] == 164 and proposal['arms'] ==
            ['incumbent_local67', 'activity_risk', 'variance_control'], 'Exact three-arm prospective proposal')
    require(design['proposal_sha256'] == REVIEWED['work/activity164_proposal.json'],
            'Current final proposal must be bound by design review')
    require(design['input_hashes'] == design['source_hashes'], 'Source closure maps must agree')
    for name in ('activity164_source_probe.py', 'activity164_proposal.json', 'activity164_novelty_review.json'):
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
            design['strategy_admitted'] is False, 'Conditional source review required')
    summary = design['summary']
    require(summary['event_months'] == 122 and summary['training_months'] == 180 and
            summary['all_overlap_sets_equal'] is True and summary['noninitial_dates_verified_monthend'] is True and
            summary['historical_names_absent_from_price_panel'] == [] and summary['market_proxy_present'] is True,
            'Metadata identity review incomplete')
    require(novelty['passed'] is True and novelty['status'] == 'BOUNDED_REVIEW_NO_EXACT_DUPLICATE_FOUND' and
            novelty['registration_authorized_by_review'] is False, 'Bounded novelty scope required')
    novelty_pins = {str(path(name)): sha for name, sha in novelty['checked_files_sha256'].items()}
    require(all(digest(name) == sha for name, sha in novelty_pins.items()), 'Novelty source snapshot changed')
    state_path = 'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
    state = read_pinned(state_path, EXPECTED_STATE)
    require(len(state['completed']) == 163 and sum(row['trials'] for row in state['completed']) == 1004 and
            state['registered_queue'] == [] and state['completed'][-1]['directory'] == 'monthly_run163',
            'Expected unregistered163/1004 central state')

    report = dict(
        passed=True, schema='activity164_registration_review_v1', round=164,
        created_utc=datetime.now(timezone.utc).isoformat(),
        host_sha256=implementation[str(ROOT/'work/activity164_host.py')],
        registration_tests_sha256=EXPECTED_RECEIPT, implementation_hashes=implementation,
        tests_run=119, suite_counts=suites, test_evidence_hashes=evidence, design_evidence_hashes=source_pins,
        source_input_hashes=design['input_hashes'], raw_source_count=64, novelty_snapshot_hashes=novelty_pins,
        preregistration_state=dict(path=str(ROOT/state_path), sha256=EXPECTED_STATE, completed=163, trials=1004,
                                  registered_queue_empty=True),
        reviewer_scope='Independent source-only host lifecycle, fixed proposal, source binding and final synthetic evidence review; reviewer authored the bounded novelty audit but did not implement the producer or numerical oracle.',
        checks={
            'complete_current20_validation_paths_match_all_four_suites': True,
            'all119_synthetic_tests_pass_with_unchanged_code_and_bound_logs': True,
            'fixed210_pair_212_adjusted_211_raw_partial_correlation_matches_proposal': True,
            'matched_variance_control_shared_support_negative04dose_max02delta': True,
            'registration_precedes_actual_features_orders_and_outcomes': True,
            'source_probe_hashes_price_bytes_but_decodes_only_dates_and_symbols': True,
            'all64_raw_sources_official_calendar_original_axes_and_novelty_bound': True,
            'novelty_source_snapshot_verified_before_central_registration': True,
            'first88_once_selection_precedes_later34_diagnostics': True,
            'candidate_must_pass_incumbent_and_control_guards_both_execution_paths': True,
            'variance_control_cannot_be_selected_or_promoted': True,
            'all5978_cells366_orders123_prefixes_required_by_prepare_gate': True,
            'original27_arrays_plus2_deltas_once_then_original_episode': True,
            'all9_paths_required_by_independent_outcome_gate': True,
            'full21444_protected_manifest_required_before_completion': True,
            'prior163_1004_history_preserved_and_completed_counts164_1007': True,
            'new_best_requires_training_admission_and_verified_full_reference_gain': True,
            'completion_and_central_best_share_verified_expected_object': True,
            'write_once_artifacts_and_locked_atomic_state_commit': True,
            'proposal_roundoff_failure_wording_clarified_before_final_hash_binding': True,
        },
        unresolved_blocking_findings=[],
        limitations=[
            'This approves registration mechanics and the fixed reviewed design, not numerical coverage, profitability or achievement of70percent.',
            'Bounded local novelty review found no exact duplicate; missing historical volume/GARCH source prevents exhaustive nonduplication certification.',
            'Hashing market input bytes is not numerical decoding; this reviewer did not compute actual features, orders or returns.',
            'Actual preparation and outcomes still require registered independent verification; the21444-file archive audit is enforced later and was not repeated here.',
            'Historical price vintages and estimated historical pools remain conditional research inputs; later34 has already been used in prior development.',
            'The central best is a descriptive full-period reference among first88-admitted activity candidates; it neither changes the frozen first88 winner nor promotes observation accounts.',
            'Preregistration central-state bytes are evidence for the registration precondition, not a requirement that central state remain unchanged after registration.',
            'This receipt binds the prospective design and synthetic evidence only; no registration or new research phase is executed by the generator.',
        ],
        actual_features_computed=0, actual_orders_computed=0, actual_returns_computed=0,
        producer_modules_imported=False, strategy_admitted=False, observer_promotion=False,
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
    print(json.dumps(dict(passed=True, implementation_files=len(implementation), tests_run=119,
                         raw_source_count=64, path=str(DESTINATION), sha256=digest(DESTINATION))))


if __name__ == '__main__':
    main()
