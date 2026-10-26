"""Bind the independent source-only registration review to final test evidence.

Read only source, metadata and file hashes. Do not import the producer, execute
market features, decode numerical market arrays, or modify central state.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import run163_independent as independent


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / 'work/run163_registration_tests.json'
EXPECTED_RECEIPT = '1e6e3cf946013e571118a4ccb18a19d711adacfe3e43a7de692500a513216e6e'
DESTINATION = ROOT / 'work/run163_registration_review.json'
REVIEWED = {
    'work/run163_host.py': '6690ba93cfff04a68ee470ed14843b6a7715fac594aa32d5c95cabedbd26eb7b',
    'work/run163_kernel.py': '910e3fdecc0a4192722ebd038283bda41d546776756f0d07cc927f702b3dc4ca',
    'work/run163_runner.py': '669a061909251779a765cda39de76307fae3be15772c36a2f8ee30557c50deb4',
    'work/run163_validate_code.py': '93a8b4170eb1832fbc70f0dca9fc2a70ae9d2e284468a0b759fdf25473f9037d',
    'work/run163_proposal.json': '65786d4dcb30919009de3847d968df7dc9fc3ad4c3505774a1c34f6a3e70327f',
    'work/run163_source_probe.py': '98dc152bca7eadfdffaa285112cf64441b55e81a22f58b95fb8f70808e0ded91',
    'work/run163_design_review.json': 'c03876631f8a853d19f269baac8e807ae9391765bc0ccd6e2bacfd45cb3715e2',
    'work/run163_novelty_review.json': '23f14b9831c65140c6d4897659039b44ab4a6cbb8b4b6e8fafdec22ff67225a4',
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
    require(not (ROOT / 'outputs/monthly_run163/PLAN.json').exists(), 'Review must precede registration')
    independent.prepare.independent_imports()
    require(all(digest(name) == sha for name, sha in REVIEWED.items()), 'Reviewed source bytes changed')
    expected_paths = sorted(set(independent.source_files(ROOT) + [ROOT / 'work' / name for name in (
        'run163_host.py', 'run163_kernel.py', 'run163_runner.py',
        'test_run163_kernel.py', 'test_run163_runner.py', 'test_run163_host.py',
        'run163_validate_code.py')]))
    implementation = {str(item): digest(item) for item in expected_paths}
    require(len(implementation) == 20, 'Twenty exact validation paths required')
    receipt = read_pinned(RECEIPT, EXPECTED_RECEIPT)
    require(receipt['passed'] is True and receipt['synthetic_only'] is True and
            receipt['real_features_orders_or_returns_computed'] is False, 'Synthetic-only aggregate required')
    require(receipt['implementation_hashes'] == implementation, 'Exact complete validation_paths closure required')
    suites = {'kernel': 24, 'runner': 34, 'host': 23, 'independent': 45}
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
    require(receipt['tests_run'] == sum(suites.values()) == 126, '126 final tests required')
    source_names = ('run163_source_probe.py', 'run163_proposal.json',
                    'run163_design_review.json', 'run163_novelty_review.json')
    source_pins = {str(ROOT/'work'/name): digest(ROOT/'work'/name) for name in source_names}
    design = read_pinned('work/run163_design_review.json', REVIEWED['work/run163_design_review.json'])
    novelty = read_pinned('work/run163_novelty_review.json', REVIEWED['work/run163_novelty_review.json'])
    proposal = read_pinned('work/run163_proposal.json', REVIEWED['work/run163_proposal.json'])
    zero_activity(design); zero_activity(novelty); zero_activity(proposal)
    require(design['proposal_sha256'] == novelty['proposal_sha256'] == REVIEWED['work/run163_proposal.json'],
            'Current proposal must be bound by both reviews')
    require(design['input_hashes'] == design['source_hashes'], 'Source closure maps must agree')
    for name in ('run163_source_probe.py', 'run163_proposal.json', 'run163_novelty_review.json'):
        key = str(ROOT/'work'/name)
        require(design['input_hashes'].get(key) == source_pins[key], 'Current source review binding required')
    require(all(digest(name) == sha for name, sha in design['input_hashes'].items()), 'Source inputs changed')
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
    state = read_pinned('outputs/monthly_target70_research/CONTINUATION_STATUS.json',
                        novelty_pins[str(ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json')])
    require(len(state['completed']) == 162 and sum(row['trials'] for row in state['completed']) == 1001 and
            state['registered_queue'] == [], 'Expected unregistered162/1001 central state')
    report = dict(
        passed=True, schema='run163_registration_review_v1', round=163,
        created_utc=datetime.now(timezone.utc).isoformat(),
        host_sha256=implementation[str(ROOT/'work/run163_host.py')],
        registration_tests_sha256=EXPECTED_RECEIPT, implementation_hashes=implementation,
        tests_run=126, test_evidence_hashes=evidence, design_evidence_hashes=source_pins,
        source_input_hashes=design['input_hashes'], novelty_snapshot_hashes=novelty_pins,
        reviewer_scope='Independent bounded novelty, host lifecycle, fixed proposal, source binding and final synthetic evidence review; reviewer did not implement the producer or numerical oracle.',
        checks={
            'complete_current20_validation_paths_match_all_four_suites': True,
            'all126_synthetic_tests_pass_with_unchanged_code_and_bound_logs': True,
            'fixed_terminal_run_rank_formula_and_opposite_doses_match_proposal': True,
            'registration_precedes_actual_features_orders_and_outcomes': True,
            'source_probe_hashes_price_bytes_but_decodes_only_dates_and_symbols': True,
            'official_calendar_original_axes_and_novelty_review_bound': True,
            'novelty12_file_snapshot_verified_before_central_registration': True,
            'first88_once_selection_precedes_later34_diagnostics': True,
            'all5978_cells366_orders123_prefixes_required_by_prepare_gate': True,
            'all9_paths_required_by_independent_outcome_gate': True,
            'full21444_protected_manifest_required_before_completion': True,
            'prior_history_preserved_and_completed_counts163_1004': True,
            'new_best_requires_training_admission_and_verified_full_reference_gain': True,
            'completion_and_central_best_share_verified_expected_object': True,
            'write_once_artifacts_and_locked_atomic_state_commit': True,
        },
        unresolved_blocking_findings=[],
        limitations=[
            'This approves registration mechanics and the fixed reviewed design, not numerical coverage, profitability or achievement of70%.',
            'Bounded local novelty review found no exact duplicate; missing prior variance-ratio and ordinal-complexity source prevents exhaustive nonduplication certification.',
            'Hashing market input bytes is not numerical decoding; this reviewer did not compute actual features, orders or returns.',
            'Actual preparation and outcomes still require the registered independent verification; the21444-file archive audit is enforced later and was not repeated here.',
            'Historical price vintages and estimated historical pools remain conditional research inputs; later34 has already been used in prior development.',
            'The central best is a descriptive full-period reference among first88-admitted candidates; it neither changes the frozen first88 winner nor promotes observation accounts.',
            'The novelty snapshot includes mutable central-state bytes as pre-registration evidence only; those bytes are not a requirement that central state remain unchanged after registration.',
        ],
        actual_features_computed=0, actual_orders_computed=0, actual_returns_computed=0,
        producer_modules_imported=False, strategy_admitted=False, observer_promotion=False,
        review_generator=dict(path=str(Path(__file__).resolve()), sha256=digest(__file__)),
    )
    require(implementation == {str(item): digest(item) for item in expected_paths}, 'Code changed during binding review')
    require(digest(RECEIPT) == EXPECTED_RECEIPT, 'Aggregate changed during binding review')
    require(source_pins == {str(ROOT/'work'/name): digest(ROOT/'work'/name) for name in source_names},
            'Design changed during review')
    independent.prepare.independent_imports()
    with DESTINATION.open('x') as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    print(json.dumps(dict(passed=True, implementation_files=len(implementation), tests_run=126,
                         path=str(DESTINATION), sha256=digest(DESTINATION))))


if __name__ == '__main__':
    main()
