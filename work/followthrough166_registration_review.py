"""Bind the reviewed166 sources to final synthetic evidence, before registration.

No producer imports, market-array decoding, feature/return calculation or central
state changes. The sole output is a new write-once registration-review receipt.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import followthrough166_independent as independent


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / 'work/followthrough166_registration_tests.json'
DESTINATION = ROOT / 'work/followthrough166_registration_review.json'
EXPECTED_STATE = '3e1b6e9b467bc82e9467dc5dca9e67f361fbad688082c79c96e3c4e9b6e466df'
SUITES = {'kernel': 22, 'runner': 30, 'host': 24, 'independent': 50}
REVIEWED = {
    'work/followthrough166_host.py': 'fa77d4115ae883c70d5110e8ef05f61468122c8eb99fec03e34c347d993563dc',
    'work/followthrough166_kernel.py': '02a9fe83edf9e488a50443a88c7c0215a4e2ef94fcb41373eb2bc60b6dcb2da9',
    'work/followthrough166_runner.py': '0fc5a70a53b3ffc004b17d51844cd9a8e1e134e61fa4057fadebf31d1b943404',
    'work/followthrough166_validate_code.py': 'a791c7f668941371c6ef4d321524ecb60b8dd0cc67fff0a2aeeecd3a5b19e3b2',
    'work/followthrough166_independent.py': 'c0b4acf82109b3168ce2bc6654b0c5304ed41e86b9c24e0ab45e8f8cc13457c7',
    'work/followthrough166_independent_prepare.py': '931e487f0d0b8fe51c6f59cf1860db3e86facf492a1d62f553ff56cdebaa88bd',
    'work/followthrough166_independent_outcome.py': '23d49c7a708839977322c8cd25a36092ddff0aa4b97acac52c13e0dde73b0c25',
    'work/test_followthrough166_kernel.py': '752993c42d6133043172f12504ad216e880b015650267dd854dba8b90bef7cbd',
    'work/test_followthrough166_runner.py': 'd26bbcdc90a936e6c9f1abb5280353a0e6b87e0ff85d1140053f122148a47694',
    'work/test_followthrough166_host.py': 'caac650d24e1af1b5499d2af3191524053fdf9f90a6d19fd5d4b3ae08013e257',
    'work/test_followthrough166_independent.py': '5643f07bdfcf269e4cca883156f74087a7ecb313d35cb83844831e23443b7492',
    'work/followthrough166_proposal.json': '5c183d80b751e70b6ed150f394c22478acd0e47c4f9bca7748f31f62613fd587',
    'work/followthrough166_source_probe.py': '4ec5f593bd5041c77b9730dd4bd8f88224cd278de655aea2b8871c841720a1c9',
    'work/followthrough166_design_review.json': 'cd9b3f1b435e03988ba9350c370a29ddb54b9c252e5157824cb1fb4dd4fdc5f4',
    'work/followthrough166_novelty_review.json': 'efd309827b7f2eca533c7ad507e668ddb6b9aa94a3a5a6e3e8b7fe84b67d6fc4',
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
    def unique(items):
        result = {}
        for key, item in items:
            require(key not in result, 'Duplicate JSON key')
            result[key] = item
        return result
    def nonfinite(value):
        raise ValueError('Nonfinite JSON token: ' + value)
    return json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite)


def zero_activity(item):
    require(all(type(item.get(key)) is int and item[key] == 0 for key in (
        'actual_features_computed', 'actual_orders_computed', 'actual_returns_computed')),
        'Source-only zero-activity counters required')


def main(expected_receipt):
    require(re.fullmatch('[0-9a-f]{64}', expected_receipt) is not None, 'Explicit final aggregate SHA256 required')
    require(not DESTINATION.exists(), 'Write-once registration review already exists')
    for name in ('PLAN.json', 'PREPARE_STARTED.json', 'DATA.npz', 'CACHE.json', 'SELECT_STARTED.json',
                 'FROZEN.json', 'DIAGNOSE_STARTED.json', 'RESULTS.json'):
        require(not (ROOT / 'outputs/monthly_followthrough166' / name).exists(),
                'Review must precede registration and real calculations: ' + name)
    independent.prepare.independent_imports()
    require(all(digest(name) == sha for name, sha in REVIEWED.items()), 'Reviewed source bytes changed')
    expected_paths = sorted(set(independent.source_files(ROOT) + [ROOT / 'work' / name for name in (
        'followthrough166_host.py', 'followthrough166_kernel.py', 'followthrough166_runner.py',
        'test_followthrough166_kernel.py', 'test_followthrough166_runner.py', 'test_followthrough166_host.py',
        'followthrough166_validate_code.py')]))
    implementation = {str(item): digest(item) for item in expected_paths}
    require(len(implementation) == 20, 'Twenty exact validation paths required')
    receipt = read_pinned(RECEIPT, expected_receipt)
    require(receipt['passed'] is True and receipt['synthetic_only'] is True and
            receipt['real_features_orders_or_returns_computed'] is False, 'Synthetic-only aggregate required')
    require(receipt['implementation_hashes'] == implementation, 'Exact complete validation_paths closure required')
    require([row['suite'] for row in receipt['suites']] == list(SUITES), 'Four exact suites required')
    evidence = {str(RECEIPT): expected_receipt}
    for row in receipt['suites']:
        require(type(row['tests_run']) is int and row['tests_run'] == SUITES[row['suite']], 'Final suite count changed')
        item = read_pinned(row['receipt']['path'], row['receipt']['sha256'])
        require(item['suite'] == row['suite'] and item['tests_run'] == row['tests_run'], 'Suite identity/count mismatch')
        require(item['passed'] is True and item['synthetic_only'] is True and item['code_unchanged'] is True,
                'Suite must pass with unchanged implementation')
        require(item['failures'] == item['errors'] == item['skipped'] == 0, 'No failed or skipped tests')
        require(item['implementation_hashes'] == implementation, 'Suite used different code')
        require(item['log'] == row['log'] and digest(row['log']['path']) == row['log']['sha256'], 'Suite log binding')
        evidence[str(path(row['receipt']['path']))] = row['receipt']['sha256']
        evidence[str(path(row['log']['path']))] = row['log']['sha256']
    require(type(receipt['tests_run']) is int and receipt['tests_run'] == sum(SUITES.values()) == 126,
            '126 final synthetic tests required')

    source_names = ('followthrough166_source_probe.py', 'followthrough166_proposal.json',
                    'followthrough166_design_review.json', 'followthrough166_novelty_review.json')
    source_pins = {str(ROOT/'work'/name): digest(ROOT/'work'/name) for name in source_names}
    design = read_pinned('work/followthrough166_design_review.json', REVIEWED['work/followthrough166_design_review.json'])
    novelty = read_pinned('work/followthrough166_novelty_review.json', REVIEWED['work/followthrough166_novelty_review.json'])
    proposal = read_pinned('work/followthrough166_proposal.json', REVIEWED['work/followthrough166_proposal.json'])
    zero_activity(design); zero_activity(novelty); zero_activity(proposal)
    require(proposal['round'] == 166 and proposal['arms'] ==
            ['incumbent_local67', 'volume_followthrough', 'persistence_control'], 'Exact three-arm prospective proposal')
    require(design['proposal_sha256'] == REVIEWED['work/followthrough166_proposal.json'],
            'Current final proposal must be bound by design review')
    require(design['input_hashes'] == design['source_hashes'], 'Source closure maps must agree')
    for name in ('followthrough166_source_probe.py', 'followthrough166_proposal.json', 'followthrough166_novelty_review.json'):
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
    require(novelty['passed'] is True and
            novelty['status'] == 'BOUNDED_CONDITIONAL_REFINEMENT_NO_EXACT_DUPLICATE_FOUND' and
            novelty['registration_authorized_by_review'] is False and
            novelty['new_information_family_claimed'] is False and novelty['exhaustive_novelty_certified'] is False,
            'Bounded conditional-refinement scope required')
    require(novelty['candidate']['arms'] == proposal['arms'], 'Novelty/proposal arm names differ')
    novelty_pins = {str(path(name)): sha for name, sha in novelty['checked_files_sha256'].items()}
    require(len(novelty_pins) == 16 and all(digest(name) == sha for name, sha in novelty_pins.items()),
            'Novelty source snapshot changed')
    state_path = 'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
    state = read_pinned(state_path, EXPECTED_STATE)
    require(len(state['completed']) == 165 and sum(row['trials'] for row in state['completed']) == 1010 and
            state['registered_queue'] == [] and state['completed'][-1]['directory'] == 'monthly_recovery165',
            'Expected unregistered165/1010 central state')

    report = dict(
        passed=True, schema='followthrough166_registration_review_v1', round=166,
        created_utc=datetime.now(timezone.utc).isoformat(),
        host_sha256=implementation[str(ROOT/'work/followthrough166_host.py')],
        registration_tests_sha256=expected_receipt, implementation_hashes=implementation,
        tests_run=126, suite_counts=SUITES, test_evidence_hashes=evidence, design_evidence_hashes=source_pins,
        source_input_hashes=design['input_hashes'], raw_source_count=64, novelty_snapshot_hashes=novelty_pins,
        preregistration_state=dict(path=str(ROOT/state_path), sha256=EXPECTED_STATE, completed=165, trials=1010,
                                  registered_queue_empty=True),
        reviewer_scope='Source-only host lifecycle, fixed proposal, source binding and final synthetic evidence review; reviewer authored the bounded novelty audit but did not implement the producer or numerical oracle.',
        checks={
            'complete_current20_validation_paths_match_all_four_suites': True,
            'all126_synthetic_tests_pass_with_unchanged_code_and_bound_logs': True,
            'fixed210_pairs_212_adjusted_231_volume_and_prior21_baseline': True,
            'volume_only_interaction_residualized_on_current_return_and_activity': True,
            'matched_persistence_control_shared_support_positive04dose_max02delta': True,
            'all49_names_including_etfs_ranked_before_unchanged_eligibility': True,
            'registration_precedes_actual_features_orders_and_outcomes': True,
            'source_probe_hashes_price_bytes_but_decodes_only_dates_and_symbols': True,
            'all64_raw_sources_official_calendar_original_axes_and_novelty_bound': True,
            'conditional_refinement_and_missing_originals_are_explicit': True,
            'first88_once_selection_precedes_later34_diagnostics': True,
            'candidate_must_pass_incumbent_and_control_guards_both_execution_paths': True,
            'persistence_control_cannot_be_selected_or_promoted': True,
            'all5978_cells366_orders123_prefixes_required_by_prepare_gate': True,
            'original27_arrays_plus2_deltas_once_then_original_episode': True,
            'all9_paths_required_by_independent_outcome_gate': True,
            'full21444_protected_manifest_required_before_completion': True,
            'prior165_1010_history_preserved_and_completed_counts166_1013': True,
            'new_best_requires_training_admission_and_verified_full_reference_gain': True,
            'completion_and_central_best_share_verified_expected_object': True,
            'write_once_artifacts_and_locked_atomic_state_commit': True,
        },
        unresolved_blocking_findings=[],
        resolved_preregistration_findings=[dict(
            issue='An initially near-collinear synthetic interaction-degeneracy fixture amplified log/exp roundoff and failed to isolate the intended residual-degeneracy case.',
            resolution='The fixture now uses returns in [-0.2,0.2] and activity=1/(return-1), preserving r*a=1+a while explicitly checking that the two controls are adequately separated. Only the synthetic fixture changed; production formula and1e-12 numerical cutoffs were not relaxed.',
            evidence='work/test_followthrough166_kernel.py:test_interaction_degenerate_after_independent_controls; final source and successful isolated suite are hash-bound here.',
            actual_data_computed_during_repair=False)],
        limitations=[
            'This approves the reviewed registration mechanics and fixed design, not numerical coverage, profitability or achievement of70percent.',
            '166 is a conditional refinement of a previously tested volume/next-return family. Missing original lag/direction/liquidity code prevents exhaustive nonduplication certification.',
            'Stored share volume is not verified float turnover or order flow. Constant unit scaling cancels; time-varying splits, vendor conventions and source splices may affect activity.',
            'Hashing market input bytes is not numerical decoding; this reviewer did not compute actual features, orders or returns.',
            'Actual preparation and outcomes still require registered independent verification; the21444-file archive audit is enforced later and was not repeated here.',
            'Historical price vintages and estimated historical pools remain conditional research inputs; first88 and later34 have already been used in prior development.',
            'The central best is a descriptive full-period reference among first88-admitted candidates; it neither changes the frozen first88 winner nor promotes observation accounts.',
            'Preregistration central-state bytes are evidence for the registration precondition, not a requirement that central state remain unchanged after registration.',
            'This receipt binds the prospective design and synthetic evidence only; no registration or new research phase is executed by the generator.',
        ],
        actual_features_computed=0, actual_orders_computed=0, actual_returns_computed=0,
        producer_modules_imported=False, strategy_admitted=False, observer_promotion=False,
        review_generator=dict(path=str(Path(__file__).resolve()), sha256=digest(__file__)),
    )
    require(implementation == {str(item): digest(item) for item in expected_paths}, 'Code changed during binding review')
    require(digest(RECEIPT) == expected_receipt, 'Aggregate changed during binding review')
    require(source_pins == {str(ROOT/'work'/name): digest(ROOT/'work'/name) for name in source_names},
            'Design changed during review')
    require(digest(state_path) == EXPECTED_STATE, 'Central state changed during review')
    independent.prepare.independent_imports()
    with DESTINATION.open('x') as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    print(json.dumps(dict(passed=True, implementation_files=len(implementation), tests_run=126,
                         raw_source_count=64, path=str(DESTINATION), sha256=digest(DESTINATION))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-tests-sha', required=True)
    args = parser.parse_args()
    main(args.expected_tests_sha)
