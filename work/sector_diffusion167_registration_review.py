"""Bind the reviewed167 sources to final synthetic evidence, before registration.

No producer imports, market-array decoding, feature/return calculation or central
state changes. The sole output is a new write-once registration-review receipt.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import sector_diffusion167_independent as independent


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / 'work/sector_diffusion167_registration_tests.json'
DESTINATION = ROOT / 'work/sector_diffusion167_registration_review.json'
EXPECTED_STATE = '43d4eef535edb3b0e9c36bf1a28506f8e044fdf4a5dba29dabf1d8d6d1185d67'
SUITES = {'kernel': 23, 'runner': 33, 'host': 24, 'independent': 52}
REVIEWED = {'work/sector_diffusion167_host.py': '0dbc2eb0efad0f0845b39cdaebc662474e78edadf4727ea37f9727979c2da6a8', 'work/sector_diffusion167_kernel.py': '419c8929d305b14fff67baa25006fefdf8fb13645070d61665c0a43ff266c1b5', 'work/sector_diffusion167_runner.py': '7ba1bbd09777419c36dd6dcf5bf93266dd98d4c969912ea966ca5bc45ecac481', 'work/sector_diffusion167_validate_code.py': '316b20f151152819b793db039dff94520573abb1f5320524069bd6eb0e7499f1', 'work/sector_diffusion167_independent.py': '062ffbc9dc253f2238aa2aead9c4f00aecff2ee29b9bd751aa65a5945c07bc50', 'work/sector_diffusion167_independent_prepare.py': '7418ca237ae22dbd8cf7ecb10fecc6307795ef08b749ac32c0db369e9e7cdeee', 'work/sector_diffusion167_independent_outcome.py': 'dc39e1216737464e62f0592437ce42e53ff2c69ee74403735113165d2870e8e6', 'work/test_sector_diffusion167_kernel.py': 'd65b53ae54bb45e56634ecc233e2903418b70455e1d4a69a9eccbda0232744c0', 'work/test_sector_diffusion167_runner.py': 'f3dbf327aee9b03cf18f8271bfd390e11eb4d2aad289e7ca6c8f5c19c2d1d8ce', 'work/test_sector_diffusion167_host.py': 'beb71ff9c178411763bd2f33ca6335f07f1af62fa548d79c15bdba1bcd0ca9a4', 'work/test_sector_diffusion167_independent.py': '8ceca5337e672956c74efbfd053e828f91ed33d50d1b567fcc9ed4706cb4d96e', 'work/sector_diffusion167_proposal.json': '67f0998242b9c6231307c5dd850b8a69ea56cd974be05ad61311bffd75cd557f', 'work/sector_diffusion167_source_probe.py': '9a679454acde726ca9491eba44675b477592cc0d1ba26e895aaa3d1cdf610456', 'work/sector_diffusion167_design_review.json': '82e5b0005139209589b7c33202817455ebf5e33691cfda8b24ab2578cb20452d', 'work/sector_diffusion167_novelty_review.json': '35ba52bf2dd3d634b108cd248dcfc0cc8d1038b54585f52cdfa0f5ad271f4835', 'work/sector_diffusion167_source_review.json': '42d6f636142b60267b1d651473a0c0f95894fc8febf960b08e7f28dfeb90ee4a'}


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
        require(not (ROOT / 'outputs/monthly_sector_diffusion167' / name).exists(),
                'Review must precede registration and real calculations: ' + name)
    independent.prepare.independent_imports()
    require(all(digest(name) == sha for name, sha in REVIEWED.items()), 'Reviewed source bytes changed')
    expected_paths = sorted(set(independent.source_files(ROOT) + [ROOT / 'work' / name for name in (
        'sector_diffusion167_host.py', 'sector_diffusion167_kernel.py', 'sector_diffusion167_runner.py',
        'test_sector_diffusion167_kernel.py', 'test_sector_diffusion167_runner.py', 'test_sector_diffusion167_host.py',
        'sector_diffusion167_validate_code.py')]))
    implementation = {str(item): digest(item) for item in expected_paths}
    require(len(implementation) == 21, 'Twenty-one exact validation paths required')
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
    require(type(receipt['tests_run']) is int and receipt['tests_run'] == sum(SUITES.values()) == 132,
            '132 final synthetic tests required')

    source_names = ('sector_diffusion167_source_probe.py', 'sector_diffusion167_proposal.json',
                    'sector_diffusion167_design_review.json', 'sector_diffusion167_novelty_review.json', 'sector_diffusion167_source_review.json')
    source_pins = {str(ROOT/'work'/name): digest(ROOT/'work'/name) for name in source_names}
    design = read_pinned('work/sector_diffusion167_design_review.json', REVIEWED['work/sector_diffusion167_design_review.json'])
    novelty = read_pinned('work/sector_diffusion167_novelty_review.json', REVIEWED['work/sector_diffusion167_novelty_review.json'])
    proposal = read_pinned('work/sector_diffusion167_proposal.json', REVIEWED['work/sector_diffusion167_proposal.json'])
    zero_activity(design); zero_activity(novelty); zero_activity(proposal)
    require(proposal['round'] == 167 and proposal['arms'] ==
            ['incumbent_local67', 'sector_volume_diffusion', 'sector_additive_control'], 'Exact three-arm prospective proposal')
    require(design['proposal_sha256'] == REVIEWED['work/sector_diffusion167_proposal.json'],
            'Current final proposal must be bound by design review')
    require(design['input_hashes'] == design['source_hashes'], 'Source closure maps must agree')
    for name in ('sector_diffusion167_source_probe.py', 'sector_diffusion167_proposal.json', 'sector_diffusion167_novelty_review.json'):
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
            summary['historical_names_absent_from_price_panel'] == [] and summary['market_proxy_present'] is True and summary['missing_sector_symbols'] == [],
            'Metadata identity review incomplete')
    require(novelty['passed'] is True and
            novelty['status'] == 'BOUNDED_CONDITIONAL_EXTENSION_DUPLICATION_UNRESOLVED' and
            novelty['registration_authorized_by_review'] is False and
            novelty['new_information_family_claimed'] is False and novelty['exhaustive_novelty_certified'] is False,
            'Bounded conditional-refinement scope required')
    require(novelty['candidate']['arms'] == proposal['arms'], 'Novelty/proposal arm names differ')
    novelty_pins = {str(path(name)): sha for name, sha in novelty['checked_files_sha256'].items()}
    require(len(novelty_pins) == 23 and all(digest(name) == sha for name, sha in novelty_pins.items()),
            'Novelty source snapshot changed')
    state_path = 'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
    state = read_pinned(state_path, EXPECTED_STATE)
    require(len(state['completed']) == 166 and sum(row['trials'] for row in state['completed']) == 1013 and
            state['registered_queue'] == [] and state['completed'][-1]['directory'] == 'monthly_followthrough166',
            'Expected unregistered166/1013 central state')

    report = dict(
        passed=True, schema='sector_diffusion167_registration_review_v1', round=167,
        created_utc=datetime.now(timezone.utc).isoformat(),
        host_sha256=implementation[str(ROOT/'work/sector_diffusion167_host.py')],
        registration_tests_sha256=expected_receipt, implementation_hashes=implementation,
        tests_run=132, suite_counts=SUITES, test_evidence_hashes=evidence, design_evidence_hashes=source_pins,
        source_input_hashes=design['input_hashes'], raw_source_count=64, novelty_snapshot_hashes=novelty_pins,
        preregistration_state=dict(path=str(ROOT/state_path), sha256=EXPECTED_STATE, completed=166, trials=1013,
                                  registered_queue_empty=True),
        reviewer_scope='Source-only host lifecycle, fixed proposal, source binding and final synthetic evidence review; reviewer authored the bounded novelty audit but did not implement the producer or numerical oracle.',
        checks={
            'complete_current21_validation_paths_match_all_four_suites': True,
            'all132_synthetic_tests_pass_with_unchanged_code_and_bound_logs': True,
            'fixed210_pairs_212_stock_and_sector_adjusted_231_sector_volume_prior21_baseline': True,
            'candidate_sector_return_volume_interaction_four_controls': True,
            'matched_additive_sector_lag_three_controls_not_price_only': True,
            'relative_svd_rank_threshold_both_arms_and_common_residual_mask': True,
            'common_valid_direct_stocks_ranked_before_unchanged_eligibility': True,
            'etfs_unknown_mapping_and_initial_partial_are_neutral': True,
            'signal_date_original_mapping_no_alternate_sector_or_spy_fallback': True,
            'fixed_positive04dose_max02delta_once_then_original_episode': True,
            'registration_precedes_actual_features_orders_and_outcomes': True,
            'source_probe_hashes_price_bytes_but_decodes_only_dates_and_symbols': True,
            'all64_raw_sources_official_calendar_original_axes_and_novelty_bound': True,
            'conditional_extension_and_missing_64_95_109_originals_explicit': True,
            'first88_once_selection_precedes_later34_diagnostics': True,
            'candidate_passes_incumbent_and_matched_control_guards_both_execution_paths': True,
            'additive_control_cannot_be_selected_or_promoted': True,
            'all5978_cells366_orders123_prefixes_required_by_prepare_gate': True,
            'original27_arrays_plus2_deltas_preserved': True,
            'all9_paths_required_by_independent_outcome_gate': True,
            'full21444_protected_manifest_required_before_completion': True,
            'prior166_1013_history_preserved_completed_counts167_1016': True,
            'new_best_requires_training_admission_and_verified_full_reference_gain': True,
            'completion_and_central_best_share_verified_expected_object': True,
            'write_once_artifacts_and_locked_atomic_state_commit': True,
        },
        unresolved_blocking_findings=[],
        resolved_preregistration_findings=[dict(
            issue='A copied independent scoring fixture retained nonzero ETF corrections although this sector-only experiment requires ETF neutrality.',
            resolution='Independent score_oracle now rejects any nonzero ETF correction; the fixture retains nonzero corrections only for direct stocks and explicitly tests even1e-20 ETF tampering is rejected. Producer validation independently enforces the same boundary.',
            evidence='work/test_sector_diffusion167_independent.py:test_ineligible_stock_keeps_delta_but_etf_stays_neutral; final source and isolated successful suite are hash-bound here.',
            actual_data_computed_during_repair=False), dict(
            issue='The first registration-review invocation retained the prior round twenty-file count and stopped before writing its receipt.',
            resolution='The current independently enumerated closure contains twenty-one files, including the frozen strict_study.py mapping source. The reviewer count now matches that complete set; equality against all four final suite bindings remains required.',
            evidence='Final implementation_hashes and independent.source_files both include outputs/monthly_strict_relative/strict_study.py; no producer, oracle, test or source bytes changed.',
            actual_data_computed_during_repair=False)],
        limitations=[
            'This approves reviewed registration mechanics and a fixed design, not numerical coverage, profitability or70percent attainment.',
            '167 is a conditional extension of prior sector-diffusion and volume-conditioned continuation families. Missing64/95/109 originals prevent exhaustive nonduplication certification.',
            'SectorETF volume is not constituent turnover, fund flow or informed orderflow. Candidate stocks may be ETF constituents; no leave-one-out historical weights exist.',
            'The next-sector-return control is historical by signal t but unavailable at predictor date s; this is a descriptive old-window association, not structural causality.',
            'Constant volume scaling cancels but time-varying source/split conventions or revisions may affect the statistic.',
            'Hashing market bytes is not decoding numerical arrays; this review computes no real features, orders or returns.',
            'Registered preparation and outcomes still need independent verification; the21444-file protection audit is required later and was not repeated here.',
            'Historical vintages, estimated Top32, assumed theme17 and signal-date sector proxies remain conditional; repeatedly reused first88/later34 history is not pristine OOS.',
            'Central best is a descriptive full-period reference among training-admitted candidates; no frozen observation account is changed or promoted.',
            'The fixed positive direction,21-session baseline and0.04dose remain researcher choices; no post-result retuning is permitted in this round.',
            'The preregistration central-state digest is a registration precondition, not a requirement that central state remain unchanged afterward.',
            'This receipt binds prospective design and synthetic evidence only; no registration or other research phase is executed by this generator.',
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
    print(json.dumps(dict(passed=True, implementation_files=len(implementation), tests_run=132,
                         raw_source_count=64, path=str(DESTINATION), sha256=digest(DESTINATION))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-tests-sha', required=True)
    args = parser.parse_args()
    main(args.expected_tests_sha)
