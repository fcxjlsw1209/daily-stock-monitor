"""Publish the independent static WC194 review; no production module execution.

The findings below record a human-readable source review of the pinned six
files and their frozen adapters. This script checks those reviewed bytes and
the already-completed synthetic receipt; it does not rerun any strategy phase.
"""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SELF = 'work/wc194_independent_code_review.py'
BASE = 'outputs/monthly_working_capital194_source/'
TESTS = 'work/monthly_trade_wc194_TESTS.json'
CODE = {
    'work/monthly_trade_wc194_host.py': 'ae144c74d2e3423317df30d6d40c16e41c973d8dc5852d382f49a58bf37c0692',
    'work/test_monthly_trade_wc194_host.py': '8281b0425f293bca8615b394ccfe3ce4835b4dd8eb5f049c82c7a4b37e97e0e2',
    'work/monthly_trade_wc194_factor.py': 'cb9f0998e99a6e0a8f3d8901c3fbdb154fc8d10311f43cd8f48ad2d4a6030f34',
    'work/test_monthly_trade_wc194_factor.py': '44fb069922bbace451ec9161c998afa399a3281da0459d7f79b8e47b154c4489',
    'work/monthly_trade_wc194_independent.py': '6c583cd00e4a597d9eb50a9d76e5e2a09bf6e9bc46e3893e8980c1567e75b895',
    'work/test_monthly_trade_wc194_independent.py': '71e949e3196d4ef0787711806f190146defff267f22220c48f63ed203de2a9da',
}
INPUTS = {
    TESTS: 'e3ab462622aa3f5c3b14482a6f769585894d4b3391b72a294f71cea36602e990',
    'work/working_capital194_experiment_concept.json': '380aae3cf18288dd668cefb3a9be94d0e9c90b76e8d085f6d59ff0ddafed9a83',
    'work/working_capital194_source_scope.json': 'f2dda7ce4d3d61084bfbbb6e3fabca2388250575588054a41bd33890dc4948be',
    BASE+'admitted/DATASET.json': '0eca5d80051f4081db8d7f0bf0405efa895365090feb3c989c21786c44a17133',
    BASE+'admitted/POLICY.json': 'd37e9606c03bb2e384a482fe9d326043127c86f471e0242424c6da1dbb9327a6',
    BASE+'admitted/ADMISSION.json': '8371d21eb44f1590f545b0808bfdf05ad63a52bf6ce63c285a3e889b758fde1c',
    BASE+'independent/REVIEW.json': '0bb19df028a429cd81e304cfc43bf4c2f7b31ff122e551a565bd8d498f83b1d6',
    BASE+'independent/ADMISSION_REVIEW.json': 'db43540f98c865dac5a718845ac5917c0c6305569edc7260a5136d1298a1b736',
    'work/wc194_independent_admission_review.py': '936997743dfb57d7470d958c4212755ed0cc2105862c556df407d38a8441eba8',
    'work/wc194_admission.py': '6aef1142c413cb0197db3710d5e061787726c3be8e5755cab80c10826e01ed37',
    'work/wc194_assemble.py': 'fe9d554e0ef6cff934f448fa556e9fc3ac62651946cd523aae5726a2c370f4a2',
    'work/monthly_cash_quality146_factor.py': '6f5d819488c7952cb32894cf4b5a4f549c73adb570e949fd47c3c6ba2e561eea',
    'work/monthly_same_report145_runner.py': 'f400350c20a6d95f1d7ed60753ff529b5544b2b8901abbd0e801999843500939',
    'outputs/monthly_local_linear70/study.py': 'e8cfe0c27a6805cbee4b59d24ce9771ece7caa01c29f1a30aa2d5acf9837f3fb',
    'outputs/monthly_core_timing70_research/core_adapter.py': 'b1c45a7e9fc671675b461e6f016eda4e17a7885c1ff74630cadddf9ce8d1544a',
    'outputs/monthly_cash_ppe193/PLAN.json': '548653cc06de37290867c4384a4d748d4bf70c3f47373fe8c64542b739340b1c',
    'work/monthly_sector_recurrence192_independent.py': 'f78cc005827fd977ec431bdec4af0f4b7160694cb3d382df0115e8e00e0bedf8',
    'work/monthly_sector_transition189_independent.py': 'e83305d9f784ef9aa5ae0554cd1deb558d927b4182010af599b977efa69618dc',
    'work/price_distribution171_independent.py': '94e2a984affad28ae1ba62de245519ffd9f4763c1a6ab3a5cd95dea02e0a1841',
    'work/ptr169_results_independent.py': '6264571ad6ddacbc408449180c74a32b37659757c9bdfc5ce0e2672467607d59',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()


def read(path):
    return json.loads((ROOT/path).read_text())


def imported_modules(tree):
    result = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.add(node.module)
    return result


def main():
    for path, expected in {**INPUTS, **CODE}.items():
        require(sha(path) == expected, 'Reviewed bytes changed: '+path)
    trees = {path: ast.parse((ROOT/path).read_text()) for path in CODE}
    oracle_imports = imported_modules(trees['work/monthly_trade_wc194_independent.py'])
    require(not any('monthly_trade_wc194_factor' in module or 'monthly_cash_quality146_factor' in module for module in oracle_imports), 'Oracle imports producer arithmetic')
    counts = {path: sum(isinstance(node, ast.FunctionDef) and node.name.startswith('test_') for node in ast.walk(tree))
              for path, tree in trees.items() if Path(path).name.startswith('test_')}
    require(counts == {
        'work/test_monthly_trade_wc194_host.py': 31,
        'work/test_monthly_trade_wc194_factor.py': 29,
        'work/test_monthly_trade_wc194_independent.py': 23,
    }, 'Reviewed synthetic coverage changed')
    receipt = read(TESTS)
    require(receipt['passed'] is True and receipt['synthetic_only'] is True and receipt['tests'] == sum(counts.values()) == 83, 'Formal synthetic result incomplete')
    require(receipt['runtime'] == dict(python='3.9.6', numpy='2.0.2', pandas='2.3.3'), 'Synthetic runtime changed')
    for path, expected in CODE.items():
        require(receipt['code_sha256'].get(str(ROOT/path)) == expected, 'Formal tests did not cover reviewed file: '+path)
    for path, expected in receipt['code_sha256'].items():
        require(sha(path) == expected, 'Formal test dependency changed: '+path)
    source_review = read(BASE+'independent/ADMISSION_REVIEW.json')
    require(source_review['passed'] is True and source_review['source_only'] is True and
            (source_review['events'], source_review['admitted_events'], source_review['held_events'],
             source_review['inherited_holds_preserved'], source_review['closure_pins_verified']) == (44, 29, 15, 13, 267), 'Source admission review incomplete')
    policy = read(BASE+'admitted/POLICY.json')
    require(len(policy['input_hashes']) == 267, 'Source closure census')
    for path, expected in policy['input_hashes'].items():
        require(sha(path) == expected, 'Reviewed source dependency changed: '+path)
    # Only metadata is read. No source monetary cells are evaluated here.
    require(not (ROOT/'outputs/monthly_trade_wc194/PLAN.json').exists(), 'This review must precede registration')
    require(not any(name in sys.modules for name in ('monthly_trade_wc194_host', 'monthly_trade_wc194_factor', 'monthly_trade_wc194_independent')), 'Review script executed a production module')
    pins = dict(INPUTS, **{SELF: sha(SELF)})
    checks = {
        'all_three_signed_cashflow_contributions_required': True,
        'component_signs_preserved_not_absolute_or_inverted': True,
        'current_cfo_minus_components_before_one_bounded_mapping': True,
        'same_report_current_and_comparison_assets_positive': True,
        'original146_net_income_gate_retained_but_not_new_ratio_input': True,
        'comparison_flow_values_unused_and_no_duration_rescaling': True,
        'latest_known_report_selected_before_validity': True,
        'held_missing_stale_or_ambiguous_latest_has_no_older_fallback': True,
        'future_numerical_fields_not_read_by_report_selection': True,
        'first_session_after_release_is_frozen_source_gate': True,
        'annual330_to400_contiguous_periods_and550_day_inclusive_expiry': True,
        'both_admissions_same_common_mask_and_etfs_neutral': True,
        'normalized_cells_preserve_raw_source_period_clock_and_signed_literal': True,
        'same_cfo_source_table_or_pdf_page_and_native_currency_evidence': True,
        'exact_scaled_fraction_before_single_bounded_float': True,
        'independent_oracle_decimal_parser_and_closed_form_not_producer_import': True,
        'oracle_verifies_complete_arrays_and_each_audit_field_exactly': True,
        'original_eligible_and_common_only_direct_point01_delta': True,
        'no_cross_sectional_rank_or_second_nonlinearity': True,
        'no_additional_first_partial_suppression_per_fixed_contract': True,
        'only_nonzero_delta_cells_written_to_local67_pre_score': True,
        'original27_arrays_and_episode_support_unchanged': True,
        'single_original_episode_per_arm': True,
        'missing_and_zero_feature_exact_incumbent_fallback_checks': True,
        'five_release_truncated_feature_prefixes_and123_decision_prefixes': True,
        'source_and_code_review_gates_before_actual_prepare_loader': True,
        'explicit_plan_runtime_source_and_code_hash_gates_each_phase': True,
        'first88_candidate_only_selection_control_never_promoted': True,
        'frozen145_guard_with_strict_close_both_references': True,
        'next_open5bp_noninferior_incumbent_strict_control': True,
        'risk_sharpe_point05_mdd_point03_era_cagr_point025': True,
        'full_training_early53_recent35_and_walk_prefix_half_split': True,
        'walkforward36_60_76_uses_only_prefix_selection': True,
        'walkforward_stitched_diagnostic_not_primary_admission': True,
        'later34_after_frozen_selection_diagnostic_only': True,
        'all3arms_9paths_45slices_and18_training_metric_checks': True,
        'pinned_independent_fills_fees_nav_no_terminal_liquidation': True,
        'write_once_outputs_and_central_compare_swap': True,
        'predecessor193_and21444_protected_archive_retained': True,
        'correct191_1090_to192_1093_closure_counters': True,
    }
    result = dict(
        schema='wc194_independent_code_review_v1', created_utc=datetime.now(timezone.utc).isoformat(),
        passed=True, unresolved_blockers=[], code_sha256=CODE, input_sha256=pins,
        review_code_sha256=sha(SELF),
        review_scope='Independent read-only review of final source, tests and frozen adapters; existing source/metadata and formal synthetic receipt only. No actual194 numerical execution.',
        source_admission=dict(events=44, admitted=29, held=15, inherited_holds_preserved=13,
                              closure_pins_verified=267, all_original_fields_preserved=True),
        checks=checks,
        evidence=[
            dict(path='work/monthly_trade_wc194_factor.py', functions=['evaluate_report', '_trade_contributions', '_cell', 'build_features'], finding='Both admission gates and original146 annual/identity/NI gates precede exact signed arithmetic; latest known report is selected before qualification.'),
            dict(path='work/monthly_trade_wc194_independent.py', functions=['evaluate_oracle', 'trade_amounts', 'select_oracle', 'verify_features', 'verify_deltas'], finding='Independent decimal-to-Fraction reconstruction and closed form 2*n/(A_current+A_comparison+abs(2*n)); whole audit and all arrays compared exactly.'),
            dict(path='work/monthly_trade_wc194_host.py', functions=['financial_source_binding', 'code_review_binding', 'gate', 'prepare', 'decisions'], finding='Registered pins and completed source/code reviews precede loading actual inputs; unchanged original arrays, original orders, five source prefixes and123 decision prefixes are required.'),
            dict(path='work/monthly_trade_wc194_host.py', functions=['select_first88', 'training_curves', 'walkforward_training', 'frozen_selection', 'diagnose', 'close'], finding='Only fixed candidate may pass pinned145 gates on first88. Frozen selections and first88 metrics are rechecked before closure; later34 remains diagnostic.'),
            dict(path='outputs/monthly_core_timing70_research/core_adapter.py', functions=['score_and_choose'], finding='blend0/all-missing new-core input leaves original baseline/core/support intact while applying the modified pre-score through exactly one original selector.'),
            dict(path='work/price_distribution171_independent.py', functions=['audit_outcomes'], finding='Frozen independent accounts consume all3 remapped arms,3 executions and5 fixed slices, checking exported fills, holdings, daily positions, fees and metrics.'),
        ],
        synthetic_tests=dict(source_test_functions=counts, total=83, formal_combined_receipt=dict(path=TESTS, sha256=INPUTS[TESTS]),
                             passed=True, synthetic_only=True, reviewer_reran_suites=False,
                             review='Inspected tests for source/sign/scope/clock mutations, held-latest/future poison, exact cancellation, zero/missing common masks, full synthetic122x49 integration, one episode, eligibility, prefixes, risk guards and missing receipts.'),
        division_of_responsibility='The normalized source gate and independent original-document audit establish annual operating-row scope, geometry and strict exchange availability. The pure numerical modules additionally recheck normalized/raw identity, dollar evidence and distinct rows; they do not re-extract or certify documents.',
        limitations=[
            'This review does not establish actual prepare, selection, outcome or protection success; those require the future registered lifecycle.',
            'Only the reviewed four issuers are covered; inherited13 holds remain and two new combined-AP holds remain. No source coverage was inferred from missing values.',
            'Three selected trade contributions are not full working capital; the residual is neither GAAP CFO nor organic cash generation.',
            'First-public bytes, complete revision history and daily security identity are not fully certified. Historical reuse means later34 and walkforward are not pristine OOS.',
            'The historical shared PDFium failure is preserved. Only the separately verified193 isolated source runtime is admitted.',
            'The financial145 contract deliberately has no added0.5percentage-point gain, changed-month minimum or first-row suppression; this review does not add those unrelated gates.',
        ],
        actual_features_computed=False, actual_financial_ratios_computed=False,
        actual_portfolio_returns_computed=False, production_files_modified=False,
        strategy_registered_by_reviewer=False,
    )
    for path, expected in {**CODE, **pins}.items():
        require(sha(path) == expected, 'File changed during review publication: '+path)
    output = ROOT/BASE/'independent/CODE_REVIEW.json'
    with output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(dict(passed=True, unresolved_blockers=[], tests=83,
        path=str(output.relative_to(ROOT)), sha256=hashlib.sha256(output.read_bytes()).hexdigest(), review_code_sha256=sha(SELF))))


if __name__ == '__main__':
    main()
