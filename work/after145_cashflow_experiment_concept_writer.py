"""Write/verify one unregistered cash-flow concept; no financial computation."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'work/after145_cashflow_experiment_concept.json'
SOURCES = {
    'AGENTS.md': 'Local execution and frozen-evidence safeguards.',
    'LOCAL_SETUP.md': 'Mandatory isolated local wrapper and source/replay separation.',
    'work/after145_cashflow_hypothesis.json': 'Fixed 44-event source scope and cash-versus-accrual motivation.',
    'work/after145_scientific_next_step.json': 'Source feasibility, novelty limits and matched controls.',
    'work/after145_next_step_review.json': 'Independent accounting, scope, missingness and experiment-readiness review.',
    'work/monthly_same_report144_proposal.json': 'Earlier gross-profit measurement, timing, controls and auxiliary budget.',
    'work/monthly_same_report145_proposal.json': 'Unchanged core and fixed four-issuer source census.',
    'work/monthly_same_report144_factor.py': 'Earlier GP/assets calculation, annual boundaries, 550-day expiry and no-fallback behavior.',
}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def new_concept():
    return {
        'schema_version': 1,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'stage': 'unregistered_concept_requires_source_admission',
        'status': 'input_to_later_formal_preregistration_only',
        'not_authorized_by_this_artifact': ['source_admission', 'factor_computation', 'strategy_registration', 'portfolio_computation', 'return_computation', 'observer_changes'],
        'hypothesis': 'Among unchanged eligible long-momentum candidates, low accrual intensity may provide incremental selection information beyond earnings level and operating-cash-flow level. This is a falsifiable research claim, not established efficacy.',
        'mechanism_label': 'low_accrual_quality_not_generic_earnings_confirmation',
        'source_scope': {
            'fixed_census': 'All 44 original annual-release events of AAPL, AMD, MSFT and NVDA underlying round145, including previously rejected events.',
            'issuer_symbols': ['AAPL', 'AMD', 'MSFT', 'NVDA'],
            'event_count': 44,
            'data_admitted': False,
            'scope_is_fixed_source_inventory_not_factor_coverage': True,
            'raw_source_preflight': 'Current and comparison annual CFO, matching consolidated GAAP net income, and year-end assets: at most 264 physical monetary cells. Existing GP/assets admission never admits these new fields.',
            'minimum_fields_for_concept': ['current_annual_CFO', 'current_annual_matching_consolidated_GAAP_net_income', 'current_year_end_total_assets', 'comparison_year_end_total_assets', 'annual_period_boundaries', 'current_release_availability', 'issuer_security_scope'],
            'comparison_flows_role': 'Collect and audit under the fixed census, but do not use comparison CFO or net income to create additional growth arms in this concept.',
            'coverage_limit': 'Four technology-heavy issuers are not a whole-universe or cross-sector cash-quality sample. Repeated monthly reuse of annual reports does not create independent annual observations.',
        },
        'unchanged_strategy': {
            'core': 'Original dominant 50%/52% long-term momentum; skip the latest 42 exchange sessions.',
            'universe': 'Existing historical monthly32 equities plus17 ETFs; no membership or eligibility expansion.',
            'rebalance': 'Month-end equal-weight three holdings with existing two/cash fallback.',
            'intramonth': 'Fixed existing units; no intramonth trades or new exits.',
            'other_controls': 'Existing timing, costs, auxiliary-state and episode-selection rules remain fixed.',
            'new_model_fits': 0,
        },
        'measurement': {
            'formulae_are_definitions_only_not_computed_values': True,
            'period': 'Current complete fiscal year from the same current release; both annual periods must satisfy existing 330..400 inclusive calendar-day and contiguous-boundary rules.',
            'scope': 'Consolidated GAAP CFO and consolidated GAAP net income matching the cash-flow reconciliation starting line, before parent/noncontrolling attribution. Do not substitute parent-only, continuing-only, non-GAAP, FCF or EBITDA amounts.',
            'average_assets': 'A_avg = (A_prior_end + A_current_end) / 2',
            'earnings': 'E = NI_current / A_avg',
            'cash_flow': 'C = CFO_current / A_avg',
            'primary_low_accrual': 'Q = C - E = (CFO_current - NI_current) / A_avg',
            'direction': 'Larger Q favors cash relative to accounting earnings; this is not a causal or fraud score.',
            'bounded_continuous_mapping': 'h(x) = x / (1 + abs(x))',
            'positive_assets_required': True,
            'negative_income_or_CFO': 'Preserve signed values; never divide by net income or replace losses by zero.',
            'scale_and_currency': 'Require field-specific admitted common currency/scale within the event; normalize only under the future explicit source policy.',
            'no_annualization': 'No inferred 365-day year or duration normalization; actual annual duration remains recorded. Residual 330..400-day variation is an interpretation limitation shared by all arms.',
            'excluded_variants': ['annual_change', 'quarterly_subtraction', 'TTM', 'cross-sectional_ranking', 'fitted_coefficients', 'interaction_terms', 'post-result_rescaling'],
        },
        'fixed_four_arms': [
            {'name': 'incumbent', 'feature': None, 'role': 'Unchanged portfolio and execution reference.'},
            {'name': 'earnings_only', 'feature': 'h(E)', 'role': 'Controls whether any improvement is only reported earnings scaled by assets.'},
            {'name': 'cash_flow_only', 'feature': 'h(C)', 'role': 'Controls whether any improvement is only operating cash flow scaled by assets.'},
            {'name': 'low_accrual_quality', 'feature': 'h(Q)', 'role': 'Only primary challenger; tests the cash-minus-earnings component.'},
        ],
        'dose_concept': {
            'status': 'tentative_until_formal_preregistration_before_any_real_feature_or_outcome',
            'coefficient': 0.01,
            'proposed_adjustment': 'delta_score = 0.01 * h(feature_input)',
            'absolute_score_cap': 0.01,
            'units': 'score units, not portfolio-weight percentage',
            'placement': 'Replace one declared auxiliary correction, never stack it with an old GP correction. Exact interface/normalization must be fixed and independently checked before formal registration.',
            'same_for_all_three_experimental_arms': True,
            'reason': 'A conservative engineering cap below the prior auxiliary maximum absolute 0.0125 score adjustment, preserving the dominant core. It is not an economically estimated optimal dose.',
            'no_dose_or_mapping_sweep': True,
            'control_limit': 'Identical mapping and score cap do not equalize empirical feature variance or guarantee equal portfolio turnover. Report those differences without tuning them away.',
        },
        'common_mask_and_time': {
            'identical_mask_all_three_experimental_arms': True,
            'release_rule': 'Current-release values become available only on the first pinned exchange session strictly after that release date; later filings and comparative revisions are never backdated.',
            'max_age_calendar_days_after_current_fiscal_end': 550,
            'expiry_reason': 'Reuse the existing causal annual-report lifetime to avoid another outcome-selected degree of freedom; 550 is not asserted to be optimal.',
            'latest_before_validity': True,
            'invalid_latest_blocks_older_fallback': True,
            'unknown_fields': 'Unadmitted, missing, stale, conflicting, mismatched-scope, unresolved-restatement or invalid-latest records make all three experiment features unavailable.',
            'strategy_fallback': 'Unavailable feature means exact incumbent score/selection fallback, not zero financial amounts and not removal from the candidate universe.',
            'etfs_and_other_issuers': 'No invented company metrics; exact incumbent fallback.',
            'source_holds': 'Re-audit CFO, net income, average-assets endpoints and accounting restatements separately from old GP/assets clearance.',
        },
        'falsification_and_controls': {
            'required_contrast': 'Primary must meet prospectively fixed next-open net-cost and risk gates versus incumbent and both same-mask controls; improvement over earnings_only alone cannot establish incremental accrual information.',
            'no_evidence_cases': ['identical ordered portfolios', 'improvement only in zero-cost same-close replay', 'failure against either matched earnings or matched cash-flow control', 'unresolved source/identity/accounting gate'],
            'unchanged_evaluation_protocol': 'Formal preregistration must bind existing 88-training/34-later month partition and execution/risk thresholds before numerical work. Later34 and full history have been reused; they are diagnostic, not pristine out-of-sample validation.',
            'limited_interpretation': 'A failing pilot need not disprove the economic effect globally; success is not broad cross-sector validation, exact author replication or evidence that a 70% target is attainable.',
            'no_outcome_guided_choices': 'No selecting issuers, fiscal years, periods, weights, staleness, maps or definitions using particular profitable or losing months.',
        },
        'distinction_from_GP_assets': 'Rounds144/145 compare annual gross-profit growth and GP/period-end-assets improvement signs. This concept uses current-year net-income/CFO reconciliation and average assets with a continuous level mapping. It does not expand the old GP coverage or reuse its admission for new fields.',
        'economic_limitations': ['working-capital timing', 'depreciation and other noncash adjustments', 'stock-based compensation', 'tax timing', 'acquisitions/disposals and discontinued-operation scope', 'unequal annual durations', 'sparse source coverage and issuer-identity proxy risk', 'unverified first-public versions and correction history unless separately admitted'],
        'performed': {'new_monetary_cell_extractions': 0, 'financial_calculations': 0, 'factor_calculations': 0, 'ranks_or_orders': 0, 'return_calculations': 0, 'source_admissions': 0, 'strategy_registrations': 0, 'observer_changes': 0, 'other_files_modified': []},
        'source_bindings': {name: {'sha256': sha(ROOT / name), 'role': role} for name, role in SOURCES.items()},
        'writer': {'path': str(Path(__file__).relative_to(ROOT)), 'sha256': sha(Path(__file__))},
    }

def validate(value):
    checks = []
    def check(ok, label):
        if not ok:
            raise ValueError(label)
        checks.append(label)
    check(value['stage'] == 'unregistered_concept_requires_source_admission', 'unregistered_stage')
    check(value['source_scope']['data_admitted'] is False, 'not_admitted')
    check(value['source_scope']['event_count'] == 44 and value['source_scope']['issuer_symbols'] == ['AAPL','AMD','MSFT','NVDA'], 'fixed_census')
    check([a['name'] for a in value['fixed_four_arms']] == ['incumbent','earnings_only','cash_flow_only','low_accrual_quality'], 'four_fixed_arms')
    check(value['measurement']['bounded_continuous_mapping'] == 'h(x) = x / (1 + abs(x))', 'fixed_continuous_mapping')
    check(value['dose_concept']['coefficient'] == 0.01 and value['dose_concept']['absolute_score_cap'] == 0.01, 'tentative_score_budget')
    check(value['dose_concept']['status'].startswith('tentative_'), 'dose_not_registered')
    check(value['common_mask_and_time']['identical_mask_all_three_experimental_arms'] is True, 'common_mask')
    check(value['common_mask_and_time']['invalid_latest_blocks_older_fallback'] is True, 'no_fallback')
    check(value['common_mask_and_time']['max_age_calendar_days_after_current_fiscal_end'] == 550, 'inherited_expiry')
    check(all(v == 0 for k,v in value['performed'].items() if k != 'other_files_modified'), 'no_financial_or_strategy_activity')
    check(value['performed']['other_files_modified'] == [], 'scope_isolated')
    check(set(value['source_bindings']) == set(SOURCES), 'source_set')
    for name, binding in value['source_bindings'].items():
        check(binding['sha256'] == sha(ROOT / name), 'source_sha:' + name)
    check(value['writer']['sha256'] == sha(Path(__file__)), 'writer_sha')
    return checks

p = argparse.ArgumentParser()
p.add_argument('--verify-only', action='store_true')
args = p.parse_args()
if args.verify_only:
    concept = json.loads(TARGET.read_text())
else:
    concept = new_concept()
    validate(concept)
    payload = json.dumps(concept, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    with TARGET.open('x', encoding='utf-8') as handle:
        handle.write(payload)
    concept = json.loads(TARGET.read_text())
checks = validate(concept)
print(json.dumps({'status':'PASS','mode':'verify_only' if args.verify_only else 'created_and_verified','artifact':str(TARGET.relative_to(ROOT)),'sha256':sha(TARGET),'checks_passed':len(checks),'stage':concept['stage'],'financial_calculations':0,'factor_calculations':0,'return_calculations':0},ensure_ascii=False))
