"""Read-only independent verification of the fixed completed admission package."""
from bisect import bisect_right
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
import json
import os
import sys

from cashflow_source_union_v1_independent import Evidence, digest, encoded, need, unique

ROOT = Path(__file__).resolve().parents[1]
SELF = 'work/cash_quality_source_admission_v1_independent.py'
OUT = 'outputs/monthly_cash_quality_source_admission_v1'
POLICY_SHA = '862c3696a2e39a476dbbd6ec450e7b27dd1c1e869eb8334902cd20d673bf5cb7'
DATA_SHA = '3725b384ab4f48a3cabee1fc1adb1acd896be3f4967e8b6cd649a9a690c953d5'
HELPER = 'work/cashflow_source_union_v1_independent.py'
HELPER_SHA = 'a71c7e16a5cf97a020726511d657a4bf33eb40dbcc7f02190ee7413561497929'
METHOD = 'current_annual_cash_quality_v1'
DEFINITION = 'consolidated_gaap_cfo_net_income_average_assets_v1'
REQUIRED = (('current', 'operating_cash_flow'), ('current', 'net_income'),
            ('current', 'total_assets'), ('comparison', 'total_assets'))
EXPECTED_YEARS = {'AAPL': list(range(2017, 2026)), 'AMD': [2016, 2017, 2023, 2024],
                  'MSFT': list(range(2016, 2027)), 'NVDA': list(range(2020, 2027))}
LEGACY = {('AAPL', 2016), ('AMD', 2015), ('AMD', 2018), ('AMD', 2025),
          ('MSFT', 2015), ('NVDA', 2015)}


def day(value):
    need(type(value) is str, 'Explicit date required')
    result = date.fromisoformat(value)
    need(result.isoformat() == value, 'Canonical date required')
    return result


def audit_event(event, sessions):
    """Check constraints independently, without producer assess or factor calls."""
    released, available = day(event['release_date']), day(event['available_from'])
    pos = bisect_right(sessions, str(released))
    need(pos < len(sessions) and sessions[pos] == str(available), 'Availability must be first later pinned session')
    identity = event['identity']
    need(identity['security_id'] == event['symbol'] and type(identity['issuer_id']) is str
         and identity['issuer_id'].strip(), 'Issuer/security identity')
    current_end = day(event['periods']['current']['period_end'])
    need(identity['identity_valid_from'] == str(available)
         and identity['identity_valid_through'] == str(current_end+timedelta(days=550)), 'Causal identity window')
    need(current_end <= released < available, 'Release chronology')
    annual = True
    parsed = {}
    for side in ('current', 'comparison'):
        record = event['periods'][side]
        try:
            start, end = day(record['period_start']), day(record['period_end'])
            duration = (end-start).days+1
            need(330 <= duration <= 400 and type(record['days_inclusive']) is int
                 and record['days_inclusive'] == duration, 'Annual duration')
            parsed[side] = (start, end)
        except (ValueError, TypeError, KeyError):
            annual = False
    if annual:
        annual = parsed['comparison'][1]+timedelta(days=1) == parsed['current'][0]
    missing = []
    for side, name in REQUIRED:
        cell = event['fields'][side][name]
        if cell is None:
            missing.append(side+'.'+name)
            continue
        need(cell['source'] == event['source'] and cell['period'] == event['periods'][side]
             and cell['available_from'] == str(available), 'Required cell source/period/availability')
        locator = cell['locator']
        need(locator['period_end'] == event['periods'][side]['period_end'], 'Locator annual/instant end')
        need(cell['currency_literal'] == locator['currency_literal'] == '$'
             and cell.get('currency_iso') in (None, 'USD'), 'Same-report literal currency')
        need(type(cell['value_decimal']) is str and type(cell['scale']) is str, 'Decimal/scale strings')
        number, scale = Decimal(cell['value_decimal']), Decimal(cell['scale'])
        need(number.is_finite() and scale.is_finite() and scale > 0, 'Finite exact source values and scale')
        need(Decimal(locator['scale']) == scale, 'Locator scale')
        need(name != 'total_assets' or number > 0, 'Positive asset endpoint')
        need(cell['fact_admitted'] is False, 'Original raw field flag')
        if name == 'net_income':
            need(locator['ni_scope'] == 'indirect_method_consolidated_start', 'Consolidated reconciliation NI')
    if not missing:
        cash = event['fields']['current']['operating_cash_flow']['locator']
        income = event['fields']['current']['net_income']['locator']
        need(cash['format'] == income['format']
             and cash['statement_title'] == income['statement_title'], 'Same CF statement')
        keys = ('page_1based', 'financial_column_index_0based') if cash['format'] == 'pdf' else (
            'table_index_zero_based', 'logical_columns')
        need(all(cash[k] == income[k] for k in keys), 'Same physical CF annual column')
    legacy = bool(event['inherited_rejection_codes']) or event['prior_field_admitted'] is not True
    expected = not (legacy or event['metadata_gaps'] or missing or not annual)
    admission = event['cash_quality_admission']
    need(set(admission) == {'admitted', 'reasons', 'definition'}
         and type(admission['admitted']) is bool and admission['definition'] == DEFINITION, 'New admission schema')
    need(admission['admitted'] == expected, 'Independently recomputed admission mismatch')
    reasons = admission['reasons']
    need(type(reasons) is list and all(type(x) is str for x in reasons), 'Explicit reasons')
    expected_categories = set()
    if legacy:
        expected_categories.add('preserved_inherited_source_hold')
    if event['metadata_gaps']:
        expected_categories.add('unresolved_source_metadata')
    if not annual:
        expected_categories.add('period_identity_or_clock')
    expected_categories.update('missing_required:'+x for x in missing)
    categories = {x.split(':', 1)[0] if x.startswith('period_identity_or_clock:') else x for x in reasons}
    need(categories == expected_categories and len(reasons) == len(categories), 'Every rejection justified and retained')
    return dict(event_id=event['event_id'], symbol=event['symbol'], fiscal_year=event['fiscal_year'],
                admitted=expected, legacy_hold=legacy, annual_periods_valid=annual,
                missing_required=missing, available_from=str(available))


def main():
    e = Evidence(ROOT)
    e.hash(HELPER, HELPER_SHA)
    policy = e.read(OUT+'/POLICY.json', POLICY_SHA)
    dataset = e.read(OUT+'/DATASET.json', DATA_SHA)
    receipt = e.read(OUT+'/ADMISSION.json')
    need(policy['runtime'] == sys.version and policy['method'] == dataset['method'] == METHOD
         and dataset['schema_version'] == 1 and dataset['conditional_research_only'] is True, 'Fixed method/runtime')
    need(policy['definition'] == DEFINITION and policy['annual_days'] == [330, 400]
         and policy['max_age_days'] == 550 and policy['expected_events'] == 44, 'Fixed temporal policy')
    need(policy['preserve_all_inherited_holds'] is True and policy['use_comparison_flows'] is False
         and policy['same_report_literal_currency_assumption'] is True
         and policy['financial_ratios_permitted'] is False
         and policy['strategy_registration_permitted'] is False, 'Fixed source-only policy')
    refs = policy['inputs']
    inputs = {key: e.read(ref['path'], ref['sha256']) for key, ref in refs.items()}
    e.hash('work/cash_quality_source_admission_v1.py', policy['implementation_sha256'])
    code_ref = policy['independent_code_review']
    code_review = e.read(code_ref['path'], code_ref['sha256'])
    need(code_review['passed'] is True, 'Prior independent code review')
    e.pins(code_review['source_hashes'])
    union_review = inputs['union_independent']
    need(union_review['passed'] is True, 'Independent completed union review')
    e.pins(union_review['input_hashes'])
    for key in ('union_data', 'union_plan', 'union_manifest'):
        ref = refs[key]
        need(union_review['input_hashes'][ref['path']] == ref['sha256'], 'Union review input binding')
    for group in ('source_hashes', 'source_code_hashes'):
        e.pins(inputs['union_plan'][group])
    need(inputs['union_manifest']['output_hashes']['RAW_FACTS.json'] == refs['union_data']['sha256']
         and inputs['union_data']['plan_sha256'] == inputs['union_manifest']['plan_sha256']
         == refs['union_plan']['sha256'], 'Union embedded hashes')
    need(inputs['tests']['passed'] is True
         and inputs['tests']['source_sha256'] == policy['implementation_sha256'], 'Completed source tests')
    need(receipt['passed'] is True and receipt['stage'] == 'conditional_cash_quality_source_admission'
         and receipt['policy_sha256'] == dataset['policy_sha256'] == POLICY_SHA
         and receipt['data_sha256'] == DATA_SHA, 'Actual admission receipt links')
    need(receipt['source_hashes'] == {x['path']: x['sha256'] for x in refs.values()}, 'Admission source closure')
    for key in ('financial_ratios', 'factor_calculations', 'portfolio_calculations',
                'return_calculations', 'network_requests', 'observer_changes'):
        need(type(receipt[key]) is int and receipt[key] == 0, 'Zero operation counter')
    for key in ('first_public_version_verified', 'complete_correction_history_verified', 'certified_daily_security_master'):
        need(receipt[key] is False, 'No upgraded certification')
    original = unique(inputs['union_data']['events'])
    actual = unique(dataset['events'])
    accounting = unique(inputs['accounting']['events'])
    need(len(actual) == 44 and list(actual) == list(original) and set(actual) == set(accounting), 'Exact ordered 44-event census')
    sessions = inputs['sessions']['sessions']
    need(sessions == sorted(set(sessions)) and all(day(x).isoformat() == x for x in sessions), 'Pinned ordered calendar')
    findings = []
    for key, event in actual.items():
        need({k: v for k, v in event.items() if k != 'cash_quality_admission'} == original[key]
             and set(event) == set(original[key]) | {'cash_quality_admission'}, 'Exact raw event preservation')
        need(event['union_provenance']['accounting_review_reference'] == refs['accounting']
             and event['union_provenance']['accounting_event_review'] == accounting[key], 'Exact accounting attachment')
        need(all(event[x] is False for x in ('fact_admitted', 'model_admitted', 'strategy_admitted')), 'Original raw flags')
        findings.append(audit_event(event, sessions))
    admitted = [x['event_id'] for x in findings if x['admitted']]
    holds = [dict(event_id=key, **event['cash_quality_admission'])
             for key, event in actual.items() if not event['cash_quality_admission']['admitted']]
    need(receipt['admitted_event_ids'] == admitted and receipt['holds'] == holds
         and (receipt['events'], receipt['admitted_events'], receipt['held_events']) == (44, 31, 13), 'Exact receipt census')
    years = {symbol: [x['fiscal_year'] for x in findings if x['admitted'] and x['symbol'] == symbol]
             for symbol in EXPECTED_YEARS}
    need(years == EXPECTED_YEARS, 'Independently expected conditional coverage')
    legacy = {(x['symbol'], x['fiscal_year']) for x in findings if x['legacy_hold']}
    need(legacy == LEGACY, 'Exact six original holds')
    e.hash(SELF)
    e.stable()
    result = dict(
        schema_version=1, stage='independent_completed_cash_quality_source_admission_review', passed=True,
        policy_sha256=POLICY_SHA, data_sha256=DATA_SHA,
        admitted_events=31, held_events=13,
        counts=dict(events=44, conditional_admitted=31, held=13, legacy_holds=6,
                    raw_complete_four_fields=sum(not x['missing_required'] for x in findings)),
        admitted_years_by_symbol=years, holds=holds,
        checks=dict(all44_raw_event_dictionaries_exactly_preserved=True,
                    independent_admission_mask_and_rejection_reasons=True,
                    first_later_pinned_session_and_causal_identity_expiry=True,
                    both_annual_durations_and_contiguous_boundaries=True,
                    same_release_required_cell_source_period_date_units=True,
                    current_CFO_and_consolidated_start_NI_same_statement_column=True,
                    all_six_legacy_holds_preserved=True, unused_comparison_flows_not_required=True,
                    source_closure_and_embedded_hashes_verified=True),
        input_hashes=dict(e.hashes), closure_files_verified=len(e.hashes),
        verified_closure_sha256=digest(encoded(e.hashes)),
        financial_ratios=0, factor_calculations=0, model_fits=0, portfolio_calculations=0,
        return_calculations=0, source_admissions_performed=0,
        first_public_version_verified=False, complete_correction_history_verified=False,
        certified_daily_security_master=False,
        limitations=policy['limitations'],
        accounting_interpretation='Current as-reported values under the explicit policy; preserved findings do not certify constant-accounting or organic comparability.',
        review_scope='Completed admission transformation and bound existing source reviews; no repeated monetary extraction and no factor evaluation.')
    path = e.path(OUT+'/INDEPENDENT_REVIEW.json')
    with path.open('xb') as stream:
        stream.write(encoded(result)); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps(dict(passed=True, counts=result['counts'], review_sha256=digest(encoded(result)),
                          closure_files_verified=len(e.hashes)), sort_keys=True))


if __name__ == '__main__':
    main()
