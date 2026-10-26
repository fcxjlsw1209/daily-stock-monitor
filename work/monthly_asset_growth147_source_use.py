"""Freeze a new purpose policy and verify existing asset endpoints, without ratios."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import re
from decimal import Decimal
import cash_quality_source_admission_v1_independent as prior
from cashflow_source_union_v1_independent import Evidence, encoded, digest, need, unique

ROOT = Path(__file__).resolve().parents[1]
BASE = 'work/monthly_asset_growth147'
POLICY = BASE+'_source_use.json'
REVIEW = BASE+'_source_review.json'
SELF = BASE+'_source_use.py'
TEST = BASE+'_source_use_tests.py'
LOG = BASE+'_source_use_tests.log'
OLD = 'outputs/monthly_cash_quality_source_admission_v1'
REFS = {
    'dataset': (OLD+'/DATASET.json', prior.DATA_SHA),
    'admission': (OLD+'/ADMISSION.json', 'e9d2bf123cdc351fecf1c794fd3a0e929b1d1a6307232d094f402d29b7463494'),
    'prior_policy': (OLD+'/POLICY.json', prior.POLICY_SHA),
    'independent': (OLD+'/INDEPENDENT_REVIEW.json', '94c92b953ea88528f43e62c7c0021f50a9bf63945c16e9cfff994337c653c493'),
    'restatement': ('work/after144_msft_nvda_source_review.json', '5460f92bd8892dfa35514c54493d265d83c153d37030c92a7f23911973e73ae9'),
    'accounting': ('work/cashflow_accounting_review_v1.json', 'ec6d74579b43c7cb8f6fda0c9c4855684ee4e16f334add46836c5854ceaebf9b'),
    'sessions': ('outputs/monthly_forward_guidance_preflight_v1/SESSIONS.json', 'c05c15f19bc1f7a3428e456a71199d64f05799ef3c432fa45f7bc485d0ff3cf4')}
DEPS = {
    'work/cash_quality_source_admission_v1_independent.py': '3e8a32f31ffbbc6d2cdb95db16e4d548abef8d8165df0e5f32f076e17022e783',
    'work/cashflow_source_union_v1_independent.py': 'a71c7e16a5cf97a020726511d657a4bf33eb40dbcc7f02190ee7413561497929'}


def write_once(path, value):
    with path.open('xb') as stream:
        stream.write(encoded(value))
        stream.flush()
        os.fsync(stream.fileno())


def policy_fields():
    return dict(
        schema_version=1, round=147, stage='prospective_source_use_policy',
        purpose='same_report_total_asset_expansion_v1',
        dataset_method=prior.METHOD,
        scope='Only the identical31 conditionally admitted events among all44 frozen146 records; retain13 holds and all six legacy holds.',
        required_endpoints=['current.total_assets', 'comparison.total_assets'],
        eligibility='Retain complete frozen146 four-field gate even though only two asset endpoints define the new variable.',
        same_current_report=True, current_release_next_pinned_session=True,
        annual_days=[330, 400], contiguous_periods=True, max_age_days=550,
        latest_event_before_validity=True, invalid_latest_blocks_older_fallback=True,
        identity='Own release anchors; valid from availability through current fiscal-end plus550 days.',
        measurement_definition='g=(A_current-A_comparison)/A_comparison; later registered factor uses -g/(1+abs(g)). No growth values computed in source review.',
        same_report_currency_literal_assumption=True,
        accounting='As-reported consolidated total-asset expansion; acquisitions, disposals, currency translation and changing business perimeter are included, not adjusted or selectively excluded.',
        restatement='MSFT FY2018 uses both asset columns from its current release on the explicitly restated revenue/lease basis; availability is that release, never backfilled.',
        old_source_policy_unchanged=True, new_events_admitted=0,
        factor_calculations_permitted_in_source_review=False,
        first_public_version_verified=False, complete_correction_history_verified=False,
        certified_daily_security_master=False, iso_currency_newly_certified=False,
        organic_investment_or_CMA_replication=False,
        interpretation=[
            'Four technology-heavy issuers are not broad-market financial coverage.',
            'Same-report endpoints do not certify constant business scope or organic investment.',
            'Negative correction for expansion and neutral uncovered names can alter source-issuer exposure.',
            'Exact mapped range(-1,0.5), or score correction(-0.01,0.005) at dose0.01; no empirical centering or rescaling.',
            'Winning against earnings and low-accrual controls does not statistically isolate every denominator effect.'])


def freeze(root):
    e = Evidence(root)
    refs = {key: dict(path=path, sha256=e.hash(path, sha)) for key, (path, sha) in REFS.items()}
    for path, sha in DEPS.items():
        e.hash(path, sha)
    log = e.path(LOG).read_text()
    need(re.search(r'Ran [1-9][0-9]* tests? in ', log) and log.rstrip().endswith('OK'),
         'Passing source-only synthetic log required')
    sources = {path: e.hash(path) for path in (SELF, TEST, LOG, *DEPS)}
    result = dict(policy_fields(), created_utc=datetime.now(timezone.utc).isoformat(),
                  inputs=refs, source_code_hashes=sources)
    e.stable()
    write_once(e.path(POLICY), result)
    return result


def review_asset_event(event, sessions):
    """Metadata/source checks and decimal positivity only; no growth arithmetic."""
    result = prior.audit_event(event, sessions)
    endpoints = {}
    for side in ('current', 'comparison'):
        cell = event['fields'][side]['total_assets']
        present = cell is not None
        if present:
            value, scale = Decimal(cell['value_decimal']), Decimal(cell['scale'])
            need(value.is_finite() and value > 0 and scale.is_finite() and scale > 0,
                 'Finite positive endpoint and scale')
            need(cell['source'] == event['source'] and cell['period'] == event['periods'][side]
                 and cell['available_from'] == event['available_from'], 'Same current release endpoint')
            need(cell['locator']['period_end'] == event['periods'][side]['period_end'],
                 'Exact year-end endpoint')
        endpoints[side] = dict(present=present, positive_if_present=present)
    need(not result['admitted'] or all(x['present'] for x in endpoints.values()),
         'Both endpoints required for admitted record')
    return dict(result, asset_endpoints=endpoints,
                conditional_asset_use=result['admitted'],
                inherited_rejection_codes=list(event['inherited_rejection_codes']),
                old_admission=dict(event['cash_quality_admission']),
                source=dict(event['source']), periods=event['periods'],
                identity=event['identity'], release_date=event['release_date'])


def verify(root, policy_sha):
    e = Evidence(root)
    policy = e.read(POLICY, policy_sha)
    need({key: policy.get(key) for key in policy_fields()} == policy_fields(), 'Exact fixed source-purpose policy')
    need(policy['inputs'] == {key: dict(path=path, sha256=sha) for key, (path, sha) in REFS.items()},
         'Fixed input references')
    e.pins(policy['source_code_hashes'])
    for name, sha in DEPS.items():
        e.hash(name, sha)
    objects = {key: e.read(ref['path'], ref['sha256']) for key, ref in policy['inputs'].items()}
    old_review = objects['independent']
    need(old_review['passed'] is True and old_review['admitted_events'] == 31
         and old_review['held_events'] == 13, 'Passing fixed146 source review')
    e.pins(old_review['input_hashes'])
    for key in ('dataset', 'admission', 'prior_policy'):
        ref = policy['inputs'][key]
        need(old_review['input_hashes'].get(ref['path']) == ref['sha256'], 'Prior review embedded binding')
    dataset, receipt = objects['dataset'], objects['admission']
    need(dataset['method'] == prior.METHOD and dataset['schema_version'] == 1
         and receipt['passed'] is True and receipt['data_sha256'] == prior.DATA_SHA
         and receipt['policy_sha256'] == prior.POLICY_SHA, 'Existing source admission identity')
    sessions = objects['sessions']['sessions']
    need(sessions == sorted(set(sessions)), 'Ordered unique pinned sessions')
    events = unique(dataset['events'])
    need(len(events) == 44, 'Full44 events required')
    findings = [review_asset_event(event, sessions) for event in events.values()]
    admitted = [x['event_id'] for x in findings if x['conditional_asset_use']]
    held = [dict(event_id=x['event_id'], **x['old_admission']) for x in findings if not x['conditional_asset_use']]
    need(admitted == receipt['admitted_event_ids'] and held == receipt['holds'] == old_review['holds'],
         'Exact original admitted set and rejected sentinels')
    years = {symbol: [x['fiscal_year'] for x in findings if x['admitted'] and x['symbol'] == symbol]
             for symbol in prior.EXPECTED_YEARS}
    need(years == prior.EXPECTED_YEARS and len(admitted) == 31 and len(held) == 13, 'Unchanged31/13 coverage')
    accounting = unique(objects['accounting']['events'])
    need(set(accounting) == set(events), 'All44 accounting records preserved')
    for key, event in events.items():
        need(event['union_provenance']['accounting_event_review'] == accounting[key],
             'Same bound field-specific accounting review')
    restated = unique(objects['restatement']['events'])['MSFT:FY2018:2018-07-19']['accounting']
    need(restated['status'] == 'resolved'
         and restated['resolution_code'] == 'both_comparatives_restated_to_current_revenue_and_lease_basis',
         'Explicit2018 common restatement basis')
    for evidence in restated['evidence']:
        e.hash(evidence['path'], evidence['sha256'])
    current = events['MSFT:FY2018:2018-07-19']['source']
    need(any(x['path'] == current['path'] and x['sha256'] == current['sha256']
             and 'prior periods presented here have been restated' in x['quote']
             for x in restated['evidence']), 'Current release explicit restatement note')
    e.stable()
    return dict(
        schema_version=1, round=147, stage='independent_asset_growth_source_use_review',
        passed=True, policy_sha256=policy_sha, dataset_sha256=prior.DATA_SHA,
        conditional_source_use_only=True, events=44, admitted_events=31, held_events=13,
        admitted_event_ids=admitted, holds=held, findings=findings, admitted_years_by_symbol=years,
        checks=dict(frozen146_dataset_and_full_source_closure_verified=True,
                    all44_records_and_exact31_13_gate_preserved=True,
                    same_current_report_positive_assets_and_units=True,
                    annual_periods_availability_identity_rechecked=True,
                    current_restatement_basis_supported_without_backfill=True),
        input_hashes=dict(e.hashes), reviewed_utc=datetime.now(timezone.utc).isoformat(),
        financial_ratios=0, asset_growth_values_computed=0, factor_calculations=0,
        model_fits=0, portfolio_calculations=0, return_calculations=0,
        newly_admitted_events=0, first_public_version_verified=False,
        complete_correction_history_verified=False, certified_daily_security_master=False,
        limitations=policy['interpretation'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--policy-sha')
    args = parser.parse_args()
    need(args.freeze != bool(args.policy_sha), 'Choose freeze or exact policy-sha verification')
    if args.freeze:
        value = freeze(ROOT)
        path = POLICY
    else:
        value = verify(ROOT, args.policy_sha)
        path = REVIEW
        write_once(ROOT/path, value)
    print(json.dumps(dict(path=path, sha256=digest(encoded(value)), passed=True)))


if __name__ == '__main__':
    main()
