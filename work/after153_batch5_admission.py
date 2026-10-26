"""Offline conditional PLTR/PYPL/TMO source admission; no ratios, models or returns.

The five previous source packages remain immutable references. Full annual
catalogs and stock-month axes are preserved; latest invalid events block use.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from decimal import InvalidOperation
import hashlib
import json
from pathlib import Path
import re

import after152_batch4_admission as previous
import after149_batch1_admission as old

ROOT = Path(__file__).resolve().parents[1]
OUT = 'outputs/monthly_financial_coverage_admission_batch5_v1'
BATCH = 'outputs/monthly_financial_coverage_batch5_v1'
BATCH4, BATCH3, BATCH2, BATCH1 = previous.OUT, previous.BATCH3, previous.BATCH2, old.OUT
POLICY = 'work/after153_batch5_admission_policy.json'
REVIEW = 'work/after153_batch5_independent_review.json'
ACCOUNTING = 'work/after153_batch5_accounting_review.json'
CONTEXT = 'work/after153_next_source_context.json'
STATUS = 'standard_HTML_annual_header_context_located_not_admitted'
CENSUS, META, SESSIONS = old.CENSUS, old.META, old.SESSIONS
OLD_DATA, OLD_ADMISSION = old.OLD_DATA, old.OLD_ADMISSION
FIELDS, METHOD, DEFINITION = old.FIELDS, old.METHOD, old.DEFINITION
SYMBOLS = frozenset(('PLTR', 'PYPL', 'TMO'))
FULL = [('PLTR',y) for y in range(2020,2026)]+[('PYPL',y) for y in range(2015,2026)]+[('TMO',y) for y in range(2012,2023)]
SELECTED = [('PLTR',y) for y in range(2020,2026)]+[('PYPL',y) for y in range(2015,2020)]+[('TMO',y) for y in range(2012,2022)]
ASSUMPTIONS = {s: s+'_consecutive_annual_boundaries_v1' for s in SYMBOLS}
ISSUERS = {'PLTR':'Palantir Technologies Inc.', 'PYPL':'PayPal Holdings, Inc.', 'TMO':'Thermo Fisher Scientific Inc.'}
SCALES = {'PLTR':'1000', 'PYPL':'1000000', 'TMO':'1000000'}
CATALOG_KEYS = ('event_id','symbol','fiscal_year','publication_date','available_session',
                'available_session_verified','official_actual_period_end','source','source_kind',
                'original_report_bytes_verified','context_status','context_holds')
ACCOUNTING_SCOPE = 'current_annual_net_income_same_report_assets'
MAX_AGE = 550
check, day, decimal, canonical, digest = old.check, old.day, old.decimal, old.canonical, old.digest
normalize_sessions, first_session, select_latest = old.normalize_sessions, old.first_session, old.select_latest
Evidence = previous.Evidence
_bound = previous._bound


def source_files():
    return tuple(dict.fromkeys(('work/after153_batch5_admission.py',
        'work/test_after153_batch5_admission.py') + previous.source_files()))


def policy_contract(policy):
    check(policy.get('schema_version') == 1 and policy.get('policy_id') == 'pltr_pypl_tmo_annual_ni_same_report_assets_batch5_v1'
          and policy.get('stage') == 'prospective_source_policy_not_strategy_registration', 'Wrong source policy')
    check(policy.get('definition') == DEFINITION and policy.get('required_fields') == list(FIELDS), 'Measurement changed')
    scope = policy.get('source_scope', {})
    check(scope.get('symbols') == ['PLTR','PYPL','TMO'] and scope.get('catalog_events') == 28
          and scope.get('extracted_events') == 21 and scope.get('unprocessed_catalog_events') == 7
          and scope.get('selection') == [list(x) for x in SELECTED] and scope.get('no_automatic_admission') is True,
          'Fixed source scope changed')
    timing = policy.get('known_time', {})
    check(timing.get('age_days') == MAX_AGE and timing.get('age_from') == 'current annual period end'
          and timing.get('first_public_version_verified') is False, 'Time policy changed')
    periods = policy.get('periods', {})
    check(periods.get('allowed_days') == [365,366] and periods.get('required_year_end') == '12-31',
          'Calendar annual boundary policy required')
    boundaries = periods.get('boundary_assumptions', {})
    check(set(boundaries) == SYMBOLS, 'All issuer boundary policies required')
    for symbol in SYMBOLS:
        boundary = boundaries[symbol]
        check(boundary.get('not_a_quoted_fact') is True and boundary.get('requires_independent_header_verification') is True
              and boundary.get('assumption_id') == ASSUMPTIONS[symbol]
              and boundary.get('formula') == 'prior annual assets date + one calendar day',
              'Explicit unquoted boundary policy required')
    preservation = policy.get('preservation', {})
    check(preservation.get('stock_month_axis') == 3904, 'Stock-month population changed')
    for name in ('all28_original_catalog_entries','no_observer_changes','no_new_strategy_trials',
                 'no_ratio_or_score_or_return','no_first_public_time_or_currency_ISO_or_security_master_certification'):
        check(preservation.get(name) is True, 'Source preservation policy changed: '+name)
    check(policy.get('accounting', {}).get('all_context_holds_retained') is True, 'Context holds must be preserved')


def catalog_projection(context):
    return [{k:deepcopy(row[k]) for k in CATALOG_KEYS} for row in context['events']]


def validate_census_context(raw, context):
    """Retain raw provenance while checking the context's documented labels."""
    kinds = {
        'issuer_original_release_full_bytes': ('HTML_original', True),
        'issuer_original_earnings_release_html': ('HTML_original', True),
        'public_index_observation_of_original_issuer_release': ('index_or_unresolved_not_original_report', False),
    }
    check(raw.get('source_kind') in kinds, 'Unknown census source classification')
    expected_kind, original_bytes = kinds[raw['source_kind']]
    check(context.get('census_source_kind') == raw['source_kind'] and
          context.get('source_kind') == expected_kind and
          context.get('original_report_bytes_verified') is original_bytes,
          'Census/context source classification mismatch')
    check(all(context.get(k) == v for k, v in raw.items() if k != 'source_kind'),
          'Frozen census/context metadata mismatch')


def context_map(context):
    docs = context['events']
    check(len(docs) == len({r['event_id'] for r in docs}) == 28 and
          [(r['symbol'],r['fiscal_year']) for r in docs] == FULL, 'Full28 ordered context required')
    chosen = [r for r in docs if r.get('context_status') == STATUS]
    check([(r['symbol'],r['fiscal_year']) for r in chosen] ==
          SELECTED, 'Fixed21 context required')
    check(all(r.get('source_kind') == 'HTML_original' and r.get('original_report_bytes_verified') is True
              for r in chosen), 'Only original reports can be extracted')
    return {r['event_id']:r for r in chosen}


def accounting_resolved(reviewed, source):
    assessment = reviewed.get('accounting_review')
    if not isinstance(assessment, dict) or assessment.get('status') != 'resolved':
        return False
    check(assessment.get('scope') == ACCOUNTING_SCOPE and assessment.get('source') == source, 'Accounting scope/source mismatch')
    checks = assessment.get('field_checks', {})
    check(set(checks) == set(FIELDS) and all(v is True for v in checks.values()) and
          assessment.get('current_prior_common_basis') is True, 'Resolved accounting lacks field evidence')
    check(type(assessment.get('basis')) is str and assessment['basis'].strip(), 'Accounting basis missing')
    evidence = assessment.get('evidence')
    check(isinstance(evidence, list) and evidence, 'Accounting source evidence missing')
    for item in evidence:
        check(item.get('path') == source['path'] and item.get('sha256') == source['sha256'] and
              isinstance(item.get('locator'), dict) and item['locator'] and
              type(item.get('quote')) is str and item['quote'].strip(), 'Accounting evidence source mismatch')
    check(isinstance(assessment.get('reasons'), list) and all(type(x) is str for x in assessment['reasons']),
          'Accounting reasons malformed')
    return True


def verify_independent_review(review, events, plan_sha, extraction_sha):
    rows = old.verify_independent_review(review, events, plan_sha, extraction_sha)
    for event in events:
        row = rows[event['event_id']]
        check(row.get('symbol') == event['symbol'] and row.get('fiscal_year') == event['fiscal_year'], 'Review identity mismatch')
        check(row.get('identity_verified') is True and row.get('issuer_id') == ISSUERS[event['symbol']]
              and row.get('security_id') == event['symbol'], 'Reviewed report-local identity required')
        check(row.get('retained_context_holds') == event['context_holds'] and
              row.get('accounting_context_numbers_redacted') == event['accounting_context_numbers_redacted'],
              'Review lost inherited source context')
        check(row.get('source') == event['source'] and row.get('available_from') == event['available_from']
              and row.get('release_date') == event['publication_date'], 'Review source/time mismatch')
        if any(event['fields'].values()):
            check(row.get('periods_verified') is True and row.get('currency_verified') is True,
                  'Independent period/currency evidence required')
        for name in FIELDS:
            field = row['fields'][name]
            if event['fields'][name] is None:
                check(type(field.get('null_classification')) is str and field['null_classification'], 'Precise null cause required')
            else:
                for flag in ('value_and_sign_verified','units_verified','period_verified','source_locator_verified'):
                    check(field.get(flag) is True, 'Missing independent field verification: '+flag)
        if row.get('boundary_assumption_verified') is True:
            check(row.get('assumption_id') == ASSUMPTIONS[event['symbol']] and row.get('start_is_source_quoted') is False,
                  'Boundary assumption identity mismatch')
            evidence = row.get('boundary_evidence', {})
            check(set(evidence) == set(FIELDS), 'Three boundary source locators required')
            for name in FIELDS:
                field = event['fields'][name]
                endpoint = event['periods']['comparison' if name == 'prior_total_assets' else 'current']['period_end']
                # Header/date evidence remains independently verifiable when a
                # frozen monetary cell is null. Missing values still hold below.
                check(type(evidence[name].get('table')) is int and evidence[name].get('headers')
                      and evidence[name].get('date') == endpoint, 'Boundary source locator mismatch')
                if field is not None:
                    check(field.get('as_of') == endpoint, 'Field/boundary endpoint mismatch')
            check(row.get('derived_start') == event['periods']['current']['period_start'], 'Reviewed annual start mismatch')
        accounting_resolved(row, event['source'])
    return rows


def assess_event(event, context, reviewed, sessions):
    result = deepcopy(event)
    check(reviewed.get('event_id') == event['event_id'] and reviewed.get('passed') is True, 'Event review mismatch')
    check(not any(k in event for k in ('ni_assets_admission','source_review','clock_review')), 'Gate already present')
    check(event['symbol'] in SYMBOLS and all(event[k] == context[k] for k in ('symbol','event_id','fiscal_year')),
          'Context event mismatch')
    check(event['event_id'] == event['symbol']+'_'+event['publication_date'], 'Event/date mismatch')
    check(set(event.get('fields', {})) == set(FIELDS) and
          all(event.get(k) is False for k in ('fact_admitted','model_admitted','strategy_admitted')), 'Invalid extraction flags/fields')
    for flag in ('first_public_version_verified','first_public_time_verified','complete_correction_history_verified',
                 'certified_daily_security_master'):
        check(event.get(flag) is False, 'Unsupported extraction certification')
    check(type(event.get('financial_calculations')) is int and event['financial_calculations'] == 0,
          'Unexpected financial calculations')
    source = event['source']
    check(source == context['source'] and source['path'] == context['evidence_path'] and source['sha256'] == context['evidence_sha256']
          and source['url'] == context['source_url'], 'Source/context mismatch')
    check(event['context_holds'] == context['context_holds'] and
          event['accounting_context_numbers_redacted'] == context['accounting_context_numbers_redacted'], 'Inherited context changed')
    check(event.get('accounting_review_flags') == context.get('accounting_terms', []), 'Accounting flags changed')
    release, available = event['publication_date'], event['available_from']
    check(release == context['publication_date'] and available == context['available_session'] and day(release) < day(available),
          'Event/context time mismatch')
    derived = first_session(release, sessions)
    clock_ok = derived is not None and derived == available
    check(context.get('available_session_verified') is clock_ok, 'Context clock differs from pinned calendar')
    check(reviewed.get('clock_verified') is clock_ok, 'Independent clock differs from pinned calendar')
    check(reviewed.get('issuer_id') == ISSUERS[event['symbol']] and reviewed.get('security_id') == event['symbol']
          and reviewed.get('identity_verified') is True, 'Missing reviewed source identity')
    reasons = []
    if not clock_ok:
        reasons.append('unverified_exchange_calendar' if derived is None else 'availability_not_first_pinned_session')
    boundary_ok = (reviewed.get('boundary_assumption_verified') is True and
                   reviewed.get('assumption_id') == ASSUMPTIONS[event['symbol']])
    if not boundary_ok:
        reasons.append('annual_boundary_assumption_unverified')
    if not accounting_resolved(reviewed, source):
        reasons.append('accounting_comparability_unresolved')
    allowed_holds = {'monetary_cells_and_sign_geometry_not_yet_independently_verified',
                     'currency_ISO_and_first_public_version_not_certified',
                     'publication_outside_pinned_calendar_range',
                     'accounting_notes_require_field_specific_independent_review',
                     'proposed_annual_start_boundary_assumption_not_yet_approved'}
    for hold in event['context_holds']:
        if hold not in allowed_holds:
            reasons.append('retained_context_hold:'+hold)
    gaps = event.get('gaps', [])
    check(all(isinstance(g,dict) and g.get('field') in FIELDS and type(g.get('reason')) is str and g['reason'] for g in gaps),
          'Malformed field gap')
    check(len({g['field'] for g in gaps}) == len(gaps) and
          {g['field'] for g in gaps} == {n for n,f in event['fields'].items() if f is None}, 'Gap/null positions differ')
    check(event.get('complete_required_fields') is (not gaps), 'Field completeness mismatch')
    for name in FIELDS:
        if event['fields'][name] is None:
            reasons.append('missing_'+name)
    start = end = None
    try:
        period = event['fields']['annual_net_income']['period']
        start, end = day(period['period_start']), day(period['period_end'])
        prior = day(context['periods']['prior_end'])
        count = (end-start).days+1
        check(type(period['days_inclusive']) is int and period['days_inclusive'] == count and count in (365,366),
              'Invalid annual duration')
        check(end.isoformat() == context['official_actual_period_end'] == context['periods']['current_end']
              and start-timedelta(days=1) == prior and str(start) == context['periods']['proposed_current_start']
              and end < day(release), 'Period/context endpoints mismatch')
        check(event['periods']['current'] == period and event['periods']['comparison']['period_end'] == str(prior)
              and event['periods'].get('start_assumption_certified') is False and
              event['periods'].get('start_basis') == 'derived_same_report_prior_annual_end_plus_one_not_directly_quoted',
              'Annual boundary derivation not explicit')
        check(reviewed.get('derived_start') == str(start) and reviewed.get('start_is_source_quoted') is False,
              'Independent annual boundary mismatch')
        year = event['fiscal_year']
        check(count in (365,366) and end.year == year and prior.year == year-1,
              'Issuer calendar annual boundary mismatch')
        check(end.month == prior.month == 12 and end.day == prior.day == 31 and event['periods'].get('assumption_id') == ASSUMPTIONS[event['symbol']],
              'Issuer annual month or assumption mismatch')
    except (KeyError, TypeError, ValueError):
        reasons.append('invalid_current_annual_period')
    for name, cell in event['fields'].items():
        if cell is None:
            continue
        try:
            check(cell['source'] == source and cell['available_from'] == available and cell.get('fact_admitted') is False,
                  'Cell source/time/certification mismatch')
            check(type(cell['displayed_amount']) is str and cell['displayed_amount'].strip() and
                  isinstance(cell['locator'], dict) and cell['locator'], 'Missing physical source cell')
            amount, scale = decimal(cell['value_decimal']), decimal(cell['scale'])
            check(type(cell['value_decimal']) is str and type(cell['scale']) is str, 'Exact decimal strings required')
            check(scale == decimal(SCALES[event['symbol']]) and cell['scale'] == context['fields'][name]['scale'] and
                  (name == 'annual_net_income' or amount > 0), 'Scale or nonpositive assets')
            check(cell['currency_literal'] == '$' and 'currency_iso' in cell and cell['currency_iso'] is None, 'Uncertified currency change')
            check(start is not None and end is not None, 'Annual endpoints unavailable')
            expected = start-timedelta(days=1) if name == 'prior_total_assets' else end
            check(day(cell['as_of']) == expected, 'Cell endpoint mismatch')
            check(cell.get('period') == (event['periods']['current'] if name == 'annual_net_income' else None), 'Cell period mismatch')
        except (KeyError, TypeError, ValueError, InvalidOperation):
            reasons.append('invalid_'+name)
    identity = deepcopy(event.get('identity', {}))
    check(isinstance(identity, dict) and not any(k in identity for k in ('issuer_id','security_id','identity_valid_from','identity_valid_through')),
          'Prior use-window identity cannot be overwritten')
    check(identity.get('issuer') == ISSUERS[event['symbol']] and identity.get('symbol') == event['symbol'] and
          identity.get('source_anchors') == context['identity_anchors'] and
          identity.get('certified_daily_security_master') is False, 'Report-local identity evidence changed')
    identity.update(issuer_id=reviewed['issuer_id'], security_id=event['symbol'], identity_valid_from=available,
                    identity_valid_through=(end+timedelta(days=MAX_AGE)).isoformat() if end else None,
                    scope='source_local_report_binding_and_use_window_not_continuous_security_master', certified_daily_security_master=False)
    result['identity'] = identity
    result['ni_assets_admission'] = dict(admitted=not reasons, reasons=list(dict.fromkeys(reasons)), definition=DEFINITION)
    result['source_review'] = dict(event_id=reviewed['event_id'], metadata_verified=True, all_three_positions_reviewed=True,
        boundary_assumption_verified=boundary_ok, accounting_review=deepcopy(reviewed.get('accounting_review')),
        original_context_holds_preserved=True)
    result['clock_review'] = dict(first_pinned_session=derived, verified=clock_ok)
    return result


def annual_catalog(census, contexts, sessions):
    rows = [deepcopy(x) for x in census['annual_documents'] if x['symbol'] in SYMBOLS]
    check(len(rows) == len({x['event_id'] for x in rows}) == 28 and Counter(x['symbol'] for x in rows)=={'PLTR':6,'PYPL':11,'TMO':11}, 'Complete28 catalog required')
    check(set(contexts) <= {x['event_id'] for x in rows}, 'Context outside full catalog')
    for row in rows:
        check(row['fiscal_quarter'] == 4, 'Nonannual catalog event')
        derived = first_session(row['publication_date'],sessions)
        advertised = row.get('available_session')
        check(advertised is None or day(advertised)>day(row['publication_date']), 'Invalid catalog time')
        row['clock_review'] = dict(first_pinned_session=derived, verified=derived is not None and derived==advertised)
        row['processed_in_batch'] = row['event_id'] in contexts
    return rows


def coverage(meta,census,catalog,events,strict=True):
    result = old.coverage(meta,census,catalog,events,strict)
    by_id = {e['event_id']:e for e in events}
    for row in result['monthly_candidate_coverage']:
        signal,symbol = row['session'],row['symbol']
        detail = select_latest(catalog,by_id,symbol,signal) if symbol in SYMBOLS else dict(
            available=False,reason='outside_new_batch_prior_evidence_preserved',event_id=None)
        row.clear();row.update(session=signal,symbol=symbol,**detail)
    rows=result['monthly_candidate_coverage']
    result['scope']='incremental_batch5_source_availability_not_strategy_eligibility'
    result['counts'].update(conditional_new_slots=sum(x['available'] for x in rows),reasons=dict(Counter(x['reason'] for x in rows)))
    result['prior_twelve_issuer_coverage']='Unchanged and separately referenced; no prior gate is reopened.'
    return result


def preserved_references(evidence):
    original,batch1,batch2,batch3 = previous.preserved_references(evidence)
    manifest=evidence.read(BATCH4+'/MANIFEST.json');evidence.pins(manifest['input_hashes'])
    check(set(manifest['outputs'])=={'DATASET.json','COVERAGE.json','ADMISSION.json'},'Batch4 output set changed')
    package={name:evidence.read(BATCH4+'/'+name,h) for name,h in manifest['outputs'].items()}
    data,adm=package['DATASET.json'],package['ADMISSION.json'];evidence.pins(adm['input_hashes'])
    check(adm.get('passed') is True and data.get('method')==METHOD,'Batch4 source package not accepted')
    check(data['original_four_issuer_reference']==original and data['prior_batch1_reference']==batch1
          and data['prior_batch2_reference']==batch2 and data['prior_batch3_reference']==batch3, 'Prior chain differs')
    check(len(data['events'])==len({e['event_id'] for e in data['events']})==9 and
          Counter(e['symbol'] for e in data['events'])=={'MU':2,'PEP':7} and len(data['latest_event_catalog'])==23,
          'Batch4 population/catalog changed')
    check(all(type(e['ni_assets_admission']['admitted']) is bool for e in data['events']) and
          sum(e['ni_assets_admission']['admitted'] for e in data['events'])==5 and
          [adm['counts'][k] for k in ('events','admitted','held')]==[9,5,4], 'Batch4 gate counts changed')
    check(package['COVERAGE.json']['counts']['stock_slots']==3904,'Batch4 stock axis changed')
    ref=dict(path=BATCH4+'/DATASET.json',sha256=evidence.pin(BATCH4+'/DATASET.json'),
             admission_path=BATCH4+'/ADMISSION.json',admission_sha256=evidence.pin(BATCH4+'/ADMISSION.json'),
             manifest_path=BATCH4+'/MANIFEST.json',manifest_sha256=evidence.pin(BATCH4+'/MANIFEST.json'),
             coverage_path=BATCH4+'/COVERAGE.json',coverage_sha256=evidence.pin(BATCH4+'/COVERAGE.json'),
             events=9,admitted=5,held=4,annual_catalog_events=23,
             records_canonical_sha256=hashlib.sha256(canonical(data['events'])).hexdigest(),
             catalog_canonical_sha256=hashlib.sha256(canonical(data['latest_event_catalog'])).hexdigest(),preserved_without_modification=True)
    return original,batch1,batch2,batch3,ref


def build(root,plan_sha,extraction_sha,review_sha,policy_sha):
    check(all(re.fullmatch(r'[0-9a-f]{64}',x or '') for x in (plan_sha,extraction_sha,review_sha,policy_sha)), 'Four explicit hashes required')
    evidence=Evidence(root)
    plan=evidence.read(BATCH+'/PLAN.json',plan_sha);evidence.pins(plan['inputs']);evidence.pins(plan['code'])
    policy=evidence.read(POLICY,policy_sha);policy_contract(policy);evidence.pins(policy['input_hashes'])
    _bound(evidence,plan['inputs'],POLICY,policy_sha)
    for name in (BATCH4+'/DATASET.json',BATCH4+'/ADMISSION.json',BATCH4+'/MANIFEST.json'):
        _bound(evidence,plan['inputs'],name,evidence.pin(name))
    context=evidence.read(CONTEXT);cref=dict(path=CONTEXT,sha256=evidence.pin(CONTEXT))
    check(plan['context']==CONTEXT and plan['policy']==POLICY, 'Context/policy PLAN linkage')
    _bound(evidence,plan['inputs'],CONTEXT,cref['sha256'])
    _bound(evidence,policy['input_hashes'],CONTEXT,cref['sha256'])
    check(plan.get('stage')=='frozen_source_only_plan' and plan.get('no_admission_by_extraction') is True
          and plan.get('new_strategy_trials')==0, 'Wrong source-only PLAN contract')
    cmap=context_map(context)
    extraction=evidence.read(BATCH+'/EXTRACTIONS.json',extraction_sha)
    check(extraction['plan_sha256']==plan_sha and extraction['catalog_ref']==cref and
          extraction['status']=='source_values_extracted_pending_independent_review','Extraction linkage')
    for k in ('new_financial_admissions','new_strategy_trials','ratios_computed','returns_computed'):
        check(type(extraction.get(k)) is int and extraction[k]==0,'Premature financial processing')
    events=extraction['events'];catalog_ids=[e['event_id'] for e in context['events']]
    check([e['event_id'] for e in events]==list(cmap)==[r['event_id'] for r in plan['records']], 'Exact21 extraction order/population')
    check(plan['catalog_ids']==extraction['catalog_ids']==catalog_ids and plan['catalog_events']==28,'Full28 catalog missing')
    review=evidence.read(REVIEW,review_sha)
    check(review.get('stage')=='independent_source_value_and_null_classification_review','Wrong review stage')
    for k in ('financial_calculations','factor_calculations','returns_computed','new_admissions'):
        check(type(review.get(k)) is int and review[k]==0,'Review exceeded source-only scope')
    evidence.pins(review['input_hashes']);evidence.pins(review['source_hashes'])
    for name in ('work/after153_batch5_independent_review.py','work/test_after153_batch5_independent_review.py'):
        _bound(evidence,review['source_hashes'],name,evidence.pin(name))
    for name,h in review['source_hashes'].items():_bound(evidence,review['input_hashes'],name,h)
    for name in (BATCH+'/PLAN.json',BATCH+'/EXTRACTIONS.json',POLICY,CONTEXT,SESSIONS,ACCOUNTING):
        _bound(evidence,review['input_hashes'],name,evidence.pin(name))
    accounting=evidence.read(ACCOUNTING)
    evidence.pins(accounting['input_hashes'])
    check(accounting.get('stage')=='independent_field_specific_accounting_review' and accounting.get('passed') is True and
          accounting.get('plan_sha256')==plan_sha and accounting.get('extractions_sha256')==extraction_sha, 'Accounting report linkage')
    accounting_rows={r['event_id']:r for r in accounting['events']}
    check(len(accounting_rows)==len(accounting['events'])==len(events) and set(accounting_rows)==set(cmap),'Accounting event census differs')
    reviewed=verify_independent_review(review,events,plan_sha,extraction_sha)
    for event in events:
        check(reviewed[event['event_id']].get('accounting_review')==accounting_rows[event['event_id']],
              'Detached accounting decision')
        source=cmap[event['event_id']]['source']
        for name,h in ((source['path'],source['sha256']),(source['receipt_path'],source['receipt_sha256'])):
            evidence.pin(name,h);_bound(evidence,review['input_hashes'],name,h)
    sessions=normalize_sessions(evidence.read(SESSIONS))
    admitted=[assess_event(e,cmap[e['event_id']],reviewed[e['event_id']],sessions) for e in events]
    census,meta=evidence.read(CENSUS),evidence.read(META)
    catalog=annual_catalog(census,cmap,sessions)
    raw_catalog=[r for r in census['annual_documents'] if r['symbol'] in SYMBOLS]
    check([r['event_id'] for r in raw_catalog]==catalog_ids,'Catalog order mismatch')
    projection=catalog_projection(context)
    check(policy['fixed_catalog_metadata']==projection==plan['catalog_metadata']==extraction['catalog_metadata'],
          'Policy/PLAN/extraction catalog differs from context')
    check(plan['original_catalog_metadata']==extraction['original_catalog_metadata']==raw_catalog,
          'Full original catalog metadata changed')
    for raw,ctx in zip(raw_catalog,context['events']):
        validate_census_context(raw,ctx)
    original,batch1,batch2,batch3,batch4=preserved_references(evidence)
    dataset=dict(schema_version=1,method=METHOD,policy_sha256=policy_sha,source_plan_sha256=plan_sha,
                 independent_review_sha256=review_sha,events=admitted,latest_event_catalog=catalog,
                 original_four_issuer_reference=original,prior_batch1_reference=batch1,prior_batch2_reference=batch2,prior_batch3_reference=batch3,prior_batch4_reference=batch4,
                 conditional_research_only=True,first_public_version_verified=False,first_public_time_verified=False,complete_correction_history_verified=False,
                 certified_daily_security_master=False,scores_computed=0,ratios_computed=0,returns_computed=0,new_strategy_trials=0)
    monthly=coverage(meta,census,catalog,admitted)
    for name in source_files():evidence.pin(name)
    evidence.stable()
    counts=dict(events=21,admitted=sum(e['ni_assets_admission']['admitted'] for e in admitted),
                held=sum(not e['ni_assets_admission']['admitted'] for e in admitted),
                reviewed_nonnull_fields=sum(e['fields'][f] is not None for e in events for f in FIELDS),
                reviewed_null_fields=sum(e['fields'][f] is None for e in events for f in FIELDS),
                annual_catalog_events=28,unprocessed_annual_catalog_events=7,
                original_events=44,original_admitted=31,original_held=13,
                prior_batch1_events=26,prior_batch1_admitted=17,prior_batch1_held=9,
                prior_batch2_events=19,prior_batch2_admitted=14,prior_batch2_held=5,
                prior_batch3_events=10,prior_batch3_admitted=5,prior_batch3_held=5,
                prior_batch4_events=9,prior_batch4_admitted=5,prior_batch4_held=4)
    admission=dict(schema_version=1,stage='conditional_source_admission_not_strategy_registration',passed=True,
                   created_utc=datetime.now(timezone.utc).isoformat(),policy_sha256=policy_sha,source_plan_sha256=plan_sha,
                   independent_review_sha256=review_sha,counts=counts,input_hashes=dict(evidence.seen),
                   numerical_work='Finiteness/sign/period checks only; no sum of assets, ratio, score or return.',
                   actual_factors_calculated=0,actual_returns_calculated=0,new_strategy_trials=0)
    return {'DATASET.json':dataset,'COVERAGE.json':monthly,'ADMISSION.json':admission},evidence


def publish(root,objects,evidence):
    directory=Path(root)/OUT;directory.mkdir(parents=True,exist_ok=True)
    check(set(objects)=={'DATASET.json','COVERAGE.json','ADMISSION.json'},'Unexpected output set')
    check(not any((directory/name).exists() for name in list(objects)+['MANIFEST.json']),'Write-once output exists')
    evidence.stable()
    for name,obj in objects.items():
        with (directory/name).open('xb') as stream:
            stream.write((json.dumps(obj,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode())
    evidence.stable()
    manifest=dict(schema_version=1,input_hashes=dict(evidence.seen),outputs={name:digest(directory/name) for name in objects},
                  status='conditional_source_package_no_strategy_registration')
    with (directory/'MANIFEST.json').open('xb') as stream:
        stream.write((json.dumps(manifest,sort_keys=True,indent=2,allow_nan=False)+'\n').encode())
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','extractions','review','policy'):parser.add_argument('--'+name+'-sha',required=True)
    args=parser.parse_args()
    objects,evidence=build(ROOT,args.plan_sha,args.extractions_sha,args.review_sha,args.policy_sha)
    manifest=publish(ROOT,objects,evidence)
    print(json.dumps(dict(status='conditional_source_package_complete',counts=objects['ADMISSION.json']['counts'],outputs=manifest['outputs']),sort_keys=True))


if __name__=='__main__':main()
