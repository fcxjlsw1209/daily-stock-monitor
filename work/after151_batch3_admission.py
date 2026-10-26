"""Offline conditional KO/LRCX source admission; no ratios, models or returns.

The three previous source packages remain immutable references. Full annual
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

import after150_batch2_admission as previous
import after149_batch1_admission as old

ROOT = Path(__file__).resolve().parents[1]
OUT = 'outputs/monthly_financial_coverage_admission_batch3_v1'
BATCH = 'outputs/monthly_financial_coverage_batch3_v1'
BATCH2, BATCH1 = previous.OUT, old.OUT
POLICY = 'work/after151_batch3_admission_policy.json'
REVIEW = 'work/after151_batch3_independent_review.json'
ACCOUNTING = 'work/after151_batch3_accounting_review.json'
CONTEXT = 'work/after151_next_source_context.json'
STATUS = 'standard_HTML_annual_header_context_located_not_admitted'
CENSUS, META, SESSIONS = old.CENSUS, old.META, old.SESSIONS
OLD_DATA, OLD_ADMISSION = old.OLD_DATA, old.OLD_ADMISSION
FIELDS, METHOD, DEFINITION = old.FIELDS, old.METHOD, old.DEFINITION
SYMBOLS = frozenset(('KO', 'LRCX'))
ASSUMPTIONS = {'KO': 'KO_calendar_annual_boundaries_v1', 'LRCX': 'LRCX_consecutive_annual_boundaries_v1'}
ISSUERS = {'KO': 'The Coca-Cola Company', 'LRCX': 'Lam Research Corporation'}
ACCOUNTING_SCOPE = 'current_annual_net_income_same_report_assets'
MAX_AGE = 550
check, day, decimal, canonical, digest = old.check, old.day, old.decimal, old.canonical, old.digest
normalize_sessions, first_session, select_latest = old.normalize_sessions, old.first_session, old.select_latest
Evidence = previous.Evidence
_bound = previous._bound


def source_files():
    return ('work/after151_batch3_admission.py', 'work/test_after151_batch3_admission.py',
            'work/after150_batch2_admission.py', 'work/after149_batch1_admission.py')


def policy_contract(policy):
    check(policy.get('schema_version') == 1 and policy.get('policy_id') == 'ko_lrcx_annual_ni_same_report_assets_batch3_v1'
          and policy.get('stage') == 'prospective_source_policy_not_strategy_registration', 'Wrong source policy')
    check(policy.get('definition') == DEFINITION and policy.get('required_fields') == list(FIELDS), 'Measurement changed')
    scope = policy.get('source_scope', {})
    check(scope.get('symbols') == ['KO','LRCX'] and scope.get('catalog_events') == 32
          and scope.get('html_context_reviewed_events') == 10 and scope.get('no_automatic_admission') is True,
          'Fixed source scope changed')
    check(policy.get('known_time', {}).get('maximum_age_days') == MAX_AGE, 'Age limit changed')
    boundary = policy.get('periods', {}).get('LRCX_52_53_week_assumption', {})
    check(boundary.get('enabled') is True and boundary.get('not_a_quoted_fact') is True
          and boundary.get('assumption_id') == ASSUMPTIONS['LRCX'], 'Explicit LRCX boundary policy required')
    for name in ('first_public_version_verified','complete_correction_history_verified','certified_daily_security_master'):
        check(policy.get('preservation', {}).get(name) is False, 'Unsupported certification')
    check(policy.get('preservation', {}).get('all_context_holds') is True, 'Context holds must be preserved')


def context_map(context):
    docs = context['events']
    check(len(docs) == len({r['event_id'] for r in docs}) == 32 and
          Counter(r['symbol'] for r in docs) == {'KO':16,'LRCX':16}, 'Full32 context required')
    chosen = [r for r in docs if r.get('context_status') == STATUS]
    check([r['fiscal_year'] for r in chosen if r['symbol']=='KO'] == list(range(2011,2017)) and
          [r['fiscal_year'] for r in chosen if r['symbol']=='LRCX'] == list(range(2013,2017)), 'Fixed10 context required')
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
    source = event['source']
    check(source == context['source'] and source['path'] == context['evidence_path'] and source['sha256'] == context['evidence_sha256']
          and source['url'] == context['source_url'], 'Source/context mismatch')
    check(event['context_holds'] == context['context_holds'] and
          event['accounting_context_numbers_redacted'] == context['accounting_context_numbers_redacted'], 'Inherited context changed')
    release, available = event['publication_date'], event['available_from']
    check(release == context['publication_date'] and available == context['available_session'] and day(release) < day(available),
          'Event/context time mismatch')
    derived = first_session(release, sessions)
    clock_ok = derived is not None and derived == available
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
    for name in FIELDS:
        if event['fields'][name] is None:
            reasons.append('missing_'+name)
    start = end = None
    try:
        period = event['fields']['annual_net_income']['period']
        start, end = day(period['period_start']), day(period['period_end'])
        prior = day(context['periods']['prior_end'])
        count = (end-start).days+1
        check(type(period['days_inclusive']) is int and period['days_inclusive'] == count and 330 <= count <= 400,
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
        if event['symbol'] == 'KO':
            check(count in (365,366) and str(start) == f'{year}-01-01' and str(end) == f'{year}-12-31'
                  and str(prior) == f'{year-1}-12-31', 'KO calendar annual mismatch')
        else:
            check(count in (364,371) and end.year == year and prior.year == year-1, 'LRCX annual boundary mismatch')
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
            check(scale == (decimal('1000000') if event['symbol']=='KO' else decimal('1000')) and
                  (name == 'annual_net_income' or amount > 0), 'Scale or nonpositive assets')
            check(cell['currency_literal'] == '$' and cell.get('currency_iso') is None, 'Uncertified currency change')
            check(start is not None and end is not None, 'Annual endpoints unavailable')
            expected = start-timedelta(days=1) if name == 'prior_total_assets' else end
            check(day(cell['as_of']) == expected, 'Cell endpoint mismatch')
            check(cell.get('period') == (event['periods']['current'] if name == 'annual_net_income' else None), 'Cell period mismatch')
        except (KeyError, TypeError, ValueError, InvalidOperation):
            reasons.append('invalid_'+name)
    identity = deepcopy(event.get('identity', {}))
    check(isinstance(identity, dict) and not any(k in identity for k in ('issuer_id','security_id','identity_valid_from','identity_valid_through')),
          'Prior use-window identity cannot be overwritten')
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
    check(len(rows) == len({x['event_id'] for x in rows}) == 32 and Counter(x['symbol'] for x in rows)=={'KO':16,'LRCX':16}, 'Complete32 catalog required')
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
    result['scope']='incremental_batch3_source_availability_not_strategy_eligibility'
    result['counts'].update(conditional_new_slots=sum(x['available'] for x in rows),reasons=dict(Counter(x['reason'] for x in rows)))
    result['prior_eight_issuer_coverage']='Unchanged and separately referenced; no prior gate is reopened.'
    return result


def preserved_references(evidence):
    original,batch1 = previous.preserved_references(evidence)
    manifest=evidence.read(BATCH2+'/MANIFEST.json');evidence.pins(manifest['input_hashes'])
    check(set(manifest['outputs'])=={'DATASET.json','COVERAGE.json','ADMISSION.json'},'Batch2 output set changed')
    package={name:evidence.read(BATCH2+'/'+name,h) for name,h in manifest['outputs'].items()}
    data,adm=package['DATASET.json'],package['ADMISSION.json'];evidence.pins(adm['input_hashes'])
    check(adm.get('passed') is True and data.get('method')==METHOD,'Batch2 source package not accepted')
    check(data['original_four_issuer_reference']==original and data['prior_batch1_reference']==batch1,'Prior chain differs')
    check(len(data['events'])==len({e['event_id'] for e in data['events']})==19 and
          Counter(e['symbol'] for e in data['events'])=={'INTC':3,'KLAC':16} and len(data['latest_event_catalog'])==32,
          'Batch2 population/catalog changed')
    check(all(type(e['ni_assets_admission']['admitted']) is bool for e in data['events']) and
          sum(e['ni_assets_admission']['admitted'] for e in data['events'])==14 and
          [adm['counts'][k] for k in ('events','admitted','held')]==[19,14,5], 'Batch2 gate counts changed')
    check(package['COVERAGE.json']['counts']['stock_slots']==3904,'Batch2 stock axis changed')
    ref=dict(path=BATCH2+'/DATASET.json',sha256=evidence.pin(BATCH2+'/DATASET.json'),
             admission_path=BATCH2+'/ADMISSION.json',admission_sha256=evidence.pin(BATCH2+'/ADMISSION.json'),
             manifest_path=BATCH2+'/MANIFEST.json',manifest_sha256=evidence.pin(BATCH2+'/MANIFEST.json'),
             coverage_path=BATCH2+'/COVERAGE.json',coverage_sha256=evidence.pin(BATCH2+'/COVERAGE.json'),
             events=19,admitted=14,held=5,annual_catalog_events=32,
             records_canonical_sha256=hashlib.sha256(canonical(data['events'])).hexdigest(),
             catalog_canonical_sha256=hashlib.sha256(canonical(data['latest_event_catalog'])).hexdigest(),preserved_without_modification=True)
    return original,batch1,ref


def build(root,plan_sha,extraction_sha,review_sha,policy_sha):
    check(all(re.fullmatch(r'[0-9a-f]{64}',x or '') for x in (plan_sha,extraction_sha,review_sha,policy_sha)), 'Four explicit hashes required')
    evidence=Evidence(root)
    plan=evidence.read(BATCH+'/PLAN.json',plan_sha);evidence.pins(plan['inputs']);evidence.pins(plan['code'])
    policy=evidence.read(POLICY,policy_sha);policy_contract(policy);evidence.pins(policy['input_hashes'])
    _bound(evidence,plan['inputs'],POLICY,policy_sha)
    for name in (BATCH2+'/DATASET.json',BATCH2+'/ADMISSION.json'):
        _bound(evidence,plan['inputs'],name,evidence.pin(name))
    context=evidence.read(CONTEXT);cref=dict(path=CONTEXT,sha256=evidence.pin(CONTEXT))
    check(policy['source_scope']['source_context']==cref==plan['catalog_ref'] and plan['context']==CONTEXT and plan['policy']==POLICY,
          'Context/policy PLAN linkage')
    cmap=context_map(context)
    extraction=evidence.read(BATCH+'/EXTRACTIONS.json',extraction_sha)
    check(extraction['plan_sha256']==plan_sha and extraction['catalog_ref']==cref and
          extraction['status']=='source_values_extracted_pending_independent_review','Extraction linkage')
    for k in ('new_financial_admissions','new_strategy_trials','ratios_computed','returns_computed'):
        check(type(extraction.get(k)) is int and extraction[k]==0,'Premature financial processing')
    events=extraction['events'];catalog_ids=[e['event_id'] for e in context['events']]
    check([e['event_id'] for e in events]==list(cmap)==[r['event_id'] for r in plan['records']], 'Exact10 extraction order/population')
    check(plan['catalog_ids']==extraction['catalog_ids']==catalog_ids and plan['catalog_events']==32,'Full32 catalog missing')
    review=evidence.read(REVIEW,review_sha)
    check(review.get('stage')=='independent_source_value_and_null_classification_review','Wrong review stage')
    for k in ('financial_calculations','factor_calculations','returns_computed','new_admissions'):
        check(type(review.get(k)) is int and review[k]==0,'Review exceeded source-only scope')
    evidence.pins(review['input_hashes']);evidence.pins(review['source_hashes'])
    for name in ('work/after151_batch3_independent_review.py','work/test_after151_batch3_independent_review.py'):
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
    projection=[{k:r[k] for k in ('event_id','symbol','fiscal_year','publication_date','available_session','official_actual_period_end','source','context_status')}
                for r in context['events']]
    check(policy['fixed_catalog_metadata']==projection,'Policy catalog differs from context')
    for raw,ctx in zip(raw_catalog,context['events']):
        check(all(ctx.get(k)==v for k,v in raw.items()),'Frozen census/context metadata mismatch')
    original,batch1,batch2=preserved_references(evidence)
    dataset=dict(schema_version=1,method=METHOD,policy_sha256=policy_sha,source_plan_sha256=plan_sha,
                 independent_review_sha256=review_sha,events=admitted,latest_event_catalog=catalog,
                 original_four_issuer_reference=original,prior_batch1_reference=batch1,prior_batch2_reference=batch2,
                 conditional_research_only=True,first_public_version_verified=False,complete_correction_history_verified=False,
                 certified_daily_security_master=False,scores_computed=0,ratios_computed=0,returns_computed=0,new_strategy_trials=0)
    monthly=coverage(meta,census,catalog,admitted)
    for name in source_files():evidence.pin(name)
    evidence.stable()
    counts=dict(events=10,admitted=sum(e['ni_assets_admission']['admitted'] for e in admitted),
                held=sum(not e['ni_assets_admission']['admitted'] for e in admitted),
                reviewed_nonnull_fields=sum(e['fields'][f] is not None for e in events for f in FIELDS),
                reviewed_null_fields=sum(e['fields'][f] is None for e in events for f in FIELDS),
                annual_catalog_events=32,unprocessed_annual_catalog_events=22,
                original_events=44,original_admitted=31,original_held=13,
                prior_batch1_events=26,prior_batch1_admitted=17,prior_batch1_held=9,
                prior_batch2_events=19,prior_batch2_admitted=14,prior_batch2_held=5)
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
