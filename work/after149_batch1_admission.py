"""Conditional NI/assets source admission and census coverage; no ratios or returns.

Build is write-once, requires explicit hashes and a successful independent
source review. Existing four-issuer evidence is referenced, never rewritten.
"""
import argparse
from bisect import bisect_right
from collections import Counter
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = 'outputs/monthly_financial_coverage_admission_batch1_v1'
BATCH = 'outputs/monthly_financial_coverage_batch1_v1'
POLICY = 'work/after149_batch1_admission_policy.json'
REVIEW = 'work/after149_batch1_independent_review.json'
CENSUS = 'outputs/monthly_financial_coverage_census_v1/CENSUS.json'
META = 'outputs/monthly_local_linear70/META.json'
SESSIONS = 'outputs/monthly_forward_guidance_preflight_v1/SESSIONS.json'
OLD_DATA = 'outputs/monthly_stock_comp_source_admission_v2/DATASET.json'
OLD_ADMISSION = 'outputs/monthly_stock_comp_source_admission_v2/ADMISSION.json'
CONTEXTS = {'AMZN': 'work/after149_amzn_source_context.json',
            'CSCO': 'work/after149_csco_source_context.json'}
FIELDS = ('annual_net_income', 'current_total_assets', 'prior_total_assets')
METHOD = 'expanded_annual_earnings_coverage_v1'
DEFINITION = 'current_annual_net_income_average_assets_v1'
SYMBOLS = frozenset(CONTEXTS)
MAX_AGE = 550


def check(ok, reason):
    if not ok:
        raise ValueError(reason)


def day(value):
    check(type(value) is str, 'Date string required')
    result = date.fromisoformat(value)
    check(result.isoformat() == value, 'Canonical date required')
    return result


def decimal(value):
    check(type(value) is str and value.strip() == value and value, 'Exact decimal string required')
    result = Decimal(value)
    check(result.is_finite(), 'Finite decimal required')
    return result


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode()


class Evidence:
    """All consumed bindings are verified locally and rechecked before publish."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.seen = {}

    def path(self, name):
        p = Path(name)
        p = p if p.is_absolute() else self.root / p
        check(not p.is_symlink(), 'Symlink source rejected')
        p = p.resolve()
        p.relative_to(self.root)
        check('CloudStorage' not in p.parts, 'Cloud source rejected')
        return p

    def pin(self, name, expected=None):
        p = self.path(name)
        actual = digest(p)
        check(expected is None or actual == expected, 'Hash mismatch: ' + str(name))
        key = str(p.relative_to(self.root))
        check(key not in self.seen or self.seen[key] == actual, 'Source changed: ' + key)
        self.seen[key] = actual
        return actual

    def read(self, name, expected=None):
        self.pin(name, expected)
        return json.loads(self.path(name).read_bytes())

    def pins(self, mapping):
        check(isinstance(mapping, dict) and mapping, 'Nonempty hash bindings required')
        for name, expected in mapping.items():
            check(type(expected) is str and len(expected) == 64, 'SHA256 binding required')
            self.pin(name, expected)

    def stable(self):
        for name, expected in list(self.seen.items()):
            self.pin(name, expected)


def policy_contract(policy):
    check(policy.get('schema_version') == 1 and
          policy.get('stage') == 'prospective_source_policy_not_strategy_registration',
          'Prospective source policy required')
    check(policy.get('required_fields') == list(FIELDS), 'Three exact source fields required')
    check(policy.get('source_scope', {}).get('symbols') == ['AMZN', 'CSCO'], 'Source scope changed')
    check(policy['source_scope'].get('no_automatic_admission') is True, 'Independent admission required')
    check(policy.get('known_time', {}).get('maximum_age_days') == MAX_AGE, 'Age limit changed')
    for name in ('first_public_version_verified', 'complete_correction_history_verified',
                 'certified_daily_security_master'):
        check(policy.get('preservation', {}).get(name) is False, 'Unproven source certification')
    for name in ('AMZN_FY2019', 'CSCO_FY2019'):
        check(str(policy.get('accounting', {}).get(name, '')).startswith('Hold '),
              'Required accounting hold removed')


def normalize_sessions(sessions):
    values = sessions['sessions'] if isinstance(sessions, dict) else sessions
    parsed = [day(value) for value in values]
    check(parsed and parsed == sorted(set(parsed)), 'Unique sorted pinned sessions required')
    return parsed


def first_session(released, sessions):
    """No extrapolation outside the pinned calendar; early dates remain unknown."""
    released = day(released)
    if released < sessions[0] or released >= sessions[-1]:
        return None
    index = bisect_right(sessions, released)
    return sessions[index].isoformat() if index < len(sessions) else None


def context_map(contexts):
    result = {}
    for symbol in SYMBOLS:
        document = contexts[symbol]
        check(document.get('symbol') == symbol, 'Context issuer mismatch')
        records = document['documents'] if symbol == 'AMZN' else document['events']
        cik = document['identity']['cik']
        check(type(cik) is str and len(cik) == 10 and cik.isdigit(), 'Reviewed CIK required')
        for row in records:
            check(row['event_id'] not in result, 'Duplicate context event')
            result[row['event_id']] = (symbol, cik, row)
    return result


def verify_independent_review(review, events, plan_sha, extraction_sha):
    check(review.get('passed') is True, 'Independent source review did not pass')
    check(review.get('plan_sha256') == plan_sha and
          review.get('extractions_sha256') == extraction_sha, 'Independent review binding mismatch')
    rows = review.get('event_reviews')
    check(isinstance(rows, list), 'Per-event source reviews required')
    reviewed = {row['event_id']: row for row in rows}
    check(len(reviewed) == len(rows) == len(events) and
          set(reviewed) == {event['event_id'] for event in events}, 'Review event population mismatch')
    for event in events:
        row = reviewed[event['event_id']]
        check(row.get('passed') is True, 'Event source review failed')
        check(row.get('metadata_verified') is True, 'Independent metadata review required')
        check(set(row.get('fields', {})) == set(FIELDS), 'Three field reviews required')
        for name in FIELDS:
            cell = row['fields'][name]
            missing = event['fields'][name] is None
            check(cell.get('is_null') is missing, 'Review null status mismatch')
            if missing:
                check(cell.get('null_reason_verified') is True, 'Missing field cause not reviewed')
            else:
                check(cell.get('verified') is True and cell.get('source_value_present') is True,
                      'Source cell not independently verified')
    return reviewed


def assess_event(event, context_item, reviewed, sessions):
    """Inspect source eligibility only. Numeric operations are finiteness/sign tests."""
    result = deepcopy(event)
    check(not any(key in event for key in ('ni_assets_admission', 'identity')), 'New gate/identity already present')
    symbol, cik, context = context_item
    check(symbol == event.get('symbol') and event['event_id'] == context['event_id'], 'Context identity mismatch')
    check(event['fiscal_year'] == context['fiscal_year'], 'Fiscal year mismatch')
    check(set(event.get('fields', {})) == set(FIELDS), 'Exact three-field extraction required')
    for flag in ('fact_admitted', 'model_admitted', 'strategy_admitted'):
        check(event.get(flag) is False, 'Source extraction certification changed')
    source = event['source']
    expected_source = context if symbol == 'AMZN' else context['source']
    check(source['path'] == expected_source['path'] and source['sha256'] == expected_source['sha256'],
          'Event/context source mismatch')
    release = event.get('release_date', event.get('publication_date'))
    context_release = context.get('release_date', context.get('publication_date'))
    check(release == context_release and event.get('available_from') == context['available_session'],
          'Event/context clock mismatch')
    released = day(release)
    advertised = event['available_from']
    check(day(advertised) > released, 'Availability must follow release')
    derived = first_session(release, sessions)
    reasons = []
    if derived is None:
        reasons.append('unverified_exchange_calendar')
    elif advertised != derived:
        reasons.append('availability_not_first_pinned_session')
    if int(event['fiscal_year']) == 2019:
        reasons.append('AMZN_FY2019_lease_transition_unresolved' if symbol == 'AMZN' else
                       'CSCO_FY2019_ASC606_transition_unresolved')
    for name in FIELDS:
        if event['fields'][name] is None:
            reasons.append('missing_' + name)
    period = (event['fields'].get('annual_net_income') or {}).get('period')
    start = end = None
    try:
        check(isinstance(period, dict), 'Current annual period missing')
        start, end = day(period['period_start']), day(period['period_end'])
        duration = (end - start).days + 1
        check(330 <= duration <= 400 and period['days_inclusive'] == duration,
              'Current period not explicit annual')
        check(end <= released, 'Publication before period end')
        expected_end = context['current_end'] if symbol == 'AMZN' else context['periods']['current']['period_end']
        expected_prior = context['comparison_end'] if symbol == 'AMZN' else context['periods']['comparison']['period_end']
        check(end.isoformat() == expected_end and (start-timedelta(days=1)).isoformat() == expected_prior,
              'Current/prior context endpoints mismatch')
        if symbol == 'CSCO':
            check(period == context['periods']['current'], 'Context current annual period mismatch')
    except (KeyError, TypeError, ValueError):
        reasons.append('invalid_current_annual_period')
    currencies = set()
    for name in FIELDS:
        cell = event['fields'][name]
        if cell is None:
            continue
        try:
            check(cell['source']['path'] == source['path'] and cell['source']['sha256'] == source['sha256'],
                  'Cell source mismatch')
            check(cell['available_from'] == advertised, 'Cell availability mismatch')
            check(cell.get('fact_admitted') is False, 'Premature cell admission')
            check(type(cell.get('displayed_amount')) is str and cell['displayed_amount'].strip(),
                  'Displayed amount missing')
            check(isinstance(cell.get('locator'), dict) and cell['locator'], 'Cell locator missing')
            value, scale = decimal(cell['value_decimal']), decimal(cell['scale'])
            check(scale > 0, 'Positive declared scale required')
            if name != 'annual_net_income':
                check(value > 0, 'Positive total assets required')
            check(cell['currency_literal'] == '$', 'Source-local dollar required')
            iso = cell.get('currency_iso')
            check(iso is None or type(iso) is str and len(iso) == 3 and iso.isupper(), 'Invalid currency ISO')
            currencies.add(iso)
            check(start is not None and end is not None, 'No valid annual endpoints')
            expected = start-timedelta(days=1) if name == 'prior_total_assets' else end
            check(day(cell['as_of']) == expected, 'Wrong cell instant')
        except (KeyError, TypeError, ValueError, InvalidOperation):
            reasons.append('invalid_' + name)
    if len(currencies) > 1:
        reasons.append('inconsistent_currency_iso')
    result['identity'] = dict(issuer_id=cik, security_id=symbol,
                              identity_valid_from=advertised,
                              identity_valid_through=(end+timedelta(days=MAX_AGE)).isoformat() if end else None,
                              scope='source_local_report_binding_and_use_window_not_continuous_security_master',
                              certified_daily_security_master=False)
    result['ni_assets_admission'] = dict(admitted=not reasons, reasons=reasons, definition=DEFINITION)
    result['source_review'] = dict(event_id=reviewed['event_id'], metadata_verified=True,
                                   all_three_positions_reviewed=True)
    result['clock_review'] = dict(first_pinned_session=derived, verified=derived == advertised)
    return result


def annual_catalog(census, contexts, sessions):
    """Retain all source catalog metadata, including every unprocessed release."""
    wanted = [row for row in census['annual_documents'] if row.get('symbol') in SYMBOLS]
    check(len({row['event_id'] for row in wanted}) == len(wanted), 'Duplicate annual catalog event')
    result = []
    for row in wanted:
        check(row.get('fiscal_quarter') == 4, 'Nonannual catalog event')
        copied = deepcopy(row)
        derived = first_session(row['publication_date'], sessions)
        advertised = row.get('available_session')
        if advertised is not None:
            check(day(advertised) > day(row['publication_date']), 'Invalid catalog availability')
        copied['clock_review'] = dict(first_pinned_session=derived,
                                      verified=derived is not None and derived == advertised)
        copied['processed_in_batch'] = row['event_id'] in contexts
        result.append(copied)
    check(set(contexts) <= {row['event_id'] for row in result}, 'Extracted event outside annual catalog')
    return result


def select_latest(catalog, events, symbol, signal):
    """Latest publication knowledge first, then completeness/clock/age. No fallback."""
    signal_day = day(signal)
    known = []
    for row in catalog:
        if row['symbol'] != symbol:
            continue
        release = day(row['publication_date'])
        advertised = row.get('available_session')
        clock = row['clock_review']
        if clock['verified']:
            if day(advertised) <= signal_day:
                known.append((release, row))
        elif release < signal_day:
            # Unknown clock must never silently disappear in favor of an old event.
            known.append((release, row))
    if not known:
        return dict(available=False, reason='no_known_annual_event', event_id=None)
    last = max(item[0] for item in known)
    tied = [row for released, row in known if released == last]
    if len(tied) != 1:
        return dict(available=False, reason='ambiguous_latest_annual_event', event_id=None)
    selected = tied[0]
    event_id = selected['event_id']
    result = dict(available=False, reason=None, event_id=event_id)
    if not selected['clock_review']['verified']:
        result['reason'] = 'latest_annual_clock_unverified'
    elif event_id not in events:
        result['reason'] = 'latest_annual_event_unprocessed'
    else:
        event = events[event_id]
        check(event['symbol'] == symbol and event['available_from'] == selected['available_session'],
              'Catalog/event binding mismatch')
        gate = event['ni_assets_admission']
        if not gate['admitted']:
            result['reason'] = 'latest_annual_event_held'
            result['holds'] = deepcopy(gate['reasons'])
        else:
            end = day(event['fields']['annual_net_income']['period']['period_end'])
            if (signal_day - end).days > MAX_AGE:
                result['reason'] = 'latest_annual_event_stale'
            elif not day(event['identity']['identity_valid_from']) <= signal_day <= day(event['identity']['identity_valid_through']):
                result['reason'] = 'latest_annual_identity_window_invalid'
            else:
                result.update(available=True, reason='conditionally_source_available')
    return result


def coverage(meta, census, catalog, events, strict=True):
    dates, panels, etfs = meta['dates'], meta['symbols'], set(meta['etfs'])
    check(len(dates) == len(panels) and [day(x) for x in dates] == sorted({day(x) for x in dates}),
          'Sorted unique monthly panels required')
    expected_slots = [(row['session'], row['symbol']) for row in census['monthly_candidate_coverage']]
    actual_slots = []
    for signal, names in zip(dates, panels):
        check(len(names) == len(set(names)) and etfs <= set(names), 'Panel duplicates or missing ETF')
        if strict:
            check(len(names) == 49 and len(etfs) == 17, 'Original 32+17 panel required')
        actual_slots.extend((signal, symbol) for symbol in names if symbol not in etfs)
    check(len(expected_slots) == len(set(expected_slots)) and
          len(actual_slots) == len(set(actual_slots)) and
          set(actual_slots) == set(expected_slots),
          'All unique census stock slot identities must be preserved')
    all_stocks = {symbol for _, symbol in actual_slots}
    check(all_stocks == {row['symbol'] for row in census['stocks']}, 'Stock census population changed')
    if strict:
        check(len(dates) == 122 and len(actual_slots) == 3904 and len(all_stocks) == 73,
              'Original 122/3904/73 population required')
    by_id = {event['event_id']: event for event in events}
    check(len(by_id) == len(events), 'Duplicate admitted event')
    rows = []
    for signal, symbol in actual_slots:
        detail = (select_latest(catalog, by_id, symbol, signal) if symbol in SYMBOLS else
                  dict(available=False, reason='outside_new_batch_original_evidence_preserved', event_id=None))
        rows.append(dict(session=signal, symbol=symbol, **detail))
    return dict(schema_version=1, scope='incremental_batch1_source_availability_not_strategy_eligibility',
                axis_order='Original META chronological panels then original stock column order; census identity set unchanged',
                counts=dict(months=len(dates), stocks=len(all_stocks), stock_slots=len(rows),
                            conditional_new_slots=sum(row['available'] for row in rows),
                            reasons=dict(Counter(row['reason'] for row in rows))),
                stocks=sorted(all_stocks), etfs=sorted(etfs), monthly_candidate_coverage=rows,
                original_four_issuer_coverage='Unchanged and separately referenced; false outside this batch is not exclusion.',
                scores_computed=0, ratios_computed=0, returns_computed=0, new_strategy_trials=0)


def old_population(old):
    records = old['events']
    check(len(records) == 44 and len({row['event_id'] for row in records}) == 44,
          'Original 44-event population required')
    check({row['symbol'] for row in records} == {'AAPL', 'AMD', 'MSFT', 'NVDA'}, 'Original issuers changed')
    gates = [row['stock_compensation_admission'] for row in records]
    check(all(type(gate.get('admitted')) is bool for gate in gates), 'Original admission missing')
    check(sum(gate['admitted'] for gate in gates) == 31, 'Original 31/13 population changed')
    return dict(events=44, admitted=31, held=13,
                records_canonical_sha256=hashlib.sha256(canonical(records)).hexdigest())


def build(root, plan_sha, review_sha, policy_sha):
    evidence = Evidence(root)
    plan = evidence.read(BATCH+'/PLAN.json', plan_sha)
    evidence.pins(plan['inputs'])
    evidence.pins(plan['code'])
    policy = evidence.read(POLICY, policy_sha)
    check(plan['inputs'].get(POLICY) == policy_sha, 'Policy not frozen in extraction plan')
    policy_contract(policy)
    extraction_sha = evidence.pin(BATCH+'/EXTRACTIONS.json')
    extraction = evidence.read(BATCH+'/EXTRACTIONS.json', extraction_sha)
    check(extraction['plan_sha256'] == plan_sha, 'Extraction plan mismatch')
    check(extraction.get('status') == 'source_values_extracted_pending_independent_review', 'Unrecognized extraction stage')
    for counter in ('new_financial_admissions', 'new_strategy_trials', 'ratios_computed', 'returns_computed'):
        check(extraction.get(counter) == 0, 'Premature financial processing')
    events = extraction['events']
    check(len(events) == 26 and Counter(row['symbol'] for row in events) == {'AMZN': 10, 'CSCO': 16},
          'Fixed 26-event batch required')
    check([row['event_id'] for row in events] == [row['event_id'] for row in plan['records']],
          'Extraction order/population differs from frozen plan')
    review = evidence.read(REVIEW, review_sha)
    evidence.pins(review['input_hashes'])
    for name, required in ((BATCH+'/PLAN.json', plan_sha), (BATCH+'/EXTRACTIONS.json', extraction_sha)):
        check(any(evidence.path(key) == evidence.path(name) and value == required
                  for key, value in review['input_hashes'].items()), 'Independent review missing mandatory binding')
    reviewed = verify_independent_review(review, events, plan_sha, extraction_sha)
    for required_name in list(CONTEXTS.values()) + [row['receipt_path'] for row in plan['records']]:
        required_sha = evidence.pin(required_name)
        check(any(evidence.path(key) == evidence.path(required_name) and value == required_sha
                  for key, value in review['input_hashes'].items()),
              'Independent context/receipt binding missing')
    contexts = {symbol: evidence.read(name) for symbol, name in CONTEXTS.items()}
    cmap = context_map(contexts)
    check(set(cmap) == {row['event_id'] for row in events}, 'Context/extraction population mismatch')
    for event in events:
        source = event['source']
        evidence.pin(source['path'], source['sha256'])
        check(any(evidence.path(key) == evidence.path(source['path']) and value == source['sha256']
                  for key, value in review['input_hashes'].items()), 'Independent raw-source binding missing')
    sessions = normalize_sessions(evidence.read(SESSIONS))
    admitted = [assess_event(event, cmap[event['event_id']], reviewed[event['event_id']], sessions)
                for event in events]
    census, meta, old = evidence.read(CENSUS), evidence.read(META), evidence.read(OLD_DATA)
    old_summary = old_population(old)
    original_admission = evidence.read(OLD_ADMISSION)
    check(original_admission.get('passed') is True and
          original_admission.get('data_sha256') == evidence.pin(OLD_DATA) and
          [original_admission.get(key) for key in ('events', 'admitted_events', 'held_events')] == [44, 31, 13],
          'Original dataset/admission binding mismatch')
    old_ref = dict(path=OLD_DATA, sha256=evidence.pin(OLD_DATA), **old_summary,
                   admission_path=OLD_ADMISSION, admission_sha256=evidence.pin(OLD_ADMISSION),
                   preserved_without_modification=True)
    catalog = annual_catalog(census, cmap, sessions)
    check(len(catalog) == 32 and Counter(row['symbol'] for row in catalog) == {'AMZN': 16, 'CSCO': 16},
          'Complete 32-event two-issuer annual catalog required')
    by_catalog = {row['event_id']: row for row in catalog}
    for event in admitted:
        row = by_catalog[event['event_id']]
        check(row['symbol'] == event['symbol'] and row['fiscal_year'] == event['fiscal_year'] and
              row['publication_date'] == event.get('release_date', event.get('publication_date')) and
              row.get('available_session') == event['available_from'], 'Catalog/context metadata mismatch')
        income = event['fields']['annual_net_income']
        if income is not None and row.get('official_actual_period_end') is not None:
            check(row['official_actual_period_end'] == income['period']['period_end'], 'Catalog/annual endpoint mismatch')
    dataset = dict(schema_version=1, method=METHOD, policy_sha256=policy_sha,
                   source_plan_sha256=plan_sha, independent_review_sha256=review_sha,
                   events=admitted, latest_event_catalog=catalog, original_four_issuer_reference=old_ref,
                   conditional_research_only=True, first_public_version_verified=False,
                   complete_correction_history_verified=False, certified_daily_security_master=False,
                   scores_computed=0, ratios_computed=0, returns_computed=0, new_strategy_trials=0)
    monthly = coverage(meta, census, catalog, admitted)
    for filename in ('work/after149_batch1_admission.py', 'work/test_after149_batch1_admission.py'):
        evidence.pin(filename)
    evidence.stable()
    counts = dict(events=len(admitted), admitted=sum(e['ni_assets_admission']['admitted'] for e in admitted),
                  held=sum(not e['ni_assets_admission']['admitted'] for e in admitted),
                  reviewed_nonnull_fields=sum(e['fields'][f] is not None for e in events for f in FIELDS),
                  reviewed_null_fields=sum(e['fields'][f] is None for e in events for f in FIELDS),
                  annual_catalog_events=len(catalog), unprocessed_annual_catalog_events=len(catalog)-len(admitted),
                  original_events=44, original_admitted=31, original_held=13)
    admission = dict(schema_version=1, stage='conditional_source_admission_not_strategy_registration',
                     passed=True, created_utc=datetime.now(timezone.utc).isoformat(),
                     policy_sha256=policy_sha, source_plan_sha256=plan_sha, independent_review_sha256=review_sha,
                     counts=counts, input_hashes=evidence.seen,
                     source_selection='fixed_existing_context_batch_no_return_selection',
                     numerical_work='Finiteness/sign checks only; no scaling arithmetic, ratios, scores or returns.',
                     actual_factors_calculated=0, actual_returns_calculated=0, new_strategy_trials=0)
    return {'DATASET.json': dataset, 'COVERAGE.json': monthly, 'ADMISSION.json': admission}, evidence


def publish(root, objects, evidence):
    directory = Path(root)/OUT
    directory.mkdir(parents=True, exist_ok=True)
    targets = [directory/name for name in list(objects)+['MANIFEST.json']]
    check(not any(path.exists() for path in targets), 'Write-once output already exists')
    evidence.stable()
    for name, data in objects.items():
        with (directory/name).open('xb') as stream:
            stream.write((json.dumps(data, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode())
    evidence.stable()
    manifest = dict(schema_version=1, input_hashes=evidence.seen,
                    outputs={name: digest(directory/name) for name in objects},
                    status='conditional_source_package_no_strategy_registration')
    with (directory/'MANIFEST.json').open('xb') as stream:
        stream.write((json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False)+'\n').encode())
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-sha', required=True)
    parser.add_argument('--review-sha', required=True)
    parser.add_argument('--policy-sha', required=True)
    args = parser.parse_args()
    objects, evidence = build(ROOT, args.plan_sha, args.review_sha, args.policy_sha)
    manifest = publish(ROOT, objects, evidence)
    print(json.dumps({'status': 'conditional_source_package_complete',
                      'counts': objects['ADMISSION.json']['counts'], 'outputs': manifest['outputs']}, sort_keys=True))


if __name__ == '__main__':
    main()
