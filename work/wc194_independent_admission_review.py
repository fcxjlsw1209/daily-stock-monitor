"""Independent source-only WC194 assembly review; no production imports or sums."""
from bisect import bisect_right
from copy import deepcopy
from datetime import date, datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = 'outputs/monthly_working_capital194_source/'
SELF = 'work/wc194_independent_admission_review.py'
KEYS = ('accounts_receivable', 'inventories', 'accounts_payable')
DEFINITION = 'current_annual_trade_working_capital_signed_contributions_v1'
METHOD = 'current_annual_cash_before_trade_working_capital_v1'
OLD = 'outputs/monthly_cash_quality_source_admission_v1/DATASET.json'
SCOPE = 'work/working_capital194_source_scope.json'
CONCEPT = 'work/working_capital194_experiment_concept.json'
CALENDAR = 'outputs/monthly_forward_guidance_preflight_v1/SESSIONS.json'
PINS = {
    BASE+'admitted/ADMISSION.json': '8371d21eb44f1590f545b0808bfdf05ad63a52bf6ce63c285a3e889b758fde1c',
    BASE+'admitted/DATASET.json': '0eca5d80051f4081db8d7f0bf0405efa895365090feb3c989c21786c44a17133',
    BASE+'admitted/POLICY.json': 'd37e9606c03bb2e384a482fe9d326043127c86f471e0242424c6da1dbb9327a6',
    OLD: '3725b384ab4f48a3cabee1fc1adb1acd896be3f4967e8b6cd649a9a690c953d5',
    SCOPE: 'f2dda7ce4d3d61084bfbbb6e3fabca2388250575588054a41bd33890dc4948be',
    CONCEPT: '380aae3cf18288dd668cefb3a9be94d0e9c90b76e8d085f6d59ff0ddafed9a83',
    BASE+'independent/REVIEW.json': '0bb19df028a429cd81e304cfc43bf4c2f7b31ff122e551a565bd8d498f83b1d6',
    CALENDAR: 'c05c15f19bc1f7a3428e456a71199d64f05799ef3c432fa45f7bc485d0ff3cf4',
}
CODE_PINS = {
    'work/wc194_admission.py': '6aef1142c413cb0197db3710d5e061787726c3be8e5755cab80c10826e01ed37',
    'work/test_wc194_admission.py': 'dec03095b82a095c28d6eb0c9cf979cb7f76cc9cd9ec87c5b0cd480df528866a',
    'work/wc194_assemble.py': 'fe9d554e0ef6cff934f448fa556e9fc3ac62651946cd523aae5726a2c370f4a2',
    'work/test_wc194_assemble.py': 'abb4050d389c610021570b226ce4fb827099420781305794d76d0e36a0ee3d0c',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()


def read(path):
    return json.loads((ROOT/path).read_text())


def indexed(records):
    require(type(records) is list and all(type(r) is dict for r in records), 'Record list required')
    result = {r['event_id']: r for r in records}
    require(len(result) == len(records), 'Duplicate event IDs')
    return result


def literal_decimal(literal):
    # Reparse the signed source token, never add distinct cash-flow components.
    require(type(literal) is str and literal.strip(), 'Source literal required')
    token = re.sub(r'\s+|\$|/dollarsign', '', literal).replace('\u2212', '-')
    if token in ('-', '\u2013', '\u2014'):
        return Decimal(0)
    require(bool(re.fullmatch(r'(?:\((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\)|-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)', token)), 'Malformed monetary literal')
    if token.startswith('('):
        token = '-' + token[1:-1]
    return Decimal(token.replace(',', ''))


def normalize(raw, target, old):
    if raw is None:
        return None
    result = deepcopy(raw)
    for key, value in (
        ('source', target['source']), ('period', old['periods']['current']),
        ('available_from', old['available_from']), ('fact_admitted', False),
    ):
        result.setdefault(key, deepcopy(value))
        require(result[key] == value, 'Conflicting normalized field: '+key)
    return result


def check_field(raw, normalized, component, target, old, reviewed):
    require(normalized == normalize(raw, target, old), 'Normalized source values changed')
    if raw is None:
        require(reviewed['verified'] is False and all(reviewed[k] is None for k in ('signed_value_decimal', 'scale', 'annual_period_end')), 'Missing component became a value')
        return None
    require(reviewed['verified'] is True and raw['component'] == component, 'Wrong/unverified component')
    for key in ('signed_value_decimal', 'scale'):
        require(normalized[key] == reviewed[key], 'Independent '+key+' differs')
    require(re.fullmatch(r'-?\d+(?:\.\d+)?', normalized['signed_value_decimal']) is not None, 'Plain signed decimal required')
    require(literal_decimal(normalized['signed_literal']) == Decimal(normalized['signed_value_decimal']), 'Literal sign/value mismatch')
    require(Decimal(normalized['scale']).is_finite() and Decimal(normalized['scale']) > 0, 'Invalid units')
    require(hashlib.sha256(normalized['row_text'].encode()).hexdigest() == normalized['row_text_sha256'], 'Row text changed')
    patterns = {
        'accounts_receivable': r'(?:trade\s+)?accounts? receivable(?:,?\s+net)?',
        'inventories': r'inventor(?:y|ies)',
        'accounts_payable': r'(?:trade\s+)?accounts? payable',
    }
    require(re.fullmatch(patterns[component], re.sub(r'\s+', ' ', normalized['row_label']).strip(), re.I) is not None, 'Mixed or wrong component row')
    loc = normalized['locator']
    cfo = old['fields']['current']['operating_cash_flow']
    cfo_loc = cfo['locator']
    require(loc['original_cfo_locator'] == cfo_loc == target['current_cfo_locator'], 'Original CFO locator differs')
    require(loc['period_end'] == reviewed['annual_period_end'] == old['periods']['current']['period_end'], 'Wrong annual period')
    require(cfo['source'] == target['source'] and cfo['scale'] == normalized['scale'] and cfo['currency_literal'] == '$', 'Paired CFO source/units differ')
    require(normalized.get('currency_iso') is None and normalized.get('currency_literal') in (None, '$', '/dollarsign'), 'Currency invented or conflicting')
    require(normalized.get('statement_currency_literal') in (None, '$'), 'Conflicting statement currency')
    evidence = normalized['statement_currency_evidence']
    require(evidence['original_cfo_locator'] == cfo_loc and evidence['format'] == loc['format'] == cfo_loc['format'], 'Same-statement currency evidence missing')
    if loc['format'] == 'html':
        table = loc['table_index_zero_based']
        require(table == cfo_loc['table_index_zero_based'] == evidence['table_index_zero_based'], 'Wrong cash-flow table')
        require(loc.get('statement_segment') == cfo_loc.get('statement_segment'), 'Wrong cash-flow scope')
        if 'statement_segment' in evidence:
            require(evidence['statement_segment'] == cfo_loc.get('statement_segment'), 'Currency evidence scope differs')
        if normalized.get('statement_currency_literal') == '$':
            require(normalized['currency_scope'] == 'same formal cash-flow table; ISO unverified', 'Currency evidence scope missing')
        else:
            require(evidence.get('literal') == '$' and evidence.get('rows'), 'No literal dollar evidence')
            require(all('$' in r['row_text'] and re.fullmatch('[0-9a-f]{64}', r['row_html_sha256']) for r in evidence['rows']), 'Invalid dollar row evidence')
        row = loc['row_index_zero_based']
        headings = loc.get('operating_reconciliation_heading_rows')
        if headings is None:
            headings = [loc.get('operating_changes_heading_row_zero_based')]
        require(headings and all(type(h) is int for h in headings) and any(h < row < cfo_loc['row_index_zero_based'] for h in headings), 'Outside operating reconciliation')
        return ('html', table, json.dumps(loc.get('statement_segment'), sort_keys=True), row)
    require(loc['format'] == 'pdf', 'Unknown source format')
    page = loc['page_1based']
    require(page == cfo_loc['page_1based'] == evidence['page_1based'], 'Wrong cash-flow PDF page')
    require(normalized['statement_currency_literal'] == '$' and normalized['currency_scope'] == 'same formal cash-flow page; ISO unverified', 'PDF currency evidence missing')
    row = loc['line_1based']
    require(loc['operating_changes_heading_line_1based'] < row < cfo_loc['line_1based'], 'Outside PDF operating reconciliation')
    require(normalized['pdfium_text_agrees'] is True and normalized['pdfium_geometry'], 'PDF independent geometry missing')
    return ('pdf', page, row)


def verify():
    for path, expected in {**PINS, **CODE_PINS}.items():
        require(digest(path) == expected, 'Pinned bytes differ: '+path)
    admission = read(BASE+'admitted/ADMISSION.json')
    data = read(BASE+'admitted/DATASET.json')
    policy = read(BASE+'admitted/POLICY.json')
    original, scope, review = read(OLD), read(SCOPE), read(BASE+'independent/REVIEW.json')
    sessions = read(CALENDAR)['sessions']
    require(sessions == sorted(set(sessions)), 'Calendar ordering')
    require(admission['passed'] is True and (admission['events'], admission['admitted_events'], admission['held_events']) == (44, 29, 15), 'Admission census')
    require(admission['data_sha256'] == PINS[BASE+'admitted/DATASET.json'] and admission['policy_sha256'] == data['policy_sha256'] == PINS[BASE+'admitted/POLICY.json'], 'Root cross-linked SHA')
    require(admission['independent_review'] == dict(path=BASE+'independent/REVIEW.json', sha256=PINS[BASE+'independent/REVIEW.json']), 'Independent review binding')
    require(data['method'] == policy['method'] == METHOD and policy['definition'] == DEFINITION and data['conditional_research_only'] is True, 'Source method')
    require(policy['source_scope_sha256'] == review['scope_sha256'] == PINS[SCOPE] and policy['concept_sha256'] == review['concept_sha256'] == PINS[CONCEPT], 'Scope/concept binding')
    require(all(policy[k] is False for k in ('component_sum_permitted', 'factor_computation_permitted', 'financial_ratios_permitted', 'return_computation_permitted', 'use_comparison_flows')), 'Source-only boundary')
    require(all(policy[k] is True for k in ('current_report_trade_working_capital_only', 'all_three_components_required', 'preserve_all_original_holds', 'conditional_research_only')), 'Source qualification changed')
    require(policy['clock'] == 'first pinned exchange session strictly after report release date' and policy['annual_days'] == [330, 400] and policy['max_age_days'] == 550, 'Source time policy')
    require(review['passed'] is True and review['source_only'] is True, 'Original source review incomplete')
    require(all(review[k] is False for k in ('component_sum_computed', 'ratios_computed', 'factors_computed', 'orders_computed', 'returns_computed')), 'Raw review boundary')
    require(all(admission[k] == 0 for k in ('component_sums', 'financial_ratios', 'factor_calculations', 'return_calculations', 'observer_changes')), 'Assembly performed non-source computation')
    closure = policy['input_hashes']
    require(len(closure) == 267, 'Closure census')
    for path, expected in closure.items():
        require(digest(path) == expected, 'Closure bytes differ: '+path)
    for key in ('input_sha256', 'producer_refs', 'producer_visual_refs', 'expected_ref', 'preflight_ref', 'tests_ref', 'code_sha256', 'producer_code_sha256'):
        for path, expected in review[key].items():
            require(closure.get(path) == expected, 'Review closure omitted: '+path)
    for path, expected in CODE_PINS.items():
        require(closure.get(path) == expected, 'Source code missing from closure: '+path)
    require(closure['work/capex193_pdf_runtime/RUNTIME.json'] == review['runtime']['manifest_sha256'], 'New runtime binding')
    require(review['runtime']['historical_runtime_currently_passes'] is False and review['runtime']['old_runtime_mismatches'], 'Historical runtime failure wrongly cleared')
    visuals = indexed(review['independent_visual_reviews'])
    require(len(visuals) == 10, 'PDF visual coverage')
    for visual in visuals.values():
        require(closure.get(visual['render_path']) == visual['render_sha256'], 'Independent rendered evidence missing')
        require(visual['three_distinct_operating_reconciliation_rows'] is True and visual['signs_and_values_match_independent_extraction'] is True, 'Independent visual review incomplete')
    targets, reviewed, producers = indexed(scope['prospective_targets']), indexed(review['events']), {}
    for path in review['producer_refs']:
        for eid, producer in indexed(read(path)['records']).items():
            require(eid not in producers, 'Duplicate producer event')
            producers[eid] = producer
    ids = [e['event_id'] for e in original['events']]
    require(len(ids) == len(set(ids)) == 44 and [e['event_id'] for e in data['events']] == ids, 'Original44 identity/order')
    require(set(ids) == set(targets) == set(reviewed) == set(producers), 'Coverage44')
    added = {'trade_working_capital', 'trade_working_capital_source_evidence', 'trade_working_capital_admission'}
    event_reviews, component_counts = [], {k: 0 for k in KEYS}
    for old, new in zip(original['events'], data['events']):
        eid = old['event_id']
        target, checked, producer = targets[eid], reviewed[eid], producers[eid]
        require(set(new) == set(old) | added and {k: v for k, v in new.items() if k not in added} == old, 'Original event changed: '+eid)
        require(new['trade_working_capital_source_evidence'] == producer, 'Raw producer evidence changed: '+eid)
        require(checked['review_passed'] is True and checked['source_sha256'] == target['source']['sha256'], 'Independent source binding: '+eid)
        require(producer['source'] == target['source'] == old['source'], 'Event source changed: '+eid)
        inherited = old['cash_quality_admission']['admitted']
        require(producer['inherited_admitted'] is inherited and target['inherited_admitted'] is inherited, 'Inherited gate changed: '+eid)
        require(closure.get(target['source']['path']) == target['source']['sha256'], 'Raw source missing from closure: '+eid)
        for k in ('event_id', 'symbol', 'release_date', 'available_from'):
            require(producer[k] == old[k], 'Producer metadata differs: '+eid+'/'+k)
        for day in (old['release_date'], old['available_from']):
            require(date.fromisoformat(day).isoformat() == day, 'Noncanonical date')
        ix = bisect_right(sessions, old['release_date'])
        require(ix < len(sessions) and sessions[ix] == old['available_from'] == target['inherited_available_from'] == checked['available_from'], 'Availability not strict next exchange day: '+eid)
        require(set(producer['fields']) == set(new['trade_working_capital']) == set(checked['components']) == set(KEYS), 'Component schema: '+eid)
        positions, components, reasons = [], {}, []
        if not inherited:
            reasons.append('preserved_original_cash_quality_hold')
        for key in KEYS:
            raw = producer['fields'][key]
            field = new['trade_working_capital'][key]
            position = check_field(raw, field, key, target, old, checked['components'][key])
            if raw is None:
                require(producer['hold_reasons'], 'Missing component without hold reason: '+eid)
                reasons.append(key+':missing_or_ambiguous')
            else:
                component_counts[key] += 1
                positions.append(position)
                require(checked['paired_cfo_verified'] is True, 'Paired CFO not verified: '+eid)
                if inherited:
                    require(checked['paired_assets_verified'] is True, 'Paired assets not verified: '+eid)
            components[key] = dict(verified=raw is not None, normalized_exact=True, missing_preserved=raw is None)
        require(len(positions) == len(set(positions)), 'Duplicate source component rows: '+eid)
        complete = all(producer['fields'][k] is not None for k in KEYS)
        require(producer['extraction_verified'] is complete and checked['three_signed_rows_verified'] is complete, 'Completeness changed: '+eid)
        reasons = list(dict.fromkeys(reasons + producer['hold_reasons']))
        should_admit = inherited and complete and checked['paired_cfo_verified'] and checked['paired_assets_verified'] and not producer['hold_reasons']
        gate = new['trade_working_capital_admission']
        require(gate == dict(admitted=should_admit, reasons=reasons, definition=DEFINITION), 'Independent admission/reasons differ: '+eid)
        require(producer['status'] == ('verified' if should_admit else 'held'), 'Producer status differs: '+eid)
        event_reviews.append(dict(event_id=eid, original_values_exact=True, raw_evidence_exact=True, normalization_exact=True, source_clock_verified=True, components=components, paired_cfo_verified=checked['paired_cfo_verified'], paired_assets_verified=checked['paired_assets_verified'], inherited_admitted=inherited, admitted=should_admit, hold_reasons=deepcopy(reasons)))
    admitted = [r['event_id'] for r in event_reviews if r['admitted']]
    held = [r['event_id'] for r in event_reviews if not r['admitted']]
    require(admitted == admission['admitted_event_ids'] and held == admission['held_event_ids'], 'Admission sets/order differ')
    oldheld = {e['event_id'] for e in original['events'] if not e['cash_quality_admission']['admitted']}
    require(len(oldheld) == 13 and oldheld <= set(held), 'Original holds lost')
    require(set(held)-oldheld == {'AMD:FY2016:2017-01-31', 'AMD:FY2017:2018-01-30'}, 'New component holds differ')
    require(component_counts == dict(accounts_receivable=34, inventories=34, accounts_payable=31), 'Verified component coverage differs')
    require(not any(name in sys.modules for name in ('wc194_admission', 'wc194_assemble', 'working_capital194_factor', 'monthly_working_capital194_host', 'monthly_working_capital194_independent')), 'Production module imported')
    # Check all bindings a second time before publishing the single receipt.
    for path, expected in {**closure, **PINS, **CODE_PINS}.items():
        require(digest(path) == expected, 'Input changed during review: '+path)
    return dict(
        schema='wc194_independent_admission_review_v1', created_utc=datetime.now(timezone.utc).isoformat(),
        passed=True, source_only=True, source_audit_is_not_strategy_admission=True,
        events=44, admitted_events=29, held_events=15, inherited_holds_preserved=13, new_holds=2,
        closure_pins_verified=len(closure), component_cells_verified=component_counts,
        all_original_event_values_preserved=True, all_raw_producer_evidence_preserved=True,
        normalized_fields_and_source_clock_checked=True, source_review_closure_preserved=True,
        source_runtime_status='New193 isolated runtime is pinned; historical shared PDFium failure remains recorded and is not represented as a current pass.',
        source_code_sha256=CODE_PINS, input_sha256=PINS, review_code_sha256=digest(SELF),
        event_reviews=event_reviews, component_sum_computed=False, component_sums_computed=False,
        ratios_computed=False, factors_computed=False, orders_computed=False, returns_computed=False,
        production_modules_imported=False,
        source_review_reused='Frozen independent original-source REVIEW and both producer/visual receipts were verified against current bytes; this normalization review does not repeat PDF extraction or change original values.',
        limitations=['Source qualification is conditional research admission, not first-public-version certification.', 'Older13 holds remain; AMD2016/2017 combined AP rows add2 holds. Missing rows remain null.', 'No component aggregation, financial ratio, factor, portfolio or observer operation performed.'],
        remaining='Numerical factor, oracle and strategy host require a separate CODE_REVIEW and prospective strategy registration.',
    )


def main():
    result = verify()
    output = ROOT/BASE/'independent/ADMISSION_REVIEW.json'
    with output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(dict(passed=True, events=44, admitted=29, held=15, closure=267, path=str(output.relative_to(ROOT)), sha256=hashlib.sha256(output.read_bytes()).hexdigest(), review_code_sha256=digest(SELF))))


if __name__ == '__main__':
    main()
