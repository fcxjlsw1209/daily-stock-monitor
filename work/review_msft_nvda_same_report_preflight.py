"""Independent source-only audit: no issuer extractor imports or financial math.

Verifies physical HTML cells, header columns, period provenance, own publication
availability, and the limited scope of four Microsoft accounting notes. It does
not admit any facts or certify the originally published historical page version.
"""
from pathlib import Path
from datetime import date, timedelta, datetime, timezone
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import argparse
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
STUDY = 'outputs/monthly_msft_nvda_same_report_preflight_v1'
SELF = 'work/review_msft_nvda_same_report_preflight.py'
TEST = 'work/test_review_msft_nvda_same_report_preflight.py'
OUTPUT = 'work/after144_msft_nvda_source_review.json'
PLAN_SHA = '33024f402deb6dbc0c36ea8b2b1219cf6c69036b417a7865b128345e574882b2'
EVENTS_SHA = 'e2c6cada88d94d020d01472641adaace1d9847c5e17b500773224b05b46f2f8a'
SCHEMA_SHA = 'eebdc65dd627bb49a7e803fe7598586fe48f4d7ce0fdfdcb817955b156f9e258'
METHOD = 'same_current_release_annual_comparison'
SCOPE = [(s, y) for s in ('MSFT', 'NVDA') for y in range(2015, 2027)]
FLAGGED = {('MSFT', y) for y in (2018, 2023, 2025, 2026)}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def norm(value):
    return re.sub(r'\s+', ' ', str(value).replace('\xa0', ' ')).strip()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def parse_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('nonfinite JSON token')
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


class Sources:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.hashes = {}

    def read(self, relative, sha=None):
        p = Path(relative)
        need(type(relative) is str and not p.is_absolute() and '..' not in p.parts,
             'relative source path required')
        need(not any(v in relative.lower() for v in ('cloudstorage', 'onedrive')), 'cloud source forbidden')
        p = self.root / p
        need(p.resolve() == p, 'source symlink forbidden')
        raw = p.read_bytes()
        actual = digest(raw)
        need(sha is None or actual == sha, 'source checksum mismatch: ' + relative)
        need(relative not in self.hashes or self.hashes[relative] == actual, 'source changed during review')
        self.hashes[relative] = actual
        return raw

    def json(self, relative, sha=None):
        return parse_json(self.read(relative, sha))


def iso(text):
    need(type(text) is str, 'date string required')
    value = date.fromisoformat(text)
    need(value.isoformat() == text, 'canonical date required')
    return value


def strict_next_session(release, sessions):
    need(sessions == sorted(set(sessions)) and bool(sessions), 'invalid sessions')
    for value in sessions:
        iso(value)
    iso(release)
    need(sessions[0] <= release < sessions[-1], 'release outside sessions')
    return next(value for value in sessions if value > release)


def expanded_table(table):
    """Independent physical cell -> rectangular logical grid, preserving spans."""
    rows = [r for r in table.find_all('tr') if r.find_parent('table') is table]
    grid, cells = {}, {}
    for ri, row in enumerate(rows):
        col = 0
        for pi, node in enumerate(row.find_all(['td', 'th'], recursive=False)):
            while (ri, col) in grid:
                col += 1
            height, width = int(node.get('rowspan', 1)), int(node.get('colspan', 1))
            need(1 <= height <= 100 and 1 <= width <= 100, 'invalid HTML span')
            item = {'node': node, 'text': norm(node.get_text(' ', strip=True)),
                    'columns': list(range(col, col + width)), 'row': ri, 'physical': pi}
            cells[(ri, pi)] = item
            for y in range(ri, ri + height):
                for x in range(col, col + width):
                    need((y, x) not in grid, 'overlapping HTML span')
                    grid[(y, x)] = item
            col += width
    return rows, grid, cells


def raw_decimal(display):
    value = norm(display).replace('$', '').replace(',', '').replace(' ', '')
    if value.startswith('(') and value.endswith(')'):
        value = '-' + value[1:-1]
    need(re.fullmatch(r'-?\d+(?:\.\d+)?', value), 'not one plain monetary amount')
    return value


def header_for_column(rows, grid, target_row, column):
    # Headers end before first named financial data row, not before target cell.
    stop = target_row
    for ri, row in enumerate(rows[:target_row]):
        labels = [norm(c.get_text(' ', strip=True)) for c in row.find_all(['td', 'th'], recursive=False)]
        if any(re.fullmatch(r'(?:total )?revenues?:?|assets|cash(?: and cash equivalents)?', s, re.I) for s in labels):
            stop = ri
            break
    headers = []
    for ri in range(stop):
        item = grid.get((ri, column))
        if item and item['text'] and item['text'] not in headers:
            headers.append(item['text'])
    return headers


def dates_in_header(headers):
    joined = norm(' '.join(headers))
    result = set()
    for match in re.finditer(r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})', joined, re.I):
        result.add(datetime.strptime(' '.join(match.groups()), '%B %d %Y').date().isoformat())
    return result


def nearby_context(table):
    # At most five adjacent heading nodes in Microsoft's statement wrapper;
    # Nvidia puts titles inside its table. Never borrow a heading across a table.
    caption = table.find('caption')
    if caption is not None:
        return norm(caption.get_text(' ', strip=True))
    inside = [norm(row.get_text(' ', strip=True)) for row in table.find_all('tr')[:6]]
    if re.search(r'INCOME STATEMENTS|STATEMENTS OF (?:INCOME|OPERATIONS)|BALANCE SHEETS|CASH FLOWS', ' '.join(inside), re.I):
        return norm(' '.join(inside))
    preceding = []
    node = table.parent if table.parent and 'c-table' in table.parent.get('class', []) else table
    for sibling in node.previous_siblings:
        if getattr(sibling, 'name', None) in ('table', 'div') or (getattr(sibling, 'find', None) and not isinstance(sibling, str) and sibling.find('table')):
            break
        if getattr(sibling, 'get_text', None):
            value = norm(sibling.get_text(' ', strip=True))
        else:
            value = norm(sibling)
        if value:
            preceding.append(value)
        if len(preceding) >= 5:
            break
    inside = [norm(r.get_text(' ', strip=True)) for r in table.find_all('tr')[:6]]
    return norm(' '.join(list(reversed(preceding)) + inside))


def verify_cell(soup, spec, event, which, field):
    cell = event[which][field]
    loc = cell['locator']
    ti, ri, pi = [loc[k] for k in ('table_index_zero_based', 'row_index_zero_based', 'physical_cell_index_zero_based')]
    need(all(type(x) is int and x >= 0 for x in (ti, ri, pi)), 'invalid cell locator')
    table = soup.find_all('table')[ti]
    rows, grid, cells = expanded_table(table)
    node = cells[(ri, pi)]
    need(node['columns'] == loc['logical_columns'], 'logical column mismatch')
    need(digest(str(node['node']).encode()) == cell['parsed_cell_sha256'], 'physical cell hash mismatch')
    need(node['text'] == cell['displayed_amount'], 'displayed monetary value mismatch')
    need(raw_decimal(node['text']) == cell['value_decimal'], 'decimal string mismatch')
    expected_label = 'Total assets' if field == 'assets' else ('Gross margin' if event['symbol'] == 'MSFT' else 'Gross profit')
    matching_rows = sorted({r for (r, _), item in cells.items() if item['text'].lower() == expected_label.lower()})
    need(matching_rows == [ri], 'financial row label not unique')
    need(not any('%' in c['text'] for (r, _), c in cells.items() if r == ri), 'percentage target row')
    kind = 'year_end_assets' if field == 'assets' else 'annual_gross_amount'
    specs = [r for r in spec['rows'] if r['kind'] == kind]
    need(len(specs) == 1 and specs[0]['row_label'] == expected_label, 'planned financial row mismatch')
    need(specs[0]['table_index_zero_based'] == ti and specs[0]['row_index_zero_based'] == ri, 'planned locator mismatch')
    end = event[which]['period']['end']
    matched_headers = []
    candidates = []
    for (r, p), item in cells.items():
        if r != ri:
            continue
        try:
            raw_decimal(item['text'])
        except ValueError:
            continue
        for column in item['columns']:
            headers = header_for_column(rows, grid, ri, column)
            annual = bool(re.search(r'Twelve\s+Months\s+Ended', ' '.join(headers), re.I))
            if dates_in_header(headers) == {end} and (field == 'assets' or annual):
                candidates.append((r, p))
                if p == pi:
                    matched_headers.append(headers)
    need(set(candidates) == {(ri, pi)}, 'ambiguous or wrong annual/date column')
    need(cell['header_literals'] in matched_headers, 'literal header mismatch')
    context = nearby_context(table)
    table_text = norm(table.get_text(' ', strip=True))
    heading = r'BALANCE SHEETS' if field == 'assets' else r'INCOME STATEMENTS|STATEMENTS OF (?:INCOME|OPERATIONS)'
    need(re.search(heading, context, re.I), 'statement type not independently established: ' + str((event.get('event_id'), field, ti, context)))
    need(not re.search(r'non.gaap|reconciliation|as adjusted', context + ' ' + table_text, re.I), 'non-GAAP or adjusted table')
    need(cell['unit_literal'] == specs[0]['units']['literal'], 'planned monetary unit differs')
    need(norm(cell['unit_literal']) in context + ' ' + table_text, 'monetary unit not in source')
    scales = {'millions': '1000000', 'thousands': '1000'}
    units = re.findall(r'\bin (millions|thousands)\b', cell['unit_literal'], re.I)
    need(len(units) == 1 and cell['scale'] == scales[units[0].lower()], 'monetary scale mismatch')
    need('$' in table_text and cell['currency_literal'] == '$' and cell['currency_iso'] is None, 'same-report dollar unit mismatch')
    need(cell['source'] == event['source'], 'cell borrowed from another publication')
    need(cell['available_from'] == event['available_from'], 'comparison prematurely available')
    need(cell['period_end'] == end, 'period end mismatch')
    return dict(side=which, field=field, locator=loc, value_decimal=cell['value_decimal'],
                scale=cell['scale'], currency_literal='$', header_literals=cell['header_literals'],
                parsed_cell_sha256=cell['parsed_cell_sha256'], verified=True)


def verify_periods(spec, event, previous, sources):
    current, comp = event['current']['period'], event['comparison']['period']
    need(current == spec['current_period'] and comp == spec['comparison_period'], 'period differs from frozen plan')
    need(iso(current['end']) < iso(event['release_date']), 'period ends after publication')
    need(iso(current['end']).year == event['fiscal_year'] and iso(comp['end']).year == event['fiscal_year'] - 1, 'annual fiscal year mismatch')
    need(iso(current['start']) == iso(comp['end']) + timedelta(days=1), 'current annual boundary not contiguous')
    for period in (current, comp):
        need(period['derived_not_direct_quote'] is True and bool(period['evidence']), 'missing derived boundary provenance')
        for evidence in period['evidence']:
            sources.read(evidence['path'], evidence['sha256'])
            if 'release_date' in evidence:
                need(iso(evidence['release_date']) <= iso(event['release_date']), 'future boundary evidence')
        if period['start'] is not None:
            days = (iso(period['end']) - iso(period['start'])).days + 1
            need(type(period['days_inclusive']) is int and days == period['days_inclusive'] and 330 <= days <= 400, 'invalid annual duration')
    if event['fiscal_year'] == 2015:
        need(previous is None and comp['start'] is None and comp['days_inclusive'] is None, 'unknown initial comparison duration must stay unknown')
        return False
    need(previous is not None and previous['symbol'] == event['symbol'], 'previous issuer boundary unavailable')
    need(iso(previous['release_date']) < iso(event['release_date']), 'future prior annual report')
    need(comp['end'] == previous['current']['period']['end'], 'prior year end mismatch')
    need(iso(comp['start']) == iso(previous['comparison']['period']['end']) + timedelta(days=1), 'prior duration not derived from previously available boundary')
    need(any(e['path'] == previous['source']['raw_path'] and e.get('release_date') == previous['release_date'] for e in comp['evidence']), 'comparison boundary lacks previous publication evidence')
    return True


def verify_event(spec, event, previous, sessions, sources):
    need(event['symbol'] == spec['symbol'] and event['fiscal_year'] == spec['fiscal_year'], 'issuer/year identity mismatch')
    need(event['release_date'] == spec['release_date'], 'current publication date mismatch')
    need(event['event_id'] == '%s:FY%s:%s' % (event['symbol'], event['fiscal_year'], event['release_date']), 'event identity mismatch')
    source = {key: spec[key] for key in ('raw_path', 'raw_sha256', 'source_url', 'receipt_path', 'receipt_sha256', 'receipt_entry_key')}
    need(event['source'] == source, 'event source mismatch')
    raw = sources.read(source['raw_path'], source['raw_sha256'])
    receipt = sources.json(source['receipt_path'], source['receipt_sha256'])
    if source['receipt_entry_key'] is not None:
        receipt = receipt[source['receipt_entry_key']]
    need(receipt['http_status'] == 200 and receipt['sha256'] == digest(raw) and receipt['url'] == source['source_url'], 'retrieval receipt mismatch')
    need(urlparse(source['source_url']).hostname in ('www.microsoft.com', 'nvidianews.nvidia.com'), 'not own issuer source')
    need(datetime.fromisoformat(receipt['retrieved_at'].replace('Z', '+00:00')).utcoffset() is not None, 'unaware retrieval timestamp')
    need('bytes' not in receipt or receipt['bytes'] == len(raw), 'retrieval byte count mismatch')
    soup = BeautifulSoup(raw, 'html.parser')
    text = norm(soup.get_text(' ', strip=True))
    need(norm(spec['release_date_literal']) in text, 'own publication date missing')
    need(norm(spec['identity_anchor']) in text, 'own issuer/ticker anchor missing')
    need(event['available_from'] == strict_next_session(event['release_date'], sessions), 'wrong first known session')
    identity = event['identity']
    need(identity['anchor'] == spec['identity_anchor'] and identity['anchor_in_current_source'] is True, 'identity anchor mismatch')
    need(identity['valid_from'] == event['available_from'], 'identity premature availability')
    need(identity['valid_through'] == (iso(event['current']['period']['end']) + timedelta(days=550)).isoformat(), 'identity stale expiry mismatch')
    need(identity['certified_daily_security_master'] is False, 'unearned daily-master certification')
    for flag in ('first_public_version_verified', 'complete_correction_history_verified', 'fact_admitted', 'model_admitted', 'strategy_admitted'):
        need(event[flag] is False, 'premature admission or certification')
    need(event['cell_errors'] == [] and event['four_cells_extracted'] is True, 'incomplete event')
    cells = [verify_cell(soup, spec, event, side, field) for side in ('current', 'comparison') for field in ('gross_profit', 'assets')]
    complete = verify_periods(spec, event, previous, sources)
    codes = ([] if complete else ['unknown_comparison_period_start'])
    if (event['symbol'], event['fiscal_year']) in FLAGGED:
        codes += ['accounting_context_requires_independent_review']
    need(event['rejection_codes'] == codes, 'unexplained producer rejection codes')
    return dict(event_id=event['event_id'], symbol=event['symbol'], fiscal_year=event['fiscal_year'],
                cells_verified=4, raw_cells_verified=True, availability_verified=True,
                identity_verified_under_assumptions=True, periods_verified=True,
                period_completeness=complete, remaining_rejection_codes=codes,
                cell_reviews=cells, fact_admitted=False, model_admitted=False, strategy_admitted=False), soup


def accounting_scope(event, previous, soup):
    if (event['symbol'], event['fiscal_year']) not in FLAGGED:
        return {'status': 'no_flag', 'resolution_code': 'no_preflight_accounting_flag', 'scope': 'No accounting hold identified by the frozen preflight; this is not a certification of the absence of accounting changes.', 'evidence': []}
    year = event['fiscal_year']
    text = norm(soup.get_text(' ', strip=True))
    tables = soup.find_all('table')
    path, sha = event['source']['raw_path'], event['source']['raw_sha256']
    parity = {f: event['comparison'][f]['value_decimal'] == previous['current'][f]['value_decimal'] and event['comparison'][f]['scale'] == previous['current'][f]['scale'] for f in ('gross_profit', 'assets')}
    evidence = []
    if year == 2018:
        quote = 'We adopted new accounting standards related to revenue recognition and leases effective July 1, 2017. The prior periods presented here have been restated to reflect adoption of these new standards.'
        need(quote in text and event['current']['period']['start'] == '2017-07-01', '2018 common-basis restatement not established')
        need(not any(parity.values()), '2018 expected comparative restatement not corroborated in both selected totals')
        evidence.append(dict(path=path, sha256=sha, quote=quote, locator='New Accounting Standards section; all prior periods presented in current release'))
        scope = ('Revenue recognition and lease standards apply from 2017-07-01, the start of FY2018. The current release explicitly restates the prior periods presented here, covering its consolidated income statement and balance sheet, not only a segment table. Both the selected FY2017 gross margin and total assets differ from the FY2017 original release. Thus the selected current/comparison columns are conditionally usable together on the current release\'s restated basis; old original FY2017 values must not be mixed in or made available retroactively. This does not establish organic growth, constant business scope, first-public historical version, or unchanged accounting across releases.')
        code = 'both_comparatives_restated_to_current_revenue_and_lease_basis'
    else:
        quote = ('We have recast certain prior period amounts to conform to the current period presentation.' if year == 2026 else 'We have recast certain prior period amounts to conform to the way we internally manage and monitor our business.')
        paragraphs = [p for p in soup.find_all('p') if norm(p.get_text(' ', strip=True)) == quote]
        need(len(paragraphs) == (1 if year == 2026 else 2), 'unexpected recast note count')
        locations = []
        for p in paragraphs:
            table = p.find_previous('table')
            need(table is not None, 'recast note has no preceding table')
            ti = next(i for i, t in enumerate(tables) if t is table)
            content = norm(table.get_text(' ', strip=True))
            context = nearby_context(table)
            if year == 2026:
                need(re.search(r'CASH FLOWS', context, re.I), '2026 recast not located at cash-flow table')
            else:
                need(all(term in content for term in ('Productivity and Business Processes', 'Intelligent Cloud', 'More Personal Computing')), 'recast not located at segment table')
            need(ti not in {event['current'][f]['locator']['table_index_zero_based'] for f in ('gross_profit', 'assets')}, 'recast note applies directly to selected statement')
            locations.append(ti)
        need(all(parity.values()), 'selected consolidated comparatives changed: recast scope unresolved')
        evidence.append(dict(path=path, sha256=sha, quote=quote, locator='Paragraph immediately after table indices (zero-based): ' + ', '.join(map(str, locations))))
        scope = ('The recast note is attached to ' + ('the cash-flow statement presentation' if year == 2026 else 'the segment revenue/constant-currency and segment results tables') + ', separate from the selected consolidated gross-margin and total-assets tables. Both selected prior-year totals match the corresponding original prior release literally at the same scale. This supports removing this specific recast-scope hold for these two totals only; it does not establish absence of other accounting-estimate changes or first-public historical versions.')
        code = 'cashflow_presentation_only_selected_totals_unchanged' if year == 2026 else 'segment_recast_only_selected_totals_unchanged'
        if year == 2023:
            caveat = 'a reduction in depreciation expense due to the change in accounting estimate for the useful lives of our server and network equipment'
            need(caveat in text, '2023 useful-life interpretation caveat missing')
            evidence.append(dict(path=path, sha256=sha, quote=caveat, locator='Financial Highlights / fourth-quarter cost of revenue and gross margin discussion'))
            scope += ' The current release separately states that longer server/network useful-life estimates reduced depreciation expense and affected gross margin. Retain this material as-reported accounting-estimate effect: the figures are not restated to constant equipment-life assumptions, and this review does not classify all observed improvement as operating quality.'
    # Only literal equality comparisons: no differences, normalized values, ratios or signs.
    for field in ('gross_profit', 'assets'):
        evidence.append(dict(path=path, sha256=sha, quote=event['comparison'][field]['displayed_amount'], locator=json.dumps(event['comparison'][field]['locator'], sort_keys=True)))
        evidence.append(dict(path=previous['source']['raw_path'], sha256=previous['source']['raw_sha256'], quote=previous['current'][field]['displayed_amount'], locator=json.dumps(previous['current'][field]['locator'], sort_keys=True)))
    return dict(status='resolved', resolution_code=code, scope=scope, evidence=evidence,
                selected_comparatives_equal_original_prior_release=parity,
                resolution_scope='conditional same-current-report source use only; not fact/model/strategy admission')


def run_review(root=ROOT):
    sources = Sources(root)
    plan = sources.json(STUDY + '/PLAN.json', PLAN_SHA)
    schema = sources.json(STUDY + '/SCHEMA.json', SCHEMA_SHA)
    data = sources.json(STUDY + '/EVENTS.json', EVENTS_SHA)
    audit = sources.json(STUDY + '/AUDIT.json')
    manifest = sources.json(STUDY + '/MANIFEST.json')
    sources.json(STUDY + '/TESTS.json')
    for mapping in (plan['source_hashes'], plan['source_code_hashes']):
        for path, sha in mapping.items():
            sources.read(path, sha)
    need(plan['method'] == schema['method'] == data['method'] == METHOD, 'same-report method mismatch')
    need(audit['stage'] == 'source_preflight' and audit['passed'] is True and audit['all_cells_extracted'] is True, 'preflight failed')
    need(audit['plan_sha256'] == manifest['plan_sha256'] == PLAN_SHA, 'preflight plan binding mismatch')
    need(audit['data_sha256'] == EVENTS_SHA, 'preflight events binding mismatch')
    need(manifest['output_hashes'] == {'EVENTS.json': EVENTS_SHA, 'AUDIT.json': sources.hashes[STUDY + '/AUDIT.json']}, 'preflight output hashes mismatch')
    need(audit['source_hashes'] == plan['source_hashes'] and audit['source_code_hashes'] == plan['source_code_hashes'], 'source closure mismatch')
    need(audit['financial_calculations'] == audit['network_requests'] == 0, 'unexpected preflight computation')
    for flag in ('first_public_version_verified', 'complete_correction_history_verified', 'certified_daily_security_master'):
        need(data[flag] is False, 'unearned preflight certification')
    for count in ('ratios_computed', 'signs_computed', 'ranks_computed', 'model_fits', 'returns_computed'):
        need(type(data[count]) is int and data[count] == 0, 'not source-only preflight')
    events = data['events']
    need([(e['symbol'], e['fiscal_year']) for e in events] == SCOPE, 'missing/reordered/duplicate source event')
    need([(s['symbol'], s['fiscal_year']) for s in plan['sources']] == SCOPE, 'source plan scope mismatch')
    sessions = sources.json(plan['sessions']['path'], plan['sessions']['sha256'])['sessions']
    reviewed, previous = [], {}
    for spec, event in zip(plan['sources'], events):
        prior = previous.get(event['symbol'])
        result, soup = verify_event(spec, event, prior, sessions, sources)
        result['accounting'] = accounting_scope(event, prior, soup)
        if result['accounting']['status'] == 'resolved':
            result['remaining_rejection_codes'].remove('accounting_context_requires_independent_review')
        reviewed.append(result)
        previous[event['symbol']] = event
    need(sum(x['cells_verified'] for x in reviewed) == 96, 'not all 96 cells verified')
    need(sum(not x['period_completeness'] for x in reviewed) == 2, 'missing-duration sentinels changed')
    for path in (SELF, TEST):
        sources.read(path)
    test_log_path = 'work/after144_msft_nvda_source_review_tests.log'
    test_log = sources.read(test_log_path).decode()
    need(re.search(r'Ran 26 tests in .*\n\nOK\s*$', test_log), 'independent synthetic test log missing success')
    return dict(schema_version=1, stage='independent_same_report_source_review', passed=True,
                method=METHOD, source_only=True, financial_calculations=0, ratios_computed=0,
                signs_computed=0, model_fits=0, ranks_computed=0, returns_computed=0,
                events_reviewed=24, cells_reviewed=96, synthetic_tests_passed=26, source_hashes=dict(sorted(sources.hashes.items())),
                preflight_plan_sha256=PLAN_SHA, events_sha256=EVENTS_SHA, events=reviewed,
                first_public_version_verified=False, complete_correction_history_verified=False,
                certified_daily_security_master=False, fact_admissions=0, model_admissions=0, strategy_admissions=0,
                remaining_rejections=[{'event_id': r['event_id'], 'codes': r['remaining_rejection_codes']} for r in reviewed if r['remaining_rejection_codes']],
                limitations=['All four monetary cells first usable only at the current release next session.',
                             'Archived current-retrieval pages do not certify their first-public historical versions.',
                             'Two FY2015 comparison starts remain unknown and cannot support an annual-duration factor.',
                             'Microsoft FY2023 depreciation useful-life changes remain an as-reported interpretation limitation.',
                             'This audit neither admits financial data nor authorizes a strategy experiment.'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write-review', action='store_true')
    args = parser.parse_args()
    result = run_review()
    if args.write_review:
        raw = (json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()
        with (ROOT / OUTPUT).open('xb') as handle:
            handle.write(raw)
        print(json.dumps({'passed': True, 'events': 24, 'cells': 96, 'remaining_rejections': result['remaining_rejections'], 'path': OUTPUT, 'sha256': digest(raw)}))
    else:
        print(json.dumps({'passed': True, 'events': 24, 'cells': 96, 'remaining_rejections': result['remaining_rejections']}))


if __name__ == '__main__':
    main()
