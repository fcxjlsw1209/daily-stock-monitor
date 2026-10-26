"""Independent source-only audit. Imports no producer, runner or extraction helper.

The pre-extraction context chooses report/table/row/year anchors. This module
reconstructs HTML spans, headers, signs and exact monetary text independently.
It never fills a frozen null or admits a source for financial use.
"""
import argparse
from bisect import bisect_right
from collections import Counter
from datetime import date, datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import sys
from bs4 import BeautifulSoup, NavigableString

ROOT = Path(__file__).resolve().parents[1]
PLAN = 'outputs/monthly_financial_coverage_batch1_v1/PLAN.json'
EXTRACTIONS = 'outputs/monthly_financial_coverage_batch1_v1/EXTRACTIONS.json'
PLAN_SHA = 'e7f86db539c1f727c6684197ba8b8bc5cd448b233f0a72caea55f92729177fa9'
EXTRACTIONS_SHA = '4f34aa88dbffded0e303e324ba1a9936f2e67181bc97dcba5df4e1b753761b28'
SELF = 'work/after149_batch1_independent_review.py'
TEST = 'work/after149_batch1_independent_review_test.py'
OUTPUT = 'work/after149_batch1_independent_review.json'
FIELDS = ('annual_net_income', 'current_total_assets', 'prior_total_assets')
MONTHS = {}
for i, word in enumerate(('January February March April May June July August September October November December').split(), 1):
    MONTHS[word.lower()] = i
    MONTHS[word[:3].lower()] = i
DATE_RE = re.compile(r'\b(' + '|'.join(sorted(MONTHS, key=len, reverse=True)) + r')\.?\s+(\d{1,2}),?\s+(\d{4})\b', re.I)
ANNUAL = re.compile(r'\b(?:twelve|12)\s+months?\s+ended\b|\byears?\s+ended\b', re.I)
SHORT = re.compile(r'\b(?:three|six|nine|3|6|9)\s+months?\s+ended\b', re.I)
NI = re.compile(r'net\s+(?:income|loss)(?:\s*\((?:income|loss)\))?', re.I)
BODY = re.compile(r'(?:operating activities|assets|current assets|net (?:income|loss).*|cash(?:\s+and\s+cash\s+equivalents)?(?:,.*)?)\s*:?', re.I)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def norm(value):
    return ' '.join(value.split()).strip()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class Reader:
    def __init__(self):
        self.hashes = {}

    def read(self, value, expected=None):
        p = Path(value)
        p = p if p.is_absolute() else ROOT / p
        require(not p.is_symlink(), 'Symlink source')
        p = p.resolve()
        rel = str(p.relative_to(ROOT))
        require('CloudStorage' not in p.parts, 'Cloud source')
        raw = p.read_bytes()
        digest = sha(raw)
        require(expected is None or digest == expected, 'SHA mismatch: ' + rel)
        require(rel not in self.hashes or self.hashes[rel] == digest, 'Input changed: ' + rel)
        self.hashes[rel] = digest
        return raw

    def load(self, value, expected=None):
        return json.loads(self.read(value, expected))


def parsed_dates(text):
    found = set()
    for month, day, year in DATE_RE.findall(text):
        try:
            found.add(date(int(year), MONTHS[month.lower()], int(day)).isoformat())
        except ValueError:
            pass
    return found


def signed_text(text):
    """Return an exact signed decimal string, using no rounded arithmetic."""
    text = norm(text).replace('\u2212', '-')
    text = re.sub(r'\s*,\s*', ',', text)
    require(text.count('$') <= 1, 'Multiple currency signs')
    if '$' in text:
        require(re.match(r'^(?:\$|\(\s*\$)', text), 'Misplaced currency sign')
        text = text.replace('$', '').strip()
    parentheses = text.startswith('(') and text.endswith(')')
    if parentheses:
        text = text[1:-1].strip()
    require('(' not in text and ')' not in text, 'Unbalanced sign')
    require(not (parentheses and text.startswith('-')), 'Double negative')
    require(re.fullmatch(r'-?\s*(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?', text), 'Ambiguous monetary syntax')
    text = text.replace(',', '').replace(' ', '')
    return ('-' if parentheses else '') + text


def grid_from_html(table):
    """Independent physical-cell map, including occupied rowspans."""
    rows = [r for r in table.find_all('tr') if r.find_parent('table') is table]
    cells, occupancy = {}, {}
    for ri, row in enumerate(rows):
        x = 0
        for pi, node in enumerate(row.find_all(['td', 'th'], recursive=False)):
            while (ri, x) in occupancy:
                x += 1
            width, height = int(node.get('colspan', 1)), int(node.get('rowspan', 1))
            require(1 <= width <= 100 and 1 <= height <= 100, 'Invalid span')
            item = {'row': ri, 'physical_cell': pi, 'columns': list(range(x, x + width)),
                    'rowspan': height, 'text': norm(node.get_text(' ', strip=True)),
                    'raw_text': node.get_text(), 'markup_sha256': sha(str(node).encode())}
            cells[ri, pi] = item
            for dy in range(height):
                for dx in range(width):
                    key = ri + dy, x + dx
                    require(key not in occupancy, 'Overlapping span')
                    occupancy[key] = item
            x += width
    return cells, occupancy


def source_coordinates(context, symbol, field):
    side = 'comparison' if field == 'prior_total_assets' else 'current'
    if symbol == 'AMZN':
        spec = context['cf' if field == 'annual_net_income' else 'assets']
        return spec['table'], spec['NI_row' if field == 'annual_net_income' else 'row'], spec[side + '_header_columns']
    matches = []
    title = re.compile('consolidated statements? of cash flows?', re.I) if field == 'annual_net_income' else re.compile('consolidated balance sheets?', re.I)
    ending = context['periods'][side]['period_end']
    for table in context['tables']:
        if not any(title.search(t) for t in table['title']):
            continue
        columns = [c['column'] for c in table['columns'] if c['dates'] == [ending] and
                   (field != 'annual_net_income' or (c['annual'] and not c['quarter']))]
        for row in table['labels']:
            label = row['label'].rstrip(':').strip()
            if (NI.fullmatch(label) if field == 'annual_net_income' else label.lower() == 'total assets'):
                matches.append((table['table'], row['row'], columns))
    require(len(matches) == 1 and matches[0][2], 'Frozen context not unique')
    return matches[0]


def left_label(cells, row, position):
    # Statement labels may follow empty indentation cells.
    return not any(c['text'] for (r, p), c in cells.items() if r == row and p < position)


def external_heading(table):
    # Only text between this table and the nearest preceding formal title;
    # never take text from a prior table or more distant non-GAAP discussion.
    parts = []
    for node in table.previous_elements:
        if getattr(node, 'name', None) == 'table':
            break
        if not isinstance(node, NavigableString):
            continue
        if node.find_parent('table') is not None:
            break
        text = norm(str(node))
        if not text:
            continue
        parts.insert(0, text)
        combined = ' '.join(parts)
        if re.search(r'(?:condensed\s+)?consolidated\s+(?:statements?\s+of\s+cash\s+flows?|balance\s+sheets?)', combined, re.I):
            return parts
        if len(parts) > 25 or len(combined) > 1200:
            break
    return []


def header_facts(cells, row, columns, field, ending, outside=None):
    body_rows = [r for (r, p), c in cells.items() if r <= row and left_label(cells, r, p) and BODY.fullmatch(c['text'])]
    stop = min(body_rows + [row])
    headers = [c for (r, _), c in cells.items() if r < stop]
    title_regex = re.compile(r'(?:condensed\s+)?consolidated\s+statements?\s+of\s+cash\s+flows?', re.I) if field == 'annual_net_income' else re.compile(r'(?:condensed\s+)?consolidated\s+balance\s+sheets?', re.I)
    titles = [c for c in headers if title_regex.search(c['text'].rstrip(':').strip())]
    if not titles and outside:
        combined = ' '.join(outside)
        matches = list(title_regex.finditer(combined))
        require(len(matches) == 1, 'External formal title not unique')
        require(not re.search(r'non[ -]*gaap|pro forma', combined, re.I), 'Non-GAAP external header')
        for i, text in enumerate(outside):
            headers.append({'row': -1, 'physical_cell': i, 'columns': list(columns), 'text': text,
                            'markup_sha256': sha(text.encode()), 'external_source_text': True})
        titles = [matches[0].group(0)]
    require(len(titles) == 1, 'Formal title not unique above source row')
    require(not any(re.search(r'non[ -]*gaap|pro forma', c['text'], re.I) for c in headers), 'Non-GAAP header')
    units = {u.lower() for c in headers for u in re.findall(r'\bin\s+(millions|thousands)\b', c['text'], re.I)}
    require(len(units) == 1, 'Missing or conflicting scale')
    scale = '1000000' if units == {'millions'} else '1000'
    evidence = []
    for column in columns:
        hs = [c for c in headers if column in c['columns']]
        text = ' '.join(c['text'] for c in hs)
        years = set(re.findall(r'\b(?:19|20)\d{2}\b', text))
        require(years == {ending[:4]}, 'Wrong or ambiguous year at anchor')
        full_dates = parsed_dates(text)
        if full_dates:
            require(full_dates == {ending}, 'Wrong explicit date at anchor')
        else:
            require(ending.endswith('-12-31') and re.search(r'December\s+31\b', text, re.I), 'Missing full date')
        if field == 'annual_net_income':
            require(ANNUAL.search(text) and not SHORT.search(text), 'Not exclusively annual')
        evidence.extend(hs)
    unique = {(c['row'], c['physical_cell']): c for c in evidence}
    return stop, headers, list(unique.values()), scale


def independently_read_field(soup, context, symbol, field, ending):
    ti, row, columns = source_coordinates(context, symbol, field)
    tables = soup.find_all('table')
    require(0 <= ti < len(tables), 'Table outside source')
    table = tables[ti]
    require(not table.find('table'), 'Nested statement table')
    cells, grid = grid_from_html(table)
    labels = [c for (r, p), c in cells.items() if left_label(cells, r, p) and
              (NI.fullmatch(c['text'].rstrip(':')) if field == 'annual_net_income' else c['text'].rstrip(':').lower() == 'total assets')]
    require(len(labels) == 1 and labels[0]['row'] == row, 'Independent row identity differs')
    stop, headers, header_evidence, scale = header_facts(cells, row, columns, field, ending, external_heading(table))
    if field == 'annual_net_income':
        later = [(r, c['text']) for (r, p), c in cells.items() if left_label(cells, r, p) and r > row]
        adjustments = [r for r, t in later if re.search(r'adjustments? to reconcile.*(?:income|loss)', t, re.I)]
        operating_totals = [r for r, t in later if re.fullmatch(r'(?:net\s+)?cash\s+(?:provided\s+by|from|generated\s+by)(?:\s*\(used in\))?\s+operating\s+activities(?:\s*\(\d+\))?', t, re.I)]
        require(adjustments and operating_totals and min(adjustments) < min(operating_totals), 'Not indirect consolidated NI start')
        require(not any(re.match(r'(?:net\s+(?:income|loss)|(?:income|earnings|profit)\s+(?:before|from continuing))', c['text'], re.I)
                        for (r, p), c in cells.items() if left_label(cells, r, p) and r < row), 'Earlier earnings scope')
    selected = {(grid[row, col]['row'], grid[row, col]['physical_cell']): grid[row, col]
                for col in columns if (row, col) in grid}
    require(selected, 'No source cells')
    ordered = sorted((c for (r, _), c in cells.items() if r == row), key=lambda c: c['physical_cell'])
    chosen = sorted(selected.values(), key=lambda c: c['physical_cell'])
    # A year printed over a single numeric cell may leave sign-only cells
    # adjacent. Complete that one syntactic amount, never add a second number.
    indices = [ordered.index(c) for c in chosen]
    lo, hi = min(indices), max(indices)
    if symbol == 'AMZN':
        if lo and re.fullmatch(r'[\s$()\-\u2212]+', ordered[lo - 1]['text']) and ordered[lo - 1]['text']:
            chosen.insert(0, ordered[lo - 1])
        if hi + 1 < len(ordered) and ordered[hi + 1]['text'] == ')':
            chosen.append(ordered[hi + 1])
    for c in chosen:
        require(c['row'] == row and c['rowspan'] == 1, 'Cross-row amount')
        for col in c['columns']:
            text = ' '.join(h['text'] for h in headers if col in h['columns'])
            years = set(re.findall(r'\b(?:19|20)\d{2}\b', text))
            require(not years or years == {ending[:4]}, 'Cross-year amount cell')
            if field == 'annual_net_income':
                require(not SHORT.search(text), 'Amount invades short period')
    displayed = ' '.join(c['text'] for c in chosen).strip()
    value = signed_text(displayed)
    table_text = table.get_text(' ', strip=True)
    require('$' in table_text and not re.search(r'[€£¥]|\b(?:EUR|GBP|JPY|CAD|AUD|HKD)\b', table_text), 'Currency evidence conflict')
    return {'value_decimal': value, 'displayed_amount': displayed, 'scale': scale,
            'currency_literal': '$', 'as_of': ending, 'table': ti, 'row': row,
            'anchor_columns': columns, 'physical_cells': chosen, 'headers': header_evidence,
            'statement_title': [c['text'] for c in headers if re.search(r'consolidated.*(?:cash flows|balance sheets)', c['text'], re.I)],
            'header_end_row': stop}


def receipt_evidence(reader, row, context, raw):
    receipt = reader.load(row['receipt_path'])
    items = [receipt] if 'sha256' in receipt else list(receipt.values())
    candidates = [v for v in items if isinstance(v, dict) and v.get('sha256') == sha(raw)]
    require(len(candidates) == 1, 'Receipt missing or ambiguous for raw bytes')
    item = candidates[0]
    require(item.get('status', item.get('http_status')) == 200, 'Receipt not HTTP200')
    if 'bytes' in item:
        require(item['bytes'] == len(raw), 'Receipt byte length')
    stamp = datetime.fromisoformat(item['retrieved_at'].replace('Z', '+00:00'))
    require(stamp.tzinfo is not None, 'Receipt retrieval timezone')
    url = item['url']
    require(url.startswith('https://press.aboutamazon.com/') if row['symbol'] == 'AMZN' else url.startswith('https://newsroom.cisco.com/'), 'Issuer URL mismatch')
    if row['symbol'] == 'CSCO':
        require(url == context['source']['url'], 'Receipt URL differs from context')
    return {'path': row['receipt_path'], 'sha256': reader.hashes[row['receipt_path']], 'url': url,
            'http_status': 200, 'retrieved_at': item['retrieved_at']}


def review_event(reader, row, event, context, sessions):
    require(event['event_id'] == row['event_id'] and event['symbol'] == row['symbol'], 'Event identity')
    require(event['fiscal_year'] == row['fiscal_year'], 'Fiscal year differs')
    raw = reader.read(row['source_path'], row['source_sha256'])
    receipt = receipt_evidence(reader, row, context, raw)
    soup = BeautifulSoup(raw, 'html.parser')
    for n in soup(['script', 'style']):
        n.decompose()
    text = norm(soup.get_text(' ', strip=True))
    symbol = row['symbol']
    require(re.search(r'Amazon\.com,?\s+Inc\.?', text, re.I) if symbol == 'AMZN' else re.search(r'Cisco\s+Systems,?\s+Inc', text, re.I), 'Raw issuer name')
    require(re.search(r'NASDAQ\s*[:(]\s*' + symbol + r'\b', text, re.I), 'Raw security anchor')
    released = context['publication_date'] if symbol == 'AMZN' else context['release_date']
    available = context['available_session']
    require(released in parsed_dates(text), 'Release date absent from raw text')
    require(event.get('publication_date', event.get('release_date')) == released, 'Extracted release mismatch')
    require(event['available_from'] == available and released < available, 'Available date mismatch')
    if symbol == 'AMZN':
        current, prior = context['current_end'], context['comparison_end']
        start = current[:4] + '-01-01'
    else:
        current = context['periods']['current']['period_end']
        prior = context['periods']['comparison']['period_end']
        start = context['periods']['current']['period_start']
    days = (date.fromisoformat(current) - date.fromisoformat(start)).days + 1
    require(330 <= days <= 400 and current < released, 'Annual duration or release order')
    require((date.fromisoformat(start) - date.fromisoformat(prior)).days == 1, 'Prior closing assets not opening boundary')
    expected_period = {'period_start': start, 'period_end': current, 'days_inclusive': days}
    require(event['periods']['current'] == expected_period, 'Current annual period differs')
    clock = False
    if sessions[0] <= released < sessions[-1]:
        require(sessions[bisect_right(sessions, released)] == available, 'Not first next pinned session')
        clock = True
    require(set(event['fields']) == set(FIELDS), 'Field census')
    require(all(event[k] is False for k in ('fact_admitted', 'model_admitted', 'strategy_admitted')), 'Unintended admission')
    fields = {}
    for field in FIELDS:
        end = prior if field == 'prior_total_assets' else current
        source = independently_read_field(soup, context, symbol, field, end)
        cell = event['fields'][field]
        result = {'verified': True, 'is_null': cell is None, 'null_reason_verified': None,
                  'source_value_present': True, 'source': {'path': row['source_path'], 'sha256': sha(raw)},
                  'independent_source_reading': source}
        if cell is None:
            gaps = [g for g in event['gaps'] if g['field'] == field]
            require(len(gaps) == 1 and gaps[0]['reason'], 'Null missing explicit reason')
            result.update(null_reason_verified=True, frozen_null_preserved=True,
                          null_classification='source_value_present_but_frozen_parser_no_candidate',
                          null_diagnosis='Formal balance-sheet title is inside the source table. An unrelated preceding non-GAAP reconciliation notice contaminates the frozen parser external context and causes its BAD-context rejection before candidate collection.',
                          frozen_gap=gaps[0], value_admitted=False)
        else:
            require(Decimal(cell['value_decimal']) == Decimal(source['value_decimal']), field + ': numeric mismatch')
            require(Decimal(signed_text(cell['displayed_amount'])) == Decimal(source['value_decimal']), field + ': displayed sign mismatch')
            require(cell['scale'] == source['scale'] and cell['currency_literal'] == '$', field + ': units mismatch')
            require(cell.get('currency_iso') is None and cell['fact_admitted'] is False, 'Unintended field certification')
            require(cell['source']['path'] == row['source_path'] and cell['source']['sha256'] == row['source_sha256'], field + ': source mismatch')
            require(cell['available_from'] == available and cell['as_of'] == end, field + ': clock mismatch')
            if field == 'annual_net_income':
                require(cell['period'] == expected_period, 'NI period mismatch')
            else:
                require(Decimal(source['value_decimal']) > 0, 'Nonpositive asset source')
            loc = cell['locator']
            require(loc.get('table', loc.get('table_index_zero_based')) == source['table'] and loc.get('row', loc.get('row_index_zero_based')) == source['row'], 'Published locator differs')
            published = loc['physical_cells']
            pcoords = {c.get('physical_cell', c.get('physical_cell_index_zero_based')) for c in published}
            require(pcoords == {c['physical_cell'] for c in source['physical_cells']}, 'Published physical-cell membership differs')
            result.update(value_and_sign_verified=True, period_verified=True, units_verified=True, source_locator_verified=True)
        fields[field] = result
    return {'event_id': row['event_id'], 'symbol': symbol, 'fiscal_year': row['fiscal_year'], 'passed': True,
            'metadata_verified': True, 'identity_verified': True, 'periods_verified': True, 'currency_verified': True,
            'clock_verified': clock, 'clock_scope': 'first_pinned_next_session' if clock else 'outside_pinned_calendar_not_certified',
            'release_date': released, 'available_from': available, 'receipt': receipt,
            'source': {'path': row['source_path'], 'sha256': sha(raw)}, 'fields': fields,
            'retained_context_holds': event.get('context_holds', []),
            'first_public_version_verified': False, 'certified_daily_security_master': False,
            'source_admitted': False}


def audit(publish=False):
    reader = Reader()
    plan = reader.load(PLAN, PLAN_SHA)
    data = reader.load(EXTRACTIONS, EXTRACTIONS_SHA)
    require(data['plan_sha256'] == PLAN_SHA, 'Extraction/plan binding')
    for path, digest in {**plan['inputs'], **plan['code']}.items():
        reader.read(path, digest)
    reader.read(SELF)
    reader.read(TEST)
    sessions = reader.load('outputs/monthly_forward_guidance_preflight_v1/SESSIONS.json')['sessions']
    require(sessions == sorted(set(sessions)), 'Pinned sessions not sorted unique')
    require(len(plan['records']) == len(data['events']) == 26, '26-event census')
    index = {e['event_id']: e for e in data['events']}
    require(len(index) == 26 and set(index) == {r['event_id'] for r in plan['records']}, 'Event set mismatch')
    contexts, events, failures = {}, [], []
    for row in plan['records']:
        cp = row['context_path']
        if cp not in contexts:
            document = reader.load(cp)
            contexts[cp] = {e['event_id']: e for e in document['documents' if row['symbol'] == 'AMZN' else 'events']}
        try:
            result = review_event(reader, row, index[row['event_id']], contexts[cp][row['event_id']], sessions)
            events.append(result)
        except Exception as exc:
            failures.append({'event_id': row['event_id'], 'error_type': type(exc).__name__, 'error': str(exc)})
    fields = [f for e in events for f in e['fields'].values()]
    nonnull = sum(not f['is_null'] for f in fields)
    nulls = sum(f['is_null'] for f in fields)
    passed = not failures and nonnull == 76 and nulls == 2
    for path, digest in list(reader.hashes.items()):
        reader.read(path, digest)
    result = {'schema_version': 1, 'stage': 'independent_source_value_and_null_classification_review',
              'created_utc': datetime.now(timezone.utc).isoformat(), 'passed': passed,
              'plan_sha256': PLAN_SHA, 'extractions_sha256': EXTRACTIONS_SHA,
              'scope': 'Verify frozen 26 reports, 76 published NI/assets values and two explicit nulls against source HTML. Does not admit data or fill nulls.',
              'coverage_complete': False, 'event_reviews': events, 'failures': failures,
              'counters': {'events': len(events), 'published_nonnull_values_verified': nonnull,
                           'null_fields_reviewed': nulls, 'source_value_present_fields': sum(f['source_value_present'] for f in fields),
                           'events_with_verified_pinned_next_session': sum(e['clock_verified'] for e in events),
                           'by_symbol': dict(Counter(e['symbol'] for e in events)), 'input_hashes_verified': len(reader.hashes)},
              'input_hashes': reader.hashes, 'runtime': sys.version,
              'independence': 'Own HTML span reconstruction, date/header parsing and exact signed-decimal syntax. No producer module/helper imports. Context coordinates were frozen before extraction.',
              'limitations': ['The two frozen CSCO FY2011 asset nulls remain null even if their original cells are readable.',
                              'Outside-calendar availability is retained but not independently calendar-certified.',
                              'Source values are not an accounting-comparability or financial-use admission.',
                              'First-public versions, complete revision histories, ISO currency and daily security master are not newly certified.',
                              'All prior source decisions and 13 legacy holds remain unchanged; no ratios, factors, orders or returns are calculated.'],
              'financial_calculations': 0, 'factor_calculations': 0, 'returns_computed': 0, 'new_admissions': 0}
    if publish:
        require(passed, 'Review failed; no final receipt published')
        with (ROOT / OUTPUT).open('x') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
    print(json.dumps({'passed': passed, 'counters': result['counters'], 'failures': failures,
                      'receipt_sha256': sha((ROOT / OUTPUT).read_bytes()) if publish else None}, ensure_ascii=False))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    result = audit(args.publish)
    raise SystemExit(0 if result['passed'] else 1)
