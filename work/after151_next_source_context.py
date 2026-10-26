"""KO/LRCX pre-extraction metadata only; never parse or emit monetary cells.

Build from the bounded 32-document feasibility census. Standard HTML headers
are re-read for date/column ownership; PRE/PDF geometry stays explicitly held.
"""
from collections import Counter
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re

from bs4 import BeautifulSoup
from after141_amd_source_extract import table_grid
import after150_next_source_context as metadata

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = 'work/after151_next_source_feasibility.json'
PREVIOUS_SHA = '2117201c32e4a8defec8d726b05bf9fb731f1af071406ca7a33325336b704262'
OUTPUT = 'work/after151_next_source_context.json'
MONTH = metadata.MONTH
DATE = re.compile(r'\b(' + MONTH + r')\s+(\d{1,2})\s*,?\s+((?:19|20)\d{2})\b', re.I)
ANNUAL = re.compile(r'^(?:(?:fiscal\s+)?years?|twelve\s+months?|12\s+months?)\s+ended$', re.I)
INCOME = re.compile(r'consolidated\s+statements?\s+of\s+(?:income|operations)\b', re.I)
BALANCE = re.compile(r'consolidated\s+balance\s+sheets?\b', re.I)
TOTAL_NI = re.compile(r'^(?:consolidated\s+)?net\s+(?:income|loss)(?:\s*\(loss\))?\s*$', re.I)
PARENT_NI = re.compile(r'^net\s+(?:income|loss).*attributable\s+to', re.I)
BODY = re.compile(r'^(?:assets|current assets|cash(?: and cash equivalents)?|cash and short.term investments|net operating revenues|net revenues?|revenues?|net sales|sales|operating activities)\s*:?$', re.I)
SHA = re.compile(r'^[0-9a-f]{64}$')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def norm(s):
    return metadata.norm(s)


def read_local(path, expected, pins):
    p = ROOT / path
    if p.is_symlink() or not p.resolve().is_relative_to(ROOT) or 'CloudStorage' in p.parts:
        raise ValueError('Nonlocal input: ' + path)
    raw = p.read_bytes()
    actual = digest(raw)
    if expected is not None and actual != expected:
        raise ValueError('Hash mismatch: ' + path)
    pins[path] = actual
    return raw


def parsed_dates(parts):
    text = norm(' '.join(parts))
    result = set()
    for m in DATE.finditer(text):
        month = m.group(1).rstrip('.')
        text_date = '%s %s %s' % (month, m.group(2), m.group(3))
        for fmt in ('%B %d %Y', '%b %d %Y'):
            try:
                result.add(datetime.strptime(text_date, fmt).date().isoformat())
                break
            except ValueError:
                pass
    return sorted(result)


def cell_metadata(cell):
    return {k: cell[k] for k in ('row', 'physical_cell', 'columns', 'rowspan')} | {'text': norm(cell['text'])}


def table_context(table, ti, symbol):
    """Only whitelisted header strings and row labels leave this function."""
    rows, grid, physical = table_grid(table)
    cells = list(physical.values())
    ni = [c for c in cells if TOTAL_NI.fullmatch(norm(c['text']))]
    assets = [c for c in cells if norm(c['text']).lower() == 'total assets']
    targets = ni + assets
    if not targets:
        return None
    first = min(c['row'] for c in targets)
    body = [c['row'] for c in cells if c['row'] < first and BODY.fullmatch(norm(c['text']))]
    if not body:
        return None
    stop = min(body)
    before = [c for c in cells if c['row'] < stop]
    titles = [c for c in before if INCOME.search(norm(c['text'])) or BALANCE.search(norm(c['text']))]
    if not titles:
        return None
    texts = [norm(c['text']) for c in before]
    if any(re.search(r'non.gaap|reconciliation', t, re.I) for t in texts):
        return None
    types = set('income' if INCOME.search(norm(c['text'])) else 'balance' for c in titles)
    if len(types) != 1:
        return None
    kind = next(iter(types))
    labels = ni if kind == 'income' else assets
    # A formal income table with only parent-attributable NI is not promoted.
    if len(labels) != 1:
        return None
    label = labels[0]
    annual = [c for c in before if ANNUAL.fullmatch(norm(c['text']))]
    allowed = set(col for c in annual for col in c['columns']) if kind == 'income' else set(gridcol for _, gridcol in grid)
    groups = {}
    for col in sorted(allowed):
        parts = [norm(c['text']) for c in before if col in c['columns']]
        ds = parsed_dates(parts)
        if len(ds) == 1:
            groups.setdefault(ds[0], []).append(col)
    headers = [cell_metadata(c) for c in before if metadata.is_metadata(norm(c['text']))
               or parsed_dates([norm(c['text'])])
               or re.fullmatch(MONTH + r'\s+\d{1,2}\s*,?', norm(c['text']), re.I)]
    units = [cell_metadata(c) for c in before if metadata.UNIT.fullmatch(norm(c['text']))]
    scales = set('1000000' if 'million' in c['text'].lower() else '1000' if 'thousand' in c['text'].lower() else '1000000000' for c in units)
    row_geometry = [{k: c[k] for k in ('row', 'physical_cell', 'columns', 'rowspan')} for c in cells if c['row'] == label['row']]
    parent = [cell_metadata(c) for c in cells if PARENT_NI.fullmatch(norm(c['text']))]
    return {'table': ti, 'kind': kind, 'header_stop_row': stop,
            'headers': headers, 'annual_header_locators': [cell_metadata(c) for c in annual],
            'date_columns': groups, 'label': cell_metadata(label),
            'parent_attributable_labels_excluded': parent,
            'unit_evidence': units, 'scale': next(iter(scales)) if len(scales) == 1 else None,
            'currency_literal': '$' if '$' in table.get_text() else None,
            'currency_iso': None, 'currency_assumption': 'same_report_dollar_literal_only',
            'row_geometry_without_values': row_geometry,
            'monetary_cell_geometry_verified': False}


def choose_pair(tables, fy, claimed_end):
    income = [t for t in tables if t['kind'] == 'income' and t['annual_header_locators']]
    balance = [t for t in tables if t['kind'] == 'balance']
    pairs = []
    for inc in income:
        cur = [d for d in inc['date_columns'] if int(d[:4]) == fy]
        prev = [d for d in inc['date_columns'] if int(d[:4]) == fy - 1]
        if len(cur) != 1 or len(prev) != 1 or cur[0] != claimed_end:
            continue
        for bs in balance:
            if cur[0] in bs['date_columns'] and prev[0] in bs['date_columns']:
                pairs.append((inc, bs, cur[0], prev[0]))
    if len(pairs) != 1:
        return None, ['unique_same_report_annual_NI_assets_date_pair_not_established']
    inc, bs, cur, prev = pairs[0]
    if inc['scale'] is None or bs['scale'] is None or inc['currency_literal'] is None or bs['currency_literal'] is None:
        return None, ['unit_or_currency_context_unresolved']
    def locator(t, d):
        return {'table': t['table'], 'row': t['label']['row'],
                'label_cell': t['label']['physical_cell'], 'label': t['label']['text'],
                'logical_columns': t['date_columns'][d], 'as_of': d,
                'scale': t['scale'], 'currency_literal': t['currency_literal'], 'currency_iso': None,
                'header_evidence': t['headers'], 'row_geometry_without_values': t['row_geometry_without_values'],
                'monetary_cell_geometry_verified': False}
    return {'fields': {'annual_net_income': locator(inc, cur),
                       'current_total_assets': locator(bs, cur),
                       'prior_total_assets': locator(bs, prev)},
            'periods': {'current_end': cur, 'prior_end': prev,
                        'proposed_current_start': (date.fromisoformat(prev) + timedelta(days=1)).isoformat(),
                        'start_is_source_quoted': False,
                        'start_assumption': 'same_current_report_prior_annual_end_plus_one_calendar_day',
                        'start_assumption_approved': False},
            'excluded_parent_NI_labels': inc['parent_attributable_labels_excluded']}, []


def self_check():
    assert parsed_dates(['June 30,', '2013']) == ['2013-06-30']
    assert parsed_dates(['June 30 ,', '2013']) == ['2013-06-30']
    assert parsed_dates(['June 30, 2013', 'June 24, 2012']) == ['2012-06-24', '2013-06-30']
    assert TOTAL_NI.fullmatch('Consolidated Net Income')
    assert not TOTAL_NI.fullmatch('Net Income Attributable to Shareowners')
    assert not TOTAL_NI.fullmatch('Net income per share')
    assert ANNUAL.fullmatch('Twelve Months Ended') and not ANNUAL.fullmatch('Three Months Ended')


def main():
    self_check()
    pins = {}
    previous = json.loads(read_local(PREVIOUS, PREVIOUS_SHA, pins))
    docs = previous['documents']
    if len(docs) != 32 or len({x['event_id'] for x in docs}) != 32:
        raise ValueError('Expected complete unique 32-event census')
    output = []
    for doc in docs:
        raw = read_local(doc['evidence_path'], doc['evidence_sha256'], pins)
        receipt = json.loads(read_local(doc['receipt']['path'], doc['receipt']['sha256'], pins))
        if (receipt.get('sha256') != doc['evidence_sha256'] or receipt.get('url') != doc['source_url']
            or receipt.get('status', receipt.get('http_status')) != 200):
            raise ValueError('Source receipt mismatch')
        item = {k: v for k, v in doc.items() if k not in ('context',)}
        item['inherited_feasibility_context'] = doc['context']
        item['source'] = {'path': doc['evidence_path'], 'sha256': doc['evidence_sha256'], 'url': doc['source_url'],
                          'receipt_path': doc['receipt']['path'], 'receipt_sha256': doc['receipt']['sha256']}
        holds = ['monetary_cells_and_sign_geometry_not_yet_independently_verified',
                 'currency_ISO_and_first_public_version_not_certified']
        if not doc['available_session_verified']:
            holds.append('publication_outside_pinned_calendar_range')
        if doc['accounting_terms']:
            holds.append('accounting_notes_require_field_specific_independent_review')
        item.update(fields={k: None for k in ('annual_net_income', 'current_total_assets', 'prior_total_assets')},
                    periods=None, statement_contexts=[], fact_admitted=False, model_admitted=False, strategy_admitted=False)
        if raw.startswith(b'%PDF'):
            status = 'pdf_text_leads_only_geometry_pending'
            holds.append('PDF_header_row_and_column_geometry_not_verified')
        elif doc['context'].get('pre_candidates'):
            status = 'PRE_text_leads_only_geometry_pending'
            holds.append('PRE_fixed_width_header_row_and_column_geometry_not_verified')
        else:
            soup = BeautifulSoup(raw, 'html.parser')
            tables = []
            for ti, table in enumerate(soup.find_all('table')):
                if table.find('table'):
                    continue
                try:
                    context = table_context(table, ti, doc['symbol'])
                except ValueError:
                    context = None
                if context is not None:
                    tables.append(context)
            item['statement_contexts'] = tables
            selected, gaps = choose_pair(tables, doc['fiscal_year'], doc['official_actual_period_end'])
            if selected:
                item.update(selected)
                for field in item['fields'].values():
                    field['source'] = dict(item['source'])
                    field['available_from'] = doc['available_session']
                    field['availability_verified'] = doc['available_session_verified']
                status = 'standard_HTML_annual_header_context_located_not_admitted'
                holds.append('proposed_annual_start_boundary_assumption_not_yet_approved')
            else:
                status = 'current_page_formal_pair_unresolved'
                holds += gaps
                if not tables:
                    holds.append('linked_statement_attachment_not_inventoried_in_this_scope')
        item['context_status'] = status
        item['context_holds'] = sorted(set(holds))
        item['page_without_table_does_not_prove_report_missing'] = True
        output.append(item)
    for path in ('work/after151_next_source_context.py', 'work/after150_next_source_context.py', 'work/after141_amd_source_extract.py'):
        read_local(path, None, pins)
    result = {'schema_version': 1, 'stage': 'pre_extraction_source_structure_context_not_admission',
              'created_utc': datetime.now(timezone.utc).isoformat(), 'source_scope': ['KO', 'LRCX'],
              'parent_feasibility': {'path': PREVIOUS, 'sha256': PREVIOUS_SHA},
              'events': output, 'input_hashes': pins,
              'counts': {'catalog_events': len(output), 'by_symbol': dict(Counter(d['symbol'] for d in output)),
                         'by_context_status': dict(Counter(d['context_status'] for d in output)),
                         'clock_verified': sum(d['available_session_verified'] for d in output)},
              'coordinate_convention': 'HTML table,row,physical_cell,logical_column zero based; inherited PDF pages/lines one based',
              'scope_limits': ['No source downloads or linked attachment reads. No numeric cell parsing or monetary output.',
                               'Preserve all32 catalog entries and prior metadata. Unresolved newest events must later block fallback.',
                               'Total consolidated NI is distinct from parent-attributable NI; never substitute the latter.',
                               'No inferred GAAP/currency/version/security-master certification from a title or dollar sign.',
                               'Field numeric geometry, accounting comparability and boundary assumptions require separate review before admission.'],
              'financial_values_extracted': 0, 'ratios_computed': 0, 'new_factors_computed': 0,
              'orders_or_returns_computed': 0, 'new_admissions': 0, 'new_strategy_trials': 0}
    target = ROOT / OUTPUT
    with target.open('x') as stream:
        stream.write(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'counts': result['counts'], 'path': OUTPUT, 'sha256': digest(target.read_bytes())}, sort_keys=True))


if __name__ == '__main__':
    main()
