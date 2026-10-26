"""Pure KO FY2011–2016 annual NI/assets extraction; never performs I/O.

The caller binds the frozen context and receipt. Raw bytes, source references,
statement headers and physical geometry are rechecked here. Successful fields
remain unadmitted candidates and do not clear any inherited context hold.
"""
import copy
import datetime as dt
import re
from urllib.parse import urlsplit

from bs4 import BeautifulSoup
import after149_amzn_extract as old
from after141_amd_source_extract import table_grid

FIELDS = old.FIELDS
YEARS = tuple(range(2011, 2017))
STATUS = 'standard_HTML_annual_header_context_located_not_admitted'
SourceError, FieldGap, sha256 = old.SourceError, old.FieldGap, old.sha256
INCOME = re.compile(r'(?:condensed\s+)?consolidated\s+statements?\s+of\s+income', re.I)
BALANCE = re.compile(r'(?:condensed\s+)?consolidated\s+balance\s+sheets?', re.I)
BODY = re.compile(r'net operating revenues|net revenues?|revenues?|net sales|sales|assets|current assets|cash(?: and cash equivalents)?', re.I)
NI = re.compile(r'consolidated net income', re.I)
ASSETS = re.compile(r'total assets', re.I)
SHA = re.compile(r'[0-9a-f]{64}')
GEOMETRY = ('row', 'physical_cell', 'columns', 'rowspan')


def source_files():
    return ('work/after151_ko_extract.py', 'work/test_after151_ko_extract.py',
            'work/after149_amzn_extract.py', 'work/after141_amd_source_extract.py')


def _text(text):
    return old._text(text)


def _reference(c, raw):
    if not isinstance(raw, bytes) or not isinstance(c, dict):
        raise SourceError('bytes_and_context_object_required')
    if (c.get('symbol') != 'KO' or type(c.get('fiscal_year')) is not int
            or c['fiscal_year'] not in YEARS or c.get('context_status') != STATUS):
        raise SourceError('outside_fixed_issuer_year_context_scope')
    if c.get('event_id') != 'KO_' + str(c.get('publication_date')):
        raise SourceError('event_identity_mismatch')
    source = c.get('source')
    if not isinstance(source, dict):
        raise SourceError('missing_source_reference')
    required = ('path', 'sha256', 'url', 'receipt_path', 'receipt_sha256')
    if any(not isinstance(source.get(k), str) or not source[k] for k in required):
        raise SourceError('missing_source_reference')
    if (not SHA.fullmatch(source['sha256']) or sha256(raw) != source['sha256']
            or not SHA.fullmatch(source['receipt_sha256'])):
        raise SourceError('raw_source_sha256_or_receipt_digest_mismatch')
    for key in ('path', 'receipt_path'):
        path = source[key]
        if path.startswith('/') or '..' in path.split('/') or 'CloudStorage' in path:
            raise SourceError('invalid_local_source_reference')
    url = urlsplit(source['url'])
    if (url.scheme != 'https' or url.hostname != 'investors.coca-colacompany.com'
            or url.username or url.password or url.port not in (None, 443)):
        raise SourceError('invalid_official_source_url')
    if (c.get('evidence_path') != source['path'] or c.get('evidence_sha256') != source['sha256']
            or c.get('source_url') != source['url']):
        raise SourceError('conflicting_source_references')
    receipt = c.get('receipt', {})
    if receipt.get('path') != source['receipt_path'] or receipt.get('sha256') != source['receipt_sha256']:
        raise SourceError('conflicting_receipt_reference')
    holds = c.get('context_holds')
    if not isinstance(holds, list) or any(not isinstance(h, str) for h in holds):
        raise SourceError('invalid_context_holds')
    soup = BeautifulSoup(raw, 'html.parser')
    for node in soup(['script', 'style']):
        node.decompose()
    text = _text(soup.get_text(' ', strip=True))
    if not re.search(r'\b(?:The )?Coca-Cola Company\b', text, re.I):
        raise SourceError('raw_issuer_anchor_missing')
    anchors = c.get('identity_anchors')
    if not isinstance(anchors, list) or not anchors or any(
            not isinstance(a, str) or not a or _text(a).lower() not in text.lower() for a in anchors):
        raise SourceError('raw_context_identity_anchors_missing')
    return copy.deepcopy(source), soup


def _periods(c):
    p = c.get('periods')
    if not isinstance(p, dict):
        raise FieldGap('missing_period_metadata')
    current, prior = old._date(p.get('current_end')), old._date(p.get('prior_end'))
    start = old._date(p.get('proposed_current_start'))
    year = c['fiscal_year']
    if (current != dt.date(year, 12, 31) or prior != dt.date(year-1, 12, 31)
            or old._date(c.get('official_actual_period_end')) != current):
        raise FieldGap('context_year_end_mismatch')
    if start != prior + dt.timedelta(days=1) or not 330 <= (current-start).days+1 <= 400:
        raise FieldGap('derived_start_mismatch')
    if (p.get('start_is_source_quoted') is not False or p.get('start_assumption_approved') is not False
            or p.get('start_assumption') != 'same_current_report_prior_annual_end_plus_one_calendar_day'):
        raise FieldGap('unrecognized_or_upgraded_start_assumption')
    publication, available = old._date(c.get('publication_date')), old._date(c.get('available_session'))
    if not current < publication < available:
        raise FieldGap('invalid_publication_availability_order')
    return {'current': {'period_start': str(start), 'period_end': str(current),
                        'days_inclusive': (current-start).days+1},
            'comparison': {'period_end': str(prior)},
            'start_basis': 'derived_same_report_prior_annual_end_plus_one_not_directly_quoted',
            'start_assumption_certified': False}


def _field(tables, spec, kind, source, available, period, endpoint):
    if not isinstance(spec, dict):
        raise FieldGap('missing_field_context')
    if spec.get('source') != source:
        raise FieldGap('field_source_mismatch')
    if spec.get('available_from') != available or spec.get('as_of') != endpoint:
        raise FieldGap('field_period_or_availability_mismatch')
    if spec.get('currency_literal') != '$' or spec.get('currency_iso') is not None:
        raise FieldGap('unrecognized_currency_context')
    ti, row, label_cell = spec.get('table'), spec.get('row'), spec.get('label_cell')
    if (any(type(x) is not int or x < 0 for x in (ti, row, label_cell))
            or ti >= len(tables) or label_cell != 0):
        raise FieldGap('invalid_table_or_row_locator')
    table = tables[ti]
    if table.find('table') is not None:
        raise FieldGap('nested_statement_table')
    try:
        rows, grid, physical = table_grid(table)
    except ValueError as exc:
        raise FieldGap('invalid_table_geometry') from exc
    if row >= len(rows):
        raise FieldGap('invalid_table_or_row_locator')
    for (r, pi), cell in physical.items():
        cell['original_text'] = rows[r].find_all(['td', 'th'], recursive=False)[pi].get_text()
    target = NI if kind == 'income' else ASSETS
    matches = [cell for (_, pi), cell in physical.items() if pi == 0 and target.fullmatch(_text(cell['text']))]
    if len(matches) != 1 or matches[0]['row'] != row or not target.fullmatch(_text(str(spec.get('label', '')))):
        raise FieldGap('exact_unique_consolidated_row_label_not_found')
    starts = [r for (r, pi), cell in physical.items()
              if pi == 0 and r < row and BODY.fullmatch(_text(cell['text']))]
    if not starts:
        raise FieldGap('statement_body_boundary_not_found')
    stop = min(starts)
    headers = [cell for (r, _), cell in physical.items() if r < stop]
    title = INCOME if kind == 'income' else BALANCE
    title_cells = [cell for cell in headers if title.fullmatch(_text(cell['text']))]
    if len(title_cells) != 1:
        raise FieldGap('formal_consolidated_title_not_unique')
    if any(re.search(r'non[\s-]*gaap|reconciliation|as adjusted|pro forma', cell['text'], re.I) for cell in headers):
        raise FieldGap('non_gaap_statement')
    units = [cell for cell in headers if re.search(r'\bin (millions|thousands)\b', cell['text'], re.I)]
    scales = set(unit.lower() for cell in units for unit in re.findall(r'\bin (millions|thousands)\b', cell['text'], re.I))
    if len(scales) != 1:
        raise FieldGap('missing_or_conflicting_scale')
    scale = '1000000' if scales == {'millions'} else '1000'
    if spec.get('scale') != scale:
        raise FieldGap('scale_context_mismatch')
    text = table.get_text(' ', strip=True)
    if '$' not in text or old.OTHER_CURRENCY.search(text):
        raise FieldGap('missing_or_conflicting_currency')
    evidence = spec.get('header_evidence')
    if not isinstance(evidence, list) or not evidence:
        raise FieldGap('missing_fixed_header_evidence')
    for anchor in evidence:
        if not isinstance(anchor, dict):
            raise FieldGap('invalid_fixed_header_evidence')
        cell = physical.get((anchor.get('row'), anchor.get('physical_cell')))
        if (cell is None or cell['row'] >= stop or any(cell.get(k) != anchor.get(k) for k in GEOMETRY)
                or _text(cell['text']) != _text(str(anchor.get('text', '')))):
            raise FieldGap('fixed_header_evidence_mismatch')
    expected_geometry = spec.get('row_geometry_without_values')
    geometry = [{k: cell[k] for k in GEOMETRY} for cell in old._cells(physical, row)]
    if expected_geometry != geometry:
        raise FieldGap('fixed_row_geometry_mismatch')
    columns = spec.get('logical_columns')
    column_evidence = old._column_evidence(headers, columns, int(endpoint[:4]), kind == 'income')
    owned = old._owned_columns(headers, physical, row, columns, int(endpoint[:4]), kind == 'income')
    if any((row, col) not in grid for col in columns):
        raise FieldGap('incomplete_amount_geometry')
    value, display, selected = old._value(physical, grid, row, columns, owned)
    return {'displayed_amount': display, 'value_decimal': value, 'scale': scale,
            'currency_literal': '$', 'currency_iso': None,
            'currency_basis': 'same_statement_dollar_symbol_not_ISO_certification',
            'period': copy.deepcopy(period) if kind == 'income' else None,
            'as_of': endpoint, 'available_from': available, 'source': copy.deepcopy(source),
            'fact_admitted': False,
            'locator': {'table': ti, 'row': row, 'label': copy.deepcopy(matches[0]),
                        'header_columns': list(columns), 'owned_logical_columns': sorted(owned),
                        'physical_cells': copy.deepcopy(selected),
                        'header_evidence': copy.deepcopy(column_evidence),
                        'scale_evidence': copy.deepcopy(units), 'statement_title_evidence': copy.deepcopy(title_cells)}}


def extract_document(raw_bytes, context):
    """Extract exactly three candidate fields; input objects remain unchanged."""
    if not isinstance(context, dict):
        raise SourceError('bytes_and_context_object_required')
    c = copy.deepcopy(context)
    source, soup = _reference(c, raw_bytes)
    result = {'schema_version': 1, 'stage': 'source_only_annual_NI_assets_candidates',
              'symbol': 'KO', 'event_id': c['event_id'], 'fiscal_year': c['fiscal_year'],
              'identity': {'issuer': 'The Coca-Cola Company', 'symbol': 'KO',
                           'source_anchors': copy.deepcopy(c['identity_anchors']), 'certified_daily_security_master': False},
              'source': source, 'publication_date': c.get('publication_date'), 'available_from': c.get('available_session'),
              'periods': {}, 'fields': dict.fromkeys(FIELDS), 'gaps': [],
              'context_holds': copy.deepcopy(c['context_holds']),
              'accounting_review_flags': copy.deepcopy(c.get('accounting_terms', [])),
              'accounting_context_numbers_redacted': copy.deepcopy(c.get('accounting_context_numbers_redacted', [])),
              'fact_admitted': False, 'model_admitted': False, 'strategy_admitted': False,
              'first_public_version_verified': False, 'complete_correction_history_verified': False,
              'certified_daily_security_master': False, 'financial_calculations': 0,
              'notes': ['All fields use current-report availability, including prior assets.',
                        'Annual start is a proposed boundary assumption, not a directly quoted or certified date.',
                        'Inherited context holds are retained exactly; successful parsing does not confer admission.']}
    try:
        result['periods'] = _periods(c)
    except (FieldGap, KeyError, TypeError) as exc:
        result['gaps'] = [{'field': name, 'reason': str(exc)} for name in FIELDS]
        result['complete_required_fields'] = False
        return result
    current = result['periods']['current']
    for name in FIELDS:
        endpoint = result['periods']['comparison']['period_end'] if name == 'prior_total_assets' else current['period_end']
        try:
            result['fields'][name] = _field(soup.find_all('table'), c.get('fields', {}).get(name),
                                          'income' if name == 'annual_net_income' else 'assets',
                                          source, c['available_session'], current, endpoint)
        except (FieldGap, KeyError, TypeError) as exc:
            result['gaps'].append({'field': name, 'reason': str(exc)})
    result['complete_required_fields'] = not result['gaps']
    return result
