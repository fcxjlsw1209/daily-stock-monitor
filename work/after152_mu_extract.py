"""Pure MU annual NI/assets candidates; no I/O, admission, or financial ratios.

Shared private mechanics are also used by the separate PEP entry point. The
caller must freeze the context/receipt and check the next pinned session. All
period/identity assumptions and inherited holds remain explicitly unapproved.
"""
import copy
import datetime as dt
import re
from urllib.parse import urlsplit

from bs4 import BeautifulSoup
import after149_amzn_extract as old
from after141_amd_source_extract import table_grid, DATE, MONTHS

FIELDS = old.FIELDS
STATUS = 'standard_HTML_annual_header_context_located_not_admitted'
SourceError, FieldGap, sha256 = old.SourceError, old.FieldGap, old.sha256
GEOMETRY = ('row', 'physical_cell', 'columns', 'rowspan')
SHA = re.compile(r'[0-9a-f]{64}')
INCOME = re.compile(r'(?:condensed\s+)?consolidated\s+(?:condensed\s+)?statements?\s+of\s+(?:operations|income)\b', re.I)
BALANCE = re.compile(r'(?:condensed\s+)?consolidated\s+(?:condensed\s+)?balance\s+sheets?\b', re.I)
BODY = re.compile(r'(?:assets|current assets|cash(?: and cash equivalents)?|cash and short.term investments|net operating revenues|net revenues?|revenues?|net sales|sales)', re.I)
NI = re.compile(r'(?:consolidated\s+)?net\s+(?:income|loss)(?:\s*\((?:loss|income)\))?', re.I)
ASSETS = re.compile(r'total assets', re.I)
ANNUAL = re.compile(r'(?:for\s+the\s+)?(?:years?|(?:fiscal\s+years?)|twelve\s+months?|12\s+months?|52\s+weeks?|53\s+weeks?)\s+ended', re.I)
SHORT = re.compile(r'(?:quarters?|three\s+months?|six\s+months?|nine\s+months?|(?:3|6|9)\s+months?|13\s+weeks?|year.to.date|ytd)', re.I)
SLASH_DATE = re.compile(r'(\d{1,2})/(\d{1,2})/((?:19|20)\d{2}|\d{2})')
YEARS = (2024, 2025)
PROFILE = dict(symbol='MU', issuer='Micron Technology, Inc.', years=YEARS,
               hosts=('micron.gcs-web.com',), issuer_pattern=r'\bMicron Technology,?\s+Inc\.',
               ticker_pattern=r'\bNasdaq\s*:\s*MU\b', month=8, income_groups=5, balance_groups=3,
               assumption_id='MU_consecutive_annual_boundaries_v1')


def source_files():
    return ('work/after152_mu_extract.py', 'work/test_after152_mu_extract.py',
            'work/after149_amzn_extract.py', 'work/after141_amd_source_extract.py')


def _text(value):
    return old._text(value)


def _reference(raw, c, profile):
    if not isinstance(raw, bytes) or not isinstance(c, dict):
        raise SourceError('bytes_and_context_object_required')
    if (c.get('symbol') != profile['symbol'] or type(c.get('fiscal_year')) is not int
            or c['fiscal_year'] not in profile['years'] or c.get('context_status') != STATUS
            or c.get('format') != 'html_or_text' or c.get('source_kind') != 'HTML_original'
            or c.get('original_report_bytes_verified') is not True):
        raise SourceError('outside_fixed_original_issuer_year_context_scope')
    if c.get('event_id') != profile['symbol'] + '_' + str(c.get('publication_date')):
        raise SourceError('event_identity_mismatch')
    source = c.get('source')
    required = ('path', 'sha256', 'url', 'receipt_path', 'receipt_sha256')
    if not isinstance(source, dict) or any(not isinstance(source.get(k), str) or not source[k] for k in required):
        raise SourceError('missing_source_reference')
    if not SHA.fullmatch(source['sha256']) or sha256(raw) != source['sha256'] or not SHA.fullmatch(source['receipt_sha256']):
        raise SourceError('raw_source_sha256_or_receipt_digest_mismatch')
    for key in ('path', 'receipt_path'):
        path = source[key]
        if path.startswith('/') or any(p in ('', '.', '..') for p in path.split('/')) or 'CloudStorage' in path or '\\' in path:
            raise SourceError('invalid_local_source_reference')
    try:
        url = urlsplit(source['url'])
        valid = (url.scheme == 'https' and url.hostname in profile['hosts']
                 and not url.username and not url.password and url.port in (None, 443))
    except ValueError:
        valid = False
    if not valid:
        raise SourceError('invalid_official_source_url')
    if (c.get('evidence_path') != source['path'] or c.get('evidence_sha256') != source['sha256']
            or c.get('source_url') != source['url']):
        raise SourceError('conflicting_source_references')
    receipt = c.get('receipt')
    if not isinstance(receipt, dict) or receipt.get('path') != source['receipt_path'] or receipt.get('sha256') != source['receipt_sha256']:
        raise SourceError('conflicting_receipt_reference')
    if not isinstance(c.get('context_holds'), list) or any(not isinstance(x, str) for x in c['context_holds']):
        raise SourceError('invalid_context_holds')
    soup = BeautifulSoup(raw, 'html.parser')
    for node in soup(['script', 'style']):
        node.decompose()
    text = _text(soup.get_text(' ', strip=True))
    if not re.search(profile['issuer_pattern'], text, re.I) or not re.search(profile['ticker_pattern'], text, re.I):
        raise SourceError('raw_issuer_security_anchor_missing')
    anchors = c.get('identity_anchors')
    if not isinstance(anchors, list) or not anchors or any(not isinstance(a, str) or not a or _text(a).lower() not in text.lower() for a in anchors):
        raise SourceError('raw_context_identity_anchors_missing')
    return copy.deepcopy(source), soup


def _periods(c, profile):
    p = c.get('periods')
    if not isinstance(p, dict):
        raise FieldGap('missing_period_metadata')
    current, prior, start = (old._date(p.get(k)) for k in ('current_end', 'prior_end', 'proposed_current_start'))
    if (current.year != c['fiscal_year'] or prior.year != c['fiscal_year'] - 1
            or current.month != profile['month'] or prior.month != profile['month']
            or old._date(c.get('official_actual_period_end')) != current):
        raise FieldGap('context_year_end_mismatch')
    if start != prior + dt.timedelta(days=1) or (current-start).days+1 not in (364, 371):
        raise FieldGap('derived_start_mismatch')
    if (p.get('start_is_source_quoted') is not False or p.get('start_assumption_approved') is not False
            or p.get('start_assumption') != 'same_current_report_prior_annual_end_plus_one_calendar_day'
            or p.get('annual_date_binding') != 'same_report_full_date_headers'):
        raise FieldGap('unrecognized_or_upgraded_start_assumption')
    if not current < old._date(c.get('publication_date')) < old._date(c.get('available_session')):
        raise FieldGap('invalid_publication_availability_order')
    return dict(current=dict(period_start=str(start), period_end=str(current), days_inclusive=(current-start).days+1),
                comparison=dict(period_end=str(prior)),
                start_basis='derived_same_report_prior_annual_end_plus_one_not_directly_quoted',
                start_assumption_certified=False, assumption_id=profile['assumption_id'])


def _is_label(physical, cell):
    return all(not _text(c['text']) for c in old._cells(physical, cell['row']) if c['physical_cell'] < cell['physical_cell'])


def _header_date(text):
    """Only whole physical date labels qualify; title audit dates never do."""
    text = _text(text)
    match = DATE.fullmatch(text)
    slash = SLASH_DATE.fullmatch(text)
    if match:
        month, day, year = match.groups()
        parts = (int(year), MONTHS[month.lower()], int(day))
    elif slash:
        month, day, year = map(int, slash.groups())
        parts = (year + 2000 if year < 100 else year, month, day)
    else:
        return None
    try:
        return dt.date(*parts).isoformat()
    except ValueError as exc:
        raise FieldGap('invalid_header_date') from exc


def _column_geometry(headers, physical, row, columns, endpoint, annual, current, prior, profile):
    if (not isinstance(columns, list) or not columns or columns != sorted(set(columns))
            or any(type(c) is not int or c < 0 for c in columns)):
        raise FieldGap('invalid_header_columns')
    dated = [(c, _header_date(c['text'])) for c in headers]
    dated = [(c, day) for c, day in dated if day is not None]
    if any(c['rowspan'] != 1 for c, _ in dated):
        raise FieldGap('date_header_crosses_rows')
    for col in {x for c, _ in dated for x in c['columns']}:
        if sum(col in c['columns'] for c, _ in dated) != 1:
            raise FieldGap('conflicting_or_duplicate_date_headers')
    groups = sorted((dict(date=day, columns=list(c['columns'])) for c, day in dated), key=lambda g: min(g['columns']))
    if len(groups) != profile['income_groups' if annual else 'balance_groups']:
        raise FieldGap('unexpected_statement_date_group_count')
    if annual:
        parents = [c for c in headers if ANNUAL.fullmatch(_text(c['text']))]
        if len(parents) != 1 or parents[0]['rowspan'] != 1 or not set(columns) <= set(parents[0]['columns']):
            raise FieldGap('not_exclusively_annual_column')
        domain = set(parents[0]['columns'])
        if any(SHORT.search(c['text']) and set(c['columns']) & domain for c in headers):
            raise FieldGap('overlapping_annual_short_period_parent')
        if any(set(g['columns']) & domain and not set(g['columns']) <= domain for g in groups):
            raise FieldGap('date_header_crosses_annual_boundary')
        chosen_groups = [g for g in groups if set(g['columns']) <= domain]
        if len(chosen_groups) != 2 or {g['date'] for g in chosen_groups} != {current, prior}:
            raise FieldGap('annual_comparison_endpoint_mismatch')
    else:
        label = next(c for c in old._cells(physical, row) if _is_label(physical, c) and ASSETS.fullmatch(_text(c['text'])))
        domain = set(range(max(label['columns'])+1, max(max(c['columns']) for c in old._cells(physical, row))+1))
        chosen_groups = groups
        dates = [g['date'] for g in groups]
        if len(dates) != len(set(dates)) or not {current, prior} <= set(dates):
            raise FieldGap('balance_annual_endpoints_not_unique')
        if any(not prior < d < current for d in dates if d not in (current, prior)):
            raise FieldGap('balance_intermediate_period_invalid')
    if any(not set(g['columns']) <= domain for g in chosen_groups):
        raise FieldGap('date_header_outside_numeric_domain')
    selected = [i for i, g in enumerate(chosen_groups) if g['date'] == endpoint and set(columns) <= set(g['columns'])]
    if len(selected) != 1:
        raise FieldGap('header_anchor_not_single_period')
    i = selected[0]
    lower = min(domain) if i == 0 else (max(chosen_groups[i-1]['columns']) + min(chosen_groups[i]['columns']))//2+1
    upper = max(domain) if i == len(chosen_groups)-1 else (max(chosen_groups[i]['columns']) + min(chosen_groups[i+1]['columns']))//2
    owned = {c for c in domain if lower <= c <= upper}
    evidence = old._unique(c for c in headers if set(c['columns']) & set(columns))
    return evidence, owned, groups


def _field(tables, spec, kind, source, available, period, endpoint, prior, profile):
    if not isinstance(spec, dict):
        raise FieldGap('missing_field_context')
    if spec.get('source') != source:
        raise FieldGap('field_source_mismatch')
    if spec.get('available_from') != available or spec.get('as_of') != endpoint:
        raise FieldGap('field_period_or_availability_mismatch')
    if spec.get('currency_literal') != '$' or spec.get('currency_iso') is not None:
        raise FieldGap('unrecognized_currency_context')
    if spec.get('external_metadata', []) != []:
        raise FieldGap('external_statement_context_not_supported')
    ti, row, label_cell = (spec.get(k) for k in ('table', 'row', 'label_cell'))
    if any(type(x) is not int or x < 0 for x in (ti, row, label_cell)) or ti >= len(tables):
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
        cell['markup_sha256'] = cell['parsed_cell_markup_sha256']
    target = NI if kind == 'income' else ASSETS
    matches = [cell for cell in physical.values() if target.fullmatch(_text(cell['text'])) and _is_label(physical, cell)]
    if (len(matches) != 1 or matches[0]['row'] != row or matches[0]['physical_cell'] != label_cell
            or matches[0]['rowspan'] != 1 or not target.fullmatch(_text(str(spec.get('label', ''))))):
        raise FieldGap('exact_unique_consolidated_row_label_not_found')
    starts = [c['row'] for c in physical.values() if c['row'] < row and BODY.fullmatch(_text(c['text'])) and _is_label(physical, c)]
    if not starts:
        raise FieldGap('statement_body_boundary_not_found')
    stop = min(starts)
    headers = [c for c in physical.values() if c['row'] < stop]
    title = INCOME if kind == 'income' else BALANCE
    other_title = BALANCE if kind == 'income' else INCOME
    title_cells = [c for c in headers if title.search(_text(c['text']))]
    if len(title_cells) != 1 or any(other_title.search(c['text']) for c in headers):
        raise FieldGap('formal_consolidated_title_not_unique')
    if any(re.search(r'non[\s\-\u2010-\u2015]*gaap|reconciliation|as adjusted|pro forma', c['text'], re.I) for c in headers):
        raise FieldGap('adjusted_or_non_gaap_statement_requires_review')
    units = [c for c in headers if re.search(r'\bin (millions|thousands)\b', c['text'], re.I)]
    scales = {x.lower() for c in units for x in re.findall(r'\bin (millions|thousands)\b', c['text'], re.I)}
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
    if spec.get('row_geometry_without_values') != [{k: c[k] for k in GEOMETRY} for c in old._cells(physical, row)]:
        raise FieldGap('fixed_row_geometry_mismatch')
    columns = spec.get('logical_columns')
    col_evidence, owned, groups = _column_geometry(headers, physical, row, columns, endpoint, kind == 'income', period['period_end'], prior, profile)
    if any((row, col) not in grid for col in columns):
        raise FieldGap('incomplete_amount_geometry')
    value, display, selected = old._value(physical, grid, row, columns, owned)
    ids = {(x['row'], x['physical_cell']) for x in selected}
    if any(_text(x['text']) and set(x['columns']) & owned and (x['row'], x['physical_cell']) not in ids for x in old._cells(physical, row)):
        raise FieldGap('unconsumed_content_within_period_group')
    return dict(displayed_amount=display, value_decimal=value, scale=scale, currency_literal='$', currency_iso=None,
                currency_basis='same_statement_dollar_symbol_not_ISO_certification',
                period=copy.deepcopy(period) if kind == 'income' else None, as_of=endpoint,
                available_from=available, source=copy.deepcopy(source), fact_admitted=False,
                locator=dict(table=ti, row=row, label=copy.deepcopy(matches[0]), header_columns=list(columns),
                             logical_columns=list(columns), owned_logical_columns=sorted(owned), date_groups=groups,
                             physical_cells=copy.deepcopy(selected), header_evidence=copy.deepcopy(col_evidence),
                             scale_evidence=copy.deepcopy(units), statement_title_evidence=copy.deepcopy(title_cells)))


def _extract(raw_bytes, context, profile):
    c = copy.deepcopy(context)
    source, soup = _reference(raw_bytes, c, profile)
    result = dict(schema_version=1, stage='source_only_annual_NI_assets_candidates', symbol=profile['symbol'],
        event_id=c['event_id'], fiscal_year=c['fiscal_year'],
        identity=dict(issuer=profile['issuer'], symbol=profile['symbol'], source_anchors=copy.deepcopy(c['identity_anchors']), certified_daily_security_master=False),
        source=source, publication_date=c['publication_date'], available_from=c['available_session'],
        periods={}, fields=dict.fromkeys(FIELDS), gaps=[], context_holds=copy.deepcopy(c['context_holds']),
        accounting_review_flags=copy.deepcopy(c.get('accounting_terms', [])),
        accounting_context_numbers_redacted=copy.deepcopy(c.get('accounting_context_numbers_redacted', [])),
        fact_admitted=False, model_admitted=False, strategy_admitted=False, first_public_version_verified=False,
        complete_correction_history_verified=False, first_public_time_verified=False, certified_daily_security_master=False, financial_calculations=0,
        notes=['Current-release availability applies to all fields, including prior assets.',
               'The next pinned session must be verified by the caller; this parser only checks chronological ordering.',
               'Annual start is an unapproved derived boundary, not a directly quoted or certified date.',
               'Successful parsing does not remove inherited holds or certify GAAP accounting comparability.'])
    try:
        result['periods'] = _periods(c, profile)
    except (FieldGap, KeyError, TypeError) as exc:
        result['gaps'] = [dict(field=name, reason=str(exc)) for name in FIELDS]
        result['complete_required_fields'] = False
        return result
    current, prior = result['periods']['current'], result['periods']['comparison']['period_end']
    for name in FIELDS:
        try:
            result['fields'][name] = _field(soup.find_all('table'), c.get('fields', {}).get(name),
                'income' if name == 'annual_net_income' else 'assets', source, c['available_session'], current,
                prior if name == 'prior_total_assets' else current['period_end'], prior, profile)
        except (FieldGap, KeyError, TypeError) as exc:
            result['gaps'].append(dict(field=name, reason=str(exc)))
    result['complete_required_fields'] = not result['gaps']
    return result


def extract_document(raw_bytes, context):
    return _extract(raw_bytes, context, PROFILE)
