"""Pure KLAC annual operations NI and same-report asset candidate extraction.

Accepts one frozen after150_next_source_context.documents record plus its raw
HTML bytes. No I/O, financial ratios, admission or price/strategy imports.
The caller must freeze a source-only PLAN before passing real source bytes.
"""
from copy import deepcopy
from datetime import date, timedelta
from decimal import Decimal
import hashlib
import re
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from after141_amd_source_extract import table_grid, dates, norm

FIELDS = ('annual_net_income', 'current_total_assets', 'prior_total_assets')
TITLE_PREFIX = r'(?:condensed\s+)?consolidated\s+(?:(?:unaudited|condensed|summary)\s+)*'
INCOME = re.compile(TITLE_PREFIX+r'statements?\s+of\s+operations', re.I)
BALANCE = re.compile(TITLE_PREFIX+r'balance\s+sheets?', re.I)
NI = re.compile(r'net\s+(?:income|loss)(?:\s*\((?:loss|income)\))?', re.I)
ASSETS = re.compile(r'total\s+assets', re.I)
BODY = re.compile(r'(?:assets|current\s+assets|cash(?:\s+and\s+cash\s+equivalents)?|(?:net\s+|total\s+)?revenues?|(?:net\s+)?sales|net\s+income(?:\s*\(loss\))?)\s*:?', re.I)
ANNUAL = re.compile(r'\b(?:twelve|12)\s+months?\s+ended\b', re.I)
SHORT = re.compile(r'\b(?:(?:three|six|nine|3|6|9)\s+months?|quarters?)\s+ended\b', re.I)
BAD = re.compile(r'non[\s\-\u2010-\u2015]*gaap|pro\s+forma|as\s+adjusted|reconciliation', re.I)
CURRENCY = re.compile(r'[€£¥]|\b(?:EUR|GBP|JPY|CAD|HKD|AUD)\b', re.I)
ISSUER = re.compile(r'KLA(?:[\s-]+Tencor)?\s+Corporation', re.I)
TICKER = re.compile(r'NASDAQ\s*[:(]\s*KLAC\b', re.I)
PRE_HOLD = 'KLAC_FY2011_preformatted_column_geometry_unverified'
ASC_HOLD = 'KLAC_FY2019_ASC606_current_comparison_not_restated__average_assets_use_hold'


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def schema():
    return dict(schema_version=1, stage='unadmitted_klac_annual_source_candidates', fields=list(FIELDS),
                scope='Fixed KLAC FY2011..2026 source contexts; FY2011 remains an explicit PRE-layout gap.',
                annual_net_income='Exact consolidated Net income/loss in formal annual Statements of Operations; exclude attributable-to-parent, per-share, non-GAAP and quarterly cash-flow alternatives.',
                assets='Exact same-report current/prior June30 total-assets endpoints; no other-report substitution.',
                admission='Always false. FY2011 PRE geometry and FY2019 ASC606 purpose hold remain; source availability is not calendar certification.')


def source_files():
    return ['work/after150_klac_extract.py','work/test_after150_klac_extract.py','work/after141_amd_source_extract.py']


def _day(value):
    require(type(value) is str and re.fullmatch(r'\d{4}-\d{2}-\d{2}', value), 'canonical_date_required')
    return date.fromisoformat(value)


def _clean(text):
    return norm(text).strip().rstrip(':').strip()


def _amount(texts):
    displayed = ' '.join(t for t in texts if t.strip()).strip()
    normalized = re.sub(r'\s*,\s*', ',', norm(displayed)).replace('\u2212','-')
    clean = normalized
    require(clean.count('$') <= 1, 'multiple_currency_markers')
    if '$' in clean:
        require(re.match(r'^(?:\$|\(\s*\$)',clean), 'misplaced_currency_marker')
        clean = clean.replace('$','').strip()
    negative = clean.startswith('(') and clean.endswith(')')
    if negative:
        clean = clean[1:-1].strip()
    require('(' not in clean and ')' not in clean, 'unbalanced_monetary_parentheses')
    require(not (negative and clean.startswith('-')), 'conflicting_monetary_signs')
    require(re.fullmatch(r'-?\s*(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?',clean), 'ambiguous_or_missing_monetary_literal')
    literal = ('-' if negative else '')+clean.replace(',','').replace(' ','')
    # Sign is applied to the string; Decimal unary minus can round under a
    # caller's low precision context. There is no monetary arithmetic here.
    value = format(Decimal(literal),'f')
    return displayed, normalized, value


def _first_label(physical, row, cell):
    return not any(v['text'] for (r,p),v in physical.items() if r == row and p < cell)


def _info(table):
    require(table.find('table') is None, 'nested_statement_table')
    rows, grid, physical = table_grid(table)
    starts = [r for (r,p),v in physical.items() if _first_label(physical,r,p) and BODY.fullmatch(_clean(v['text']))]
    require(starts, 'statement_body_boundary_missing')
    stop = min(starts)
    headers = [v for (r,_),v in physical.items() if r < stop]
    return rows, grid, physical, headers, stop


def _find_table(soup, kind, locator):
    """Require a single formal source table, before inspecting its amounts."""
    pattern = INCOME if kind == 'income' else BALANCE
    matches = []
    for ti,table in enumerate(soup.find_all('table')):
        try:
            info = _info(table)
        except ValueError:
            continue
        headers = info[3]
        titles = [c for c in headers if pattern.search(c['text'])]
        if titles and not any(BAD.search(c['text']) for c in headers):
            matches.append((ti,table,info,titles))
    require(len(matches) == 1, 'formal_statement_missing_or_duplicate')
    ti,table,info,titles = matches[0]
    require(ti == locator['table'] and len(titles) == 1, 'formal_statement_context_mismatch')
    return table,info,titles[0]


def _metadata(physical, headers, stop, frozen):
    require(stop == frozen['header_stop_row'], 'frozen_header_boundary_changed')
    for old in frozen['headers']:
        key = old['row'],old['physical_cell']
        require(key in physical, 'frozen_header_cell_missing')
        actual = physical[key]
        require(actual['row'] < stop and all(actual[k] == old[k] for k in ('columns','rowspan')) and
                norm(actual['text']) == norm(old['text']), 'frozen_header_cell_changed')
    units = []
    for cell in headers:
        for match in re.finditer(r'\bin\s+(millions|thousands|billions)\b',cell['text'],re.I):
            units.append((match[1].lower(),cell))
    require(units and {u for u,_ in units} == {'thousands'}, 'missing_or_conflicting_KLAC_scale')
    return [c for _,c in units]


def _headers_for(headers, column):
    return [c for c in headers if column in c['columns']]


def _columns(headers, locator, side, ending, annual):
    selected = locator[side+'_header_columns']
    require(type(selected) is list and selected and all(type(c) is int and c >= 0 for c in selected) and
            selected == sorted(set(selected)) and selected == list(range(min(selected),max(selected)+1)), 'invalid_context_columns')
    evidence = {}
    for col in selected:
        cells = _headers_for(headers,col)
        text = ' '.join(c['text'] for c in cells)
        require(dates([text]) == [ending], 'exclusive_date_header_required')
        if annual:
            require(ANNUAL.search(text) and not SHORT.search(text), 'exclusive_annual_header_required')
        for c in cells:
            evidence[c['row'],c['physical_cell']] = c
    if annual:
        wanted = locator['annual_header']
        key = wanted['row'],wanted['physical_cell']
        require(key in evidence and ANNUAL.search(evidence[key]['text']) and
                evidence[key]['columns'] == wanted['columns'] and norm(evidence[key]['text']) == norm(wanted['text']), 'annual_parent_context_mismatch')
        require(set(selected) <= set(wanted['columns']), 'annual_group_boundary_crossed')
    return selected,list(evidence.values())


def _cell_ref(cell):
    return dict(row_index_zero_based=cell['row'],physical_cell_index_zero_based=cell['physical_cell'],
                logical_columns=cell['columns'],rowspan=cell['rowspan'],text=cell['text'],
                markup_sha256=cell['parsed_cell_markup_sha256'])


def _field(soup, context, field, source, available, periods):
    annual = field == 'annual_net_income'
    kind = 'income' if annual else 'assets'
    summary = context['context_summary']
    locator = summary['net_income_locator' if annual else 'assets_locator']
    table,info,title = _find_table(soup,kind,locator)
    rows,grid,physical,headers,stop = info
    frozen = [t for t in context['context']['statement_tables'] if t['table'] == locator['table']]
    require(len(frozen) == 1, 'frozen_statement_context_missing_or_duplicate')
    scales = _metadata(physical,headers,stop,frozen[0])
    label = NI if annual else ASSETS
    matches = [c for (r,p),c in physical.items() if _first_label(physical,r,p) and label.fullmatch(_clean(c['text']))]
    require(len(matches) == 1, 'exact_total_row_missing_or_duplicate')
    target = matches[0]
    require(target['rowspan'] == 1, 'row_label_spans_multiple_rows')
    require(target['row'] == locator['row'] and target['physical_cell'] == locator['label_cell'] and
            _clean(target['text']) == _clean(locator['label']), 'frozen_total_row_changed')
    require(not CURRENCY.search(table.get_text(' ',strip=True)), 'conflicting_currency_evidence')
    # Both date groups must still identify the exact same-report endpoints.
    current = periods['current']['period_end']
    prior = periods['comparison']['period_end']
    curr, curr_headers = _columns(headers,locator,'current',current,annual)
    prev, prev_headers = _columns(headers,locator,'comparison',prior,annual)
    require(not set(curr) & set(prev), 'overlapping_year_groups')
    selected, used_headers = (prev,prev_headers) if field == 'prior_total_assets' else (curr,curr_headers)
    ending = prior if field == 'prior_total_assets' else current
    row = target['row']
    inside = sorted([c for (r,p),c in physical.items() if r == row and set(c['columns']) & set(selected)],key=lambda c:c['physical_cell'])
    require(inside and all(c['rowspan'] == 1 and set(c['columns']) <= set(selected) for c in inside), 'amount_crosses_year_or_row')
    require(all((row,col) in grid and grid[row,col]['row'] == row for col in selected), 'amount_inherited_from_other_row')
    allcells = sorted([c for (r,p),c in physical.items() if r == row],key=lambda c:c['physical_cell'])
    indexes = [allcells.index(c) for c in inside]
    for i in (min(indexes)-1,max(indexes)+1):
        if 0 <= i < len(allcells) and allcells[i] not in inside:
            require(not re.fullmatch(r'[\s$()\-\u2212]+',allcells[i]['text']) or not allcells[i]['text'], 'sign_outside_declared_year_group')
    dollar = [c for (r,p),c in physical.items() if r >= stop and set(c['columns']) & set(selected) and '$' in c['text']]
    require(dollar, 'selected_year_dollar_evidence_missing')
    rawcells = []
    for c in inside:
        node = rows[row].find_all(['td','th'],recursive=False)[c['physical_cell']]
        rawcells.append(dict(_cell_ref(c),raw_text=node.get_text(' ',strip=False)))
    displayed,normalized,value = _amount([c['raw_text'] for c in rawcells])
    loc = dict(table_index_zero_based=locator['table'],row_index_zero_based=row,
               row_label=target['text'],label_cell_index_zero_based=target['physical_cell'],logical_columns=selected,
               statement_title=title['text'],header_cells=[_cell_ref(c) for c in used_headers],
               scale_cells=[_cell_ref(c) for c in scales],physical_cells=rawcells,
               dollar_evidence=[{k:v for k,v in _cell_ref(c).items() if k != 'text'} for c in dollar])
    return dict(measure='net_income' if annual else 'total_assets',
                basis='current_annual_consolidated_operations_total' if annual else 'same_report_year_end_balance',
                displayed_amount=displayed,normalized_monetary_literal=normalized,value_decimal=value,scale='1000',
                currency_literal='$',currency_iso=None,period=deepcopy(periods['current']) if annual else None,
                as_of=ending,available_from=available,source=deepcopy(source),locator=loc,fact_admitted=False)


def _validate(raw,context):
    require(type(raw) is bytes and raw and not raw.startswith(b'%PDF-'), 'nonempty_HTML_bytes_required')
    require(type(context) is dict and context.get('symbol') == 'KLAC', 'KLAC_context_required')
    year = context.get('fiscal_year')
    require(type(year) is int and 2011 <= year <= 2026 and context.get('fiscal_quarter') == 4, 'fixed_KLAC_annual_scope')
    released,available = _day(context['publication_date']),_day(context['available_session'])
    require(context['event_id'] == 'KLAC_'+str(released) and released < available, 'event_or_available_time_mismatch')
    require(context['evidence_sha256'] == digest(raw) and type(context['evidence_path']) is str and context['evidence_path'], 'source_SHA_or_path_mismatch')
    url = urlparse(context['source_url'])
    require(url.scheme == 'https' and url.netloc == 'ir.kla.com' and url.path.startswith('/news-events/press-releases/detail/'), 'original_KLA_source_required')
    for flag in ('source_hash_verified','catalog_claims_issuer_full_body','html_context_reviewed'):
        require(context.get(flag) is True, 'unverified_source_context:'+flag)
    require(context.get('format') == 'html_or_text' and context.get('source_kind') == 'issuer_original_release_full_bytes', 'full_HTML_context_required')
    detail,summary = context['context'],context['context_summary']
    require(detail.get('source_sha_verified') is True and detail.get('financial_values_extracted') == 0 and summary.get('fact_admitted') is False, 'preextraction_context_required')
    require(detail['receipt']['status'] == 200 and detail['receipt']['url_and_raw_sha_match'] is True, 'receipt_context_mismatch')
    end = _day(context['official_actual_period_end'])
    require(str(end) == str(year)+'-06-30' and end < released, 'reported_June30_annual_endpoint_required')
    if year != 2011:
        period = summary['current_annual_period']
        require(summary['status'] == 'formal_annual_NI_and_two_assets_endpoints_located_no_values_extracted' and
                period['start'] == str(year-1)+'-07-01' and period['end'] == str(end) and
                summary['comparison_assets_date'] == str(year-1)+'-06-30', 'annual_context_endpoints_mismatch')
        require(summary['units']['scale'] == '1000' and summary['units']['currency_literal'] == '$' and summary['units']['currency_iso'] is None, 'context_units_mismatch')
    else:
        require(summary['status'] == 'preformatted_original_requires_separate_column_geometry_review', 'PRE_context_hold_required')
    return year


def extract_document(raw_bytes, context_document):
    year = _validate(raw_bytes,context_document)
    e = context_document
    soup = BeautifulSoup(raw_bytes,'html.parser')
    for node in soup(['script','style']):
        node.decompose()
    text = soup.get_text(' ',strip=True)
    require(ISSUER.search(text) and TICKER.search(text), 'original_issuer_and_ticker_not_found')
    source = dict(path=e['evidence_path'],sha256=e['evidence_sha256'],url=e['source_url'])
    current_end = e['official_actual_period_end']
    if year == 2011:
        periods = dict(current=dict(period_start=None,period_end=current_end,days_inclusive=None),
                       comparison=dict(period_start=None,period_end=None,days_inclusive=None))
    else:
        p = e['context_summary']['current_annual_period']
        periods = dict(current=dict(period_start=p['start'],period_end=p['end'],days_inclusive=(_day(p['end'])-_day(p['start'])).days+1),
                       comparison=dict(period_start=None,period_end=e['context_summary']['comparison_assets_date'],days_inclusive=None))
    result = dict(schema_version=1,stage=schema()['stage'],symbol='KLAC',event_id=e['event_id'],fiscal_year=year,
                  publication_date=e['publication_date'],release_date=e['publication_date'],available_from=e['available_session'],
                  source=source,receipt=deepcopy(e['context']['receipt']),periods=periods,fields={k:None for k in FIELDS},gaps=[],
                  original_context=deepcopy(e),accounting_review_flags=deepcopy(e['context_summary'].get('accounting_attention',[])),
                  accounting_context=deepcopy(e['context'].get('accounting_contexts_numbers_redacted',[])),
                  context_holds=[],availability_calendar_certified=False,
                  current_period_basis=deepcopy(e['context_summary'].get('current_annual_period', {}).get('basis')),
                  first_public_version_verified=False,currency_iso_verified=False,
                  fact_admitted=False,model_admitted=False,strategy_admitted=False,
                  financial_ratios_computed=0,factors_computed=0,returns_computed=0)
    if year == 2011:
        result['context_holds'].append(PRE_HOLD)
        result['gaps'] = [dict(field=k,reason=PRE_HOLD) for k in FIELDS]
        return result
    if year == 2019:
        result['context_holds'].append(ASC_HOLD)
    for field in FIELDS:
        try:
            result['fields'][field] = _field(soup,e,field,source,e['available_session'],periods)
        except ValueError as exc:
            result['gaps'].append(dict(field=field,reason=str(exc)))
    return result
