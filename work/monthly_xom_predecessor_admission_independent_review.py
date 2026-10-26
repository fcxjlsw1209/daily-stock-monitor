"""Independent XOM predecessor evidence review; never imports producer code."""
from collections import Counter
from bisect import bisect_right
from copy import deepcopy
from datetime import date, datetime
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
import argparse
import calendar
import json
import re

from bs4 import BeautifulSoup, Comment
import monthly_sec_revenue_data_review_v3 as structural
import monthly_wmt_crosscheck_independent_review as common

ROOT = Path(__file__).resolve().parent.parent
STUDY = 'outputs/monthly_sec_xom_predecessor_admission_v1/'
POLICY = 'monthly_sec_xom_predecessor_admission_v1'
CIK = '0000034088'
ORIGINAL_CIK = '0002115436'
SOURCE_CLASS = 'issuer_original_periodic_filing'
EVIDENCE_CLASS = 'issuer_original_filing_and_legacy_facts_with_predecessor_resolution'
FLAGS = ('factor_ready', 'identity_verified', 'semantic_verified')
require = common.require
digest = common.digest
strict_json = common.strict_json


def source_files():
    return [Path(__file__), ROOT / 'work/monthly_xom_predecessor_admission_independent_test.py',
            Path(structural.__file__)] + common.source_files()


def clean(value):
    return ' '.join(value.split())


def text(node):
    return clean(node.get_text(' ', strip=True))


def html_soup(raw):
    require(type(raw) is bytes, 'original_bytes_required')
    # Original XHTML is intentionally read as an HTML table. Remove only its
    # serialization declaration in this parsing view; source bytes stay bound.
    declaration = re.match(br'\A\s*<\?xml\s+[^?]+\?>', raw)
    if declaration:
        require(re.search(br'<html(?:\s|>)', raw, re.I) is not None, 'HTML root required for XML serialization')
        raw = raw[declaration.end():]
    return BeautifulSoup(raw, 'html.parser')


def document(raw):
    soup = html_soup(raw)
    for node in list(soup.find_all(True)):
        if node.parent is None:
            continue
        if node.name.lower() in ('script', 'style', 'ix:header') or re.search(r'display\s*:\s*none', node.get('style', ''), re.I):
            node.extract()
    return soup


def identity(soup):
    cover = text(soup)[:14000]
    forms = re.findall(r'\bFORM\s+(10-[KQ])\b', cover, re.I)
    require(forms and len(set(v.upper() for v in forms)) == 1, 'form_identity_missing_or_conflicting')
    require(re.search(r'\bEXXON\s+MOBIL\s+CORP\s*ORATION\b', cover, re.I), 'issuer_name_missing')
    require(re.search(r'Commission\s+File\s+(?:Number|No\.?)\s*:?\s*0*1-0*2256\b', cover, re.I), 'commission_file_missing')
    require(re.search(r'\b13\s*-\s*5409005\b', cover), 'issuer_ein_missing')
    hits = re.findall(r'For the (?:fiscal year|quarterly period|quarter)\s+ended\s+([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})', cover, re.I)
    months = {calendar.month_name[i].lower(): i for i in range(1, 13)}
    require(hits, 'cover_report_date_missing_or_conflicting')
    values = []
    for month, day, year in hits:
        require(month.lower() in months, 'cover_month_invalid')
        values.append(date(int(year), months[month.lower()], int(day)).isoformat())
    require(len(set(values)) == 1, 'cover_report_date_missing_or_conflicting')
    return dict(issuer='EXXON MOBIL CORPORATION', commission_file='1-2256', ein='13-5409005',
                form=forms[0].upper(), report_date=values[0], document_identity_verified=True)


def anchor(raw):
    soup = document(raw)
    ident = identity(soup)
    require(ident['form'] == '10-K' and ident['report_date'] == '2015-12-31', 'anchor_period_mismatch')
    matches = re.findall(r'.{0,180}\bXOM\b.{0,180}', text(soup))
    matches = [v for v in matches if 'New York Stock Exchange' in v and re.search('common stock', v, re.I)]
    require(matches, 'anchor_ticker_exchange_missing')
    return dict(ident, symbol='XOM', exchange='NYSE', security_evidence=matches[0], continuous_security_identity_verified=False)


def grid(table):
    """Fill a coordinate matrix directly from each origin cell's full rectangle."""
    trs = [tr for tr in table.find_all('tr') if tr.find_parent('table') is table]
    matrix = {}
    for row_index, tr in enumerate(trs):
        cursor = 0
        for cell in tr.find_all(['td', 'th'], recursive=False):
            while (row_index, cursor) in matrix:
                cursor += 1
            try:
                width, height = int(cell.get('colspan', 1)), int(cell.get('rowspan', 1))
            except (TypeError, ValueError):
                raise ValueError('invalid_table_span')
            require(1 <= width <= 100 and 1 <= height <= 100, 'invalid_table_span')
            item = dict(text=text(cell), row=row_index, col=cursor, node=cell)
            for dy in range(height):
                for dx in range(width):
                    coord = (row_index + dy, cursor + dx)
                    require(coord not in matrix, 'overlapping_table_span')
                    matrix[coord] = item
            cursor += width
    return [{c: matrix[r, c] for rr, c in matrix if rr == r} for r in range(len(trs))]


def number(value):
    token = clean(value)
    unsigned = r'(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'
    require(re.fullmatch(r'-?' + unsigned + r'|\(' + unsigned + r'\)', token), 'invalid_printed_amount')
    if token.startswith('('):
        token = '-' + token[1:-1]
    result = Decimal(token.replace(',', ''))
    require(result.is_finite(), 'nonfinite_amount')
    return result


LABELS = {
    'sales_and_other_operating_revenue': r'Sales and other operating revenue(?: \(\d+\))?',
    'income_from_equity_affiliates': r'Income from equity affiliates(?: \(\d+\))?',
    'other_income': r'Other income(?: \(\d+\))?',
    'total_revenues_and_other_income': r'Total revenues and other income(?: \(\d+\))?',
}


def inline_info(cell):
    return [dict(name=n.get('name'), context_ref=n.get('contextref'), unit_ref=n.get('unitref'),
                 scale=n.get('scale'), decimals=n.get('decimals'), sign=n.get('sign'), printed=text(n))
            for n in cell['node'].find_all('ix:nonfraction')]


def preceding_text(table):
    strings = [str(v) for v in table.find_all_previous(string=True)
               if not isinstance(v, Comment) and str(v).strip()][:12]
    return clean(' '.join(reversed(strings)))


def statement(soup, ident):
    selected = []
    for table_index, table in enumerate(soup.find_all('table')):
        value = text(table)
        if not re.search(LABELS['sales_and_other_operating_revenue'], value, re.I) or 'Total revenues and other income' not in value:
            continue
        cells = grid(table)
        origins = [(r, item) for r, row in enumerate(cells) for c, item in sorted(row.items())
                   if item['row'] == r and item['col'] == c]
        matched = {key: [(r, item) for r, item in origins if re.fullmatch(pattern, item['text'], re.I)]
                   for key, pattern in LABELS.items()}
        if any(len(v) != 1 for v in matched.values()):
            continue
        labels = {key: value[0] for key, value in matched.items()}
        first = min(v[0] for v in labels.values())
        header = ' '.join(item['text'] for r, item in origins if r < first)
        title = ('CONDENSED ' if ident['form'] == '10-Q' else '') + 'CONSOLIDATED STATEMENT OF INCOME'
        if title not in header.upper() and not preceding_text(table).upper().endswith(title):
            continue
        require(re.search(r'\(millions of dollars(?:, unless noted)?\)', header, re.I), 'statement_million_dollar_units_missing')
        year_rows = []
        for r in range(first):
            year_cells = [(item['col'], item) for rr, item in origins if rr == r and re.fullmatch(r'20\d{2}', item['text'])]
            if len(year_cells) in (2, 3, 4):
                year_rows.append((r, year_cells))
        require(len(year_rows) == 1, 'statement_year_columns_not_unique')
        year_row, years = year_rows[0]
        y = date.fromisoformat(ident['report_date']).year
        require([int(v['text']) for _, v in years] in ([y, y-1], [y, y-1, y-2], [y, y-1, y, y-1]), 'statement_year_direction_mismatch')
        columns = []
        for col, cell in years:
            items = {}
            for r in range(year_row):
                item = cells[r].get(col)
                if item is not None:
                    items.setdefault((item['row'], item['col']), item['text'])
            heading = clean(' '.join(items.values()))
            year = int(cell['text'])
            if ident['form'] == '10-K':
                require(len(years) == 3 and ident['report_date'].endswith('-12-31'), 'annual_period_invalid')
                duration, end = 12, date(year, 12, 31)
                basis = 'annual_form_fiscal_year_ended_cover_and_printed_year_column'
            else:
                durations = re.findall(r'\b(Three|Six|Nine) Months Ended\b', heading, re.I)
                ends = re.findall(r'\b(March 31|June 30|September 30),?\b', heading, re.I)
                require(len(durations) == len(ends) == 1, 'quarter_column_period_missing')
                duration = {'three': 3, 'six': 6, 'nine': 9}[durations[0].lower()]
                month = {'march 31': 3, 'june 30': 6, 'september 30': 9}[ends[0].lower()]
                end = date(year, month, calendar.monthrange(year, month)[1])
                basis = 'printed_month_duration_end_and_year_column'
            require(end.month-duration+1 >= 1, 'period_start_invalid')
            start = date(year, end.month-duration+1, 1)
            columns.append(dict(grid_column=col, year=year, months=duration, start=str(start), end=str(end),
                                duration_days=(end-start).days+1, period_basis=basis, period_start_derived=True))
        rows = {}
        for key, (r, label) in labels.items():
            values, inline = [], []
            for col, _ in years:
                item = cells[r].get(col)
                require(item is not None and item['text'] and item['col'] > label['col'], 'amount_column_missing')
                values.append(str(number(item['text'])))
                inline.append(inline_info(item))
            rows[key] = dict(label=label['text'], row_index=r, values_millions=values, inline=inline)
        disclosure = re.findall(r'Due to rounding, numbers presented may not add up precisely to the totals indicated\.', text(soup), re.I)
        bound = 2 if disclosure else 0
        residuals = []
        for i in range(len(columns)):
            delta = sum(Decimal(rows[k]['values_millions'][i]) for k in LABELS if k != 'total_revenues_and_other_income') - Decimal(rows['total_revenues_and_other_income']['values_millions'][i])
            require(abs(delta) <= bound, 'printed_components_inconsistent')
            residuals.append(str(delta))
        selected.append(dict(table_index=table_index, title=title, unit='USD_millions', currency='USD', columns=columns,
                             rows=rows, rounding_disclosure=disclosure[0] if disclosure else None,
                             component_rounding_max_millions=bound, component_sum_deltas_millions=residuals))
    require(len(selected) == 1, 'consolidated_statement_not_unique')
    return selected[0]


def parse_html(raw):
    soup = document(raw)
    ident = identity(soup)
    parsed = statement(soup, ident)
    refs = {item['context_ref'] for row in parsed['rows'].values() for entries in row['inline'] for item in entries}
    original = html_soup(raw)
    contexts = {}
    for node in original.find_all(['xbrli:context', 'context']):
        ref = node.get('id')
        if ref not in refs:
            continue
        require(ref not in contexts, 'duplicate_inline_context_id')
        children = list(node.find_all(True))
        get = lambda names: [text(n) for n in children if n.name.lower() in names]
        contexts[ref] = dict(identifiers=get(('xbrli:identifier', 'identifier')),
                             start_dates=get(('xbrli:startdate', 'startdate')), end_dates=get(('xbrli:enddate', 'enddate')),
                             instant_dates=get(('xbrli:instant', 'instant')),
                             dimensions=[dict(dimension=n.get('dimension'), member=text(n)) for n in children if n.name.lower() in ('xbrldi:explicitmember', 'xbrldi:typedmember')],
                             scenario_present=any(n.name.lower() in ('xbrli:scenario', 'scenario') for n in children))
    require(set(contexts) == refs, 'inline_context_missing')
    return dict(identity=ident, statement=parsed, inline_contexts=contexts)


def blob(design, documents, path):
    value = documents[path]
    require(type(value) is bytes and sha256(value).hexdigest() == design['source_hashes'][path], 'bound source bytes: '+path)
    return value


def read(design, documents, path):
    return strict_json(blob(design, documents, path).decode())


def flags(record):
    require(all(record.get(k) is False for k in FLAGS), 'original flags')
    return {k: False for k in FLAGS}


def legacy(design, documents, path, url):
    raw = blob(design, documents, path)
    receipt = str(Path(path).with_suffix('.provenance.json'))
    meta = read(design, documents, receipt)
    urls = [meta[k] for k in ('source', 'url') if k in meta]
    require(urls and all(x == url for x in urls) and meta['sha256'] == sha256(raw).hexdigest(), 'legacy source binding')
    require(not any(k in meta for k in ('http_status', 'status', 'status_code')), 'legacy HTTP grade')
    _, day = structural.retrieval_clock(meta['retrieved_at'])
    return dict(path=path, sha256=sha256(raw).hexdigest(), provenance_path=receipt,
                provenance_sha256=sha256(documents[receipt]).hexdigest(), source_url=url,
                retrieved_at=meta['retrieved_at'], snapshot_date=day, provenance_verified=True,
                http_status_recorded=False, modern_http200_provenance_verified=False)


def source_state(design, documents):
    r = legacy(design, documents, design['filings_root_path'], 'https://data.sec.gov/submissions/CIK'+CIK+'.json')
    a = legacy(design, documents, design['filings_archive_path'], 'https://data.sec.gov/submissions/CIK'+CIK+'-submissions-001.json')
    f = legacy(design, documents, design['facts_path'], 'https://data.sec.gov/api/xbrl/companyfacts/CIK'+CIK+'.json')
    root, page, facts = [read(design, documents, e['path']) for e in (r, a, f)]
    require(str(root['cik']).zfill(10) == str(facts['cik']).zfill(10) == CIK, 'legacy issuer identity')
    name = Path(a['path']).name
    advertisements = [v for v in root['filings']['files'] if v['name'] == name]
    require(len(advertisements) == 1, 'one advertised archive')
    ad = advertisements[0]; dates = page['filingDate']
    require(dates and all(isinstance(v, list) and len(v) == len(dates) for v in page.values()), 'archive columns')
    require(ad['filingCount'] == len(dates) and min(dates) == ad['filingFrom'] and max(dates) <= ad['filingTo'], 'archive inclusive boundaries')
    oracle = structural.oracle_catalog(root, {name: page}, r['snapshot_date'])
    require(all(v['reason'] == 'missing_archive' for v in oracle['holes']), 'malformed predecessor catalog')
    recent_dates = root['filings']['recent']['filingDate']
    sources = [dict(name='recent', status='present', row_count=len(recent_dates), range_start=min(recent_dates),
                    range_end=r['snapshot_date'], actual_min=min(recent_dates), actual_max=max(recent_dates)),
               dict(name=name, status='present', range_start=ad['filingFrom'], range_end=ad['filingTo'],
                    row_count=len(dates), expected_count=ad['filingCount'], actual_min=min(dates), actual_max=max(dates))]
    for hole in oracle['holes']:
        matching = [v for v in root['filings']['files'] if v['name'] == hole['source']]
        require(len(matching) == 1, 'missing archive manifest ambiguity')
        manifest = matching[0]
        sources.append(dict(name=hole['source'], status='missing', range_start=hole['start'], range_end=hole['end'], row_count=None, expected_count=manifest['filingCount']))
    sources = sources[:1] + sorted(sources[1:], key=lambda v: v['name'])
    catalog = dict(schema_version=1, cik=CIK, snapshot_date=oracle['snapshot_date'],
                   filings=[dict(accession=v['accessionNumber'], form=v['form'], filing_date=v['filingDate'],
                                 report_date=v['reportDate'], metadata_issues=v['metadata_issues'], sources=v['sources']) for v in oracle['filings']],
                   sources=sources, coverage_ranges=[dict(start=x, end=y) for x, y in oracle['coverage_ranges']],
                   holes=deepcopy(oracle['holes']), ignored_archive_names=oracle['ignored_archive_names'], raw_row_count=oracle['raw_row_count'],
                   unique_accessions=oracle['unique_accessions'], report_count=oracle['report_count'])
    status = dict(semantic_source_sufficient=True, raw_facts_provenance_verified=False,
                  filing_catalog_http_status_verified=False, legacy_official_url_hash_time_bound=True,
                  atomic_catalog_vintage_verified=False, root=r, archive=a, facts=f,
                  association=dict(compatible=True, issues=[], evidence='verified_official_page_count_and_inclusive_advertised_range_not_atomic_vintage'))
    return oracle, catalog, facts, status, {'recent': root['filings']['recent'], name: page}


def transfer(item, design, documents):
    meta = read(design, documents, item['receipt_path'])
    raw = blob(design, documents, item['response_path']); decoded = blob(design, documents, item['document_path'])
    require(meta.get('status') == 200 and meta.get('request_count') == 1 and meta.get('public_unauthenticated') is True, 'HTTP receipt')
    require(meta['url'] == meta['final_url'] == item['url'] and item['url'].startswith('https://investor.exxonmobil.com/sec-filings/all-sec-filings/content/'), 'issuer exact URL')
    require(meta['response_sha256'] == sha256(raw).hexdigest() and meta['decoded_sha256'] == sha256(decoded).hexdigest(), 'transfer hashes')
    require(meta['response_bytes'] == len(raw) and meta['decoded_bytes'] == len(decoded), 'transfer byte lengths')
    require(meta.get('content_encoding') in (None, '', 'identity') and meta.get('decoder') == 'identity' and raw == decoded and meta.get('accept_encoding') == 'identity', 'identity transfer')
    structural.retrieval_clock(meta['retrieved_at'])
    require(item['url'].endswith('/'+item['accession']+'/'+item['primary_document']), 'selected document URL')
    folder = str(Path(item['receipt_path']).parent.parent)
    require(folder+'/'+meta['response_path'] == item['response_path'] and folder+'/'+meta['decoded_path'] == item['document_path'], 'receipt relative paths')
    return dict(document_path=item['document_path'], document_sha256=sha256(decoded).hexdigest(), receipt_path=item['receipt_path'],
                receipt_sha256=sha256(documents[item['receipt_path']]).hexdigest(), url=item['url'], retrieved_at=meta['retrieved_at'],
                original_document_http200_verified=True, original_ir_archive_outlink_verified=False,
                historical_http_availability_verified=False, financial_evidence_available_date=item['filing_date'])


def catalog_anchor(item, tables):
    found = []
    keys = ('accessionNumber', 'filingDate', 'reportDate', 'form', 'primaryDocument')
    expected = dict(zip(keys, [item['accession'], item['filing_date'], item['report_date'], item['form'], item['primary_document']]))
    for table in tables.values():
        for i, acc in enumerate(table['accessionNumber']):
            if acc == item['accession']:
                actual = {k: table[k][i] for k in keys}
                require(actual == expected, 'catalog selected primary document')
                found.append(actual)
    require(found, 'document absent from catalog')


def reconstruct(design, documents, oracle, facts, sessions):
    records = read(design, documents, design['source_records']['path'])
    diagnostic = read(design, documents, design['diagnostic_path'])
    by_index = {v['record_index_0based']: v for v in diagnostic['rows']}
    require(len(records) == design['expected_record_count'] and len(by_index) == len(diagnostic['rows']) == len(design['resolved_indices'])+len(design['post_effective_excluded_indices']), 'original/diagnostic scope')
    require(sessions and sessions == sorted(set(sessions)), 'session axis')
    for day in sessions: structural.iso_day(day)
    require(len({(v['symbol'], v['signal_date']) for v in records}) == len(records), 'original record keys')
    resolved = []
    for i, old in enumerate(records):
        flags(old); flags(old['pairing'])
        if old['symbol'] != 'XOM': continue
        require(old['candidate_cik'] == ORIGINAL_CIK and old['catalog']['status'] == 'coverage_unknown', 'original XOM identity')
        d = by_index[i]
        require(d['original_record_sha256'] == digest(old) and d['original_cik'] == ORIGINAL_CIK and d['signal_date'] == old['signal_date'], 'diagnostic original record binding')
        if old['signal_date'] >= '2026-07-01':
            require(d['catalog'] is None and d['pairing'] is None, 'post-cutover excluded')
            continue
        cat = structural.oracle_asof(oracle, old['signal_date'], sessions)
        require(cat == d['catalog'] and cat['status'] == 'selected' and cat['coverage_complete'], 'independent latest catalog')
        expected = structural.oracle_pairs(facts, cat['selected'], old['signal_date'])
        pair = d['pairing']
        require(pair['pairs'] == expected['pairs'] and pair['structural_pair_available'] is expected['structural_pair_available'], 'independent full ordered pairs')
        require({k: v['reason'] for k, v in pair['diagnostics']['concepts'].items()} == expected['concept_reasons'], 'independent concept rejection reasons')
        require(pair['factor_ready'] is False and pair['status'] == ('paired' if expected['pairs'] else 'unavailable') and pair['reason'] == ('structural_pair_available' if expected['pairs'] else 'no_structural_pair'), 'structural-only pairing status')
        resolved.append(dict(record_index_0based=i, source_record_sha256=digest(old), original_record=deepcopy(old),
                             original_candidate_cik=ORIGINAL_CIK, resolved_candidate_cik=CIK, resolved_catalog=cat,
                             resolved_pairing=deepcopy(pair), resolution_basis='pre_effective_issuer_anchor_plus_original_filing',
                             continuous_security_identity_verified=False, original_source_flags=flags(old), factor_ready=False))
    require([v['record_index_0based'] for v in resolved] == design['resolved_indices'], 'resolved exact scope')
    require([i for i, r in enumerate(records) if r['symbol'] == 'XOM' and r['signal_date'] >= '2026-07-01'] == design['post_effective_excluded_indices'], 'post-cutover exact scope')
    return records, resolved


def validate_pair(pair, selected, facts):
    require(pair['taxonomy'] == 'us-gaap' and pair['unit'] == 'USD' and pair['tag'] in ('Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax'), 'raw supported concept')
    common.raw_contexts(pair, selected, facts)
    for role in ('current', 'prior'):
        v = pair[role]; start = structural.iso_day(v['start']); end = structural.iso_day(v['end'])
        require(start <= end and v['duration_days'] == (end-start).days+1 and v['accn'] == selected['accession'] and v['filed'] == selected['filingDate'] and v['form'] == selected['form'], 'raw context period and accession')
        require(type(v['val']) in (int, float) and Decimal(str(v['val'])).is_finite() and v['val'] >= 0, 'raw finite amount')


def compare(pair, parsed, mapping):
    fail = lambda reason: dict(status='unverified', reason=reason, selected_period=None)
    if parsed is None: return fail('original_filing_not_parsed')
    st = parsed['statement']; semantic = mapping['semantic_concept']
    require(semantic in st['rows'], 'registered statement semantic')
    row = st['rows'][semantic]; periods = {}; inline = []
    for role in ('current', 'prior'):
        original = pair[role]
        matches = [(i, c) for i, c in enumerate(st['columns']) if (c['start'], c['end']) == (original['start'], original['end'])]
        if len(matches) != 1: return fail('printed_period_not_unique')
        i, col = matches[0]
        if col['duration_days'] != original['duration_days'] or Decimal(row['values_millions'][i])*1000000 != Decimal(str(original['val'])):
            return fail('printed_value_or_period_mismatch')
        periods[role] = dict(deepcopy(col), printed_millions=row['values_millions'][i], usd=original['val'])
        corroborated = []
        for fact in row['inline'][i]:
            context = parsed['inline_contexts'][fact['context_ref']]
            if fact['name'] == 'us-gaap:'+pair['tag'] and context['identifiers'] == [CIK] and context['start_dates'] == [original['start']] and context['end_dates'] == [original['end']] and not context['dimensions'] and not context['scenario_present']:
                corroborated.append(dict(fact=deepcopy(fact), context=deepcopy(context)))
        inline.append(corroborated)
    return dict(status='exact_match', reason='all_printed_values_and_periods_exact', selected_period=dict(
        semantic_concept=semantic, printed_label=row['label'], table_index=st['table_index'], row_index=row['row_index'],
        statement_title=st['title'], unit=st['unit'], currency=st['currency'], periods=periods, mapping_basis=mapping['basis'],
        inline_nondimensional_corroboration=inline, original_inline_duration_contexts_verified=False,
        component_sum_deltas_millions=st['component_sum_deltas_millions'], component_rounding_max_millions=st['component_rounding_max_millions'], rounding_disclosure=st['rounding_disclosure']))


def admit(resolution, entries, status, ident):
    require(status.get('semantic_source_sufficient') is True and status.get('raw_facts_provenance_verified') is False and status.get('filing_catalog_http_status_verified') is False, 'source grade not admissible')
    old = resolution['original_record']; selected = resolution['resolved_catalog']['selected']; pairs = resolution['resolved_pairing']['pairs']
    expected = [digest(dict(accession=selected['accession'], pair=p)) for p in pairs]
    require(len(pairs) in (1, 2) and expected == [v['evidence_id'] for v in entries], 'ordered complete pair evidence')
    for p, e in zip(pairs, entries): require(e['pair'] == p and e['pair_sha256'] == digest(p), 'original evidence pair unchanged')
    reasons = list(dict.fromkeys(e['reason'] for e in entries if e['status'] != 'exact_match'))
    coverage = dict(record_index_0based=resolution['record_index_0based'], source_record_sha256=resolution['source_record_sha256'],
                    selected_accession=selected['accession'], evidence_ids=expected, status='unverified' if reasons else 'admitted', reasons=reasons)
    if reasons: return None, coverage
    approved = []
    for pair, evidence in zip(pairs, entries):
        approved.append(dict(evidence_id=evidence['evidence_id'], pair_sha256=evidence['pair_sha256'], taxonomy=pair['taxonomy'], tag=pair['tag'], unit=pair['unit'],
            semantic_concept=evidence['semantic_concept'], source_class=SOURCE_CLASS, evidence_class=EVIDENCE_CLASS,
            current=deepcopy(pair['current']), prior=deepcopy(pair['prior']), selected_filing=deepcopy(selected),
            selected_period=deepcopy(evidence['selected_period']), release_evidence=deepcopy(evidence['release_evidence']), source_status=deepcopy(status), identity_evidence=deepcopy(ident),
            checks=dict(raw_contexts_bound=True, all_ordered_pairs_present=True, values_match=True, periods_match=True, first_usable_session_verified=True, original_document_identity_verified=True, resolved_catalog_rebuilt=True),
            original_accession_body_directly_verified=True, original_xbrl_instance_directly_verified=False, original_inline_duration_contexts_verified=False,
            period_starts_directly_printed=False, continuous_security_identity_verified=False, factor_ready=False))
    return dict(record_index_0based=resolution['record_index_0based'], source_record_sha256=resolution['source_record_sha256'], symbol='XOM',
        candidate_cik=ORIGINAL_CIK, original_candidate_cik=ORIGINAL_CIK, resolved_candidate_cik=CIK, signal_date=old['signal_date'], selected_accession=selected['accession'],
        selected_from_resolved_catalog=True, original_selected_accession=None, resolved_catalog=deepcopy(resolution['resolved_catalog']), resolved_pairing=deepcopy(resolution['resolved_pairing']),
        original_record=deepcopy(old), source_flags=flags(old), semantic_admitted=True, factor_ready=False, source_class=SOURCE_CLASS, evidence_class=EVIDENCE_CLASS,
        raw_facts_provenance_verified=False, evidence_ids=expected, pair_sha256s=[digest(p) for p in pairs], approval=approved), coverage


def produce(design, documents):
    require(design['policy_id'] == POLICY and design['source_class'] == SOURCE_CLASS and design['evidence_class'] == EVIDENCE_CLASS, 'registered policy')
    for certificate in design['source_certificates']:
        obj = read(design, documents, certificate['path'])
        require(obj['status'] == certificate['status'], 'source certificate status')
        for field in certificate['pin_fields']:
            require(all(design['source_hashes'].get(p) == h for p, h in obj[field].items()), 'source certificate closure')
    require(design['source_records']['sha256'] == design['source_hashes'][design['source_records']['path']], 'original source record binding')
    oracle, catalog, facts, status, tables = source_state(design, documents)
    sessions = read(design, documents, design['sessions_path'])
    records, resolved = reconstruct(design, documents, oracle, facts, sessions)
    identity_source = design['identity_anchor']
    catalog_anchor(identity_source, tables)
    receipt = transfer(identity_source, design, documents)
    ident = anchor(blob(design, documents, identity_source['document_path']))
    require(ident['report_date'] == identity_source['report_date'] and identity_source['filing_date'] == '2016-02-24', 'causal identity anchor')
    ident = dict(ident, **receipt, filing_date=identity_source['filing_date'], retrospective_bridge_used_as_historical_evidence=False)
    sources = {v['accession']: v for v in design['scope']}
    require(len(sources) == len(design['scope']) == design['expected_accessions'], 'exact document scope')
    releases, evidence, rows, admitted, used = {}, {}, [], [], set()
    for acc, item in sources.items():
        catalog_anchor(item, tables)
        receipt = transfer(item, design, documents)
        parsed, error = None, None
        try:
            parsed = parse_html(blob(design, documents, item['document_path']))
        except ValueError as exc:
            error = str(exc)
        if parsed is not None:
            require(parsed['identity']['form'] == item['form'] and parsed['identity']['report_date'] == item['report_date'], 'parsed filing identity')
        releases[acc] = dict(receipt, accession=acc, form=item['form'], report_date=item['report_date'], filing_date=item['filing_date'], parsed=parsed, parse_reason=error)
    for resolution in resolved:
        index = resolution['record_index_0based']; original = resolution['original_record']; selected = resolution['resolved_catalog']['selected']; acc = selected['accession']
        require(acc in sources and index in sources[acc]['record_indices'], 'exact selected source membership')
        source = sources[acc]; used.add(acc)
        require(source['record_sha256s'][str(index)] == digest(original), 'original object hash')
        next_pos = bisect_right(sessions, selected['filingDate'])
        require(next_pos < len(sessions) and selected['first_usable_date'] == sessions[next_pos] <= original['signal_date'] and original['signal_date'] in sessions, 'first next official session')
        require(identity_source['filing_date'] < selected['filingDate'] < selected['first_usable_date'], 'identity available before filing')
        pairs = resolution['resolved_pairing']['pairs']
        if not pairs:
            rows.append(dict(record_index_0based=index, source_record_sha256=digest(original), selected_accession=acc, evidence_ids=[], status='unavailable', reasons=['no_structural_pair']))
            continue
        require(len(pairs) in (1, 2), 'complete one/two pairs')
        entries = []
        for pair in pairs:
            validate_pair(pair, selected, facts)
            mapping = source['semantic_map'][pair['tag']]
            eid = digest(dict(accession=acc, pair=pair))
            require(source['evidence_pair_hashes'].get(eid) == digest(pair), 'registered original pair')
            e = dict(evidence_id=eid, accession=acc, pair_sha256=digest(pair), pair=deepcopy(pair), semantic_concept=mapping['semantic_concept'], semantic_mapping=deepcopy(mapping), **compare(pair, releases[acc]['parsed'], mapping), release_evidence=deepcopy(releases[acc]))
            require(eid not in evidence or evidence[eid] == e, 'duplicate evidence consistency')
            evidence[eid] = e; entries.append(e)
        sidecar, row = admit(resolution, entries, status, ident)
        rows.append(row)
        if sidecar is not None: admitted.append(sidecar)
    require(used == set(sources), 'no unused registered source')
    for acc, source in sources.items():
        observed = {eid: e['pair_sha256'] for eid, e in evidence.items() if e['accession'] == acc}
        require(observed == source['evidence_pair_hashes'], 'all registered pairs checked')
    require(len(resolved) == design['expected_rows'] and len(evidence) == design['expected_pairs'] and sum(len(r['evidence_ids']) for r in rows) == design['expected_pair_occurrences'], 'all candidate counts')
    require([r['record_index_0based'] for r in rows if not r['evidence_ids']] == design['no_pair_indices'], 'all no-pair rows preserved')
    shared = dict(schema_version=1, policy_id=POLICY, source_class=SOURCE_CLASS, evidence_class=EVIDENCE_CLASS, raw_facts_provenance_verified=False,
                  source_status=status, source_hashes=design['source_hashes'], source_records=design['source_records'], record_count=len(records), factor_ready_count=0, independent_review_pending=True)
    ad = dict(shared, semantic_admitted_count=len(admitted), records=admitted)
    ev = dict(shared, identity_anchor=ident, resolved_catalog=catalog, resolved_records=resolved, releases=releases, evidence=evidence)
    coverage = dict(schema_version=1, policy_id=POLICY, source_class=SOURCE_CLASS, evidence_class=EVIDENCE_CLASS, raw_facts_provenance_verified=False,
        record_count=len(records), scope_rows=len(rows), scope_accessions=len(releases), structural_candidate_rows=sum(bool(r['evidence_ids']) for r in rows),
        no_pair_count=len(design['no_pair_indices']), unique_pairs=len(evidence), pair_occurrences=sum(len(r['evidence_ids']) for r in rows), semantic_admitted_count=len(admitted),
        unverified_count=sum(r['status'] == 'unverified' for r in rows), unavailable_count=sum(r['status'] == 'unavailable' for r in rows), factor_ready_count=0,
        admitted_indices=[r['record_index_0based'] for r in admitted], rows=rows, post_effective_excluded_indices=design['post_effective_excluded_indices'],
        pair_status_counts=dict(Counter(e['status'] for e in evidence.values())), source_flags_unchanged=True, original_records_unchanged=True, concept_contexts_kept_separate=True,
        growth_factor_rank_order_return_computed=False, strategy_trials=0, network_requests=0, independent_review_pending=True)
    return {'ADMISSIONS.json': ad, 'EVIDENCE.json': ev, 'COVERAGE.json': coverage}



def registered_design(design):
    require(design['policy_id'] == POLICY and design['stage'] == 'frozen_scope_before_actual_build', 'registered stage')
    require(design['source_class'] == SOURCE_CLASS and design['evidence_class'] == EVIDENCE_CLASS, 'registered evidence class')
    require((design['expected_record_count'], design['expected_rows'], design['expected_accessions'], design['expected_structural_rows'], design['expected_pairs'], design['expected_pair_occurrences']) == (3904, 111, 38, 94, 46, 130), 'registered fixed candidate scope')
    require(len(design['no_pair_indices']) == 17 and len(design['post_effective_excluded_indices']) == 2, 'excluded original scope')
    require(design['source_records']['sha256'] == '758e2e6edd44099131ca88c9ca83d59f9f8489fb92bacdd087a1f9fde62c4ab3', 'original 3904 record bytes')
    contract = {
        'original_records_and_flags_unchanged': True, 'original_candidate_cik_retained': True,
        'explicit_resolved_entity_sidecar': True, 'predecessor_cik': CIK, 'original_candidate_cik': ORIGINAL_CIK,
        'effective_cutoff': '2026-07-01', 'latest_filing_selected_before_pairing': True,
        'no_older_filing_fallback': True, 'coverage_days': 400, 'first_strictly_next_official_session': True,
        'causal_2015_anchor_and_each_original_document_identity': True, 'continuous_security_identity_verified': False,
        'july_2026_bridge_retrospective_only': True, 'raw_facts_provenance_verified': False,
        'filing_catalog_http_status_verified': False, 'no_atomic_catalog_vintage_claim': True,
        'filing_date_is_financial_availability': True, 'record_level_all_pairs_required': True,
        'separate_sales_and_total_other_income': True, 'amounts_exact_no_tolerance': True,
        'component_qa_bound': '0 unless original rounding disclosure; then abs(sum(three)-total)<=2 printed million units from four half-unit rounding bounds',
        'original_inline_context_verification_claimed': False,
        'period_start_derivation': 'nominal printed months and independently printed end/year; calendar start, exact day count',
        'table_year_columns_exclude_note_reference_column': True, 'individual_amounts_never_derived': True,
        'no_growth_factor_rank_order_return': True, 'semantic_admission_count_not_promised': True}
    require(design['contract'] == contract and design['factor_ready'] is False and design['strategy_trials'] == design['network_requests'] == 0, 'registered immutable contract')
    early_sales = {'0000034088-17-000017', '0000034088-18-000015', '0000034088-18-000024',
        '0000034088-18-000039', '0000034088-18-000048', '0000034088-19-000010',
        '0000034088-19-000017', '0000034088-19-000042'}
    observed_early = set()
    for filing in design['scope']:
        for tag, rule in filing['semantic_map'].items():
            require(tag in ('Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax'), 'registered concept tag')
            sales = tag != 'Revenues' or filing['accession'] in early_sales
            expected = dict(semantic_concept='sales_and_other_operating_revenue' if sales else 'total_revenues_and_other_income',
                basis='registered_accession_specific_printed_consolidated_row_reconstruction',
                numeric_matching_used_to_select_row=False, original_inline_context_verification_claimed=False)
            require(rule == expected, 'registered accession-specific printed row mapping')
            if tag == 'Revenues' and sales: observed_early.add(filing['accession'])
    require(observed_early == early_sales, 'all eight early sales mappings retained')


def audit(root, design_sha, plan_sha):
    root = Path(root)
    load = lambda p: strict_json((root/p).read_text())
    h = lambda p: sha256((root/p).read_bytes()).hexdigest()
    require(h(STUDY+'DESIGN.json') == design_sha and h(STUDY+'PLAN.json') == plan_sha, 'registration digests')
    design, plan = load(STUDY+'DESIGN.json'), load(STUDY+'PLAN.json')
    registered_design(design)
    require(plan['policy_id'] == POLICY and plan['factor_ready'] is False and plan['strategy_trials'] == 0 and plan['design_sha256'] == design_sha and plan['actual_executed'] is False and plan['source_hashes'] == design['source_hashes'], 'plan source binding')
    require(type(plan['synthetic_tests_passed']) is int and plan['synthetic_tests_passed'] > 0 and h(STUDY+'TESTS.txt') == plan['tests_sha256'], 'synthetic test receipt')
    require(plan['synthetic_tests_passed'] == plan['producer_tests_passed'] + plan['independent_tests_passed'] and plan['producer_tests_passed'] > 0 and plan['independent_tests_passed'] > 0, 'test suite count binding')
    hashes = common.verify_hashes(root, design['source_hashes'])
    implementation = common.verify_hashes(root, plan['implementation_hashes'])
    independent = common.verify_hashes(root, plan['independent_review_hashes'])
    require(set(implementation) == {STUDY+n for n in ('parser.py', 'admission.py', 'build.py', 'register.py', 'test_parser.py', 'test_admission.py')}, 'implementation closure')
    reviewer = {str(p.relative_to(ROOT)): h(str(p.relative_to(ROOT))) for p in source_files()}
    require(all(independent.get(p) == v or hashes.get(p) == v for p, v in reviewer.items()), 'independent source closure')
    require(h(STUDY+'SOURCE_INVENTORY.json') == design['source_inventory_sha256'] and load(STUDY+'SOURCE_INVENTORY.json')['source_hashes'] == hashes, 'source inventory exact')
    artifacts = {STUDY+n: h(STUDY+n) for n in ('DESIGN.json', 'PLAN.json', 'SOURCE_INVENTORY.json', 'TESTS.txt', 'ADMISSIONS.json', 'EVIDENCE.json', 'COVERAGE.json', 'PROOF.json')}
    documents = {p: (root/p).read_bytes() for p in hashes}
    expected = produce(design, documents)
    for name, value in expected.items():
        require(digest(load(STUDY+name)) == digest(value), 'complete independent output differs: '+name)
    ev = expected['EVIDENCE.json']; coverage = expected['COVERAGE.json']
    proof = dict(schema_version=1, policy_id=POLICY, design_sha256=design_sha, plan_sha256=plan_sha,
        source_hashes=hashes, implementation_hashes=implementation, independent_review_hashes=independent,
        output_hashes={n: h(STUDY+n) for n in expected}, original_documents_loaded=39, selected_filings_loaded=38,
        original_filings_parsed=sum(v['parsed'] is not None for v in ev['releases'].values()), original_record_objects_preserved=111,
        resolved_catalog_records=111, all_original_3904_records_unchanged=True, structural_candidate_rows=94, no_pair_rows=17,
        post_effective_excluded_rows=2, raw_pair_occurrences_checked=130, source_class=SOURCE_CLASS, evidence_class=EVIDENCE_CLASS,
        raw_facts_provenance_verified=False, filing_catalog_http_status_verified=False, continuous_security_identity_verified=False,
        retrospective_bridge_used_as_historical_evidence=False, source_rows_unchanged=True, factor_ready_count=0,
        strategy_trials=0, network_requests=0, independent_review_pending=True)
    require(digest(load(STUDY+'PROOF.json')) == digest(proof), 'complete proof differs')
    for mapping in (hashes, implementation, independent, reviewer, artifacts): common.verify_hashes(root, mapping)
    return dict(schema_version=1, policy_id=POLICY, passed=True, errors=[], design_sha256=design_sha, plan_sha256=plan_sha,
        source_hashes=hashes, implementation_hashes=implementation, independent_review_hashes=independent,
        auditor_source_hashes=reviewer, artifact_hashes=artifacts, summary=coverage,
        original_3904_objects_hash_verified=True, predecessor111_catalogs_independently_rebuilt=True,
        all46_unique_pairs_and130_occurrences_checked=True, original_candidate_cik_retained=True,
        explicit_predecessor_resolution_preserved=True, no_pair17_remain_unavailable=True, post_cutover2_excluded=True,
        causal_identity_anchor_and_document_identity_checked=True, contemporaneous_identity_not_backdated=True,
        source_class=SOURCE_CLASS, evidence_class=EVIDENCE_CLASS, raw_facts_provenance_verified=False,
        filing_catalog_http_status_verified=False, continuous_security_identity_verified=False,
        raw_original_html_independently_parsed=True, production_modules_imported=False,
        all_original_ordered_pairs_preserved=True, concept_contexts_kept_separate=True, source_flags_unchanged=True,
        factor_ready=False, factor_ready_count=0, growth_factor_rank_order_return_computed=False,
        network_requests=0, strategy_trials=0, observer_files_read_or_written=False)


if __name__ == '__main__':
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument('--design-sha', required=True)
    command.add_argument('--plan-sha', required=True)
    command.add_argument('--output', default='work/monthly_xom_predecessor_admission_independent_review.json')
    args = command.parse_args()
    result = audit(ROOT, args.design_sha, args.plan_sha)
    common.immutable_write(ROOT/args.output, result)
    print(json.dumps(dict(passed=True, output=args.output, admitted=result['summary']['semantic_admitted_count'],
                          unverified=result['summary']['unverified_count'], unavailable=result['summary']['unavailable_count']), sort_keys=True))
