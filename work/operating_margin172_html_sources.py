"""Source-only extraction of four annual statement cells from 25 bound HTMLs.

No price, ratio, rank, model or portfolio code is imported.  The pure parser
accepts bytes; only ``run`` reads real sources, after the parent source PLAN
has passed its complete hash gate.  Extraction is never field admission.
"""
import argparse
from copy import deepcopy
from datetime import date, datetime, timezone
from decimal import Decimal
import hashlib
import json
import re
from urllib.parse import urlparse

from bs4 import BeautifulSoup
import operating_margin172_common as common


SCOPE = {'MSFT': tuple(range(2015, 2027)), 'ABBV': tuple(range(2013, 2026))}
LIMITS = [
    'Source extraction only; operating-income accounting comparability and field admission require independent review.',
    'Both annual cells inherit the CURRENT report known_by; a comparative is never backdated.',
    'Original-dated documents were retrieved later; historical first-public bytes and complete amendment history are not certified.',
    'Dollar-literal evidence is preserved; no new ISO-currency certification or FX conversion.',
    'Reported GAAP amounts may include acquisitions, disposals, impairments and reclassifications; no normalization or organic-growth claim.',
]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def text(tag):
    return ' '.join(tag.get_text(' ', strip=True).split())


def decimal_amount(value):
    """Parse only an explicitly printed signed amount, never a dash or note."""
    value = ' '.join(value.split()).replace('\u2212', '-')
    require(value.count('$') <= 1, 'Multiple dollar markers in amount')
    if '$' in value:
        require(value.startswith('$'), 'Dollar marker must precede amount')
        value = value[1:].strip()
    negative = value.startswith('(') and value.endswith(')')
    if negative:
        value = value[1:-1].strip()
    require(re.fullmatch(r'-?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?', value)
            is not None, 'Ambiguous printed amount: ' + value)
    require(not (negative and value.startswith('-')), 'Double negative amount')
    number = Decimal(value.replace(',', ''))
    if negative:
        number = -number
    return format(number, 'f')


def table_grid(table):
    rows = [row for row in table.find_all('tr') if row.find_parent('table') is table]
    grid, physical = {}, {}
    for ri, row in enumerate(rows):
        ci = 0
        for pi, cell in enumerate(row.find_all(['td', 'th'], recursive=False)):
            while (ri, ci) in grid:
                ci += 1
            spans = []
            for key in ('rowspan', 'colspan'):
                raw = str(cell.get(key, '1'))
                require(re.fullmatch(r'[1-9]\d*', raw) is not None, 'Invalid table span')
                spans.append(int(raw))
            rs, cs = spans
            require(rs <= 100 and cs <= 100, 'Excessive table span')
            item = dict(text=text(cell), tag=cell, row=ri, physical_cell=pi,
                        columns=list(range(ci, ci + cs)), rowspan=rs)
            physical[ri, pi] = item
            for rr in range(ri, ri + rs):
                for cc in item['columns']:
                    require((rr, cc) not in grid, 'Overlapping table cells')
                    grid[rr, cc] = item
            ci += cs
    return rows, grid, physical


def headers(grid, col, stop):
    result = []
    for ri in range(stop):
        item = grid.get((ri, col))
        if item and item['text'] and item['text'] not in result:
            result.append(item['text'])
    return result


def annual_end(parts, issuer):
    """An annual duration AND a unique year owned by this column are required."""
    joined = ' '.join(parts)
    if issuer == 'MSFT':
        if not re.search(r'\bTwelve Months Ended\b', joined, re.I):
            return None
        years = re.findall(r'June\s+30\s*,?\s+(\d{4})\b', joined, re.I)
        return years[0] + '-06-30' if len(set(years)) == 1 else None
    annual = [p for p in parts if re.fullmatch(
        r'Twelve Months Ended (?:December|Dec\.) 31', p, re.I)]
    years = [p for p in parts if re.fullmatch(r'\d{4}', p)]
    return years[0] + '-12-31' if len(annual) == 1 and len(years) == 1 else None


def statement_context(table, rows, stop):
    inside = ' '.join(text(row) for row in rows[:stop])
    outside = []
    caption = table.find('caption', recursive=False)
    if caption:
        outside.append(text(caption))
    if table.parent.name == 'div' and 'c-table' in table.parent.get('class', []):
        for sibling in table.parent.previous_siblings:
            if not getattr(sibling, 'name', None):
                continue
            if sibling.name == 'div' or sibling.find('table'):
                break
            outside.append(text(sibling))
            if len(outside) >= 5:
                break
    return ' '.join(list(reversed(outside)) + [inside]).strip()


def check_receipt(report, raw, receipt):
    source = report['source']
    require(digest(raw) == source['raw']['sha256'], 'Raw source digest mismatch')
    if 'bytes' in source['raw']:
        require(len(raw) == source['raw']['bytes'], 'Raw source size mismatch')
    issuer = report['issuer']
    expected_url = (source['expected_url'] if issuer == 'MSFT'
                    else source['metadata']['source_url'])
    require(receipt.get('http_status', receipt.get('status')) == 200,
            'Original receipt is not HTTP 200')
    require(receipt.get('sha256') == digest(raw), 'Receipt/source digest mismatch')
    require(receipt.get('url') == expected_url, 'Receipt URL mismatch')
    host = 'www.microsoft.com' if issuer == 'MSFT' else 'news.abbvie.com'
    require(urlparse(expected_url).hostname == host, 'Unexpected issuer source host')
    if 'final_url' in receipt:
        require(receipt['final_url'] == expected_url, 'Receipt redirected source')
    if 'bytes' in receipt:
        require(receipt['bytes'] == len(raw), 'Receipt source size mismatch')
    retrieved = datetime.fromisoformat(receipt['retrieved_at'].replace('Z', '+00:00'))
    known = datetime.fromisoformat(report['known_by'].replace('Z', '+00:00'))
    require(retrieved.utcoffset() is not None and known.utcoffset() is not None,
            'Timezone-aware source/known clocks required')
    release = date.fromisoformat(report['release_date'])
    require(retrieved.date() >= release and known.date() > release,
            'Current-release conservative source clock mismatch')
    return dict(release_date=release.isoformat(), known_by=report['known_by'],
                retrieved_at=receipt['retrieved_at'],
                publication_precision='inherited_current_release_date_only',
                first_public_bytes_verified=False,
                comparative_backdated=False)


def _periods(report):
    result = {}
    for side, prefix in (('current', ''), ('comparative', 'comparative_')):
        start, end = report[prefix + 'period_start'], report[prefix + 'period_end']
        a, b = date.fromisoformat(start), date.fromisoformat(end)
        require(330 <= (b - a).days + 1 <= 400, 'Unknown/nonannual period')
        expected_end = f"{report['fiscal_year'] - (side == 'comparative')}-" + (
            '06-30' if report['issuer'] == 'MSFT' else '12-31')
        require(end == expected_end, 'Unexpected annual period end')
        expected_start = f'{b.year - 1}-07-01' if report['issuer'] == 'MSFT' else f'{b.year}-01-01'
        require(start == expected_start, 'Unexpected annual period start')
        result[side] = dict(start=start, end=end, days_inclusive=(b - a).days + 1,
                            start_source='inherited_reviewed_revenue_period',
                            start_is_direct_quote=False)
    return result


def _extract(report, raw, receipt):
    issuer, year = report['issuer'], report['fiscal_year']
    require(issuer in SCOPE and year in SCOPE[issuer], 'Report outside fixed HTML scope')
    clock = check_receipt(report, raw, receipt)
    periods = _periods(report)
    soup = BeautifulSoup(raw, 'html.parser')
    tables = soup.find_all('table')
    old_cur, old_cmp = report['current_locator'], report['comparative_locator']
    ti = old_cur['table_index_zero_based']
    require(type(ti) is int and 0 <= ti < len(tables), 'Missing bound statement table')
    require(old_cmp['table_index_zero_based'] == ti, 'Revenue cells are in different tables')
    table = tables[ti]
    rows, grid, physical = table_grid(table)
    cur_ri, cmp_ri = old_cur['row_index_zero_based'], old_cmp['row_index_zero_based']
    require(cur_ri == cmp_ri and 0 <= cur_ri < len(rows), 'Missing bound revenue row')
    labels = {}
    for ri, row in enumerate(rows):
        row_cells = row.find_all(['td', 'th'], recursive=False)
        if row_cells:
            labels[ri] = text(row_cells[0]).rstrip(':').strip()
    if issuer == 'MSFT':
        total = [r for r, label in labels.items() if re.fullmatch(r'Total revenues?', label, re.I)]
        revenue_rows = total or [r for r, label in labels.items() if re.fullmatch(r'Revenues?', label, re.I)]
        oi_re = r'Operating income(?:\s*\(loss\))?'
        title_re = r'INCOME STATEMENTS|CONSOLIDATED STATEMENTS OF (?:INCOME|OPERATIONS)'
        cost_re = r'Cost of (?:revenue|goods)'
    else:
        revenue_rows = [r for r, label in labels.items() if label.lower() == (
            'net sales' if year <= 2014 else 'net revenues')]
        oi_re = r'Operating earnings(?:\s*\(loss\))?'
        title_re = r'AbbVie Inc\..*Consolidated Statements of Earnings'
        cost_re = r'Cost of products sold'
    require(revenue_rows == [cur_ri], 'Revenue total row is ambiguous or shifted')
    oi_rows = [r for r, label in labels.items() if re.fullmatch(oi_re, label, re.I)]
    require(len(oi_rows) == 1, 'Exactly one operating-income total row required')
    require(oi_rows[0] > cur_ri, 'Operating-income row precedes revenue')
    context = statement_context(table, rows, cur_ri)
    require(re.search(title_re, context, re.I), 'Formal consolidated statement title absent')
    require(not re.search(r'non[ -]?GAAP|\badjusted\b|reconciliation|pro forma', context, re.I),
            'Non-GAAP/adjusted statement heading')
    require(re.search(cost_re, text(table), re.I), 'Formal income-statement cost row absent')
    units = re.findall(r'\b[Ii]n\s+(millions|thousands)\b', context, re.I)
    require(len(set(u.lower() for u in units)) == 1, 'Missing/ambiguous monetary unit')
    scale = {'millions': '1000000', 'thousands': '1000'}[units[0].lower()]
    require(scale == str(report['scale']), 'Original revenue scale mismatch')
    require('$' in text(rows[cur_ri]), 'Revenue row dollar evidence absent')
    result_cells = {}
    for measure, ri in (('revenue', cur_ri), ('operating_income', oi_rows[0])):
        by_side = {side: [] for side in periods}
        for (r, pi), item in physical.items():
            if r != ri:
                continue
            owned = [(headers(grid, col, cur_ri), col) for col in item['columns']]
            ends = [annual_end(parts, issuer) for parts, _ in owned]
            for side, period in periods.items():
                if period['end'] not in ends:
                    continue
                require(all(e == period['end'] for e in ends), 'Cell crosses annual column ownership')
                if item['text'] in ('', '$'):
                    continue
                amount = decimal_amount(item['text'])
                require(item['rowspan'] == 1, 'Amount cell spans statement rows')
                if measure == 'revenue':
                    require(Decimal(amount) > 0, 'Positive revenue source amount required')
                raw_cell, raw_row = str(item['tag']), str(rows[ri])
                locator = dict(table_index_zero_based=ti, row_index_zero_based=ri,
                    physical_cell_index_zero_based=pi, logical_columns=item['columns'])
                cell = dict(amount_decimal=amount, displayed_amount=item['text'], scale=scale,
                    measure=measure, measure_label=labels[ri], locator=locator,
                    header_literals=owned[0][0], unit_literal='In ' + units[0].lower(),
                    currency_literal='$', currency_evidence_scope='same_statement_revenue_row',
                    currency_iso=None, currency_iso_newly_certified=False,
                    inherited_revenue_currency=report.get('currency'),
                    period_start=period['start'], period_end=period['end'],
                    known_by=clock['known_by'], release_date=clock['release_date'],
                    source_raw_sha256=digest(raw),
                    source_path=report['source']['raw']['path'],
                    cell_html=raw_cell, cell_html_sha256=digest(raw_cell.encode()),
                    row_html=raw_row, row_html_sha256=digest(raw_row.encode()),
                    row_text=text(rows[ri]), fact_admitted=False,
                    role='own_year_current' if side == 'current' else 'current_report_comparative')
                by_side[side].append(cell)
        require(all(len(v) == 1 for v in by_side.values()),
                'Unique current/comparative annual cells required for ' + measure)
        result_cells[measure] = {side: value[0] for side, value in by_side.items()}
    for side, old in (('current', old_cur), ('comparative', old_cmp)):
        extracted = result_cells['revenue'][side]
        old_cols = old.get('logical_columns', old.get('logical_columns_zero_based'))
        require(extracted['locator'] == dict(table_index_zero_based=ti,
            row_index_zero_based=cur_ri, physical_cell_index_zero_based=old['physical_cell_index_zero_based'],
            logical_columns=old_cols), 'Bound revenue locator mismatch')
        require(Decimal(extracted['amount_decimal']) == Decimal(report[side]), 'Bound revenue amount mismatch')
        require(extracted['cell_html_sha256'] == report[side + '_cell_html_sha256'],
                'Bound revenue cell digest mismatch')
        if 'row_html_sha256' in report:
            require(extracted['row_html_sha256'] == report['row_html_sha256'], 'Bound revenue row digest mismatch')
    notes = []
    for node in soup.find_all(['p', 'li']):
        if node.find(['p', 'li']):
            continue
        value = text(node)
        if re.search(r'restat|reclassif|recast|retrospect|accounting standard|revenue recognition|'
                     r'discontinued|impairment|acquisition', value, re.I):
            notes.append(dict(text=value, html_sha256=digest(str(node).encode()),
                interpretation='Unclassified source lead; not an accounting-clearance decision'))
    return dict(clock=clock, periods=periods,
        statement=dict(table_index_zero_based=ti, table_html_sha256=digest(str(table).encode()),
            context=context, header_rows=[[text(c) for c in row.find_all(['td', 'th'], recursive=False)]
                                         for row in rows[:cur_ri]],
            unit_literal='In ' + units[0].lower(), scale=scale),
        cells=result_cells, accounting_note_leads=notes)


def extract_report(report, raw, receipt):
    """Represent every supplied report, preserving failures as explicit holds."""
    result = dict(issuer=report.get('issuer'), fiscal_year=report.get('fiscal_year'),
        report_id=report.get('report_id'), source=deepcopy(report.get('source')),
        release_date=report.get('release_date'), known_by=report.get('known_by'),
        status='hold', cells={}, holds=['operating_income_field_review_required',
            'accounting_comparability_not_admitted'], limits=list(LIMITS),
        inherited_record=deepcopy(report), source_fields_admitted=False,
        financial_ratios_computed=0, financial_features_computed=0)
    try:
        result.update(_extract(report, raw, receipt))
        result['status'] = 'extracted'
        result['extraction_errors'] = []
    except (ValueError, KeyError, IndexError, TypeError, ArithmeticError) as exc:
        result['extraction_errors'] = [str(exc)]
        result['holds'].append('extraction_failed:' + str(exc))
    return result


def run(expected_plan_sha256):
    plan = common.gate(expected_plan_sha256)
    manifest = common.read(common.source_path(plan['source_manifests']['html']))
    for issuer, years in SCOPE.items():
        require(sorted(r['fiscal_year'] for r in manifest[issuer]) == list(years),
                'Complete fixed report scope required: ' + issuer)
    destination = common.OUT / 'html'
    require(not destination.exists(), 'HTML extraction is write-once')
    reports = []
    for issuer in SCOPE:
        for report in sorted(manifest[issuer], key=lambda r: r['fiscal_year']):
            require(report['issuer'] == issuer, 'Manifest issuer mismatch')
            source = report['source']
            raw = common.source_path(source['raw']['path']).read_bytes()
            receipt = common.read(common.source_path(source['receipt']['path']))
            reports.append(extract_report(report, raw, receipt))
    common.gate(expected_plan_sha256)
    payload = dict(schema='operating_margin172_html_sources_v1',
        plan_sha256=expected_plan_sha256, created_utc=datetime.now(timezone.utc).isoformat(),
        reports=reports, report_count=len(reports),
        extracted_reports=sum(r['status'] == 'extracted' for r in reports),
        held_reports=sum(r['status'] == 'hold' for r in reports),
        source_fields_admitted=False, financial_ratios_computed=0,
        financial_features_computed=0, models_fit=0, orders_or_returns_computed=0,
        source_manifest_sha256=common.sha(common.source_path(plan['source_manifests']['html'])))
    destination.mkdir()
    common.dump(destination / 'EXTRACTION.json', payload)
    print(json.dumps(dict(path=str(destination / 'EXTRACTION.json'),
        sha256=common.sha(destination / 'EXTRACTION.json'),
        reports=payload['report_count'], extracted=payload['extracted_reports'],
        held=payload['held_reports'], admitted=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-sha256', required=True)
    run(parser.parse_args().plan_sha256)
