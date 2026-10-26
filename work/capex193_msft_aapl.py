"""Source-only CAPEX193 extraction: all twelve MSFT and ten AAPL original reports.

No financial ratio, factor, portfolio, admission, or network operation. Existing
CFO table locators establish statement scope; the PP&E row is independently found.
PDF values require pypdf/PDFium agreement with complete annual-column geometry.
"""
import argparse
from bisect import bisect_right
from copy import deepcopy
from datetime import date, datetime, timezone
from decimal import Decimal, localcontext
import hashlib
import io
import json
from pathlib import Path
import re
import sys

from bs4 import BeautifulSoup
import cashflow_pdf_repair_v4 as pdf4
import cashflow_html_repair_v2 as html2

v1 = pdf4.v1
ROOT = Path(__file__).resolve().parents[1]
SCOPE = 'work/capex193_source_scope.json'
SCOPE_SHA = '71ccee9c200ae377010fb03a155bc7a6976f7f3f139db70a0a0ebdf49098fe08'
OUT = ROOT / 'outputs/monthly_capex193_source/msft_aapl'
DATA = 'outputs/monthly_cash_quality_source_admission_v1/DATASET.json'
PDF_PLAN = 'outputs/monthly_cashflow_pdf_repair_v4/PLAN.json'
PDF_RUNTIME = 'work/capex193_pdf_runtime/RUNTIME.json'
PDF_SITE = ROOT / 'work/capex193_pdf_runtime/site-packages'
CALENDAR = 'outputs/monthly_paired_target143/CALENDAR.json'
PINS = {
    PDF_RUNTIME: 'ad437674d06761ef278da476110c1d3aedf386cd5a0276161304627d2140c98a',
    'work/capex193_pdf_runtime/ACQUISITION.json': '477fe83ca171825d760284c9f1480702271c22e0446f179049d672b4e2570d34',
    PDF_PLAN: '6acd731fb3bcbcbaf0cf5c219c6bc5abaccd45fe7be85b95f79ce25ce707df11',
    CALENDAR: '8ee4da69fba84db1edee37a7e0f5af5d3ab16c6257e5d37cddbd531f01404622',
    'work/cashflow_pdf_repair_v4.py': '30629722f1dcec6dafd77f9f886d50b810e164ff562a4221723d4ab6caaed265',
    'work/cashflow_html_repair_v2.py': 'fb70dad51f2c4fcb5f79c74827fcadb616ceae89f4965555c9f883a91aa78f15',
    'work/cashflow_source_preflight_v1.py': 'c19b3b76bc2cabed6b4c1e3873be1a76e88e7675178debdc7b063e44afe5fbc9',
    'work/after141_amd_source_extract.py': '868cae81055318b9df934f4bfe2849c4971f861c4b480942139c8c7606ae4766',
    'work/after141_aapl_source_extract.py': 'a4e18ca94928e4924e8e6a40c8863e184c5fc42bf92a5da7cd9213313d496fda',
    'work/profitability141_issuer_statements.py': 'd6fea019c3ae155ddb61234b0ccf920092ab642de0a1fce166a24b87d29443de',
    'work/same_report144_admission.py': 'afb62d9286327d9849d9c42ca579c4566cbdc3bfdf7bcdc2274bfd8b30fd41c1',
}
HTML_LABEL = 'Additions to property and equipment'
PDF_LABEL = 'Payments for acquisition of property, plant and equipment'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def norm(text):
    return ' '.join(text.split())


def local(value):
    p = Path(value)
    p = (p if p.is_absolute() else ROOT / p).resolve()
    require(p.is_relative_to(ROOT) and not any(v in p.parts for v in ('CloudStorage', 'OneDrive')),
            'Source path outside local research root')
    return p


def verify(pins):
    for path, expected in pins.items():
        require(sha(local(path).read_bytes()) == expected, 'SHA mismatch: ' + path)


def publish(path, data):
    with Path(path).open('x') as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def outflow(displayed, scale):
    """Preserve the signed cash-flow literal; never abs an unexplained inflow."""
    normalized = re.sub(r'\s*,\s*', ',', displayed.strip())
    normalized = re.sub(r'[ \t]+(?=\)$)', '', normalized)
    value = v1.html.exact_amount(normalized)
    signed = Decimal(value)
    require(signed.is_finite() and signed <= 0, 'PP&E cash row must display an outflow or explicit zero')
    require(str(scale) in ('1000', '1000000'), 'Unsupported monetary scale')
    positive = signed.copy_abs()
    with localcontext() as ctx:
        ctx.prec = max(40, len(signed.as_tuple().digits) + 12)
        scaled = positive * Decimal(str(scale))
    return dict(signed_literal=displayed, normalized_monetary_literal=normalized,
                signed_value_decimal=format(signed, 'f'), positive_outflow_decimal=format(positive, 'f'),
                scale=str(scale), scaled_positive_outflow_decimal=format(scaled, 'f'))


def verify_cfo_locator(observed, event):
    locs = observed['candidates']['current']['operating_cash_flow']
    expected = event['fields']['current']['operating_cash_flow']['locator']
    require(len(locs) == 1 and locs[0] == expected, 'Original annual CFO statement locator differs')
    return deepcopy(locs[0])


def html_field(raw, event):
    loc = verify_cfo_locator(html2.inventory(raw, event['periods']), event)
    soup = BeautifulSoup(raw, 'html.parser')
    table = soup.find_all('table')[loc['table_index_zero_based']]
    rows, grid, physical = v1.html.table_grid(table)
    segment = loc.get('statement_segment', {})
    start, end = segment.get('start_row', 0), segment.get('end_row_exclusive', len(rows))
    labels = {ri: [x['text'] for (r, _), x in physical.items() if r == ri and re.search('[A-Za-z]', x['text'])]
              for ri in range(start, end)}
    matches = [ri for ri, items in labels.items() if any(norm(s).casefold() == HTML_LABEL.casefold() for s in items)]
    require(len(matches) == 1, 'Unique pure cash PP&E row required')
    ri = matches[0]
    investing = [r for r, items in labels.items() if any(re.fullmatch(r'Investing\s*:?', norm(x), re.I) for x in items)]
    financing = [r for r, items in labels.items() if any(re.fullmatch(r'Financing\s*:?', norm(x), re.I) for x in items)]
    require(len(investing) == 1 and investing[0] < ri, 'PP&E must be in explicit investing section')
    require(not any(investing[0] < r < ri for r in financing), 'PP&E row crossed financing boundary')
    columns = loc['logical_columns']
    chosen = sorted({grid[(ri, c)]['physical_cell'] for c in columns if (ri, c) in grid})
    require(chosen, 'Missing annual PP&E cell')
    require(all(physical[(ri, p)]['rowspan'] == 1 and set(physical[(ri, p)]['columns']).issubset(columns)
                for p in chosen), 'PP&E cell spans multiple financial periods')
    display = ' '.join(physical[(ri, p)]['text'] for p in chosen if physical[(ri, p)]['text']).strip()
    value = outflow(display, loc['scale'])
    # Independently retain the exact header-cell/year evidence for the selected columns.
    cfo_stop = next((r for r in range(start, loc['row_index_zero_based'])
                     if any(v1.NI.fullmatch(norm(s)) for s in labels[r])), None)
    require(cfo_stop is not None, 'Formal indirect-method cash-flow start missing')
    header_cells = []
    for c in columns:
        evidence = [x for x in v1.html.header_evidence(grid, cfo_stop, c) if x['row'] >= start]
        parts = [x['text'] for x in evidence]
        require(v1.html.dates(parts) == [loc['period_end']], 'Annual PP&E year column differs')
        require(v1.ANNUAL.search(' '.join(parts)) and not v1.QUARTER.search(' '.join(parts)), 'Interim PP&E column excluded')
        header_cells.append(dict(logical_column=c, cells=deepcopy(evidence)))
    row = rows[ri]
    cells = row.find_all(['td', 'th'], recursive=False)
    context = v1.html.external_context(table)
    if not v1.CF_TITLE.search(' '.join(context)):
        context.append(v1.legacy.statement_context(table))
    unit_evidence = [x['text'] for (r, _), x in physical.items() if start <= r < cfo_stop and re.search(r'\bmillions|\bthousands', x['text'], re.I)]
    unit_evidence += [x for x in context if re.search(r'\bmillions|\bthousands', x, re.I)]
    require(unit_evidence and v1.units(unit_evidence) == loc['scale'], 'Cash PP&E unit context absent')
    row_text = norm(row.get_text(' ', strip=True))
    own_currency = '$' if '$' in display else None
    require('$' in table.get_text(), 'Statement literal currency unavailable')
    field = dict(value, measure='current_annual_cash_ppe_purchases', row_label=HTML_LABEL,
                 row_text=row_text, row_text_sha256=sha(row_text.encode()),
                 row_html=str(row), row_html_sha256=sha(str(row).encode()),
                 cell_html=[str(cells[p]) for p in chosen],
                 cell_html_sha256=[sha(str(cells[p]).encode()) for p in chosen],
                 currency_literal=own_currency, currency_iso=None,
                 statement_currency_literal='$', currency_scope='same formal cash-flow table; ISO unverified',
                 period=deepcopy(event['periods']['current']), source=deepcopy(event['source']),
                 locator=dict(format='html', table_index_zero_based=loc['table_index_zero_based'],
                     row_index_zero_based=ri, physical_cell_indices_zero_based=chosen, logical_columns=columns,
                     statement_segment=segment or None, statement_title=loc['statement_title'],
                     table_html_sha256=sha(str(table).encode()), header_literals=loc['header_literals'],
                     header_cells=header_cells, unit_evidence=unit_evidence, period_end=loc['period_end'],
                     investing_section_row_zero_based=investing[0], original_cfo_locator=loc),
                 extraction_verified=True, fact_admitted=False)
    return field


def pdf_monetary_cells(tail, count):
    """Only whitespace before a closing parenthesis is normalized, never digit gaps."""
    tail = tail.replace('/dollarsign', '$').replace('$', '')
    number = r'(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?'
    token = re.compile(r'\(' + number + r'[ \t]*\)|-?' + number)
    matches = list(token.finditer(tail))
    require(len(matches) == count, 'PDF annual PP&E column count differs')
    position = 0
    raw, normalized = [], []
    for match in matches:
        require(not tail[position:match.start()].strip(), 'Unmapped PDF monetary characters')
        raw.append(match[0])
        clean = re.sub(r'[ \t]+(?=\)$)', '', match[0])
        v1.pdf.decimal_cell(clean)
        normalized.append(clean)
        position = match.end()
    require(not tail[position:].strip(), 'Unmapped PDF monetary characters')
    return raw, normalized


def pdf_layout(texts, event):
    loc = verify_cfo_locator(pdf4.inventory(texts, event['periods']), event)
    lines = texts[loc['page_1based'] - 1].splitlines()
    matches = [(i, v1.pdf_row_match(line, PDF_LABEL)) for i, line in enumerate(lines)
               if v1.pdf_row_match(line, PDF_LABEL)]
    require(len(matches) == 1, 'Unique pure cash PP&E PDF row required')
    ri, match = matches[0]
    investing = [i for i, s in enumerate(lines) if re.fullmatch(r'\s*Investing\s+activities\s*:\s*', s, re.I)]
    financing = [i for i, s in enumerate(lines) if re.fullmatch(r'\s*Financing\s+activities\s*:\s*', s, re.I)]
    require(len(investing) == 1 and investing[0] < ri, 'PDF PP&E must be in investing activities')
    require(not any(investing[0] < r < ri for r in financing), 'PDF PP&E crossed financing boundary')
    signed_cells, row_cells = pdf_monetary_cells(match[1], len(loc['header_dates']))
    chosen = signed_cells[loc['financial_column_index_0based']]
    caploc = dict(loc, row_label=PDF_LABEL, line_1based=ri + 1)
    return dict(outflow(chosen, loc['scale']), measure='current_annual_cash_ppe_purchases', row_label=PDF_LABEL,
                row_text=lines[ri], row_text_sha256=sha(lines[ri].encode()), all_row_cells=row_cells, all_row_signed_literals=signed_cells,
                monetary_normalization='Only blank space before closing parenthesis; no digit joining',
                currency_literal='$' if '$' in lines[ri] else '/dollarsign' if '/dollarsign' in lines[ri] else None,
                currency_iso=None, statement_currency_literal='$',
                currency_scope='same formal cash-flow page; ISO unverified',
                period=deepcopy(event['periods']['current']), source=deepcopy(event['source']),
                locator=dict(caploc, original_cfo_locator=loc, investing_section_line_1based=investing[0] + 1),
                fact_admitted=False)


def configure_pdf193(runtime):
    """Use only the explicitly admitted193 toolchain; never load the changed shared binary."""
    require(sha(local(PDF_RUNTIME).read_bytes()) == PINS[PDF_RUNTIME], 'CAPEX193 PDF runtime manifest changed')
    expected = json.loads(local(PDF_RUNTIME).read_text())
    require(runtime == expected, 'Exact CAPEX193 PDF runtime required')
    require(Path(runtime['pdf_runtime']['bundle_site_packages']).resolve() == PDF_SITE,
            'CAPEX193 isolated runtime path required')
    for name, digest in runtime['pdf_runtime_hashes'].items():
        path = Path(name)
        require(path.resolve() == path and path.is_relative_to(PDF_SITE), 'PDF dependency outside isolated193 runtime')
        require(sha(path.read_bytes()) == digest, 'CAPEX193 PDF dependency hash changed: ' + name)
    if str(PDF_SITE) not in sys.path:
        sys.path.insert(0, str(PDF_SITE))
    import pypdf
    import pypdfium2
    import pypdfium2_raw
    for module in (pypdf, pypdfium2, pypdfium2_raw):
        require(Path(module.__file__).resolve().is_relative_to(PDF_SITE), 'PDF imported from wrong runtime')
    return pypdf.PdfReader, pypdfium2


def pdf_field(raw, event, pdf_plan, render_dir=None):
    reader, pdfium = configure_pdf193(pdf_plan)
    parsed_pdf = reader(io.BytesIO(raw))
    require(1 <= len(parsed_pdf.pages) <= 12, 'Bounded annual PDF required')
    texts = [p.extract_text(extraction_mode='layout') for p in parsed_pdf.pages]
    field = pdf_layout(texts, event)
    loc = field['locator']
    doc = pdfium.PdfDocument(raw)
    page = doc[loc['page_1based'] - 1]
    textpage = page.get_textpage()
    try:
        field['pdfium_geometry'] = pdf4.geometry_rows(pdf4.physical_rows(textpage), loc, field['all_row_cells'])
        field['pdfium_text_agrees'] = True
        field['extraction_verified'] = True
        if render_dir is not None:
            path = Path(render_dir) / ('AAPL_FY' + str(event['fiscal_year']) + '_cashflow.png')
            require(not path.exists(), 'Render output already exists')
            bitmap = page.render(scale=1.6)
            try:
                with path.open('xb') as output:
                    bitmap.to_pil().save(output, format='PNG')
            finally:
                bitmap.close()
            field['render'] = dict(path=str(path.relative_to(ROOT)), sha256=sha(path.read_bytes()), scale=1.6,
                                   page_1based=loc['page_1based'], visual_review_pending=True)
    finally:
        textpage.close(); page.close(); doc.close()
    return field


def record(event, field=None, error=None):
    old = deepcopy(event['cash_quality_admission']['reasons'])
    reasons = [('inherited:' + x) for x in old]
    if not event['cash_quality_admission']['admitted'] and not reasons:
        reasons.append('inherited:admission_false')
    if error:
        reasons.append('capex_extraction:' + error)
    return dict(event_id=event['event_id'], symbol=event['symbol'], fiscal_year=event['fiscal_year'],
                release_date=event['release_date'], available_from=event['available_from'],
                inherited_admitted=event['cash_quality_admission']['admitted'], inherited_holds=old,
                inherited_rejection_codes=deepcopy(event['inherited_rejection_codes']),
                inherited_accounting_review_flags=deepcopy(event['accounting_review_flags']),
                inherited_identity=deepcopy(event['identity']),
                source=deepcopy(event['source']), field=field,
                status='held' if reasons else 'verified', hold_reasons=reasons,
                extraction_verified=field is not None and field.get('extraction_verified') is True,
                fact_admitted=False, strategy_admitted=False,
                first_public_version_verified=False, complete_correction_history_verified=False,
                certified_daily_security_master=False)


def gate(expected, needs_pdf=True):
    require(expected == SCOPE_SHA, 'Exact CAPEX193 scope SHA required')
    verify({SCOPE: expected, **PINS})
    scope = json.loads(local(SCOPE).read_text())
    verify(scope['input_sha256'])
    dataset = json.loads(local(DATA).read_text())
    events = [e for e in dataset['events'] if e['symbol'] in ('MSFT', 'AAPL')]
    targets = [e for e in scope['prospective_targets'] if e['symbol'] in ('MSFT', 'AAPL')]
    require(len(events) == len(targets) == 22 and len({e['event_id'] for e in events}) == 22, 'All22 distinct reports required')
    require([e['event_id'] for e in events] == [e['event_id'] for e in targets], 'Scope event order differs')
    calendar = json.loads(local(CALENDAR).read_text())['sessions']
    pins = {SCOPE: expected, **PINS, **scope['input_sha256']}
    for event, target in zip(events, targets):
        require(event['source'] == target['source'] and event['available_from'] == target['inherited_available_from']
                and event['cash_quality_admission']['admitted'] == target['inherited_admitted'], 'Scope source/availability changed')
        require(event['fields']['current']['operating_cash_flow']['locator'] == target['current_cfo_locator'], 'Scope CFO locator changed')
        current = event['periods']['current']
        require(current['period_start'] and 330 <= current['days_inclusive'] <= 400, 'Current annual period required')
        require((date.fromisoformat(current['period_end']) - date.fromisoformat(current['period_start'])).days + 1
                == current['days_inclusive'], 'Annual duration inconsistent')
        require(calendar[bisect_right(calendar, event['release_date'])] == event['available_from'], 'Next-session publication clock differs')
        for item in (event['source'], event['source']['receipt']):
            pins[item['path']] = item['sha256']
    verify(pins)
    pdf_plan = json.loads(local(PDF_RUNTIME).read_text())
    if needs_pdf:
        configure_pdf193(pdf_plan)
    return scope, events, pins, pdf_plan


def run(expected):
    scope, events, pins, pdf_plan = gate(expected)
    require(not (OUT / 'EXTRACTION.json').exists(), 'Write-once extraction already exists')
    codes = {str(Path(__file__).resolve().relative_to(ROOT)): sha(Path(__file__).read_bytes()),
             'work/test_capex193_msft_aapl.py': sha(local('work/test_capex193_msft_aapl.py').read_bytes())}
    OUT.mkdir(parents=True, exist_ok=True)
    renders = OUT / 'renders'; renders.mkdir(exist_ok=True)
    records = []
    for event in events:
        raw = local(event['source']['path']).read_bytes()
        require(sha(raw) == event['source']['sha256'], 'Raw source changed')
        try:
            field = pdf_field(raw, event, pdf_plan, renders) if event['symbol'] == 'AAPL' else html_field(raw, event)
            records.append(record(event, field))
        except (ValueError, IndexError, KeyError) as exc:
            records.append(record(event, error=str(exc)))
    verify(pins); verify(codes); configure_pdf193(pdf_plan)
    data = dict(schema='capex193_msft_aapl_source_extraction_v1', generated_utc=datetime.now(timezone.utc).isoformat(),
                scope_sha256=expected, records=records, report_count=22,
                extraction_verified_count=sum(r['extraction_verified'] for r in records),
                verified_count=sum(r['status'] == 'verified' for r in records), held_count=sum(r['status'] == 'held' for r in records),
                inherited_admitted_count=sum(r['inherited_admitted'] for r in records), source_hashes=pins,
                code_hashes=codes, pdf_runtime_hashes=pdf_plan['pdf_runtime_hashes'], runtime=sys.version,
                fact_admissions=0, financial_ratios=0, factors=0, model_fits=0, returns=0, network_requests=0,
                pdf_visual_review_pending=True, limits=scope['source_limitations'])
    publish(OUT / 'EXTRACTION.json', data)
    return {k: data[k] for k in ('report_count', 'extraction_verified_count', 'verified_count', 'held_count')} | {
        'sha256': sha((OUT / 'EXTRACTION.json').read_bytes()),
        'holds': [{'event_id': r['event_id'], 'reasons': r['hold_reasons']} for r in records if r['status'] == 'held']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scope-sha256', required=True)
    parser.add_argument('--inspect-html', action='store_true')
    parser.add_argument('--inspect-pdf', action='store_true')
    args = parser.parse_args()
    require(not (args.inspect_html and args.inspect_pdf), "Choose one inspection mode")
    if args.inspect_html or args.inspect_pdf:
        _, events, pins, runtime = gate(args.scope_sha256, needs_pdf=args.inspect_pdf)
        findings = []
        for e in events:
            if e['symbol'] != ('AAPL' if args.inspect_pdf else 'MSFT'):
                continue
            try:
                raw = local(e['source']['path']).read_bytes()
                f = pdf_field(raw, e, runtime) if args.inspect_pdf else html_field(raw, e)
                findings.append(dict(event_id=e['event_id'], extraction_verified=True, row=f['row_label'], column=f['locator'].get('logical_columns', f['locator'].get('financial_column_index_0based'))))
            except (ValueError, IndexError, KeyError) as ex:
                findings.append(dict(event_id=e['event_id'], extraction_verified=False, error=str(ex)))
        verify(pins)
        print(json.dumps(findings, indent=2))
    else:
        print(json.dumps(run(args.scope_sha256), indent=2))
