"""Pinned Apple annual operating-income source evidence, with no ratios or trading.

The real CLI is gated by the common source PLAN before it reads any original
PDF. Pure parsing and geometry APIs allow fabricated tests before registration.
"""
import argparse
from bisect import bisect_right
from copy import deepcopy
from datetime import date, datetime, timezone
from decimal import Decimal
import hashlib
import io
import json
from pathlib import Path
import re
import sys
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
import operating_margin172_common as common

TITLE = 'CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS'
MONTHS = {name: i + 1 for i, name in enumerate(('January February March April May June July August September October November December').split())}
DATE = re.compile(r'(' + '|'.join(MONTHS) + r')\s+(\d{1,2}),')
AMOUNT = re.compile(r'\(?-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\)?')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def norm(text):
    return ' '.join(text.split())


def decimal_cell(cell):
    require(isinstance(cell, str) and AMOUNT.fullmatch(cell), 'Invalid exact monetary cell')
    negative = cell.startswith('(')
    require(negative == cell.endswith(')') and not (negative and '-' in cell), 'Invalid monetary sign')
    value = Decimal(cell.strip('()').replace(',', ''))
    require(value.is_finite(), 'Nonfinite monetary cell')
    return format(-value if negative else value, 'f')


def headers(lines, count):
    found = []
    for i, line in enumerate(lines[:12]):
        parts = DATE.findall(line)
        if not parts:
            continue
        years, following = [], ''
        for following in lines[i + 1:i + 4]:
            years = re.findall(r'\b(?:19|20)\d{2}\b', following)
            if years:
                break
        require(len(parts) == len(years) == count, 'Ambiguous date/year columns')
        dates = [date(int(y), MONTHS[m], int(d)).isoformat() for (m, d), y in zip(parts, years)]
        found.append(dict(dates=dates, line_1based=i + 1, lines=[norm(line), norm(following)]))
    require(len(found) == 1, 'Unique financial date header required')
    return found[0]


def label_pattern(label):
    return re.compile(r'\s*' + r'\s*'.join(re.escape(c) for c in label if not c.isspace()) + r'\s+(.+)', re.I)


def row_at(lines, labels, start, stop, sales=False):
    matches = []
    for label in labels:
        pattern = label_pattern(label)
        for i in range(start, stop):
            match = pattern.fullmatch(lines[i])
            if not match:
                continue
            cells = match[1].replace('/dollarsign', '$').replace('$', '').split()
            annotation = None
            if sales and label == 'Total net sales' and len(cells) == 5 and cells[0] == '(1)':
                refs = [dict(page_1based=1, line_1based=j + 1, text=v) for j, v in enumerate(lines)
                        if re.match(r'\s*\(1\)\s+Net sales by\s+', v)]
                require(refs, 'Unresolved net-sales footnote')
                annotation, cells = dict(literal='(1)', reference_lines=refs), cells[1:]
            require(len(cells) == 4, 'Exactly four monetary columns required for ' + label)
            for cell in cells:
                decimal_cell(cell)
            matches.append(dict(label=label, cells=cells, line_1based=i + 1,
                                text=lines[i], label_footnote=annotation))
    require(len(matches) == 1, 'Unique exact statement row required: ' + '/'.join(labels))
    return matches[0]


def currency_context(lines, sales):
    line, number = sales['text'], sales['line_1based']
    scope = 'same statement total net-sales row'
    if '$' not in line and '/dollarsign' not in line:
        sections = [i for i, v in enumerate(lines[:number - 1]) if re.fullmatch(r'\s*Net sales:\s*', v)]
        require(len(sections) == 1, 'Unique sales currency section required')
        supporting = [(j + 1, lines[j]) for j in range(sections[0] + 1, number - 1)
                      if re.match(r'^\s*Products\s+', lines[j])]
        require(len(supporting) == 1, 'Unique Products currency row required')
        number, line = supporting[0]
        require(line.replace('/dollarsign', '$').count('$') == 4, 'Four sales-block currency glyphs required')
        scope = 'same statement net-sales block Products row'
    return dict(literal='$' if '$' in line else '/dollarsign', text=line,
                text_sha256=digest(line.encode()), locator=dict(page_1based=1, line_1based=number),
                scope=scope, iso_currency=None, ISO_currency_verified=False)


def parse_layout(texts, prior):
    """Extract four annual cells from fabricated or original layout text.

    Prior is the pinned v4 record, used for source scope and exact revenue parity,
    never as a replacement for a missing operating-income observation.
    """
    require(len(texts) >= 2, 'Operations and balance-sheet pages required')
    op, bs = texts[0].splitlines(), texts[1].splitlines()
    for text, title in zip(texts[:2], (TITLE, 'CONDENSED CONSOLIDATED BALANCE SHEETS')):
        require('Apple Inc.' in text and title in text and 'unaudited' in text.lower(), 'Original consolidated Apple statement identity required')
    require(not re.search(r'NON[ -]?GAAP|RECONCILIATION', texts[0], re.I), 'Non-GAAP statement excluded')
    head = re.sub(r'\s+', '', ''.join(op[:9]))
    require(head.count('ThreeMonthsEnded') == head.count('TwelveMonthsEnded') == 1
            and head.index('ThreeMonthsEnded') < head.index('TwelveMonthsEnded'), 'Annual duration columns ambiguous')
    hs, hb = headers(op, 4), headers(bs, 2)
    dates = hs['dates']
    require(dates == prior['metadata']['header_dates'], 'Frozen revenue date headers differ')
    require(dates[0] == dates[2] == hb['dates'][0] and dates[1] == dates[3] == hb['dates'][1], 'Annual operations/balance-sheet dates disagree')
    require(date.fromisoformat(dates[2]).year == prior['fiscal_year'], 'Wrong fiscal year')
    require(dates[2] < prior['publication']['release_date'], 'Release must follow fiscal end')
    unit_lines = [(i + 1, v) for i, v in enumerate(op[:9]) if re.search(r'\(In millions,', v)]
    require(len(unit_lines) == 1, 'Unique explicit in-millions unit required')
    cost = [i for i, v in enumerate(op) if re.match(r'^\s*Cost\s+of\s+sales(?:\s*:|\s+)', v, re.I)]
    require(cost, 'Formal cost-of-sales boundary required')
    net = [i for i, v in enumerate(op) if label_pattern('Net income').fullmatch(v)]
    require(net and net[0] > cost[0], 'Formal net-income boundary required')
    sales = row_at(op, ['Net sales', 'Total net sales'], 0, cost[0], sales=True)
    income = row_at(op, ['Operating income'], cost[0] + 1, net[0])
    currency = currency_context(op, sales)
    fields = {}
    for name, row in [('revenue', sales), ('operating_income', income)]:
        fields[name] = {}
        for role, col in [('current', 2), ('comparative', 3)]:
            period = deepcopy(prior['metadata'][role + '_period'])
            require(period['end'] == dates[col] and period['duration_literal'] == 'Twelve Months Ended', 'Prior period metadata inconsistent')
            value = decimal_cell(row['cells'][col])
            if name == 'revenue':
                old = prior['revenue'][role]
                require(old['amount_decimal'] == value and old['all_row_cells'] == row['cells']
                        and old['date'] == dates[col] and old['scale'] == 1000000, 'Fresh extraction does not match pinned revenue')
            own_currency = '$' if '$' in row['text'] else '/dollarsign' if '/dollarsign' in row['text'] else None
            fields[name][role] = dict(measure='total_net_sales' if name == 'revenue' else name,
                measure_label=row['label'], displayed_amount=row['cells'][col], amount_decimal=value,
                sign=-1 if Decimal(value) < 0 else 1 if Decimal(value) > 0 else 0,
                scale=1000000, unit_literal=unit_lines[0][1],
                unit_locator=dict(page_1based=1, line_1based=unit_lines[0][0]),
                currency_literal=own_currency, statement_currency_evidence=deepcopy(currency),
                iso_currency=None, ISO_currency_verified=False, period=period, date=dates[col],
                header_dates=dates, date_header_lines=hs['lines'], all_row_cells=row['cells'],
                row_text=row['text'], row_text_sha256=digest(row['text'].encode()),
                label_footnote=row['label_footnote'],
                locator=dict(page_1based=1, line_1based=row['line_1based'], financial_column_index_0based=col),
                known_at_session=prior['availability']['candidate_first_usable_session'],
                release_date=prior['publication']['release_date'],
                vintage_role='own_year_original' if role == 'current' else 'same_current_release_prior_year_comparative',
                historical_original_event_created=role == 'current', fact_admitted=False, strategy_admitted=False)
    statement = dict(title=TITLE, page_1based=1, annual_duration_literal='Twelve Months Ended',
        annual_columns_0based=dict(current=2, comparative=3), header_dates=dates,
        date_header_lines=hs['lines'], header_line_1based=hs['line_1based'],
        unit_literal=unit_lines[0][1], unit_line_1based=unit_lines[0][0],
        currency_evidence=currency, revenue_exact_parity=True,
        comparative_boundaries_reused_from_pinned_v4=True, accounting_chain_reaudited=False)
    return dict(statement=statement, fields=fields)


def boxes(textpage, term):
    search, out = textpage.search(term, match_case=False), []
    try:
        while True:
            hit = search.get_next()
            if hit is None:
                break
            index, count = hit
            coords = [textpage.get_charbox(j) for j in range(index, index + count)]
            out.append(dict(index=index, count=count, text=textpage.get_text_range(index, count),
                bbox=[min(c[0] for c in coords), min(c[1] for c in coords),
                      max(c[2] for c in coords), max(c[3] for c in coords)]))
    finally:
        search.close()
    return out


def attach_geometry(parsed, textpage, finder=boxes):
    def cy(box):
        return (box['bbox'][1] + box['bbox'][3]) / 2
    titles, costs, nets = finder(textpage, TITLE), finder(textpage, 'Cost of sales'), finder(textpage, 'Net income')
    require(len(titles) == 1 and costs and nets, 'Formal statement geometry boundaries missing')
    title_y, cost_y = cy(titles[0]), max(map(cy, costs))
    lower = [cy(b) for b in nets if cy(b) < cost_y]
    require(lower, 'Net-income geometry below cost boundary missing')
    net_y = max(lower)
    parsed['statement']['pdfium_title_bbox'] = titles[0]['bbox']
    parsed['statement']['pdfium_boundaries'] = dict(cost_y=cost_y, net_income_y=net_y)
    for term, key in [('Twelve Months Ended', 'pdfium_annual_duration_bbox'), ('In millions', 'pdfium_unit_bbox')]:
        header_boxes = [b for b in finder(textpage, term) if cost_y < cy(b) < title_y]
        require(len(header_boxes) == 1, 'Unique scoped PDF header required: ' + term)
        parsed['statement'][key] = header_boxes[0]['bbox']
    for name, roles in parsed['fields'].items():
        row = roles['current']
        labels = [b for b in finder(textpage, row['measure_label'])
                  if (cost_y < cy(b) < title_y if name == 'revenue' else net_y < cy(b) < cost_y)]
        require(len(labels) == 1, 'Unique scoped PDF row label required: ' + name)
        label, matches = labels[0], {}
        for value in set(row['all_row_cells']):
            for found in finder(textpage, value):
                if abs(cy(found) - cy(label)) < 4 and found['bbox'][0] > label['bbox'][2]:
                    matches[found['index']] = found
        ordered = sorted(matches.values(), key=lambda b: b['bbox'][0])
        require(len(ordered) == 4 and [norm(x['text']) for x in ordered] == row['all_row_cells'], 'Independent PDF engines disagree: ' + name)
        require(all(ordered[i]['bbox'][2] < ordered[i + 1]['bbox'][0] for i in range(3)), 'Overlapping numeric columns')
        for role, col in [('current', 2), ('comparative', 3)]:
            cell = ordered[col]
            roles[role]['locator'].update(pdfium_label_bbox=label['bbox'], pdfium_cell_bbox=cell['bbox'],
                pdfium_char_index=cell['index'], pdfium_char_count=cell['count'])
            roles[role]['pdfium_all_row_cells'] = ordered
            roles[role]['independent_PDFium_cell_text_match'] = True


def first_session(release, sessions):
    date.fromisoformat(release)
    require(sessions and sessions == sorted(set(sessions)), 'Ordered unique calendar required')
    for day in sessions:
        date.fromisoformat(day)
    require(sessions[0] <= release < sessions[-1], 'Release outside pinned calendar')
    return sessions[bisect_right(sessions, release)]


def receipt_check(meta, raw, url, pdf=False):
    require(meta.get('status', meta.get('http_status')) == 200, 'Receipt is not HTTP200')
    require(urlparse(url).scheme == 'https' and urlparse(url).hostname == 'www.apple.com', 'Original Apple HTTPS URL required')
    require(meta.get('url', meta.get('source_url')) == url and meta.get('final_url') == url, 'Receipt URL mismatch')
    require(meta.get('public_unauthenticated') is True and not meta.get('error'), 'Public source receipt required')
    require(meta.get('response_sha256', meta.get('sha256')) == digest(raw), 'Receipt digest mismatch')
    require(meta.get('response_bytes', meta.get('size')) == len(raw), 'Receipt byte count mismatch')
    stamp = datetime.fromisoformat(meta['retrieved_at'].replace('Z', '+00:00'))
    require(stamp.utcoffset() is not None, 'Retrieval timezone required')
    require(meta.get('content_type', '').split(';')[0].lower() == ('application/pdf' if pdf else 'text/html'), 'Receipt MIME mismatch')
    if pdf:
        require(raw.startswith(b'%PDF-') and meta.get('decoder') == 'identity'
                and meta.get('decoded_sha256') == digest(raw), 'Identity PDF required')


def jsonld_dates(value):
    if isinstance(value, dict):
        own = [value['datePublished'][:10]] if isinstance(value.get('datePublished'), str) else []
        return own + [d for item in value.values() for d in jsonld_dates(item)]
    return [d for item in value for d in jsonld_dates(item)] if isinstance(value, list) else []


def publication(raw, prior, parent_url):
    soup = BeautifulSoup(raw, 'html.parser')
    release = prior['publication']['release_date']
    visible = [norm(x.get_text(' ', strip=True)) for x in soup.select('.category-eyebrow__date')]
    require(len(visible) == 1 and datetime.strptime(visible[0], '%B %d, %Y').date().isoformat() == release, 'Visible parent date mismatch')
    structured = [d for script in soup.select('script[type="application/ld+json"]')
                  for d in jsonld_dates(json.loads(script.get_text()))]
    require(structured and set(structured) == {release}, 'Structured parent date mismatch')
    titles = [norm(x.get_text(' ', strip=True)) for x in soup.find_all('h1')]
    require(len(titles) == 1 and 'fourth quarter' in titles[0].lower(), 'Fourth-quarter parent title required')
    require(prior['source_url'] in [urljoin(parent_url, a['href']) for a in soup.find_all('a', href=True)], 'Direct original PDF link missing')
    return dict(release_date=release, precision='date_only', visible_date=visible[0], structured_dates=structured,
        headline=titles[0], publication_timezone='America/Los_Angeles', direct_pdf_link_verified=True,
        first_public_version_verified=False, historical_web_snapshot_verified=False)


def read_bound(ref, pins):
    path = common.source_path(ref['path'])
    require(pins.get(str(path)) == ref['sha256'], 'Reference absent from source PLAN: ' + str(path))
    raw = path.read_bytes()
    require(digest(raw) == ref['sha256'] and len(raw) == ref['bytes'], 'Original source binding changed')
    return raw


def extract_record(prior, pins, out, plan_sessions):
    """Call only after common.gate; source integrity is fatal, semantic gaps held."""
    raw = read_bound(prior['source'], pins)
    parent = read_bound(prior['parent_release'], pins)
    receipt = json.loads(read_bound(prior['receipt'], pins))
    parent_receipt = json.loads(read_bound(prior['parent_receipt'], pins))
    parent_url = parent_receipt.get('url', parent_receipt.get('source_url'))
    receipt_check(receipt, raw, prior['source_url'], pdf=True)
    receipt_check(parent_receipt, parent, parent_url)
    calendar = json.loads(read_bound(prior['availability']['calendar'], pins))
    sessions = calendar['sessions'] if isinstance(calendar, dict) else calendar
    known = first_session(prior['publication']['release_date'], sessions)
    require(known == prior['availability']['candidate_first_usable_session'], 'Pinned next-session clock differs')
    require(known == first_session(prior['publication']['release_date'], plan_sessions), 'Original and new registered calendars disagree')
    for role in ('current', 'comparative'):
        require(prior['revenue'][role]['known_at_session'] == known, 'Prior revenue clock differs')
    common.verify_pdf_runtime()
    if str(common.PDF_BUNDLE) not in sys.path:
        sys.path.append(str(common.PDF_BUNDLE))
    from pypdf import PdfReader
    import pypdfium2 as pdfium
    reader = PdfReader(io.BytesIO(raw))
    require(len(reader.pages) >= 2, 'Original Apple statement pages missing')
    texts = [reader.pages[i].extract_text(extraction_mode='layout') for i in range(2)]
    record = {k: deepcopy(prior[k]) for k in ('symbol', 'fiscal_year', 'source', 'source_url', 'receipt', 'parent_release', 'parent_receipt')}
    record.update(availability=deepcopy(prior['availability']), original_pdf_sha256=digest(raw),
        original_page_count=len(reader.pages), text_sha256=[digest(x.encode()) for x in texts],
        known_at_session=known, status='HELD', fields=None, statement=None,
        publication=None, source_retrieved_at=receipt['retrieved_at'],
        parent_retrieved_at=parent_receipt['retrieved_at'], registered_calendar_clock_verified=True,
        holds=['currency_iso_unverified', 'first_public_version_unverified', 'accounting_continuity_review_pending',
               'independent_operating_income_review_pending'], fact_admitted=False, strategy_admitted=False)
    if prior['metadata']['comparative_period'].get('start') is None:
        record['holds'].append('comparative_period_start_unknown')
    doc = pdfium.PdfDocument(raw)
    page = doc[0]
    textpage = page.get_textpage()
    try:
        imagepath = out / ('FY%d_operations.png' % prior['fiscal_year'])
        require(not imagepath.exists(), 'Render target already exists')
        page.render(scale=1.5).to_pil().save(imagepath)
        record['render'] = dict(path=str(imagepath.relative_to(common.ROOT)), sha256=common.sha(imagepath), page_1based=1)
        try:
            record['publication'] = publication(parent, prior, parent_url)
            parsed = parse_layout(texts, prior)
            attach_geometry(parsed, textpage)
            record.update(parsed)
            record['status'] = 'EXTRACTED_SOURCE_CANDIDATE'
            for roles in record['fields'].values():
                for cell in roles.values():
                    cell['source'] = deepcopy(prior['source'])
        except (ValueError, KeyError, TypeError) as error:
            record['holds'].append('source_semantics: ' + str(error))
    finally:
        textpage.close()
        page.close()
        doc.close()
    return record


def run(plan_sha256):
    plan = common.gate(plan_sha256)
    prior = common.read(plan['source_manifests']['aapl'])
    rows = prior['rows']
    require([r['fiscal_year'] for r in rows] == list(range(2016, 2026))
            and all(r['symbol'] == 'AAPL' for r in rows), 'Exact ten-original source scope required')
    out = common.OUT / 'aapl'
    require(not out.exists(), 'Apple source extraction is write-once')
    out.mkdir()
    plan_sessions = common.read(plan['calendar'])['sessions']
    records = [extract_record(row, plan['input_sha256'], out, plan_sessions) for row in rows]
    common.gate(plan_sha256)
    result = dict(schema='operating_margin172_aapl_source_v1', plan_sha256=plan_sha256,
        generated_utc=datetime.now(timezone.utc).isoformat(), records=records,
        report_count=len(records), extracted_count=sum(r['status'] == 'EXTRACTED_SOURCE_CANDIDATE' for r in records),
        quantities_per_extracted_report=4, financial_ratios_factors_orders_returns=0,
        source_fields_admitted=False, accounting_chain_reaudited=False, observer_writes=0,
        limitations=['Original bytes were retrieved later, not a certified pristine historical web vintage.',
          'Current and comparative values use the same current parent-release next-session clock.',
          'Dollar glyphs are preserved as table context; no ISO currency inferred.',
          'Pinned v4 comparative period boundaries are reused, not newly audited.',
          'All extracted values await independent operating-income semantic review.'])
    common.dump(out / 'REPORTS.json', result)
    print(json.dumps(dict(status='SOURCE_EXTRACTION_RECORDED', reports=10, extracted=result['extracted_count'],
                         sha256=common.sha(out / 'REPORTS.json'))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan-sha256', required=True)
    run(parser.parse_args().plan_sha256)
