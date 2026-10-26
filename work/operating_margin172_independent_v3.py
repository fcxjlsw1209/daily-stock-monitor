"""Independent v3 source verification with explicit statement-caption ownership.

Parses original document bytes, never producer-selected amounts or financial
ratios. PDF content-order text is independent of production layout/geometry.
No prices, factors, rankings, orders, returns, or strategy admission occur here.
"""
from datetime import date, datetime, timedelta
from decimal import Decimal
import hashlib
import io
import re
import sys
from urllib.parse import urljoin, urlparse, unquote

from bs4 import BeautifulSoup


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def normalize(text):
    return ' '.join(str(text).replace('\u00a0', ' ').split())


def node_text(node):
    return normalize(node.get_text(' ', strip=True))


NUMBER = r'(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?'
MONEY = re.compile(r'(?:\(\s*\$?\s*' + NUMBER + r'\s*\)|\$?\s*[-+−]?' + NUMBER + r')')
PERIOD_DATE = re.compile(r'\b(September)\s+(\d{1,2}),?\s+(20\d{2})\b', re.I)
OI = re.compile(r'^(?:operating (?:income|earnings|loss)(?:\s*\(loss\))?|income from operations)$', re.I)


def amount(text):
    literal = normalize(text).replace('−', '-')
    require(MONEY.fullmatch(literal) is not None, 'Ambiguous monetary cell: ' + literal)
    neg = literal.startswith('(')
    cleaned = literal.strip('() ').replace('$', '').replace(',', '').replace(' ', '')
    value = Decimal(cleaned)
    require(value.is_finite(), 'Nonfinite monetary cell')
    return format(-value if neg else value, 'f')


def _rows_and_grid(table):
    rows = [row for row in table.find_all('tr') if row.find_parent('table') is table]
    physical = [row.find_all(['th', 'td'], recursive=False) for row in rows]
    grid, occupied = {}, {}
    for ri, cells in enumerate(physical):
        col = 0
        for ci, cell in enumerate(cells):
            while (ri, col) in grid:
                col += 1
            width, height = int(cell.get('colspan', 1)), int(cell.get('rowspan', 1))
            require(0 < width < 100 and 0 < height < 100, 'Invalid cell span')
            occupied[ri, ci] = list(range(col, col + width))
            for y in range(ri, ri + height):
                for x in range(col, col + width):
                    require((y, x) not in grid, 'Overlapping table cells')
                    grid[y, x] = (ri, ci, cell)
            col += width
    return rows, physical, grid, occupied


def _headers(grid, stop, column):
    texts, seen = [], set()
    for ri in range(stop):
        entry = grid.get((ri, column))
        if entry is not None and entry[:2] not in seen:
            seen.add(entry[:2])
            value = node_text(entry[2])
            if value:
                texts.append(value)
    return texts


def _header_year(texts, year, month_day):
    years = [int(v) for value in texts for v in re.findall(r'(?<!\d)(?:19|20)\d{2}(?!\d)', value)]
    # Only pure-year cells identify a column; broad title cells may name both years.
    exact = [int(value) for value in texts if re.fullmatch(r'(?:19|20)\d{2}', value)]
    heading = ' '.join(texts)
    annual = re.search(r'\b(?:twelve months|years?)\s+ended\b', heading, re.I)
    return bool(annual and re.search(month_day, heading, re.I) and
                ((exact and set(exact) == {year}) or (not exact and set(years) == {year})))


def statement_context(table, header_rows):
    """Own caption/header plus adjacent headings, never another table's text."""
    inside = [node_text(row) for row in header_rows]
    caption = table.find('caption', recursive=False)
    if caption is not None:
        inside.insert(0, node_text(caption))
    anchor = table
    parent = table.parent
    if parent.name == 'div' and 'c-table' in parent.get('class', []):
        anchor = parent
    outside = []
    for node in anchor.previous_siblings:
        if not getattr(node, 'name', None):
            if normalize(node): break
            continue
        if node.name == 'br': continue
        if node.name not in ('p', 'h1', 'h2', 'h3', 'h4') or node.find('table'):
            break
        value = node_text(node)
        if not value: continue
        if len(value) > 250: break
        outside.append(value)
        if len(outside) == 6: break
    # When a caption owns the formal title, unrelated preceding prose is not
    # statement evidence. This also prevents a nearby reconciliation caption
    # from contaminating a separate, fully captioned GAAP statement.
    formal = r'INCOME STATEMENTS|CONSOLIDATED STATEMENTS OF (?:INCOME|OPERATIONS|EARNINGS)'
    if any(re.search(formal, value, re.I) for value in inside):
        outside = []
    return inside + list(reversed(outside))


def parse_html_statement(raw, issuer, fiscal_year):
    """Independently select one whole annual consolidated table and four cells."""
    require(issuer in ('MSFT', 'ABBV'), 'Unsupported HTML issuer')
    soup = BeautifulSoup(raw.decode('utf-8-sig'), 'html.parser')
    body = node_text(soup)
    require(('Microsoft' if issuer == 'MSFT' else 'AbbVie') in body, 'Issuer identity absent')
    revenue_names = ('revenue', 'revenues', 'total revenue', 'total revenues') if issuer == 'MSFT' else (('net sales',) if fiscal_year <= 2014 else ('net revenues',))
    candidates = []
    for ti, table in enumerate(soup.find_all('table')):
        rows, physical, grid, occupied = _rows_and_grid(table)
        found = {'revenue': [], 'operating_income': []}
        for ri, cells in enumerate(physical):
            for ci, cell in enumerate(cells):
                text = node_text(cell)
                if any(node_text(c) for c in cells[:ci]):
                    continue
                if text.lower() in revenue_names:
                    found['revenue'].append((ri, ci))
                expected_oi = r'Operating income(?:\s*\(loss\))?' if issuer == 'MSFT' else r'Operating earnings(?:\s*\(loss\))?'
                if re.fullmatch(expected_oi, text, re.I):
                    found['operating_income'].append((ri, ci))
        if issuer == 'MSFT':
            totals = [(ri, ci) for ri, ci in found['revenue'] if node_text(physical[ri][ci]).lower().startswith('total ')]
            if totals:
                found['revenue'] = totals
        if not found['revenue'] or not found['operating_income']:
            continue
        first = min(ri for ri, _ in found['revenue'])
        head = ' '.join(node_text(row) for row in rows[:first])
        context_parts = statement_context(table, rows[:first])
        context = ' '.join(context_parts)
        full = node_text(table)
        if issuer == 'ABBV':
            if not re.search(r'AbbVie Inc\.?\s+Consolidated Statements of (?:Earnings|Income)', context, re.I):
                continue
            if 'cost of products sold' not in full.lower():
                continue
        else:
            if not re.search(r'cost of (?:revenue|goods)', full, re.I) or 'diluted' not in full.lower():
                continue
            if not re.search(r'INCOME STATEMENTS|CONSOLIDATED STATEMENTS OF (?:INCOME|OPERATIONS)', context, re.I):
                continue
        if re.search(r'non[ -]?gaap|reconciliation|pro forma|adjusted', context, re.I):
            continue
        require(len(found['revenue']) == len(found['operating_income']) == 1, 'Nonunique statement row')
        require(found['revenue'][0][0] < found['operating_income'][0][0], 'Operating income precedes revenue')
        currency_literal = '$' if '$' in node_text(rows[found['revenue'][0][0]]) else None
        unit_matches = [value for value in context_parts if re.search(r'\bin millions\b', value, re.I)]
        require(unit_matches and set(re.findall(r'\bin\s+(millions|thousands|billions)\b', context.lower())) == {'millions'},
                'Unique statement millions denomination required')
        month_day = r'June\s+30' if issuer == 'MSFT' else r'(?:December|Dec\.)\s+31'
        dates = {role: (f'{year}-07-01', f'{year + 1}-06-30') if issuer == 'MSFT'
                 else (f'{year + 1}-01-01', f'{year + 1}-12-31')
                 for role, year in (('current', fiscal_year - 1), ('comparative', fiscal_year - 2))}
        result = {}
        for measure, [(ri, label_ci)] in found.items():
            result[measure] = {}
            for role, year in (('current', fiscal_year), ('comparative', fiscal_year - 1)):
                cells = []
                for ci, cell in enumerate(physical[ri]):
                    cols = occupied[ri, ci]
                    match = [_header_year(_headers(grid, first, col), year, month_day) for col in cols]
                    if not any(match):
                        continue
                    require(all(match), 'Cell straddles ambiguous annual columns')
                    literal = node_text(cell)
                    if not literal or literal == '$':
                        continue
                    cells.append((ci, cell, cols, literal))
                require(len(cells) == 1, 'Unique annual monetary cell required')
                ci, cell, cols, literal = cells[0]
                require(int(cell.get('rowspan', 1)) == 1, 'Monetary cell spans multiple statement rows')
                value = amount(literal)
                if measure == 'revenue':
                    require(Decimal(value) > 0, 'Nonpositive revenue')
                start, end = dates[role]
                result[measure][role] = dict(amount_decimal=value, displayed_amount=literal,
                    scale=1000000, period_start=start, period_end=end,
                    label=node_text(physical[ri][label_ci]),
                    locator=dict(table_index_zero_based=ti, row_index_zero_based=ri,
                                 physical_cell_index_zero_based=ci, logical_columns_zero_based=cols),
                    cell_html=str(cell), cell_html_sha256=digest(str(cell).encode()),
                    row_html=str(rows[ri]), row_html_sha256=digest(str(rows[ri]).encode()),
                    header_literals=_headers(grid, first, cols[0]))
        candidates.append(dict(issuer=issuer, fiscal_year=fiscal_year, cells=result,
            table_index_zero_based=ti, table_html_sha256=digest(str(table).encode()),
            unit_literal=unit_matches[0], scale=1000000, source_sha256=digest(raw),
            currency_literal=currency_literal, currency_iso=None, currency_iso_certified=False))
    require(len(candidates) == 1, 'Unique consolidated annual statement required')
    return candidates[0]


def _letters(value):
    return r'\s*'.join(re.escape(c) for c in value.replace(' ', ''))


def _dates(text):
    return [datetime.strptime(' '.join(hit), '%B %d %Y').date().isoformat()
            for hit in PERIOD_DATE.findall(text)]


def _four_amounts(text):
    result, rest = [], text.lstrip()
    for _ in range(4):
        hit = MONEY.match(rest)
        require(hit is not None, 'Four monetary columns required')
        literal = normalize(hit.group())
        result.append(dict(displayed_amount=literal, amount_decimal=amount(literal)))
        rest = rest[hit.end():].lstrip()
    require(not re.match(r'[$(0-9−]', rest), 'Extra monetary column or ambiguous row boundary')
    return result


def parse_aapl_statements(texts, fiscal_year):
    """Read content-order text; no production geometry/line selection is reused."""
    pages = [normalize(text or '').replace('/dollarsign', '$') for text in texts]
    title = 'CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS'
    selected = [(i, text) for i, text in enumerate(pages) if title in text]
    require(len(selected) == 1, 'Unique operations page required')
    page, text = selected[0]
    require(text.count(title) == 1 and 'Apple Inc.' in text[:text.index(title)] and
            'unaudited' in text.lower(), 'Apple statement identity or audit status missing')
    require(not re.search(r'non[ -]?gaap|reconciliation', text, re.I), 'Adjusted operations page')
    first = re.search(_letters('Net sales'), text, re.I)
    cost = re.search(_letters('Cost of sales'), text, re.I)
    require(first is not None and cost is not None and first.start() < cost.start(), 'Formal revenue boundary missing')
    header = text[:first.start()]
    require(header.count('Three Months Ended') == header.count('Twelve Months Ended') == 1 and
            header.index('Three Months Ended') < header.index('Twelve Months Ended'), 'Quarter/annual header order')
    dates = _dates(header)
    require(len(dates) == 4 and dates[:2] == dates[2:] and date.fromisoformat(dates[2]).year == fiscal_year and
            date.fromisoformat(dates[3]).year == fiscal_year - 1, 'Four annual/quarter endpoint identities')
    units = re.findall(r'\(In millions,[^)]+\)', header, re.I)
    require(len(units) == 1, 'Unique millions denomination')
    revenue_scope = text[first.start():cost.start()]
    total = list(re.finditer(_letters('Total net sales'), revenue_scope, re.I))
    require(len(total) <= 1, 'Duplicate formal total revenue')
    revenue_tail = revenue_scope[total[0].end():] if total else revenue_scope[first.end()-first.start():]
    if re.match(r'\s*\(1\)', revenue_tail):
        require(total and re.search(r'\(1\)\s*Net sales by', text, re.I), 'Unresolved revenue footnote')
        revenue_tail = re.sub(r'^\s*\(1\)', '', revenue_tail)
    revenue = _four_amounts(revenue_tail)
    oi_hits = list(re.finditer(_letters('Operating income'), text, re.I))
    require(len(oi_hits) == 1 and oi_hits[0].start() > cost.start(), 'Unique formal operating income row')
    require(text[:oi_hits[0].start()].rstrip()[-1] in '0123456789)', 'Operating-income label has an unsupported prefix')
    net = re.search(_letters('Net income'), text[oi_hits[0].end():], re.I)
    require(net is not None, 'Operating income must precede net income')
    operating = _four_amounts(text[oi_hits[0].end():])
    balance = [(i, value) for i, value in enumerate(pages) if 'CONSOLIDATED BALANCE SHEETS' in value]
    require(len(balance) == 1 and 'ASSETS:' in balance[0][1], 'Unique balance sheet endpoint evidence')
    end_dates = _dates(balance[0][1].split('ASSETS:', 1)[0])
    require(end_dates == dates[2:], 'Operations/balance annual endpoints differ')
    end, prior = map(date.fromisoformat, end_dates)
    duration = (end - prior).days
    require(duration in (364, 371) and end.weekday() == prior.weekday() == 5 and
            all(x.month == 9 and 24 <= x.day <= 30 for x in (end, prior)), 'Invalid 52/53-week Apple endpoints')
    fields = {}
    for measure, row in (('revenue', revenue), ('operating_income', operating)):
        fields[measure] = {}
        for role, column in (('current', 2), ('comparative', 3)):
            cell = dict(row[column], scale=1000000, period_end=dates[column],
                period_start=(prior + timedelta(days=1)).isoformat() if role == 'current' else None,
                header_dates=dates, all_amount_decimals=[v['amount_decimal'] for v in row],
                locator=dict(page_1based=page + 1, financial_column_index_0based=column))
            if measure == 'revenue': require(Decimal(cell['amount_decimal']) > 0, 'Nonpositive revenue')
            fields[measure][role] = cell
    if not all('$' in v['displayed_amount'] for v in revenue):
        products = re.search(_letters('Products') + r'\s+', revenue_scope, re.I)
        require(products is not None and all('$' in v['displayed_amount'] for v in _four_amounts(revenue_scope[products.end():])), 'Four sales-block dollar glyphs required; ISO remains uncertified')
    return dict(fiscal_year=fiscal_year, cells=fields, page_1based=page + 1,
        header_dates=dates, unit_literal=units[0], scale=1000000,
        current_period=dict(start=(prior + timedelta(days=1)).isoformat(), end=end.isoformat(), days_inclusive=duration),
        comparative_period=dict(start=None, end=prior.isoformat()),
        currency_literal='$', currency_iso=None, currency_iso_certified=False)


def publication_date(raw, issuer):
    soup = BeautifulSoup(raw.decode('utf-8-sig'), 'html.parser')
    if issuer in ('AAPL', 'ABBV'):
        nodes = soup.select('.category-eyebrow__date' if issuer == 'AAPL' else '.wd_date')
        require(len(nodes) == 1, 'Unique visible publication date')
        value = datetime.strptime(node_text(nodes[0]), '%B %d, %Y').date().isoformat()
        if issuer == 'AAPL':
            structured = set(v[:10] for v in re.findall(r'"datePublished"\s*:\s*"([^"]+)"', raw.decode('utf-8-sig')))
            require(structured == {value}, 'Visible/structured release dates disagree')
        return value
    require(issuer == 'MSFT', 'Unsupported publication issuer')
    found = set()
    for node in soup.find_all(['p', 'li', 'h1', 'h2', 'h3']):
        text = node_text(node)
        if re.search(r'REDMOND,\s*Wash', text, re.I):
            hit = re.search(r'\bJuly\s+(\d{1,2}),?\s+(20\d{2})\b', text)
            if hit: found.add(f'{hit.group(2)}-07-{int(hit.group(1)):02d}')
    require(len(found) == 1, 'Unique Microsoft original dateline')
    return found.pop()


def next_session(release, sessions):
    require(isinstance(release, str) and date.fromisoformat(release).isoformat() == release, 'ISO release date required')
    require(sessions and list(sessions) == sorted(set(sessions)), 'Unique ordered calendar required')
    require(all(date.fromisoformat(day).isoformat() == day for day in sessions) and sessions[0] <= release < sessions[-1], 'Release outside calendar or malformed session')
    later = [session for session in sessions if session > release]
    require(bool(later), 'Calendar has no session after publication')
    return later[0]


def pdf_content_pages(raw):
    import operating_margin172_common as common
    common.verify_pdf_runtime()
    require(raw.startswith(b'%PDF'), 'Expected original PDF bytes')
    bundle = str(common.PDF_BUNDLE)
    if bundle not in sys.path:
        sys.path.append(bundle)
    from pypdf import PdfReader
    return [page.extract_text() for page in PdfReader(io.BytesIO(raw)).pages]


def verify_scope(records, expected):
    require(isinstance(records, list), 'Records must be explicit list')
    keys = [(r.get('issuer', r.get('symbol')), r['fiscal_year']) for r in records]
    require(len(keys) == len(set(keys)) and set(keys) == set(expected), 'Missing, duplicate, or unexpected source report')
    return True


def verify_receipt(raw, receipt, url, issuer, pdf=False):
    """Check independently the stored HTTP provenance, without granting PIT status."""
    host = {'MSFT': 'www.microsoft.com', 'ABBV': 'news.abbvie.com', 'AAPL': 'www.apple.com'}[issuer]
    require(urlparse(url).scheme == 'https' and urlparse(url).hostname == host, 'Unexpected original issuer URL')
    require(receipt.get('status', receipt.get('http_status')) == 200, 'Original response was not HTTP200')
    require(receipt.get('url', receipt.get('source_url')) == url, 'Receipt requested URL differs')
    require(receipt.get('final_url', url) == url, 'Receipt redirected source')
    require(receipt.get('response_sha256', receipt.get('sha256')) == digest(raw), 'Receipt raw digest differs')
    size = receipt.get('response_bytes', receipt.get('bytes', receipt.get('size')))
    require(size is None or size == len(raw), 'Receipt byte count differs')
    require(not receipt.get('error'), 'Receipt records an error')
    stamp = datetime.fromisoformat(receipt['retrieved_at'].replace('Z', '+00:00'))
    require(stamp.utcoffset() is not None, 'Receipt retrieval timezone missing')
    if issuer == 'AAPL':
        require(receipt.get('public_unauthenticated') is True, 'Public source provenance missing')
        require(receipt.get('content_type', '').split(';')[0].lower() == ('application/pdf' if pdf else 'text/html'), 'Wrong original MIME')
    if pdf:
        require(raw.startswith(b'%PDF-') and receipt.get('decoder') == 'identity' and
                receipt.get('decoded_sha256') == digest(raw), 'PDF bytes were transformed')
    return stamp


def compare_html_record(parsed, produced, release, known_by):
    require(produced['status'] == 'extracted' and produced['issuer'] == parsed['issuer'] and
            produced['fiscal_year'] == parsed['fiscal_year'], 'HTML extraction identity/status differs')
    require(produced['release_date'] == release and produced['known_by'] == known_by, 'HTML release availability differs')
    statement = produced['statement']
    require(statement['table_index_zero_based'] == parsed['table_index_zero_based'] and
            statement['table_html_sha256'] == parsed['table_html_sha256'] and
            Decimal(str(statement['scale'])) == 1000000, 'HTML statement identity/scale differs')
    require(set(produced['cells']) == {'revenue', 'operating_income'}, 'Exactly two measures required')
    for measure, roles in parsed['cells'].items():
        require(set(produced['cells'][measure]) == {'current', 'comparative'}, 'Exactly two annual vintages required')
        for role, expected in roles.items():
            cell = produced['cells'][measure][role]
            require(Decimal(cell['amount_decimal']) == Decimal(expected['amount_decimal']) and
                    amount(cell['displayed_amount']) == expected['amount_decimal'], 'HTML original amount differs')
            require(Decimal(str(cell['scale'])) == 1000000 and
                    re.search(r'in millions', cell['unit_literal'], re.I), 'HTML unit differs')
            for key in ('period_start', 'period_end', 'cell_html', 'cell_html_sha256', 'row_html', 'row_html_sha256'):
                require(cell[key] == expected[key], 'HTML original evidence differs: ' + key)
            require(cell['release_date'] == release and cell['known_by'] == known_by and
                    cell['source_raw_sha256'] == parsed['source_sha256'], 'HTML cell current-release source clock differs')
            loc = dict(cell['locator'])
            loc['logical_columns_zero_based'] = loc.pop('logical_columns')
            require(loc == expected['locator'], 'HTML physical/logical column locator differs')
            require(cell['currency_literal'] == parsed['currency_literal'] and cell.get('currency_iso') is None and
                    cell.get('currency_iso_newly_certified') is False and cell.get('fact_admitted') is False,
                    'HTML source-only currency/admission limit lost')
            period = produced['periods'][role]
            require(period['start'] == expected['period_start'] and period['end'] == expected['period_end'] and
                    period['days_inclusive'] == (date.fromisoformat(period['end']) - date.fromisoformat(period['start'])).days + 1,
                    'HTML annual period differs')
    return True


def compare_aapl_record(parsed, produced, release, known_session, earlier=None):
    require(produced['symbol'] == 'AAPL' and produced['fiscal_year'] == parsed['fiscal_year'] and
            produced['status'] == 'EXTRACTED_SOURCE_CANDIDATE', 'Apple extraction identity/status differs')
    require(produced['publication']['release_date'] == release and produced['known_at_session'] == known_session and
            produced['availability']['candidate_first_usable_session'] == known_session, 'Apple publication/availability differs')
    require(set(produced['fields']) == {'revenue', 'operating_income'}, 'Exactly two Apple measures required')
    comparative_start = None
    if earlier is not None:
        require(earlier['fiscal_year'] == parsed['fiscal_year'] - 1 and earlier['release_date'] < release and
                earlier['current_period']['end'] == parsed['comparative_period']['end'], 'Prior original boundary source differs')
        comparative_start = earlier['current_period']['start']
    require(produced['statement']['header_dates'] == parsed['header_dates'] and
            produced['statement']['page_1based'] == parsed['page_1based'], 'Apple statement/header identity differs')
    for measure, roles in parsed['cells'].items():
        require(set(produced['fields'][measure]) == {'current', 'comparative'}, 'Exactly two Apple annual vintages required')
        for role, expected in roles.items():
            cell = produced['fields'][measure][role]
            require(Decimal(cell['amount_decimal']) == Decimal(expected['amount_decimal']) and
                    amount(cell['displayed_amount']) == expected['amount_decimal'] and
                    [amount(v) for v in cell['all_row_cells']] == expected['all_amount_decimals'], 'Apple original row/amount differs')
            require(cell['scale'] == 1000000 and re.search(r'\(In millions,', cell['unit_literal']), 'Apple denomination differs')
            for key, value in expected['locator'].items():
                require(cell['locator'][key] == value, 'Apple annual physical column differs')
            require(cell['date'] == expected['period_end'] and cell['header_dates'] == parsed['header_dates'], 'Apple annual period end differs')
            start = expected['period_start'] if role == 'current' else comparative_start
            period = cell['period']
            require(period['start'] == start and period['end'] == expected['period_end'] and
                    period['duration_literal'] == 'Twelve Months Ended', 'Apple annual boundary differs')
            if start is not None:
                days = (date.fromisoformat(period['end']) - date.fromisoformat(start)).days + 1
                require(days in (364, 371) and period['days_inclusive'] == days, 'Apple annual duration differs')
            require(cell['release_date'] == release and cell['known_at_session'] == known_session and
                    cell['vintage_role'] == ('own_year_original' if role == 'current' else 'same_current_release_prior_year_comparative'), 'Apple comparative backdating')
            require(cell['iso_currency'] is None and cell['ISO_currency_verified'] is False and
                    cell['fact_admitted'] is False and cell['strategy_admitted'] is False, 'Apple source-only limitations lost')
            require(digest(cell['row_text'].encode()) == cell['row_text_sha256'], 'Apple row text digest differs')
    if comparative_start is None:
        require('comparative_period_start_unknown' in produced['holds'], 'Unknown comparative period must be held')
    return True


def _char_evidence(textpage, index, count):
    require(type(index) is int and type(count) is int and index >= 0 and count > 0, 'Invalid PDF character span')
    coords = [textpage.get_charbox(j) for j in range(index, index + count)]
    return dict(text=textpage.get_text_range(index, count), bbox=[min(b[0] for b in coords), min(b[1] for b in coords),
        max(b[2] for b in coords), max(b[3] for b in coords)])


def _pdf_hits(textpage, term):
    search, found = textpage.search(term, match_case=False), []
    try:
        while True:
            hit = search.get_next()
            if hit is None:
                return found
            found.append(_char_evidence(textpage, *hit))
    finally:
        search.close()


def verify_pdf_geometry(textpage, parsed, produced):
    """Re-read source characters and positions; supplied PDFium hashes are not trusted."""
    mid = lambda b: (b['bbox'][1] + b['bbox'][3]) / 2
    titles = _pdf_hits(textpage, 'CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS')
    costs, nets = _pdf_hits(textpage, 'Cost of sales'), _pdf_hits(textpage, 'Net income')
    require(len(titles) == 1 and costs and nets, 'PDF statement geometry missing')
    title_y, cost_y = mid(titles[0]), max(map(mid, costs))
    net_y = max(mid(b) for b in nets if mid(b) < cost_y)
    for measure, roles in parsed['cells'].items():
        cell = produced['fields'][measure]['current']
        hits = [b for b in _pdf_hits(textpage, cell['measure_label'])
                if (cost_y < mid(b) < title_y if measure == 'revenue' else net_y < mid(b) < cost_y)]
        require(len(hits) == 1, 'PDF formal row label geometry ambiguous')
        label = hits[0]
        raw_cells = cell['pdfium_all_row_cells']
        require(len(raw_cells) == 4, 'Four PDF monetary columns required')
        boxes = []
        for i, item in enumerate(raw_cells):
            actual = _char_evidence(textpage, item['index'], item['count'])
            require(actual['bbox'] == item['bbox'] and actual['text'] == item['text'], 'PDF raw geometry evidence differs')
            require(amount(actual['text']) == roles['current']['all_amount_decimals'][i] and
                    abs(mid(actual) - mid(label)) < 4 and actual['bbox'][0] > label['bbox'][2], 'PDF numeric source is outside the formal row')
            boxes.append(actual['bbox'])
        require(all(boxes[i][2] < boxes[i + 1][0] for i in range(3)), 'PDF columns overlap or are reordered')
        for role, col in (('current', 2), ('comparative', 3)):
            loc = produced['fields'][measure][role]['locator']
            require(loc['pdfium_char_index'] == raw_cells[col]['index'] and loc['pdfium_char_count'] == raw_cells[col]['count'] and
                    loc['pdfium_cell_bbox'] == boxes[col] and loc['pdfium_label_bbox'] == label['bbox'], 'PDF annual locator differs')
    return True


def bound_read(ref, pins):
    import operating_margin172_common as common
    path = common.source_path(ref['path'])
    require(pins.get(str(path)) == ref['sha256'], 'Source reference is absent from frozen PLAN')
    raw = path.read_bytes()
    require(digest(raw) == ref['sha256'] and ('bytes' not in ref or len(raw) == ref['bytes']), 'Source digest/size changed')
    return raw


def calendar_check(plan):
    import importlib.util
    import operating_margin172_common as common
    saved = common.read(plan['calendar'])
    module_path = common.ROOT / 'outputs/monthly_earnings_repair/research_calendar.py'
    require(plan['input_sha256'].get(str(module_path)) == common.sha(module_path), 'Calendar implementation not pinned')
    spec = importlib.util.spec_from_file_location('_operating172_independent_calendar', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calendar = module.market_calendar(saved['start'], saved['end'])
    sessions = [v.date().isoformat() for v in calendar.sessions]
    require(sessions == saved['sessions'], 'Calendar sessions do not reproduce from pinned implementation')
    return sessions, calendar


def run(plan_sha, html_sha, aapl_sha):
    import argparse
    import json
    from datetime import timezone
    import operating_margin172_v3_common as common
    import operating_margin172_common as original_common
    plan = common.gate(plan_sha)
    hp, ap = original_common.OUT / 'v2/html/EXTRACTION.json', original_common.OUT / 'aapl/REPORTS.json'
    require(common.sha(hp) == html_sha and common.sha(ap) == aapl_sha, 'Explicit completed extraction SHA required')
    html, aapl = common.read(hp), common.read(ap)
    require(html['plan_sha256'] == plan['source_v2_plan_sha256'] and aapl['plan_sha256'] == plan['original_plan_sha256'],
            'Extraction uses another original/v2 source PLAN')
    expected = [(s, y) for s, years in plan['contract']['scope'].items() for y in years]
    verify_scope(html['reports'] + aapl['records'], expected)
    sessions, calendar = calendar_check(plan)
    source_html = common.read(plan['source_manifests']['html'])
    source_aapl = common.read(plan['source_manifests']['aapl'])['rows']
    html_by = {(r['issuer'], r['fiscal_year']): r for r in html['reports']}
    apple_by = {r['fiscal_year']: r for r in aapl['records']}
    records = []
    for issuer in ('MSFT', 'ABBV'):
        for original in source_html[issuer]:
            src = original['source']
            raw = bound_read(src['raw'], plan['input_sha256'])
            receipt = json.loads(bound_read(src['receipt'], plan['input_sha256']))
            url = src['expected_url'] if issuer == 'MSFT' else src['metadata']['source_url']
            retrieved = verify_receipt(raw, receipt, url, issuer)
            release = publication_date(raw, issuer)
            require(release == original['release_date'] and retrieved.date() >= date.fromisoformat(release), 'Original publication/retrieval date differs')
            session = next_session(release, sessions)
            known = calendar.session_close(session).isoformat()
            require(known == original['known_by'], 'Original source clock differs from independent calendar')
            parsed = parse_html_statement(raw, issuer, original['fiscal_year'])
            require(parsed['cells']['revenue']['current']['period_end'] < release, 'HTML release precedes fiscal end')
            produced = html_by[issuer, original['fiscal_year']]
            require(produced['source'] == original['source'] and produced['report_id'] == original['report_id'], 'HTML source identity mutated')
            compare_html_record(parsed, produced, release, known)
            records.append(dict(issuer=issuer, fiscal_year=original['fiscal_year'], passed=True, source_sha256=digest(raw),
                checked_annual_cells=4, table_index_zero_based=parsed['table_index_zero_based'],
                currency_literal=parsed['currency_literal'], currency_iso=None, currency_iso_certified=False,
                release_date=release, known_session=session, known_by=known, next_session_clock_rederived=True,
                inherited_holds=original.get('original_holds_preserved', []), source_fields_admitted=False))
    earlier = None
    for original in sorted(source_aapl, key=lambda r: r['fiscal_year']):
        raw = bound_read(original['source'], plan['input_sha256'])
        parent = bound_read(original['parent_release'], plan['input_sha256'])
        receipt = json.loads(bound_read(original['receipt'], plan['input_sha256']))
        parent_receipt = json.loads(bound_read(original['parent_receipt'], plan['input_sha256']))
        parent_url = parent_receipt.get('source_url', parent_receipt.get('url'))
        verify_receipt(raw, receipt, original['source_url'], 'AAPL', pdf=True)
        verify_receipt(parent, parent_receipt, parent_url, 'AAPL')
        release = publication_date(parent, 'AAPL')
        links = [urljoin(parent_url, a['href']) for a in BeautifulSoup(parent, 'html.parser').find_all('a', href=True)]
        require(original['source_url'] in links and release == original['publication']['release_date'], 'Parent release/PDF association differs')
        parsed = parse_aapl_statements(pdf_content_pages(raw), original['fiscal_year'])
        require(parsed['current_period']['end'] < release, 'Release precedes fiscal end')
        session = next_session(release, sessions)
        produced = apple_by[original['fiscal_year']]
        for key in ('source', 'source_url', 'receipt', 'parent_release', 'parent_receipt'):
            require(produced[key] == original[key], 'Apple source identity mutated')
        compare_aapl_record(parsed, produced, release, session, earlier)
        import pypdfium2 as pdfium
        document = pdfium.PdfDocument(raw)
        page = document[parsed['page_1based'] - 1]
        textpage = page.get_textpage()
        try:
            verify_pdf_geometry(textpage, parsed, produced)
        finally:
            textpage.close(); page.close(); document.close()
        records.append(dict(issuer='AAPL', fiscal_year=original['fiscal_year'], passed=True, source_sha256=digest(raw),
            checked_annual_cells=4, source_page=parsed['page_1based'], content_and_geometry_independently_checked=True,
            release_date=release, known_session=session, known_by=calendar.session_close(session).isoformat(),
            next_session_clock_rederived=True, comparative_start_known=earlier is not None,
            inherited_holds=original.get('holds', []), source_fields_admitted=False))
        earlier = dict(parsed, release_date=release)
    verify_scope(records, expected)
    common.gate(plan_sha)
    require(common.sha(hp) == html_sha and common.sha(ap) == aapl_sha, 'Extraction changed during verification')
    out = common.OUT / 'independent'
    require(not out.exists(), 'Independent source review is write-once')
    out.mkdir()
    result = dict(schema='operating_margin172_independent_v3', plan_sha256=plan_sha,
        original_plan_sha256=plan['original_plan_sha256'],
        source_v2_plan_sha256=plan['source_v2_plan_sha256'],
        generated_utc=datetime.now(timezone.utc).isoformat(), passed=True, source_reports=35,
        checked_annual_cells=140, reports=records,
        producer_outputs={str(hp): html_sha, str(ap): aapl_sha},
        source_fields_admitted=False, financial_ratios_factors_orders_returns=0, new_strategy_trials=0,
        independent_of_production_extractors=True,
        limits=['ISO currency and historical first-public bytes remain unverified.',
                'Missing revenue-row dollar glyphs remain currency_literal=None; no ISO is inferred from issuer nationality.',
                'Same-report annual cells checked; full accounting continuity/amendment history not certified.',
                'AAPL FY2016 comparative start remains unknown; no synthetic boundary or prior original created.',
                'All facts retain current-release next-session availability; no comparative backdating.',
                'This review does not authorize a factor, strategy trial, or live observer change.'])
    common.dump(out / 'REVIEW.json', result)
    common.dump(out / 'COMPLETION.json', dict(passed=True, review_sha256=common.sha(out / 'REVIEW.json'),
        plan_sha256=plan_sha, reports=35, cells=140, source_fields_admitted=False))
    print(json.dumps(dict(passed=True, reports=35, cells=140, review_sha256=common.sha(out / 'REVIEW.json'))))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--html-sha256', required=True)
    parser.add_argument('--aapl-sha256', required=True)
    args = parser.parse_args()
    run(args.plan_sha256, args.html_sha256, args.aapl_sha256)
