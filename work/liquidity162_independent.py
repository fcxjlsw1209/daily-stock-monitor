"""Read-only, independent HTML audit for the liquidity162 source pilot.

No issuer extractor or historical monetary output is imported. ``verify`` reads
only the plan's pinned local files; it never saves output or computes a ratio.
The command-line entry point runs synthetic tests only.
"""
from copy import deepcopy
from datetime import date, datetime, time, timedelta
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
import unittest
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ('liquid_financial_assets', 'short_term_debt',
          'current_portion_long_term_debt', 'long_term_debt', 'total_assets')
SECTIONS = dict(zip(FIELDS, ('current_assets', 'current_liabilities',
                           'current_liabilities', 'noncurrent_liabilities', 'all_assets')))
MONTHS = ('January February March April May June July August September October '
          'November December').split()


def need(condition, message):
    if not condition:
        raise ValueError(message)


def norm(value):
    return re.sub(r'\s+', ' ', str(value).replace('\xa0', ' ')).strip()


def canonical(value):
    value = norm(value).casefold().replace('–', '-').replace('—', '-')
    return re.sub(r'[^a-z0-9]+', ' ', value).strip()


def decimal_token(value):
    """A dash, empty cell, nonfinite token or malformed grouping is not zero."""
    token = norm(value).replace('$', '').strip()
    negative = token.startswith('(') and token.endswith(')')
    if negative:
        token = token[1:-1].strip()
    need(bool(re.fullmatch(r'-?(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.[0-9]+)?', token)),
         'unavailable_or_malformed_amount')
    need(not (negative and token.startswith('-')), 'double_negative_amount')
    result = Decimal(token.replace(',', ''))
    return -result if negative else result


def iso(value):
    need(type(value) is str, 'date_string_required')
    parsed = date.fromisoformat(value)
    need(parsed.isoformat() == value, 'canonical_date_required')
    return parsed


def dates(text):
    """Parse dates only from the independent header/publication evidence."""
    result = set()
    for match in re.finditer(r'\b(20\d{2}|19\d{2})-(\d{2})-(\d{2})\b', text):
        try:
            result.add(date(*map(int, match.groups())).isoformat())
        except ValueError:
            pass
    names = '|'.join(m + '|' + m[:3] + r'\.?' for m in MONTHS)
    for match in re.finditer(r'\b(' + names + r')\s+(\d{1,2}),?\s+(\d{4})\b', text, re.I):
        month = next(i + 1 for i, name in enumerate(MONTHS)
                     if name[:3].casefold() == match.group(1)[:3].casefold())
        try:
            result.add(date(int(match.group(3)), month, int(match.group(2))).isoformat())
        except ValueError:
            pass
    return result


def table_grid(table):
    """Independent physical cells and span-aware header alignment."""
    rows = [r for r in table.find_all('tr') if r.find_parent('table') is table]
    physical, grid = [], {}
    for ri, row in enumerate(rows):
        cells, col = [], 0
        for pi, node in enumerate(row.find_all(['td', 'th'], recursive=False)):
            while (ri, col) in grid:
                col += 1
            width, height = int(node.get('colspan', 1)), int(node.get('rowspan', 1))
            need(1 <= width <= 100 and 1 <= height <= 100, 'invalid_html_span')
            cell = dict(text=norm(node.get_text(' ', strip=True)), row=ri,
                        physical=pi, columns=list(range(col, col + width)))
            cells.append(cell)
            for r in range(ri, ri + height):
                for c in cell['columns']:
                    need((r, c) not in grid, 'overlapping_html_span')
                    grid[(r, c)] = cell
            col += width
        physical.append(cells)
    return physical, grid


def context(table):
    chunks = []
    caption = table.find('caption', recursive=False)
    if caption:
        chunks.append(norm(caption.get_text(' ', strip=True)))
    node = table.parent if table.parent and 'c-table' in table.parent.get('class', []) else table
    preceding = []
    for sibling in node.previous_siblings:
        if getattr(sibling, 'name', None) in ('table', 'div'):
            break
        if getattr(sibling, 'find', None) and not isinstance(sibling, str) and sibling.find('table'):
            break
        text = norm(sibling.get_text(' ', strip=True)) if hasattr(sibling, 'get_text') else norm(sibling)
        if text:
            preceding.append(text)
        if len(preceding) == 5:
            break
    return norm(' '.join(list(reversed(preceding)) + chunks))


def label_of(cells):
    return next((c['text'] for c in cells if re.search(r'[A-Za-z]', c['text'])), '')


def matches(field, label, ticker):
    label = canonical(label)
    choices = {
        'liquid_financial_assets': ({'total cash cash equivalents and short term investments'}
                                   if ticker == 'MSFT' else {'cash cash equivalents and marketable securities'}),
        'short_term_debt': {'short term debt'} if ticker == 'MSFT' else {'short term debt', 'convertible short term debt'},
        'current_portion_long_term_debt': {'current portion of long term debt', 'current maturities of long term debt'},
        'long_term_debt': {'long term debt'}, 'total_assets': {'total assets'},
    }
    return label in choices[field]


def publication(soup, source):
    """Use visible release lead/metadata, never a date inside financial tables."""
    clone = BeautifulSoup(str(soup), 'html.parser')
    for node in clone.find_all(['table', 'script', 'style']):
        node.decompose()
    lead = norm(clone.get_text(' ', strip=True))
    issuer = 'Microsoft' if source['ticker'] == 'MSFT' else 'NVIDIA'
    need(re.search(r'\b' + issuer + r'\b', lead, re.I), 'issuer_not_in_release')
    need(re.search(r'\b' + source['ticker'] + r'\b', lead), 'ticker_not_in_release')
    if source['ticker'] == 'MSFT':
        identity = re.search(r'Earnings Release FY(\d{2}) Q([1-4])', lead, re.I)
        need(identity and int(identity[1]) + 2000 == source['fiscal_year'] and identity[2] == '4',
             'release_fiscal_identity_mismatch')
    else:
        titles = soup.select('.article-title')
        title = norm(titles[0].get_text(' ', strip=True)) if len(titles) == 1 else lead[:12000]
        identity = re.search(r'fourth\s+quarter\s+(?:and\s+)?fiscal\s+(\d{4})', title, re.I)
        need(identity and int(identity[1]) == source['fiscal_year'], 'release_fiscal_identity_mismatch')
    ended = [m.group(1) for m in re.finditer(r'\bended\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})', lead, re.I)]
    need(any(source['fiscal_period_end'] in dates(text) for text in ended), 'release_period_end_not_evidenced')
    candidates = []
    # A dated release dateline is stronger than an arbitrary date in navigation.
    for match in re.finditer(r'(?:REDMOND|SANTA CLARA)[\s\S]{0,250}', lead, re.I):
        candidates.append(match.group())
    for node in soup.find_all(['meta', 'time']):
        keys = ' '.join(str(node.get(k, '')) for k in ('name', 'property', 'itemprop')).lower()
        if node.name == 'time' or any(k in keys for k in ('datepublished', 'published_time', 'publication_date')):
            candidates.append(str(node.get('content') or node.get('datetime') or node.get_text(' ', strip=True)))
    # NVIDIA's release pages place the publication date immediately above their
    # company release heading; scan only the lead before its first dateline.
    if source['ticker'] == 'NVDA':
        article_dates = soup.select('.article-date')
        if article_dates:
            need(len(article_dates) == 1, 'ambiguous_release_date_node')
            candidates.append(norm(article_dates[0].get_text(' ', strip=True)))
        lead_end = re.search(r'\bSANTA CLARA\b', lead, re.I)
        candidates.append(lead[:lead_end.start() if lead_end else min(len(lead), 12000)])
    expected = source['release_date']
    evidence = [text for text in candidates if expected in dates(text)]
    need(evidence, 'release_date_not_in_publication_evidence')
    need(iso(source['fiscal_period_end']) < iso(expected), 'release_precedes_period_end')
    need(source['precision'] == 'date_only' and source['publication_timezone'] == 'America/Los_Angeles',
         'unsupported_publication_precision')
    host = urlparse(source['source_url']).hostname
    allowed = {'www.microsoft.com', 'microsoft.com'} if source['ticker'] == 'MSFT' else {'nvidianews.nvidia.com'}
    need(host in allowed, 'nonissuer_source_url')
    return dict(release_date=expected, evidence=evidence, issuer=issuer,
                exact_publication_clock_verified=False, original_vintage_verified=False)


def inspect_html(raw, source):
    """Reconstruct source evidence without consulting the proposed record."""
    soup = BeautifulSoup(raw, 'html.parser')
    pub = publication(soup, source)
    candidates = []
    for ti, table in enumerate(soup.find_all('table')):
        if table.find('table'):
            continue
        rows, grid = table_grid(table)
        labels = [canonical(label_of(row)) for row in rows]
        if 'total assets' not in labels or 'total current liabilities' not in labels:
            continue
        stops = [i for i, label in enumerate(labels) if label in ('assets', 'current assets')]
        need(stops, 'assets_section_not_explicit')
        stop = min(stops)
        head = context(table) + ' ' + ' '.join(c['text'] for row in rows[:stop] for c in row)
        if not re.search(r'\bbalance sheets?\b', head, re.I):
            continue
        need(not re.search(r'non.gaap|cash flows?|as adjusted', head, re.I), 'wrong_statement_basis')
        candidates.append((ti, table, rows, grid, labels, stop, norm(head)))
    need(len(candidates) == 1, 'balance_sheet_not_unique')
    ti, table, rows, grid, labels, stop, head = candidates[0]
    units = {m.casefold() for m in re.findall(r'\bin\s+(thousands|millions|billions)\b', head, re.I)}
    need(len(units) == 1, 'statement_unit_missing_or_ambiguous')
    scale = {'thousands': 1000, 'millions': 1000000, 'billions': 1000000000}[next(iter(units))]
    table_text = norm(table.get_text(' ', strip=True))
    need({m.casefold() for m in re.findall(r'\bin\s+(thousands|millions|billions)\b', head + ' ' + table_text, re.I)} == units,
         'conflicting_statement_units')
    need(not re.search(r'non.gaap|reconciliation|as adjusted', table_text, re.I), 'wrong_statement_basis')
    need(not re.search(r'\bCAD\b|\bAUD\b|\bHKD\b|\bSGD\b|Canadian dollars|Australian dollars|[€£¥]', head, re.I),
         'ambiguous_currency')
    need('$' in table_text or re.search(r'\bU\.?S\.? dollars?\b|\bUSD\b', head, re.I), 'statement_currency_missing')
    currency = 'explicit_usd' if re.search(r'\bU\.?S\.? dollars?\b|\bUSD\b|\bUnited States dollars\b', head, re.I) else 'dollar_symbol_only'
    columns, all_dates = [], set()
    for col in sorted({c for r, c in grid if r < stop}):
        parts = []
        for ri in range(stop):
            item = grid.get((ri, col))
            if item and item['text'] not in parts:
                parts.append(item['text'])
        found_dates = dates(' '.join(parts))
        all_dates.update(found_dates)
        if found_dates == {source['fiscal_period_end']}:
            columns.append(col)
    need(columns, 'current_period_column_not_found')
    need(max(all_dates) == source['fiscal_period_end'], 'selected_period_is_not_current')

    def boundary(name):
        locations = [i for i, label in enumerate(labels) if label == name]
        need(len(locations) == 1, 'section_boundary_not_unique:' + name)
        return locations[0]

    ca = next((i for i, label in enumerate(labels) if label == 'current assets'), stop)
    tca, ta = boundary('total current assets'), boundary('total assets')
    cl, tcl = boundary('current liabilities'), boundary('total current liabilities')
    ends = [i for i, label in enumerate(labels) if i > tcl and
            (label == 'total liabilities' or label in ('stockholders equity', 'shareholders equity'))]
    need(ends, 'noncurrent_liabilities_end_missing')
    end = min(ends)
    need(ca < tca <= ta < cl < tcl < end, 'balance_sheet_section_order')
    ranges = {'current_assets': (ca, tca), 'current_liabilities': (cl, tcl),
              'noncurrent_liabilities': (tcl, end)}
    result, rejections, scope = {}, [], []
    matched_rows = set()
    for field in FIELDS:
        positions = [i for i, row in enumerate(rows) if matches(field, label_of(row), source['ticker'])]
        if not positions:
            result[field] = None
            if field != 'current_portion_long_term_debt' or source['ticker'] == 'MSFT':
                rejections.append('missing_field:' + field)
            continue
        if len(positions) != 1:
            result[field] = None
            rejections.append('ambiguous_field:' + field)
            continue
        ri = positions[0]
        matched_rows.add(ri)
        if field == 'total_assets':
            valid_section = ri == ta
        else:
            lower, upper = ranges[SECTIONS[field]]
            valid_section = lower < ri < upper
        if not valid_section:
            rejections.append('wrong_section:' + field)
        selected = [c for c in rows[ri] if any(col in columns for col in c['columns'])]
        if any(any(col not in columns for col in c['columns']) for c in selected):
            rejections.append('cross_period_span:' + field)
        raw_value = norm(' '.join(c['text'] for c in selected))
        # These two provenance representations are both lossless: all cells
        # under the date heading, or the one amount cell excluding $/spacers.
        amounts = [c for c in selected if c['text'] not in ('', '$')]
        alternative = None
        if len(amounts) == 1:
            alternative = dict(raw_value=amounts[0]['text'], selected_cells=[amounts[0]['physical']],
                               selected_columns=amounts[0]['columns'])
        try:
            value = decimal_token(raw_value)
            # A reported zero is a valid source amount, including liquid and
            # total assets. Denominator eligibility belongs to a later stage.
            if value < 0:
                rejections.append('nonpositive_balance:' + field)
        except ValueError:
            value = None
            rejections.append('unavailable_amount:' + field)
        result[field] = dict(label=label_of(rows[ri]), row_index=ri,
                             raw_cells=[c['text'] for c in rows[ri]], raw_value=raw_value,
                             value_reported=None if value is None else format(value, 'f'),
                             scale=scale, section=SECTIONS[field],
                             selected_columns=columns,
                             selected_cells=[c['physical'] for c in selected],
                             amount_cell_alternative=alternative)
    for ri, row in enumerate(rows):
        label = label_of(row)
        key = canonical(label)
        if not key:
            continue
        flags = []
        if re.search(r'lease', key): flags.append('lease_liabilities_excluded')
        if re.search(r'restricted cash', key): flags.append('restricted_cash_scope')
        if re.search(r'securities lend|securities loan|loaned|securities pledged', key): flags.append('securities_lending_scope')
        if 'convertible debt conversion obligation' in key: flags.append('convertible_conversion_obligation_excluded')
        if 'convertible short term debt' in key: flags.append('convertible_short_term_debt_scope')
        for flag in flags:
            scope.append(dict(code=flag, row_index=ri, label=label))
        if cl < ri < end and ri not in matched_rows and re.search(r'\bdebt\b|borrow|commercial paper|notes payable|credit facilit', key):
            if not any('lease' in flag for flag in flags):
                rejections.append('unmapped_borrowing_row:' + str(ri) + ':' + label)
    if source['ticker'] == 'NVDA' and result['current_portion_long_term_debt'] is not None:
        rejections.append('nvda_current_portion_overlap_requires_review')
    return dict(table_index=ti, header_rows=[[c['text'] for c in row] for row in rows[:stop]],
                selected_period=source['fiscal_period_end'], scale=scale, scale_context=head,
                currency_symbol='$' if '$' in table_text else None, currency_status=currency,
                fields=result, source_rejections=rejections, conditional_coverage=scope,
                publication=pub)


def compare_record(raw, source, record):
    """All detected source and extractor failures survive in a JSON-safe audit."""
    base = dict(ticker=source['ticker'], fiscal_year=source['fiscal_year'],
                fiscal_period_end=source['fiscal_period_end'], raw_sha256=source['raw_sha256'])
    discrepancies = []

    def check(condition, text):
        if not condition:
            discrepancies.append(text)

    for key in ('ticker', 'fiscal_year', 'fiscal_period_end'):
        check(record.get(key) == source[key], 'record_identity:' + key)
    check(record.get('status') in ('candidate', 'rejected'), 'record_status')
    check(type(record.get('reasons')) is list, 'record_reasons_schema')
    check(type(record.get('notes')) is list, 'record_notes_schema')
    if record.get('status') == 'rejected':
        check(bool(record.get('reasons')), 'rejection_without_reason')
    try:
        oracle = inspect_html(raw, source)
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        check(record.get('status') == 'rejected', 'source_failure_claimed_candidate')
        return dict(base, status='rejected', verified=not discrepancies,
                    discrepancies=discrepancies, source_rejections=[str(exc)], evidence=None)
    statement = record.get('statement') or {}
    for key in ('table_index', 'header_rows', 'selected_period', 'scale', 'currency_symbol'):
        check(statement.get(key) == oracle[key], 'statement_mismatch:' + key)
    check(str(statement.get('currency_status', '')).lower() == oracle['currency_status'], 'statement_mismatch:currency_status')
    check(type(statement.get('scale')) is int, 'statement_scale_type')
    evidence = statement.get('scale_evidence')
    check(type(evidence) is str and bool(evidence) and all(
        word in canonical(oracle['scale_context']).split() for word in canonical(evidence).split()), 'scale_evidence_not_in_header')
    if isinstance(evidence, str):
        unit = re.findall(r'\bin\s+(thousands|millions|billions)\b', evidence, re.I)
        check(len({x.lower() for x in unit}) == 1 and bool(unit), 'scale_evidence_without_unit')
    actual_fields = record.get('fields') or {}
    check(set(actual_fields) == set(FIELDS), 'field_set_mismatch')
    for field, expected in oracle['fields'].items():
        actual = actual_fields.get(field)
        if expected is None:
            check(actual is None, 'invented_field:' + field)
            continue
        if actual is None and (expected['value_reported'] is None or
                               any(r == 'wrong_section:' + field or r == 'nonpositive_balance:' + field for r in oracle['source_rejections'])):
            continue
        if not isinstance(actual, dict):
            check(False, 'omitted_source_field:' + field)
            continue
        for key in ('label', 'row_index', 'raw_cells', 'scale', 'section'):
            check(actual.get(key) == expected[key], 'field_mismatch:' + field + ':' + key)
        check(type(actual.get('row_index')) is int and type(actual.get('scale')) is int, 'field_index_scale_type:' + field)
        value = actual.get('value_reported')
        if expected['value_reported'] is None:
            check(value is None, 'unavailable_amount_coerced:' + field)
        else:
            try:
                need(type(value) is str and bool(re.fullmatch(r'-?\d+(?:\.\d+)?', value)), 'not_plain_decimal_string')
                check(Decimal(value) == Decimal(expected['value_reported']), 'numeric_mismatch:' + field)
            except (ValueError, InvalidOperation):
                check(False, 'numeric_format:' + field)
        variants = [expected] + ([expected['amount_cell_alternative']] if expected['amount_cell_alternative'] else [])
        check(any(all(actual.get(key) == variant[key] for key in ('raw_value', 'selected_columns', 'selected_cells'))
                  for variant in variants), 'field_mismatch:' + field + ':physical_current_cell_selection')
    current_keys = ('short_term_debt', 'current_portion_long_term_debt') if source['ticker'] == 'MSFT' else ('short_term_debt',)
    # These are provenance lists of present reported rows, not usable-value
    # lists; a located dash can remain listed only on a rejected record.
    current = [k for k in current_keys if isinstance(actual_fields.get(k), dict)]
    noncurrent = ['long_term_debt'] if isinstance(actual_fields.get('long_term_debt'), dict) else []
    for key, expected in (('current_debt_components', current), ('noncurrent_debt_components', noncurrent)):
        actual = record.get(key)
        check(type(actual) is list and len(actual) == len(set(actual)) and set(actual) == set(expected), 'debt_component_mismatch:' + key)
    note_text = canonical(json.dumps(record.get('notes', []), ensure_ascii=False))
    for flag in oracle['conditional_coverage']:
        term = ('lease' if flag['code'].startswith('lease') else 'restricted' if flag['code'].startswith('restricted')
                else 'securit' if flag['code'].startswith('securities') else 'convertible')
        check(term in note_text, 'unacknowledged_scope:' + flag['code'])
    if oracle['source_rejections']:
        check(record.get('status') == 'rejected', 'source_rejection_claimed_candidate')
    else:
        check(record.get('status') != 'candidate' or not record.get('reasons'), 'candidate_has_rejections')
    return dict(base, status='candidate' if record.get('status') == 'candidate' and not oracle['source_rejections'] and not discrepancies else 'rejected',
                verified=not discrepancies, discrepancies=discrepancies,
                source_rejections=oracle['source_rejections'],
                conditional_coverage=oracle['conditional_coverage'], evidence=oracle)


def availability_check(source, actual, sessions, closes):
    """Independently derive the date-only bound and exact next-session close."""
    import pandas as pd
    need(sessions == sorted(set(sessions)), 'calendar_not_unique_ordered')
    release = source['release_date']
    need(sessions[0] <= release < sessions[-1], 'release_outside_calendar')
    following = next(s for s in sessions if s > release)
    midnight = datetime.combine(iso(release) + timedelta(days=1), time(),
                                tzinfo=ZoneInfo(source['publication_timezone']))
    upper = pd.Timestamp(midnight).tz_convert('UTC') - pd.Timedelta(nanoseconds=1)
    close = pd.Timestamp(closes[following])
    need(close.tz is not None and close > upper, 'invalid_next_session_close')
    expected = dict(status='ready', symbol=source['ticker'], asset_type='equity', precision='date_only',
                    publication_date=release, publication_at=None,
                    publication_timezone=source['publication_timezone'],
                    official_actual_period_end=source['fiscal_period_end'], source_url=source['source_url'],
                    available_session=following, known_by_is_actual_publication=False,
                    known_by_at=close.isoformat(), date_only_upper_bound_at=upper.isoformat(),
                    exchange='XNYS', policy_version='earnings_publication_availability_v1', issues=[])
    need(type(actual) is dict and all(actual.get(k) == v for k, v in expected.items()), 'availability_mismatch')
    return expected


def verify(plan, records):
    """Read every pinned source before auditing; structural/provenance errors raise."""
    need(type(plan) is dict and plan.get('schema') == 'liquidity162_source_scope_v1', 'source_plan_schema')
    need(plan.get('no_network') is True and plan.get('no_observer_writes') is True, 'source_plan_scope')
    sources = plan['source_records']
    need(type(sources) is list and bool(sources) and type(records) is list, 'record_list_required')
    keys = [(s['ticker'], s['fiscal_year']) for s in sources]
    need(len(keys) == len(set(keys)), 'duplicate_plan_source')
    need([(r['ticker'], r['fiscal_year']) for r in records] == keys, 'complete_ordered_source_records_required')
    pinned = {}
    for name, expected in plan['source_hashes'].items():
        path = Path(name)
        need(path.is_absolute() and '..' not in path.parts and path.resolve() == path, 'absolute_nonsymlink_source_required')
        need(not any('onedrive' in p.lower() or 'cloudstorage' in p.lower() for p in path.parts), 'cloud_path_forbidden')
        need(path.is_relative_to(ROOT), 'source_outside_research_root')
        need(not any(p.endswith('_monitor') for p in path.parts), 'observer_source_forbidden')
        raw = path.read_bytes()
        need(hashlib.sha256(raw).hexdigest() == expected, 'source_hash_mismatch:' + name)
        pinned[name] = raw
    audits = []
    calendar_name = str(ROOT / 'outputs/monthly_paired_target143/CALENDAR.json')
    need(calendar_name in pinned, 'pinned_calendar_missing')
    calendar = json.loads(pinned[calendar_name])
    # The registered date axis includes checksum-verified exceptional closures.
    # Obtain close times from XNYS, then restrict to that independently pinned axis.
    from exchange_calendars.exchange_calendar_xnys import XNYSExchangeCalendar
    exchange = XNYSExchangeCalendar(start=calendar['start'], end=calendar['end'])
    close_map = {str(index.date()): row['close'] for index, row in exchange.schedule.iterrows()}
    need(all(s in close_map for s in calendar['sessions']), 'pinned_session_absent_from_xnys')
    for source, record in zip(sources, records):
        need(source['raw_path'] in pinned and plan['source_hashes'][source['raw_path']] == source['raw_sha256'], 'unbound_original_html')
        need(record.get('source_binding') == source, 'record_source_binding_mismatch')
        audit = compare_record(pinned[source['raw_path']], source, record)
        try:
            audit['availability'] = availability_check(source, record.get('availability'), calendar['sessions'], close_map)
        except (ValueError, KeyError, TypeError) as exc:
            audit['discrepancies'].append(str(exc)); audit['verified'] = False; audit['status'] = 'rejected'
        audits.append(audit)
    return dict(schema='liquidity162_independent_review_v1', passed=all(a['verified'] for a in audits),
                records=audits, source_hashes=deepcopy(plan['source_hashes']),
                candidate_count=sum(a['status'] == 'candidate' for a in audits),
                rejected_count=sum(a['status'] == 'rejected' for a in audits),
                source_only=True, numerical_factors_computed=0, orders_computed=0, returns_computed=0,
                original_vintage_verified=False, complete_correction_history_verified=False,
                scope='Reported current borrowings versus reported current cash and investments only; no total debt, unrestricted cash, or strategy admission.')


class SyntheticTests(unittest.TestCase):
    def fixture(self, ticker='MSFT', short='12', extra=''):
        liquid = ('Total cash, cash equivalents, and short-term investments' if ticker == 'MSFT'
                  else 'Cash, cash equivalents and marketable securities')
        company = 'Microsoft (MSFT)' if ticker == 'MSFT' else 'NVIDIA (NVDA)'
        location = 'REDMOND' if ticker == 'MSFT' else 'SANTA CLARA'
        portion = '<tr><td>Current portion of long-term debt</td><td>3</td><td>9</td></tr>' if ticker == 'MSFT' else ''
        title = 'Earnings Release FY25 Q4' if ticker == 'MSFT' else 'Fourth Quarter and Fiscal 2025'
        raw = f'''<h1>{company} {title}</h1><p>{location}, July 25, 2025. Fiscal year ended June 30, 2025.</p>
        <h2>Consolidated Balance Sheets (in millions)</h2><table>
        <tr><td></td><th>June 30, 2025</th><th>June 30, 2024</th></tr>
        <tr><td>Assets</td><td></td><td></td></tr><tr><td>Current assets:</td></tr>
        <tr><td>{liquid}</td><td>$ 1,234</td><td>$ 8,765</td></tr>
        <tr><td>Total current assets</td><td>2,000</td><td>9,000</td></tr>
        <tr><td>Total assets</td><td>3,000</td><td>10,000</td></tr>
        <tr><td>Current liabilities:</td></tr><tr><td>Short-term debt</td><td>{short}</td><td>8</td></tr>
        {portion}{extra}<tr><td>Total current liabilities</td><td>100</td><td>200</td></tr>
        <tr><td>Long-term debt</td><td>300</td><td>700</td></tr>
        <tr><td>Total liabilities</td><td>400</td><td>900</td></tr></table>'''
        source = dict(ticker=ticker, fiscal_year=2025, fiscal_period_end='2025-06-30', release_date='2025-07-25',
                      precision='date_only', publication_timezone='America/Los_Angeles',
                      source_url='https://www.microsoft.com/release' if ticker == 'MSFT' else 'https://nvidianews.nvidia.com/release',
                      raw_sha256=hashlib.sha256(raw.encode()).hexdigest())
        oracle = inspect_html(raw, source)
        record = {k: source[k] for k in ('ticker', 'fiscal_year', 'fiscal_period_end')}
        record.update(status='candidate' if not oracle['source_rejections'] else 'rejected',
                      reasons=deepcopy(oracle['source_rejections']), notes=[],
                      statement={k: oracle[k] for k in ('table_index', 'header_rows', 'selected_period', 'scale', 'currency_symbol', 'currency_status')},
                      fields=deepcopy(oracle['fields']),
                      current_debt_components=[k for k in ('short_term_debt', 'current_portion_long_term_debt') if oracle['fields'][k] is not None],
                      noncurrent_debt_components=['long_term_debt'])
        record['statement']['scale_evidence'] = 'Consolidated Balance Sheets (in millions)'
        return raw, source, record

    def test_msft_candidate(self):
        raw, source, record = self.fixture()
        self.assertEqual(compare_record(raw, source, record)['status'], 'candidate')

    def test_nvda_candidate_without_portion(self):
        raw, source, record = self.fixture('NVDA')
        self.assertEqual(compare_record(raw, source, record)['status'], 'candidate')

    def test_wrong_current_value(self):
        raw, source, record = self.fixture()
        record['fields']['short_term_debt']['value_reported'] = '8'
        self.assertFalse(compare_record(raw, source, record)['verified'])

    def test_wrong_period(self):
        raw, source, record = self.fixture()
        record['statement']['selected_period'] = '2024-06-30'
        self.assertFalse(compare_record(raw, source, record)['verified'])

    def test_wrong_scale(self):
        raw, source, record = self.fixture()
        record['statement']['scale'] = 1000
        self.assertFalse(compare_record(raw, source, record)['verified'])

    def test_dollar_not_iso(self):
        raw, source, record = self.fixture()
        record['statement']['currency_status'] = 'explicit_usd'
        self.assertFalse(compare_record(raw, source, record)['verified'])

    def test_dash_is_not_zero(self):
        raw, source, record = self.fixture(short='—')
        self.assertTrue(compare_record(raw, source, record)['verified'])
        record['fields']['short_term_debt']['value_reported'] = '0'
        self.assertFalse(compare_record(raw, source, record)['verified'])

    def test_numeric_zero_is_present(self):
        raw, source, record = self.fixture(short='0')
        self.assertEqual(compare_record(raw, source, record)['status'], 'candidate')

    def assert_reported_balance(self, field, token, candidate):
        for ticker in ('MSFT', 'NVDA'):
            with self.subTest(ticker=ticker, field=field, token=token):
                raw, source, record = self.fixture(ticker)
                cell = record['fields'][field]
                old = '<td>' + cell['label'] + '</td><td>' + cell['raw_value'] + '</td>'
                new = '<td>' + cell['label'] + '</td><td>' + token + '</td>'
                self.assertIn(old, raw)
                raw = raw.replace(old, new)
                cell.update(raw_value=token, value_reported=token if candidate else None)
                cell['raw_cells'][1] = token
                if not candidate:
                    record.update(status='rejected', reasons=['unavailable_amount:' + field])
                audit = compare_record(raw, source, record)
                self.assertTrue(audit['verified'], audit['discrepancies'])
                self.assertEqual(audit['status'], 'candidate' if candidate else 'rejected')
                self.assertEqual(audit['evidence']['fields'][field]['value_reported'], token if candidate else None)

    def test_zero_liquid_assets_is_valid_source_amount(self):
        self.assert_reported_balance('liquid_financial_assets', '0', True)

    def test_zero_total_assets_is_valid_source_amount(self):
        self.assert_reported_balance('total_assets', '0', True)

    def test_blank_dash_assets_stay_unavailable(self):
        for field in ('liquid_financial_assets', 'total_assets'):
            for token in ('', '—'):
                self.assert_reported_balance(field, token, False)

    def test_unknown_current_borrowings(self):
        raw, source, record = self.fixture(extra='<tr><td>Commercial paper</td><td>5</td><td>7</td></tr>')
        self.assertTrue(any('unmapped_borrowing' in r for r in inspect_html(raw, source)['source_rejections']))
        record['status'] = 'candidate'; record['reasons'] = []
        self.assertFalse(compare_record(raw, source, record)['verified'])

    def test_debt_double_count(self):
        raw, source, record = self.fixture()
        record['current_debt_components'].append('short_term_debt')
        self.assertFalse(compare_record(raw, source, record)['verified'])

    def test_omitted_available_field(self):
        raw, source, record = self.fixture()
        record['fields']['short_term_debt'] = None
        self.assertFalse(compare_record(raw, source, record)['verified'])

    def test_wrong_row(self):
        raw, source, record = self.fixture()
        record['fields']['short_term_debt']['row_index'] += 1
        self.assertFalse(compare_record(raw, source, record)['verified'])

    def test_missing_lease_note(self):
        raw, source, record = self.fixture(extra='<tr><td>Operating lease liabilities</td><td>5</td><td>7</td></tr>')
        self.assertFalse(compare_record(raw, source, record)['verified'])
        record['notes'] = ['Operating lease liabilities excluded from narrow borrowing proxy.']
        self.assertTrue(compare_record(raw, source, record)['verified'])

    def test_wrong_release(self):
        raw, source, record = self.fixture()
        source['release_date'] = '2025-07-26'
        self.assertFalse(compare_record(raw, source, record)['verified'])

    def test_wrong_statement(self):
        raw, source, record = self.fixture()
        raw = raw.replace('Consolidated Balance Sheets', 'Consolidated Cash Flows')
        self.assertFalse(compare_record(raw, source, record)['verified'])

    def test_ambiguous_statement(self):
        raw, source, record = self.fixture()
        self.assertFalse(compare_record(raw + raw, source, record)['verified'])

    def test_numeric_parser(self):
        self.assertEqual(decimal_token('(1,234.50)'), Decimal('-1234.50'))
        for token in ('', '-', '—', 'NaN', 'inf', '1,23', '1 23', '1e3', '( -2 )'):
            with self.assertRaises(ValueError): decimal_token(token)

    def test_missing_short_term_debt_is_not_zero(self):
        raw, source, record = self.fixture()
        raw = raw.replace('<tr><td>Short-term debt</td><td>12</td><td>8</td></tr>', '')
        self.assertIn('missing_field:short_term_debt', inspect_html(raw, source)['source_rejections'])
        self.assertFalse(compare_record(raw, source, record)['verified'])

    def test_negative_balance_rejected(self):
        raw, source, record = self.fixture(short='(12)')
        self.assertEqual(compare_record(raw, source, record)['status'], 'rejected')
        record['status'] = 'candidate'; record['reasons'] = []
        self.assertFalse(compare_record(raw, source, record)['verified'])

    def test_invalid_amount_cannot_be_candidate(self):
        for token in ('NaN', 'inf', '', '1,23'):
            raw, source, record = self.fixture(short=token)
            record['status'] = 'candidate'; record['reasons'] = []
            self.assertFalse(compare_record(raw, source, record)['verified'])

    def test_wrong_section_cannot_pass(self):
        raw, source, record = self.fixture()
        raw = raw.replace('<tr><td>Short-term debt</td><td>12</td><td>8</td></tr>', '')
        raw = raw.replace('<tr><td>Long-term debt</td>', '<tr><td>Short-term debt</td><td>12</td><td>8</td></tr><tr><td>Long-term debt</td>')
        self.assertIn('wrong_section:short_term_debt', inspect_html(raw, source)['source_rejections'])
        self.assertFalse(compare_record(raw, source, record)['verified'])

    def test_table_unit_conflict(self):
        raw, source, record = self.fixture()
        raw = raw.replace('</table>', '<tr><td>in thousands</td></tr></table>')
        self.assertFalse(compare_record(raw, source, record)['verified'])

    def test_competitive_currency(self):
        for currency in ('CAD', 'AUD', 'Canadian dollars', 'Australian dollars', 'USD and CAD', 'USD and AUD'):
            with self.subTest(currency=currency):
                raw, source, record = self.fixture()
                raw = raw.replace('(in millions)', '(in millions of ' + currency + ')')
                result = compare_record(raw, source, record)
                self.assertFalse(result['verified'])
                self.assertIn('ambiguous_currency', result['source_rejections'])

    def test_explicit_usd_case_compatible(self):
        raw, source, record = self.fixture()
        raw = raw.replace('(in millions)', '(in millions, USD)')
        record['statement']['currency_status'] = 'explicit_USD'
        record['statement']['scale_evidence'] = 'Consolidated Balance Sheets (in millions, USD)'
        self.assertTrue(compare_record(raw, source, record)['verified'])

    def test_release_date_in_table_not_publication(self):
        raw, source, record = self.fixture()
        raw = raw.replace('July 25, 2025', 'July 24, 2025')
        raw = raw.replace('</table>', '<tr><td>July 25, 2025</td></tr></table>')
        self.assertFalse(compare_record(raw, source, record)['verified'])

    def test_wrong_release_year(self):
        raw, source, record = self.fixture()
        raw = raw.replace('Release FY25 Q4', 'Release FY24 Q4')
        self.assertFalse(compare_record(raw, source, record)['verified'])

    def test_wrong_release_period(self):
        raw, source, record = self.fixture()
        raw = raw.replace('ended June 30, 2025', 'ended March 31, 2025')
        self.assertFalse(compare_record(raw, source, record)['verified'])

    def test_header_must_be_current_not_comparative(self):
        raw, source, record = self.fixture()
        raw = raw.replace('<th>June 30, 2024</th>', '<th>June 30, 2026</th>')
        self.assertFalse(compare_record(raw, source, record)['verified'])

    def test_currency_spacer_columns_and_amount_cell_forms(self):
        raw, source, record = self.fixture()
        raw = raw.replace('<th>June 30, 2025</th>', '<th colspan="2">June 30, 2025</th>')
        # Every financial row has a separate current-period dollar cell.
        raw = re.sub(r'(<tr><td>[^<]+</td>)(<td>)([^<]*)(</td><td>[^<]*</td></tr>)',
                     lambda m: m[1] + '<td>$</td><td>' + m[3].replace('$ ', '') + m[4], raw)
        oracle = inspect_html(raw, source)
        record['statement']['header_rows'] = oracle['header_rows']
        record['fields'] = deepcopy(oracle['fields'])
        self.assertTrue(compare_record(raw, source, record)['verified'])
        for field in record['fields'].values():
            if field is not None:
                field.update(field['amount_cell_alternative'])
        self.assertTrue(compare_record(raw, source, record)['verified'])
        record['fields']['short_term_debt']['selected_cells'] = [3]
        self.assertFalse(compare_record(raw, source, record)['verified'])

    def test_overlap_portion_nvda_rejected(self):
        raw, source, record = self.fixture('NVDA', extra='<tr><td>Current portion of long-term debt</td><td>3</td><td>2</td></tr>')
        record['current_debt_components'] = ['short_term_debt']
        self.assertTrue(compare_record(raw, source, record)['verified'])
        record['status'] = 'candidate'; record['reasons'] = []
        self.assertFalse(compare_record(raw, source, record)['verified'])

    def test_availability_next_close_not_midnight(self):
        _, source, _ = self.fixture()
        sessions = ['2025-07-24', '2025-07-25', '2025-07-28', '2025-07-29']
        closes = {s: s + 'T20:00:00+00:00' for s in sessions}
        actual = dict(status='ready', symbol='MSFT', asset_type='equity', precision='date_only',
                      publication_date='2025-07-25', publication_at=None,
                      publication_timezone='America/Los_Angeles',
                      official_actual_period_end='2025-06-30', source_url=source['source_url'],
                      available_session='2025-07-28', known_by_is_actual_publication=False,
                      known_by_at='2025-07-28T20:00:00+00:00',
                      date_only_upper_bound_at='2025-07-26T06:59:59.999999999+00:00',
                      exchange='XNYS', policy_version='earnings_publication_availability_v1', issues=[])
        self.assertEqual(availability_check(source, actual, sessions, closes), actual)
        for key, value in [('known_by_at', '2025-07-28T00:00:00+00:00'),
                           ('available_session', '2025-07-25'), ('known_by_is_actual_publication', True)]:
            bad = deepcopy(actual); bad[key] = value
            with self.assertRaises(ValueError): availability_check(source, bad, sessions, closes)


if __name__ == '__main__':
    unittest.main()
