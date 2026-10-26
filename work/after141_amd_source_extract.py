"""Pure, source-only AMD annual statement extraction; no I/O or factor calculations.

``extract(raw, spec)`` accepts the AMD coverage-manifest source dictionaries
(fiscal_year, fiscal_year_end, release_date, raw_html, receipt, source_url).
The shorter year/end/release aliases are also accepted, but conflicting aliases
are rejected. Returned candidates always require separate source admission.
"""
from datetime import date, timedelta
from decimal import Decimal
import hashlib
import re
from bs4 import BeautifulSoup, Tag


class ExtractionError(ValueError):
    """A fail-closed source or statement ambiguity with an inspectable reason."""


MONTHS = {name.lower(): i for i, name in enumerate((
    'January', 'February', 'March', 'April', 'May', 'June', 'July',
    'August', 'September', 'October', 'November', 'December'), 1)}
MONTHS.update({name[:3]: number for name, number in list(MONTHS.items())})
DATE = re.compile(r'\b(' + '|'.join(sorted(MONTHS, key=len, reverse=True)) +
                  r')\.?\s+(\d{1,2}),?\s+(\d{4})\b', re.I)
TITLES = {'gross_amount': re.compile(r'(?:condensed\s+)?consolidated\s+statements?\s+of\s+(?:operations|income)', re.I),
          'total_assets': re.compile(r'(?:condensed\s+)?consolidated\s+balance\s+sheets?', re.I)}
ADJUSTED = re.compile(r'non[\s\-\u2010-\u2015]*GAAP|reconciliation|as adjusted|pro forma', re.I)
BODY_START = re.compile(r'(?:net\s+|total\s+)?revenues?:?|assets:?|cash(?: and cash equivalents)?:?', re.I)


def norm(value):
    return re.sub(r'\s+', ' ', value).strip()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def exact_amount(display):
    text = norm(display).replace('\u2212', '-')
    if '%' in text or re.search(r'\bpercent', text, re.I):
        raise ExtractionError('Percentage is not a monetary amount')
    match = re.fullmatch(r'\$?\s*(\()?\s*(-?\s*(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*(\))?', text)
    if match is None or bool(match[1]) != bool(match[3]):
        raise ExtractionError('Missing or ambiguous displayed monetary amount')
    number = match[2].replace(' ', '').replace(',', '')
    if match[1] and number.startswith('-'):
        raise ExtractionError('Conflicting monetary signs')
    value = Decimal(number)
    return format(-value if match[1] else value, 'f')


def dates(parts):
    result = set()
    for month, day, year in DATE.findall(' '.join(parts)):
        try:
            result.add(date(int(year), MONTHS[month.lower()], int(day)).isoformat())
        except ValueError as exc:
            raise ExtractionError('Invalid statement date') from exc
    return sorted(result)


def table_grid(table):
    """Expand spans while preserving physical cells and their logical columns."""
    rows = [row for row in table.find_all('tr') if row.find_parent('table') is table]
    grid, physical = {}, {}
    for ri, row in enumerate(rows):
        column = 0
        for pi, cell in enumerate(row.find_all(['td', 'th'], recursive=False)):
            while (ri, column) in grid:
                column += 1
            try:
                rs, cs = int(cell.get('rowspan', 1)), int(cell.get('colspan', 1))
            except ValueError as exc:
                raise ExtractionError('Noninteger table span') from exc
            if not (1 <= rs <= 100 and 1 <= cs <= 100):
                raise ExtractionError('Invalid table span')
            item = {'text': norm(cell.get_text(' ', strip=True)), 'row': ri,
                    'physical_cell': pi, 'columns': list(range(column, column + cs)),
                    'rowspan': rs, 'parsed_cell_markup_sha256': digest(str(cell).encode('utf-8'))}
            physical[(ri, pi)] = item
            for r in range(ri, ri + rs):
                for c in item['columns']:
                    if (r, c) in grid:
                        raise ExtractionError('Overlapping table spans')
                    grid[(r, c)] = item
            column += cs
    return rows, grid, physical


def header_evidence(grid, stop, column):
    evidence, seen = [], set()
    for row in range(stop):
        item = grid.get((row, column))
        if item and item['text'] and (item['row'], item['physical_cell']) not in seen:
            evidence.append(item)
            seen.add((item['row'], item['physical_cell']))
    return evidence


def external_context(table):
    """Nearest preceding heading/units, without crossing another table."""
    parts, seen = [], set()
    for step, node in enumerate(table.previous_elements):
        if step > 250:
            break
        if not isinstance(node, Tag):
            continue
        if node.name == 'table' or node.find_parent('table') is not None:
            break
        if node.name not in ('div', 'p', 'font', 'strong', 'b', 'h1', 'h2', 'h3', 'h4') or node.find('table'):
            continue
        text = norm(node.get_text(' ', strip=True))
        if not text or len(text) > 350 or text in seen:
            continue
        parts.append(text)
        seen.add(text)
        if any(pattern.search(text) for pattern in TITLES.values()):
            break
        if len(parts) >= 12:
            break
    return list(reversed(parts))


def scale_evidence(parts):
    hits = []
    for part in parts:
        for match in re.finditer(r'\b(millions|thousands)\b', part, re.I):
            hits.append({'literal': part, 'unit': match[1].lower()})
    units = {hit['unit'] for hit in hits}
    if len(units) != 1:
        raise ExtractionError('Missing or conflicting statement scale')
    unit = next(iter(units))
    return {'multiplier': {'millions': 1000000, 'thousands': 1000}[unit],
            'unit': unit, 'evidence_literals': sorted({hit['literal'] for hit in hits})}


def locate_statement(table, ti, kind, label, end):
    rows, grid, physical = table_grid(table)
    starts = [ri for ri, row in enumerate(rows) if any(BODY_START.fullmatch(norm(cell.get_text(' ', strip=True)))
              for cell in row.find_all(['td', 'th'], recursive=False))]
    if not starts:
        raise ExtractionError('Cannot establish statement header boundary')
    stop = starts[0]
    header_parts = [item['text'] for (ri, _), item in physical.items() if ri < stop and item['text']]
    if table.caption:
        header_parts.insert(0, norm(table.caption.get_text(' ', strip=True)))
    context_parts = header_parts if TITLES[kind].search(' '.join(header_parts)) else external_context(table) + header_parts
    context = ' '.join(context_parts)
    if not TITLES[kind].search(context):
        raise ExtractionError('Formal consolidated statement title missing')
    table_text = norm(table.get_text(' ', strip=True))
    if ADJUSTED.search(context) or ADJUSTED.search(table_text):
        raise ExtractionError('Non-GAAP, reconciliation, adjusted or pro-forma statement')
    selected_rows = {ri for (ri, _), item in physical.items() if item['text'].casefold() == label.casefold()}
    if len(selected_rows) != 1:
        raise ExtractionError('Missing or duplicate exact statement row: ' + label)
    ri = next(iter(selected_rows))
    if ri < stop:
        raise ExtractionError('Selected amount precedes statement body')
    if kind == 'gross_amount' and not re.search(r'cost of (?:sales|revenue|goods)', table_text, re.I):
        raise ExtractionError('Monetary gross amount lacks cost-of-sales statement context')
    all_columns = sorted({column for (_, column) in grid})
    by_column = {column: header_evidence(grid, stop, column) for column in all_columns}
    target_columns, period_dates = [], set()
    for column, evidence in by_column.items():
        texts = [item['text'] for item in evidence]
        column_dates = dates(texts)
        periods = [item['text'] for item in evidence if re.search(r'\b(?:year|months?)\s+ended\b', item['text'], re.I)]
        annual = any(re.fullmatch(r'year\s+ended(?:\s+.*)?', period, re.I) for period in periods)
        quarterly = any(re.search(r'\b(?:three|six|nine|3|6|9)\s+months?\s+ended', period, re.I) for period in periods)
        if kind == 'total_assets' or annual:
            period_dates.update(column_dates)
        if column_dates == [end] and (kind == 'total_assets' or (annual and not quarterly)):
            target_columns.append(column)
    if not target_columns or target_columns != list(range(min(target_columns), max(target_columns) + 1)):
        raise ExtractionError('Missing or ambiguous current annual/date column group')
    if not period_dates or max(period_dates) != end:
        raise ExtractionError('Requested period is not the latest printed statement period')
    target = set(target_columns)
    amount_cells = [item for (r, _), item in physical.items() if r == ri and target.intersection(item['columns'])]
    if not amount_cells or any(not set(item['columns']).issubset(target) or item['rowspan'] != 1 for item in amount_cells):
        raise ExtractionError('Amount cell spans multiple date groups or rows')
    displayed = norm(' '.join(item['text'] for item in amount_cells if item['text']))
    amount = exact_amount(displayed)
    currency_cells = [item for (r, _), item in physical.items() if r >= stop and target.intersection(item['columns']) and '$' in item['text']]
    if not currency_cells:
        raise ExtractionError('No literal dollar evidence in selected statement column')
    if re.search(r'[€£¥]|\b(?:EUR|GBP|JPY|CAD|AUD|HKD)\b', context + ' ' + table_text, re.I):
        raise ExtractionError('Conflicting or unsupported statement currency')
    headers = []
    seen = set()
    for column in target_columns:
        for item in by_column[column]:
            key = (item['row'], item['physical_cell'])
            if key not in seen:
                seen.add(key)
                headers.append(item)
    scale = scale_evidence(context_parts)
    return {'measure_label': label, 'displayed_amount': displayed, 'amount_decimal': amount,
            'scale': scale['multiplier'], 'scale_evidence': scale,
            'currency_literal': '$', 'currency_iso': None, 'date': end,
            'locator': {'table_index_zero_based': ti, 'row_index_zero_based': ri,
                        'logical_columns_zero_based': target_columns},
            'amount_cells': amount_cells, 'header_cells': headers,
            'all_statement_period_dates': sorted(period_dates),
            'period_header_cells': [item for (r, _), item in physical.items() if r < stop and dates([item['text']])],
            'statement_title_and_unit_context': context,
            'currency_evidence_cells': currency_cells,
            'fact_admitted': False, 'context_independent_review_required': True}


def _field(spec, key, alias):
    if key in spec and alias in spec and spec[key] != spec[alias]:
        raise ExtractionError('Conflicting spec aliases: ' + key)
    value = spec.get(key, spec.get(alias))
    if value is None:
        raise ExtractionError('Missing spec field: ' + key)
    return value


def extract(raw: bytes, spec: dict) -> dict:
    """Extract exactly one current annual pair, never infer historical admission.

    No files, sockets or subprocesses are opened. A caller must independently
    verify the receipt, event/date provenance, identity and frozen plan.
    """
    if not isinstance(raw, bytes) or not raw or not isinstance(spec, dict):
        raise ExtractionError('Expected nonempty source bytes and a source specification')
    year = _field(spec, 'fiscal_year', 'year')
    end = _field(spec, 'fiscal_year_end', 'end')
    release = _field(spec, 'release_date', 'release')
    if type(year) is not int or year not in (2015, 2016, 2017, 2018, 2020, 2021, 2022, 2023, 2024, 2025):
        raise ExtractionError('Year outside reviewed AMD source scope; FY2019 remains missing')
    if spec.get('symbol', 'AMD') != 'AMD':
        raise ExtractionError('Wrong issuer specification')
    try:
        end_day, release_day = date.fromisoformat(end), date.fromisoformat(release)
    except (TypeError, ValueError) as exc:
        raise ExtractionError('Invalid end or release date') from exc
    if end_day.year != year or not 0 < (release_day - end_day).days < 120:
        raise ExtractionError('Inconsistent fiscal year/end/release dates')
    binding = spec.get('raw_html', {})
    expected_sha = binding.get('sha256', spec.get('raw_sha256'))
    if expected_sha is not None and expected_sha != digest(raw):
        raise ExtractionError('Source SHA-256 mismatch')
    if binding.get('sha256') and spec.get('raw_sha256') and binding['sha256'] != spec['raw_sha256']:
        raise ExtractionError('Conflicting raw source hash bindings')
    soup = BeautifulSoup(raw, 'html.parser')
    for node in soup(['script', 'style']):
        node.decompose()
    page_text = norm(soup.get_text(' ', strip=True))
    label = 'Gross margin' if year <= 2018 else 'Gross profit'
    if any(spec.get(key, label).casefold() != label.casefold() for key in ('numerator_label', 'expected_label')):
        raise ExtractionError('Unexpected numerator semantics for this source year')
    statements = {}
    for kind, row_label in (('gross_amount', label), ('total_assets', 'Total Assets')):
        candidates, failures, formal_conflicts = [], [], []
        for ti, table in enumerate(soup.find_all('table')):
            own_rows = [row for row in table.find_all('tr') if row.find_parent('table') is table]
            own_cells = [cell for row in own_rows for cell in row.find_all(['td', 'th'], recursive=False)]
            if not any(norm(cell.get_text(' ', strip=True)).casefold() == row_label.casefold() for cell in own_cells):
                continue
            try:
                candidates.append(locate_statement(table, ti, kind, row_label, end))
            except ExtractionError as exc:
                failures.append(str(exc))
                text = norm(table.get_text(' ', strip=True))
                context = text if TITLES[kind].search(text) else ' '.join(external_context(table))
                if TITLES[kind].search(context) and not ADJUSTED.search(context + ' ' + text):
                    formal_conflicts.append(str(exc))
        if candidates and formal_conflicts:
            raise ExtractionError(kind + ': conflicting additional formal statement: ' + '; '.join(formal_conflicts))
        if len(candidates) != 1:
            raise ExtractionError(kind + ': expected one unambiguous formal statement; found ' + str(len(candidates)) +
                                  ('; rejected: ' + '; '.join(sorted(set(failures))) if failures else ''))
        statements[kind] = candidates[0]
    gp, assets = statements['gross_amount'], statements['total_assets']
    if gp['scale'] != assets['scale']:
        raise ExtractionError('Paired statements have conflicting scales')
    # Older releases include an interim comparative balance-sheet column.
    # Require the prior boundary to be printed in this same balance sheet AND
    # the same report's annual income header; never use the interim date.
    prior_dates = [day for day in assets['all_statement_period_dates']
                   if day < end and day in gp['all_statement_period_dates']]
    if len(prior_dates) != 1:
        raise ExtractionError('Prior annual boundary missing or ambiguous in same balance sheet and annual income header')
    prior = date.fromisoformat(prior_dates[0])
    start = prior + timedelta(days=1)
    days = (end_day - start).days + 1
    if days not in (364, 371):
        raise ExtractionError('AMD annual interval is not 52 or 53 weeks')
    discontinued = re.search(r'.{0,120}discontinued operations.{0,180}', page_text, re.I)
    gaap = re.search(r'(?:consolidated financial statements|financial statements|financial results).{0,80}?'
                     r'(?:prepared|reported|presented) in accordance with (?:U\.?S\.?\s+)?GAAP', page_text, re.I)
    flags = ['source_context_independent_review_required', 'derived_annual_start_not_verbatim',
             'ISO_currency_unverified_literal_dollar_only', 'historical_first_public_version_unverified',
             'receipt_event_release_and_identity_not_verified_by_pure_extractor']
    if year <= 2018:
        flags.append('monetary_Gross_margin_semantic_review_required')
    if year == 2025 or discontinued:
        flags.append('discontinued_operations_measurement_compatibility_review_required')
    if not gaap:
        flags.append('explicit_issuer_GAAP_policy_not_located')
    return {'schema': 'amd_source_only_annual_pair_v1', 'symbol': 'AMD', 'fiscal_year': year,
            **statements,
            'annual_period': {'start': start.isoformat(), 'end': end, 'days_inclusive': days,
                              'prior_year_end': prior.isoformat(), 'start_is_direct_quote': False,
                              'prior_year_end_balance_header_cells': [item for item in assets['period_header_cells'] if prior.isoformat() in dates([item['text']])],
                              'prior_year_end_also_printed_in_annual_income_header': True,
                              'derivation': 'day_after_explicit_prior_year_end_in_same_source_balance_sheet',
                              'derivation_requires_independent_review': True},
            'source': {'raw_sha256': digest(raw), 'raw_path': binding.get('path', spec.get('raw_path')),
                       'receipt': spec.get('receipt'), 'receipt_path': spec.get('receipt_path'),
                       'source_url': spec.get('source_url'), 'receipt_and_event_verified': False},
            'availability': {'release_date': release, 'precision': 'date_only',
                             'release_date_verified': False, 'first_public_version_verified': False},
            'gaap_evidence': {'formal_unadjusted_consolidated_statement_candidates': True,
                              'issuer_policy_literal': gaap.group(0) if gaap else None,
                              'independent_semantic_admission': False},
            'discontinued_operations_literal': discontinued.group(0) if discontinued else None,
            'semantic_review_required': True, 'review_flags': flags,
            'fact_admitted': False, 'context_independent_review_required': True,
            'strategy_admitted': False, 'factor_or_strategy_admitted': False}


if __name__ == '__main__':
    raise SystemExit('Pure extractor only; run work/after141_amd_source_extract_tests.py through ./run_local.sh.')
