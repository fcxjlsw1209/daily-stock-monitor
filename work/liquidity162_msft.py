"""Pure MSFT Q4 balance-sheet candidate extraction for liquidity162.

The caller must verify SOURCE_PLAN membership, original-byte/receipt hashes and
publication evidence before calling extract. Importing this module reads no
data. This module performs no aggregation, ratio, ranking or return calculation.
All numeric strings retain the reported statement units; candidates still need
independent admission review.
"""
from datetime import date
from decimal import Decimal
import re

from bs4 import BeautifulSoup
from profitability141_issuer_statements import (
    exact_amount, header_dates, headers_for, norm, scale_from,
    statement_context, table_grid, validate_statement,
)


FIELD_LABELS = {
    'liquid_financial_assets': (
        'total cash, cash equivalents, and short-term investments',
        'total cash, cash equivalents and short-term investments',
    ),
    'short_term_debt': ('short-term debt',),
    'current_portion_long_term_debt': ('current portion of long-term debt',),
    'long_term_debt': ('long-term debt',),
    'total_assets': ('total assets',),
}
FIELD_SECTIONS = {
    'liquid_financial_assets': 'current_assets',
    'short_term_debt': 'current_liabilities',
    'current_portion_long_term_debt': 'current_liabilities',
    'long_term_debt': 'noncurrent_liabilities',
    'total_assets': 'all_assets',
}


def _label(text):
    text = norm(text).casefold().replace('\u2019', "'").replace('\u2018', "'")
    text = re.sub(r'\s*[-\u2010\u2011\u2012\u2013\u2014]\s*', '-', text)
    return text.rstrip(':').strip()


def _cells(row):
    return [norm(c.get_text(' ', strip=True))
            for c in row.find_all(['td', 'th'], recursive=False)]


def _empty(event):
    return {
        'ticker': event.get('ticker'),
        'fiscal_year': event.get('fiscal_year'),
        'fiscal_period_end': event.get('fiscal_period_end'),
        'status': 'rejected', 'reasons': [],
        'statement': {
            'table_index': None, 'header_rows': [], 'selected_period': None,
            'scale': None, 'scale_evidence': None, 'currency_symbol': None,
            'currency_status': 'unavailable',
        },
        'fields': {key: None for key in FIELD_LABELS},
        'current_debt_components': [], 'noncurrent_debt_components': [],
        'notes': [],
    }


def _one(labels, target):
    matches = [i for i, label in enumerate(labels) if label == target]
    if len(matches) != 1:
        raise ValueError('missing_or_ambiguous_section:' + target)
    return matches[0]


def _sections(labels):
    ordered = [
        ('assets', 'assets'),
        ('current_assets', 'current assets'),
        ('current_assets_end', 'total current assets'),
        ('total_assets', 'total assets'),
        ('liabilities', "liabilities and stockholders' equity"),
        ('current_liabilities', 'current liabilities'),
        ('current_liabilities_end', 'total current liabilities'),
        ('equity', "stockholders' equity"),
    ]
    boundaries = {key: _one(labels, text) for key, text in ordered}
    indices = [boundaries[key] for key, _ in ordered]
    if indices != sorted(set(indices)):
        raise ValueError('invalid_asset_liability_section_order')
    return boundaries


def _columns(grid, stop, period_end):
    columns = sorted({col for row, col in grid if row < stop})
    selected, all_dates = [], set()
    for col in columns:
        headers = headers_for(grid, stop, col, stop)
        if re.search(r'\b(?:months?|years?)\s+ended\b', ' '.join(headers), re.I):
            raise ValueError('duration_header_in_balance_sheet')
        dates = header_dates(headers)
        if len(dates) > 1:
            raise ValueError('ambiguous_period_column')
        all_dates.update(dates)
        if dates == [period_end]:
            selected.append(col)
    if not selected or not all_dates or max(all_dates) != period_end:
        raise ValueError('current_period_missing_or_future_comparison_present')
    if selected != list(range(min(selected), max(selected) + 1)):
        raise ValueError('disjoint_current_period_columns')
    return selected


def _in_section(row, section, bounds):
    if section == 'current_assets':
        return bounds['current_assets'] < row < bounds['current_assets_end']
    if section == 'current_liabilities':
        return bounds['current_liabilities'] < row < bounds['current_liabilities_end']
    if section == 'noncurrent_liabilities':
        return bounds['current_liabilities_end'] < row < bounds['equity']
    return row == bounds['total_assets']


def _read_field(rows, grid, row, selected, scale, section):
    physical = []
    for col in selected:
        item = grid.get((row, col))
        if item is None or item['row'] != row:
            raise ValueError('missing_or_inherited_current_cell')
        if not set(item['columns']).issubset(selected):
            raise ValueError('current_cell_crosses_period_boundary')
        if item['physical_cell'] not in physical:
            physical.append(item['physical_cell'])
    cells = _cells(rows[row])
    raw_value = norm(' '.join(cells[i] for i in physical))
    # exact_amount deliberately rejects blanks, dash tokens and percentages.
    try:
        value = exact_amount(raw_value)
    except (ValueError, ArithmeticError):
        raise ValueError('unavailable_or_ambiguous_current_value') from None
    if Decimal(value) < 0:
        raise ValueError('negative_balance_requires_review')
    return {
        'label': cells[0], 'row_index': row, 'raw_cells': cells,
        'raw_value': raw_value, 'value_reported': value, 'scale': scale,
        'section': section, 'selected_columns': list(selected),
        'selected_cells': physical,
    }


def _scope_notes(result, labels, cells, bounds):
    recognized = {label for aliases in FIELD_LABELS.values() for label in aliases}
    for row, label in enumerate(labels):
        if not label:
            continue
        if re.search(r'securities lending payable|securities loaned|loaned (?:securities|investments)', label):
            result['notes'].append({
                'code': 'securities_lending_scope_not_comparable',
                'row_index': row, 'label': cells[row][0],
                'treatment': 'excluded; no debt offset or liquid-asset adjustment inferred',
            })
            result['reasons'].append('securities_lending_scope_requires_review')
        if 'restricted cash' in label:
            result['notes'].append({
                'code': 'restricted_cash_not_added', 'row_index': row,
                'label': cells[row][0],
                'treatment': 'excluded; liquid-aggregate inclusion needs independent evidence',
            })
            result['reasons'].append('restricted_cash_scope_requires_review')
        if 'lease' in label:
            result['notes'].append({
                'code': 'lease_row_not_added', 'row_index': row, 'label': cells[row][0],
                'treatment': 'separately reported lease item is not a debt component here',
            })
        elif (bounds['current_liabilities'] < row < bounds['equity']
              and label not in recognized
              and re.search(r'\bdebt\b|\bborrowings?\b|\bcommercial paper\b|\bcredit facilit', label)):
            result['notes'].append({
                'code': 'unmapped_debt_row', 'row_index': row, 'label': cells[row][0],
                'treatment': 'not added; overlap and maturity scope require review',
            })
            result['reasons'].append('unmapped_debt_scope_requires_review')


def extract(html: str, event: dict) -> dict:
    """Return all five fields or explicit rejection; never repair missing data.

    event is one SOURCE_PLAN source entry. Amounts are decimal strings in the
    original scale. The two component arrays contain field keys, never totals.
    Missing/invalid fields remain None, including dash and blank values. The
    caller is responsible for SOURCE_PLAN hashes and issuer/publication proof.
    """
    if not isinstance(event, dict):
        event = {}
    result = _empty(event)
    year, period_end = event.get('fiscal_year'), event.get('fiscal_period_end')
    if (event.get('ticker') != 'MSFT' or type(year) is not int
            or not 2015 <= year <= 2026 or event.get('fiscal_quarter') != 4):
        result['reasons'].append('event_outside_msft_q4_source_scope')
        return result
    try:
        parsed_end = date.fromisoformat(period_end)
    except (TypeError, ValueError):
        result['reasons'].append('invalid_event_period_end')
        return result
    if parsed_end.isoformat() != '%04d-06-30' % year:
        result['reasons'].append('msft_q4_fiscal_period_mismatch')
        return result
    if not isinstance(html, str) or not html.strip():
        result['reasons'].append('empty_or_nontext_html')
        return result

    soup = BeautifulSoup(html, 'html.parser')
    candidates = []
    for index, table in enumerate(soup.find_all('table')):
        context = statement_context(table)
        if not re.search(r'\bBALANCE SHEETS\b', context, re.I):
            continue
        try:
            validate_statement(context, norm(table.get_text(' ', strip=True)),
                               'balance_sheet', 'MSFT')
        except ValueError:
            continue
        if re.search(r'CASH FLOWS?|INCOME STATEMENTS|STATEMENTS OF OPERATIONS', context, re.I):
            continue
        candidates.append((index, table, context))
    if len(candidates) != 1:
        result['reasons'].append('missing_or_ambiguous_gaap_balance_sheet')
        return result
    index, table, context = candidates[0]
    result['statement']['table_index'] = index
    try:
        rows, grid, _ = table_grid(table)
        cells = [_cells(row) for row in rows]
        labels = [_label(row[0]) if row else '' for row in cells]
        bounds = _sections(labels)
        stop = bounds['assets']
        header_rows = cells[:stop]
        result['statement']['header_rows'] = header_rows
        selected = _columns(grid, stop, period_end)
        result['statement']['selected_period'] = period_end
        # Unit evidence is restricted to the formal title and header cells.
        snippets = [context] + [text for row in header_rows for text in row]
        unit_snippets = [text for text in snippets
                         if re.search(r'\bin\s+(?:millions|thousands)\b', text, re.I)]
        scale_evidence = norm(' '.join(dict.fromkeys(unit_snippets)))
        scale = scale_from(scale_evidence)
        # Reject a conflicting unit declaration anywhere in this same table.
        if scale_from(context + ' ' + table.get_text(' ', strip=True)) != scale:
            raise ValueError('conflicting_statement_units')
        full_text = norm(context + ' ' + table.get_text(' ', strip=True))
        # The same dollar glyph can denote CAD/AUD and other currencies. Any
        # competing currency explicitly present in this statement requires
        # review, including a foreign-currency footnote or a mixed USD/CAD unit.
        # Do not search unrelated narrative or other tables on the release page.
        competing_currency = re.search(
            r'\b(?:CAD|AUD|NZD|HKD|SGD|TWD|CNY|RMB|JPY|EUR|GBP|CHF|INR|KRW|'
            r'SEK|NOK|DKK|BRL|MXN|ZAR)\b|'
            r'\b(?:Canadian|Australian|New Zealand|Hong Kong|Singapore|Taiwan)\s+dollars?\b|'
            r'\b(?:CA?|AU?|NZ|HK|SG?|NT)\s*\$|'
            r'\b(?:euros?|yen|yuan|renminbi|pounds? sterling|Swiss francs?|rupees?)\b|'
            r'[€£¥₹₩]', full_text, re.I)
        if competing_currency:
            result['statement'].update(
                scale=scale, scale_evidence=scale_evidence,
                currency_symbol='$' if '$' in full_text else None,
                currency_status='explicit_non_usd_or_mixed')
            result['notes'].append({
                'code': 'explicit_competing_currency',
                'evidence': competing_currency.group(0),
                'treatment': 'rejected; no currency inference or conversion',
            })
            raise ValueError('explicit_non_usd_or_mixed_currency')
        if '$' not in full_text:
            raise ValueError('statement_dollar_symbol_missing')
        currency_status = ('explicit_usd' if re.search(
            r'\bUSD\b|\bU\.?S\.?\s+dollars?\b|\bUnited States dollars?\b|\bU\.?S\.?\s*\$',
            ' '.join(snippets), re.I) else 'dollar_symbol_only')
        result['statement'].update(scale=scale, scale_evidence=scale_evidence,
                                   currency_symbol='$', currency_status=currency_status)
        for key, aliases in FIELD_LABELS.items():
            matches = [row for row, label in enumerate(labels) if label in aliases]
            if len(matches) != 1:
                result['reasons'].append(key + ':missing_or_ambiguous_row')
                continue
            row = matches[0]
            section = FIELD_SECTIONS[key]
            if not _in_section(row, section, bounds):
                result['reasons'].append(key + ':wrong_statement_section')
                continue
            try:
                result['fields'][key] = _read_field(rows, grid, row, selected, scale, section)
            except ValueError as exc:
                result['reasons'].append(key + ':' + str(exc))
        _scope_notes(result, labels, cells, bounds)
    except (ValueError, IndexError, KeyError) as exc:
        result['reasons'].append('statement:' + str(exc))
        return result

    result['current_debt_components'] = [key for key in (
        'short_term_debt', 'current_portion_long_term_debt') if result['fields'][key] is not None]
    result['noncurrent_debt_components'] = (['long_term_debt']
        if result['fields']['long_term_debt'] is not None else [])
    result['notes'].append({
        'code': 'reported_aggregate_only',
        'treatment': 'Use the single cash/cash-equivalents/short-term-investments total; '
                     'no component addition, maturity completeness or unrestricted-cash claim.',
    })
    result['reasons'] = list(dict.fromkeys(result['reasons']))
    if not result['reasons'] and all(result['fields'].values()):
        result['status'] = 'candidate'
    return result
