"""Pure NVDA Q4 balance-sheet source extraction; no I/O or derived ratios.

Amounts remain exact displayed-unit decimal strings. A candidate is a source
record awaiting independent review, not admission of a financial feature.
"""
from datetime import date
from decimal import Decimal
import re

from bs4 import BeautifulSoup
from profitability141_issuer_statements import (
    exact_amount, header_dates, headers_for, norm, scale_from, table_grid,
)

KEYS = ('liquid_financial_assets', 'short_term_debt',
        'current_portion_long_term_debt', 'long_term_debt', 'total_assets')
_FINANCE = re.compile(r'\b(?:debt|borrowings?|commercial paper|notes payable|lease)\b', re.I)


def _label(text):
    return norm(text).lower().replace('\u2010', '-').replace('\u2011', '-').replace('\u2013', '-').replace('\u2014', '-').rstrip(':').strip()


def _row_cells(row):
    return [norm(c.get_text(' ', strip=True)) for c in row.find_all(['td', 'th'], recursive=False)]


def _row_label(cells):
    words = [c for c in cells if re.search('[A-Za-z]', c)]
    # The leftmost textual cell is the account label; a later N/A amount
    # remains available to the amount parser rather than hiding the row.
    return words[0] if words else None


def _matches(key, label):
    label = _label(label or '')
    if key == 'liquid_financial_assets':
        return bool(re.fullmatch(r'cash,? cash equivalents,? and marketable securities', label))
    if key == 'short_term_debt':
        return label in ('short-term debt', 'convertible short-term debt')
    if key == 'current_portion_long_term_debt':
        return bool(re.fullmatch(r'current (?:portion|maturities) of long-term debt', label))
    return label == {'long_term_debt': 'long-term debt', 'total_assets': 'total assets'}.get(key)


class _Reject(ValueError):
    pass


def _need(condition, reason):
    if not condition:
        raise _Reject(reason)


def _unique(labels, literal):
    rows = [i for i, text in enumerate(labels) if _label(text or '') == literal]
    _need(len(rows) == 1, 'missing_or_ambiguous_section:' + literal)
    return rows[0]


def _prepare_table(table, table_index, end):
    rows, grid, physical = table_grid(table)
    raw = [_row_cells(row) for row in rows]
    labels = [_row_label(cells) for cells in raw]
    ca = _unique(labels, 'current assets')
    tca = _unique(labels, 'total current assets')
    ta = _unique(labels, 'total assets')
    cl = _unique(labels, 'current liabilities')
    tcl = _unique(labels, 'total current liabilities')
    _need(ca < tca < ta < cl < tcl, 'balance_sheet_section_order')
    assets = [i for i, value in enumerate(labels[:ca]) if _label(value or '') == 'assets']
    _need(len(assets) <= 1, 'ambiguous_assets_header')
    stop = assets[0] if assets else ca
    header_rows = raw[:stop]
    header_text = ' '.join(' '.join(row) for row in header_rows)
    caption = table.find('caption', recursive=False)
    context = norm((caption.get_text(' ', strip=True) if caption else '') + ' ' + header_text)
    _need(re.search(r'\b(?:condensed )?consolidated balance sheets\b', context, re.I), 'formal_consolidated_balance_sheet_title_missing')
    _need(re.search(r'\bNVIDIA\b', context, re.I), 'issuer_title_missing')
    text = norm(table.get_text(' ', strip=True))
    _need(not re.search(r'non[ -]?gaap|reconciliation|as adjusted', text, re.I), 'non_gaap_or_adjusted_table')
    scale = scale_from(context)
    evidence = [cell for row in header_rows for cell in row if re.search(r'\bin\s+(?:millions|thousands)\b', cell, re.I)]
    if caption and re.search(r'\bin\s+(?:millions|thousands)\b', caption.get_text(' ', strip=True), re.I):
        evidence.insert(0, norm(caption.get_text(' ', strip=True)))
    _need('$' in text, 'dollar_symbol_not_evidenced')
    _need(not re.search(r'\b(?:CAD|AUD|HKD|SGD|Canadian dollars|Australian dollars)\b|[€£¥]', context, re.I), 'ambiguous_currency')
    currency = 'explicit_usd' if re.search(r'\b(?:USD|U\.?S\.? dollars|United States dollars)\b', context, re.I) else 'dollar_symbol_only'
    columns = sorted({c for r, c in grid if r < stop})
    dates = sorted({d for col in columns for d in header_dates(headers_for(grid, stop, col, stop))})
    _need(dates and dates[-1] == end, 'current_period_end_not_latest_header_date')
    current_cols = [col for col in columns if header_dates(headers_for(grid, stop, col, stop)) == [end]]
    _need(current_cols, 'current_period_column_not_unique_date')
    boundaries = [i for i in range(tcl + 1, len(labels)) if _label(labels[i] or '') == 'total liabilities' or
                  re.fullmatch(r'(?:stockholders|shareholders)[\u2019\u0027]? equity', _label(labels[i] or ''))]
    _need(boundaries, 'noncurrent_liability_boundary_missing')
    boundary = min(boundaries)
    sections = {}
    for i in range(len(rows)):
        sections[i] = ('current_assets' if ca < i < tca else
                       'all_assets' if i == ta else
                       'current_liabilities' if cl < i < tcl else
                       'noncurrent_liabilities' if tcl < i < boundary else 'outside_selected_sections')
    statement = dict(table_index=table_index, header_rows=header_rows, selected_period=end,
                     scale=scale, scale_evidence=' | '.join(dict.fromkeys(evidence)),
                     currency_symbol='$', currency_status=currency)
    return rows, grid, physical, raw, labels, stop, sections, statement


def _field(key, ri, grid, physical, raw, labels, stop, section, statement):
    end = statement['selected_period']
    candidates = []
    for (row, _), item in physical.items():
        if row != ri:
            continue
        dates_by_col = [header_dates(headers_for(grid, ri, col, stop)) for col in item['columns']]
        if not dates_by_col or not any(d == [end] for d in dates_by_col):
            continue
        _need(all(d == [end] for d in dates_by_col), 'amount_cell_spans_different_periods:' + key)
        value = item['text']
        if value in ('', '$'):
            continue
        candidates.append(item)
    _need(len(candidates) == 1, 'missing_or_ambiguous_current_amount:' + key)
    cell = candidates[0]
    try:
        value = exact_amount(cell['text'])
    except ValueError:
        value = None
    return dict(label=labels[ri], row_index=ri, raw_cells=raw[ri], raw_value=cell['text'],
                value_reported=value, scale=statement['scale'], section=section,
                selected_cells=[cell['physical_cell']], selected_columns=cell['columns'])


def extract(html: str, event: dict) -> dict:
    """Read supplied HTML only; retain all missing/ambiguous source boundaries."""
    metadata = event if isinstance(event, dict) else {}
    result = dict(ticker=metadata.get('ticker'), fiscal_year=metadata.get('fiscal_year'),
                  fiscal_period_end=metadata.get('fiscal_period_end'), status='rejected', reasons=[],
                  statement=None, fields=dict.fromkeys(KEYS), current_debt_components=[],
                  noncurrent_debt_components=[], notes=[])
    try:
        _need(isinstance(html, str) and isinstance(event, dict), 'invalid_input')
        _need(event.get('ticker') == 'NVDA' and type(event.get('fiscal_year')) is int and
              2015 <= event['fiscal_year'] <= 2026 and event.get('fiscal_quarter') == 4, 'outside_fixed_nvda_q4_scope')
        end = event['fiscal_period_end']
        _need(type(end) is str and date.fromisoformat(end).isoformat() == end and
              date.fromisoformat(end).year == event['fiscal_year'], 'invalid_fiscal_period_end')
        soup = BeautifulSoup(html, 'html.parser')
        candidates = []
        for ti, table in enumerate(soup.find_all('table')):
            if table.find('table') is not None:
                continue  # A layout parent cannot provide independent table evidence.
            if re.search(r'\b(?:condensed )?consolidated balance sheets\b', norm(table.get_text(' ', strip=True)), re.I):
                candidates.append((ti, table))
        _need(len(candidates) == 1, 'missing_or_ambiguous_balance_sheet')
        ti, table = candidates[0]
        rows, grid, physical, raw, labels, stop, sections, statement = _prepare_table(table, ti, end)
        result['statement'] = statement
        wanted_sections = dict(liquid_financial_assets='current_assets', short_term_debt='current_liabilities',
                               current_portion_long_term_debt='current_liabilities',
                               long_term_debt='noncurrent_liabilities', total_assets='all_assets')
        used = set()
        for key in KEYS:
            matches = [i for i, label in enumerate(labels) if _matches(key, label)]
            if not matches:
                if key != 'current_portion_long_term_debt':
                    result['reasons'].append('missing_field:' + key)
                continue
            if len(matches) != 1 or sections[matches[0]] != wanted_sections[key]:
                result['reasons'].append('ambiguous_or_wrong_section:' + key)
                continue
            ri = matches[0]
            try:
                item = _field(key, ri, grid, physical, raw, labels, stop, sections[ri], statement)
                result['fields'][key] = item
                used.add(ri)
                if item['value_reported'] is None:
                    result['reasons'].append('unavailable_amount:' + key)
                elif Decimal(item['value_reported']) < 0:
                    result['reasons'].append('negative_balance_requires_review:' + key)
            except _Reject as exc:
                result['reasons'].append(str(exc))
        if any(_matches('current_portion_long_term_debt', label) for label in labels):
            result['reasons'].append('current_portion_overlap_requires_review')
        for ri, label in enumerate(labels):
            if not label or not _FINANCE.search(label) or ri in used:
                continue
            section = sections[ri]
            result['notes'].append('Excluded financing row %d (%s): %s' % (ri, section, label))
            if section in ('current_liabilities', 'noncurrent_liabilities') and 'lease' not in _label(label):
                result['reasons'].append('unclassified_financing_row:%d' % ri)
        if result['fields']['short_term_debt'] is not None:
            result['current_debt_components'] = ['short_term_debt']
            if _label(result['fields']['short_term_debt']['label']) == 'convertible short-term debt':
                result['notes'].append(
                    'Convertible short-term debt is the reported current carrying amount; '
                    'it is not certified contractual principal due within one year.')
        if result['fields']['long_term_debt'] is not None:
            result['noncurrent_debt_components'] = ['long_term_debt']
        result['notes'].extend([
            'Reported current liquid-financial-assets aggregate; unrestricted availability is not certified.',
            'Reported debt carrying amounts only; leases and conversion obligations are not silently added.',
            'No missing amount is replaced with zero; no ratios, scores or admission decisions are computed.',
        ])
        if not result['reasons']:
            result['status'] = 'candidate'
    except (ValueError, KeyError, TypeError) as exc:
        result['reasons'].append(str(exc))
    return result
