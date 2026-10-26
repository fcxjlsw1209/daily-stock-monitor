"""Single-field, source-only stock-compensation reconciliation extraction.

No event filtering, admission, source-path reads, financial arithmetic or CLI.
The caller freezes inventory output before calling extract_current. Periods are
exactly the existing cashflow PLAN's current/comparison period dictionaries;
pdf_plan is the existing PDF extraction PLAN with its pinned runtime fields.
"""
from copy import deepcopy
from pathlib import Path
import re
from bs4 import BeautifulSoup
import cashflow_source_preflight_v1 as base
import cashflow_html_repair_v2 as html_repair
import cashflow_pdf_repair_v4 as pdf_repair

SELF = 'work/stock_comp_source_v1.py'
TEST = 'work/test_stock_comp_source_v1.py'
FIELD = 'stock_compensation_addback'
require = base.require
LABEL = re.compile(r'^(?:stock|share)[\s\-\u2010-\u2015]+based\s+compensation(?:\s+expense)?$', re.I)
RELATED = re.compile(r'(?:stock|share)[\s\-\u2010-\u2015]*based\s*compensation', re.I)
PDF_LABELS = tuple(a+s+'based compensation'+b for a in ('Stock', 'Share')
                   for s in ('-', ' ', '\u2010', '\u2011', '\u2012', '\u2013', '\u2014', '\u2015')
                   for b in ('', ' expense'))
LOCAL_DEPENDENCIES = (
    SELF, TEST, base.SELF, html_repair.SELF, pdf_repair.SELF,
    'work/after141_amd_source_extract.py', 'work/after141_aapl_source_extract.py',
    'work/profitability141_issuer_statements.py', 'work/same_report144_admission.py')


def code_hashes(root):
    """All local code imported by this adapter; caller also binds runtime pins."""
    return {path: base.sha(Path(root), path) for path in LOCAL_DEPENDENCIES}


def _periods(periods):
    require(type(periods) is dict and set(periods) == {'current', 'comparison'},
            'Existing current/comparison period dictionaries required')
    for side in ('current', 'comparison'):
        p = periods[side]
        require(type(p) is dict and set(p) == {'period_start', 'period_end', 'days_inclusive'},
                'Exact existing period fields required')
        end = base.safe.day(p['period_end'])
        if p['period_start'] is None:
            require(p['days_inclusive'] is None, 'Missing start requires missing duration')
        else:
            start = base.safe.day(p['period_start'])
            require(type(p['days_inclusive']) is int and
                    (end-start).days+1 == p['days_inclusive'] and 330 <= p['days_inclusive'] <= 400,
                    'Exact full-year period required')
    require(periods['comparison']['period_end'] < periods['current']['period_end'],
            'Comparison period must precede current period')
    return periods['current']['period_start'] is not None


def _pair(observed, fmt):
    ni = observed['candidates']['current']['net_income']
    cfo = observed['candidates']['current']['operating_cash_flow']
    if len(ni) != 1 or len(cfo) != 1:
        return None, 'unique_formal_current_annual_NI_and_CFO_required'
    ni, cfo = ni[0], cfo[0]
    keys = ('table_index_zero_based', 'statement_segment', 'logical_columns') if fmt == 'html' else (
        'page_1based', 'header_dates', 'financial_column_index_0based')
    keys += ('scale', 'currency_literal', 'period_end', 'header_literals')
    if any(ni.get(k) != cfo.get(k) for k in keys):
        return None, 'NI_and_CFO_must_share_statement_columns_and_units'
    index = 'row_index_zero_based' if fmt == 'html' else 'line_1based'
    if ni[index] >= cfo[index]:
        return None, 'indirect_start_must_precede_operating_total'
    if ni.get('ni_scope') != 'indirect_method_consolidated_start':
        return None, 'consolidated_indirect_start_required'
    if cfo.get('currency_literal') != '$':
        return None, 'explicit_same_report_currency_required'
    return (ni, cfo), None


def _html_inventory(raw, periods):
    observed = html_repair.inventory(raw, periods)
    pair, reason = _pair(observed, 'html')
    result = dict(candidates=[], gaps=[], accounting_review_flags=observed['accounting_review_flags'])
    if reason:
        result['gaps'].append(dict(reason=reason)); return result
    ni, cfo = pair
    soup = BeautifulSoup(raw, 'html.parser')
    for node in soup(['script', 'style']): node.decompose()
    table = soup.find_all('table')[cfo['table_index_zero_based']]
    _, grid, physical = base.html.table_grid(table)
    lower, upper = ni['row_index_zero_based'], cfo['row_index_zero_based']
    adjustment_rows = sorted({r for (r, _), cell in physical.items()
                              if lower < r < upper and base.ADJUSTMENT.search(base.norm(cell['text']))})
    targets = []
    for (ri, pi), cell in physical.items():
        if not lower < ri < upper: continue
        label = base.norm(cell['text'])
        if not RELATED.search(label): continue
        if not LABEL.fullmatch(label):
            result['gaps'].append(dict(reason='combined_or_unrecognized_compensation_label',
                                      table_index_zero_based=cfo['table_index_zero_based'],
                                      row_index_zero_based=ri))
            continue
        targets.append((ri, pi, label))
    for ri, pi, label in targets:
        if not any(r < ri for r in adjustment_rows):
            result['gaps'].append(dict(reason='compensation_not_after_indirect_adjustment_heading',
                                      row_index_zero_based=ri)); continue
        columns = cfo['logical_columns']
        if any((ri, col) not in grid for col in columns):
            result['gaps'].append(dict(reason='missing_current_annual_cells', row_index_zero_based=ri)); continue
        selected = sorted({grid[(ri, col)]['physical_cell'] for col in columns})
        if any(re.search('[A-Za-z]', physical[(ri, p)]['text']) or
               physical[(ri, p)]['rowspan'] != 1 or
               not set(physical[(ri, p)]['columns']).issubset(columns) for p in selected):
            result['gaps'].append(dict(reason='compensation_cell_crosses_period_or_contains_text',
                                      row_index_zero_based=ri)); continue
        loc = deepcopy(cfo)
        loc.update(row_index_zero_based=ri, row_label=label,
                   physical_cell_indices_zero_based=selected,
                   label_cell_sha256=physical[(ri, pi)]['parsed_cell_markup_sha256'],
                   indirect_start_row_zero_based=lower, operating_total_row_zero_based=upper,
                   adjustment_rows_zero_based=adjustment_rows,
                   field_scope='explicit_component_of_consolidated_indirect_operating_reconciliation')
        loc.pop('ni_scope', None)
        result['candidates'].append(loc)
    if not targets and not result['gaps']:
        result['gaps'].append(dict(reason='missing_explicit_stock_compensation_row'))
    if len(result['candidates']) > 1:
        result['gaps'].append(dict(reason='multiple_stock_compensation_rows'))
    return result


def _pdf_inventory(texts, periods):
    observed = pdf_repair.inventory(texts, periods)
    pair, reason = _pair(observed, 'pdf')
    result = dict(candidates=[], gaps=[], accounting_review_flags=observed['accounting_review_flags'])
    if reason:
        result['gaps'].append(dict(reason=reason)); return result
    ni, cfo = pair
    lines = texts[cfo['page_1based']-1].splitlines()
    lower, upper = ni['line_1based'], cfo['line_1based']
    adjustment_rows = [i+1 for i, line in enumerate(lines)
                       if lower < i+1 < upper and 'adjustmentstoreconcile' in pdf_repair.compact(line)]
    for i in range(lower, upper-1):
        line = lines[i]
        if not RELATED.search(pdf_repair.compact(line)): continue
        matches = [label for label in PDF_LABELS if base.pdf_row_match(line, label)]
        if len(matches) != 1:
            result['gaps'].append(dict(reason='combined_or_unrecognized_compensation_label',
                                      page_1based=cfo['page_1based'], line_1based=i+1)); continue
        if not any(r < i+1 for r in adjustment_rows):
            result['gaps'].append(dict(reason='compensation_not_after_indirect_adjustment_heading',
                                      line_1based=i+1)); continue
        loc = deepcopy(cfo)
        loc.update(line_1based=i+1, row_label=matches[0],
                   indirect_start_line_1based=lower, operating_total_line_1based=upper,
                   adjustment_lines_1based=adjustment_rows,
                   field_scope='explicit_component_of_consolidated_indirect_operating_reconciliation')
        loc.pop('ni_scope', None)
        result['candidates'].append(loc)
    if not result['candidates'] and not result['gaps']:
        result['gaps'].append(dict(reason='missing_explicit_stock_compensation_row'))
    if len(result['candidates']) > 1:
        result['gaps'].append(dict(reason='multiple_stock_compensation_rows'))
    return result


def inventory(raw, periods, pdf_plan=None):
    """Metadata only: hashes, headers, coordinates, labels, flags; no amounts."""
    require(type(raw) is bytes and bool(raw), 'Nonempty original response bytes required')
    complete = _periods(periods)
    fmt = 'pdf' if raw.startswith(b'%PDF-') else 'html'
    result = dict(source_sha256=base.safe.digest(raw), format=fmt,
                  current_period=deepcopy(periods['current']), candidates=[], gaps=[],
                  accounting_review_flags=[], new_field_independent_review_required=True)
    if not complete:
        result['gaps'].append(dict(reason='missing_current_annual_duration')); return result
    if fmt == 'pdf':
        require(type(pdf_plan) is dict, 'Pinned existing PDF runtime plan required')
        found = _pdf_inventory(base.pdf_texts(raw, pdf_plan), periods)
    else:
        found = _html_inventory(raw, periods)
    result.update(found)
    return result


def _pdf_value(raw, texts, loc, pdf_plan):
    parsed = pdf_repair.extract_cell(raw, texts, loc, pdf_plan)
    geometry = deepcopy(parsed['pdfium_geometry'])
    # The reused verifier checks the full row to prove column alignment. Only
    # the current value is emitted: comparison numerals and row text stay out.
    for cell in geometry['pdfium_all_row_cells']:
        cell.pop('text', None)
    return dict(displayed_amount=parsed['displayed_amount'], value_decimal=parsed['value_decimal'],
                row_text_sha256=parsed['row_text_sha256'], pdfium_geometry=geometry,
                pdfium_text_agrees=True, independent_visual_review_required=True)


def extract_current(raw, periods, frozen_inventory, pdf_plan=None):
    """Extract at most one current signed decimal after frozen-metadata parity.

    The root-owned runner preserves all44 events and their13 prior holds, binds
    source/receipt/availability, and obtains any later admission independently.
    This function never changes or interprets an old event-admission decision.
    """
    current = inventory(raw, periods, pdf_plan)
    require(current == frozen_inventory, 'Frozen source/period/locator inventory changed')
    result = dict(field=FIELD, value=None, gaps=deepcopy(current['gaps']), candidate_only=True,
                  fact_admitted=False, model_admitted=False, strategy_admitted=False,
                  accounting_review_flags=deepcopy(current['accounting_review_flags']),
                  first_public_version_verified=False, complete_correction_history_verified=False,
                  certified_daily_security_master=False)
    if result['gaps'] or len(current['candidates']) != 1:
        if not result['gaps']:
            result['gaps'].append(dict(reason='unique_stock_compensation_candidate_required'))
        return result
    loc = current['candidates'][0]
    try:
        parsed = (_pdf_value(raw, base.pdf_texts(raw, pdf_plan), loc, pdf_plan)
                  if current['format'] == 'pdf' else html_repair.extract(raw, loc))
        result['value'] = dict(parsed, scale=loc['scale'], currency_literal=loc['currency_literal'],
                               currency_iso=None, period=deepcopy(periods['current']),
                               locator=deepcopy(loc), source_sha256=current['source_sha256'],
                               fact_admitted=False)
        if parsed['value_decimal'].startswith('-'):
            result['accounting_review_flags'].append(
                dict(code='negative_stock_compensation_addback_requires_review'))
    except (ValueError, IndexError, KeyError) as exc:
        result['gaps'].append(dict(reason='invalid_or_ambiguous_current_monetary_cell',
                                  error_type=type(exc).__name__))
    return result
