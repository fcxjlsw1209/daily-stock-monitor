"""Three-event Apple SBC repair: unique syntax-only pypdf column partition.

The caller must freeze the original locator inventory before using this pure
parser. No file reads, event filtering, admission or financial arithmetic.
PDFium source-index, numeric-spacing and column checks remain frozen v4 code.
"""
from copy import deepcopy
import re
import stock_comp_source_v1 as old

base = old.base
pdf = old.pdf_repair
require = base.require
SELF = 'work/stock_comp_pdf_repair_v2.py'
TEST = 'work/test_stock_comp_pdf_repair_v2.py'
RUNNER = 'work/stock_comp_pdf_repair_v2_runner.py'
MAX_TOKENS = 64


def code_hashes(root):
    return dict(old.code_hashes(root), **{p: base.sha(root, p) for p in (SELF, TEST, RUNNER, 'work/test_cashflow_pdf_repair_v4.py')})


def pypdf_columns(tail, columns):
    """Partition contiguous tokens uniquely, without consulting PDFium/values.

    Only whitespace is removed within a candidate column. Signs, parentheses,
    commas and decimal points remain literal. Every possible boundary between
    the two frozen annual columns is considered; two valid boundaries fail,
    even if PDFium or a human could pick one. This is not a widest-gap heuristic.
    """
    require(type(tail) is str and type(columns) is int and columns == 2,
            'exact_two_annual_columns_required')
    tokens = tail.replace('/dollarsign', '$').replace('$', '').split()
    require(2 <= len(tokens) <= MAX_TOKENS, 'bounded_pypdf_token_count_required')
    require(all(re.fullmatch(r'[0-9(),.\-]+', token) for token in tokens),
            'nonmonetary_pypdf_character')
    candidates = []
    for boundary in range(1, len(tokens)):
        cells = [''.join(tokens[:boundary]), ''.join(tokens[boundary:])]
        try:
            for cell in cells:
                base.pdf.decimal_cell(cell)
        except ValueError:
            continue
        candidates.append((cells, boundary))
    require(len(candidates) == 1, 'unique_syntax_only_pypdf_column_partition_required')
    cells, boundary = candidates[0]
    return cells, dict(method='unique_contiguous_token_partition_by_exact_monetary_syntax',
                       original_token_count=len(tokens),
                       column_token_counts=[boundary, len(tokens)-boundary],
                       removed_characters='whitespace_only_after_legacy_currency_marker_handling',
                       numeric_characters_preserved=True, whitespace_rejoined=len(tokens) > columns,
                       candidate_partitions=1, maximum_tokens=MAX_TOKENS)


def extract_cell(raw, texts, loc, pdf_plan):
    require(loc.get('format') == 'pdf' and loc.get('row_label') in old.PDF_LABELS,
            'explicit_stock_compensation_pdf_row_required')
    require(len(loc['header_dates']) == 2 and loc['financial_column_index_0based'] == 0,
            'current_first_of_two_annual_columns_required')
    line = texts[loc['page_1based']-1].splitlines()[loc['line_1based']-1]
    match = base.pdf_row_match(line, loc['row_label'])
    require(match is not None, 'frozen_exact_row_changed')
    cells, proof = pypdf_columns(match[1], len(loc['header_dates']))
    _, pdfium = base.configure_pdf(pdf_plan)
    doc = pdfium.PdfDocument(raw)
    page = doc[loc['page_1based']-1]
    textpage = page.get_textpage()
    try:
        geometry = pdf.geometry_rows(pdf.physical_rows(textpage), loc, cells)
    finally:
        textpage.close(); page.close(); doc.close()
    chosen = cells[loc['financial_column_index_0based']]
    return dict(displayed_amount=chosen, value_decimal=base.pdf.decimal_cell(chosen),
                row_text_sha256=base.safe.digest(line.encode()), pdfium_geometry=geometry,
                pdfium_text_agrees=True, pypdf_column_reconstruction=proof)


def extract_current(raw, periods, frozen_inventory, pdf_plan):
    """Same result interface as frozen stock_comp_source_v1.extract_current."""
    require(type(raw) is bytes and raw.startswith(b'%PDF-'), 'original_pdf_bytes_required')
    current = old.inventory(raw, periods, pdf_plan)
    require(current == frozen_inventory, 'Frozen source/period/locator inventory changed')
    result = dict(field=old.FIELD, value=None, gaps=deepcopy(current['gaps']), candidate_only=True,
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
        parsed = extract_cell(raw, base.pdf_texts(raw, pdf_plan), loc, pdf_plan)
        # Full-row PDFium validation proves alignment; do not publish the
        # comparison amount or the raw monetary line as a second new field.
        geometry = deepcopy(parsed['pdfium_geometry'])
        for cell in geometry['pdfium_all_row_cells']:
            cell.pop('text', None)
        parsed['pdfium_geometry'] = geometry
        parsed['independent_visual_review_required'] = True
        result['value'] = dict(parsed, scale=loc['scale'], currency_literal=loc['currency_literal'],
                               currency_iso=None, period=deepcopy(periods['current']),
                               locator=deepcopy(loc), source_sha256=current['source_sha256'],
                               fact_admitted=False)
        if parsed['value_decimal'].startswith('-'):
            result['accounting_review_flags'].append(
                dict(code='negative_stock_compensation_addback_requires_review'))
    except (ValueError, IndexError, KeyError) as exc:
        result['gaps'].append(dict(reason='invalid_or_ambiguous_current_monetary_cell',
                                  error_type=type(exc).__name__, error_message=str(exc)))
    return result
