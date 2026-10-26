"""Write-once producer visual QA for ten already rendered WC194 cash-flow pages.

The observations below were read with view_image, not inferred from extraction.
This records source evidence only: no component sum, ratio, factor, or returns.
"""
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTRACTION = 'outputs/monthly_working_capital194_source/msft_aapl/EXTRACTION.json'
EXTRACTION_SHA = '6ff8c01f9fc3cffb6f47a3f7ceb4792d3647208f008aa1189139565292c5c727'
OUT = ROOT / 'outputs/monthly_working_capital194_source/msft_aapl/VISUAL_REVIEW.json'
LABELS = {'accounts_receivable': 'Accounts receivable, net',
          'inventories': 'Inventories', 'accounts_payable': 'Accounts payable'}
# Current/prior annual date, then AR, inventory, AP visible current/prior cells.
OBSERVED = {
    2016: ('2016-09-24', '2015-09-26', ('1,095','611'), ('217','(238)'), ('1,791','5,400')),
    2017: ('2017-09-30', '2016-09-24', ('(2,093)','527'), ('(2,723)','217'), ('9,618','1,837')),
    2018: ('2018-09-29', '2017-09-30', ('(5,322)','(2,093)'), ('828','(2,723)'), ('9,175','8,966')),
    2019: ('2019-09-28', '2018-09-29', ('245','(5,322)'), ('(289)','828'), ('(1,923)','9,175')),
    2020: ('2020-09-26', '2019-09-28', ('6,917','245'), ('(127)','(289)'), ('(4,062)','(1,923)')),
    2021: ('2021-09-25', '2020-09-26', ('(10,125)','6,917'), ('(2,642)','(127)'), ('12,326','(4,062)')),
    2022: ('2022-09-24', '2021-09-25', ('(1,823)','(10,125)'), ('1,484','(2,642)'), ('9,448','12,326')),
    2023: ('2023-09-30', '2022-09-24', ('(1,688)','(1,823)'), ('(1,618)','1,484'), ('(1,889)','9,448')),
    2024: ('2024-09-28', '2023-09-30', ('(3,788)','(1,688)'), ('(1,046)','(1,618)'), ('6,020','(1,889)')),
    2025: ('2025-09-27', '2024-09-28', ('(6,682)','(3,788)'), ('1,400','(1,046)'), ('902','6,020')),
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def local(path):
    value = Path(path)
    value = (value if value.is_absolute() else ROOT / value).resolve()
    require(value.is_relative_to(ROOT), 'Path outside local research root')
    require(not any(x in value.parts for x in ('OneDrive', 'CloudStorage')), 'Cloud path rejected')
    return value


def digest(path):
    return hashlib.sha256(local(path).read_bytes()).hexdigest()


def verify(pins):
    for path, expected in pins.items():
        require(digest(path) == expected, 'SHA mismatch: ' + path)


def signed_value(literal):
    value = literal.replace(',', '')
    return Decimal('-' + value[1:-1] if value.startswith('(') else value)


def review_record(record):
    year = record['fiscal_year']
    current, prior, *observed_fields = OBSERVED[year]
    require(record['symbol'] == 'AAPL' and record['extraction_verified'] is True,
            'Expected successfully extracted AAPL source')
    require(set(record['fields']) == set(LABELS), 'Exactly three components required')
    render = record['render']
    require(render['page_1based'] == 3, 'Reviewed page differs')
    require(digest(render['path']) == render['sha256'], 'Reviewed image changed')
    require(digest(record['source']['path']) == record['source']['sha256'], 'Original PDF changed')
    fields = {}
    for (key, label), pair in zip(LABELS.items(), observed_fields):
        field = record['fields'][key]
        loc = field['locator']
        require(field['row_label'] == loc['row_label'] == label, 'Operating contribution row differs')
        require(loc['statement_title'] == 'consolidated statements of cash flows', 'Statement title differs')
        require(loc['unit_literal'] == '(In millions)' and field['scale'] == '1000000', 'Units differ')
        require(loc['financial_column_index_0based'] == 0 and loc['page_1based'] == 3, 'Annual column/page differs')
        require(loc['header_dates'] == [current, prior], 'Annual dates differ')
        require(loc['period_end'] == field['period']['period_end'] == current, 'Current annual period differs')
        require(field['all_row_cells'] == list(pair), 'Visible current/comparative cells differ')
        require(field['normalized_monetary_literal'] == pair[0], 'Current signed literal differs')
        require(Decimal(field['signed_value_decimal']) == signed_value(pair[0]), 'Signed current value differs')
        require(field['pdfium_text_agrees'] is True and field['render'] == render, 'Geometry/image binding missing')
        require(field['fact_admitted'] is False, 'Unexpected field admission')
        fields[key] = dict(observed_row_label=label, observed_current_signed_literal=pair[0],
                           observed_comparative_signed_literal=pair[1],
                           annual_column_index_0based=0, page_1based=3,
                           source_row_text_sha256=field['row_text_sha256'],
                           source_locator=loc, original_signed_literal=field['signed_literal'],
                           layout_whitespace_reconciled=field.get('layout_whitespace_reconciled', False),
                           source_glyph_geometry_matches=True, extracted_value_matches_visual=True)
    require(record['fact_admitted'] is False and record['strategy_admitted'] is False, 'Unexpected record admission')
    return dict(event_id=record['event_id'], fiscal_year=year, source=record['source'], render=render,
                observed_statement='Consolidated Statements of Cash Flows',
                observed_annual_header='Twelve Months Ended', observed_unit='(In millions)',
                observed_current_date=current, observed_comparative_date=prior, fields=fields,
                pure_trade_ar_separate_from_vendor_nontrade=True,
                inventories_and_ap_distinct_from_other_assets_and_liabilities=True,
                all_three_rows_within_operating_cashflow_changes=True,
                legible_unclipped_annual_columns=True, producer_visual_review_completed=True,
                independent_review=False, inherited_admitted=record['inherited_admitted'],
                inherited_status=record['status'], preserved_hold_reasons=record['hold_reasons'],
                fact_admitted=False, strategy_admitted=False)


def main():
    require(not OUT.exists(), 'Write-once visual review already exists')
    require(digest(EXTRACTION) == EXTRACTION_SHA, 'Extraction SHA differs')
    data = json.loads(local(EXTRACTION).read_text())
    pins = {EXTRACTION: EXTRACTION_SHA, **data['source_hashes'], **data['code_hashes'],
            **data['pdf_runtime_hashes']}
    verify(pins)
    require((data['report_count'], data['extraction_verified_count'], data['verified_count'], data['held_count'])
            == (22,22,20,2), 'Complete source census differs')
    apple = [r for r in data['records'] if r['symbol'] == 'AAPL']
    require(len(apple) == 10 and {r['fiscal_year'] for r in apple} == set(OBSERVED), 'Exact ten-report scope required')
    records = [review_record(r) for r in apple]
    image_hashes = {r['render']['path']: r['render']['sha256'] for r in records}
    verify(pins); verify(image_hashes)
    result = dict(schema='working_capital194_msft_aapl_producer_visual_review_v1',
                  generated_utc=datetime.now(timezone.utc).isoformat(),
                  extraction_path=EXTRACTION, extraction_sha256=EXTRACTION_SHA,
                  review_kind='producer_visual_inspection_of_final_rendered_pages',
                  tool='view_image', passed=True, report_count=10, current_cells_reviewed=30,
                  comparative_cells_reviewed=30, records=records, image_hashes=image_hashes,
                  source_hashes=pins, review_code_sha256=digest(__file__),
                  visual_review_pending=False, independent_review=False,
                  immutable_extraction_pending_flag_preserved=True, original_holds_preserved=True,
                  component_sums=0, financial_ratios=0, factors=0, model_fits=0, orders=0,
                  returns=0, fact_admissions=0, strategy_admissions=0, network_requests=0,
                  limits=['This supplements, and does not rewrite, the immutable extraction pending flag.',
                          'Producer visual QA is not the separate independent source audit.',
                          'Every comparative cell is retained as printed in that report, without cross-report reconciliation.',
                          'Original source holds, publication and security-identity limitations remain unchanged.'])
    with OUT.open('x') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(path=str(OUT.relative_to(ROOT)), sha256=digest(OUT),
                          passed=True, reviewed_pages=10, reviewed_current_cells=30)))


if __name__ == '__main__':
    main()
