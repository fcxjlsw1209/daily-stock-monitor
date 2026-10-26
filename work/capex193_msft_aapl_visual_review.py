"""Record the producer's completed visual review of all ten CAPEX193 PDF pages.

This is a separate write-once review receipt, not independent source admission.
The observations were read from the final PNGs with the image-viewing tool.
No source extraction, ratio, factor, model or portfolio calculation is performed.
"""
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTRACTION = 'outputs/monthly_capex193_source/msft_aapl/EXTRACTION.json'
EXTRACTION_SHA = '288f1af295a25cadbf2eb26437e53513382942fc50db46997164e50b3edebf2d'
OUT = ROOT / 'outputs/monthly_capex193_source/msft_aapl/VISUAL_REVIEW.json'
LABEL = 'Payments for acquisition of property, plant and equipment'
# Current annual date, comparative annual date, visible current and prior literals.
OBSERVED = {
    2016: ('2016-09-24', '2015-09-26', '(12,734)', '(11,247)'),
    2017: ('2017-09-30', '2016-09-24', '(12,451)', '(12,734)'),
    2018: ('2018-09-29', '2017-09-30', '(13,313)', '(12,451)'),
    2019: ('2019-09-28', '2018-09-29', '(10,495)', '(13,313)'),
    2020: ('2020-09-26', '2019-09-28', '(7,309)', '(10,495)'),
    2021: ('2021-09-25', '2020-09-26', '(11,085)', '(7,309)'),
    2022: ('2022-09-24', '2021-09-25', '(10,708)', '(11,085)'),
    2023: ('2023-09-30', '2022-09-24', '(10,959)', '(10,708)'),
    2024: ('2024-09-28', '2023-09-30', '(9,447)', '(10,959)'),
    2025: ('2025-09-27', '2024-09-28', '(12,715)', '(9,447)'),
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
    for path, sha in pins.items():
        require(digest(path) == sha, 'SHA mismatch: ' + path)


def review_record(record):
    year = record['fiscal_year']
    current, prior, current_literal, prior_literal = OBSERVED[year]
    field = record['field']; locator = field['locator']; render = field['render']
    require(record['symbol'] == 'AAPL' and record['extraction_verified'] is True,
            'Expected successfully extracted AAPL source')
    require(field['row_label'] == LABEL and locator['row_label'] == LABEL, 'PP&E row label differs')
    require(locator['statement_title'] == 'consolidated statements of cash flows', 'Statement title differs')
    require(locator['unit_literal'] == '(In millions)' and field['scale'] == '1000000', 'Monetary units differ')
    require(locator['financial_column_index_0based'] == 0, 'Current annual column differs')
    require(locator['page_1based'] == render['page_1based'] == 3, 'Reviewed page differs')
    require(locator['header_dates'] == [current, prior], 'Annual column dates differ')
    require(locator['period_end'] == field['period']['period_end'] == current, 'Current annual period differs')
    require(field['all_row_cells'] == [current_literal, prior_literal], 'Visible row literals differ')
    require(field['normalized_monetary_literal'] == current_literal, 'Current signed literal differs')
    amount = Decimal(current_literal[1:-1].replace(',', ''))
    require(Decimal(field['positive_outflow_decimal']) == amount, 'Current PP&E value differs')
    require(Decimal(field['signed_value_decimal']) == -amount, 'Outflow sign differs')
    require(field['pdfium_text_agrees'] is True, 'Text/geometry agreement absent')
    require(record['fact_admitted'] is False and record['strategy_admitted'] is False, 'Unexpected admission')
    require(digest(render['path']) == render['sha256'], 'Reviewed image changed')
    require(digest(record['source']['path']) == record['source']['sha256'], 'Original PDF changed')
    return dict(event_id=record['event_id'], fiscal_year=year, source=record['source'], render=render,
                observed_statement='Consolidated Statements of Cash Flows',
                observed_annual_header='Twelve Months Ended', observed_unit='(In millions)',
                observed_row_label=LABEL, observed_current_date=current, observed_prior_date=prior,
                observed_current_signed_literal=current_literal, observed_prior_signed_literal=prior_literal,
                observed_current_annual_column_index_0based=0, page_1based=3,
                pure_ppe_row_distinct_from_acquisitions_and_intangibles=True,
                legible_unclipped_annual_columns=True, extracted_value_matches_visual=True,
                producer_visual_review_completed=True, independent_review=False,
                inherited_admitted=record['inherited_admitted'], inherited_status=record['status'],
                preserved_hold_reasons=record['hold_reasons'], fact_admitted=False, strategy_admitted=False)


def main():
    require(not OUT.exists(), 'Write-once visual review already exists')
    require(digest(EXTRACTION) == EXTRACTION_SHA, 'Extraction SHA differs')
    data = json.loads(local(EXTRACTION).read_text())
    pins = {EXTRACTION: EXTRACTION_SHA, **data['source_hashes'], **data['code_hashes'],
            **data['pdf_runtime_hashes']}
    verify(pins)
    apple = [r for r in data['records'] if r['symbol'] == 'AAPL']
    require(len(apple) == 10 and {r['fiscal_year'] for r in apple} == set(OBSERVED), 'Exact ten-report scope required')
    records = [review_record(r) for r in apple]
    image_hashes = {r['render']['path']: r['render']['sha256'] for r in records}
    verify(pins); verify(image_hashes)
    result = dict(schema='capex193_msft_aapl_producer_visual_review_v1',
                  generated_utc=datetime.now(timezone.utc).isoformat(),
                  extraction_path=EXTRACTION, extraction_sha256=EXTRACTION_SHA,
                  review_kind='producer_visual_inspection_of_final_rendered_pages',
                  tool='view_image', passed=True, report_count=10, records=records,
                  image_hashes=image_hashes, source_hashes=pins,
                  review_code_sha256=digest(__file__),
                  visual_review_pending=False, independent_review=False,
                  immutable_extraction_pending_flag_preserved=True,
                  original_holds_preserved=True, fact_admissions=0, strategy_admissions=0,
                  financial_ratios=0, factors=0, model_fits=0, returns=0, network_requests=0,
                  limits=['This supplements the immutable extraction pending flag; it does not rewrite extraction.',
                          'Producer visual review is not the separate independent source audit.',
                          'Original source holds, publication and security-identity limitations remain unchanged.'])
    with OUT.open('x') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(path=str(OUT.relative_to(ROOT)), sha256=digest(OUT), passed=True, reviewed_pages=10)))


if __name__ == '__main__':
    main()
