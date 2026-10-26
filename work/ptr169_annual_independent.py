"""Offline independent audit of PTR 2015-2019 share-class evidence.

No producer helper imports, network, price inputs, valuation or strategy work.
Default is a read-only probe; --write creates only the new independent report.
"""
import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/monthly_ptr_pool_source169'
OUT = BASE / 'annual'
PINS = {}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def track(path, expected=None):
    path = Path(path)
    value = digest(path)
    if expected is not None:
        assert value == expected, str(path)
    PINS[str(path.relative_to(ROOT))] = value
    return path


def load(path, expected=None):
    return json.loads(track(path, expected).read_text())


def compact(text):
    return ' '.join(text.split())


def parse_table(text):
    text = compact(text)
    assert 'Unit: Shares' in text
    assert 'Pre-movement' in text and 'Post-movement' in text
    patterns = {
        'A': r'1\. RMB.*?(?=2\. Shares)',
        'H': r'3\. Shares.*?(?=4\. Others)',
        'total': r'Shares without selling restrictions.*?(?=1\. RMB)',
    }
    result = {}
    for key, pattern in patterns.items():
        matches = re.findall(pattern, text)
        assert len(matches) == 1, (key, matches)
        counts = [int(v.replace(',', '')) for v in re.findall(r'\b\d{1,3}(?:,\d{3})+\b', matches[0])]
        assert len(counts) == 2, (key, counts)
        assert counts[0] == counts[1], (key, counts)
        result[key] = dict(pre=counts[0], post=counts[1])
    assert 'there was no issue of shares' in text
    assert result['A']['post'] == 161922077818
    assert result['H']['post'] == 21098900000
    assert result['total']['post'] == 183020977818
    return result


def main(write=False):
    annual = load(OUT/'AUDIT.json', 'a1f4bb12c733cdf8dbb7307d664205ff18ceca7a075a0dcaa893c3a37e19eefd')
    document = load(BASE/'DOCUMENT_AUDIT.json', 'eaaa6ea43e74c692d931a255e960fcbd428bd345bcdf1fbf18efbfea8a8f9714')
    plan = load(OUT/'PLAN.json')
    extraction = load(OUT/'EXTRACTION.json')
    for parent in (annual, plan):
        for name, expected in parent['input_sha256'].items():
            track(ROOT/name, expected)
    track(ROOT/'work/ptr169_annual_audit.py', annual['code_sha256'])
    track(ROOT/'work/ptr169_annual_sources.py', plan['code_sha256'])
    assert extraction['plan_sha256'] == digest(OUT/'PLAN.json')
    assert extraction['code_sha256'] == plan['code_sha256']
    sec = load(ROOT/'outputs/dynamic_top100/priority_sec/PTR_submissions.json')['filings']['recent']
    facts = load(ROOT/'outputs/dynamic_top100/priority_sec/PTR_facts.json')
    summary = load(ROOT/'outputs/monthly_2016_extension/FOREIGN_EVIDENCE.json')['PTR']
    quotes = load(BASE/'QUOTE_AUDIT.json')
    first_asof = quotes['target_scope']['first_asof']
    assert first_asof == '2016-08-16'
    runtime = load(ROOT/'outputs/monthly_profitability_extension_sources_v1/EXTRACTION_PLAN.json', extraction['runtime_sha256'])
    runtime_checked = {}
    for name, expected in runtime['pdf_runtime_hashes'].items():
        assert digest(name) == expected
        runtime_checked[name] = expected
    sys.path.append(runtime['pdf_runtime']['bundle_site_packages'])
    import pypdfium2 as pdfium

    records = []
    for year in range(2015, 2020):
        if year == 2015:
            source = track(ROOT/document['source']['pdf'], document['source']['pdf_sha256'])
            receipt = load(ROOT/document['source']['receipt'], document['source']['receipt_sha256'])
            derived = load(BASE/'derived/INDEX.json', document['derived_evidence']['index_sha256'])
            page = next(row for row in derived['pages'] if row['pdf_page'] == 13)
            expected = dict(A=document['share_classes']['A_shares'], H=document['share_classes']['H_shares'], total=document['share_classes']['reported_total_shares'])
            assert receipt['source_plan_sha256'] == digest(BASE/'SOURCE_PLAN.json')
            producer_row = None
        else:
            row = extraction['reports'][str(year)]
            source = track(OUT/(str(year)+'.pdf'), row['source_sha256'])
            receipt = load(OUT/(str(year)+'.receipt.json'), row['receipt_sha256'])
            assert receipt['plan_sha256'] == digest(OUT/'PLAN.json')
            assert receipt['url'] == plan['sources'][str(year)]
            page, = row['pages']
            producer_row = next(row for row in annual['rows'] if row['period_end'] == str(year)+'-12-31')
            expected = dict(A=producer_row['A_shares'], H=producer_row['H_shares'], total=producer_row['total'])
        assert receipt['status'] == 200 and receipt['body_sha256'] == digest(source)
        assert receipt['bytes'] == source.stat().st_size
        text_path = track(ROOT/page['text'], page['text_sha256'])
        image_path = track(ROOT/page['image'], page['image_sha256'])
        assert page['pdf_page'] == 13
        pdf = pdfium.PdfDocument(source.read_bytes())
        fresh = pdf[12].get_textpage().get_text_range()
        assert compact(fresh) == compact(text_path.read_text()), ('fresh extraction mismatch', year)
        counts = parse_table(fresh)
        assert all(counts[key]['post'] == value for key,value in expected.items())
        if year == 2015:
            assert 'each representing 100 H shares' in compact(pdf[222].get_textpage().get_text_range())
            assert 'March 23, 2016' in compact(pdf[235].get_textpage().get_text_range())
        pdf.close()
        matched = [i for i, form in enumerate(sec['form']) if form == '20-F' and sec['reportDate'][i] == str(year)+'-12-31']
        assert len(matched) == 1
        i = matched[0]
        index = {key:sec[key][i] for key in ['form','reportDate','filingDate','acceptanceDateTime','accessionNumber','primaryDocument']}
        if producer_row:
            assert producer_row['sec_index_filing_date'] == index['filingDate']
            assert producer_row['sec_index_accession'] == index['accessionNumber']
            assert producer_row['sec_index_primary_document'] == index['primaryDocument']
        records.append(dict(year=year, period_end=str(year)+'-12-31', source=str(source.relative_to(ROOT)),
                            source_sha256=digest(source), pdf_page=13, printed_page='009', units='individual shares',
                            independent_counts=counts, matches_existing_audit=True, independently_viewed_existing_page_image=True,
                            fresh_original_PDF_text_matches_saved_text=True, no_share_issue_sentence_present=True,
                            sec_index=index, index_filing_precedes_first_asof=index['filingDate']<first_asof))

    issued = facts['facts']['ifrs-full']['NumberOfSharesIssued']['units']['shares']
    assert len(issued) == 3
    assert all(row['end'] == '1999-11-05' and row['filed'] >= '2018-01-01' for row in issued)
    assert summary['known'] == records[0]['sec_index']['filingDate']
    assert summary['shares'] == records[0]['independent_counts']['total']['post']
    report = dict(
        schema='ptr169_annual_independent_review_v1', created_utc=datetime.now(timezone.utc).isoformat(),
        verdict='PASS_TRANSCRIPTION_AND_CONDITIONAL_CONSTANT_ONLY_NOT_STRICT_PIT',
        method='Fresh original-PDF text extraction and row-bounded share-count parsing; every saved page13 image independently viewed; source and receipt hashes verified; no producer helper imports.',
        rows=records, first_target_asof=first_asof, source_and_input_sha256=PINS,
        code_sha256=digest(__file__), pdf_runtime_sha256=runtime_checked,
        findings=[
            'All five original annual reports give identical exact A/H and total pre/post movement counts in individual shares.',
            'The 2015 report distinguishes the exact Shares table from rounded RMB-million share capital and explicitly describes ADSs as representing 100 H shares.',
            'The SEC index establishes a 2015-period 20-F filing before the first target asof; it is not the filed document body and does not itself contain the class counts.',
            'FOREIGN_EVIDENCE is an existing derived summary with total shares and source URL, not independently preserved A/H-specific filing text.',
            'The local SEC NumberOfSharesIssued facts are 1999 initial-issuance figures reported in 2018-2020; they do not supply the 2015 A/H split or its pre-start publication proof.'
        ],
        pre_start_publication=dict(
            assessment='supported_indirectly_not_document_level_proven',
            index_filing_date='2016-04-28', index_precedes_first_target=True,
            annual_report_board_approval_date='2016-03-23', exchange_URL_date_hint='2016-04-15',
            exact_HKEX_PDF_publication_timestamp_proven=False,
            original_2015_SEC_20F_body_with_AH_counts_located=False,
            note='Filename-oriented search found no locally named d264935d20f or 561899 filing body; this is not a claim that every anonymous archived byte was exhaustively searched.'),
        fixed_constant_sensitivity=dict(
            acceptable_as_explicit_conditional_research_scenario=True,
            constants_source='2015 period-end report only',
            later_reports_allowed_use='Retrospective consistency cross-check only; never pretend their future disclosures were already known.',
            required_label='Conditional fixed-2015-share-count sensitivity, with indirect pre-start publication support and no claim of fully validated historical dynamic shares.',
            requirements=[
                'Register A=161922077818 and H=21098900000 as unchanged assumed constants before valuation; never switch them using later report knowledge.',
                'Keep the pre-start document-level evidence gap explicit rather than relabeling the SEC index as the filed contents.',
                'Do not equate zero reported new issuance or equal annual endpoints with an exhaustive intrayear corporate-action audit.',
                'Keep all 2020 target dates conditional: the reviewed final report ends in 2019.',
                'A/H classes and ADS underlying representation must remain distinct; no double counting ADSs.'
            ]),
        unresolved=[
            'No exact-content bridge between the cached 2015 SEC filing index and the retrieved HKEX PDF has been verified.',
            'No exhaustive daily/intrayear share-class conversion, cancellation, treasury or ADS-ratio history was reviewed.',
            'Matching subsequent annual endpoints supports the assumption retrospectively but cannot upgrade it to contemporaneous knowledge.',
            'Prices, FX availability, market-calendar handling and valuation conventions require their separate checks.'
        ],
        work_performed=dict(original_PDFs_reread=5, tables_parsed=5, visual_pages_reviewed=5,
                            new_network_requests=0, prices_filled=False, capitalization_or_pool_calculated=False,
                            strategy_returns_calculated=False, existing_files_modified=False))
    for name, expected in PINS.items():
        assert digest(ROOT/name) == expected
    if write:
        with (OUT/'INDEPENDENT_REVIEW.json').open('x') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write('\n')
    print(json.dumps(dict(verdict=report['verdict'], original_PDFs_reread=5, first_target_asof=first_asof,
                          input_files_verified=len(PINS), report_written=write)))


if __name__ == '__main__':
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument('--write', action='store_true')
    main(args.parse_args().write)
