"""Bounded independent check of the FY2024 mirror discrepancy; no financial values."""
import json
import re
from pathlib import Path

from after157_annual_body_coverage import Reader, require
from after157_annual_body_collect import write_json, utc

ROOT = Path(__file__).resolve().parents[1]
RAW = 'outputs/after157_walmart_bodies_late_v1/wmt_fy2024_original_10k.html'
REVIEW = 'work/after157_walmart_late_review.json'
OUT = 'work/after157_walmart_2024_root_review.json'


def main():
    reader = Reader(ROOT)
    reader.read('work/after157_walmart_2024_root_review.py')
    review = reader.json(REVIEW, 'f320a6992dbfca98eb2bad79bcc6c736a33f8b0a7de381fd2c428b8ab247248f')
    original = next(r for r in review['originals'] if r['fiscal_year'] == 2024)
    raw = reader.read(RAW, '248c60ee5992b75bf1d2e1f10282a9213aaf97022f9f1cd001285cd4f7e68ee7').decode('utf-8')
    receipt = reader.json(original['receipt_path'], original['receipt_sha256'])
    index_path = 'outputs/after157_walmart_index_v1/wmt_annual_filings_page1.html'
    index = reader.read(index_path, 'bfe5650dc8113c13bdc29ce7cf8178e8db7a1346c4f4eeb0cabaae3748bd5efc').decode('utf-8')
    url = 'https://stock.walmart.com/sec-filings/all-sec-filings/content/0000104169-24-000056/wmt-20240131.htm'
    bad_canonical = url + '/wmtexhibit21fy24.htm'
    exhibit = url.rsplit('/', 1)[0] + '/wmtexhibit21fy24.htm'
    lines = raw.splitlines()
    checks = {
        'exact_official_index_href': ('href="' + url + '"') in index.splitlines()[810],
        'response_url_no_redirect': receipt['source_url'] == receipt['final_url'] == url and receipt['redirect_chain'] == [],
        'full_response': receipt['http_status'] == 200 and not receipt['body_truncated_at_limit'],
        'canonical_and_og_discrepancy_preserved': bad_canonical in lines[10] and bad_canonical in lines[11],
        'primary_navigation_lines101_102': url in lines[100] and '>10-K' in lines[101],
        'separate_exhibit_lines121_122': exhibit in lines[120] and '>EX-21' in lines[121],
        'document_type_line178': 'name="dei:DocumentType"' in lines[177] and '>10-K<' in lines[177],
        'registrant_line218': 'WALMART INC.' in lines[217],
        'signature_section_line13821': 'SIGNATURES' in lines[13820],
    }
    context_match = re.search(r'<xbrli:context\s+id="c-1">(.*?)</xbrli:context>', raw, re.S | re.I)
    require(context_match is not None, 'Current context missing')
    context = context_match.group(1).lower()
    checks['current_unsegmented_annual_context'] = all(x in context for x in [
        '>0000104169</xbrli:identifier>', '<xbrli:startdate>2023-02-01</xbrli:startdate>',
        '<xbrli:enddate>2024-01-31</xbrli:enddate>']) and 'segment' not in context
    for field, value in [('DocumentFiscalYearFocus', '2024'), ('DocumentFiscalPeriodFocus', 'FY'), ('AmendmentFlag', 'false')]:
        tags = re.findall(r'<ix:nonnumeric\b([^>]*)>(.*?)</ix:nonnumeric>', raw, re.S | re.I)
        checks[field] = any(('name="dei:' + field + '"').lower() in attrs.lower() and
                            'contextref="c-1"' in attrs.lower() and text == value for attrs, text in tags)
    require(all(checks.values()), 'FY2024 discrepancy check failed: ' + str(checks))
    payload = dict(schema='after157_wmt_fy2024_root_check_v1', created_utc=utc(), passed=True,
        checks=checks, input_hashes=dict(reader.hashes), source_admission_changed=False,
        reviewed_original_event=original['event_id'],
        correction=dict(unchanged_prior_review=REVIEW,
            fields={'originals[3].recorded_discrepancies[0].primary_navigation_lines':[101,102],
                    'originals[3].recorded_discrepancies[0].separate_exhibit_navigation_lines':[121,122]},
            reason='Prior review cited correct links but four-line-later locations; original raw bytes unchanged.'),
        judgement='Conditional original-version binding is supported by exact official index, response URL, primary-document navigation, annual context and full report/signatures. Erroneous canonical/og metadata is retained and not used as positive evidence.',
        limits=['No SEC-host byte comparison or historical mirror-vintage certification.',
                'No change to original timing, identity or amendment gates.', 'No lag, factor or return calculation.'])
    reader.reverify()
    write_json(reader.path(OUT)[0], payload)
    print(json.dumps(dict(output=OUT, checks=checks, passed=True)))


if __name__ == '__main__':
    main()
