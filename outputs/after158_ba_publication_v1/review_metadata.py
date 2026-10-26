"""Offline, write-once check of the five collected HTML datelines and links."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import after158_ba_publication_collect as c

ROOT = Path(__file__).resolve().parents[2]
OUT = 'outputs/after158_ba_publication_v1/'
SELF = OUT + 'review_metadata.py'
SCOPE_SHA = '19c6588aa8763e8d01956bbc056a47305c2931d000a4e31b50dba44af0566785'
COLLECTION_SHA = '46c05c3141b74f6a853245b8436cf57983b076b5c44f1d0f29df1174e0288c91'
METADATA_SHA = '77f516d24be889361dd208c8c2e868ab48082d9186974f6933af2a6003732615'
HELPER_SHA = 'a7e5c1d0de9c10266a18815ea2163c064018111431b58b4b9caff9828ab5438a'

def main():
    scope = c.load_scope(SCOPE_SHA, HELPER_SHA)
    pins = dict(scope['input_hashes'])
    pins.update({OUT + 'SCOPE.json': SCOPE_SHA, OUT + 'COLLECTION.json': COLLECTION_SHA,
                 OUT + 'METADATA.json': METADATA_SHA})
    collection = json.loads(c.read(c.local(ROOT, OUT + 'COLLECTION.json'), COLLECTION_SHA))
    metadata = json.loads(c.read(c.local(ROOT, OUT + 'METADATA.json'), METADATA_SHA))
    c.require(collection['all_successful'] is True and len(collection['targets']) == len(metadata['records']) == 5,
              'Expected five successful sources')
    records = []
    for target, item, derived in zip(scope['targets'], collection['targets'], metadata['records']):
        key = target['id']
        c.require(item['target_id'] == derived['target_id'] == key, 'Target order')
        rp = OUT + item['receipt_file']
        receipt = json.loads(c.read(c.local(ROOT, rp), item['receipt_sha256']))
        pins[rp] = item['receipt_sha256']
        pins[OUT + key + '.attempt.json'] = receipt['attempt_sha256']
        pins[OUT + key + '.html'] = receipt['body_sha256']
        raw = c.read(c.local(ROOT, OUT + key + '.html'), receipt['body_sha256'])
        c.verify_existing(c.local(ROOT, OUT.rstrip('/')), target, SCOPE_SHA, HELPER_SHA)
        pins[OUT + derived['path']] = derived['sha256']
        parsed = json.loads(c.read(c.local(ROOT, OUT + derived['path']), derived['sha256']))
        actual = c.parse_page(raw, target['url'])
        for field in actual:
            c.require(actual[field] == parsed[field], 'Derived metadata differs: ' + field)
        expected = datetime.strptime(target['expected_dateline'], '%Y-%m-%d').strftime('Jan. %d, %Y')
        datelines = [x for x in parsed['visible_date_candidates'] if x['text'] == expected
                     and 'CHICAGO' in x['context'] and '/ PRNewswire /' in x['context']]
        c.require(len(datelines) == 1, 'One exact visible Chicago dateline required')
        html = raw.decode('utf-8')
        positions = [m.start() for m in re.finditer(re.escape(expected), html)]
        c.require(positions, 'Dateline must exist in original bytes')
        raw_refs = [dict(line_1based=html[:pos].count('\n') + 1,
                         snippet=html[max(0, pos - 140):pos + len(expected) + 130]) for pos in positions]
        export_url = target['url'] + '?asPDF=1'
        c.require(export_url in html, 'Expected dynamic export link')
        c.require(not parsed['explicit_publication_metadata_candidates'] and not parsed['pdf_links'],
                  'New publication fields or direct PDF links require review')
        c.require(target['original_pdf_url'] not in html, 'Exact original PDF link unexpectedly present')
        records.append(dict(
            fiscal_year=target['fiscal_year'], source_url=target['url'], final_url=receipt['final_url'],
            http_status=receipt['http_status'], bytes=receipt['bytes'], raw_sha256=receipt['body_sha256'],
            receipt_path=rp, receipt_sha256=item['receipt_sha256'],
            request_started_utc=receipt['request_started_utc'], retrieved_utc=receipt['retrieved_utc'],
            visible_dateline=datelines[0], original_html_dateline_refs=raw_refs,
            dateline_matches_previously_reviewed_pdf=True, publication_date=target['expected_dateline'],
            explicit_publication_time_found=False, exact_first_publication_time_certified=False,
            http_date=receipt['headers'].get('Date'), http_last_modified=receipt['headers'].get('Last-Modified'),
            http_time_is_publication_time=False,
            original_pdf_path=target['original_pdf_path'], original_pdf_sha256=target['original_pdf_sha256'],
            exact_original_pdf_outlink_present=False, dynamic_page_export_url=export_url,
            dynamic_export_fetched=False, dynamic_export_is_original_pdf=False,
            publication_basis='Visible original-dated official HTML dateline matching the separate PDF; current retrieval, not contemporaneous capture.',
            source_admission_changed=False))
    pins[SELF] = c.sha(c.read(c.local(ROOT, SELF)))
    for path, digest in pins.items():
        c.read(c.local(ROOT, path), digest)
    report = dict(schema='after158_ba_publication_metadata_review_v1', created_utc=c.utc(),
                  passed=True, input_hashes=pins, records=records,
                  checks=dict(five_fixed_urls_only=True, five_http_200=True,
                              no_redirects_or_retries=True, source_raw_and_receipt_hashes_verified=True,
                              all_five_datelines_match_pdf=True, modified_not_used_as_publication=True,
                              dynamic_export_not_confused_with_original_pdf=True),
                  source_admission_changed=False, new_strategy_trials=0, financial_factors_computed=0,
                  ranks_computed=0, returns_computed=0,
                  conclusion='The five separately retrieved official HTML pages corroborate the dates printed in the previously reviewed PDF releases. They do not independently authenticate historical byte versions or first-public instants. The existing Boeing archive-to-PDF link evidence remains separate; these pages offer a current HTML-to-PDF export, not the exact saved original-PDF link.',
                  remaining=['BA source timing policy and conservative session mapping are not executed here.',
                             'No exact historical publication clock or contemporaneous archive is certified.',
                             'Currency/accounting admission remains outside this collection; no source qualification changes.'])
    digest = c.write(c.local(ROOT, OUT + 'REVIEW.json'), report)
    print(json.dumps(dict(path=OUT + 'REVIEW.json', sha256=digest, checked_inputs=len(pins),
                         datelines_verified=len(records), exact_publication_times=0,
                         direct_original_pdf_outlinks=0, source_admission_changed=False)))

if __name__ == '__main__':
    main()
