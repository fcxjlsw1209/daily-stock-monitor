"""Pre-registration source-only diagnostics; never a formal success receipt."""
from datetime import datetime, timezone
import json
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import operating_margin172_v3_common as common
import operating_margin172_common as original_common
import operating_margin172_independent_v4 as audit

plan = common.gate('d9034467d0116008fe27091257bc70abb0244476d0cdd4b3731f6c269895367e')
hp = original_common.OUT / 'v2/html/EXTRACTION.json'
ap = original_common.OUT / 'aapl/REPORTS.json'
assert common.sha(hp) == '791f1a6994e6a2193125b788fcb841d855bc5d4aa6ce5487013f49718629df34'
assert common.sha(ap) == 'be0c7bb99a996bd37a8c0b383d134b35fdb686280eb232a14a505128c73f6c2f'
html = {(r['issuer'], r['fiscal_year']): r for r in common.read(hp)['reports']}
apples = {r['fiscal_year']: r for r in common.read(ap)['records']}
sources = common.read(plan['source_manifests']['html'])
sessions, calendar = audit.calendar_check(plan)
records = []


def failure(error):
    trace = error.__traceback__
    stack = []
    while trace:
        stack.append(dict(function=trace.tb_frame.f_code.co_name, line=trace.tb_lineno))
        trace = trace.tb_next
    return dict(error=type(error).__name__ + ': ' + str(error), stack=stack)


for issuer in ('MSFT', 'ABBV'):
    for source in sources[issuer]:
        record = dict(issuer=issuer, fiscal_year=source['fiscal_year'], source=source['source']['raw'])
        try:
            raw = audit.bound_read(source['source']['raw'], plan['input_sha256'])
            receipt = json.loads(audit.bound_read(source['source']['receipt'], plan['input_sha256']))
            url = source['source']['expected_url'] if issuer == 'MSFT' else source['source']['metadata']['source_url']
            audit.verify_receipt(raw, receipt, url, issuer)
            release = audit.publication_date(raw, issuer)
            assert release == source['release_date']
            known = calendar.session_close(audit.next_session(release, sessions)).isoformat()
            assert known == source['known_by']
            parsed = audit.parse_html_statement(raw, issuer, source['fiscal_year'])
            audit.compare_html_record(parsed, html[issuer, source['fiscal_year']], release, known)
            record.update(passed=True, checked_cells=4, currency_literal=parsed['currency_literal'])
        except Exception as error:
            record.update(passed=False, **failure(error))
        records.append(record)
        print(json.dumps({k: record[k] for k in ('issuer','fiscal_year','passed')} | ({'error':record['error']} if not record['passed'] else {})))

earlier = None
for source in common.read(plan['source_manifests']['aapl'])['rows']:
    record = dict(issuer='AAPL', fiscal_year=source['fiscal_year'], source=source['source'])
    try:
        raw = audit.bound_read(source['source'], plan['input_sha256'])
        parent = audit.bound_read(source['parent_release'], plan['input_sha256'])
        receipt = json.loads(audit.bound_read(source['receipt'], plan['input_sha256']))
        parent_receipt = json.loads(audit.bound_read(source['parent_receipt'], plan['input_sha256']))
        parent_url = parent_receipt.get('source_url', parent_receipt.get('url'))
        audit.verify_receipt(raw, receipt, source['source_url'], 'AAPL', pdf=True)
        audit.verify_receipt(parent, parent_receipt, parent_url, 'AAPL')
        release = audit.publication_date(parent, 'AAPL')
        assert release == source['publication']['release_date']
        assert source['source_url'] in [urljoin(parent_url, a['href']) for a in BeautifulSoup(parent, 'html.parser').find_all('a', href=True)]
        parsed = audit.parse_aapl_statements(audit.pdf_content_pages(raw), source['fiscal_year'])
        session = audit.next_session(release, sessions)
        audit.compare_aapl_record(parsed, apples[source['fiscal_year']], release, session, earlier)
        import pypdfium2 as pdfium
        doc = pdfium.PdfDocument(raw); page = doc[parsed['page_1based']-1]; tp = page.get_textpage()
        try: audit.verify_pdf_geometry(tp, parsed, apples[source['fiscal_year']])
        finally: tp.close(); page.close(); doc.close()
        record.update(passed=True, checked_cells=4)
        earlier = dict(parsed, release_date=release)
    except Exception as error:
        record.update(passed=False, **failure(error)); earlier = None
    records.append(record)
    print(json.dumps({k: record[k] for k in ('issuer','fiscal_year','passed')} | ({'error':record['error']} if not record['passed'] else {})))

result = dict(status='PREFLIGHT_ONLY_NOT_FORMAL_VERIFICATION', generated_utc=datetime.now(timezone.utc).isoformat(),
              records=records, passed=sum(r['passed'] for r in records), failed=sum(not r['passed'] for r in records),
              financial_ratios_features_returns=0, source_fields_admitted=False,
              parser_sha256=common.sha(common.ROOT / 'work/operating_margin172_independent_v4.py'))
target = common.ROOT / 'work' / ('operating_margin172_preflight_v4_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.json')
common.dump(target, result)
print(json.dumps(dict(passed=result['passed'], failed=result['failed'], diagnostic_path=str(target), diagnostic_sha256=common.sha(target))))
