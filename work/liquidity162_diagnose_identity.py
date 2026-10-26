"""Record the failed frozen v1 identity pass; no financial-amount extraction."""
import json
from bs4 import BeautifulSoup
import liquidity162_audit as a

plan = a.bound()
rows = []
for record in plan['source_records']:
    html = a.pinned(record['raw_path'], record['raw_sha256']).decode('utf-8')
    try:
        result = a.release_evidence(html, record)
        rows.append(dict(ticker=record['ticker'], fiscal_year=record['fiscal_year'], passed=True))
    except ValueError as exc:
        soup = BeautifulSoup(html, 'html.parser')
        node = soup.select_one('.article-body')
        text = a.norm((node or soup).get_text(' ', strip=True))
        if record['ticker'] == 'MSFT':
            text = text[text.find('REDMOND,'):]
        rows.append(dict(ticker=record['ticker'], fiscal_year=record['fiscal_year'],
                         passed=False, error=str(exc), expected_period=record['fiscal_period_end'],
                         opening=text[:2200]))
receipt = dict(created_utc=a.now(), status='v1_real_extraction_failed_before_publication',
    original_error='Missing actual fiscal period in release opening',
    original_extraction_code_sha256=a.digest(a.OUT/'EXTRACTION_CODE.json'),
    diagnostic_sha256=a.digest(__file__), rows=rows,
    numerical_strategy_work=0, frozen_v1_unchanged=True)
a.write_new('IDENTITY_FAILURE.json', receipt)
print(json.dumps([r for r in rows if not r['passed']], ensure_ascii=False, indent=2))
