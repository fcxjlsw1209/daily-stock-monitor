"""Offline source-only PDF inspection; no strategy imports or calculations."""
from pathlib import Path
import hashlib
import json
import sys
import re

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BUNDLE = Path('/Users/shiweili/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/lib/python3.12/site-packages')
sys.path.append(str(BUNDLE))
from pypdf import PdfReader

PDF = OUT / 'FY2019.response.pdf'
EXPECTED = 'ece287b69ac1a6a030af4c1dce5d1e76cc0c5b85cfed13302440a1a0c372714c'
assert hashlib.sha256(PDF.read_bytes()).hexdigest() == EXPECTED
reader = PdfReader(str(PDF))
rows = []
for number, page in enumerate(reader.pages, 1):
    content = page.extract_text()
    rows.append(dict(page_1based=number, text=content,
        text_sha256=hashlib.sha256(content.encode()).hexdigest()))
result = dict(source_path=str(PDF.relative_to(ROOT)), source_sha256=EXPECTED,
    page_count=len(rows), extractor='pypdf', extractor_version=__import__('pypdf').__version__,
    extractor_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    accounting_admission=False, pages=rows)
with (OUT/'FY2019.pages.json').open('x') as stream:
    json.dump(result, stream, indent=2, ensure_ascii=False)
    stream.write('\n')
for row in rows:
    text = row['text']
    if row['page_1based']==1:
        print('COVER PAGE', row['page_1based'], text[:1800])
    pattern = r'revenue recognition|recategor|reclassif|Accounting Standards Update No\. 2014-09|modified retrospective|full retrospective|beginning of fiscal 2019|no impact.*revenue|net sales.*impact'
    matches = list(re.finditer(pattern, text, re.I))
    if matches:
        print('SOURCE PAGE', row['page_1based'])
        for match in matches:
            print(text[max(0, match.start()-180):min(len(text), match.end()+650)])
