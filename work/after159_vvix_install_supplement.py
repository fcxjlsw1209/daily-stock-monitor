"""Preserve already collected public Cboe notice; no network or model reads."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

OUT = Path('outputs/after159_vvix_source')
SOURCE = Path('/private/tmp/after159_cboe_filter.pdf')
EXPECTED = 'a52428a70355692ab5e37fb88f8a5dd9e7f8689dbb445c9a5d69b9d047a9049b'
URL = 'https://cdn.cboe.com/resources/release_notes/2024/20240524_Consultation-Results-Regarding-the-Volatility-Index-Filtering-Mechanism-for-Select-Index-Equity-and-ETF-Volatility-Indices.pdf'

raw = SOURCE.read_bytes()
assert hashlib.sha256(raw).hexdigest() == EXPECTED and raw.startswith(b'%PDF-')
header_lines = Path('/private/tmp/after159_cboe_filter.headers').read_text().splitlines()
status = [line for line in header_lines if line.startswith('HTTP/')][-1]
assert status.split()[1] == '200'
keep = {'date', 'content-type', 'last-modified', 'etag'}
headers = {line.split(':', 1)[0].lower(): line.split(':', 1)[1].strip()
           for line in header_lines if ':' in line and line.split(':', 1)[0].lower() in keep}
assert headers['content-type'].split(';')[0] == 'application/pdf'
target = OUT / 'filter_change_20240708.pdf'
receipt = OUT / 'SUPPLEMENT.json'
assert not target.exists() and not receipt.exists()
with target.open('xb') as f:
    f.write(raw)
record = {
    'schema': 'after159_vvix_methodology_supplement_v1',
    'recorded_utc': datetime.now(timezone.utc).isoformat(),
    'requested_url': URL, 'status': 200, 'headers': headers,
    'retrieval_time_note': 'Original curl request completed before this receipt. Exact client retrieval timestamp was not recorded; HTTP Date is server response metadata, not a fabricated client receipt time.',
    'file': str(target), 'bytes': len(raw), 'sha256': EXPECTED,
    'collection_relation': 'Additional source only. Original five-source COLLECTION.json remains unchanged.',
    'purpose': 'Document the known 2024-07-08 VVIX filtering methodology change, not certify all historical revisions.',
    'new_features': 0, 'new_strategy_paths': 0,
}
with receipt.open('x') as f:
    f.write(json.dumps(record, indent=2, sort_keys=True) + '\n')
print(json.dumps({'receipt': str(receipt), 'sha256': hashlib.sha256(receipt.read_bytes()).hexdigest()}))
