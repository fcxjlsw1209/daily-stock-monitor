"""Separate, bounded public-source collection; never a research replay.

Only exact original AAPL 10-K primary documents identified in the pinned SEC
submissions cache are requested. No return calculations or accounting admission.
"""
import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/monthly_aapl_annual_notes156'
CATALOG = ROOT / 'outputs/dynamic_top100/registry_sec/0000320193_submissions.json'
CATALOG_SHA = 'e89635068cd46772d1d3859ce2bc839fab781630787943a1874f258fa354e7a9'
UA = 'MonthlyStockResearch/1.0 (automated public historical filing verification)'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, obj):
    with path.open('x') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write('\n')


def targets():
    raw = CATALOG.read_bytes()
    if sha(raw) != CATALOG_SHA:
        raise ValueError('Pinned SEC submissions cache has changed')
    doc = json.loads(raw)
    if doc['cik'] != '0000320193':
        raise ValueError('Wrong issuer')
    recent = doc['filings']['recent']
    result = []
    for i, form in enumerate(recent['form']):
        if form != '10-K' or recent['reportDate'][i][:4] not in map(str, range(2016, 2026)):
            continue
        row = {k: recent[k][i] for k in ('form', 'reportDate', 'filingDate',
               'acceptanceDateTime', 'accessionNumber', 'primaryDocument')}
        if '/' in row['primaryDocument'] or '..' in row['primaryDocument']:
            raise ValueError('Unsafe primary document name')
        accession = row['accessionNumber'].replace('-', '')
        if not accession.isdigit() or len(accession) != 18:
            raise ValueError('Invalid accession')
        row['fiscal_year'] = int(row['reportDate'][:4])
        row['url'] = 'https://www.sec.gov/Archives/edgar/data/320193/{}/{}'.format(
            accession, row['primaryDocument'])
        row['catalog_path'] = str(CATALOG.relative_to(ROOT))
        row['catalog_sha256'] = CATALOG_SHA
        result.append(row)
    if len(result) != 10 or {x['fiscal_year'] for x in result} != set(range(2016, 2026)):
        raise ValueError('Expected exactly one original 10-K for each FY2016–2025')
    # FY2019 directly addresses the already identified revenue-adoption edge.
    return sorted(result, key=lambda x: (x['fiscal_year'] != 2019, x['fiscal_year']))


def plan():
    rows = targets()
    provenance = CATALOG.with_name('0000320193_submissions.provenance.json')
    return {'schema_version': 1, 'created_at': now(), 'purpose':
            'Original annual-note source collection only; no accounting or strategy admission',
            'catalog_sha256': CATALOG_SHA, 'catalog_provenance_path': str(provenance.relative_to(ROOT)),
            'catalog_provenance_sha256': sha(provenance.read_bytes()),
            'catalog_receipt_limitation':
            'Existing submissions receipt records source URL, retrieval time and digest but no HTTP status.',
            'acceptance_time_limitation':
            'Acceptance times are unmodified catalog strings, not independently certified historical availability.',
            'request_policy': {'minimum_start_interval_seconds': 1.1, 'user_agent': UA,
            'timeout_seconds': 45, 'maximum_requests': 10, 'retry_count': 0,
            'on_http_429': 'Save response and Retry-After then stop this source; no same-run retry',
            'on_http_403': 'Save response then stop this source; no alternate endpoint or evasion',
            'reuse': 'Only already saved receipt with exact target URL and matching raw SHA is reused'},
            'targets': rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    plan_path = OUT / 'PLAN.json'
    if not plan_path.exists():
        write_json(plan_path, plan())
    saved_plan = json.loads(plan_path.read_text())
    if saved_plan['targets'] != targets() or saved_plan['request_policy']['user_agent'] != UA:
        raise ValueError('Existing plan does not match this collector')
    if args.plan_only:
        print(json.dumps({'plan_sha256': sha(plan_path.read_bytes()), 'targets': len(saved_plan['targets'])}))
        return
    summary_path = OUT / 'COLLECTION.json'
    if summary_path.exists():
        raise ValueError('Collection already terminated; inspect receipt rather than repeating requests')
    receipts = []
    stopped = None
    last_start = None
    for target in saved_plan['targets']:
        fy = target['fiscal_year']
        receipt_path = OUT / ('FY{}.receipt.json'.format(fy))
        if receipt_path.exists():
            rec = json.loads(receipt_path.read_text())
            raw_path = ROOT / rec['raw_path']
            if rec['request_url'] != target['url'] or sha(raw_path.read_bytes()) != rec['raw_sha256']:
                raise ValueError('Existing raw receipt mismatch')
            receipts.append(rec)
            if not rec['accepted_original_source']:
                stopped = 'Previously saved nonaccepted response; do not retry'
                break
            continue
        if last_start is not None:
            time.sleep(max(0, 1.1 - (time.monotonic() - last_start)))
        last_start = time.monotonic()
        started = now()
        status, headers, final_url, body, error = None, {}, target['url'], b'', None
        request = urllib.request.Request(target['url'], headers={'User-Agent': UA,
                                         'Accept': 'text/html,application/xhtml+xml',
                                         'Accept-Encoding': 'identity'})
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                status, headers, final_url = response.status, dict(response.headers), response.geturl()
                body = response.read()
        except urllib.error.HTTPError as exc:
            status, headers, final_url = exc.code, dict(exc.headers), exc.geturl()
            body = exc.read()
            error = 'HTTP {}'.format(exc.code)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            error = '{}: {}'.format(type(exc).__name__, str(exc))
        raw_path = OUT / ('FY{}.response.bin'.format(fy))
        with raw_path.open('xb') as f:
            f.write(body)
        accepted = bool(status == 200 and final_url == target['url'] and body
                        and b'<html' in body[:10000].lower())
        rec = {'target': target, 'request_url': target['url'], 'final_url': final_url,
               'request_started_at': started, 'retrieved_at': now(), 'http_status': status,
               'response_headers': headers, 'retry_after': headers.get('Retry-After'),
               'raw_path': str(raw_path.relative_to(ROOT)), 'raw_sha256': sha(body),
               'raw_bytes': len(body), 'error': error, 'accepted_original_source': accepted,
               'acceptance_meaning': 'HTTP/source identity only; document text and accounting not yet admitted',
               'plan_sha256': sha(plan_path.read_bytes()),
               'collector_sha256': sha(Path(__file__).read_bytes())}
        write_json(receipt_path, rec)
        receipts.append(rec)
        print(json.dumps({'fiscal_year': fy, 'status': status, 'bytes': len(body), 'accepted': accepted}), flush=True)
        if not accepted:
            stopped = 'Nonaccepted response; bounded collection stopped without retry'
            break
    write_json(summary_path, {'created_at': now(), 'plan_sha256': sha(plan_path.read_bytes()),
               'collector_sha256': sha(Path(__file__).read_bytes()),
               'accepted_originals': sum(x['accepted_original_source'] for x in receipts),
               'targets': len(saved_plan['targets']), 'responses': len(receipts),
               'stopped_reason': stopped, 'no_accounting_admission': True,
               'no_strategy_trials': True, 'receipts': [
                   {'path': str((OUT / ('FY{}.receipt.json'.format(x['target']['fiscal_year']))).relative_to(ROOT)),
                    'sha256': sha((OUT / ('FY{}.receipt.json'.format(x['target']['fiscal_year']))).read_bytes())}
                   for x in receipts]})


if __name__ == '__main__':
    main()
