"""Separate public-source acquisition; no strategy or return calculations."""
import hashlib
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/monthly_ptr_pool_source169'


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(body):
    return hashlib.sha256(body).hexdigest()


def write(name, value):
    with (OUT / name).open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def register():
    OUT.mkdir(exist_ok=True)
    source = 'https://query1.finance.yahoo.com/v8/finance/chart/'
    query = '?period1=1451606400&period2=1609459200&interval=1d&events=div%2Csplits'
    requests = [{'name': 'petrochina_2015_annual.pdf',
                 'url': 'https://www.hkexnews.hk/listedco/listconews/SEHK/2016/0415/LTN20160415313.pdf',
                 'kind': 'issuer_annual_report_hosted_by_exchange'}]
    for symbol in ['601857.SS', '0857.HK', 'CNY=X', 'HKD=X']:
        requests.append({'name': symbol.replace('=', '_') + '.json',
                         'symbol': symbol, 'url': source + symbol + query,
                         'kind': 'provider_daily_chart_not_certified_vintage'})
    pins = {}
    for name in ['outputs/monthly_2016_extension/FOREIGN_EVIDENCE.json',
                 'outputs/monthly_2016_extension/EARLY_RANKS.json',
                 'outputs/monthly_2016_extension/POOL_SENSITIVITIES.json',
                 'outputs/dynamic_top100/priority_sec/PTR_submissions.json']:
        pins[name] = sha((ROOT / name).read_bytes())
    write('SOURCE_PLAN.json', dict(
        schema='ptr169_public_sources_v1', registered_utc=now(),
        purpose='Investigate documented all-shares-at-H-price assumption, not tune by performance.',
        period='53 archived early signal/asof pairs, 2016-08-17 through 2020-12-31',
        scope='Collect raw independent A/H prices, USD FX quotes and pre-start share-class evidence only.',
        requests=requests, existing_input_pins=pins, collector_sha256=sha(Path(__file__).read_bytes()),
        assumptions_to_resolve=['A/H share counts and historical release dates',
          'ADR represents 100 H shares, not 100 mixed shares',
          'Raw price units, split history, exchange-local dates and currency quote direction',
          'FX timestamp availability relative to US signal cutoff',
          'Foreign market closures must not silently forward-fill prices'],
        acquisition_rules=['One request per resource, no automatic retry',
          'On 403 or 429 stop that provider for rest of UTC day; preserve Retry-After',
          'Do not use empty responses or unrelated ticker identities',
          'Keep original bodies and request time; downloaded history is not original point-in-time vintage'],
        next_gate='Offline identity/unit/coverage audit and separately declared cap methodology before any pool or model calculation.',
        strategy_trials_added=0, returns_computed=False, observers_modified=False))
    print('SOURCE_PLAN registered')


def collect():
    plan_body = (OUT / 'SOURCE_PLAN.json').read_bytes()
    p = json.loads(plan_body)
    if p['collector_sha256'] != sha(Path(__file__).read_bytes()):
        raise ValueError('collector drift')
    for name, expected in p['existing_input_pins'].items():
        if sha((ROOT / name).read_bytes()) != expected:
            raise ValueError('input drift: ' + name)
    stop = set()
    for item in p['requests']:
        receipt = OUT / (item['name'] + '.receipt.json')
        provider = 'Yahoo' if 'finance.yahoo.com/' in item['url'] else 'HKEX'
        if receipt.exists():
            old = json.loads(receipt.read_text())
            if old.get('status') in (403, 429) and old['requested_utc'][:10] == now()[:10]:
                stop.add(provider)
            if old.get('body_sha256') and sha((OUT / item['name']).read_bytes()) != old['body_sha256']:
                raise ValueError('cached source changed')
            continue
        if provider in stop:
            continue
        meta = dict(requested_utc=now(), provider=provider, url=item['url'],
                    source_plan_sha256=sha(plan_body), source_kind=item['kind'])
        req = urllib.request.Request(item['url'], headers={'User-Agent': 'Mozilla/5.0 public historical research'})
        try:
            with urllib.request.urlopen(req, timeout=40) as response:
                body = response.read()
                meta.update(status=response.status, final_url=response.url,
                            content_type=response.headers.get('Content-Type'),
                            retry_after=response.headers.get('Retry-After'))
        except urllib.error.HTTPError as e:
            body = e.read()
            meta.update(status=e.code, final_url=e.url,
                        content_type=e.headers.get('Content-Type'),
                        retry_after=e.headers.get('Retry-After'))
        except Exception as e:
            body = None
            meta.update(status='transport_error', error=type(e).__name__, message=str(e))
        if body is not None:
            with (OUT / item['name']).open('xb') as f:
                f.write(body)
            meta.update(body_sha256=sha(body), bytes=len(body))
        meta['finished_utc'] = now()
        write(item['name'] + '.receipt.json', meta)
        if meta['status'] in (403, 429):
            stop.add(provider)
        print(json.dumps({'name': item['name'], 'status': meta['status'],
                          'bytes': meta.get('bytes'), 'provider_stopped': provider in stop}), flush=True)


if __name__ == '__main__':
    {'register': register, 'collect': collect}[sys.argv[1]]()
