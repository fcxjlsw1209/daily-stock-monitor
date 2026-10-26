"""Explicit, source-only reconciliation of official H.10 index/page dates."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import unittest
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'work'))
import ptr169_fed_bulk as bulk

OUT = Path(__file__).resolve().parent
FED = OUT.parent
ORIGINAL_RATES_SHA = 'cd385a4409422c6d5e99b69ab904769a96537df423ba31745f5b28e45e6484e7'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path): return json.loads(path.read_text())
def now(): return datetime.now(timezone.utc).isoformat()
def write(name, value):
    with (OUT / name).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def indexed_dates():
    return sorted({datetime.strptime(day, '%Y%m%d').date().isoformat()
                   for year in load(FED / 'releaseDates.json')
                   for month in year['Months'] for day in month['Dates']})


def actual_date(html, url):
    parser = bulk.Inspector(url)
    parser.feed(html)
    text = ' '.join(' '.join(parser.text).split())
    matches = re.findall(r'Release Date:\s*([A-Za-z]+ \d{1,2}, \d{4})', text)
    assert len(matches) == 1, ('Missing/ambiguous official page date', matches)
    return datetime.strptime(matches[0], '%B %d, %Y').date().isoformat()


def reconciled_page(html, indexed_date, url):
    date = actual_date(html, url)
    page = bulk.parse_page(html, date, url)
    page.update(indexed_release_date=indexed_date,
                page_release_date=date,
                known_date=max(date, indexed_date),
                index_page_date_match=date == indexed_date)
    return page


def select(page, asof):
    assert page['known_date'] < asof, 'Same-day/future conservative known date excluded'
    selected = bulk.choose(page, asof)
    selected.update(indexed_release_date=page['indexed_release_date'],
                    page_release_date=page['page_release_date'], known_date=page['known_date'],
                    known_date_age_calendar_days=(datetime.fromisoformat(asof) - datetime.fromisoformat(page['known_date'])).days)
    return selected


def register():
    original = bulk.verify_plan()
    assert sha(OUT / 'RATES.json') == ORIGINAL_RATES_SHA
    method = load(FED / 'RELEASE_METHOD.json')
    pages = []
    for item in original['requests']:
        body, receipt = bulk.cached_or_new(item)
        bulk.verify_receipt(item, body, receipt)
        page = reconciled_page(body.read_text(encoding='utf-8-sig'), item['date'], item['url'])
        pages.append(dict(indexed_release_date=item['date'], page_release_date=page['page_release_date'],
                          url=item['url'], known_date=page['known_date'], body_sha256=sha(body)))
    conflicts = [p for p in pages if p['indexed_release_date'] != p['page_release_date']]
    assert len(conflicts) == 5
    affected = []
    dates = indexed_dates()
    for target in method['target_release_candidates']:
        page = next(p for p in pages if p['indexed_release_date'] == target['release_date'])
        if page['known_date'] >= target['asof']:
            candidates = list(reversed([d for d in dates if '2019-01-01' <= d < target['release_date']]))
            affected.append(dict(target=target, fallback_index_dates=candidates))
    assert len(affected) == 1 and affected[0]['target']['asof'] == '2019-02-27'
    pins = dict(load(OUT / 'RATES.json')['inputs'])
    for path in [OUT / 'PLAN.json', OUT / 'RATES.json', FED / 'RELEASE_METHOD.json',
                 FED / 'releaseDates.json', ROOT / 'work/ptr169_fed_bulk.py', ROOT / 'work/ptr169_fed_sources.py']:
        pins[str(path.relative_to(ROOT))] = sha(path)
    write('RECONCILIATION_PLAN.json', dict(
        schema='ptr169_fed_date_reconciliation_v1', registered_utc=now(), code_sha256=sha(Path(__file__)),
        original_bulk_plan_sha256=sha(OUT / 'PLAN.json'), original_rates_sha256=ORIGINAL_RATES_SHA,
        inputs=pins, observed_conflicts=conflicts, fallback_targets=affected,
        known_date_rule='max(official index date, official page Release Date); require strictly before asof. This is a conservative date convention, not proof of historical delivery time.',
        fallback_rule='For the affected target try these observed official index dates in descending order, one new page per separate acquisition step. Parse offline before requesting the next; stop at first page with known_date strictly before asof and a common non-ND observation. Reuse any cached original page; never infer a URL date not in the index.',
        currency_rule='Preserve original local-currency-per-USD CNY/HKD strings; latest common non-ND observation within eligible page; no cross-date mixing or forward fill.',
        network_rule='One request per uncached declared URL; no retries; any 403/429 stops Fed source for current UTC day.',
        preserve='Original PLAN, parser, source bodies/receipts, and failed RATES remain byte-identical.',
        cap_computed=False, pool_computed=False, returns_computed=False))
    print(json.dumps(dict(plan_sha256=sha(OUT / 'RECONCILIATION_PLAN.json'), conflicts=conflicts,
                          fallback_targets=affected), ensure_ascii=False))


def verify():
    bulk.verify_plan()
    plan = load(OUT / 'RECONCILIATION_PLAN.json')
    assert plan['code_sha256'] == sha(Path(__file__))
    for path, digest in plan['inputs'].items(): assert sha(ROOT / path) == digest, path
    return plan


def fetch(date):
    plan = verify()
    assert date in plan['fallback_targets'][0]['fallback_index_dates']
    url = 'https://www.federalreserve.gov/releases/h10/' + date.replace('-', '') + '/'
    bodyfile = OUT / (date.replace('-', '') + '.html')
    receiptfile = OUT / (bodyfile.name + '.receipt.json')
    if receiptfile.exists():
        receipt = load(receiptfile)
        assert receipt['url'] == url and sha(bodyfile) == receipt['body_sha256']
        print(json.dumps(dict(reused=True, date=date, status=receipt['status'])))
        return
    assert not bodyfile.exists()
    assert not any(p.get('status') in (403, 429) and p['requested_utc'][:10] == now()[:10]
                   for p in (load(path) for folder in [FED, OUT] for path in folder.glob('*.receipt.json'))), 'Fed source stopped today'
    receipt = dict(url=url, requested_utc=now(), reconciliation_plan_sha256=sha(OUT / 'RECONCILIATION_PLAN.json'))
    request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 public source research'})
    try:
        with urllib.request.urlopen(request, timeout=40) as response:
            body = response.read()
            receipt.update(status=response.status, final_url=response.url,
                           content_type=response.headers.get('Content-Type'), retry_after=response.headers.get('Retry-After'),
                           last_modified=response.headers.get('Last-Modified'), etag=response.headers.get('ETag'))
    except urllib.error.HTTPError as error:
        body = error.read()
        receipt.update(status=error.code, final_url=error.url, retry_after=error.headers.get('Retry-After'))
    except Exception as error:
        body = None
        receipt.update(status='transport_error', error=type(error).__name__, message=str(error))
    if body is not None:
        with bodyfile.open('xb') as stream: stream.write(body)
        receipt.update(body_sha256=sha(bodyfile), bytes=len(body))
    receipt['finished_utc'] = now()
    write(receiptfile.name, receipt)
    print(json.dumps(dict(date=date, status=receipt['status'], body_sha256=receipt.get('body_sha256'))))


def parse():
    plan = verify()
    original = load(OUT / 'PLAN.json')
    pages, inputs = {}, dict(plan['inputs'])
    inputs[str((OUT / 'RECONCILIATION_PLAN.json').relative_to(ROOT))] = sha(OUT / 'RECONCILIATION_PLAN.json')
    items = list(original['requests'])
    for date in plan['fallback_targets'][0]['fallback_index_dates']:
        if date not in [item['date'] for item in items]:
            items.append(dict(date=date, url='https://www.federalreserve.gov/releases/h10/' + date.replace('-', '') + '/',
                              name=date.replace('-', '') + '.html', cached=False, fallback=True))
    for item in items:
        body, receiptfile = bulk.cached_or_new(item)
        if not receiptfile.exists(): continue
        if item.get('fallback'):
            receipt = load(receiptfile)
            assert receipt['url'] == item['url'] and receipt['reconciliation_plan_sha256'] == sha(OUT / 'RECONCILIATION_PLAN.json')
            assert sha(body) == receipt['body_sha256'] and body.stat().st_size == receipt['bytes']
        else: receipt = bulk.verify_receipt(item, body, receiptfile)
        assert receipt['status'] == 200
        page = reconciled_page(body.read_text(encoding='utf-8-sig'), item['date'], item['url'])
        page.update(url=item['url'], body_sha256=sha(body), retrieved_utc=receipt['requested_utc'])
        pages[item['date']] = page
        for path in [body, receiptfile]: inputs[str(path.relative_to(ROOT))] = sha(path)
    selected, excluded = [], []
    for target in load(FED / 'RELEASE_METHOD.json')['target_release_candidates']:
        candidates = [target['release_date']]
        fallback = next((f for f in plan['fallback_targets'] if f['target']['asof'] == target['asof']), None)
        if fallback: candidates += fallback['fallback_index_dates']
        chosen = None
        for date in candidates:
            assert date in pages, ('Next declared page needs separate acquisition', date)
            page = pages[date]
            if page['known_date'] >= target['asof']:
                excluded.append(dict(signal_date=target['signal_date'], asof=target['asof'], url=page['url'],
                                     indexed_release_date=date, page_release_date=page['page_release_date'],
                                     known_date=page['known_date'], reason='same-day/future known date'))
                continue
            if not any(o['cny_per_usd'] is not None and o['hkd_per_usd'] is not None for o in page['observations']):
                excluded.append(dict(signal_date=target['signal_date'], url=page['url'], reason='no common non-ND quote'))
                continue
            chosen = dict(signal_date=target['signal_date'], asof=target['asof'], url=page['url'],
                          originally_selected_index_date=target['release_date'], body_sha256=page['body_sha256'],
                          used_earlier_release=date != target['release_date'], **select(page, target['asof']))
            break
        assert chosen is not None, ('No eligible declared source', target)
        selected.append(chosen)
    assert len(selected) == len({s['signal_date'] for s in selected}) == 53
    original_rates = load(OUT / 'RATES.json')['selected_rates']
    for previous in original_rates:
        current = next(s for s in selected if s['signal_date'] == previous['signal_date'])
        for key in previous: assert current[key] == previous[key], (key, current['signal_date'])
    ages = [s['quote_age_calendar_days'] for s in selected]
    result = dict(schema='ptr169_fed_reconciled_rates_v1', created_utc=now(), code_sha256=sha(Path(__file__)),
                  reconciliation_plan_sha256=sha(OUT / 'RECONCILIATION_PLAN.json'), inputs=inputs,
                  complete=True, selected_rates=selected, parsed_releases=list(pages.values()), excluded_pages=excluded,
                  date_conflicts=plan['observed_conflicts'],
                  counts=dict(targets=53, selected=53, downloaded_release_pages=len(pages),
                              index_page_date_conflicts=len(plan['observed_conflicts']),
                              earlier_release_selections=sum(s['used_earlier_release'] for s in selected),
                              original_48_selections_unchanged=True,
                              quote_age_calendar_days_min=min(ages), quote_age_calendar_days_max=max(ages),
                              ND_cells=sum(p['ND_cells'] for p in pages.values()),
                              selections_skipping_newer_ND=sum(s['ND_observations_skipped'] > 0 for s in selected)),
                  policy=plan['known_date_rule'], currency_rule=plan['currency_rule'],
                  limitations='2026 retrieval of official dated archive. Official about page says past releases are not revised, but this is not a contemporaneously captured historical vintage. Conservative known_date is a declared convention; actual per-release delivery timestamp/outage history is not independently certified. Quotes are lagged Fed New York noon buying rates, not same-day closing FX.',
                  cap_computed=False, pool_computed=False, returns_computed=False)
    for path, digest in inputs.items(): assert sha(ROOT / path) == digest, path
    write('RECONCILED_RATES.json', result)
    print(json.dumps(dict(sha256=sha(OUT / 'RECONCILED_RATES.json'), **result['counts'])))


class Tests(unittest.TestCase):
    def page(self, indexed, actual):
        return dict(indexed_release_date=indexed, page_release_date=actual, release_date=actual,
                    known_date=max(indexed, actual), observations=[dict(observation_date='2019-02-22', cny_per_usd='6.7', hkd_per_usd='7.8')])
    def test_same_day_actual_excluded(self):
        with self.assertRaises(AssertionError): select(self.page('2019-02-25', '2019-02-27'), '2019-02-27')
    def test_same_day_index_excluded(self):
        with self.assertRaises(AssertionError): select(self.page('2019-02-27', '2019-02-25'), '2019-02-27')
    def test_later_index_used_conservatively(self):
        value = select(self.page('2019-02-26', '2019-02-25'), '2019-02-27')
        self.assertEqual(value['known_date_age_calendar_days'], 1)
        self.assertEqual(value['quote_age_calendar_days'], 5)
    def test_later_page_used_conservatively(self):
        value = select(self.page('2019-02-25', '2019-02-26'), '2019-02-27')
        self.assertEqual(value['known_date'], '2019-02-26')
    def test_date_adapter_preserves_original_parser(self):
        html = bulk.Tests().html('Rates in currency units per U.S. dollar except as noted')
        page = reconciled_page(html, '2016-08-16', 'https://example.test/')
        self.assertFalse(page['index_page_date_match'])
        self.assertEqual(page['ND_cells'], 1)
        self.assertEqual(page['known_date'], '2016-08-16')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['test', 'register', 'fetch', 'parse'])
    parser.add_argument('--date')
    args = parser.parse_args()
    if args.phase == 'test':
        suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(bulk.Tests),
                                   unittest.defaultTestLoader.loadTestsFromTestCase(Tests)])
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful(): raise SystemExit(1)
    elif args.phase == 'fetch': fetch(args.date)
    else: globals()[args.phase]()
