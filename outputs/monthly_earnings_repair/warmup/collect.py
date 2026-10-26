"""Earlier original releases, isolated from previously verified source archives."""
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit

import requests

P = Path(__file__).resolve().parent
BASE = P.parent
ROOT = BASE.parents[1]


def dump(name, value):
    (P / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def fetch(url):
    raw = P / 'raw'; raw.mkdir(exist_ok=True)
    key = sha(url.encode())[:24]
    file = raw / (key + '.html'); meta = raw / (key + '.meta.json')
    if file.exists() and meta.exists():
        m = json.loads(meta.read_text())
        assert sha(file.read_bytes()) == m['sha256']
        return file, m
    sp = P / 'SOURCE_STOPS.json'
    stops = json.loads(sp.read_text()) if sp.exists() else {}
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    host = urlsplit(url).hostname
    if stops.get(host, {}).get('date') == today:
        raise RuntimeError('Source stopped today: ' + host)
    time.sleep(1.1)
    response = requests.get(url, timeout=40, headers={'User-Agent': 'Mozilla/5.0'})
    file.write_bytes(response.content)
    m = dict(url=url, final_url=response.url, http_status=response.status_code,
             sha256=sha(response.content), retrieved_at=dt.datetime.now(dt.timezone.utc).isoformat(),
             retry_after=response.headers.get('Retry-After'), content_type=response.headers.get('Content-Type'))
    meta.write_text(json.dumps(m, indent=2) + '\n')
    if response.status_code in (403, 429):
        stops[host] = dict(m, date=today)
        dump('SOURCE_STOPS.json', stops)
    return file, m


def main():
    apple = module('warmup_apple', BASE / 'apple/collect.py')
    msft = module('warmup_msft', BASE / 'microsoft/collect.py')
    events, pending = [], []
    targets = [r for r in json.loads(apple.VENDOR.read_text()) if '2011-01-01' <= r['reported_date'] < '2015-01-01']
    for target in targets:
        urls = apple.candidates(target)
        if 'Record-First' in urls[0]: urls.append(urls[0].replace('Record-First', 'First'))
        accepted = False
        errors = []
        for url in urls:
            try:
                file, m = fetch(url)
                if m['http_status'] != 200:
                    raise ValueError('HTTP ' + str(m['http_status']))
                e = apple.parse_html(file.read_bytes(), target, m['final_url'], str(file))
                e['retrieved_at'] = m['retrieved_at']
                events.append(e); accepted = True; break
            except Exception as exc:
                errors.append(dict(url=url, reason=str(exc)))
        if not accepted: pending.append(dict(ticker='AAPL', date=target['reported_date'], attempts=errors))
        dump('events.json', events); dump('PENDING.json', pending)
        print('AAPL', target['reported_date'], accepted, flush=True)
    for year in range(2011, 2016):
        for quarter in range(1, 5):
            if (year, quarter) < (2011, 2) or (year, quarter) > (2015, 1): continue
            url = msft.TEMPLATE.format(year=year, quarter=quarter)
            try:
                file, m = fetch(url)
                if m['http_status'] != 200: raise ValueError('HTTP ' + str(m['http_status']))
                e = msft.parse(file.read_text(), year, quarter, url, file, m['sha256'])
                assert '2011-01-01' <= e['release_date'] < '2015-01-01'
                e['retrieved_at'] = m['retrieved_at']
                events.append(e)
            except Exception as exc:
                pending.append(dict(ticker='MSFT', year=year, quarter=quarter, url=url, reason=str(exc)))
            dump('events.json', events); dump('PENDING.json', pending)
            print('MSFT', year, quarter, 'total', len(events), 'pending', len(pending), flush=True)
    dump('STATUS.json', dict(events=len(events), pending=len(pending), scope='2011-2014 release dates, original issuer evidence'))


if __name__ == '__main__':
    main()
