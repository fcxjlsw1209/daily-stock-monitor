"""Cache public original releases; HTTP403/429 stops a host for the UTC day."""
from pathlib import Path
from datetime import datetime, timezone
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
import hashlib
import json
import time

P = Path(__file__).resolve().parent
RAW = P / 'raw'
RAW.mkdir(exist_ok=True)

def now():
    return datetime.now(timezone.utc).isoformat()

def digest(data):
    return hashlib.sha256(data).hexdigest()

def write(name, data):
    (P / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')

def read(name, default):
    return json.loads((P / name).read_text()) if (P / name).exists() else default

def fetch(url):
    records = read('manifest.json', {})
    if url in records:
        r = records[url]
        path = P / r['evidence_path']
        if path.exists() and digest(path.read_bytes()) == r['sha256']:
            return r, path.read_text(errors='replace')
    host, day = urlsplit(url).hostname, now()[:10]
    if host not in {'ir.aboutamazon.com', 'press.aboutamazon.com', 'www.sec.gov'}:
        raise ValueError('Collector permits only official Amazon and SEC public domains')
    blocked = read('blocked_sources.json', {})
    if blocked.get(host, {}).get('date') == day:
        raise RuntimeError('Restricted source stopped for UTC day: ' + host)
    time.sleep(1.1)
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0 (compatible; historical-financial-research/1.0)', 'Accept': 'text/html,application/json'})
    started = now()
    try:
        with urlopen(req, timeout=35) as response:
            body, status, headers, final = response.read(), response.status, dict(response.headers), response.url
    except HTTPError as exc:
        body, status, headers, final = exc.read(), exc.code, dict(exc.headers), exc.url
    except URLError as exc:
        failures = read('request_failures.json', [])
        failures.append({'url': url, 'retrieved_at': started, 'error': str(exc)})
        write('request_failures.json', failures)
        return {'http_status': None, 'url': url, 'error': str(exc)}, ''
    filename = 'raw/' + digest(url.encode())[:24] + '.html'
    (P / filename).write_bytes(body)
    record = dict(url=url, final_url=final, retrieved_at=started, http_status=status,
                  content_type=headers.get('Content-Type'), retry_after=headers.get('Retry-After'),
                  evidence_path=filename, sha256=digest(body), bytes=len(body), public_unauthenticated=True)
    records[url] = record
    write('manifest.json', records)
    if status in (403, 429):
        blocked[host] = {'date': day, 'url': url, 'retrieved_at': started, 'http_status': status, 'retry_after': headers.get('Retry-After')}
        write('blocked_sources.json', blocked)
        raise RuntimeError('HTTP' + str(status) + ': stopped ' + host + ' for UTC day')
    return record, body.decode('utf-8', errors='replace')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('urls', nargs='+')
    for address in parser.parse_args().urls:
        record, _ = fetch(address)
        print(json.dumps(record), flush=True)
