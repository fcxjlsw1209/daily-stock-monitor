"""Cache original public evidence for the isolated BUD/VZ collection."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import time
import requests
from urllib.parse import urlsplit

P = Path(__file__).resolve().parent

def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')

def fetch(ticker, url):
    folder = P / ticker / 'raw'
    folder.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(url.encode()).hexdigest()[:24]
    f = folder / (key + '.bin'); m = folder / (key + '.json')
    if f.exists() and m.exists():
        meta = json.loads(m.read_text())
        assert hashlib.sha256(f.read_bytes()).hexdigest() == meta['sha256']
        return f, meta
    stops_path = P / 'BUDVZ_SOURCE_STOPS.json'
    stops = json.loads(stops_path.read_text()) if stops_path.exists() else {}
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    host = urlsplit(url).hostname
    if stops.get(host, {}).get('date') == today:
        raise RuntimeError('source stopped today: ' + host)
    time.sleep(1.1)
    r = requests.get(url, timeout=45, headers={'User-Agent':'Mozilla/5.0'})
    meta = dict(url=url, final_url=r.url, http_status=r.status_code,
                sha256=hashlib.sha256(r.content).hexdigest(),
                retrieved_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                content_type=r.headers.get('Content-Type'), retry_after=r.headers.get('Retry-After'))
    f.write_bytes(r.content); dump(m, meta)
    if r.status_code in (403, 429):
        stops[host] = dict(meta, date=today); dump(stops_path, stops)
    return f, meta
