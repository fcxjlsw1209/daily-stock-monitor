"""Public document fetch with per-URL cache; never retries 429 sources that day."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import urllib.error
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RAW = HERE / 'raw'


def sha(b):
    return hashlib.sha256(b).hexdigest()


def dump(p, v):
    p.write_text(json.dumps(v, indent=2, ensure_ascii=False) + '\n')


def fetch(url):
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / (sha(url.encode())[:20] + ('.pdf' if urlsplit(url).path.endswith('.pdf') else '.html'))
    meta_path = path.with_suffix('.meta.json')
    if not path.exists():
        for old in RAW.glob('*.meta.json'):
            metadata = json.loads(old.read_text())
            if metadata.get('source_url') == url:
                for suffix in ('.pdf', '.html'):
                    prior = old.with_name(old.name.removesuffix('.meta.json') + suffix)
                    if prior.exists():
                        assert sha(prior.read_bytes()) == metadata['sha256']
                        return prior, metadata
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        assert sha(path.read_bytes()) == meta['sha256'], 'cache checksum changed'
        return path, meta
    stop_path = HERE / 'SOURCE_STOPS.json'
    stops = json.loads(stop_path.read_text()) if stop_path.exists() else {}
    host = urlsplit(url).hostname
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    if stops.get(host, {}).get('utc_date') == today:
        raise RuntimeError('Source stopped today: ' + host)
    time.sleep(1.1)
    try:
        response = urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'}), timeout=30)
        content = response.read(10_000_001); status = response.status; headers = response.headers; final = response.url
    except urllib.error.HTTPError as e:
        content = e.read(10_000_001); status = e.code; headers = e.headers; final = e.url
    if len(content) > 10_000_000:
        raise RuntimeError('Unexpected oversized response')
    path.write_bytes(content)
    meta = dict(source_url=url, final_url=final, http_status=status,
        retrieved_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        content_type=headers.get('Content-Type'), retry_after=headers.get('Retry-After'),
        sha256=sha(content), size=len(content), public_unauthenticated=True)
    dump(meta_path, meta)
    if status == 429:
        stops[host] = dict(utc_date=today, status=status, retry_after=meta['retry_after'])
        dump(stop_path, stops)
        raise RuntimeError('HTTP429; source stopped for today')
    return path, meta
