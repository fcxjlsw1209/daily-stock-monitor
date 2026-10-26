"""Separate public issuer collection; no factor or portfolio calculations.

Fetch only an observed official URL. One request, no automatic redirects/retries.
403/429 stop the host for that Los Angeles calendar day; receipts are immutable.
HTML inspection reports source text/links only, never executes downloaded code.
"""
import argparse
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import hashlib
from html.parser import HTMLParser
from http.cookies import SimpleCookie
import json
from pathlib import Path
import re
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/monthly_etf_primary_flow_sources/other'
DOMAINS = ('ark-funds.com', 'ark-invest.com', 'vaneck.com', 'invesco.com', 'vanguard.com')
MAX_BYTES = 24 * 1024 * 1024


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest(b):
    return hashlib.sha256(b).hexdigest()


def write_once(path, data):
    raw = data if isinstance(data, bytes) else (json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()
    with path.open('xb') as f:
        f.write(raw)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def allowed(url):
    p = urllib.parse.urlsplit(url)
    if p.scheme != 'https' or p.username or p.password or p.fragment or not p.hostname:
        raise ValueError('HTTPS public URL required')
    if not any(p.hostname == d or p.hostname.endswith('.' + d) for d in DOMAINS):
        raise ValueError('Not one of the four assigned issuers: ' + p.hostname)
    return p.hostname


def fetch(a):
    host = allowed(a.url)
    if not re.fullmatch(r'[a-z0-9_]{1,80}', a.id):
        raise ValueError('Invalid receipt ID')
    day = datetime.now(ZoneInfo('America/Los_Angeles')).date().isoformat()
    OUT.mkdir(parents=True, exist_ok=True)
    stop = OUT / ('STOP_' + host + '_' + day + '.json')
    if stop.exists():
        raise SystemExit('Host already stopped today; no request: ' + host)
    rawpath = OUT / (a.id + '.bin')
    recpath = OUT / (a.id + '.receipt.json')
    if recpath.exists():
        prior = json.loads(recpath.read_text())
        if prior['requested_url'] != a.url or digest(rawpath.read_bytes()) != prior['sha256']:
            raise ValueError('Prior receipt URL/digest mismatch')
        print(json.dumps({'cached': True, 'id': a.id, 'status': prior['status']}))
        return
    if rawpath.exists():
        raise ValueError('Unbound existing raw path')
    record = dict(schema='etf_primary_flow_other_source_v1', requested_url=a.url,
                  discovery_source=a.discovered_on, discovery_evidence=a.evidence,
                  requested_utc=stamp(), los_angeles_date=day,
                  raw_file=str(rawpath.relative_to(ROOT)), redirects_followed=False,
                  retries=0, new_factors=0, new_model_fits=0, new_orders=0, new_returns=0)
    request_headers = {'User-Agent': 'Mozilla/5.0 (compatible; ETF-public-data-research/1.0)'}
    if a.cookie_source:
        previous = OUT / (a.cookie_source + '.receipt.json')
        prior = json.loads(previous.read_text())
        if allowed(prior['requested_url']) != host:
            raise ValueError('Cookie source must be same host')
        cookies = SimpleCookie()
        pairs = prior.get('headers_items', list(prior['headers'].items()))
        for key, value in pairs:
            if key.lower() == 'set-cookie': cookies.load(value)
        request_headers['Cookie'] = '; '.join(key + '=' + value.value for key, value in cookies.items())
        record['anonymous_cookie_source'] = str(previous.relative_to(ROOT))
        record['anonymous_cookie_names'] = sorted(cookies)
    req = urllib.request.Request(a.url, headers=request_headers)
    body = b''
    headers = {}
    response = None
    try:
        response = urllib.request.build_opener(NoRedirect()).open(req, timeout=40)
    except urllib.error.HTTPError as e:
        response = e
    except Exception as e:
        record.update(status=None, error=type(e).__name__ + ': ' + str(e), final_url=a.url)
    if response is not None:
        with response:
            record.update(status=response.code, final_url=response.geturl())
            headers = dict(response.headers.items())
            record['headers_items'] = list(response.headers.items())
            body = response.read(MAX_BYTES + 1)
        if len(body) > MAX_BYTES:
            raise ValueError('Response exceeds declared size limit; no further request')
    record.update(completed_utc=stamp(), headers=headers, bytes=len(body), sha256=digest(body))
    write_once(rawpath, body)
    write_once(recpath, record)
    if record['status'] in (403, 429):
        write_once(stop, dict(host=host, stopped_utc=stamp(), local_date=day,
                             status=record['status'], retry_after=headers.get('Retry-After'),
                             receipt=str(recpath.relative_to(ROOT)), policy='No further host requests today; no bypass.'))
    print(json.dumps({k: record.get(k) for k in ('requested_url', 'status', 'bytes', 'sha256', 'error')}, ensure_ascii=False))


class Page(HTMLParser):
    def __init__(self):
        super().__init__(); self.links=[]; self.text=[]; self.current=None
    def handle_starttag(self, tag, attrs):
        d=dict(attrs)
        if tag == 'a' and d.get('href'):
            self.current={'href': d['href'], 'label': ''}; self.links.append(self.current)
        if tag == 'script' and d.get('src'):
            self.links.append({'script': d['src']})
    def handle_data(self, data):
        t=data.strip()
        if t: self.text.append(t)
        if self.current: self.current['label'] += t + ' '
    def handle_endtag(self, tag):
        if tag == 'a': self.current=None


def inspect(a):
    rawpath=OUT/(a.id+'.bin'); rec=json.loads((OUT/(a.id+'.receipt.json')).read_text())
    body=rawpath.read_bytes()
    if digest(body) != rec['sha256']: raise ValueError('Raw source changed')
    raw=body.decode('utf-8', errors='replace'); page=Page(); page.feed(raw)
    pattern=re.compile(a.pattern, re.I)
    links=[]
    for item in page.links:
        value=item.get('href',item.get('script',''))
        if pattern.search(json.dumps(item)):
            links.append(dict(item, resolved_url=urllib.parse.urljoin(rec['final_url'],value)))
    snippets=[]
    for m in pattern.finditer(raw):
        snippets.append(raw[max(0,m.start()-140):min(len(raw),m.end()+230)])
        if len(snippets)>=a.limit: break
    print(json.dumps({'receipt':str((OUT/(a.id+'.receipt.json')).relative_to(ROOT)), 'links':links[:a.limit], 'snippets':snippets},ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='command',required=True)
    f=sub.add_parser('fetch'); f.add_argument('--id',required=True); f.add_argument('--url',required=True)
    f.add_argument('--cookie-source'); f.add_argument('--discovered-on',required=True); f.add_argument('--evidence',required=True); f.set_defaults(func=fetch)
    i=sub.add_parser('inspect'); i.add_argument('--id',required=True); i.add_argument('--pattern',default='history|historical|shares.outstanding|NAV|download|csv|xlsx'); i.add_argument('--limit',type=int,default=35); i.set_defaults(func=inspect)
    a=p.parse_args(); a.func(a)

if __name__ == '__main__': main()
