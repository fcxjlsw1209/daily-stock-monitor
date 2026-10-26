"""Bounded SEC original-body collection; source inspection never computes financial values."""
import argparse
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PREFIX = ROOT / 'work/after155_sec_gap_discovery'


def path(suffix):
    return Path(str(PREFIX) + suffix)


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def read(file):
    return json.loads(file.read_text())


def save(file, value):
    with file.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write('\n')


def collect():
    scope = read(path('_scope.json'))
    for name, expected in scope['input_hashes'].items():
        assert sha((ROOT / name).read_bytes()) == expected
    assert not path('_collection.json').exists()
    receipts = []
    stopped = False
    for target in scope['targets']:
        row = dict(key=target['key'], url=target['url'], request_utc=datetime.now(timezone.utc).isoformat())
        if stopped:
            row.update(status='not_attempted_after_SEC_stop')
        else:
            try:
                req = urllib.request.Request(target['url'], headers={'User-Agent': 'Personal investment research data-quality audit'})
                with urllib.request.urlopen(req, timeout=35) as response:
                    body = response.read(8 * 1024 * 1024 + 1)
                    assert len(body) <= 8 * 1024 * 1024
                    row.update(status='downloaded', http_status=response.status, final_url=response.url,
                               content_type=response.headers.get('Content-Type'), retry_after=response.headers.get('Retry-After'))
            except urllib.error.HTTPError as error:
                body = error.read(8 * 1024 * 1024)
                row.update(status='http_error', http_status=error.code, final_url=error.url,
                           content_type=error.headers.get('Content-Type'), retry_after=error.headers.get('Retry-After'))
                stopped = error.code in (401, 403, 429)
            except (urllib.error.URLError, TimeoutError) as error:
                body = None
                row.update(status='network_error', error_type=type(error).__name__, error=str(error))
                stopped = True
            if body is not None:
                raw = path('_' + target['key'] + '.html')
                with raw.open('xb') as stream:
                    stream.write(body)
                row.update(path=str(raw.relative_to(ROOT)), sha256=sha(body), bytes=len(body))
            if stopped:
                row['SEC_provider_stopped_for_UTC_date'] = row['request_utc'][:10]
        save(path('_' + target['key'] + '_receipt.json'), row)
        receipts.append(row)
    save(path('_collection.json'), dict(
        scope_sha256=sha(path('_scope.json').read_bytes()), helper_sha256=sha(Path(__file__).read_bytes()),
        receipts=receipts, direct_requests=sum(r['status'] != 'not_attempted_after_SEC_stop' for r in receipts),
        network_policy='One attempt per frozen primary; all remaining SEC requests stopped after 401/403/429 or network failure.'))
    print(json.dumps(receipts, indent=2))


class Visible(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hidden = 0
        self.text = []
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag in ('style', 'script'):
            self.hidden += 1
        if tag == 'a':
            self.links.extend(value for key, value in attrs if key == 'href' and value)

    def handle_endtag(self, tag):
        if tag in ('style', 'script') and self.hidden:
            self.hidden -= 1

    def handle_data(self, data):
        if not self.hidden:
            self.text.append(data)


def inspect():
    collection = read(path('_collection.json'))
    rows = []
    for receipt in collection['receipts']:
        row = dict(receipt)
        if receipt.get('path'):
            body = (ROOT / receipt['path']).read_bytes()
            assert sha(body) == receipt['sha256']
            if receipt['status'] == 'downloaded':
                parsed = Visible(); parsed.feed(body.decode('utf-8', errors='replace'))
                plain = ' '.join(' '.join(parsed.text).split())
                # Original 8-K bodies are brief; extract only the relevant release narrative.
                start = plain.casefold().find('item 2.02')
                if start < 0:
                    start = plain.casefold().find('results of operations')
                row.update(relevant_section=plain[start:start + 1800] if start >= 0 else None,
                           linked_exhibits=[link for link in parsed.links if '99' in link.lower()])
        rows.append(row)
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('phase', choices=('collect', 'inspect'))
    {'collect': collect, 'inspect': inspect}[args.parse_args().phase]()
