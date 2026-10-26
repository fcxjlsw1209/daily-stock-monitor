"""Separate public-source collection and offline inspection for one missing event.

No strategy inputs are modified. This checks a known official quarterly-release
gap, not investment performance or comprehensive earnings-calendar coverage.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/monthly_event_gap156_sources'
URLS = [
    ('issuer_release', 'https://investors.broadcom.com/news-releases/news-release-details/broadcom-inc-announces-second-quarter-fiscal-year-2021-financial'),
    ('issuer_distributed_release', 'https://www.prnewswire.com/news-releases/broadcom-inc-announces-second-quarter-fiscal-year-2021-financial-results-and-quarterly-dividends-301305589.html'),
]


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2)
        f.write('\n')


def utc():
    return datetime.now(timezone.utc).isoformat()


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'):
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in ('script', 'style') and self.hidden:
            self.hidden -= 1

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def freeze():
    sources = {}
    for rel in ('outputs/monthly_event_clock_census155/CENSUS.json',
                'outputs/monthly_event_clock_census155/SLOTS.json',
                'outputs/monthly_earnings_repair/EVENTS.json',
                'outputs/monthly_paired_target143/CALENDAR.json'):
        sources[rel] = digest((ROOT / rel).read_bytes())
    save(OUT / 'SOURCE_SPEC.json', {
        'created_utc': utc(), 'purpose': 'Verify AVGO FY2021Q2 missing quarterly release only',
        'selection': 'Unobserved intervening quarter flagged by source-only census, independent of holdings/returns',
        'urls': URLS, 'input_hashes': sources,
        'code_sha256': digest(Path(__file__).read_bytes()),
        'expected_identity': {'symbol': 'AVGO', 'fiscal_year': 2021, 'fiscal_quarter': 2},
        'network_policy': 'One request per URL; no retries; HTTP429 stops this source for the UTC date and preserves Retry-After.',
        'scope': 'Current original-page bytes may establish original dateline, not contemporaneous capture or entire calendar completeness.',
        'strategy_trials': 0, 'original_event_file_changes': 0,
    })


def checked_spec():
    spec = read(OUT / 'SOURCE_SPEC.json')
    assert spec['code_sha256'] == digest(Path(__file__).read_bytes())
    for rel, expected in spec['input_hashes'].items():
        assert digest((ROOT / rel).read_bytes()) == expected, rel
    return spec


def collect():
    spec = checked_spec()
    assert not (OUT / 'COLLECTION.json').exists(), 'Collection already attempted; reuse cache.'
    receipts = []
    for name, url in spec['urls']:
        receipt = {'name': name, 'requested_url': url, 'retrieved_utc': utc()}
        request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (compatible; historical-research/1.0)'})
        try:
            with urllib.request.urlopen(request, timeout=35) as response:
                blob = response.read(8 * 1024 * 1024 + 1)
                assert len(blob) <= 8 * 1024 * 1024, 'Response too large'
                receipt.update(http_status=response.status, final_url=response.url,
                               content_type=response.headers.get('Content-Type'),
                               etag=response.headers.get('ETag'),
                               last_modified=response.headers.get('Last-Modified'))
            raw = OUT / (name + '.html')
            with raw.open('xb') as f:
                f.write(blob)
            receipt.update(path=str(raw.relative_to(ROOT)), sha256=digest(blob), bytes=len(blob), status='downloaded')
        except urllib.error.HTTPError as error:
            receipt.update(status='http_error', http_status=error.code,
                           retry_after=error.headers.get('Retry-After'))
        except (urllib.error.URLError, TimeoutError) as error:
            receipt.update(status='network_error', error_type=type(error).__name__, error=str(error))
        save(OUT / (name + '.receipt.json'), receipt)
        receipts.append(receipt)
        if receipt.get('http_status') == 429:
            break
    save(OUT / 'COLLECTION.json', {'created_utc': utc(), 'receipts': receipts, 'strategy_trials': 0})
    print(json.dumps([{'name': r['name'], 'status': r['status'], 'http_status': r.get('http_status')} for r in receipts]))


def inspect():
    checked_spec()
    results = []
    for r in read(OUT / 'COLLECTION.json')['receipts']:
        row = {'name': r['name'], 'status': r['status']}
        if r['status'] == 'downloaded':
            blob = (ROOT / r['path']).read_bytes()
            assert digest(blob) == r['sha256']
            parser = Text(); parser.feed(blob.decode('utf-8', errors='replace'))
            plain = ' '.join(' '.join(parser.parts).split())
            marker = 'today reported financial results for its second quarter of fiscal year 2021, ended May 2, 2021'
            start = plain.find(marker)
            dateline = plain.find('June 3, 2021')
            title_ok = 'Broadcom Inc. Announces Second Quarter Fiscal Year 2021 Financial Results and Quarterly Dividends' in plain
            row.update(path=r['path'], sha256=r['sha256'], actual_release_language=start >= 0,
                       original_dateline_found=dateline >= 0, title_matches=title_ok,
                       short_evidence='second quarter of fiscal year 2021, ended May 2, 2021' if start >= 0 else None,
                       publication_date='2021-06-03' if start >= 0 and dateline >= 0 and title_ok else None,
                       period_end='2021-05-02' if start >= 0 else None,
                       evidence_locator={'normalized_text_start': start, 'dateline_offset': dateline})
        results.append(row)
    found = [r for r in results if r.get('publication_date')]
    save(OUT / 'INSPECTION.json', {
        'created_utc': utc(), 'results': results,
        'finding': 'original_release_identified_pending_independent_admission' if found else 'source_unresolved',
        'symbol': 'AVGO', 'fiscal_year': 2021, 'fiscal_quarter': 2,
        'candidate_publication_date': '2021-06-03' if found else None,
        'candidate_conservative_available_session': '2021-06-04' if found else None,
        'date_mapping_needs_independent_review': True,
        'event_admitted': False, 'whole_calendar_admitted': False,
        'strategy_trials': 0, 'models_fit': 0, 'orders': 0,
    })
    print(json.dumps({'original_releases_found': len(found), 'event_admitted': False}))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('phase', choices=('freeze', 'collect', 'inspect'))
    {'freeze': freeze, 'collect': collect, 'inspect': inspect}[ap.parse_args().phase]()
