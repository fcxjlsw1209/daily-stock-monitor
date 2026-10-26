"""PTR169 calendar-only source collection and offline evidence checks.

register/extract/verify run through run_local.sh. collect is a separately
authorized public acquisition step, with one request per registered resource.
No security prices, capitalization, rankings or strategy returns are read.
"""
import argparse
import hashlib
import html.parser
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/monthly_ptr_pool_source169/calendar'
QUOTE = ROOT / 'outputs/monthly_ptr_pool_source169/QUOTE_AUDIT.json'
RUNTIME = ROOT / 'outputs/monthly_profitability_extension_sources_v1/EXTRACTION_PLAN.json'


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, obj):
    with Path(p).open('x') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def register():
    OUT.mkdir(exist_ok=True)
    sse = 'https://www.sse.com.cn/disclosure/announcement/general/c/'
    items = [
        ('sse2017annual.html', 'SSE', sse + 'c_20161222_4218613.shtml'),
        ('sse2017dragonboat.html', 'SSE', sse + 'c_20170518_4312827.shtml'),
        ('sse2019labour.html', 'SSE', sse + 'c_20190418_4771364.shtml'),
        ('sse2019southbound.html', 'SSE', 'https://www.sse.com.cn/services/hkexsc/disclo/announ/c/c_20190418_4771375.shtml'),
        ('sse2020annual.html', 'SSE', sse + 'c_20191220_4969627.shtml'),
        ('sse2020extension.html', 'SSE', 'https://star.sse.com.cn/star/media/news/c/c_20200127_4991583.shtml'),
        ('hkex2017securities.pdf', 'HKEX', 'https://www.hkex.com.hk/-/media/hkex-market/services/circulars-and-notices/participant-and-members-circulars/sehk/2016/ct01716e'),
    ]
    quote = json.loads(QUOTE.read_text())
    write(OUT / 'PLAN.json', dict(
        schema='ptr169_calendar_sources_v1', registered_utc=now(),
        purpose='Classify six missing A/H quote dates using exchange market calendars, not Stock Connect availability.',
        quote_audit_sha256=sha(QUOTE), missing_dates=quote['summary']['missing_dates'],
        requests=[dict(name=n, provider=p, url=u) for n, p, u in items],
        collector_sha256=sha(__file__), runtime_plan_sha256=sha(RUNTIME),
        rules=['One request per new URL; validate and reuse cached body/receipt.',
               'On HTTP403 or HTTP429 stop all further requests to that provider for that UTC day.',
               'No price filling, capitalization, pool, feature or return calculations.',
               'Stock Connect closure alone does not imply an exchange market closure.',
               'A nonholiday does not itself exclude a security-specific suspension.'],
        results_computed_before_registration=False))


def plan():
    p = json.loads((OUT / 'PLAN.json').read_text())
    assert p['collector_sha256'] == sha(__file__), 'script changed'
    assert p['quote_audit_sha256'] == sha(QUOTE), 'quote audit changed'
    assert p['runtime_plan_sha256'] == sha(RUNTIME), 'runtime plan changed'
    return p


def cached(item):
    rpath = OUT / (item['name'] + '.receipt.json')
    if not rpath.exists():
        return None
    r = json.loads(rpath.read_text())
    assert r['url'] == item['url'] and r['provider'] == item['provider']
    assert r['plan_sha256'] == sha(OUT / 'PLAN.json')
    if 'body_sha256' in r:
        assert sha(OUT / item['name']) == r['body_sha256'], 'body changed'
    return r


def collect():
    p = plan()
    stopped = set()
    for item in p['requests']:
        r = cached(item)
        if r and r['status'] in (403, 429) and r['requested_utc'][:10] == now()[:10]:
            stopped.add(item['provider'])
    for item in p['requests']:
        if cached(item):
            print('reused ' + item['name'], flush=True)
            continue
        if item['provider'] in stopped:
            print('provider stopped: ' + item['provider'], flush=True)
            continue
        r = dict(item, requested_utc=now(), plan_sha256=sha(OUT / 'PLAN.json'))
        req = urllib.request.Request(item['url'], headers={'User-Agent': 'Mozilla/5.0 public calendar research'})
        body = None
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                body = response.read()
                r.update(status=response.status, final_url=response.url,
                         content_type=response.headers.get('Content-Type'),
                         retry_after=response.headers.get('Retry-After'))
        except urllib.error.HTTPError as error:
            body = error.read()
            r.update(status=error.code, final_url=error.url,
                     content_type=error.headers.get('Content-Type'),
                     retry_after=error.headers.get('Retry-After'))
        except Exception as error:
            r.update(status='transport_error', error=type(error).__name__, detail=str(error))
        if body is not None:
            with (OUT / item['name']).open('xb') as f:
                f.write(body)
            r.update(body_sha256=sha(OUT / item['name']), bytes=len(body))
        r['finished_utc'] = now()
        write(OUT / (item['name'] + '.receipt.json'), r)
        if r['status'] in (403, 429):
            stopped.add(item['provider'])
        print(json.dumps({k:r.get(k) for k in ['name','status','bytes','retry_after']}, ensure_ascii=False), flush=True)


class VisibleText(html.parser.HTMLParser):
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
        if not self.hidden and data.strip():
            self.parts.append(data.strip())


def extract():
    p = plan()
    records = []
    for item in p['requests']:
        r = cached(item)
        if not r or r['status'] != 200:
            continue
        source = OUT / item['name']
        outputs = []
        if item['name'].endswith('.pdf'):
            cfg = json.loads(RUNTIME.read_text())
            for path, digest in cfg['pdf_runtime_hashes'].items():
                assert sha(path) == digest
            sys.path.append(cfg['pdf_runtime']['bundle_site_packages'])
            import pypdfium2 as pdfium
            doc = pdfium.PdfDocument(source.read_bytes())
            for i in range(len(doc)):
                page = doc[i]
                t = OUT / (item['name'] + '.page%02d.txt' % (i+1))
                image = OUT / (item['name'] + '.page%02d.png' % (i+1))
                assert not t.exists() and not image.exists()
                t.write_text(page.get_textpage().get_text_range())
                page.render(scale=1.7).to_pil().save(image)
                outputs += [dict(path=str(v.relative_to(ROOT)), sha256=sha(v)) for v in [t,image]]
            doc.close()
        else:
            parser = VisibleText()
            parser.feed(source.read_text(encoding='utf-8-sig'))
            t = OUT / (item['name'] + '.txt')
            with t.open('x') as f:
                f.write('\n'.join(parser.parts) + '\n')
            outputs.append(dict(path=str(t.relative_to(ROOT)), sha256=sha(t)))
        records.append(dict(source=item['name'], source_sha256=sha(source), outputs=outputs))
    write(OUT / 'EXTRACT_INDEX.json', dict(plan_sha256=sha(OUT/'PLAN.json'), script_sha256=sha(__file__), records=records))
    print('extracted %d official bodies' % len(records))


def verify():
    p = plan()
    index = json.loads((OUT/'EXTRACT_INDEX.json').read_text())
    assert index['plan_sha256'] == sha(OUT/'PLAN.json') and index['script_sha256'] == sha(__file__)
    for row in index['records']:
        assert sha(OUT/row['source']) == row['source_sha256']
        for out in row['outputs']:
            assert sha(ROOT/out['path']) == out['sha256']
    decisions = json.loads((OUT/'DECISIONS.json').read_text())
    allowed = set()
    for symbol, dates in p['missing_dates'].items():
        for d in dates:
            allowed.add((symbol, d))
    assert {(r['symbol'],r['date']) for r in decisions['rows']} == allowed
    assert len(decisions['rows']) == len(allowed)
    for row in decisions['rows']:
        for evidence in row['evidence']:
            assert evidence['text'] in [x['path'] for rec in index['records'] for x in rec['outputs'] if x['path'].endswith('.txt')]
            text = ''.join((ROOT/evidence['text']).read_text().split())
            for needle in evidence['exact_phrases']:
                assert ''.join(needle.split()) in text, needle
    receipts = {item['name']:cached(item) for item in p['requests']}
    write(OUT/'VERIFICATION.json', dict(
        verified_utc=now(), plan_sha256=sha(OUT/'PLAN.json'), decisions_sha256=sha(OUT/'DECISIONS.json'),
        extraction_index_sha256=sha(OUT/'EXTRACT_INDEX.json'), script_sha256=sha(__file__),
        rows_verified=len(allowed), all_exact_evidence_phrases_found=True,
        counts={label:sum(row['classification']==label for row in decisions['rows']) for label in sorted({r['classification'] for r in decisions['rows']})},
        receipt_sha256={item['name']:sha(OUT/(item['name']+'.receipt.json')) for item in p['requests'] if receipts[item['name']]},
        body_sha256={name:r['body_sha256'] for name,r in receipts.items() if r and 'body_sha256' in r},
        prices_filled=False, capitalization_or_rankings_computed=False, strategy_returns_computed=False))
    print('six calendar classifications and source pins verified offline')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['register','collect','extract','verify'])
    args = parser.parse_args()
    globals()[args.mode]()
