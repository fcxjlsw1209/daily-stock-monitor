"""Issuer-hosted original PDF supplements, with cached evidence and no date guessing."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'microsoft'))
from collect import retry_not_before, is_stopped

P = Path(__file__).resolve().parent
D = P / 'supplement'
RUNTIME = '/Users/shiweili/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3'
MONTH = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)'
DATE = MONTH + r'\s+\d{1,2},\s+\d{4}'
BASE = 'https://s2.q4cdn.com/299287126/files/'


def sha(b): return hashlib.sha256(b).hexdigest()


def dump(name, x):
    (D/name).write_text(json.dumps(x, indent=2, ensure_ascii=False)+'\n')


def fetch(url):
    if not url.startswith(BASE): raise ValueError('Not the observed official Amazon PDF directory')
    key = sha(url.encode())[:24]
    file, meta = D/(key+'.pdf'), D/(key+'.meta.json')
    if meta.exists():
        m = json.loads(meta.read_text())
        if sha(file.read_bytes()) != m['sha256']: raise ValueError('Cached PDF checksum mismatch')
        if m['status'] != 200: raise ValueError('Cached HTTP failure: '+str(m['status']))
        return file, m
    stop = D/'STOP.json'
    now = dt.datetime.now(dt.timezone.utc)
    if stop.exists() and is_stopped(json.loads(stop.read_text()), now):
        raise RuntimeError('Q4 CDN stopped; honor same-day restriction and Retry-After')
    time.sleep(1.1)
    request = Request(url, headers={'User-Agent':'Mozilla/5.0 (public historical earnings research)'})
    try:
        with urlopen(request, timeout=30) as r:
            body, status, headers = r.read(), r.status, r.headers
    except HTTPError as e:
        body, status, headers = e.read(), e.code, e.headers
    received = dt.datetime.now(dt.timezone.utc)
    file.write_bytes(body)
    m = dict(url=url, status=status, retrieved_at=received.isoformat(), sha256=sha(body),
             evidence_path=str(file), retry_after=headers.get('Retry-After'))
    meta.write_text(json.dumps(m, indent=2)+'\n')
    if status in (403,429):
        dump('STOP.json', dict(date=str(received.date()), received_at=received.isoformat(),
            status=status, retry_not_before=retry_not_before(headers.get('Retry-After'), received)))
        raise RuntimeError('HTTP%d source stopped' % status)
    if status != 200 or not body.startswith(b'%PDF'): raise ValueError('Not a successful PDF response')
    return file, m


def pdf_text(file):
    result = subprocess.run([RUNTIME, '-c',
        'from pypdf import PdfReader; import sys; print("\\n".join(p.extract_text() for p in PdfReader(sys.argv[1]).pages))',
        str(file)], text=True, capture_output=True, check=True)
    file.with_suffix('.txt').write_text(result.stdout)
    return result.stdout


def parse(text, file, m, expected):
    text = re.sub(r'\s+', ' ', text).strip()
    if not re.search(r'AMAZON\.COM\s+ANNOUNCES', text[:1200], re.I): raise ValueError('Wrong issuer release')
    line = re.search(r'SEATTLE\s*[—–-]+\s*\(BUSINESS\s+WIRE\)\s*[—–-]*\s*('+DATE+')', text[:2000], re.I)
    ended = re.search(r'(first|second|third|fourth)\s+quarter\s+ended\s+('+DATE+')', text[:2500], re.I)
    if not line or not ended: raise ValueError('Original dateline or quarter ended missing')
    day = dt.datetime.strptime(line[1], '%B %d, %Y').date()
    period = dt.datetime.strptime(ended[2], '%B %d, %Y').date()
    quarter = ['first','second','third','fourth'].index(ended[1].lower())+1
    if (period.year, quarter) != tuple(expected): raise ValueError('Fiscal identity mismatch')
    expected_end = [dt.date(period.year,3,31),dt.date(period.year,6,30),dt.date(period.year,9,30),dt.date(period.year,12,31)][quarter-1]
    if period != expected_end or not 0 < (day-period).days < 100: raise ValueError('Fiscal end or release lag invalid')
    if not dt.date(2015,1,1) <= day <= dt.date(2026,9,24): raise ValueError('Outside research release scope')
    return dict(ticker='AMZN', release_date=str(day), publication_timestamp=None, precision='date_only',
        publication_timezone='America/Los_Angeles',
        publication_timezone_basis='Original SEATTLE dateline: conservative issuer-local calendar date policy; exact first publication not inferred.',
        fiscal_period_end_actual=str(period), fiscal_year=period.year, fiscal_quarter=quarter,
        source_url=m['url'], evidence_path=str(file), evidence_sha256=m['sha256'],
        source_kind='issuer_hosted_original_earnings_release_pdf', retrieved_at=m['retrieved_at'],
        release_date_evidence=line[0], fiscal_period_evidence=ended[0],
        source_discovery='Official Amazon Q4 CDN publication; URL discovery pattern is not accepted without PDF issuer, dateline and quarter validation.',
        conference_call_time_used_as_publication=False, modified_time_used_as_publication=False,
        filing_date_used_as_publication=False, acceptance_timestamp_used_as_publication=False)


def candidates():
    # First request an already observed PDF; all dates come from content, not filenames.
    yield (2025,4), BASE+'doc_earnings/2025/q4/earnings-result/AMZN-Q4-2025-Earnings-Release.pdf'
    yield (2021,2), BASE+'doc_financials/2021/q2/AMZN-Q2-2021-Earnings-Release.pdf'
    yield (2022,4), BASE+'doc_financials/2022/q4/Q4-2022-Amazon-Earnings-Release.pdf'
    for y in range(2023,2027):
        for q in range(1,5 if y<2026 else 3):
            if (y,q)==(2025,4): continue
            name = 'Q%d-%d-Amazon-Earnings-Release.pdf' % (q,y) if y==2023 and q<3 else 'AMZN-Q%d-%d-Earnings-Release.pdf' % (q,y)
            yield (y,q), BASE+'doc_financials/%d/q%d/'%(y,q)+name
    yield (2022,3), BASE+'doc_financials/2022/q3/Q3-2022-Amazon-Earnings-Release.pdf'
    for q in [1,2]:
        yield (2026,q), BASE+'doc_earnings/2026/q%d/earnings-result/AMZN-Q%d-2026-Earnings-Release.pdf' % (q,q)


def legacy_events():
    file = D/'legacy_observations.json'
    if not file.exists(): return []
    events = []
    for row in json.loads(file.read_text()):
        match = re.search(r'/amzn-(\d{4})(\d{2})(\d{2})xex991.htm$', row['source_url'])
        if not match: raise ValueError('Unexpected original SEC exhibit identity')
        # Filename is an expected fiscal identity, never publication evidence.
        expected = (int(match[1]), int(match[2])//3)
        m = dict(url=row['source_url'], sha256=sha(file.read_bytes()), retrieved_at=row['observed_at_utc'])
        event = parse(' '.join(row['observed_lines']), file, m, expected)
        event['source_kind'] = 'public_index_observation_of_original_SEC_exhibit_not_full_html'
        event['source_discovery'] = 'Original exhibit body observed via web tool; only headline/dateline/period excerpt retained. This is not a raw HTML download or a filing-date-derived release.'
        events.append(event)
    return events


def main():
    D.mkdir(exist_ok=True)
    events, pending = [], []
    for expected,url in candidates():
        try:
            f,m = fetch(url); e=parse(pdf_text(f),f,m,expected); events.append(e)
            print('accepted',expected,e['release_date'],flush=True)
        except Exception as ex:
            pending.append(dict(quarter=expected,url=url,reason=str(ex)))
            print('pending',expected,str(ex),flush=True)
        dump('events.json',sorted(events,key=lambda e:e['release_date']))
        dump('PENDING.json',pending)
    events += legacy_events()
    if len({(e['fiscal_year'],e['fiscal_quarter']) for e in events}) != len(events):
        raise ValueError('Duplicate fiscal quarter in supplements')
    dump('events.json',sorted(events,key=lambda e:e['release_date']))
    print('Verified supplements:',len(events),flush=True)


if __name__=='__main__':main()
