"""Cache actual Microsoft releases; URL templates are discovery, not evidence."""
import datetime as dt
import hashlib
import json
import re
import time
from pathlib import Path
from email.utils import parsedate_to_datetime
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup

P = Path(__file__).resolve().parent
MONTH = r'(January|February|March|April|May|June|July|August|September|October|November|December)'
DATE = MONTH + r'\s+\d{1,2},\s+\d{4}'
TEMPLATE = 'https://www.microsoft.com/en-us/Investor/earnings/FY-{year}-Q{quarter}/press-release-webcast'


def parse(body, year, quarter, url, cache, checksum):
    soup = BeautifulSoup(body, 'html.parser')
    text = re.sub(r'\s+', ' ', ' '.join(soup.stripped_strings))
    title = re.search(r'Earnings Release FY(\d{2}) Q([1-4])', text, re.I)
    if not title or (int(title[1]) + 2000, int(title[2])) != (year, quarter):
        raise ValueError('Wrong/missing fiscal release identity')
    # Date in the actual release's Redmond dateline, not a call or page metadata.
    hit = re.search(r'REDMOND,\s+Wash\.(?:\s|—|–|-)*(' + DATE + r')', text, re.I)
    if not hit:
        raise ValueError('Missing original release dateline')
    date = dt.datetime.strptime(re.sub(r'\s+', ' ', hit[1]), '%B %d, %Y').date()
    after = text[hit.end():hit.end() + 1800]
    period = re.search(r'(?:quarter|fiscal year) ended\s+(' + DATE + r')', after, re.I)
    if not period:
        raise ValueError('Missing period ended in original release')
    end = dt.datetime.strptime(re.sub(r'\s+', ' ', period[1]), '%B %d, %Y').date()
    if end > date:
        raise ValueError('Future fiscal period')
    expected_period = [dt.date(year-1, 9, 30), dt.date(year-1, 12, 31),
                       dt.date(year, 3, 31), dt.date(year, 6, 30)][quarter-1]
    if end != expected_period:
        raise ValueError('Actual fiscal-period end conflicts with Microsoft fiscal year/quarter')
    return dict(ticker='MSFT', release_date=str(date), publication_timestamp=None,
                precision='date_only', publication_timezone='America/Los_Angeles',
                publication_timezone_basis='Release dateline REDMOND, Wash.; conservative issuer-local calendar date policy, not a claimed clock time.',
                fiscal_year=year, fiscal_quarter=quarter,
                fiscal_period_end_actual=str(end), source_url=url,
                evidence_path=str(cache), sha256=checksum,
                date_literal=hit[1], period_literal=period[1],
                evidence_kind='issuer_original_release',
                conference_call_time_used=False)


def retry_not_before(value, now):
    if not value: return None
    try:
        return (now + dt.timedelta(seconds=max(0, int(value)))).isoformat()
    except (ValueError, TypeError):
        try:
            return parsedate_to_datetime(value).astimezone(dt.timezone.utc).isoformat()
        except (ValueError, TypeError, OverflowError):
            return None


def is_stopped(state, now):
    # LA and UTC day both respected; a longer Retry-After takes precedence.
    today = state.get('date') in (str(now.date()), str(now.astimezone(ZoneInfo('America/Los_Angeles')).date()))
    until = state.get('retry_not_before')
    return today or bool(until and dt.datetime.fromisoformat(until) > now)


def main():
    raw = P / 'raw'; raw.mkdir(exist_ok=True)
    events, errors, manifest = [], [], []
    stop = P / 'STOP.json'
    now = dt.datetime.now(dt.timezone.utc)
    blocked = stop.exists() and is_stopped(json.loads(stop.read_text()), now)
    for year in range(2015, 2027):
        for q in range(1, 5):
            url = TEMPLATE.format(year=year, quarter=q)
            file = raw / ('FY%dQ%d.html' % (year, q))
            meta = file.with_suffix('.meta.json')
            if not file.exists():
                if blocked:
                    errors.append(dict(year=year, quarter=q, reason='source_stopped')); continue
                time.sleep(1.1)
                try:
                    r = requests.get(url, timeout=35)
                except requests.RequestException:
                    errors.append(dict(year=year, quarter=q, reason='network_failure')); continue
                file.write_bytes(r.content)
                meta.write_text(json.dumps(dict(url=url, http_status=r.status_code,
                    sha256=hashlib.sha256(r.content).hexdigest(),
                    retrieved_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                    retry_after=r.headers.get('Retry-After')), indent=2) + '\n')
                if r.status_code in (403, 429):
                    blocked = True
                    stopped_at = dt.datetime.now(dt.timezone.utc)
                    stop.write_text(json.dumps(dict(date=str(stopped_at.date()), http_status=r.status_code,
                        received_at=stopped_at.isoformat(),
                        retry_after=r.headers.get('Retry-After'),
                        retry_not_before=retry_not_before(r.headers.get('Retry-After'), stopped_at)), indent=2))
            m = json.loads(meta.read_text()); manifest.append(m)
            if hashlib.sha256(file.read_bytes()).hexdigest() != m['sha256']:
                raise ValueError('Cache checksum mismatch')
            if m['http_status'] != 200:
                errors.append(dict(year=year, quarter=q, reason='http_%s' % m['http_status'])); continue
            try:
                event = parse(file.read_text(), year, q, url, file, m['sha256'])
                if '2015-01-01' <= event['release_date'] <= str(now.date()): events.append(event)
            except ValueError as exc:
                errors.append(dict(year=year, quarter=q, reason=str(exc)))
            print(year, q, 'events', len(events), 'errors', len(errors), flush=True)
    (P / 'events.json').write_text(json.dumps(events, indent=2) + '\n')
    (P / 'MANIFEST.json').write_text(json.dumps(dict(source_template=TEMPLATE,
        discovery='Template observed on official FY2025Q2; every page identity and date parsed before acceptance.',
        failure_cache_policy='Failed responses retained for audit, not silently retried; a later repair must explicitly archive the attempt before requesting again.',
        events=len(events), errors=errors, responses=manifest), indent=2) + '\n')


if __name__ == '__main__':
    main()
