"""Bounded source collection for ABT's three absent 2020 quarterly events."""
import argparse
import urllib.error
import urllib.request
from pathlib import Path
from bisect import bisect_right
from after155_broadcom_gap_source import Text, digest, read, save, utc, ROOT

OUT = ROOT / 'outputs/monthly_event_gap156_abbott'
ITEMS = [
    ('q1', '2020-04-16', '2020-03-31', 'first quarter ended March 31, 2020', 'https://abbott.mediaroom.com/2020-04-16-Abbott-Reports-First-Quarter-2020-Results'),
    ('q2', '2020-07-16', '2020-06-30', 'second quarter ended June 30, 2020', 'https://abbott.mediaroom.com/2020-07-16-Abbott-Reports-Second-Quarter-2020-Results-Exceeds-Analysts-Expectations'),
    ('q3', '2020-10-21', '2020-09-30', 'third quarter ended Sept. 30, 2020', 'https://abbott.mediaroom.com/2020-10-21-Abbott-Reports-Third-Quarter-2020-Results-Achieves-Strong-Double-Digit-Earnings-Growth-and-Raises-Guidance'),
]


def freeze():
    save(OUT / 'SOURCE_SPEC.json', {
        'created_utc': utc(), 'items': ITEMS,
        'selection': 'ABT quarterly gaps inside 2021-01 formation scope; no use of holdings or returns',
        'inputs': {rel: digest((ROOT / rel).read_bytes()) for rel in (
            'outputs/monthly_event_clock_census155/SLOTS.json',
            'outputs/monthly_earnings_repair/EVENTS.json',
            'outputs/monthly_paired_target143/CALENDAR.json',
            'work/after155_broadcom_gap_source.py',
            'work/after155_abbott_gap_source.py')},
        'policy': 'One request per listed original issuer URL; no retry; any429 stops remaining source requests; save Retry-After.',
        'source_only': True, 'strategy_trials': 0,
    })


def spec():
    value = read(OUT / 'SOURCE_SPEC.json')
    for rel, expected in value['inputs'].items():
        assert digest((ROOT / rel).read_bytes()) == expected, rel
    return value


def collect():
    value = spec(); receipts = []
    assert not (OUT / 'COLLECTION.json').exists(), 'Reuse prior attempt'
    for name, pub, end, marker, url in value['items']:
        row = {'name': name, 'url': url, 'retrieved_utc': utc()}
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (compatible; historical-research/1.0)'})
            with urllib.request.urlopen(req, timeout=35) as response:
                blob = response.read(8 * 1024 * 1024 + 1)
                assert len(blob) <= 8 * 1024 * 1024
                row.update(http_status=response.status, final_url=response.url,
                           content_type=response.headers.get('Content-Type'))
            dest = OUT / (name + '.html')
            with dest.open('xb') as f: f.write(blob)
            row.update(status='downloaded', path=str(dest.relative_to(ROOT)), sha256=digest(blob), bytes=len(blob))
        except urllib.error.HTTPError as error:
            row.update(status='http_error', http_status=error.code, retry_after=error.headers.get('Retry-After'))
        except (urllib.error.URLError, TimeoutError) as error:
            row.update(status='network_error', error_type=type(error).__name__, error=str(error))
        save(OUT / (name + '.receipt.json'), row); receipts.append(row)
        if row.get('http_status') == 429: break
    save(OUT / 'COLLECTION.json', {'created_utc': utc(), 'receipts': receipts})
    print([(r['name'], r['status'], r.get('http_status')) for r in receipts])


def inspect():
    value = spec(); rows = []
    calendar = read(ROOT / 'outputs/monthly_paired_target143/CALENDAR.json')['sessions']
    for item, receipt in zip(value['items'], read(OUT / 'COLLECTION.json')['receipts']):
        name, pub, end, marker, url = item
        assert receipt['name'] == name
        row = dict(receipt)
        if row['status'] == 'downloaded':
            blob = (ROOT / row['path']).read_bytes(); assert digest(blob) == row['sha256']
            parser = Text(); parser.feed(blob.decode('utf-8', errors='replace'))
            plain = ' '.join(' '.join(parser.parts).split())
            hit = plain.find(marker)
            row.update(period_end=end if hit >= 0 else None, period_marker_found=hit >= 0,
                       actual_announcement_language='today announced financial results' in plain,
                       issuer_marker='Abbott (NYSE: ABT)' in plain,
                       marker_offset=hit, short_evidence=marker if hit >= 0 else None,
                       proposed_publication_date=pub,
                       proposed_available_session=calendar[bisect_right(calendar, pub)],
                       independent_dateline_check_required=True)
        rows.append(row)
    save(OUT / 'INSPECTION.json', {'created_utc': utc(), 'results': rows,
         'events_admitted': False, 'calendar_complete': False, 'strategy_trials': 0})
    print([(r['name'], r.get('period_marker_found')) for r in rows])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('phase', choices=('freeze', 'collect', 'inspect'))
    {'freeze': freeze, 'collect': collect, 'inspect': inspect}[ap.parse_args().phase]()
