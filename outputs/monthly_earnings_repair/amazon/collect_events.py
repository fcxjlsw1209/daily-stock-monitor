"""Conservatively parse issuer original release datelines, not migrated headers.

IR and SEC hosts blocked by HTTP403 stay pending; filing/acceptance dates are
never announcement evidence. Cached successful pages can be reparsed offline.
"""
from datetime import datetime
import re
from bs4 import BeautifulSoup
from collector import P, fetch, read, write, now

MONTHS = 'January|February|March|April|May|June|July|August|September|October|November|December|Jan\\.?|Feb\\.?|Mar\\.?|Apr\\.?|Jun\\.?|Jul\\.?|Aug\\.?|Sep\\.?|Sept\\.?|Oct\\.?|Nov\\.?|Dec\\.?'
DATE = r'((?:' + MONTHS + r')\s+\d{1,2},?\s+\d{4})'
QUARTERS = {'first': 1, 'second': 2, 'third': 3, 'fourth': 4}
RESEARCH_CUTOFF = '2026-09-24'


def normalized(text):
    return re.sub(r'\s+', ' ', text).strip()


def parse_date(text):
    cleaned = normalized(text).replace(',', '').replace('.', '').replace('Sept ', 'Sep ')
    for pattern in ('%B %d %Y', '%b %d %Y'):
        try:
            return datetime.strptime(cleaned, pattern).date().isoformat()
        except ValueError:
            pass
    raise ValueError('Unrecognized date: ' + text)


def parse_event(record, text, entry):
    soup = BeautifulSoup(text, 'html.parser')
    headline = soup.select_one('.PressReleasePage-headline')
    body_node = soup.select_one('.PressReleasePage-articleBody')
    display_node = soup.select_one('.PressReleasePage-datePublished')
    if not all([headline, body_node, display_node]):
        raise ValueError('Missing original announcement selectors')
    body = normalized(body_node.get_text(' ', strip=True))
    title = normalized(headline.get_text(' ', strip=True))
    dateline = re.search(r'SEATTLE\s*[-–—]+\s*\(BUSINESS WIRE\)\s*[-–—]+\s*' + DATE, body[:2000], re.I)
    if not dateline:
        raise ValueError('Original body dateline missing/truncated; migrated header/title dates not accepted alone')
    release_date = parse_date(dateline.group(1))
    ended = re.search(r'(first|second|third|fourth)\s+quarter\s+ended\s+' + DATE, body[:2500], re.I)
    if not ended:
        raise ValueError('Actual quarter-ended date not established in original opening text')
    period = parse_date(ended.group(2))
    quarter = QUARTERS[ended.group(1).lower()]
    if int(period[5:7]) != quarter * 3 or period[8:] not in ('30', '31'):
        raise ValueError('Fiscal quarter and period end disagree')
    lag = (datetime.fromisoformat(release_date) - datetime.fromisoformat(period)).days
    if not 0 < lag < 100:
        raise ValueError('Quarter ended date does not precede release within 100 days')
    displayed = parse_date(display_node.get_text(' ', strip=True))
    return dict(ticker='AMZN', release_date=release_date, publication_timestamp=None,
        precision='date_only', publication_timezone='America/Los_Angeles',
        publication_timezone_evidence=dict(basis='issuer_dateline_local_calendar_date_policy',
            policy='Original SEATTLE dateline date is interpreted in America/Los_Angeles; no intraday timestamp or CMS timezone is inferred.',
            literal=dateline.group(0), source_url=record['url'], evidence_path=record['evidence_path'], sha256=record['sha256']),
        fiscal_period_end_actual=period, fiscal_year=int(period[:4]), fiscal_quarter=quarter,
        source_url=record['url'], evidence_path=record['evidence_path'], evidence_sha256=record['sha256'],
        source_kind='issuer_original_press_release', source_title=title, retrieved_at=record['retrieved_at'],
        release_date_evidence=dict(selector='.PressReleasePage-articleBody', literal=dateline.group(1), context=dateline.group(0)),
        fiscal_period_evidence=dict(selector='.PressReleasePage-articleBody', literal=ended.group(2), context=ended.group(0)),
        migrated_header_date=displayed, migrated_header_date_conflicts_with_original=displayed != release_date,
        archive_source_urls=entry['archive_sources'], eligible_at_research_cutoff=release_date <= RESEARCH_CUTOFF,
        conference_call_time_used_as_publication=False, modified_time_used_as_publication=False,
        filing_date_used_as_publication=False, acceptance_timestamp_used_as_publication=False,
        global_first_public_time_verified=False)


def main():
    events, pending = [], []
    for entry in read('release_catalog.json', {})['releases']:
        if entry.get('malformed_external_ir_reference'):
            pending.append(dict(url=entry['referenced_ir_url'], title=entry['title'],
                reason='IR host stopped after HTTP403; malformed press archive external reference not fetched',
                calendar_release_year=entry['calendar_release_year']))
            continue
        try:
            record, text = fetch(entry['url'])
            if record['http_status'] != 200:
                raise ValueError('HTTP status is not 200: ' + str(record['http_status']))
            event = parse_event(record, text, entry)
            if not '2015-01-01' <= event['release_date'] <= RESEARCH_CUTOFF:
                raise ValueError('Outside requested release date scope')
            events.append(event)
        except (ValueError, RuntimeError) as exc:
            pending.append(dict(url=entry['url'], title=entry['title'], reason=str(exc),
                calendar_release_year=entry['calendar_release_year']))
        write('parse_progress.json', dict(updated_at=now(), accepted=len(events), pending=pending))
    events.sort(key=lambda event: (event['release_date'], event['source_url']))
    write('events.json', events)
    write('pending.json', pending)
    print('Accepted:', len(events), 'Pending catalog entries:', len(pending), flush=True)


if __name__ == '__main__':
    main()
