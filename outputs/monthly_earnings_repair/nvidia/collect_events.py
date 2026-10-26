"""Parse each enumerated original quarterly release, preserving HTML evidence.

Use publication text only. Neither conference-call time nor page-modified time
is interpreted as announcement time. Exact wire times are reused only from
the previously verified, separately cached publisher observations.
"""
from datetime import datetime
from pathlib import Path
import json
import re
from bs4 import BeautifulSoup
from collector import P, fetch, read, write, now, digest

MONTHS = 'January|February|March|April|May|June|July|August|September|October|November|December|Jan\\.?|Feb\\.?|Mar\\.?|Apr\\.?|Jun\\.?|Jul\\.?|Aug\\.?|Sep\\.?|Sept\\.?|Oct\\.?|Nov\\.?|Dec\\.?'
DATE = r'((?:' + MONTHS + r')\s+\d{1,2},?\s+\d{4})'
QUARTERS = {'first': 1, 'second': 2, 'third': 3, 'fourth': 4}
AS_OF = '2026-10-12'


def normalized(text):
    return re.sub(r'\s+', ' ', text).strip()


def parse_date(text):
    cleaned = normalized(text).replace(',', '').replace('.', '').replace('Sept ', 'Sep ')
    for pattern in ('%B %d %Y', '%b %d %Y'):
        try:
            return datetime.strptime(cleaned, pattern).date().isoformat()
        except ValueError:
            pass
    raise ValueError('Unrecognized date literal: ' + text)


def parse_event(record, text, catalog_entry):
    soup = BeautifulSoup(text, 'html.parser')
    title_node, date_node, body_node = soup.select_one('.article-title'), soup.select_one('.article-date'), soup.select_one('.article-body')
    if not all([title_node, date_node, body_node]):
        raise ValueError('Missing original announcement title/date/body selectors')
    title = normalized(title_node.get_text(' ', strip=True))
    day_text = normalized(date_node.get_text(' ', strip=True))
    date_match = re.fullmatch(DATE, day_text)
    if not date_match:
        raise ValueError('Article date is not an unambiguous date: ' + day_text)
    release_date = parse_date(day_text)
    fiscal = re.search(r'(first|second|third|fourth)\s+quarter\s+(?:and\s+)?fiscal\s+(\d{4})', title, re.I)
    if not fiscal:
        raise ValueError('Quarter/year cannot be established from title: ' + title)
    body = normalized(body_node.get_text(' ', strip=True))
    ended = re.search(r'\bended\s+' + DATE, body[:5000], re.I)
    if not ended:
        raise ValueError('No actual quarter-ended date in opening release text')
    period = parse_date(ended.group(1))
    if not period < release_date:
        raise ValueError('Actual fiscal end must precede release')
    if (datetime.fromisoformat(release_date) - datetime.fromisoformat(period)).days > 100:
        raise ValueError('Quarter-ended evidence is too distant from publication')
    excerpt = body[max(0, ended.start()-130):ended.end()+50]
    location = re.search(r'SANTA CLARA, (?:CA|Calif\.)', body, re.I)
    location_source = record if location else read('manifest.json', {})['https://nvidianews.nvidia.com/news/nvidia-announces-financial-results-for-fourth-quarter-and-fiscal-2015']
    event = dict(ticker='NVDA', release_date=release_date, publication_timestamp=None,
        publication_timezone='America/Los_Angeles',
        publication_timezone_evidence=dict(
            basis='explicit_issuer_local_calendar_date_policy',
            policy='Interpret a date-only issuer newsroom date in the issuer dateline locality, Santa Clara, California; this is a documented research calendar policy, not an observed intraday timestamp or proof of newsroom CMS timezone.',
            literal=location.group(0) if location else 'SANTA CLARA, CA',
            dateline_on_this_release=bool(location), source_url=location_source['url'],
            evidence_path=location_source['evidence_path'], sha256=location_source['sha256']),
        precision='date_only', fiscal_period_end_actual=period,
        fiscal_year=int(fiscal.group(2)), fiscal_quarter=QUARTERS[fiscal.group(1).lower()],
        source_url=record['url'], evidence_path=record['evidence_path'],
        evidence_sha256=record['sha256'], source_kind='issuer_original_newsroom',
        source_title=title, retrieved_at=record['retrieved_at'],
        release_date_evidence=dict(selector='.article-date', literal=day_text),
        fiscal_period_evidence=dict(selector='.article-body', literal=ended.group(1), context=excerpt),
        archive_source_urls=catalog_entry['archive_sources'],
        conference_call_time_used_as_publication=False, modified_time_used_as_publication=False,
        global_first_public_time_verified=False)
    # Date-only issuer source is sufficient; this parser does not infer midnight.
    return event


def add_known_wire_times(events):
    old = P.parents[1] / 'monthly_cost_portfolio_research/events/VALIDATED_EVENTS.json'
    for previous in json.loads(old.read_text())['events']:
        if previous.get('symbol') != 'NVDA' or not previous.get('exact_publication_timestamp'):
            continue
        matched = [event for event in events if event['release_date'] == previous['actual_release_date']]
        for event in matched:
            assert event['fiscal_period_end_actual'] == previous['issuer_fiscal_period_end']
            wire = [source for source in previous['sources'] if 'observation_cache' in source]
            assert len(wire) == 1
            observation = P / Path(wire[0]['observation_cache']).name
            assert observation.exists() and digest(observation.read_bytes()) == wire[0]['sha256']
            event.update(publication_timestamp=previous['exact_publication_timestamp'],
                publication_timezone='America/New_York',
                publication_timezone_evidence=dict(basis='original_publisher_explicit_ET',
                    source_url=wire[0]['url'], evidence_path=observation.name,
                    sha256=digest(observation.read_bytes()), literal=json.loads(observation.read_text())['excerpt']),
                precision='publisher_timestamp_minute', publication_timestamp_utc=previous['publication_timestamp_utc'],
                timestamp_meaning='Issuer-distributed original newswire publication; not proof of earliest disclosure across channels',
                timestamp_evidence=dict(source_url=wire[0]['url'], evidence_path=observation.name,
                    sha256=digest(observation.read_bytes()), reuse_source=wire[0]['observation_cache']))


def main():
    catalog = read('release_catalog.json', {})
    assert catalog.get('completed_pages') == catalog.get('archive_last_page')
    events, issues, outside = [], [], []
    for entry in catalog['releases']:
        record, text = fetch(entry['url'])
        if record['http_status'] != 200:
            issues.append(dict(url=entry['url'], problem='http_not_200', status=record['http_status']))
        else:
            try:
                event = parse_event(record, text, entry)
            except ValueError as exc:
                issues.append(dict(url=entry['url'], problem=str(exc)))
            else:
                if '2015-01-01' <= event['release_date'] <= AS_OF:
                    events.append(event)
                else:
                    outside.append(event)
        write('parse_progress.json', dict(parsed_events=len(events), issues=issues, outside_scope=outside))
        print('release', len(events), entry['url'].rsplit('/', 1)[-1], 'issues', len(issues), flush=True)
    add_known_wire_times(events)
    events.sort(key=lambda event: (event['release_date'], event['source_url']))
    write('events.json', events)
    write('parse_issues.json', issues)
    write('outside_scope_events.json', outside)
    print('Events', len(events), 'issues', len(issues), 'outside', len(outside), flush=True)


if __name__ == '__main__':
    main()
