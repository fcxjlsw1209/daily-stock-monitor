"""Offline validation of explicitly labeled public-index original-release excerpts.

Direct issuer website returned 403 and is stopped. These are observations from
the public document index, not downloaded PDF bytes. Publication and period
must both appear in the original text; filenames and scheduled dates are not
evidence of an actual release. No network access is performed by this builder.
"""
import calendar
import datetime as dt
import hashlib
import json
import re
from pathlib import Path

P = Path(__file__).resolve().parent
QWORDS = ('first', 'second', 'third', 'fourth')
MONTHS = '|'.join(calendar.month_name[1:])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse(row, evidence):
    opening = re.sub(r'\s+', ' ', row['opening_excerpt'])
    period_text = re.sub(r'\s+', ' ', row.get('period_excerpt', ''))
    if not re.search(r'BERKSHIRE\s+HATHAWAY\s+INC', opening, re.I):
        raise ValueError('Wrong issuer')
    if re.search(r'\b(?:will be|preliminary|information regarding)\b', opening, re.I):
        raise ValueError('Scheduled/preliminary statement is not a regular results release')
    date = re.search(r'FOR\s+IMMEDIATE\s+RELEASE\s+(' + MONTHS + r')\s+(\d{1,2}),?\s+(20\d{2})', opening, re.I)
    if not date:
        raise ValueError('Actual original-release dateline missing')
    released = dt.datetime.strptime(' '.join(date.groups()), '%B %d %Y').date()
    identity_text = opening + ' ' + row.get('quarter_excerpt', '')
    identity = re.search(r'\b(first|second|third|fourth)\s+quarters?(?:\s+and\s+(?:first\s+(?:six|nine)\s+months|full\s+year)|\s+Full\s+Year)?\s+(?:of\s+)?(20\d{2})', identity_text, re.I)
    if not identity:
        raise ValueError('Original current-quarter identity missing')
    quarter, year = QWORDS.index(identity[1].lower()) + 1, int(identity[2])
    month = quarter * 3
    end = dt.date(year, month, calendar.monthrange(year, month)[1])
    period_pattern = r'\b%s\s+%d,?\s+%d\b' % (calendar.month_name[month], end.day, year)
    match = re.search(period_pattern, period_text + ' ' + opening, re.I)
    period_basis = 'Calendar fiscal quarter explicitly named in original release and corresponding period-end date present in original balance-sheet/share/float summary.'
    confirmation = row.get('period_confirmation')
    if not match and confirmation:
        cover = confirmation['opening_excerpt']
        if not re.fullmatch(r'https://www.berkshirehathaway.com/qtrly/[^/]+\.pdf', confirmation['url']):
            raise ValueError('Period confirmation must reference official quarterly report')
        if not re.search(r'BERKSHIRE\s+HATHAWAY\s+INC', cover, re.I):
            raise ValueError('Period confirmation issuer mismatch')
        if confirmation['release_url'] != row['url'] or 'FORM 10-Q' not in cover:
            raise ValueError('Period confirmation identity mismatch')
        match = re.search(r'quarterly period ended\s+(' + period_pattern + ')', cover, re.I)
        period_basis = 'Original release names current fiscal quarter; separately indexed issuer-hosted 10-Q cover confirms actual period end. Filing date is not used as publication date.'
    if not match:
        raise ValueError('Actual fiscal-period date not observed in original text')
    if not 0 < (released-end).days < 100 or not '2011-01-01' <= str(released) <= '2026-09-24':
        raise ValueError('Release outside expected lag/research cutoff')
    if not re.search(r'Omaha,?\s+NE', opening, re.I):
        raise ValueError('Original issuer-local dateline missing')
    return dict(ticker='BRK.B', fiscal_year=year, fiscal_quarter=quarter,
        fiscal_period_end_actual=str(end), release_date=str(released),
        publication_timestamp=None, precision='date_only', publication_timezone='America/Chicago',
        publication_timezone_basis='Explicit conservative issuer-local date policy from original Omaha, NE dateline; not CMS timezone or a publication clock time.',
        source_url=row['url'], source_kind=row['source_kind'], evidence_path=str(evidence),
        evidence_sha256=sha(evidence), evidence_excerpt=dict(dateline=date[0],
            quarter_identity=identity[0], observed_period=match[0], original_excerpt=row),
        evidence_limitation='Public index observation of original issuer release text; original PDF bytes were not directly downloaded.',
        first_public_time_verified=False,
        period_basis=period_basis,
        publication_version='corrected_current_archive' if re.search('CORRECTED NEWS RELEASE', opening, re.I) else 'current_archive')


def main():
    events, pending, urls = [], [], set()
    for file in sorted((P/'raw').glob('indexed_releases_*.json')):
        payload = json.loads(file.read_text())
        for row in payload['observations']:
            if row['url'] in urls:
                continue
            try:
                event = parse(row, file)
                events.append(event); urls.add(row['url'])
            except ValueError as exc:
                pending.append(dict(source_url=row['url'], reason=str(exc), evidence_path=str(file)))
    events.sort(key=lambda r: r['release_date'])
    quarters = [(r['fiscal_year'], r['fiscal_quarter']) for r in events]
    if len(set(quarters)) != len(quarters):
        raise ValueError('Multiple versions for a fiscal quarter require manual reconciliation')
    target = [(y,q) for y in range(2010,2027) for q in range(1,5)
              if (2010,4) <= (y,q) <= (2026,2)]
    missing = ['%dQ%d' % pair for pair in target if pair not in quarters]
    # Failed earlier views are resolved when a later original-text observation
    # supplies the missing field. Keep the raw attempt but do not double count.
    unresolved = [r for r in pending if r['source_url'] not in urls]
    (P/'events.json').write_text(json.dumps(events, indent=2, ensure_ascii=False)+'\n')
    (P/'PENDING.json').write_text(json.dumps(dict(missing_target_quarters=missing,
        unaccepted_observations=unresolved, resolved_earlier_observations=len(pending)-len(unresolved)), indent=2)+'\n')
    validation = dict(events=len(events), earliest_release=events[0]['release_date'] if events else None,
        latest_release=events[-1]['release_date'] if events else None,
        target_quarters=len(target), missing_target_quarters=missing,
        complete_since_2015=all(p in quarters for p in target if p >= (2014,4)),
        duplicate_quarters=0, source_kind='public_index_observations_not_original_pdf_bytes',
        direct_source_stop=json.loads((P/'SOURCE_STOPS.json').read_text()),
        events_sha256=sha(P/'events.json'))
    (P/'VALIDATION.json').write_text(json.dumps(validation, indent=2)+'\n')
    print(json.dumps(validation))


if __name__ == '__main__':
    main()
