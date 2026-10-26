"""Independent, offline SEC filing-clock metadata audit. Never reads price arrays.

The Eastern conversion below is a hypothesis comparator, not an assertion that
SEC's submissions Z field or an unverified header uses any particular timezone.
"""
import argparse
from collections import Counter
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
OUT = 'work/after156_filing_clock_independent.json'
FEAS = 'work/after156_filing_event_feasibility.json'
FEAS_SHA = '1af27cbd0777d6e0542fe56c8306efc3b747e018ac00bd7e70d7400c99abc6de'
STOP = 'work/after155_sec_gap_discovery_collection.json'
BASE = 'outputs/dynamic_top100/registry_sec/'
PILOT = (('AAPL', '0000320193'), ('MSFT', '0000789019'), ('ABBV', '0001551152'))
FIELDS = ('accessionNumber', 'form', 'filingDate', 'reportDate', 'acceptanceDateTime', 'primaryDocument')
NY = ZoneInfo('America/New_York')
PREFIX_BYTES = 65536


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


class Inputs:
    def __init__(self):
        self.hashes = {}

    def read(self, relative, expected=None):
        path = ROOT / relative
        require(path.resolve().is_relative_to(ROOT), 'Path escaped research root')
        raw = path.read_bytes()
        sha = digest(raw)
        require(expected is None or sha == expected, 'Source digest changed: ' + relative)
        require(relative not in self.hashes or self.hashes[relative] == sha, 'Input mutation')
        self.hashes[relative] = sha
        return raw

    def obj(self, relative, expected=None):
        return json.loads(self.read(relative, expected))

    def finish(self):
        for relative, sha in list(self.hashes.items()):
            self.read(relative, sha)


def utc_syntax(raw):
    require(isinstance(raw, str) and re.fullmatch(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d{1,6})?Z', raw),
            'Expected preserved Z timestamp syntax')
    return datetime.fromisoformat(raw[:-1] + '+00:00')


def local_candidate(raw):
    """Resolve a hypothetical Eastern wall clock; reject DST gaps/folds."""
    require(isinstance(raw, str) and re.fullmatch(r'\d{14}', raw), 'Header clock syntax')
    wall = datetime.strptime(raw, '%Y%m%d%H%M%S')
    candidates = set()
    for fold in (0, 1):
        point = wall.replace(tzinfo=NY, fold=fold).astimezone(timezone.utc)
        if point.astimezone(NY).replace(tzinfo=None) == wall:
            candidates.add(point)
    require(len(candidates) == 1, 'Ambiguous or nonexistent hypothetical Eastern wall clock')
    return candidates.pop()


def parse_header(raw, accession):
    text = raw.decode('ascii', errors='replace')
    ids = re.findall(r'ACCESSION NUMBER:\s*(\d{10}-\d{2}-\d{6})', text)
    clocks = re.findall(r'<ACCEPTANCE-DATETIME>\s*(\d{14})(?!\d)', text)
    require(ids == [accession] and len(clocks) == 1, 'Missing, duplicate or mismatched header identity/clock')
    local_candidate(clocks[0])
    return clocks[0]


def compare_clock(raw_z, wall):
    literal = utc_syntax(raw_z)
    as_eastern = local_candidate(wall)
    same_wall = literal.replace(tzinfo=None) == datetime.strptime(wall, '%Y%m%d%H%M%S')
    return dict(literal_z_matches_hypothetical_eastern_instant=literal == as_eastern,
                literal_z_repeats_header_wall_clock=same_wall,
                seconds_literal_minus_hypothetical_eastern=(literal - as_eastern).total_seconds())


def conservative_date(filing_date, raw_z):
    filed = date.fromisoformat(filing_date)
    require(filed.isoformat() == filing_date, 'Filing date syntax')
    return max(filed, utc_syntax(raw_z).date())


def next_session(bound, sessions):
    require(sessions == sorted(set(sessions)), 'Session dates must be sorted and unique')
    for session in sessions:
        day = date.fromisoformat(session)
        if day > bound:
            return session
    raise ValueError('No later verified session in calendar')


def rows(columns):
    require(all(isinstance(columns.get(k), list) for k in FIELDS), 'Missing SEC columns')
    size = len(columns['accessionNumber'])
    require(all(len(v) == size for v in columns.values() if isinstance(v, list)), 'Ragged SEC columns')
    return [dict((k, columns[k][i]) for k in FIELDS) for i in range(size)]


def test():
    checks = []
    def ok(name, condition):
        require(condition, 'Synthetic failure: ' + name)
        checks.append(name)
    def reject(name, fn):
        try:
            fn()
        except ValueError:
            checks.append(name)
        else:
            raise ValueError('Synthetic failed to reject: ' + name)
    ok('winter Eastern hypothesis needs five-hour conversion', compare_clock('2020-01-15T22:00:00Z', '20200115170000')['literal_z_matches_hypothetical_eastern_instant'])
    ok('summer Eastern hypothesis needs four-hour conversion', compare_clock('2020-07-15T21:00:00.000Z', '20200715170000')['literal_z_matches_hypothetical_eastern_instant'])
    ok('same wall under Z detected separately', compare_clock('2020-07-15T17:00:00Z', '20200715170000')['literal_z_repeats_header_wall_clock'])
    ok('double conversion not accepted', not compare_clock('2020-07-16T01:00:00Z', '20200715170000')['literal_z_matches_hypothetical_eastern_instant'])
    ok('fractional seconds mismatch retained', not compare_clock('2020-07-15T21:00:00.001Z', '20200715170000')['literal_z_matches_hypothetical_eastern_instant'])
    ok('UTC date rollover handled', conservative_date('2020-07-15', '2020-07-16T01:00:00Z').isoformat() == '2020-07-16')
    ok('filing date can be later', conservative_date('2020-07-16', '2020-07-15T23:00:00Z').isoformat() == '2020-07-16')
    ok('strictly later session skips same date and weekend', next_session(date(2020, 7, 17), ['2020-07-17', '2020-07-20']) == '2020-07-20')
    ok('holiday omitted by supplied calendar', next_session(date(2020, 7, 2), ['2020-07-02', '2020-07-06']) == '2020-07-06')
    reject('naive clock rejected', lambda: utc_syntax('2020-07-15T17:00:00'))
    reject('offset syntax not silently normalized', lambda: utc_syntax('2020-07-15T17:00:00-04:00'))
    reject('bad day rejected', lambda: utc_syntax('2020-02-30T17:00:00Z'))
    reject('DST gap rejected', lambda: local_candidate('20200308023000'))
    reject('DST fold rejected', lambda: local_candidate('20201101013000'))
    reject('short header rejected', lambda: local_candidate('20200101'))
    reject('unsorted calendar rejected', lambda: next_session(date(2020, 1, 1), ['2020-01-03', '2020-01-02']))
    reject('duplicate calendar rejected', lambda: next_session(date(2020, 1, 1), ['2020-01-02', '2020-01-02']))
    reject('calendar exhaustion rejected', lambda: next_session(date(2020, 1, 2), ['2020-01-02']))
    fixture = b'ACCESSION NUMBER: 0000000001-20-000001\n<ACCEPTANCE-DATETIME>20200715170000\n'
    ok('exact header identity', parse_header(fixture, '0000000001-20-000001') == '20200715170000')
    reject('wrong accession rejected', lambda: parse_header(fixture, '0000000001-20-000002'))
    reject('duplicate header rejected', lambda: parse_header(fixture + fixture, '0000000001-20-000001'))
    good = {k: ['x'] for k in FIELDS}
    ok('aligned columns preserve values', rows(good)[0] == {k: 'x' for k in FIELDS})
    reject('ragged source rejected', lambda: rows(dict(good, form=[])))
    return checks


def audit_pages(inputs, feasibility):
    results, samples = [], []
    for symbol, cik in PILOT:
        prior = next(x for x in feasibility['issuers'] if x['symbol'] == symbol)
        main_path = BASE + cik + '_submissions.json'
        main = inputs.obj(main_path)
        require(str(main['cik']).zfill(10) == cik and symbol in main['tickers'], 'Issuer identity')
        pages = [(main_path, rows(main['filings']['recent']), None)]
        for item in main['filings']['files']:
            if item['filingFrom'] <= '2026-10-06' and item['filingTo'] >= '2015-01-01':
                require(re.fullmatch('CIK' + cik + r'-submissions-\d+\.json', item['name']), 'Page name')
                path = BASE + 'filing_indexes/' + item['name']
                page_rows = rows(inputs.obj(path))
                require(len(page_rows) == item['filingCount'], 'Advertised page count')
                pages.append((path, page_rows, item))
        all_rows, boundary_exceptions = {}, []
        for path, page_rows, item in pages:
            provenance = inputs.obj(str(Path(path).with_suffix('.provenance.json')))
            expected_url = 'https://data.sec.gov/submissions/' + ('CIK' + cik + '.json' if item is None else item['name'])
            require(provenance['source'] == expected_url, 'Source provenance URL')
            inputs.read(path, provenance['sha256'])
            for row in page_rows:
                key = row['accessionNumber']
                require(key not in all_rows or all_rows[key] == row, 'Conflicting accession')
                all_rows[key] = row
                if item and not item['filingFrom'] <= row['filingDate'] <= item['filingTo']:
                    boundary_exceptions.append(dict(accession=key, form=row['form'], filingDate=row['filingDate']))
        selected = sorted([x for x in all_rows.values() if x['form'] in ('10-Q', '10-K') and '2015-01-01' <= x['filingDate'] <= '2026-10-06'], key=lambda x: x['accessionNumber'])
        expected = sorted([{k: x[k] for k in FIELDS} for x in prior['filing_events']], key=lambda x: x['accessionNumber'])
        require(selected == expected, 'Feasibility differs from independent source enumeration')
        bounds, hypothesis_violations = [], []
        for row in selected:
            literal = utc_syntax(row['acceptanceDateTime'])
            bound = conservative_date(row['filingDate'], row['acceptanceDateTime'])
            # This is only a conditional check for two timezone interpretations.
            # It does not prove the source clock itself is accurate or publication occurred.
            for hypothesis, local_day in [('literal_UTC', literal.astimezone(NY).date()), ('wall_clock_under_Z', literal.date())]:
                if local_day > bound:
                    hypothesis_violations.append(dict(accession=row['accessionNumber'], hypothesis=hypothesis))
            bounds.append(dict(accession=row['accessionNumber'], filing_date=row['filingDate'], raw_z_date=literal.date().isoformat(), proposed_bound_date=bound.isoformat()))
        for year in (2015, 2020, 2026):
            for season, offset in (('winter_EST', -18000), ('summer_EDT', -14400)):
                choices = [x for x in selected if x['filingDate'][:4] == str(year) and datetime.fromisoformat(x['filingDate'] + 'T12:00:00').replace(tzinfo=NY).utcoffset().total_seconds() == offset]
                require(choices, 'Fixed sample stratum missing')
                chosen = min(choices, key=lambda x: (x['filingDate'], x['accessionNumber']))
                acc = chosen['accessionNumber']
                url = 'https://www.sec.gov/Archives/edgar/data/' + str(int(cik)) + '/' + acc.replace('-', '') + '/'
                samples.append(dict(symbol=symbol, year=year, season=season, **chosen,
                                    index_url=url + acc + '-index.html', complete_submission_url=url + acc + '.txt',
                                    purpose='Predetermined diagnostic sample only; cannot certify unsampled clocks. No request made.'))
        results.append(dict(symbol=symbol, independently_reconstructed_original_events=len(selected), pages=len(pages),
            feasibility_events_match_exactly=True, descriptor_boundary_exceptions=boundary_exceptions,
            syntax_Z_fields=len(bounds), raw_z_day_minus_filing_day_counts=dict(Counter(str((date.fromisoformat(x['raw_z_date']) - date.fromisoformat(x['filing_date'])).days) for x in bounds)),
            conditional_date_bound_violations=hypothesis_violations, proposed_date_bounds=bounds))
    return results, samples


def search_headers():
    """Bounded inventory: only source-family directories and first 64 KiB/file."""
    roots = [p for p in (ROOT / 'outputs').iterdir() if p.is_dir() and re.match(r'monthly_(?:sec_|aapl_|apple_|abbv_|msft_)', p.name)]
    roots += [ROOT / p for p in ('outputs/dynamic_top100/registry_sec', 'outputs/dynamic_top100/pilot_identity/original_reports',
              'outputs/dynamic_top100/pilot_identity/filing_indexes', 'outputs/index_union/sources/sec_identity', 'outputs/index_union/sources/sec_qqq', 'outputs/monthly_earnings_repair')]
    extensions = {'.txt', '.html', '.htm', '.sgml', '.hdr', '.bin', '.decoded'}
    inventory, hits = [], []
    for root in sorted(set(roots)):
        if not root.is_dir():
            continue
        for path in sorted(root.rglob('*')):
            if not path.is_file() or path.suffix.lower() not in extensions:
                continue
            require(path.resolve().is_relative_to(ROOT), 'Source symlink escaped root')
            with path.open('rb') as source:
                prefix = source.read(PREFIX_BYTES)
            relative = str(path.relative_to(ROOT))
            inventory.append(dict(path=relative, size=path.stat().st_size, prefix_sha256=digest(prefix)))
            if b'ACCEPTANCE-DATETIME' in prefix or re.search(rb'Accepted\s*(?:</[^>]+>\s*<[^>]+>\s*)+\d{4}-\d\d-\d\d', prefix, re.I):
                hits.append(dict(path=relative, prefix_sha256=digest(prefix), marker_only_not_admitted=True))
    return dict(roots=[str(p.relative_to(ROOT)) for p in sorted(set(roots))], extensions=sorted(extensions),
                prefix_bytes=PREFIX_BYTES, files_scanned=len(inventory), inventory_sha256=digest(canonical(inventory)),
                marker_hits=hits, complete_original_header_provenance_verified=0,
                limit='Bounded local search, not proof no header exists anywhere. Prefix markers alone do not authenticate source or timezone.')


def review_date_proposal(inputs, feasibility):
    """Compare the new metadata diagnostic with our own date/session oracle."""
    path = 'work/after156_filing_date_anchor_diagnostic.json'
    report = inputs.obj(path)
    for source, sha in report['input_hashes'].items():
        inputs.read(source, sha)
    inputs.read('work/test_after156_filing_date_anchor.py')
    sessions = inputs.obj('outputs/monthly_paired_target143/CALENDAR.json')['sessions']
    expected = {(issuer['symbol'], row['accessionNumber']): row
                for issuer in feasibility['issuers'] for row in issuer['filing_events']}
    require(len(report['rows']) == len(expected), 'Date diagnostic row count')
    seen, extra = set(), {}
    for row in report['rows']:
        key = row['symbol'], row['accession']
        require(key in expected and key not in seen, 'Date diagnostic duplicate or extra row')
        seen.add(key)
        source = expected[key]
        require(row['raw_acceptance'] == source['acceptanceDateTime'] and
                row['raw_filing_date'] == source['filingDate'], 'Date diagnostic raw fields changed')
        bound = conservative_date(source['filingDate'], source['acceptanceDateTime'])
        require(row['date_boundary'] == bound.isoformat(), 'Date boundary differs from independent oracle')
        chosen = next_session(bound, sessions)
        require(row['proposed_anchor'] == chosen, 'Date-only session differs from independent oracle')
        literal = utc_syntax(source['acceptanceDateTime'])
        for name, point in [('encoded_offset_taken_literally', literal),
                            ('raw_wall_time_interpreted_as_new_york', literal.replace(tzinfo=NY))]:
            local = point.astimezone(NY)
            day = local.date().isoformat()
            # The diagnostic explicitly uses ordinary 09:30 NY opens; this is
            # not verification of a historical exchange open-time schedule.
            before = (local.hour, local.minute, local.second, local.microsecond) < (9, 30, 0, 0)
            first = day if day in sessions and before else next_session(local.date(), sessions)
            require(row['hypothetical_first_opens'][name] == first, 'Hypothetical first open differs')
            lag = sessions.index(chosen) - sessions.index(first)
            require(lag >= 0 and lag == row['extra_sessions_vs_hypothetical'][name], 'Hypothetical lag differs')
            if name == 'encoded_offset_taken_literally':
                extra.setdefault(row['symbol'], Counter())[str(lag)] += 1
        require(row['exact_acceptance_instant_verified'] is False, 'Date diagnostic claimed exact time')
    require(seen == set(expected), 'Date diagnostic omissions')
    return dict(path=path, independent_rows_compared=len(seen), passed=True,
                actual_calendar_dates_used=True, producer_imported=False,
                extra_sessions_vs_literal_UTC={k: dict(v) for k, v in sorted(extra.items())},
                limit='Verifies the proposed mapping and two stated interpretations only. It does not establish source clock accuracy, dissemination timing or authoritative exchange intraday opens.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--test-only', action='store_true')
    args = parser.parse_args()
    tests = test()
    if args.test_only:
        print(json.dumps(dict(synthetic_checks=len(tests), passed=True, checks=tests), indent=2))
        return
    inputs = Inputs()
    feasibility = inputs.obj(FEAS, FEAS_SHA)
    for path, sha in feasibility['input_hashes'].items():
        inputs.read(path, sha)
    inputs.read('work/after156_filing_clock_independent.py')
    stop = inputs.obj(STOP)
    require(any(x.get('SEC_provider_stopped_for_UTC_date') == '2026-10-20' and x.get('http_status') == 403 for x in stop['receipts']), 'Expected SEC stop evidence')
    issuers, samples = audit_pages(inputs, feasibility)
    date_review = review_date_proposal(inputs, feasibility)
    scan = search_headers()
    inputs.finish()
    require(search_headers() == scan, 'Local header search changed during review')
    result = dict(schema_version=1, stage='INDEPENDENT_FILING_CLOCK_SOURCE_REVIEW_AFTER156',
        created_utc=datetime.now(timezone.utc).isoformat(), status='EXACT_CLOCK_UNRESOLVED_DATE_ONLY_ALTERNATIVE_CONDITIONAL',
        passed=True, passed_scope='Independent enumeration/hash verification and synthetic validator only; no event-clock admission.',
        source_scope=dict(issuers=[x[0] for x in PILOT], begin='2015-01-01', end='2026-10-06', forms=['10-Q', '10-K']),
        issuers=issuers, local_header_search=scan, diagnostic_sample_rule='First filing in each issuer x {2015,2020,2026} x {winter_EST,summer_EDT}, based only on filing date noon NY offset.',
        independent_date_proposal_review=date_review,
        future_collection_sample=samples,
        timezone_disposition=dict(raw_field='All 141 use parseable Z syntax. Z denotes UTC syntactically; it does not authenticate upstream conversion correctness.',
            original_header_timezone='Not independently authenticated from local originals or authoritative field-level specification.',
            accepted_is_publication='Acceptance and public dissemination must be distinguished. No exact dissemination claim.',
            earliest_earnings_news=False, exact_instant_verified=False,
            minimum_next_evidence='SEC authoritative field/timezone specification and original submission or filing index for fixed 18-sample diagnostic. Without a documented uniform conversion rule, each of 141 clocks needs original corroboration; successful samples alone cannot waive remaining clocks.'),
        date_only_alternative=dict(proposed=True, adopted=False,
            definition='D=max(filingDate, literal calendar date of parseable acceptanceDateTime Z field); anchor=first verified NYSE session strictly after D. Post-anchor offsets are a distinct proposed filing-date event definition.',
            conditional_result='No row violates the date bound under either accurate literal UTC or Eastern wall-clock mislabeled Z interpretation. This result is conditional and partly structural, not external corroboration.',
            causal_bound_established=False,
            missing_evidence='Need evidence that public availability was no later than the end of D for each admitted filing or a documented source-wide guarantee. Unknown clock corruption, corrected/backdated filingDate, and delayed dissemination are not excluded by these JSON snapshots.',
            distinction='Could be separately preregistered as an explicit conservative source assumption if authorized; cannot be labeled exact-clock certification or silently replace the pending source gate.',
            calendar_status='Root metadata-only proposed anchors independently reproduced for 141 records against its hash-bound session dates. No price partition, factor or return computed; no intraday exchange schedule certified.'),
        public_search_observation=dict(date_utc='2026-10-20', search_calls_before_stop_notice=1, opened_pages=0, archive_requests=0,
            observation='Search excerpts mention Eastern EDGAR operating/filing deadlines, but do not establish submissions acceptanceDateTime conversion or immediate dissemination. Not used for admission.',
            urls=['https://www.sec.gov/file/efmvol3-onlineforms810-r2', 'https://www.sec.gov/files/edgar/filermanual/edgarfm-vol2-v78_30_day_draft.pdf', 'https://www.sec.gov/submit-filings/filer-support-resources/how-do-i-guides/request-filing-date-adjustment'],
            stop='No further SEC request through any route after the parent identified the existing 2026-10-20 provider stop.'),
        synthetic_validation=dict(passed=True, checks=len(tests), names=tests), input_hashes=dict(sorted(inputs.hashes.items())),
        admission=dict(exact_filing_clock=False, date_only_causal_bound=False, complete_SEC_history=False, factor=False, strategy_registered=False),
        activity_counts=dict(offline_network_requests=0, original_archive_requests=0, price_values_read=0, factor_calculations=0, ranks=0, returns=0, new_trials=0, frozen_files_modified=0))
    if args.write:
        target = ROOT / OUT
        require(not target.exists(), 'Write-once review exists')
        target.write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
    print(json.dumps(dict(status=result['status'], source_events=sum(x['independently_reconstructed_original_events'] for x in issuers),
        source_pages=sum(x['pages'] for x in issuers), local_scan=scan, sample_count=len(samples), synthetic_checks=len(tests),
        conditional_date_bound_violations=sum(len(x['conditional_date_bound_violations']) for x in issuers),
        review_sha256=digest((ROOT / OUT).read_bytes()) if args.write else None), indent=2))


if __name__ == '__main__':
    main()
