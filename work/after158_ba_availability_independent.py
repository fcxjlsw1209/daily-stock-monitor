"""Read-only independent audit of the frozen BA source-feasibility map.

Run through run_local.sh after PLAN.json and FEASIBILITY.json exist. This module
never imports the implementation under review and never reads outcome files.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from exchange_calendars.exchange_calendar_xnys import XNYSExchangeCalendar

ROOT = Path(__file__).resolve().parents[1]
CAL = ROOT / 'outputs/monthly_paired_target143/CALENDAR.json'
BRIDGE = ROOT / 'work/after158_ba_currency_bridge.json'
REVIEW = ROOT / 'work/after158_ba_independent_review.json'
FACTS = ROOT / 'outputs/dynamic_top100/registry_sec/0000012927_facts.json'
META = ROOT / 'outputs/monthly_local_linear70/META.json'
ROWS = ROOT / 'outputs/monthly_revenue_pilot_source156/ROWS.json'
PUBLICATION = ROOT / 'outputs/after158_ba_publication_v1'
EXTERNAL = Path('/Users/shiweili/.cache/codex-runtimes/codex-primary-runtime/dependencies')
ACCESSIONS = {
    2014: '0000012927-15-000011', 2015: '0000012927-16-000099',
    2016: '0000012927-17-000006', 2017: '0000012927-18-000007',
    2018: '0000012927-19-000010',
}


def require(condition, label):
    if not condition:
        raise ValueError(label)


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pin_check(mapping, external=False):
    for name, expected in mapping.items():
        path = Path(name)
        if external:
            require(path.is_absolute() and path.resolve().is_relative_to(EXTERNAL.resolve()),
                    'Unexpected external review evidence')
        else:
            require(not path.is_absolute() and '..' not in path.parts, 'Nonrelative local pin')
            path = ROOT / path
            require(path.resolve().is_relative_to(ROOT), 'Local pin escape')
        require(digest(path) == expected, 'Pin mismatch: ' + name)


def local_name(name):
    path = Path(name)
    return str(path.relative_to(ROOT)) if path.is_absolute() else name


def pointer(document, text):
    value = document
    for segment in text.split('/')[1:]:
        segment = segment.replace('~1', '/').replace('~0', '~')
        value = value[int(segment)] if isinstance(value, list) else value[segment]
    return value


def expected_selection(events, cutoff):
    known = sorted((e for e in events if datetime.fromisoformat(e['known_at']) <= cutoff),
                   key=lambda e: e['year'], reverse=True)
    if not known:
        return dict(latest_year=None, chain_years=[], feasible=False, reason='no_known_original')
    result = dict(latest_year=known[0]['year'], chain_years=[], feasible=False)
    if not known[0]['basis_valid']:
        return dict(result, reason='latest_original_basis_break_no_fallback')
    wanted = list(range(known[0]['year'] - 3, known[0]['year'] + 1))
    by_year = {event['year']: event for event in known}
    if any(year not in by_year for year in wanted):
        return dict(result, reason='four_originals_not_yet_known')
    result['chain_years'] = wanted
    if not all(by_year[year]['basis_valid'] for year in wanted):
        return dict(result, reason='chain_basis_break')
    if not all(datetime.fromisoformat(by_year[year]['usable_at']) <= cutoff for year in wanted):
        return dict(result, reason='unit_corroboration_not_yet_usable')
    return dict(result, feasible=True, reason=None)


def build_calendar():
    """Exercise the pinned library's actual schedule interface, not a mock."""
    calendar = read(CAL)
    calendar_pins = calendar['provenance']['source_hashes']
    require(len(calendar_pins) == 9, 'Calendar source closure size')
    pin_check({local_name(name): expected for name, expected in calendar_pins.items()})
    provenance = calendar['provenance']['provenance']
    manifest_path = Path(provenance['manifest'])
    require(digest(manifest_path) == provenance['manifest_sha256'], 'Calendar manifest binding')
    manifest = read(manifest_path)
    closure_dates = {row['date'] for row in manifest['closures'] if 'XNYS' in row['markets']}
    for row in manifest['closures']:
        source = ROOT / 'outputs/index_union' / row['source_file']
        require(digest(source) == row['source_sha256'], 'Calendar closure source binding')
    raw_schedule = XNYSExchangeCalendar(start=calendar['start'], end=calendar['end']).schedule
    require('close' in raw_schedule.columns, 'Pinned schedule has no close column')
    schedule = raw_schedule.loc[[str(day.date()) not in closure_dates for day in raw_schedule.index]]
    sessions = [str(day.date()) for day in schedule.index]
    require(sessions == calendar['sessions'], 'Independent calendar reconstruction')
    closes = {str(day.date()): close.to_pydatetime() for day, close in schedule['close'].items()}
    require(all(close.tzinfo is not None and close.utcoffset() is not None for close in closes.values()),
            'Calendar close missing timezone')
    return calendar_pins, sessions, closes, list(schedule.columns)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calendar-preflight', action='store_true')
    parser.add_argument('--output-dir', choices=['after158_ba_availability_v1', 'after158_ba_availability_v2'],
                        default='after158_ba_availability_v2')
    parser.add_argument('--plan-sha')
    parser.add_argument('--feasibility-sha')
    args = parser.parse_args()
    if args.calendar_preflight:
        calendar_pins, sessions, closes, columns = build_calendar()
        print(json.dumps(dict(passed=True, calendar_source_pins_verified=len(calendar_pins),
                              independently_matched_sessions=len(sessions), schedule_columns=columns,
                              all_closes_timezone_aware=True, calendar_sha256=digest(CAL),
                              checker_sha256=digest(Path(__file__)), source_map_computed=False), indent=2))
        return
    require(args.plan_sha and args.feasibility_sha, 'Audit requires both artifact hashes')
    base = ROOT / 'outputs' / args.output_dir
    require(digest(base / 'PLAN.json') == args.plan_sha, 'Wrong PLAN bytes')
    require(digest(base / 'FEASIBILITY.json') == args.feasibility_sha, 'Wrong FEASIBILITY bytes')
    plan = read(base / 'PLAN.json')
    result = read(base / 'FEASIBILITY.json')
    pins = plan['input_hashes']
    pin_check(pins)
    pin_check(plan['source_review_external_hashes'], external=True)
    require(result['plan_sha256'] == args.plan_sha, 'Result PLAN binding')
    require(result['contract'] == plan['contract'], 'Result contract binding')
    require(plan['accessions'] == {str(year): accn for year, accn in ACCESSIONS.items()},
            'Original accession contract')
    require(plan['strategy_trials'] == 0 and result['new_strategy_trials'] == 0, 'Unexpected strategy trial')
    require(result['source_admission_changed'] is False and result['strategy_admitted'] is False
            and result['financial_factors_computed'] is False and result['observer_changes'] is False,
            'Source-only scope violation')

    calendar_pins, sessions, closes, columns = build_calendar()
    for name, expected in calendar_pins.items():
        require(pins.get(local_name(name)) == expected, 'Calendar dependency absent from PLAN: ' + name)

    def second_close(day):
        following = [session for session in sessions if session > day]
        require(len(following) >= 2, 'Insufficient calendar after source date')
        return closes[following[1]].isoformat()

    review, bridge, facts = read(REVIEW), read(BRIDGE), read(FACTS)
    publication_review = read(PUBLICATION / 'REVIEW.json')
    require(review['passed'] is True and review['original_basis_three_edges_supported'] is True,
            'Original-basis review unavailable')
    require(bridge['passed'] is True and bridge['five_original_cells_match_revenues_usd'] is True,
            'Original currency review unavailable')
    require(publication_review['passed'] is True, 'Publication review unavailable')
    releases = {row['fiscal_year']: row['original_publication_dateline'] for row in review['reports']}
    require(sorted(releases) == sorted(ACCESSIONS), 'Original report coverage')
    events = []
    for year in sorted(ACCESSIONS):
        reports = [row for row in bridge['reports'] if row['fiscal_year'] == year]
        require(len(reports) == 1, 'Duplicate or missing bridge year')
        report = reports[0]
        release = releases[year]
        require(report['original_publication_dateline'] == release, 'Bridge dateline mismatch')
        candidates = [row for row in report['concept_results']['Revenues']['exact_annual_contexts']
                      if row['source_row']['accn'] == ACCESSIONS[year]
                      and row['comparison']['currency_corroboration_matches']]
        require(len(candidates) == 1, 'Exact original currency context missing or ambiguous')
        chosen = candidates[0]
        fact = pointer(facts, chosen['json_pointer'])
        require(fact == chosen['source_row'], 'Bridge raw JSON pointer mismatch')
        require(chosen['json_pointer'].startswith('/facts/us-gaap/Revenues/units/USD/')
                and chosen['unit'] == 'USD', 'Wrong original currency/concept')
        require((fact['accn'], fact['form'], fact['fy'], fact['fp'], fact['start'], fact['end']) ==
                (ACCESSIONS[year], '10-K', year, 'FY', f'{year}-01-01', f'{year}-12-31'),
                'Wrong original annual context')
        parsed = read(PUBLICATION / f'FY{year}.parsed.json')
        require(parsed['expected_dateline'] == release, 'Publication parsed dateline mismatch')
        month = datetime.strptime(release, '%Y-%m-%d')
        normalized = ' '.join(parsed['visible_text'].split())
        require(any(f'{label} {month.day}, {month.year}' in normalized
                    for label in ('Jan.', 'Jan', 'January')), 'Dateline absent from source text')
        events.append(dict(year=year, original_release_date=release, original_usd_filing_date=fact['filed'],
                           original_usd_accession=ACCESSIONS[year], original_usd_json_pointer=chosen['json_pointer'],
                           known_at=second_close(release), usable_at=second_close(max(release, fact['filed'])),
                           basis_valid=year != 2018))
    require(result['events'] == events, 'Independent source event map mismatch')

    metadata, rows = read(META), read(ROWS)
    require(len(rows) == 5978 and len(metadata['dates']) == 122, 'Original axis dimensions')
    expected_rows = []
    for ordinal, row in enumerate(rows):
        signal, candidate = divmod(ordinal, 49)
        require((row['signal_index'], row['candidate_index'], row['symbol'], row['decision_date']) ==
                (signal, candidate, metadata['symbols'][signal][candidate], metadata['dates'][signal]),
                'Original axis row mismatch')
        require(datetime.fromisoformat(row['decision_cutoff']) == closes[row['decision_date']],
                'Original decision close mismatch')
        if signal >= 53 or row['symbol'] != 'BA':
            continue
        position = sessions.index(row['decision_date'])
        require(position >= 42, 'Insufficient t-42 axis')
        cutoff = closes[sessions[position - 42]]
        expected_rows.append(dict(signal_index=signal, candidate_index=candidate,
                                  decision_date=row['decision_date'], information_cutoff=cutoff.isoformat(),
                                  **expected_selection(events, cutoff)))
    require(len(expected_rows) == 21 and result['rows'] == expected_rows, 'Independent 21-row map mismatch')
    require(result['full_original_axis_verified'] == 5978, 'Result axis scope mismatch')
    summary = dict(total_early_ba_rows=21, feasible_rows=sum(row['feasible'] for row in expected_rows),
                   reasons={reason: sum(row['reason'] == reason for row in expected_rows)
                            for reason in sorted({row['reason'] for row in expected_rows if row['reason']})})
    require(result['summary'] == summary, 'Result summary mismatch')
    pin_check(pins)
    pin_check(plan['source_review_external_hashes'], external=True)
    require(digest(base / 'PLAN.json') == args.plan_sha
            and digest(base / 'FEASIBILITY.json') == args.feasibility_sha, 'Audit input changed')
    print(json.dumps(dict(passed=True, checked_utc=datetime.now(timezone.utc).isoformat(),
                          checker_sha256=digest(Path(__file__)), plan_sha256=args.plan_sha,
                          feasibility_sha256=args.feasibility_sha,
                          local_pins_verified=len(pins), external_review_pins_verified=len(plan['source_review_external_hashes']),
                          calendar_source_pins_verified=9, original_context_pointers_verified=5,
                          full_original_axis_verified=5978, independently_matched_rows=21,
                          independently_matched_events=5, summary=summary,
                          strategy_admitted=False, financial_factors_computed=False,
                          source_files_modified=False, implementation_under_review_imported=False,
                          limitations=['Date-only historical availability assumptions remain unverified.',
                                       'No first-public timestamp, contemporaneous bytes or strategy admission is certified.']),
                     indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
