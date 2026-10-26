"""Audit actual candidate dates, not an arbitrary all-issuer archive interval.

This is a data gate for regular-release post-known drift. It is not evidence
of complete preliminary releases, exact announcement clocks or EPS surprises.
No model fitting, price downloads, candidate exclusion or ledger writes.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pandas as pd
from research_calendar import market_calendar

P = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quarter(event):
    if event.get('fiscal_year') is None or event.get('fiscal_quarter') is None:
        return None
    return int(event['fiscal_year']) * 4 + int(event['fiscal_quarter']) - 1


def candidate_issues(scope, rows, events, hints, calendar):
    groups = defaultdict(list)
    vendor = defaultdict(list)
    for event in events:
        if event['status'] == 'ready':
            groups[event['symbol']].append(event)
    for group in groups.values():
        group.sort(key=lambda e: e['known_by_at'])
    for hint in hints:
        vendor[hint['symbol']].append(hint)
    issues = []
    pairs = [(r['symbol'], r['signal_date']) for r in rows]
    if len(pairs) != len(set(pairs)):
        raise ValueError('Duplicate candidate pair')
    for row in rows:
        if row['status'] == 'not_applicable_etf':
            continue
        day, symbol = row['signal_date'], row['symbol']
        close = calendar.session_close(pd.Timestamp(day))
        sequence = groups[symbol]
        known = [e for e in sequence if pd.Timestamp(e['known_by_at']) <= close]
        latest = known[-1] if known else None
        if row['event_id'] != (latest['event_id'] if latest else None):
            raise ValueError('Availability did not select latest actually known event')
        base = dict(scope=scope, signal_date=day, symbol=symbol,
                    prior_event_id=latest['event_id'] if latest else None,
                    prior_release=latest['publication_date'] if latest else None)

        def add(kind, **details):
            issues.append(dict(base, issue_id='|'.join((kind, scope, day, symbol)), kind=kind, **details))

        if latest is None:
            add('missing_prior_event')
            continue
        age = (close - pd.Timestamp(latest['known_by_at'])).days
        if age > 120:
            add('long_reporting_interval', age_days=age)
        index = sequence.index(latest)
        following = sequence[index + 1] if index + 1 < len(sequence) else None
        if following and quarter(latest) is not None and quarter(following) is not None:
            if quarter(following) - quarter(latest) > 1:
                add('missing_intervening_quarter', prior_quarter=quarter(latest),
                    following_quarter=quarter(following), following_release=following['publication_date'])
        if following is None:
            age_end = (pd.Timestamp(day) - pd.Timestamp(latest['official_actual_period_end'])).days
            if age_end > 100:
                add('archive_endpoint_review', days_since_fiscal_end=age_end)
        suspect = []
        for hint in vendor[symbol]:
            date = hint['provider_reported_date']
            if not (latest['publication_date'] < date < day):
                continue
            if any(abs((pd.Timestamp(e['publication_date']) - pd.Timestamp(date)).days) <= 3 for e in sequence):
                continue
            suspect.append(date)
        if suspect:
            add('unmatched_vendor_hint', provider_dates=sorted(set(suspect)))
    return issues


def verify_resolution(resolution):
    if not resolution.get('reason') or not resolution.get('evidence'):
        raise ValueError('Resolution needs reason and source evidence')
    for item in resolution['evidence']:
        path = Path(item['path'])
        if not path.is_file() or sha(path) != item['sha256']:
            raise ValueError('Resolution evidence checksum mismatch: ' + str(path))


def verify_universe(rows, meta, etfs):
    expected = {(day, symbol) for day, symbols in zip(meta['dates'], meta['symbols']) for symbol in symbols}
    actual = [(row['signal_date'], row['symbol']) for row in rows]
    if len(meta['dates']) != len(meta['symbols']) or len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('Candidate scope differs from frozen research META')
    for row in rows:
        is_etf = row['symbol'] in etfs
        if (row['status'] == 'not_applicable_etf') != is_etf or (is_etf and row['event_id'] is not None):
            raise ValueError('Candidate ETF identity differs from frozen research META')


def resolve(issues, resolutions, boundaries):
    for boundary in boundaries:
        verify_resolution(boundary)
    by_id = {}
    for resolution in resolutions:
        verify_resolution(resolution)
        if resolution['issue_id'] in by_id:
            raise ValueError('Duplicate explicit scope resolution')
        by_id[resolution['issue_id']] = resolution
    rows = []
    for issue in issues:
        row = dict(issue, resolution=None)
        if issue['kind'] == 'missing_intervening_quarter':
            for boundary in boundaries:
                prior = boundary['previous_fiscal_year'] * 4 + boundary['previous_fiscal_quarter'] - 1
                following = boundary['next_fiscal_year'] * 4 + boundary['next_fiscal_quarter'] - 1
                if (boundary['symbol'] == issue['symbol'] and prior == issue['prior_quarter']
                        and following == prior + 1 and issue['signal_date'] < boundary['release_date']):
                    row['resolution'] = dict(kind='next_regular_release_not_yet_published',
                        source_url=boundary['source_url'], next_release=boundary['release_date'],
                        source_representation=boundary['source_representation'],
                        evidence=boundary['evidence'],
                        note='Boundary fact used only for audit, never for earlier feature values.')
                    break
        if issue['issue_id'] in by_id:
            explicit = by_id[issue['issue_id']]
            if 'expected_prior_release' in explicit and explicit['expected_prior_release'] != issue['prior_release']:
                raise ValueError('Resolution refers to a different prior release')
            if 'expected_provider_dates' in explicit and explicit['expected_provider_dates'] != issue.get('provider_dates'):
                raise ValueError('Resolution refers to a different vendor hint')
            if explicit.get('next_release') and not issue['signal_date'] < explicit['next_release']:
                raise ValueError('Resolution crosses the next release boundary')
            row['resolution'] = explicit
        rows.append(row)
    return rows


def main():
    inputs = {name: sha(P / name) for name in ('audit_scope.py', 'audit_panel.py', 'event_schema.py', 'research_calendar.py')}

    def read(name):
        file = P / name
        inputs[name] = sha(file)
        return json.loads(file.read_text())

    events = read('EVENTS.json')
    meta_file = P.parent / 'monthly_architecture_restart/META.json'
    inputs[str(meta_file)] = sha(meta_file)
    frozen_meta = json.loads(meta_file.read_text())
    hints = read('VENDOR_HINTS.json')['hints']
    boundary = read('BOUNDARY_OBSERVATIONS.json')
    resolution_file = P / 'SCOPE_RESOLUTIONS.json'
    resolutions = read('SCOPE_RESOLUTIONS.json') if resolution_file.exists() else []
    cal = market_calendar(start='2010-01-01', end='2030-12-31')
    issues, counts, invalid_prices = [], {}, []
    for scope in ('monthly', 'training'):
        availability = read(scope.upper() + '_AVAILABILITY.json')
        verify_universe(availability, frozen_meta[scope], set(frozen_meta['monthly']['etfs']))
        panel = read(scope.upper() + '_FEATURES.json')
        audit = read(scope.upper() + '_PANEL_AUDIT.json')
        if not {'audit_panel.py', 'event_schema.py', 'research_calendar.py'} <= set(audit['input_sha256']):
            raise ValueError('Legacy panel audit lacks code-version provenance')
        for name, expected in audit['input_sha256'].items():
            if sha(P / name) != expected:
                raise ValueError('Stale panel audit: ' + name)
        issues += candidate_issues(scope, availability, events, hints, cal)
        counts[scope] = dict(candidate_rows=len(availability), states=dict(Counter(r['status'] for r in availability)))
        for row in panel['rows']:
            if row['status'] == 'missing_cached_price_file':
                invalid_prices.append(dict(scope=scope, signal_date=row['signal_date'], symbol=row['symbol'], status=row['status']))
            for horizon, reaction in row['reactions'].items():
                if reaction['status'] not in ('ready', 'waiting_for_closed_window'):
                    invalid_prices.append(dict(scope=scope, signal_date=row['signal_date'], symbol=row['symbol'], horizon=horizon, status=reaction['status']))
                elif reaction['status'] == 'ready' and (reaction['benchmark_status'] != 'ready' or reaction['volume_status'] != 'ready'):
                    invalid_prices.append(dict(scope=scope, signal_date=row['signal_date'], symbol=row['symbol'], horizon=horizon,
                        status='incomplete_benchmark_or_volume', benchmark_status=reaction['benchmark_status'], volume_status=reaction['volume_status']))
    reviewed = resolve(issues, resolutions, boundary)
    unresolved = [r for r in reviewed if r['resolution'] is None]
    status = read('STATUS.json')
    if status['protected_files_changed']:
        raise ValueError('Protected account/research files changed')
    result = dict(generated_at=datetime.now(timezone.utc).isoformat(),
        scope='Actual frozen monthly and five-year-training candidate dates, regular earnings release post-known-close drift only.',
        scoped_post_known_drift_data_ready=not unresolved and not invalid_prices,
        full_global_earnings_archive_complete=False,
        exact_announcement_gap_or_eps_surprise_ready=False,
        backtest_run=False, counts=counts, issues_reviewed=len(reviewed),
        unresolved_count=len(unresolved), unresolved=unresolved, reviewed=reviewed,
        invalid_candidate_price_windows=invalid_prices,
        inputs=inputs, protected_files_changed=0,
        limitations=['Date-only announcements use the next XNYS session close; first-public clock is not asserted.',
                    'Current official archives and indexed observations are not immutable contemporaneous snapshots.',
                    'ETF, pre-first-release and immature reaction values stay null, never zero or stock-pool exclusions.',
                    'Candidate-scoped quarter checks do not certify every preliminary announcement, amendment or inactive-period archive.'])
    (P / 'SCOPED_READINESS.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('scoped_post_known_drift_data_ready', 'issues_reviewed', 'unresolved_count')}, sort_keys=True))
    print('invalid_candidate_price_windows=' + str(len(invalid_prices)))
    print(json.dumps(dict(Counter((r['symbol'] + ':' + r['kind']) for r in unresolved))))


if __name__ == '__main__':
    main()
