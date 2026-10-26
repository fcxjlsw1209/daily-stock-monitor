"""One preregistered data-only sensitivity, never a certified market-cap repair."""
import argparse
import bisect
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/monthly_shell_pool_sensitivity168'
SRC = 'outputs/monthly_shell_pool_source168/'
FILES = [
    SRC + name for name in [
        'PLAN.json', 'EXTRACTION.json', 'INDEPENDENT_REVIEW.json',
        'EXTERNAL_AUDIT.json', 'FOLLOWUP_REVIEW.json', 'COMPLETION.json',
        'QUOTES_PLAN.json', 'QUOTES.json', 'QUOTES_REVIEW.json',
        'external_followup/ratio_model.body', 'external_followup/ratio_model.receipt.json',
    ]
] + [
    'outputs/monthly_2016_extension/EARLY_RANKS.json',
    'outputs/monthly_2016_extension/POOLS.json',
    'outputs/monthly_2016_extension/extend.py',
    'outputs/monthly_published44_reconstruction/prices/SHEL.csv',
    'outputs/monthly_sector_diffusion167/COMPLETION.json',
]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def now():
    return datetime.now(timezone.utc).isoformat()


def runtime():
    return {'python': platform.python_version(), **{
        name: importlib.metadata.version(name)
        for name in ['numpy', 'pandas', 'exchange-calendars']}}


def next_session(released, sessions):
    index = bisect.bisect_right(sessions, released)
    if index >= len(sessions):
        raise ValueError('Release beyond calendar: ' + released)
    return sessions[index]


def choose_report(records, asof, sessions):
    usable = []
    for record in records:
        released = record['release']['stated_date']
        available = next_session(released, sessions)
        if available <= asof and record['period_end'] <= released:
            usable.append((released, record['period_end'], record['period'], available, record))
    if not usable:
        raise ValueError('No disclosed prior report: ' + asof)
    selected = max(usable, key=lambda x: x[:3])
    return selected[4], selected[3]


def revalue(issued, quote):
    values = []
    for category in ['A', 'B']:
        count = issued[category]
        price = quote['class_' + category]['close']
        if type(count) is not int or count <= 0:
            raise ValueError('Invalid ordinary-share count')
        if not isinstance(price, (int, float)) or not math.isfinite(price) or price <= 0:
            raise ValueError('Invalid quote')
        values.append(count * price / 2.0)
    return math.fsum(values)


def rerank(old, proxy):
    if len(old) != 40 or len({r['symbol'] for r in old}) != 40:
        raise ValueError('Expected 40 unique archived issuers')
    if not math.isfinite(proxy) or proxy <= 0:
        raise ValueError('Invalid proxy')
    others = [{**r} for r in old if r['symbol'] != 'SHEL']
    rows = others + [{'symbol': 'SHEL', 'market_cap_usd': proxy}]
    if not all(math.isfinite(r['market_cap_usd']) and r['market_cap_usd'] > 0 for r in rows):
        raise ValueError('Invalid original cap')
    rows.sort(key=lambda r: (-r['market_cap_usd'], r['symbol']))
    return rows


def register():
    OUT.mkdir(exist_ok=True)
    paths = set(FILES)
    for record in read(ROOT / (SRC + 'PLAN.json'))['records']:
        paths.update(record['inputs'])
    for item in read(ROOT / (SRC + 'QUOTES_PLAN.json'))['source_files'].values():
        paths.add(item['path'])
    plan = {
        'schema': 'shell_pool_sensitivity168_plan_v1', 'registered_utc': now(),
        'purpose': 'Quantify difference between two declared Shell cap proxies, not error relative to true market cap.',
        'scope': '53 archived early signal/asof pairs only; no strategy return, model or training computation.',
        'calendar': {'name': 'XNYS', 'start': '2015-01-01', 'end': '2021-01-10'},
        'report_selection': 'Latest issuer-stated release date whose next strictly later XNYS session is <= archived asof; ties by latest period_end, then period lexicographically. Same-report A/B only.',
        'formula': '(last_published_A_issued * asof_A_ADR_Close + last_published_B_issued * asof_B_ADR_Close) / 2',
        'quote_source': 'Pinned issuer-embedded Euroland XLS transcription; exact asof only, no fill, no source switching.',
        'ranking': 'Remove SHEL from archived top40, insert new proxy, keep all other values exact, sort (-market_cap_usd, symbol), select first32. Preserve all later pool records. Candidate-set rank is not full-market rank.',
        'assumptions': [
            'Issuer-stated release dates approximate historical availability; original-vintage bytes not certified.',
            'Most recent published issued-share counts serve as a lagged proxy, not current counts. No unobserved issuance/repurchase/treasury/employee-trust adjustment.',
            'Two ordinary shares per matching class ADR throughout this period; interval continuity assumed.',
            'Euroland Close treated as non-dividend-adjusted USD on a consistent split basis; numerical corroboration is not a source-methodology certificate.',
            'Other issuers retain legacy retrospective estimates and listing assumptions; only conditional pool impact is measured.',
        ],
        'hard_failures': ['missing exact-date A/B quotes', 'missing published report', 'invalid units/numbers', 'hash drift', 'archived top32 disagrees with existing POOLS', 'positive evidence contradicts assumed ADR/quote basis'],
        'known_quote_issues': {'unresolved_A_difference': '2016-03-09', 'absent_dates': ['2016-05-03', '2016-11-25'], 'handling': 'Record intersection with target dates; never repair or switch source.'},
        'prohibitions': ['No actual-market-cap or certified-PIT or strict-upper-bound claim', 'No retrospective observer edits', 'No selection by strategy performance', 'No return calculations or strategy trials in this phase'],
        'next_gate': 'Independent data comparison audit before any separate registered strategy sensitivity; membership or training changes require full relevant feature reconstruction.',
        'inputs': {str(path): sha(ROOT / path) for path in sorted(paths)},
        'script_sha256': sha(Path(__file__)), 'runtime': runtime(),
        'strategy_trials_added': 0,
    }
    write_once(OUT / 'PLAN.json', plan)
    print(json.dumps({'plan_sha256': sha(OUT / 'PLAN.json'), 'input_count': len(paths), 'status': 'REGISTERED_NO_CAPS_COMPUTED'}))


def run():
    import exchange_calendars as xcals
    plan = read(OUT / 'PLAN.json')
    assert plan['script_sha256'] == sha(Path(__file__))
    assert plan['runtime'] == runtime()
    for path, expected in plan['inputs'].items():
        assert sha(ROOT / path) == expected, path
    calendar = xcals.get_calendar(**plan['calendar'])
    sessions = calendar.sessions.strftime('%Y-%m-%d').tolist()
    extraction = read(ROOT / (SRC + 'EXTRACTION.json'))
    quotes = read(ROOT / (SRC + 'QUOTES.json'))
    assert read(ROOT / (SRC + 'QUOTES_REVIEW.json'))['status'] == 'PASS_TRANSCRIPTION_AND_LOCAL_COMPARISONS_ONLY'
    assert quotes['currency'] == 'USD'
    by_date = {r['date']: r for r in quotes['rows']}
    assert len(by_date) == len(quotes['rows'])
    archived = read(ROOT / 'outputs/monthly_2016_extension/EARLY_RANKS.json')
    old_pools = read(ROOT / 'outputs/monthly_2016_extension/POOLS.json')
    assert len(archived) == 53
    pools = dict(old_pools)
    rows = []
    for event in archived:
        signal, asof, top40 = event['date'], event['asof'], event['top40']
        assert signal in sessions and asof in sessions and next_session(asof, sessions) == signal
        old = [r['symbol'] for r in top40[:32]]
        assert old == old_pools[signal]
        report, usable = choose_report(extraction['records'], asof, sessions)
        issued, quote = report['issued_ordinary_shares'], by_date[asof]
        assert issued['units'] == 'ordinary shares; not millions'
        proxy = revalue(issued, quote)
        ranked = rerank(top40, proxy)
        new = [r['symbol'] for r in ranked[:32]]
        prior = next((r for r in top40 if r['symbol'] == 'SHEL'), None)
        unchanged = [r for r in top40 if r['symbol'] != 'SHEL']
        boundary = sorted(unchanged, key=lambda r: (-r['market_cap_usd'], r['symbol']))[31]
        row = {
            'signal_date': signal, 'asof': asof, 'selected_period': report['period'],
            'count_effective_date': report['period_end'], 'release_date': report['release']['stated_date'], 'usable_session': usable,
            'count_age_calendar_days': (date.fromisoformat(asof)-date.fromisoformat(report['period_end'])).days,
            'issued_ordinary_shares': issued, 'count_source': report['source'],
            'quote': quote, 'quote_currency': 'USD', 'ordinary_per_ADR': 2,
            'quote_source_sha256': plan['inputs'][SRC + 'QUOTES.json'],
            'workbook_sha256': plan['inputs'][SRC + 'external_followup/euroland_browser_quotes.xls'],
            'old_SHEL_value': prior['market_cap_usd'] if prior else None,
            'old_SHEL_rank_in_archive': prior['rank'] if prior else None,
            'new_SHEL_proxy_value': proxy,
            'new_to_old_proxy_ratio': proxy/prior['market_cap_usd'] if prior else None,
            'new_SHEL_candidate_rank': next(i+1 for i,r in enumerate(ranked) if r['symbol']=='SHEL'),
            'non_SHEL_32nd': {'symbol': boundary['symbol'], 'market_cap_usd': boundary['market_cap_usd']},
            'proxy_to_non_SHEL_boundary': proxy/boundary['market_cap_usd'],
            'unchanged_non_SHEL_sha256': hashlib.sha256(json.dumps(unchanged, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
            'old_top32': old, 'new_top32': new,
            'added': sorted(set(new)-set(old)), 'removed': sorted(set(old)-set(new)),
            'not_current_net_market_cap': True, 'not_upper_bound': True, 'not_certified_PIT_universe': True,
        }
        rows.append(row); pools[signal] = new
    early = {r['signal_date'] for r in rows}
    assert all(pools[k] == v for k,v in old_pools.items() if k not in early)
    issues = set(plan['known_quote_issues']['absent_dates']+[plan['known_quote_issues']['unresolved_A_difference']])
    impacted = sorted(issues & {r['asof'] for r in rows})
    assert not impacted
    changed = [r for r in rows if r['added'] or r['removed']]
    ratios = [r['new_to_old_proxy_ratio'] for r in rows if r['new_to_old_proxy_ratio'] is not None]
    result = {'schema': 'shell_pool_sensitivity168_data_v1', 'created_utc': now(), 'plan_sha256': sha(OUT/'PLAN.json'),
              'status': 'CONDITIONAL_PROXY_COMPARISON_NOT_PIT_CERTIFICATION', 'rows': rows,
              'summary': {'events': len(rows), 'membership_changed_events': len(changed),
                          'ordered_list_changed_events': sum(r['old_top32']!=r['new_top32'] for r in rows),
                          'known_quote_issues_at_target_dates': impacted,
                          'min_proxy_ratio_vs_legacy': min(ratios), 'max_proxy_ratio_vs_legacy': max(ratios),
                          'changed_events': [{'date':r['signal_date'],'added':r['added'],'removed':r['removed']} for r in changed],
                          'old_SHEL_member_events':sum('SHEL' in r['old_top32'] for r in rows),
                          'new_SHEL_member_events':sum('SHEL' in r['new_top32'] for r in rows)},
              'source_assumptions': plan['assumptions'], 'strategy_trials_added': 0, 'returns_computed': False,
              'frozen_files_modified': False, 'all_later_pools_unchanged': True}
    write_once(OUT/'DATA_COMPARISON.json',result)
    write_once(OUT/'CONDITIONAL_POOLS.json',pools)
    for path, expected in plan['inputs'].items():
        assert sha(ROOT/path) == expected, path
    print(json.dumps(result['summary'],ensure_ascii=False,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['register','run'])
    args=parser.parse_args();register() if args.phase=='register' else run()
