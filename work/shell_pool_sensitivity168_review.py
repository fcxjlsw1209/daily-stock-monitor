"""Independent audit of the single Shell issued-share proxy sensitivity.

Only writes a new independent report. Does not import producer code, rebuild
features, execute a strategy, or claim the proxy is actual market capitalization.
"""
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import platform
import re
import sys
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/monthly_shell_pool_sensitivity168'
SRC = ROOT/'outputs/monthly_shell_pool_source168'
PINS = {
    'PLAN.json': 'c50e7a4edd9b83a23efe7a290f6fa82d5f920141b02aff1aed34520191c7d2a6',
    'DATA_COMPARISON.json': '6b6afc5fa84a3abcbeb05e108e2aecb35c6fc30e5c85c274c0b2455ac6980c32',
    'CONDITIONAL_POOLS.json': '6efd9c7e4d47b0480453fde1f549ac942205e58bacbb7a9a6ae26b96c6a1132e',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def check(value, label):
    if not value:
        raise AssertionError(label)


def match(a, b, path=''):
    if isinstance(b, dict):
        check(a.keys() == b.keys(), path+' keys')
        for k in b:
            match(a[k], b[k], path+'/'+k)
    elif isinstance(b, list):
        check(len(a) == len(b), path+' length')
        for i, pair in enumerate(zip(a, b)):
            match(*pair, path+'/'+str(i))
    elif isinstance(b, float):
        # Price*share-count floats may differ by several ULP versus decimal.
        tolerance = 0.0001 if abs(b) > 1e6 else 1e-13
        check(isinstance(a, (float, int)) and math.isfinite(a) and abs(a-b) <= tolerance, path+' numeric')
    else:
        check(a == b, path+' value')


def original_reports(plan, extra_pins):
    """Recheck original pages against the preexisting independent visual audit."""
    review = load(SRC/'INDEPENDENT_REVIEW.json')
    producer = load(SRC/'EXTRACTION.json')
    runtime_path = ROOT/producer['pdf_runtime_pins_source']
    check(digest(runtime_path) == producer['pdf_runtime_pins_sha256'], 'PDF runtime plan')
    extra_pins[str(runtime_path)] = digest(runtime_path)
    runtime = load(runtime_path)
    for path, expected in runtime['pdf_runtime_hashes'].items():
        check(digest(path) == expected, 'PDF runtime '+path)
        extra_pins[path] = expected
    sys.path.append(runtime['pdf_runtime']['bundle_site_packages'])
    import pypdfium2 as pdfium
    records = {r['period']: r for r in producer['records']}
    independently_read = []
    check(len(records) == len(review['records']) == 20, 'Twenty QRA documents')
    for r in review['records']:
        source = ROOT/r['source_binary']
        check(digest(source) == r['source_sha256'] == plan['inputs'][r['source_binary']], 'Raw QRA digest')
        doc = pdfium.PdfDocument(source.read_bytes())
        cp, rp = r['capital_pdf_page_1based'], r['release_pdf_page_1based']
        cap = ' '.join(doc[cp-1].get_textpage().get_text_range().split())
        released = ' '.join(doc[rp-1].get_textpage().get_text_range().split())
        for field in ['independent_A', 'independent_B']:
            check(format(r[field], ',') in cap, r['period']+' original count')
        for day, text in [(r['period_end'], cap), (r['release_stated_date'], released)]:
            human = datetime.fromisoformat(day).strftime('%B %d, %Y').replace(' 0', ' ')
            check(human in text, r['period']+' original date')
        check(bool(re.search(r'issued.{0,30}(?:fully|share)', cap, re.I)), 'Issued table basis')
        check('0.07' in cap, 'Ordinary nominal unit')
        p = records[r['period']]
        s = p['issued_ordinary_shares']
        check((s['A'], s['B']) == (r['independent_A'], r['independent_B']), 'QRA A/B extraction')
        check(s['units'] == 'ordinary shares; not millions', 'Exact share units')
        check(s['source_page_1based'] == cp and p['release']['source_page_1based'] == rp, 'QRA pages')
        check(p['period_end'] == r['period_end'] and p['release']['stated_date'] == r['release_stated_date'], 'QRA dates')
        check(p['source']['binary'] == r['source_binary'] and p['source']['sha256'] == r['source_sha256'], 'QRA source binding')
        independently_read.append(dict(period=r['period'], A=r['independent_A'], B=r['independent_B'],
                                       period_end=r['period_end'], release=r['release_stated_date'],
                                       capital_page=cp, release_page=rp, binary=r['source_binary'],
                                       original_sha256=r['source_sha256']))
        doc.close()
    return independently_read, records


def original_quotes(plan):
    qp = load(SRC/'QUOTES_PLAN.json')
    wheel = ROOT/qp['source_files']['wheel']['path']
    sys.path.insert(0, str(wheel))
    import xlrd
    check(str(wheel) in xlrd.__file__, 'Isolated xlrd wheel')
    xls = ROOT/qp['source_files']['xls']['path']
    check(digest(xls) == plan['inputs'][str(xls.relative_to(ROOT))], 'Raw XLS pin')
    book = xlrd.open_workbook(file_contents=xls.read_bytes())
    check(book.sheet_names() == ['Share Data'] and book.datemode == 0, 'XLS sheet and date system')
    sheet = book.sheet_by_index(0)
    check(sheet.row_values(1) == ['Share', 'Royal Dutch Shell A ADR (NYSE)', '', 'Royal Dutch Shell B ADR (NYSE)', ''], 'XLS security columns')
    check(sheet.row_values(2) == ['Currency', 'USD', '', 'USD', ''], 'XLS units')
    check(sheet.row_values(3) == ['Date', 'Close', 'Volume', 'Close', 'Volume'], 'XLS field columns')
    quotes = {}
    for i in range(4, sheet.nrows):
        serial, a, av, b, bv = sheet.row_values(i)
        check(serial == int(serial) and serial > 61, 'XLS integral date')
        key = (date(1899, 12, 30)+timedelta(days=int(serial))).isoformat()
        check(key not in quotes, 'Duplicate XLS date')
        check(all(math.isfinite(p) and p > 0 for p in [a,b]), 'Positive finite prices')
        check(all(v >= 0 and v == int(v) for v in [av,bv]), 'Integral volumes')
        quotes[key] = dict(date=key, xls_row=i+1, class_A=dict(close=a, volume=int(av)), class_B=dict(close=b, volume=int(bv)))
    check(list(quotes.values()) == load(SRC/'QUOTES.json')['rows'], 'All raw XLS cells equal transcription')
    return quotes


def main():
    for name, expected in PINS.items():
        check(digest(OUT/name) == expected, 'Frozen output '+name)
    plan = load(OUT/'PLAN.json')
    for path, expected in plan['inputs'].items():
        check(digest(ROOT/path) == expected, 'Frozen input '+path)
    check(digest(ROOT/'work/shell_pool_sensitivity168.py') == plan['script_sha256'], 'Producer code pin')
    runtime = dict(python=platform.python_version(), **{name: importlib.metadata.version(name) for name in ['numpy','pandas','exchange-calendars']})
    check(runtime == plan['runtime'], 'Pinned runtime')
    extras = {}
    qras, full_records = original_reports(plan, extras)
    quotes = original_quotes(plan)
    import exchange_calendars as xcals
    sessions = xcals.get_calendar(**plan['calendar']).sessions.strftime('%Y-%m-%d').tolist()
    spy_path = ROOT/'outputs/index_union/sources/yahoo_prices/SPY.csv'
    spy_days = sorted(r['Date'][:10] for r in csv.DictReader(io.StringIO(spy_path.read_text())))
    # Independent linear next-date search, cross-checked against the SPY archive.
    for r in qras:
        r['available'] = min(s for s in sessions if s > r['release'])
        check(r['available'] == min(s for s in spy_days if s > r['release']), 'QRA next session/SPY agreement')
    archived = load(ROOT/'outputs/monthly_2016_extension/EARLY_RANKS.json')
    old_pools = load(ROOT/'outputs/monthly_2016_extension/POOLS.json')
    actual = load(OUT/'DATA_COMPARISON.json')
    conditional = load(OUT/'CONDITIONAL_POOLS.json')
    check(actual['plan_sha256'] == PINS['PLAN.json'], 'Output plan pin')
    check(plan['registered_utc'] < actual['created_utc'], 'Registration chronology')
    check(len(archived) == len(actual['rows']) == 53, '53 comparisons')
    check(len({x['date'] for x in archived}) == 53, '53 distinct signals')
    expected_pools = dict(old_pools)
    expected_rows, audited_rows = [], []
    d = lambda x: Decimal(str(x))
    order = lambda x: (-d(x['market_cap_usd']), x['symbol'])
    for event, produced in zip(archived, actual['rows']):
        signal, asof, old = event['date'], event['asof'], event['top40']
        check(signal == min(s for s in sessions if s > asof), 'Signal follows asof')
        check(asof in sessions and asof in spy_days and signal == min(s for s in spy_days if s > asof), 'Signal/asof calendar agreement')
        check(len(old) == len({x['symbol'] for x in old}) == 40, '40 unique archived issuers')
        check(old == sorted(old, key=order), 'Original Top40 canonical order')
        check([r['rank'] for r in old] == list(range(1,41)), 'Original rank fields')
        old_names = [x['symbol'] for x in old[:32]]
        check(old_names == old_pools[signal], 'Original Top32 matches pool')
        qualifying = [r for r in qras if r['available'] <= asof and r['period_end'] <= r['release']]
        check(bool(qualifying), 'A published report exists')
        chosen = sorted(qualifying, key=lambda r: (r['release'], r['period_end'], r['period']))[-1]
        check(chosen['release'] < chosen['available'] <= asof < signal, 'Information timing')
        quote = quotes[asof]
        exact = (Decimal(chosen['A'])*d(quote['class_A']['close']) + Decimal(chosen['B'])*d(quote['class_B']['close']))/Decimal(2)
        proxy = float(exact)
        other = [r for r in old if r['symbol'] != 'SHEL']
        pairs = [(d(r['market_cap_usd']), r['symbol']) for r in other] + [(exact, 'SHEL')]
        pairs.sort(key=lambda x: (-x[0], x[1]))
        names = [symbol for cap,symbol in pairs[:32]]
        expected_pools[signal] = names
        old_shell = [r for r in old if r['symbol'] == 'SHEL']
        prior = old_shell[0] if old_shell else None
        boundary = sorted(other,key=order)[31]
        report = full_records[chosen['period']]
        expected = dict(signal_date=signal, asof=asof, selected_period=chosen['period'],
            count_effective_date=chosen['period_end'], release_date=chosen['release'], usable_session=chosen['available'],
            count_age_calendar_days=(date.fromisoformat(asof)-date.fromisoformat(chosen['period_end'])).days,
            issued_ordinary_shares=report['issued_ordinary_shares'], count_source=report['source'],
            quote=quote, quote_currency='USD', ordinary_per_ADR=2,
            quote_source_sha256=plan['inputs']['outputs/monthly_shell_pool_source168/QUOTES.json'],
            workbook_sha256=plan['inputs']['outputs/monthly_shell_pool_source168/external_followup/euroland_browser_quotes.xls'],
            old_SHEL_value=prior['market_cap_usd'] if prior else None, old_SHEL_rank_in_archive=prior['rank'] if prior else None,
            new_SHEL_proxy_value=proxy, new_to_old_proxy_ratio=float(exact/d(prior['market_cap_usd'])) if prior else None,
            new_SHEL_candidate_rank=[p[1] for p in pairs].index('SHEL')+1,
            non_SHEL_32nd={k:boundary[k] for k in ['symbol','market_cap_usd']},
            proxy_to_non_SHEL_boundary=float(exact/d(boundary['market_cap_usd'])),
            unchanged_non_SHEL_sha256=hashlib.sha256(json.dumps(other,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            old_top32=old_names, new_top32=names, added=sorted(set(names)-set(old_names)), removed=sorted(set(old_names)-set(names)),
            not_current_net_market_cap=True, not_upper_bound=True, not_certified_PIT_universe=True)
        match(produced, expected, signal)
        expected_rows.append(expected)
        audited_rows.append(dict(signal_date=signal, asof=asof, report=chosen['period'], available=chosen['available'],
                                  xls_row=quote['xls_row'], independently_computed_proxy_decimal=str(exact),
                                  maximum_proxy_float_error_usd=abs(produced['new_SHEL_proxy_value']-proxy),
                                  old_SHEL_rank=expected['old_SHEL_rank_in_archive'], new_SHEL_candidate_rank=expected['new_SHEL_candidate_rank'],
                                  every_producer_field_matches=True, membership_unchanged=set(old_names)==set(names)))
    check(expected_pools == conditional, 'Entire conditional pool output')
    early = {r['signal_date'] for r in expected_rows}
    check(all(conditional[k] == v for k,v in old_pools.items() if k not in early), 'Later pools exact')
    issues = set(plan['known_quote_issues']['absent_dates']+[plan['known_quote_issues']['unresolved_A_difference']])
    affected = sorted(issues & {r['asof'] for r in expected_rows})
    check(not affected, 'No known issue at target dates')
    changed = [r for r in expected_rows if r['added'] or r['removed']]
    ratios = [r['new_to_old_proxy_ratio'] for r in expected_rows if r['new_to_old_proxy_ratio'] is not None]
    summary = dict(events=len(expected_rows), membership_changed_events=len(changed),
        ordered_list_changed_events=sum(r['old_top32'] != r['new_top32'] for r in expected_rows),
        known_quote_issues_at_target_dates=affected, min_proxy_ratio_vs_legacy=min(ratios), max_proxy_ratio_vs_legacy=max(ratios),
        changed_events=[dict(date=r['signal_date'],added=r['added'],removed=r['removed']) for r in changed],
        old_SHEL_member_events=sum('SHEL' in r['old_top32'] for r in expected_rows), new_SHEL_member_events=sum('SHEL' in r['new_top32'] for r in expected_rows))
    match(actual['summary'],summary,'summary')
    check(actual['source_assumptions'] == plan['assumptions'], 'Assumptions retained')
    report = dict(schema='shell_pool_sensitivity168_independent_review_v1', created_utc=datetime.now(timezone.utc).isoformat(),
        status='PASS_SINGLE_DECLARED_PROXY_SENSITIVITY_ONLY', reviewer_script=str(Path(__file__).relative_to(ROOT)),
        reviewer_script_sha256=digest(Path(__file__)), producer_imported=False, runtime=runtime,
        frozen_output_pins=PINS, source_input_pins=plan['inputs'], additional_runtime_pins=extras,
        producer_script_sha256=plan['script_sha256'], summary=summary, audited_rows=audited_rows,
        QRA_source_checks=qras, QRA_original_capital_and_release_pages_rechecked=40,
        source_note='Prior independent visual transcription of twenty original QRA tables reused; all forty capital/release pages re-extracted from pinned originals to corroborate counts and dates. No producer helper imported.',
        all_53_row_fields_match=True, all_XLS_transcription_cells_match=True, all_conditional_pool_records_match=True,
        later_pool_records_unchanged=len(old_pools)-len(early),
        independent_arithmetic='Decimal counts times decimalized original XLS prices; final float tolerances 0.0001 USD for capitalization magnitudes and 1e-13 for ratios.',
        manual_assumption_review=dict(no_positive_contradiction_identified_in_reviewed_sources=True,
            automated_universal_contradiction_detector=False,
            evidence_scope='Only the pinned QRA, ratio page, quote workbook and existing source reviews; no claim to have searched all possible corporate actions or vendor methodology.',
            remaining_assumptions=plan['assumptions']),
        limitations=['The substituted value is a last-published-issued-share proxy, not current net market capitalization or a strict bound.',
            'The new ranks are conditional on archived Top40 candidates and legacy non-Shell values; no full-market coverage proof follows.',
            'No membership change is not proof that either universe is correct or that unseen candidates cannot qualify.',
            'Zero membership changes alone does not authorize reuse of a strategy result if downstream logic consumes input ordering; this audit does not execute or certify downstream strategy invariance.',
            'Historical-vintage, share-count staleness, employee-trust/treasury netting, continuous ADR ratio and quote-adjustment assumptions remain unresolved.'],
        strategy_returns_computed=False, strategy_trials_added=0, existing_files_modified=False)
    for name, expected in PINS.items():
        check(digest(OUT/name) == expected, 'Post-check output '+name)
    for path, expected in plan['inputs'].items():
        check(digest(ROOT/path) == expected, 'Post-check input '+path)
    for path, expected in extras.items():
        check(digest(path) == expected, 'Post-check runtime '+path)
    with (OUT/'INDEPENDENT_DATA_REVIEW.json').open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(dict(status=report['status'], summary=summary, report_sha256=digest(OUT/'INDEPENDENT_DATA_REVIEW.json')),indent=2))


if __name__ == '__main__':
    main()
