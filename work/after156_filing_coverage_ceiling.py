"""Offline source-only SEC coverage bounds; never an admission or market replay.

The untimed bound deliberately ignores page holes and dates. The closer screen
requires receipted pagination in a stated filing-date envelope, but cannot prove
the acceptance clock, historical identity, or the eventual event partition.
"""
import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

from after156_event_scope_local_review import Sources, require
from after156_filing_universe_inventory import rows as sec_rows

SELF = 'work/after156_filing_coverage_ceiling.py'
OUT = 'work/after156_filing_coverage_ceiling.json'
INVENTORY = 'work/after156_filing_universe_inventory.json'
META = 'outputs/monthly_local_linear70/META.json'
CAL = 'outputs/monthly_paired_target143/CALENDAR.json'
PINS = {
    INVENTORY: 'e5157e2c6da08253baa9e523be2465261b442b506fad3baf7216664f47ff7286',
    META: '3b46968f486e88fa170ac01f120dd5ef74ebd5a56243c6991c5a4fea506076f4',
    CAL: '8ee4da69fba84db1edee37a7e0f5af5d3ab16c6257e5d37cddbd531f01404622',
    'work/after156_filing_universe_inventory.py': '3251f80ec2c4c56cb9df04b22350fd6ba33b38afff8be7bf0bba6a1c943f9e86',
}
ERAS = [('early53', 0, 53), ('recent35', 53, 88), ('first88', 0, 88),
        ('later34', 88, 122), ('all122', 0, 122)]
SCOPES = ('current_metadata_links', 'all_locating_candidates_hypothesis')
TIERS = ('untimed_receipted_original_bound', 'closed_pages_untimed_original_screen',
         'conditional_date_envelope_screen', 'closed_pages_any_form_diagnostic')


def intersects(a, b, lo, hi):
    return a <= hi and b >= lo


def original_keys(records, lo=None, hi=None):
    # The report-key count does not claim different anchors or complete windows.
    return {(r[1], r[3]) for r in records if r[1] in ('10-Q', '10-K')
            and (lo is None or lo <= r[2] <= hi)}


def evaluate(snapshot, lo, hi):
    """No clock interpretation: only a conditional filing-date routing screen."""
    missing, other = [], []
    if not snapshot['root_receipted']:
        other.append('ROOT_RECEIPT_MISSING:' + snapshot['path'])
    if snapshot['retrieved_date'] is None or snapshot['retrieved_date'] < hi:
        other.append('ROOT_SNAPSHOT_NOT_KNOWN_THROUGH_ENVELOPE:' + snapshot['path'])
    for page in snapshot['pages']:
        if not intersects(page['lo'], page['hi'], lo, hi):
            continue
        if not page['present']:
            missing.append(page['path'])
        elif not page['receipt']:
            other.append('PAGE_RECEIPT_MISSING:' + page['path'])
        elif not page['count_match']:
            other.append('PAGE_COUNT_MISMATCH:' + page['path'])
    total = len(original_keys(snapshot['receipted_records']))
    dated = len(original_keys(snapshot['receipted_records'], lo, hi))
    closed = not missing and not other
    return dict(path=snapshot['path'], cik=snapshot['cik'],
        untimed_receipted_original_bound=snapshot['root_receipted'] and total >= 2,
        closed_pages_untimed_original_screen=closed and total >= 2,
        conditional_date_envelope_screen=closed and dated >= 2,
        closed_pages_any_form_diagnostic=closed,
        observed_receipted_original_report_keys_total=total,
        observed_receipted_original_report_keys_in_envelope=dated,
        missing_pages=sorted(missing), other_source_blockers=sorted(other))


def aggregate(candidates):
    return {tier: any(c[tier] for c in candidates) for tier in TIERS}


def era_counts(ids, slot_rows):
    ids = set(ids)
    return {name: sum(i in ids and lo <= r['signal_index'] < hi
                     for i, r in enumerate(slot_rows)) for name, lo, hi in ERAS}


def summaries(slot_rows, scope, tier):
    result = {}
    for name, lo, hi in ERAS:
        group = [r for r in slot_rows if lo <= r['signal_index'] < hi]
        selected = [r for r in group if r['by_identity_scope'][scope][tier]]
        month_counts = Counter(r['signal_index'] for r in selected)
        result[name] = dict(stock_candidate_slots=len(group), screened_slots=len(selected),
            fraction=len(selected) / len(group),
            distinct_symbols=len({r['symbol'] for r in selected}),
            months_with_at_least_8_symbols=sum(month_counts[i] >= 8 for i in range(lo, hi)),
            minimum_symbols_in_any_month=min(month_counts[i] for i in range(lo, hi)),
            maximum_symbols_in_any_month=max(month_counts[i] for i in range(lo, hi)))
    return result


def floor_comparison(summary):
    e, r, f = (summary[k] for k in ('early53', 'recent35', 'first88'))
    checks = dict(first88_fraction_at_least_50pct=f['fraction'] >= .50,
        early53_fraction_at_least_40pct=e['fraction'] >= .40,
        recent35_fraction_at_least_40pct=r['fraction'] >= .40,
        first88_at_least_12_symbols=f['distinct_symbols'] >= 12,
        early53_at_least_8_symbols=e['distinct_symbols'] >= 8,
        recent35_at_least_8_symbols=r['distinct_symbols'] >= 8,
        early53_at_least_40_months_with_8_symbols=e['months_with_at_least_8_symbols'] >= 40,
        recent35_at_least_26_months_with_8_symbols=r['months_with_at_least_8_symbols'] >= 26)
    return dict(numerical_comparisons=checks, all_numerical_comparisons_met=all(checks.values()),
        actual_source_identity_clock_gate_passed=False,
        note='Distinct symbols are only an optimistic proxy for distinct issuers until historical identity is verified.')


def self_test():
    a = ('a', '10-Q', '2020-04-01', '2020-03-31', 'raw', 'a.htm')
    b = ('b', '10-K', '2020-02-01', '2019-12-31', 'raw', 'b.htm')
    amended = ('c', '10-Q/A', '2020-04-01', '2020-03-31', 'raw', 'c.htm')
    foreign = ('d', '20-F', '2020-04-01', '2020-03-31', 'raw', 'd.htm')
    duplicate_key = ('e', '10-Q', '2020-04-02', '2020-03-31', 'raw', 'e.htm')
    require(len(original_keys([a, b, amended, foreign, duplicate_key])) == 2, 'Original report keys')
    require(len(original_keys([a, b], '2020-04-01', '2020-04-01')) == 1, 'Inclusive date route')
    require(intersects('2019-01-01', '2020-01-01', '2020-01-01', '2021-01-01'), 'Page boundary')
    require(not intersects('2019-01-01', '2019-12-31', '2020-01-01', '2021-01-01'), 'Page outside')
    base = dict(path='root', cik='1', root_receipted=True, retrieved_date='2020-12-31',
                pages=[], receipted_records=[a, b])
    good = evaluate(base, '2020-01-01', '2020-06-01')
    require(good['conditional_date_envelope_screen'], 'Closed two-original screen')
    hole = dict(base, pages=[dict(path='page', lo='2010-01-01', hi='2020-12-31',
                                present=False, receipt=False, count_match=False)])
    bad = evaluate(hole, '2020-01-01', '2020-06-01')
    require(bad['untimed_receipted_original_bound'] and not bad['conditional_date_envelope_screen']
            and bad['missing_pages'] == ['page'], 'Loose bound is not pagination closure')
    no_receipt = evaluate(dict(base, root_receipted=False), '2020-01-01', '2020-06-01')
    require(not no_receipt['untimed_receipted_original_bound'], 'Root receipt required')
    foreign_only = evaluate(dict(base, receipted_records=[foreign, amended]), '2020-01-01', '2020-06-01')
    require(foreign_only['closed_pages_any_form_diagnostic'] and
            not foreign_only['conditional_date_envelope_screen'], 'Any-form closure is not original closure')
    old = evaluate(dict(base, retrieved_date='2020-03-01'), '2020-01-01', '2020-06-01')
    require(not old['conditional_date_envelope_screen'], 'Stale root cannot close future interval')
    one_a = evaluate(dict(base, receipted_records=[a]), '2020-01-01', '2020-06-01')
    one_b = evaluate(dict(base, receipted_records=[b], cik='2'), '2020-01-01', '2020-06-01')
    require(not aggregate([one_a, one_b])['conditional_date_envelope_screen'], 'Never splice CIK histories')
    limited = evaluate(base, '2020-03-01', '2020-06-01')
    require(limited['closed_pages_untimed_original_screen'] and
            not limited['conditional_date_envelope_screen'], 'Date-count relaxation explicit')
    return 11


class BoundInputs:
    def __init__(self, source, inventory):
        self.source, self.inventory = source, inventory
        self.cache = {}

    def read(self, path, expected):
        if path not in self.cache:
            self.cache[path] = self.source.read(path, expected)
        else:
            require(self.source.hashes[path] == expected, 'Conflicting source pin: ' + path)
        return self.cache[path]

    def proof(self, proof):
        path = proof['path']
        raw = self.read(path, proof['sha256'])
        receipt_path = proof.get('receipt')
        if not proof['source_receipt_verified']:
            require(not receipt_path, 'Unexpected unverified receipt')
            missing = proof['missing_source_receipt']
            require(not self.source.path(missing).exists(), 'Inventory stale: receipt appeared ' + missing)
            return raw, False, None
        rec = self.read(receipt_path, self.inventory['input_hashes'][receipt_path])
        require(rec.get('sha256') == proof['sha256'] and
                rec.get('source', rec.get('url')) == proof['expected_source_url'],
                'Original source receipt mismatch: ' + path)
        retrieved = rec.get('retrieved_at')
        try:
            stamp = datetime.fromisoformat(retrieved.replace('Z', '+00:00'))
            require(stamp.tzinfo is not None, 'Naive retrieval timestamp')
            day = stamp.astimezone(timezone.utc).date().isoformat()
        except (TypeError, ValueError, AttributeError):
            day = None
        return raw, True, day

    def snapshot(self, item):
        raw, receipted, day = self.proof(item['source'])
        require(str(raw['cik']).zfill(10) == item['cik'], 'Snapshot CIK changed')
        records = sec_rows(raw['filings']['recent']) if receipted else []
        all_records = sec_rows(raw['filings']['recent'])
        require(raw['filings'].get('files', []) == [p['advertised'] for p in item['advertised_pages']],
                'Advertised descriptors changed')
        pages = []
        for p in item['advertised_pages']:
            desc, present = p['advertised'], 'source' in p
            if p['status'] == 'MISSING_LOCAL_PAGE':
                require(not self.source.path(p['path']).exists(), 'Inventory stale: page appeared ' + p['path'])
            if present:
                table, receipt, _ = self.proof(p['source'])
                page_rows = sec_rows(table)
                match = len(page_rows) == desc['filingCount']
                require(match == p['count_matches'], 'Page row count changed')
                all_records.extend(page_rows)
                if receipt:
                    records.extend(page_rows)
            else:
                receipt, match = False, False
            # Some old pages lie outside 2015..2026 and were intentionally not parsed.
            pages.append(dict(path=p['path'], lo=desc['filingFrom'], hi=desc['filingTo'],
                present=present, receipt=receipt, count_match=match, status=p['status'],
                advertised_count=desc['filingCount']))
        by_accession = defaultdict(set)
        for r in all_records:
            by_accession[r[0]].add(r)
        require(not any(len(v) > 1 for v in by_accession.values()), 'Conflicting accession rows')
        # The parent inventory only parsed advertised pages in this declared scope.
        records = sorted(set(r for r in records if '2015-01-01' <= r[2] <= '2026-10-06'))
        for r in records:
            if r[1] in ('10-Q', '10-K'):
                require(date.fromisoformat(r[3]) <= date.fromisoformat(r[2]), 'Report date after filing')
        return dict(path=item['source']['path'], cik=item['cik'], root_receipted=receipted,
            retrieved_date=day, pages=pages, receipted_records=records)


def priority_report(priority, rows):
    result = []
    for path, p in priority.items():
        item = dict(path=path, symbols=sorted(p['symbols']), ciks=sorted(p['ciks']),
                    co_required_missing_pages=sorted(p['co_pages'] - {path}))
        for field in ('required_slots', 'blocked_slots', 'single_page_screen_unlock',
                      'single_page_possible_if_originals_supplied'):
            item[field] = era_counts(p[field], rows)
        result.append(item)
    result.sort(key=lambda p: (-p['single_page_screen_unlock']['first88'],
        -p['single_page_possible_if_originals_supplied']['first88'],
        -p['blocked_slots']['first88'], -p['blocked_slots']['all122'], p['path']))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    checks = self_test()
    src = Sources()
    inventory = src.read(INVENTORY, PINS[INVENTORY])
    meta = src.read(META, PINS[META])
    calendar = src.read(CAL, PINS[CAL])['sessions']
    design = src.read('work/after156_filing_drift_design.json')
    require(design['formation_and_nuisance_beta']['formation_return_offsets'] == [-251, -42], 'Formation changed')
    # Resolve this field structurally without assuming a prose section name.
    chronology = [v for v in design.values() if isinstance(v, dict) and 'source_anchor_offsets_required' in v]
    require(len(chronology) == 1 and chronology[0]['source_anchor_offsets_required'] == [-271, -42], 'Chronology changed')
    warmup = src.read('work/after156_filing_warmup.json')
    require(calendar == sorted(set(calendar)), 'Calendar must be sorted and unique')
    require(len(meta['dates']) == 122 and all(len(s) == 49 for s in meta['symbols']), 'Original grid changed')
    require(len(meta['etfs']) == 17, 'ETF taxonomy changed')
    stock_map = {r['symbol']: r for r in inventory['stocks']}
    require(set(stock_map) == set(s for seq in meta['symbols'] for s in seq) - set(meta['etfs'])
            and len(stock_map) == 73, 'Original 73-stock inventory differs')
    data = BoundInputs(src, inventory)
    snapshots = {s['source']['path']: data.snapshot(s) for s in inventory['submission_snapshots']}
    print('Verified %d inventoried snapshots; evaluating metadata-only slot bounds.' % len(snapshots),
          file=sys.stderr, flush=True)
    rows, details = [], []
    for i, (signal, symbols) in enumerate(zip(meta['dates'], meta['symbols'])):
        t = bisect_left(calendar, signal)
        require(t < len(calendar) and calendar[t] == signal and t >= 272, 'Signal calendar axis')
        lo, hi = calendar[t-272], calendar[t-41]
        require('2015-01-01' <= lo <= hi <= '2026-10-06', 'Envelope outside inventoried date scope')
        for j, symbol in enumerate(symbols):
            if symbol in meta['etfs']:
                continue
            stock = stock_map[symbol]
            row = dict(signal_index=i, candidate_index=j, signal_date=signal, symbol=symbol,
                required_anchor_date_bounds=[calendar[t-271], calendar[t-42]],
                conditional_filing_date_envelope=[lo, hi], by_identity_scope={})
            raw_scope = {}
            for scope in SCOPES:
                ciks = stock['exact_current_ciks'] if scope == SCOPES[0] else stock['local_snapshot_paths'].keys()
                paths = sorted(set(p for c in ciks for p in stock['local_snapshot_paths'][c]))
                candidates = [evaluate(snapshots[p], lo, hi) for p in paths]
                require(all(not intersects(p['lo'], p['hi'], lo, hi)
                            or p['status'] != 'PRESENT_OUTSIDE_SCOPE_NOT_PARSED'
                            for path in paths for p in snapshots[path]['pages']), 'Required page not inventoried')
                result = aggregate(candidates)
                result['conditional_supporting_snapshot'] = next(
                    (c['path'] for c in candidates if c['conditional_date_envelope_screen']), None)
                result['maximum_observed_original_keys_in_one_snapshot_envelope'] = max(
                    [c['observed_receipted_original_report_keys_in_envelope'] for c in candidates] or [0])
                result['no_local_snapshot'] = not candidates
                row['by_identity_scope'][scope] = result
                raw_scope[scope] = candidates
            rows.append(row)
            details.append(raw_scope)
    require(len(rows) == 3904, 'Original stock slot denominator changed')
    require([(r['signal_index'], r['candidate_index'], r['symbol']) for r in rows] ==
            [(r['signal_index'], r['candidate_index'], r['symbol']) for r in warmup['rows']], 'Warmup grid differs')
    require(all(r['axis_ready'] for r in warmup['rows']), 'Known metadata warmup no longer complete')
    priority, blockers = {}, {}
    summary, floors = {}, {}
    for scope in SCOPES:
        summary[scope] = {tier: summaries(rows, scope, tier) for tier in TIERS}
        floors[scope] = {tier: floor_comparison(summary[scope][tier]) for tier in TIERS[:3]}
        pmap = defaultdict(lambda: dict(symbols=set(), ciks=set(), co_pages=set(), required_slots=set(),
            blocked_slots=set(), single_page_screen_unlock=set(), single_page_possible_if_originals_supplied=set()))
        bmap = defaultdict(set)
        for index, (row, choices) in enumerate(zip(rows, details)):
            candidates = choices[scope]
            passed = row['by_identity_scope'][scope]['conditional_date_envelope_screen']
            if not candidates:
                bmap['NO_LOCAL_SNAPSHOT'].add(index)
            if not passed:
                if candidates and all(c['observed_receipted_original_report_keys_in_envelope'] < 2 for c in candidates):
                    bmap['FEWER_THAN_TWO_OBSERVED_RECEIPTED_ORIGINAL_KEYS_IN_ENVELOPE'].add(index)
                for c in candidates:
                    for reason in c['other_source_blockers']:
                        bmap[reason].add(index)
                    if c['missing_pages']:
                        bmap['REQUIRED_ADVERTISED_PAGES_MISSING'].add(index)
            for c in candidates:
                for page in c['missing_pages']:
                    p = pmap[page]
                    p['symbols'].add(row['symbol']); p['ciks'].add(c['cik'])
                    p['co_pages'].update(c['missing_pages']); p['required_slots'].add(index)
                    if not passed:
                        p['blocked_slots'].add(index)
                        if len(c['missing_pages']) == 1 and not c['other_source_blockers']:
                            p['single_page_possible_if_originals_supplied'].add(index)
                            if c['observed_receipted_original_report_keys_in_envelope'] >= 2:
                                p['single_page_screen_unlock'].add(index)
        priority[scope] = priority_report(pmap, rows)
        blockers[scope] = [dict(reason=k, counts=era_counts(v, rows)) for k, v in sorted(bmap.items())]
    for row in rows:
        for scope in SCOPES:
            r = row['by_identity_scope'][scope]
            require(not r['conditional_date_envelope_screen'] or r['closed_pages_untimed_original_screen'], 'Screen nesting')
            require(not r['closed_pages_untimed_original_screen'] or r['untimed_receipted_original_bound'], 'Bound nesting')
        for tier in TIERS:
            require(not row['by_identity_scope'][SCOPES[0]][tier]
                    or row['by_identity_scope'][SCOPES[1]][tier], 'Identity-envelope nesting')
    for path in (SELF, 'work/after156_event_scope_local_review.py',
                 'work/after156_filing_universe_inventory.py', 'AGENTS.md', 'LOCAL_SETUP.md',
                 'run_local.sh', 'local_run.py', 'local.sandbox.sb'):
        src.read_bytes(path, PINS.get(path))
    src.recheck()
    report = dict(schema_version=1, status='SOURCE_ONLY_OPTIMISTIC_BOUNDS_NOT_ADMISSION',
        created_utc=datetime.now(timezone.utc).isoformat(), runtime_python=sys.version,
        scope=dict(original_stocks=73, excluded_etfs=17, original_months=122, stock_candidate_slots=3904,
            parsed_snapshot_count=len(snapshots), original_forms=['10-Q', '10-K'],
            no_snapshot_or_CIK_histories_concatenated=True, original_candidate_denominator_preserved=True),
        definitions=dict(
            identity_scopes={SCOPES[0]: 'Exact current metadata CIK links only, optimistically assumed historically valid; this is not historical proof.',
                SCOPES[1]: 'Any one existing local snapshot of any inventoried current or unverified locating CIK may support a slot; no CIK histories are spliced. This is an optimistic disjunction, not an admitted mapping.'},
            untimed_receipted_original_bound='Root has an original source receipt and one coherent snapshot contains at least two distinct receipted original (form, reportDate) keys anywhere in inventoried 2015-01-01..2026-10-06. Ignore missing pages, all event dates, stale snapshots, bundles, amendments, and price validity. Thus this bounds AS-IS available receipted-original support, not support after new sources are added.',
            closed_pages_untimed_original_screen='Same untimed original-key count, plus root receipt retrieved no earlier than envelope end and every advertised page intersecting the envelope locally present, row-count matching and originally receipted.',
            conditional_date_envelope_screen='Same source closure, but require two distinct receipted original report keys whose raw filingDate lies in [session(t-272),session(t-41)]. Required anchors are [t-271,t-42]. This is a conditional, deliberately relaxed routing screen, not an accepted-time mapping or a strict upper bound under arbitrary timestamp errors.',
            envelope_assumption='Assume future clock review confirms that every potentially relevant original/amendment record has its filingDate within this expanded calendar envelope, and advertised page date descriptors cover those records. No filingDate is assigned an anchor; raw acceptanceDateTime is never interpreted here. An edge page included by this conservative envelope may be unnecessary under eventual verified anchors, so failure of this closer screen alone is not proof of unavoidable gate failure.',
            closed_pages_any_form_diagnostic='Pagination/receipt closure only; explicitly NOT counted as original periodic-report coverage.',
            report_key='(CIK, exact original form, reportDate), counted separately within each coherent snapshot. Multiple accessions for a report key count once optimistically; no claim of adjudication or different full-session anchor bundles.',
            missing_page_priority='Rank by first88 single-page screen unlocks with two originals already observed, then possible single-page unlocks if the page supplies enough originals, then blocked slots. Counts are metadata-screen opportunities, never guaranteed admissions; a delivered page must have matching provenance/counts and no conflicts.',
            blocker_counts='Nonexclusive unions over alternative snapshots. A blocker listed for an alternative need not be the sole explanation for a slot. Missing-page priority excludes currently passing slots from its blocked/unlock counts.'),
        summary=summary, numerical_floor_comparisons=floors,
        missing_page_priority=priority, nonexclusive_blocker_counts=blockers, rows=rows,
        candidate_ciks_without_local_submission=inventory['candidate_ciks_without_local_submission'],
        pending_gates=[
            'Independent dated security-to-issuer/CIK identity proof, including class/ADR/successor continuity; all current and locating links remain unadmitted.',
            'Independent original acceptance timestamp/timezone corroboration, scheduled-open mapping and any boundary uncertainty adjudication.',
            'Full original-plus-amendment chronology review; page descriptors and supplied arrays alone do not independently certify SEC completeness.',
            'Original report-key ambiguity, orphan amendment, accession contradiction and same-session bundle adjudication.',
            'Verified-clock E/B/N partition: two complete uninterrupted 19-day original bundles, n_E>=38 and n_N>=80; none calculated here.',
            'Positive finite original stock/SPY prices and valid beta; previous warmup proves axes only. No price values read here.',
            'Actual common-mask coverage gate and independent review before any registration, factor, order or outcome work.'],
        validation=dict(synthetic_checks_passed=checks, original_grid_reconciled=True,
            previous_axis_warmup_grid_reconciled=True, nested_bounds_verified=True,
            all_consumed_input_hashes_rechecked=True),
        activity_counts=dict(price_values_read=0, acceptance_instants_interpreted=0, anchors_assigned=0,
            factor_calculations=0, ranks=0, orders=0, returns=0, new_trials=0, network_requests=0,
            frozen_files_modified=0), input_hashes=dict(sorted(src.hashes.items())))
    if args.write:
        payload = (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
        with src.path(OUT).open('xb') as f:
            f.write(payload)
        require(src.path(OUT).read_bytes() == payload, 'Output round-trip')
        output = dict(path=OUT, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    else:
        output = None
    compact = {s: {t: {e: dict(slots=v['screened_slots'], denominator=v['stock_candidate_slots'],
                fraction=v['fraction'], distinct_symbols=v['distinct_symbols'],
                months_with_8=v['months_with_at_least_8_symbols'])
                for e, v in summary[s][t].items()} for t in TIERS[:3]} for s in SCOPES}
    print(json.dumps(dict(summary=compact, missing_pages={s:len(priority[s]) for s in SCOPES},
        first_5_page_priorities={s:priority[s][:5] for s in SCOPES},
        synthetic_checks=checks, script_sha256=src.hashes[SELF], output=output), indent=2))


if __name__ == '__main__':
    main()
