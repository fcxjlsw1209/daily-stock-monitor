"""Bounded, offline index-event metadata join. No price or strategy imports."""
import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SELF = 'work/after159_index_event_join.py'
PLAN = 'work/after159_index_event_join_plan.json'
OUT = 'work/after159_index_event_join.json'
TABLES = [('NDX', 'outputs/index_union/ndx_primary_reviewed_rows.json', 'ticker_as_announced_or_effective'),
          ('SP100', 'outputs/index_union/sp100_modern_event_rows.json', 'ticker')]
META = 'outputs/monthly_local_linear70/META.json'
CAL = 'outputs/monthly_paired_target143/CALENDAR.json'
SPLITS = {'all122': (0,122), 'first88': (0,88), 'early53': (0,53),
          'recent35': (53,88), 'later34': (88,122)}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_path(name):
    p = Path(name)
    p = p if p.is_absolute() else ROOT / p
    require('..' not in p.parts and p.resolve() == p, 'Noncanonical or symlink path')
    require(p.is_relative_to(ROOT), 'External path forbidden')
    return p


def load(name):
    return json.loads(local_path(name).read_text())


def verify(pins):
    for name, expected in pins.items():
        require(digest(local_path(name)) == expected, 'Input hash mismatch: ' + name)


def iso_day(value):
    require(type(value) is str and len(value) == 10, 'Strict date required')
    result = date.fromisoformat(value)
    require(result.isoformat() == value, 'Noncanonical date')
    return value


def publication_day(value):
    require(type(value) is str, 'Publication timestamp missing')
    stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(stamp.tzinfo is not None and stamp.utcoffset() is not None, 'Aware publication timestamp required')
    return max(stamp.date().isoformat(), stamp.astimezone(timezone.utc).date().isoformat())


def source_reference(row):
    """Verify source bytes without parsing them; unsupported paths are retained."""
    name, expected = row.get('source_file'), row.get('source_sha256')
    item = dict(source_file=name, expected_sha256=expected, verified=False)
    try:
        require(type(name) is str and type(expected) is str and len(expected) == 64, 'Missing source reference')
        path = local_path('outputs/index_union/' + name)
        require(path.is_relative_to(ROOT / 'outputs/index_union/sources'), 'Source outside announcement cache')
        actual = digest(path)
        item.update(path=str(path.relative_to(ROOT)), actual_sha256=actual,
                    verified=actual == expected, reason=None if actual == expected else 'SOURCE_HASH_MISMATCH')
    except (OSError, ValueError) as exc:
        item['reason'] = str(exc)
    return item


def events_from_tables(tables, verify_sources=False):
    grouped = defaultdict(list)
    references = {}
    for index_name, rows, ticker_key in tables:
        require(type(rows) is list, 'Event table must be a list')
        for row_index, raw in enumerate(rows):
            require(type(raw) is dict, 'Event row must be object')
            errors = []
            ticker, action = raw.get(ticker_key), raw.get('action')
            if type(ticker) is not str or not ticker or ticker.strip() != ticker:
                errors.append('INVALID_EXACT_TICKER')
            if action not in ('addition', 'deletion'):
                errors.append('NOT_ADDITION_OR_DELETION')
            try:
                effective = iso_day(raw.get('effective_session'))
            except (TypeError, ValueError):
                effective = None
                errors.append('INVALID_EFFECTIVE_DAY')
            try:
                published = publication_day(raw.get('published_at'))
            except (TypeError, ValueError):
                published = None
                errors.append('INVALID_PUBLICATION_TIMESTAMP')
            ref_id = index_name + ':' + str(row_index)
            ref = source_reference(raw) if verify_sources else {'verified': True}
            references[ref_id] = ref
            key = (index_name, action, ticker, effective) if not errors else (index_name, 'invalid', ref_id, None)
            grouped[key].append(dict(source_row=ref_id, original_row=raw,
                                     publication_day=published, errors=errors))
    events = []
    for key, variants in grouped.items():
        rows = [v['original_row'] for v in variants]
        errors = sorted({err for v in variants for err in v['errors']})
        signatures = {(v['publication_day'], repr(v['original_row'].get('identity_verified')),
                       repr(v['original_row'].get('chain_verified'))) for v in variants}
        if len(signatures) != 1:
            errors.append('CONFLICTING_SEMANTIC_EVENT_VARIANTS')
        flags = {'identity_verified': all(r.get('identity_verified') is True for r in rows),
                 'chain_verified': all(r.get('chain_verified') is True for r in rows)}
        blockers = [name.upper() + '_NOT_TRUE' for name, value in flags.items() if not value]
        if not all(references[v['source_row']]['verified'] for v in variants):
            blockers.append('SOURCE_REFERENCE_NOT_VERIFIED')
        events.append(dict(event_id=json.dumps(key, separators=(',', ':')), index_name=key[0],
                           action=rows[0].get('action'), ticker=rows[0].get('ticker_as_announced_or_effective', rows[0].get('ticker')),
                           effective_day=key[3], publication_day=variants[0]['publication_day'],
                           metadata_errors=errors, source_blockers=blockers, original_variants=variants,
                           **flags))
    return events, references


def join_grid(meta, sessions, events, expected_shape=(122,49)):
    require(type(sessions) is list and sessions == sorted(set(sessions)), 'Unique sorted calendar required')
    for value in sessions:
        iso_day(value)
    dates, symbols, etfs = meta['dates'], meta['symbols'], meta['etfs']
    require(type(etfs) is list and len(set(etfs)) == len(etfs), 'Explicit unique ETF set required')
    require(len(dates) == len(symbols) == expected_shape[0], 'Original month axis mismatch')
    require(dates == sorted(set(dates)), 'Unique sorted signals required')
    positions = {d: i for i, d in enumerate(sessions)}
    by_ticker = defaultdict(list)
    for e in events:
        if not e['metadata_errors']:
            by_ticker[e['ticker']].append(e)
    rows = []
    for i, (signal, names) in enumerate(zip(dates, symbols)):
        require(len(names) == len(set(names)) == expected_shape[1], 'Original candidate axis mismatch')
        require(signal in positions and positions[signal] >= 294, 'Full frozen formation calendar required')
        t = positions[signal]
        start, cutoff = sessions[t-294], sessions[t-42]
        for j, symbol in enumerate(names):
            etf = symbol in etfs
            primary, older, same_cutoff = [], [], []
            for e in [] if etf else by_ticker[symbol]:
                p, d = e['publication_day'], e['effective_day']
                if start <= d <= cutoff and p <= cutoff:
                    if p < start:
                        older.append(e['event_id'])
                    else:
                        primary.append(e['event_id'])
                        if p == cutoff:
                            same_cutoff.append(e['event_id'])
            rows.append(dict(signal_index=i, candidate_index=j, symbol=symbol, signal_date=signal,
                             is_etf=etf, window_start=start, window_end=cutoff,
                             matched_event_ids=primary, publication_before_window_event_ids=older,
                             cutoff_day_publication_clock_pending=same_cutoff,
                             source_admitted=False, complete_event_coverage_verified=False))
    return rows


def summarize(rows, events, lo, hi):
    selected = [r for r in rows if lo <= r['signal_index'] < hi]
    matched = [r for r in selected if r['matched_event_ids']]
    ids = {key for r in matched for key in r['matched_event_ids']}
    lookup = {e['event_id']:e for e in events}
    month_symbols = defaultdict(set)
    for row in matched:
        month_symbols[row['signal_index']].add(row['symbol'])
    older = {key for r in selected for key in r['publication_before_window_event_ids']}
    return dict(candidate_slots=len(selected), non_etf_slots=sum(not r['is_etf'] for r in selected),
                etf_slots=sum(r['is_etf'] for r in selected), matched_non_etf_slots=len(matched),
                unique_events=len(ids), exact_ticker_issuer_proxies=len({r['symbol'] for r in matched}),
                months_with_event=len(month_symbols), months_with_at_least_two_issuer_proxies=sum(len(v)>=2 for v in month_symbols.values()),
                event_ids=sorted(ids), exact_tickers=sorted({r['symbol'] for r in matched}),
                event_counts_by_action=dict(Counter(lookup[k]['action'] for k in ids)),
                identity_not_verified_events=sum(not lookup[k]['identity_verified'] for k in ids),
                chain_not_verified_events=sum(not lookup[k]['chain_verified'] for k in ids),
                cutoff_day_publication_pending_slots=sum(bool(r['cutoff_day_publication_clock_pending']) for r in selected),
                effective_in_window_publication_before_window_unique_events=len(older),
                source_admitted_slots=0, complete_event_coverage_verified=False)


class Tests(unittest.TestCase):
    def setUp(self):
        from datetime import timedelta
        self.sessions = [(date(2000,1,1)+timedelta(days=i)).isoformat() for i in range(340)]
        self.meta = dict(dates=[self.sessions[320]], symbols=[['A','research_equity_FB','ETF']], etfs=['ETF'])
        self.base = dict(ticker='A', action='addition', effective_session=self.sessions[150],
                         published_at=self.sessions[140]+'T10:00:00Z', identity_verified=False, chain_verified=False)

    def run_rows(self, raw):
        e, _ = events_from_tables([('TEST', raw, 'ticker')])
        return join_grid(self.meta, self.sessions, e, (1,3)), e

    def test_exact_no_alias_and_etf(self):
        rows,e = self.run_rows([self.base, dict(self.base,ticker='FB'),dict(self.base,ticker='ETF')])
        self.assertEqual([len(r['matched_event_ids']) for r in rows],[1,0,0])

    def test_inclusive_edges_and_after_cutoff(self):
        a = dict(self.base,effective_session=self.sessions[26],published_at=self.sessions[26]+'T00:00:00Z')
        b = dict(self.base,effective_session=self.sessions[278],published_at=self.sessions[278]+'T00:00:00Z')
        c = dict(self.base,effective_session=self.sessions[279])
        rows,e = self.run_rows([a,b,c])
        self.assertEqual(len(rows[0]['matched_event_ids']),2)
        self.assertEqual(len(rows[0]['cutoff_day_publication_clock_pending']),1)

    def test_older_announcement_separate(self):
        rows,e = self.run_rows([dict(self.base,published_at=self.sessions[25]+'T10:00:00Z')])
        self.assertFalse(rows[0]['matched_event_ids'])
        self.assertEqual(len(rows[0]['publication_before_window_event_ids']),1)

    def test_future_announcement_excluded(self):
        rows,e = self.run_rows([dict(self.base,published_at=self.sessions[279]+'T10:00:00Z')])
        self.assertFalse(rows[0]['matched_event_ids'])

    def test_semantic_duplicate_dedup(self):
        rows,e = self.run_rows([self.base,self.base.copy()])
        self.assertEqual(len(e),1)
        self.assertEqual(len(e[0]['original_variants']),2)
        self.assertEqual(len(rows[0]['matched_event_ids']),1)

    def test_conflict_blocks_not_pick_best(self):
        rows,e = self.run_rows([self.base,dict(self.base,identity_verified=True)])
        self.assertFalse(rows[0]['matched_event_ids'])
        self.assertIn('CONFLICTING_SEMANTIC_EVENT_VARIANTS',e[0]['metadata_errors'])

    def test_missing_or_naive_timestamp_retained(self):
        rows,e = self.run_rows([dict(self.base,published_at=None),dict(self.base,published_at='2000-05-19T10:00:00')])
        self.assertEqual(len(e),2)
        self.assertFalse(rows[0]['matched_event_ids'])

    def test_false_source_flags_preserved(self):
        rows,e = self.run_rows([self.base])
        self.assertEqual(e[0]['original_variants'][0]['original_row'],self.base)
        self.assertFalse(rows[0]['source_admitted'])
        self.assertEqual(len(e[0]['source_blockers']),2)

    def test_day_offset_is_conservative(self):
        self.assertEqual(publication_day('2000-01-01T23:30:00-05:00'),'2000-01-02')

    def test_distinct_events_not_repeated_slots(self):
        rows,e = self.run_rows([self.base])
        rows.append(dict(rows[0],signal_index=1))
        summary = summarize(rows,e,0,2)
        self.assertEqual(summary['unique_events'],1)
        self.assertEqual(summary['months_with_event'],2)

    def test_insufficient_calendar_fails(self):
        with self.assertRaises(ValueError):
            join_grid(dict(self.meta,dates=[self.sessions[100]]),self.sessions,[],(1,3))

    def test_calendar_not_business_day_approximation(self):
        sparse = self.sessions[::2]
        with self.assertRaises(ValueError):
            join_grid(self.meta,sparse,[],(1,3))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-only',action='store_true')
    parser.add_argument('--plan-sha256')
    parser.add_argument('--write',action='store_true')
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    require(result.wasSuccessful(),'Synthetic tests failed')
    if args.test_only:
        return
    require(args.write and args.plan_sha256,'Explicit plan hash and --write required')
    require(not local_path(OUT).exists(),'Write-once result already exists')
    require(digest(local_path(PLAN))==args.plan_sha256,'Plan hash mismatch')
    plan = load(PLAN)
    pins = dict(plan['input_hashes'])
    verify(pins)
    meta, calendar = load(META),load(CAL)
    extra = calendar['provenance']['source_hashes']
    verify(extra)
    for name, expected in extra.items():
        pins[str(local_path(name).relative_to(ROOT))] = expected
    events, refs = events_from_tables([(index,load(path),key) for index,path,key in TABLES],True)
    rows = join_grid(meta,calendar['sessions'],events)
    require(len(rows)==5978 and sum(r['is_etf'] for r in rows)==2074,'Original full grid/ETF mismatch')
    summaries = {key:summarize(rows,events,*bounds) for key,bounds in SPLITS.items()}
    ever_candidates = {r['symbol'] for r in rows if not r['is_etf']}
    all_matched = {key for r in rows for key in r['matched_event_ids']}
    for event in events:
        event['matches_any_exact_non_etf_candidate'] = event['ticker'] in ever_candidates
        event['matched_any_formation_window'] = event['event_id'] in all_matched
        event['unmatched_reasons'] = (['INVALID_OR_CONFLICTING_METADATA'] if event['metadata_errors'] else
            ['NO_EXACT_NON_ETF_CANDIDATE'] if event['ticker'] not in ever_candidates else
            ['NO_JOINT_PUBLICATION_AND_EFFECTIVE_WINDOW'] if event['event_id'] not in all_matched else [])
    for ref in refs.values():
        if ref.get('verified'):
            name, expected = ref['path'],ref['actual_sha256']
            require(name not in pins or pins[name]==expected,'Conflicting reference pins')
            pins[name] = expected
    pins[PLAN] = args.plan_sha256
    pins[SELF] = digest(local_path(SELF))
    verify(pins)
    blocked = any(e['source_blockers'] for e in events if e['event_id'] in all_matched)
    disposition = 'CLOSE_BOUNDED_LEAD_NO_COLLECTION_OR_STRATEGY_TRIAL' if blocked or not all_matched else 'NOT_ADMITTED_UNKNOWN_COVERAGE_REMAINS'
    output = dict(schema='after159_index_event_join_result_v1',created_utc=datetime.now(timezone.utc).isoformat(),
                  input_hashes=pins,plan_sha256=args.plan_sha256,implementation_sha256=pins[SELF],
                  tests=dict(passed=True,synthetic_tests=result.testsRun),
                  summary=summaries,rows=rows,events=events,source_references=refs,
                  unmatched_event_ids=[e['event_id'] for e in events if not e['matched_any_formation_window']],
                  unique_semantic_events=len(events),original_source_rows=len(refs),
                  invalid_or_conflicting_semantic_events=sum(bool(e['metadata_errors']) for e in events),
                  source_reference_failures=sum(not r['verified'] for r in refs.values()),
                  decision=disposition,source_admission_changed=False,activity=plan['activity'],
                  limitations=['Candidate metadata coverage only; original eligible arrays and prices not read.',
                               'Distinct ticker proxies are not verified unique issuers; no alias inference.',
                               'The two tables are not certified complete event histories; absence is unknown.',
                               'Reported publication dates and source hashes do not establish historical first availability.',
                               'False or absent identity/chain flags remain blockers; no source gate is waived.'])
    encoded = (json.dumps(output,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    fd = os.open(local_path(OUT),os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o644)
    with os.fdopen(fd,'wb') as handle:
        handle.write(encoded)
    print(json.dumps(dict(output=OUT,sha256=digest(local_path(OUT)),summary=summaries,decision=disposition),sort_keys=True))


if __name__ == '__main__':
    main()
