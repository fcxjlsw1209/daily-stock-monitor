"""Source-only BA annual-chain availability under explicit two-clock assumptions.

No prices, financial ratios, ranks, portfolio decisions or returns are read.
The result is not strategy admission or a claim of contemporaneous source bytes.
"""
import argparse
from bisect import bisect_left, bisect_right
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
SELF = 'work/after158_ba_availability.py'
OUT = 'outputs/after158_ba_availability_v1'
EXTERNAL_EVIDENCE_ROOT = Path('/Users/shiweili/.cache/codex-runtimes/codex-primary-runtime/dependencies')
CAL = 'outputs/monthly_paired_target143/CALENDAR.json'
CAL_CODE = 'outputs/monthly_earnings_repair/research_calendar.py'
META = 'outputs/monthly_local_linear70/META.json'
ROWS = 'outputs/monthly_revenue_pilot_source156/ROWS.json'
REVIEW = 'work/after158_ba_independent_review.json'
BRIDGE = 'work/after158_ba_currency_bridge.json'
PUBLICATION = 'outputs/after158_ba_publication_v1/METADATA.json'
PINS = {
    CAL: '8ee4da69fba84db1edee37a7e0f5af5d3ab16c6257e5d37cddbd531f01404622',
    CAL_CODE: '112249b503050b54f62152aa5e15bcde066d4ee30735687c79f1a06a1c61474a',
    META: '3b46968f486e88fa170ac01f120dd5ef74ebd5a56243c6991c5a4fea506076f4',
    ROWS: '1732c3611fdbd14b00ef39d370a95493aec91f1a2c3ee60a2ce2ec3e27080196',
    REVIEW: 'a26da326feafc899773e8c17897b7d6fee91c17f68614d720ab0cff988f84b74',
    BRIDGE: '4e9607d7e1dcd2e2b200746ffd90d20f9e1f9e9deb2d818dffecbfab09b69289',
    PUBLICATION: '77f516d24be889361dd208c8c2e868ab48082d9186974f6933af2a6003732615',
    'outputs/after158_ba_publication_v1/REVIEW.json': 'c89347be3c7878c2bc069faf7262159809f8cc944960ed20125fd87edf0eb957',
}
ACCESSIONS = {
    2014: '0000012927-15-000011', 2015: '0000012927-16-000099',
    2016: '0000012927-17-000006', 2017: '0000012927-18-000007',
    2018: '0000012927-19-000010',
}
CONTRACT = {
    'scope': 'Original early53 BA candidate rows only; full5978-row axis checked and unchanged.',
    'known_clock': 'Close of second XNYS session strictly after original release dateline.',
    'usable_clock': 'Close of second XNYS session strictly after max(release dateline, pinned original annual USD corroboration filing date).',
    'information_cutoff': 'Actual XNYS close at original signal session minus42 exchange sessions.',
    'selection': 'Select latest known annual event BEFORE validity. An invalid or not-yet-usable latest never falls back.',
    'chain': 'Four consecutive annual originals FY2014..2017 with three independently reviewed original-basis edges.',
    'boundary': 'FY2018 is a known invalid latest because of ASC606/715 basis change; never a substitute original.',
    'vintage': 'Currently retrieved original-dated sources; date-only reconstruction assumptions, not exact first-public timestamps or historical bytes.',
    'admission': 'Feasibility overlay only. Original source flags and strategy eligibility remain unchanged.',
}


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def publish(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)+'\n').encode()
    with path.open('xb') as f:
        f.write(data)


def verify(pins):
    for rel, digest in pins.items():
        path = ROOT / rel
        require(not Path(rel).is_absolute() and '..' not in Path(rel).parts, 'Nonlocal reference')
        require(path.resolve().is_relative_to(ROOT), 'Path escape')
        require(sha(path) == digest, 'Hash mismatch: '+rel)


def partition_reference(name):
    path = Path(name)
    if not path.is_absolute():
        require('..' not in path.parts, 'Relative path escape')
        return 'local',name
    if path.is_relative_to(ROOT):
        return 'local',str(path.relative_to(ROOT))
    require(path.is_relative_to(EXTERNAL_EVIDENCE_ROOT) and
            path.resolve().is_relative_to(EXTERNAL_EVIDENCE_ROOT.resolve()),
            'Unexpected external evidence path')
    return 'external',name


def verify_external(pins):
    """Verify prior PDF-review tool evidence; never import or execute it."""
    for name,digest in pins.items():
        require(partition_reference(name)[0] == 'external', 'Expected review-tool evidence')
        require(sha(Path(name)) == digest, 'External evidence hash mismatch')


def merge_references(local, external, references):
    for name, digest in references.items():
        kind, path = partition_reference(name)
        dest = local if kind == 'local' else external
        require(path not in dest or dest[path] == digest, 'Conflicting source hash: '+path)
        dest[path] = digest


def nth_later(sessions, day, n=2):
    require(n > 0 and sessions == sorted(set(sessions)), 'Invalid session axis')
    i = bisect_right(sessions, day) + n - 1
    require(i < len(sessions), 'Insufficient future calendar')
    return sessions[i]


def select_chain(events, cutoff):
    """Both clocks are timezone-aware close timestamps, compared as instants."""
    require(cutoff.tzinfo is not None, 'Naive cutoff')
    require(len({e['year'] for e in events}) == len(events), 'Duplicate annual original')
    known = [e for e in events if datetime.fromisoformat(e['known_at']) <= cutoff]
    if not known:
        return dict(latest_year=None, chain_years=[], feasible=False, reason='no_known_original')
    latest = max(known, key=lambda e: e['year'])
    result = dict(latest_year=latest['year'], chain_years=[], feasible=False)
    if not latest['basis_valid']:
        return dict(result, reason='latest_original_basis_break_no_fallback')
    selected = [next((e for e in known if e['year'] == year), None)
                for year in range(latest['year']-3, latest['year']+1)]
    if any(e is None for e in selected):
        return dict(result, reason='four_originals_not_yet_known')
    result['chain_years'] = [e['year'] for e in selected]
    if any(not e['basis_valid'] for e in selected):
        return dict(result, reason='chain_basis_break')
    if any(datetime.fromisoformat(e['usable_at']) > cutoff for e in selected):
        return dict(result, reason='unit_corroboration_not_yet_usable')
    return dict(result, feasible=True, reason=None)


def collect_inputs():
    verify(PINS)
    pins = dict(PINS, **{SELF: sha(ROOT/SELF)})
    external = {}
    for source in (REVIEW, BRIDGE, 'outputs/after158_ba_publication_v1/REVIEW.json'):
        merge_references(pins, external, load(ROOT/source)['input_hashes'])
    # Exact close timestamps depend on the exchange-calendar implementation,
    # not just the serialized list of session dates.
    merge_references(pins, external, load(ROOT/CAL)['provenance']['source_hashes'])
    metadata = load(ROOT/PUBLICATION)
    prefix = str(Path(PUBLICATION).parent)
    pins[prefix+'/COLLECTION.json'] = metadata['collection_sha256']
    pins['work/after158_ba_publication_collect.py'] = metadata['helper_sha256']
    for row in metadata['records']:
        path = prefix+'/'+row['path']
        pins[path] = row['sha256']
        parsed = load(ROOT/path)
        pins[prefix+'/'+row['target_id']+'.html'] = parsed['raw_sha256']
        pins[prefix+'/'+row['target_id']+'.receipt.json'] = parsed['receipt_sha256']
    verify(pins)
    verify_external(external)
    return pins,external


def run(plan_sha):
    plan_path = ROOT/OUT/'PLAN.json'
    require(sha(plan_path) == plan_sha, 'Exact source plan required')
    plan = load(plan_path)
    require(plan['contract'] == CONTRACT and plan['accessions'] == {str(k):v for k,v in ACCESSIONS.items()}, 'Contract changed')
    verify(plan['input_hashes'])
    verify_external(plan['source_review_external_hashes'])
    spec = importlib.util.spec_from_file_location('ba_source_calendar', ROOT/CAL_CODE)
    cal_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cal_module)
    cal = load(ROOT/CAL)
    schedule = cal_module.market_calendar(cal['start'], cal['end']).schedule
    sessions = [str(d.date()) for d in schedule.index]
    require(sessions == cal['sessions'], 'Calendar mismatch')
    closes = {str(d.date()): datetime.fromisoformat(v.isoformat()) for d,v in schedule['market_close'].items()}
    independent = load(ROOT/REVIEW)
    require(independent['passed'] is True and independent['original_basis_three_edges_supported'] is True,
            'Original-basis independent review required')
    bridge = load(ROOT/BRIDGE)
    require(bridge['passed'] is True and bridge['five_original_cells_match_revenues_usd'] is True,
            'Original-cell currency corroboration required')
    original = {r['fiscal_year']: r for r in independent['reports']}
    events = []
    for report in bridge['reports']:
        year = report['fiscal_year']; release = original[year]['original_publication_dateline']
        require(release == report['original_publication_dateline'], 'Dateline disagreement')
        candidates = report['concept_results']['Revenues']['exact_annual_contexts']
        matches = [r for r in candidates if r['source_row']['accn'] == ACCESSIONS[year]
                   and r['comparison']['currency_corroboration_matches']]
        require(len(matches) == 1, 'Exact original annual USD context required')
        fact = matches[0]['source_row']
        require(matches[0]['unit'] == 'USD' and fact['form'] == '10-K' and
                fact['fy'] == year and fact['fp'] == 'FY' and
                fact['start'] == f'{year}-01-01' and fact['end'] == f'{year}-12-31', 'Wrong original annual context')
        parsed = load(ROOT/str(Path(PUBLICATION).parent)/f'FY{year}.parsed.json')
        stamp = datetime.strptime(release, '%Y-%m-%d')
        literal = r'CHICAGO\s*,\s*Jan(?:uary)?\.?\s+'+str(stamp.day)+r',\s*'+str(stamp.year)
        require(parsed['expected_dateline'] == release and re.search(literal, parsed['visible_text']), 'Official HTML dateline mismatch')
        events.append(dict(year=year, original_release_date=release, original_usd_filing_date=fact['filed'],
            original_usd_accession=ACCESSIONS[year], original_usd_json_pointer=matches[0]['json_pointer'],
            known_at=closes[nth_later(sessions,release)].isoformat(),
            usable_at=closes[nth_later(sessions,max(release,fact['filed']))].isoformat(), basis_valid=year<2018))
    meta, rows = load(ROOT/META), load(ROOT/ROWS)
    require(len(rows) == 5978 and len(meta['dates']) == 122, 'Original axis shape')
    output = []
    for k, row in enumerate(rows):
        i,j = divmod(k,49)
        require((row['signal_index'],row['candidate_index'],row['symbol'],row['decision_date']) ==
                (i,j,meta['symbols'][i][j],meta['dates'][i]), 'Original row mismatch')
        require(datetime.fromisoformat(row['decision_cutoff']) == closes[row['decision_date']], 'Decision clock mismatch')
        if i>=53 or row['symbol'] != 'BA':
            continue
        t = bisect_left(sessions,row['decision_date'])
        require(t>=42 and sessions[t]==row['decision_date'], 'No exact t-42 session')
        cutoff = closes[sessions[t-42]]
        output.append(dict(signal_index=i,candidate_index=j,decision_date=row['decision_date'],
            information_cutoff=cutoff.isoformat(),**select_chain(events,cutoff)))
    require(len(output) == 21, 'Unexpected early BA scope')
    verify(plan['input_hashes'])
    verify_external(plan['source_review_external_hashes'])
    summary = dict(total_early_ba_rows=len(output), feasible_rows=sum(r['feasible'] for r in output),
                   reasons={reason:sum(r['reason']==reason for r in output)
                            for reason in sorted({r['reason'] for r in output if r['reason']})})
    result = dict(schema='after158_ba_two_clock_source_feasibility_v1', plan_sha256=plan_sha,
        created_utc=datetime.now(timezone.utc).isoformat(), contract=CONTRACT, events=events, rows=output,
        summary=summary, full_original_axis_verified=5978, source_admission_changed=False,
        strategy_admitted=False, financial_factors_computed=False, new_strategy_trials=0,
        observer_changes=False, limitations=['Declared date-only assumptions; not a certified historical capture.',
            'Original-basis edge review and USD bridge apply only to the five named sources.',
            'This is source feasibility, not a new portfolio or evidence of better returns.'])
    publish(ROOT/OUT/'FEASIBILITY.json',result)
    print(json.dumps(summary))


class Tests(unittest.TestCase):
    def event(self,year,known,usable=None,valid=True):
        return dict(year=year,known_at=known,usable_at=usable or known,basis_valid=valid)

    def test_strict_second_calendar_session(self):
        self.assertEqual(nth_later(['2020-01-02','2020-01-03','2020-01-06'],'2020-01-02'),'2020-01-06')

    def test_weekend(self):
        self.assertEqual(nth_later(['2020-01-06','2020-01-07'],'2020-01-05'),'2020-01-07')

    def test_missing_calendar_fails(self):
        with self.assertRaises(ValueError):nth_later(['2020-01-06'],'2020-01-05')

    def test_invalid_latest_never_falls_back(self):
        es=[self.event(y,'2018-01-01T21:00:00+00:00') for y in range(2014,2018)]
        es.append(self.event(2018,'2019-02-01T21:00:00+00:00','2019-02-12T21:00:00+00:00',False))
        r=select_chain(es,datetime.fromisoformat('2019-02-05T21:00:00+00:00'))
        self.assertEqual((r['latest_year'],r['reason']),(2018,'latest_original_basis_break_no_fallback'))

    def test_later_currency_evidence_not_early(self):
        es=[self.event(y,'2018-01-01T21:00:00+00:00') for y in range(2014,2017)]
        es.append(self.event(2017,'2018-02-02T21:00:00+00:00','2018-02-14T21:00:00+00:00'))
        r=select_chain(es,datetime.fromisoformat('2018-02-13T21:00:00+00:00'))
        self.assertEqual(r['reason'],'unit_corroboration_not_yet_usable')
        self.assertTrue(select_chain(es,datetime.fromisoformat('2018-02-14T21:00:00+00:00'))['feasible'])

    def test_four_consecutive_originals(self):
        es=[self.event(y,'2018-01-01T21:00:00+00:00') for y in [2013,2015,2016,2017]]
        self.assertEqual(select_chain(es,datetime.fromisoformat('2018-02-01T21:00:00+00:00'))['reason'],'four_originals_not_yet_known')

    def test_future_event_does_not_contaminate(self):
        es=[self.event(y,'2018-01-01T21:00:00+00:00') for y in range(2014,2018)]
        t=datetime.fromisoformat('2018-02-01T21:00:00+00:00')
        expected=select_chain(es,t)
        es.append(self.event(2018,'2019-02-01T21:00:00+00:00',valid=False))
        self.assertEqual(select_chain(es,t),expected)

    def test_duplicate_originals_rejected(self):
        with self.assertRaises(ValueError):
            select_chain([self.event(2017,'2018-01-01T21:00:00+00:00')]*2,datetime.now(timezone.utc))

    def test_review_tools_separate_from_local_inputs(self):
        self.assertEqual(partition_reference(str(ROOT/'work/x.py')),('local','work/x.py'))
        name=str(EXTERNAL_EVIDENCE_ROOT/'bin/override/pdftoppm')
        self.assertEqual(partition_reference(name),('external',name))
        for name in ('../x','/Users/shiweili/Library/CloudStorage/x','/tmp/x'):
            with self.assertRaises(ValueError):partition_reference(name)

    def test_merge_normalizes_calendar_dependency_and_rejects_conflict(self):
        local, external = {'work/x.py': 'same'}, {}
        merge_references(local, external, {str(ROOT/'work/x.py'): 'same'})
        self.assertEqual(local, {'work/x.py': 'same'})
        self.assertEqual(external, {})
        with self.assertRaises(ValueError):
            merge_references(local, external, {str(ROOT/'work/x.py'): 'changed'})

    def test_calendar_closure_has_required_local_runtime_and_source_members(self):
        # Metadata-only: protects the actual closure used to rebuild closes.
        local, external = {}, {}
        merge_references(local, external, load(ROOT/CAL)['provenance']['source_hashes'])
        self.assertEqual(len(local), 9)
        self.assertEqual(external, {})
        self.assertIn(CAL_CODE, local)
        self.assertIn('outputs/index_union/CALENDAR_EXCEPTIONS.json', local)
        self.assertIn('work/venv/lib/python3.9/site-packages/exchange_calendars/exchange_calendar_xnys.py', local)
        verify(local)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['test','freeze','run']);p.add_argument('--plan-sha')
    args=p.parse_args()
    if args.phase=='test':
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
        raise SystemExit(0 if result.wasSuccessful() else 1)
    if args.phase=='freeze':
        pins,external = collect_inputs()
        plan=dict(schema='after158_ba_two_clock_source_plan_v1',created_utc=datetime.now(timezone.utc).isoformat(),
                  contract=CONTRACT,accessions=ACCESSIONS,input_hashes=pins,
                  source_review_external_hashes=external,
                  external_evidence_role='Hash verification of prior PDF-review toolchain only; never imported or executed.',
                  strategy_trials=0)
        path=ROOT/OUT/'PLAN.json';publish(path,plan);print(sha(path))
    else:
        run(args.plan_sha)


if __name__=='__main__':main()
