"""Bounded source-only bridge: freeze, select metadata, then compare raw cells.

No ratios, factors, ranks, prices, holdings or returns. The metadata phase never
uses a financial value to select an accession or context. Later accessions are
not substituted when the fixed earliest cached candidate fails comparison.
"""
import argparse
from datetime import date, datetime, timezone
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SELF = 'work/after158_basic_shares_gaap_bridge.py'
OUT = ROOT / 'work/after158_basic_shares_gaap_bridge_v1'
SOURCE_PLAN = 'work/after158_basic_shares_source_plan.json'
RAW = 'outputs/after158_basic_shares_extraction_v1/RAW_SHARES.json'
MANIFEST = 'outputs/after158_basic_shares_extraction_v1/MANIFEST.json'
CONCEPT = 'WeightedAverageNumberOfSharesOutstandingBasic'
CACHES = {
    'AAPL': (320193, 'outputs/dynamic_top100/registry_sec/0000320193_facts.json',
             'outputs/dynamic_top100/registry_sec/0000320193_facts.provenance.json'),
    'AMD': (2488, 'outputs/dynamic_top100/sec_coverage/0000002488.json', None),
    'MSFT': (789019, 'outputs/dynamic_top100/registry_sec/0000789019_facts.json',
             'outputs/dynamic_top100/registry_sec/0000789019_facts.provenance.json'),
    'NVDA': (1045810, 'outputs/dynamic_top100/sec_coverage/0001045810.json', None),
}
RULES = {
    'scope': 'All44 original events, original order. Exactly four fixed local companyfacts files; no downloads or substitute cache search.',
    'concept': 'Only us-gaap/WeightedAverageNumberOfSharesOutstandingBasic/units/shares. No diluted, DEI outstanding, custom tag or numeric proxy.',
    'current': 'Current context must match original fiscal_year, fp=FY, form=10-K, exact original period_end, and exact period_start when known. Its duration must be330..400inclusive days.',
    'accession': 'Select unique accession on the earliest filed date among current annual metadata candidates, independently of all values. This is earliest cached matching10-K, not proof of the original filing or archive completeness.',
    'prior': 'Require a unique annual context with the same accession, filed date, fy and fp and the exact original comparison end/start when known. No context from another accession.',
    'duplicates': 'Retain all candidate metadata. Multiple earliest accessions or multiple current/prior context rows are ambiguity, even if their values might agree. Do not value-deduplicate.',
    'comparison': 'After metadata output is frozen, compare the chosen current/prior literal shares independently with RAW_SHARES normalized decimal share counts. Never compute their ratio or change. Keep all mismatches and missing/ambiguous records.',
    'rounding': 'Separately record whether the selected companyfacts point value falls inside the closed original published rounding interval: original displayed decimal plus/minus half its last displayed place, multiplied by the original share unit scale. Preserve trailing decimal zero precision. This assumes rounding to nearest with unspecified tie convention; interval compatibility is not exact equality or proof of the latent count. No adjustment of original values or interval for companyfacts precision, and no accession substitution.',
    'no_fallback': 'A mismatch or missing prior in the earliest selected accession cannot trigger selection of a later accession.',
    'availability': 'Report filing date and max(original release date, filing date) as date-only corroboration anchors. No earlier availability, exact first-publication time or contemporaneous capture is certified.',
    'admission': 'No source, split, identity, period, strategy or model hold is lifted. Numeric equality cannot certify common split basis or full GAAP semantics alone.',
    'provenance': 'AAPL/MSFT carry existing origin/hash/retrieval sidecars. The two fixed sec_coverage files currently have no identified matching sidecars; this remains an explicit restriction.',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write('\n')


def verify(pins):
    for name, expected in pins.items():
        path = Path(name)
        require(not path.is_absolute() and '..' not in path.parts, 'Nonlocal input reference')
        path = ROOT / path
        require(path.resolve().is_relative_to(ROOT), 'Input path escape')
        require(sha(path) == expected, 'Input hash changed: ' + name)


def strict_date(value):
    parsed = date.fromisoformat(value)
    require(parsed.isoformat() == value, 'Noncanonical date')
    return parsed


def context_metadata(row, period, year):
    if row.get('form') != '10-K' or row.get('fp') != 'FY' or row.get('fy') != year:
        return None
    if row.get('end') != period['period_end']:
        return None
    try:
        start, end, filed = (strict_date(row[key]) for key in ('start', 'end', 'filed'))
    except (ValueError, TypeError, KeyError):
        return None
    if not 330 <= (end-start).days+1 <= 400 or filed < end:
        return None
    if period.get('period_start') is not None and row['start'] != period['period_start']:
        return None
    if not isinstance(row.get('accn'), str) or not row['accn']:
        return None
    return {key: row[key] for key in ('accn', 'filed', 'form', 'fp', 'fy', 'start', 'end')}


def choose_metadata(event, facts):
    result = dict(event_id=event['event_id'], symbol=event['symbol'], fiscal_year=event['fiscal_year'],
                  release_date=event['release_date'], periods=event['periods'], current_candidates=[],
                  prior_candidates=[], status=None, selected_accession=None, selected_filed=None,
                  original_filing_certified=False, source_admission_changed=False)
    concept = facts.get('facts', {}).get('us-gaap', {}).get(CONCEPT)
    if concept is None:
        return dict(result, status='missing_exact_basic_concept')
    values = concept.get('units', {}).get('shares')
    if values is None:
        return dict(result, status='missing_exact_shares_unit')
    require(isinstance(values, list), 'Malformed shares context array')
    prefix = '/facts/us-gaap/' + CONCEPT + '/units/shares/'
    for index, row in enumerate(values):
        metadata = context_metadata(row, event['periods']['current'], event['fiscal_year'])
        if metadata:
            result['current_candidates'].append(dict(metadata, json_pointer=prefix+str(index)))
    if not result['current_candidates']:
        return dict(result, status='no_current_original_annual_metadata_candidate')
    earliest = min(row['filed'] for row in result['current_candidates'])
    earliest_rows = [row for row in result['current_candidates'] if row['filed'] == earliest]
    accessions = sorted({row['accn'] for row in earliest_rows})
    if len(accessions) != 1:
        return dict(result, status='ambiguous_earliest_cached_accession')
    accession = accessions[0]
    result.update(selected_accession=accession, selected_filed=earliest)
    current = [row for row in result['current_candidates'] if row['accn'] == accession and row['filed'] == earliest]
    if len(current) != 1:
        return dict(result, status='ambiguous_current_context_in_earliest_accession')
    result['current_selected'] = current[0]
    for index, row in enumerate(values):
        if row.get('accn') != accession or row.get('filed') != earliest:
            continue
        metadata = context_metadata(row, event['periods']['comparison'], event['fiscal_year'])
        if metadata:
            result['prior_candidates'].append(dict(metadata, json_pointer=prefix+str(index)))
    if not result['prior_candidates']:
        return dict(result, status='missing_prior_context_in_earliest_accession')
    if len(result['prior_candidates']) != 1:
        return dict(result, status='ambiguous_prior_context_in_earliest_accession')
    result['prior_selected'] = result['prior_candidates'][0]
    return dict(result, status='unique_same_accession_annual_pair')


def summarize(records, key):
    return {value: sum(row[key] == value for row in records) for value in sorted({row[key] for row in records})}


def freeze():
    source = load(ROOT / SOURCE_PLAN)
    require(len(source['events']) == 44, 'Fixed44 source events required')
    test_path = OUT/'TESTS.json'
    tests = load(test_path)
    require(tests['passed'] and tests['implementation_sha256'] == sha(ROOT/SELF),
            'Passing synthetic tests for exact implementation required')
    paths = [SELF, str(test_path.relative_to(ROOT)), SOURCE_PLAN, RAW, MANIFEST, 'work/after158_basic_shares_extraction_contract.json',
             'work/after158_basic_shares_independent_manual.json', 'AGENTS.md', 'LOCAL_SETUP.md',
             'run_local.sh', 'local_run.py', 'local.sandbox.sb']
    events = []
    for event in source['events']:
        events.append({key: event[key] for key in ('event_id', 'symbol', 'fiscal_year', 'release_date',
                                                   'available_from', 'periods', 'original_index',
                                                   'inherited_rejection_codes', 'identity', 'source')})
        require(sha(ROOT/event['source']['path']) == event['source']['sha256'], 'Original raw source changed')
        require(sha(ROOT/event['source']['receipt']['path']) == event['source']['receipt']['sha256'],
                'Original raw receipt changed')
        paths.extend([event['source']['path'], event['source']['receipt']['path']])
    require(len({event['event_id'] for event in events}) == 44, 'Duplicate original event ID')
    for cik, path, provenance in CACHES.values():
        paths.append(path)
        if provenance:
            paths.append(provenance)
    manifest = load(ROOT/MANIFEST)
    require(manifest['outputs'][RAW] == sha(ROOT/RAW), 'Extraction output binding')
    pins = {path: sha(ROOT/path) for path in sorted(set(paths))}
    verify(pins)
    plan = dict(schema='after158_basic_shares_gaap_bridge_plan_v1', created_utc=datetime.now(timezone.utc).isoformat(),
                rules=RULES, caches={symbol: dict(cik=cik, path=path, provenance=provenance)
                                     for symbol,(cik,path,provenance) in CACHES.items()},
                events=events, input_hashes=pins, source_admission_changed=False,
                activity=dict(financial_values_compared=0, factors=0, ranks=0, orders=0, returns=0))
    publish(OUT/'PLAN.json', plan)
    print(json.dumps(dict(plan_sha256=sha(OUT/'PLAN.json'), events=44, local_inputs=len(pins))))


def verified_plan(plan_sha):
    require(sha(OUT/'PLAN.json') == plan_sha, 'Exact source PLAN required')
    plan = load(OUT/'PLAN.json')
    require(plan['rules'] == RULES, 'Source selection rules changed')
    verify(plan['input_hashes'])
    return plan


def read_caches(plan):
    caches = {}
    for symbol, cache in plan['caches'].items():
        facts = json.loads((ROOT/cache['path']).read_text(), parse_float=Decimal)
        require(facts.get('cik') == cache['cik'], 'Companyfacts CIK mismatch')
        if cache['provenance']:
            provenance = load(ROOT/cache['provenance'])
            require(provenance['sha256'] == sha(ROOT/cache['path']), 'Companyfacts origin sidecar hash')
            require(provenance['source'] == 'https://data.sec.gov/api/xbrl/companyfacts/CIK%010d.json' % cache['cik'],
                    'Companyfacts origin URL mismatch')
        caches[symbol] = facts
    return caches


def metadata(plan_sha):
    plan = verified_plan(plan_sha)
    facts = read_caches(plan)
    records = [choose_metadata(event, facts[event['symbol']]) for event in plan['events']]
    result = dict(schema='after158_basic_shares_gaap_bridge_metadata_v1',
                  created_utc=datetime.now(timezone.utc).isoformat(), plan_sha256=plan_sha,
                  records=records, summary=summarize(records, 'status'),
                  by_symbol={symbol:summarize([row for row in records if row['symbol']==symbol], 'status')
                             for symbol in CACHES},
                  values_used_for_selection=False, financial_values_compared=0,
                  source_admission_changed=False, original_filing_certified=False)
    verify(plan['input_hashes'])
    publish(OUT/'METADATA.json', result)
    print(json.dumps(dict(metadata_sha256=sha(OUT/'METADATA.json'), summary=result['summary'], by_symbol=result['by_symbol'])))


def at_pointer(document, pointer):
    value = document
    for part in pointer.split('/')[1:]:
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def compare_cell(raw_cell, sec_row):
    """Compare literal points and original display precision, never alter a count."""
    literal = raw_cell['displayed_value']
    require(re.fullmatch(r'(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?', literal) is not None,
            'Invalid original displayed decimal')
    displayed = Decimal(literal.replace(',', ''))
    source_value = Decimal(raw_cell['value_in_shares'])
    decimal_value = Decimal(raw_cell['value_decimal'])
    sec_value = Decimal(str(sec_row['val']))
    scale = Decimal(str(raw_cell['unit']['scale']))
    require(all(v.is_finite() for v in (displayed, source_value, decimal_value, sec_value, scale)),
            'Nonfinite source share count or scale')
    require(displayed >= 0 and sec_value >= 0 and scale in (1000, 1000000, 1000000000),
            'Invalid source share count or scale')
    with localcontext() as context:
        context.prec = max(60, len(displayed.as_tuple().digits)+30,
                           len(sec_value.as_tuple().digits)+30)
        require(displayed == decimal_value and displayed*scale == source_value,
                'Raw original display normalization mismatch')
        last_place = Decimal(1).scaleb(displayed.as_tuple().exponent)
        half_width = last_place*scale/2
        lower, upper = source_value-half_width, source_value+half_width
        return dict(equal=source_value == sec_value,
                    original_value_in_shares=str(source_value),
                    companyfacts_value_in_shares=str(sec_value),
                    original_displayed_value=literal, original_share_scale=str(scale),
                    published_last_place_in_shares=str(last_place*scale),
                    native_rounding_interval_lower_inclusive=str(lower),
                    native_rounding_interval_upper_inclusive=str(upper),
                    native_rounding_interval_compatible=lower <= sec_value <= upper)


def compare(plan_sha, metadata_sha):
    plan = verified_plan(plan_sha)
    require(sha(OUT/'METADATA.json') == metadata_sha, 'Exact pre-value METADATA required')
    meta = load(OUT/'METADATA.json')
    require(meta['plan_sha256'] == plan_sha and meta['values_used_for_selection'] is False,
            'Metadata selection PLAN binding')
    facts = read_caches(plan)
    raw = load(ROOT/RAW)
    require(len(raw['events']) == 44 and [e['event_id'] for e in raw['events']] ==
            [e['event_id'] for e in plan['events']], 'Raw44 event axis changed')
    require(meta['records'] == [choose_metadata(event, facts[event['symbol']]) for event in plan['events']],
            'Frozen metadata selection cannot be changed after seeing values')
    records = []
    for selected, event, original in zip(meta['records'], raw['events'], plan['events']):
        record = dict(event_id=event['event_id'], symbol=event['symbol'], metadata_status=selected['status'],
                      selected_accession=selected['selected_accession'], selected_filed=selected['selected_filed'],
                      comparison_status='not_compared_metadata_unavailable', cells={},
                      inherited_rejection_codes=original['inherited_rejection_codes'],
                      cache_origin_sidecar_available=bool(plan['caches'][event['symbol']]['provenance']),
                      source_admission_changed=False, holds_released=False, original_filing_certified=False)
        if selected['status'] == 'unique_same_accession_annual_pair':
            matches = []
            interval_matches = []
            for raw_side, selected_side in (('current', 'current_selected'), ('comparison', 'prior_selected')):
                raw_cell = event['cells'][raw_side]
                require(raw_cell is not None and raw_cell['source_sha256'] == original['source']['sha256'],
                        'Raw cell original source binding')
                require(raw_cell['period_end'] == selected[selected_side]['end'] ==
                        original['periods'][raw_side]['period_end'], 'Raw cell period binding')
                sec_row = at_pointer(facts[event['symbol']], selected[selected_side]['json_pointer'])
                compared = compare_cell(raw_cell, sec_row)
                matches.append(compared['equal'])
                interval_matches.append(compared['native_rounding_interval_compatible'])
                record['cells'][raw_side] = dict(compared, json_pointer=selected[selected_side]['json_pointer'],
                    source_period_end=raw_cell['period_end'], original_source_sha256=raw_cell['source_sha256'])
            record['comparison_status'] = ('both_cells_equal' if all(matches) else
                'both_cells_mismatch' if not any(matches) else 'current_cell_mismatch' if not matches[0] else 'prior_cell_mismatch')
            record['rounding_status'] = ('both_cells_interval_compatible' if all(interval_matches) else
                                        'one_or_both_cells_outside_original_interval')
            record['corroboration_filing_date'] = selected['selected_filed']
            record['later_of_release_and_filing_date'] = max(event['release_date'], selected['selected_filed'])
        else:
            record['rounding_status'] = 'not_compared_metadata_unavailable'
        records.append(record)
    verify(plan['input_hashes'])
    require(sha(OUT/'METADATA.json') == metadata_sha, 'Metadata bytes changed during comparison')
    result = dict(schema='after158_basic_shares_gaap_bridge_comparison_v1',
                  created_utc=datetime.now(timezone.utc).isoformat(), plan_sha256=plan_sha,
                  metadata_sha256=metadata_sha, records=records,
                  summary=summarize(records, 'comparison_status'),
                  rounding_summary=summarize(records, 'rounding_status'),
                  by_symbol={symbol:summarize([row for row in records if row['symbol']==symbol], 'comparison_status')
                             for symbol in CACHES},
                  later_accession_fallbacks=0, source_admission_changed=False, holds_released=False,
                  activity=dict(ratios=0, factors=0, ranks=0, orders=0, returns=0, downloads=0),
                  limitations=[RULES['accession'], RULES['admission'], RULES['availability'], RULES['provenance'], RULES['rounding']])
    publish(OUT/'COMPARISON.json', result)
    print(json.dumps(dict(comparison_sha256=sha(OUT/'COMPARISON.json'), summary=result['summary'],
                         rounding_summary=result['rounding_summary'], by_symbol=result['by_symbol'])))


class Tests(unittest.TestCase):
    def event(self):
        return dict(event_id='X:FY2020', symbol='X', fiscal_year=2020, release_date='2021-01-20',
                    periods=dict(current=dict(period_start='2020-01-01',period_end='2020-12-31'),
                                 comparison=dict(period_start='2019-01-01',period_end='2019-12-31')))

    def row(self, **updates):
        row = dict(form='10-K', fp='FY', fy=2020, accn='first', filed='2021-02-01',
                   start='2020-01-01', end='2020-12-31', val=1)
        row.update(updates)
        return row

    def facts(self, rows):
        return dict(facts={'us-gaap':{CONCEPT:dict(units={'shares':rows})}})

    def test_pair(self):
        result=choose_metadata(self.event(),self.facts([self.row(),self.row(start='2019-01-01',end='2019-12-31')]))
        self.assertEqual(result['status'],'unique_same_accession_annual_pair')

    def test_no_value_selection_or_later_fallback(self):
        rows=[self.row(val=-999),self.row(accn='later',filed='2021-03-01',val=1),
              self.row(accn='later',filed='2021-03-01',start='2019-01-01',end='2019-12-31',val=1)]
        result=choose_metadata(self.event(),self.facts(rows))
        self.assertEqual((result['selected_accession'],result['status']),('first','missing_prior_context_in_earliest_accession'))

    def test_ambiguous_earliest(self):
        result=choose_metadata(self.event(),self.facts([self.row(),self.row(accn='other')]))
        self.assertEqual(result['status'],'ambiguous_earliest_cached_accession')

    def test_duplicate_metadata_not_value_deduplicated(self):
        result=choose_metadata(self.event(),self.facts([self.row(val=1),self.row(val=1)]))
        self.assertEqual(result['status'],'ambiguous_current_context_in_earliest_accession')

    def test_interim_and_amendments_excluded(self):
        result=choose_metadata(self.event(),self.facts([self.row(start='2020-10-01'),self.row(form='10-K/A')]))
        self.assertEqual(result['status'],'no_current_original_annual_metadata_candidate')

    def test_unknown_source_start_does_not_waive_annual_length(self):
        event=self.event();event['periods']['current']['period_start']=None
        result=choose_metadata(event,self.facts([self.row(start='2020-10-01')]))
        self.assertEqual(result['status'],'no_current_original_annual_metadata_candidate')

    def raw_cell(self, literal='1.2'):
        value=Decimal(literal.replace(',', ''))
        return dict(displayed_value=literal, value_decimal=str(value),
                    value_in_shares=str(value*1000), unit=dict(scale=1000))

    def test_interval_does_not_turn_mismatch_into_exact(self):
        result=compare_cell(self.raw_cell(), dict(val=1249))
        self.assertFalse(result['equal'])
        self.assertTrue(result['native_rounding_interval_compatible'])
        self.assertEqual(result['original_value_in_shares'], '1200.0')

    def test_native_trailing_zero_changes_interval(self):
        result=compare_cell(self.raw_cell('1.20'), dict(val=1249))
        self.assertFalse(result['native_rounding_interval_compatible'])
        self.assertEqual(result['published_last_place_in_shares'], '10.00')

    def test_interval_boundary_inclusive(self):
        self.assertTrue(compare_cell(self.raw_cell(), dict(val=1250))['native_rounding_interval_compatible'])
        self.assertFalse(compare_cell(self.raw_cell(), dict(val=Decimal('1250.0001')))['native_rounding_interval_compatible'])

    def test_original_normalization_mismatch_fails(self):
        raw=self.raw_cell();raw['value_in_shares']='1201'
        with self.assertRaisesRegex(ValueError, 'normalization mismatch'):
            compare_cell(raw, dict(val=1200))

    def test_exact_equality_and_comma_literal(self):
        result=compare_cell(self.raw_cell('1,234'), dict(val=1234000))
        self.assertTrue(result['equal'])
        self.assertTrue(result['native_rounding_interval_compatible'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['test','freeze','metadata','compare'])
    parser.add_argument('--plan-sha');parser.add_argument('--metadata-sha')
    args=parser.parse_args()
    if args.phase=='test':
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
        publish(OUT/'TESTS.json', dict(schema='after158_basic_shares_gaap_bridge_tests_v1',
                created_utc=datetime.now(timezone.utc).isoformat(), implementation_sha256=sha(ROOT/SELF),
                tests_run=result.testsRun, passed=result.wasSuccessful(),
                failures=[str(test) for test, _ in result.failures], errors=[str(test) for test, _ in result.errors],
                python=sys.version, activity=dict(real_financial_values_read=0, factors=0, returns=0)))
        raise SystemExit(0 if result.wasSuccessful() else 1)
    if args.phase=='freeze':freeze()
    elif args.phase=='metadata':metadata(args.plan_sha)
    else:compare(args.plan_sha,args.metadata_sha)


if __name__=='__main__':main()
