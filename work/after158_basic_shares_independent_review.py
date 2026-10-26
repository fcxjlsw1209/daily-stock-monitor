"""Independent closure/metadata and manually reviewed source-cell checks.

No producer imports, share-change computation, factor, price or return access.
The numerical audit is explicitly limited to the 17 preselected source pairs.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
SELF = 'work/after158_basic_shares_independent_review.py'
PLAN = 'work/after158_basic_shares_source_plan.json'
INVENTORY = 'outputs/after158_basic_shares_inventory_v1/INVENTORY.json'
INSPECTION = 'work/after158_basic_shares_independent/INSPECTION.json'
MANUAL = 'work/after158_basic_shares_independent_manual.json'
PRODUCER = 'outputs/after158_basic_shares_extraction_v1/'
RESULT = 'work/after158_basic_shares_independent_review.json'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def path(name):
    p = Path(name)
    require('..' not in p.parts, 'Parent traversal forbidden')
    p = p if p.is_absolute() else ROOT / p
    require('CloudStorage' not in p.parts and not any('OneDrive' in x for x in p.parts), 'Cloud path forbidden')
    actual = p.resolve()
    allowed = (ROOT, Path('/Users/shiweili/.cache/codex-runtimes/codex-primary-runtime/dependencies'))
    require(any(actual.is_relative_to(base) for base in allowed), 'Unexpected external dependency: ' + name)
    return p


def load(name, expected=None):
    raw = path(name).read_bytes()
    require(expected is None or sha(raw) == expected, 'Hash mismatch: ' + name)
    return json.loads(raw)


def verify(pins):
    for name, expected in pins.items():
        require(re.fullmatch('[0-9a-f]{64}', expected) is not None, 'Malformed hash')
        require(sha(path(name).read_bytes()) == expected, 'Hash mismatch: ' + name)


def audit(events, originals, inventory, oracle):
    ids = [x['event_id'] for x in events]
    require(ids == [x['event_id'] for x in originals] == [x['event_id'] for x in inventory], 'Original order changed')
    require(len(ids) == len(set(ids)), 'Duplicate event')
    original_keys = ('original_index', 'event_id', 'symbol', 'fiscal_year', 'source',
                     'release_date', 'available_from', 'identity', 'periods')
    pairs, explicit, implicit = 0, 0, 0
    by_id = dict(zip(ids, events))
    for row, old, old_inventory in zip(events, originals, inventory):
        for key in original_keys:
            require(row[key] == old[key], 'Original metadata changed: ' + key)
        require(row['inherited_holds'] == old_inventory['inherited_holds'], 'Inherited holds changed')
        for key in ('source_admission_changed', 'fact_admitted', 'model_admitted', 'strategy_admitted'):
            require(row[key] is False, 'Admission promoted: ' + key)
        require(row['split_evidence']['automatic_adjustment_applied'] is False, 'Automatic split adjustment')
        require(row['split_evidence']['complete_split_history_certified'] is False, 'Split history overclaim')
        has_pair = all(row['cells'][side] is not None for side in ('current', 'comparison'))
        require(row['raw_annual_pair_extracted'] is has_pair, 'Pair status mismatch')
        require(has_pair == (len(row['failures']) == 0), 'Failure status mismatch')
        if not has_pair:
            require(all(v is None for v in row['cells'].values()), 'Partial pair published')
            continue
        pairs += 1
        semantics = set()
        for side in ('current', 'comparison'):
            cell = row['cells'][side]
            value = Decimal(cell['value_decimal'])
            scale = cell['unit']['scale']
            require(value.is_finite() and value >= 0 and scale in (1000, 1000000, 1000000000), 'Invalid source units/value')
            require(Decimal(cell['displayed_value'].replace(',', '').strip()) == value, 'Displayed source mismatch')
            require(Decimal(cell['value_in_shares']) == value * scale, 'Incorrect unit normalization')
            require(cell['source_sha256'] == old['source']['sha256'], 'Cell source mismatch')
            require(cell['period_end'] == old['periods'][side]['period_end'], 'Cell period mismatch')
            require(cell['source_admission_changed'] is False, 'Cell admission promoted')
            require(cell['weighted_average_explicit'] is (cell['semantic_basis'] == 'explicit_basic_weighted_average_shares'), 'Semantic flag mismatch')
            semantics.add(cell['semantic_basis'])
        require(len(semantics) == 1, 'Current/comparative semantics differ')
        explicit += row['cells']['current']['weighted_average_explicit']
        implicit += not row['cells']['current']['weighted_average_explicit']
    require(len({x['event_id'] for x in oracle}) == len(oracle), 'Duplicate manual oracle')
    for wanted in oracle:
        row = by_id[wanted['event_id']]
        require(row['raw_annual_pair_extracted'], 'Manual source pair missing')
        for side in ('current', 'comparison'):
            cell = row['cells'][side]
            require(Decimal(cell['value_decimal']) == Decimal(wanted[side]), 'Manual source-cell disagreement: ' + wanted['event_id'])
            require(cell['unit']['scale'] == wanted['share_scale'], 'Manual source unit disagreement')
        if row['symbol'] == 'AAPL':
            for side, column in (('current', 2), ('comparison', 3)):
                loc = row['cells'][side]['locator']
                require(loc['page_1based'] == 1 and loc['financial_column_index_0based'] == column, 'Apple annual geometry mismatch')
                require(loc['duration_literal'] == 'Twelve Months Ended', 'Apple duration mismatch')
    return dict(events=len(events), raw_annual_pairs_extracted=pairs,
                explicit_weighted_average_pairs=explicit, implicit_basic_eps_denominator_pairs=implicit,
                sampled_numeric_pairs_verified=len(oracle),
                by_symbol={s:dict(events=sum(e['symbol']==s for e in events),
                                 pairs=sum(e['symbol']==s and e['raw_annual_pair_extracted'] for e in events))
                           for s in sorted({e['symbol'] for e in events})})


def run(manifest_sha):
    manifest = load(PRODUCER+'MANIFEST.json', manifest_sha)
    pins = dict(manifest['input_hashes'])
    for extra in (manifest['implementation_hashes'], manifest['outputs']):
        for name, digest in extra.items():
            require(name not in pins or pins[name] == digest, 'Conflicting closure hash')
            pins[name] = digest
    verify(pins)
    manual = load(MANUAL)
    verify(manual['input_hashes'])
    inspection = load(INSPECTION)
    verify(inspection['input_hashes'])
    for row in inspection['inspected_sample']:
        if 'full_page_image' in row:
            ref = row['full_page_image']
            verify({ref['path']:ref['sha256']})
    data = load(PRODUCER+'RAW_SHARES.json')
    require(data['schema'] == 'after158_basic_shares_raw_extraction_v1', 'Wrong extraction schema')
    require(data['source_admission_changed'] is False and manifest['source_admission_changed'] is False, 'Source admission changed')
    for key in ('factors', 'scores', 'returns', 'orders'):
        require(data['activity'][key] == 0, 'Unexpected financial activity')
    stats = audit(data['events'], load(PLAN)['events'], load(INVENTORY)['events'], manual['manual_source_pairs'])
    require(stats['events'] == 44 and stats['sampled_numeric_pairs_verified'] == 17, 'Unexpected scope')
    for key in ('events', 'raw_annual_pairs_extracted', 'explicit_weighted_average_pairs',
                'implicit_basic_eps_denominator_pairs', 'by_symbol'):
        require(data['summary'][key] == stats[key] == manifest['summary'][key], 'Summary mismatch: ' + key)
    require(data['activity']['source_cells_extracted'] == stats['raw_annual_pairs_extracted']*2, 'Source cell count mismatch')
    for name in (SELF, MANUAL, INSPECTION, PRODUCER+'MANIFEST.json', 'work/monthly_stock_comp149_proposal.json'):
        pins[name] = sha(path(name).read_bytes())
    pins.update(inspection['input_hashes'])
    verify(pins)
    report = dict(schema='after158_basic_shares_independent_review_v1',
                  created_utc=datetime.now(timezone.utc).isoformat(), passed=True,
                  review_scope='Full 44-event closure/metadata/holds; manual source value and unit crosscheck of 17 preselected pairs, including all 10 Apple PDF page1 layouts. Not an independent extraction of the remaining 27 pairs.',
                  input_hashes=pins, summary=stats,
                  checks=dict(full_original_44_denominator_and_order=True,
                              original_source_identity_timing_periods_unchanged=True,
                              inherited_holds_unchanged=True, all_manifest_pins_verified=True,
                              no_admission_promotion=True, source_unit_normalization_checked=True,
                              manual_17_pairs_match=True, all_10_apple_page1_geometry_reviewed=True,
                              explicit_and_implicit_weighting_kept_distinct=True,
                              no_automatic_split_adjustment_or_history_certification=True,
                              all_44_pairs_independently_numerically_reextracted=False),
                  manual_findings_ref={'path':MANUAL,'sha256':pins[MANUAL]},
                  limitations=manual['method_limits'], novelty=manual['novelty'],
                  source_admission_changed=False, financial_features_computed=0,
                  returns_computed=0, orders_computed=0, new_strategy_trials=0)
    require(not path(RESULT).exists(), 'Review already exists; never overwrite')
    raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
    with path(RESULT).open('xb') as handle:
        handle.write(raw)
    print(json.dumps({'path':RESULT, 'sha256':sha(raw), 'summary':stats}))


class TestIndependentAudit(unittest.TestCase):
    def setUp(self):
        old = dict(original_index=0,event_id='TEST',symbol='TEST',fiscal_year=2020,
                   source={'sha256':'a'*64},release_date='2021-01-01',available_from='2021-01-04',
                   identity={}, periods={'current':{'period_end':'2020-12-31'},'comparison':{'period_end':'2019-12-31'}})
        row = deepcopy(old)
        row.update(inherited_holds={'unverified':True}, source_admission_changed=False,
                   fact_admitted=False, model_admitted=False, strategy_admitted=False,
                   split_evidence={'automatic_adjustment_applied':False,'complete_split_history_certified':False},
                   raw_annual_pair_extracted=True, failures=[], cells={})
        for side, value in [('current','8'),('comparison','10')]:
            row['cells'][side] = dict(value_decimal=value, displayed_value=value,
                value_in_shares=str(int(value)*1000),unit={'scale':1000}, source_sha256='a'*64,
                period_end=old['periods'][side]['period_end'],source_admission_changed=False,
                semantic_basis='explicit_basic_weighted_average_shares',weighted_average_explicit=True)
        self.old=[old];self.rows=[row];self.inventory=[{'event_id':'TEST','inherited_holds':{'unverified':True}}]
        self.oracle=[{'event_id':'TEST','current':'8','comparison':'10','share_scale':1000}]

    def test_valid_sample(self):
        self.assertEqual(audit(self.rows,self.old,self.inventory,self.oracle)['sampled_numeric_pairs_verified'],1)

    def test_swapped_values_fail(self):
        self.oracle[0]['current']='10'
        with self.assertRaisesRegex(ValueError,'source-cell'):
            audit(self.rows,self.old,self.inventory,self.oracle)

    def test_wrong_scale_fails_even_if_value_normalized(self):
        self.rows[0]['cells']['current']['unit']['scale']=1000000
        self.rows[0]['cells']['current']['value_in_shares']='8000000'
        with self.assertRaisesRegex(ValueError,'source unit'):
            audit(self.rows,self.old,self.inventory,self.oracle)

    def test_hold_removed_fails(self):
        self.rows[0]['inherited_holds']={}
        with self.assertRaisesRegex(ValueError,'holds'):
            audit(self.rows,self.old,self.inventory,self.oracle)

    def test_admission_promoted_fails(self):
        self.rows[0]['fact_admitted']=True
        with self.assertRaisesRegex(ValueError,'promoted'):
            audit(self.rows,self.old,self.inventory,self.oracle)

    def test_implicit_mislabeled_as_explicit_fails(self):
        self.rows[0]['cells']['current']['semantic_basis']='basic_eps_denominator_weighting_implicit'
        with self.assertRaisesRegex(ValueError,'Semantic'):
            audit(self.rows,self.old,self.inventory,self.oracle)

    def test_duplicate_event_fails(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            audit(self.rows*2,self.old*2,self.inventory*2,self.oracle)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--test-only',action='store_true')
    parser.add_argument('--manifest-sha')
    args=parser.parse_args()
    if args.test_only:
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TestIndependentAudit))
        raise SystemExit(0 if result.wasSuccessful() else 1)
    require(re.fullmatch('[0-9a-f]{64}',args.manifest_sha or '') is not None,'Explicit producer manifest SHA required')
    run(args.manifest_sha)
