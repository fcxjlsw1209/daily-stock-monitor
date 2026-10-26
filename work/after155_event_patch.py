"""Freeze, build and verify a four-event source-only catalog overlay.

Never rewrites the original 3,771-event JSON. The overlay manifest references its
exact original bytes and a four-row added array. No prices/features are read.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parent.parent
OUT = 'outputs/monthly_event_patch156'
SELF = 'work/after155_event_patch.py'
SPEC = OUT + '/PATCH_SPEC.json'
PARENT = 'outputs/monthly_earnings_repair/EVENTS.json'
PARENT_SHA = '1e027fc8b933af9753241c15a2e0dcc5a32daa13205ba4d74f90b64db6e2e103'
CALENDAR = 'outputs/monthly_paired_target143/CALENDAR.json'
REVIEWS = {
    'avgo': 'work/after155_broadcom_independent_review.json',
    'abt': 'work/after155_abbott_independent_review.json',
}
IDENTITIES = (
    ('ABT_2020-04-16', 'ABT', 2020, 1, '2020-03-31', '2020-04-16', '2020-04-17'),
    ('ABT_2020-07-16', 'ABT', 2020, 2, '2020-06-30', '2020-07-16', '2020-07-17'),
    ('ABT_2020-10-21', 'ABT', 2020, 3, '2020-09-30', '2020-10-21', '2020-10-22'),
    ('AVGO_2021-06-03', 'AVGO', 2021, 2, '2021-05-02', '2021-06-03', '2021-06-04'),
)
ZERO_ACTIVITY = {'downloads': 0, 'price_reads': 0, 'feature_calculations': 0,
                 'model_fits': 0, 'orders': 0, 'return_calculations': 0, 'strategy_plans': 0}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def local(rel):
    p = Path(rel)
    if p.is_absolute() or '..' in p.parts:
        raise ValueError('Expected local relative path: ' + str(rel))
    p = ROOT / p
    if p.resolve() != p:
        raise ValueError('Symlink/path redirection: ' + str(rel))
    return p


def relative(value):
    p = Path(value)
    return str(p.relative_to(ROOT)) if p.is_absolute() else str(p)


def binding(rel):
    raw = local(rel).read_bytes()
    return {'path': rel, 'sha256': sha(raw), 'bytes': len(raw)}


def pinned(b):
    raw = local(b['path']).read_bytes()
    if sha(raw) != b['sha256'] or len(raw) != b['bytes']:
        raise ValueError('Source binding mismatch: ' + b['path'])
    return raw


def write_new(rel, value):
    p = local(rel)
    p.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    with p.open('xb') as f:
        f.write(raw)
    return {'path': rel, 'sha256': sha(raw), 'bytes': len(raw)}


def candidates_from_reviews(reviews):
    avgo, abt = reviews['avgo'], reviews['abt']
    if avgo['admission_decision']['single_event_may_enter_new_derived_catalog'] is not True:
        raise ValueError('AVGO source not admitted for an overlay')
    if abt['admission_decision']['these_three_events_may_enter_new_derived_catalog'] is not True:
        raise ValueError('ABT sources not admitted for an overlay')
    result = []
    for e in abt['events']:
        if e['single_event_admissible'] is not True:
            raise ValueError('ABT single-event admission missing')
        result.append({
            'event_id': e['candidate_event_id'], 'symbol': 'ABT',
            'fiscal_year': e['fiscal_year'], 'fiscal_quarter': e['fiscal_quarter'],
            'actual_period_end': e['official_actual_period_end'],
            'publication_date': e['publication_date'], 'publication_at': None,
            'publication_timezone': e['publication_timezone'],
            'source_url': e['source_url'], 'raw_path': relative(e['source_path']),
            'retrieved_at': e['retrieved_utc'], 'source_kind': 'original_issuer_release_full_bytes',
            'review_key': 'abt', 'expected': {k: e[k] for k in [
                'precision', 'date_only_upper_bound_at', 'available_session', 'known_by_at',
                'known_by_is_actual_publication', 'first_public_time_verified']},
            'evidence': {k: e[k] for k in ['title', 'issuer_excerpt', 'actual_results_excerpt', 'period_excerpt', 'dateline']},
            'source_display_timestamp': None,
        })
    e, a = avgo['raw_evidence'], avgo['conservative_availability']
    result.append({
        'event_id': avgo['original_catalog_gap_and_conflict_checks']['candidate_event_id'],
        'symbol': 'AVGO', 'fiscal_year': e['fiscal_year'], 'fiscal_quarter': e['fiscal_quarter'],
        'actual_period_end': e['official_actual_period_end'], 'publication_date': a['publication_date'],
        'publication_at': None, 'publication_timezone': a['publication_timezone'],
        'source_url': avgo['collection_assessment']['source_url'], 'raw_path': relative(e['path']),
        'retrieved_at': avgo['collection_assessment']['retrieved_utc'],
        'source_kind': 'issuer_distributed_PRNewswire_release_full_bytes', 'review_key': 'avgo',
        'expected': {'precision': a['admitted_policy_precision'],
                     **{k: a[k] for k in ['date_only_upper_bound_at', 'available_session', 'known_by_at',
                                         'known_by_is_actual_publication', 'first_public_time_verified']}},
        'evidence': {'title': e['title'], 'issuer_excerpt': e['issuer_attribution_excerpt'],
                     'actual_results_excerpt': e['actual_release_excerpt'],
                     'period_excerpt': e['fiscal_period_excerpt'], 'dateline': e['historical_dateline']},
        'source_display_timestamp': a['source_display_timestamp'],
    })
    result.sort(key=lambda x: x['event_id'])
    identities = tuple((r['event_id'], r['symbol'], r['fiscal_year'], r['fiscal_quarter'],
                        r['actual_period_end'], r['publication_date'], r['expected']['available_session']) for r in result)
    if identities != IDENTITIES:
        raise ValueError('Reviewed candidates do not equal the exact four authorized identities')
    return result


def freeze():
    reviews = {k: json.loads(local(p).read_bytes()) for k, p in REVIEWS.items()}
    candidates = candidates_from_reviews(reviews)
    expected = {}
    for review in reviews.values():
        for rel, digest in review['verified_input_sha256'].items():
            if rel in expected and expected[rel] != digest:
                raise ValueError('Independent reviews disagree on source binding: ' + rel)
            expected[rel] = digest
    if expected.get(PARENT) != PARENT_SHA:
        raise ValueError('Wrong parent catalog')
    cal = json.loads(local(CALENDAR).read_bytes())
    for name, digest in cal['provenance']['source_hashes'].items():
        rel = relative(name)
        if rel in expected and expected[rel] != digest:
            raise ValueError('Calendar source conflict: ' + rel)
        expected[rel] = digest
    inputs = {}
    for rel, digest in expected.items():
        inputs[rel] = binding(rel)
        if inputs[rel]['sha256'] != digest:
            raise ValueError('Reviewed source changed: ' + rel)
    for rel in list(REVIEWS.values()) + [SELF, 'run_local.sh', 'local_run.py', 'local.sandbox.sb']:
        inputs[rel] = binding(rel)
    if len(json.loads(pinned(inputs[PARENT]))) != 3771:
        raise ValueError('Parent does not contain exactly3771 entries')
    spec = {
        'schema_version': 1, 'kind': 'four_event_source_only_overlay_spec',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'inputs': inputs,
        'parent': inputs[PARENT], 'parent_count': 3771, 'added_count': 4, 'combined_count': 3775,
        'review_paths': REVIEWS, 'candidates': candidates,
        'policy': {
            'normalizer': 'outputs/monthly_earnings_repair/event_schema.py:normalize_event',
            'calendar': CALENDAR,
            'precision': 'date_only for all4; retain AVGO displayed clock separately, never use it as first-public-time proof',
            'composition': 'manifest references parent original bytes plus ADDED_EVENTS array; never deserialize/reserialize parent into a new saved catalog',
            'conflicts': ['event_id', 'symbol/fiscal_year/fiscal_quarter', 'symbol/publication_date', 'symbol/actual_period_end'],
            'scope': 'Exactly4 single events. No complete-window, absence, factor, strategy or price-reaction admission.',
        },
        'activity': ZERO_ACTIVITY,
    }
    print(json.dumps(write_new(SPEC, spec)))


def keys(record):
    return {
        'event_id': (record.get('event_id'),),
        'fiscal_quarter': (record.get('symbol'), record.get('fiscal_year'), record.get('fiscal_quarter')),
        'publication_date': (record.get('symbol'), record.get('publication_date')),
        'actual_period_end': (record.get('symbol'), record.get('official_actual_period_end', record.get('actual_period_end'))),
    }


def assert_no_conflicts(parent, added):
    rows = list(parent)
    for new in added:
        nk = keys(new)
        if any(any(v is None for v in key) for key in nk.values()):
            raise ValueError('Incomplete new identity')
        for old in rows:
            ok = keys(old)
            for name, key in nk.items():
                if key == ok[name]:
                    raise ValueError('New event conflict on ' + name + ': ' + new['event_id'])
        rows.append(new)


def source_modules():
    d = str(ROOT / 'outputs/monthly_earnings_repair')
    if d not in sys.path:
        sys.path.insert(0, d)
    import event_schema
    import research_calendar
    return event_schema, research_calendar


def load_spec(spec_sha):
    raw = local(SPEC).read_bytes()
    if sha(raw) != spec_sha:
        raise ValueError('Patch spec SHA mismatch')
    spec = json.loads(raw)
    for b in spec['inputs'].values():
        pinned(b)
    reviews = {k: json.loads(pinned(spec['inputs'][p])) for k, p in REVIEWS.items()}
    if candidates_from_reviews(reviews) != spec['candidates']:
        raise ValueError('Candidate specs no longer match independent source approvals')
    parent = json.loads(pinned(spec['parent']))
    if spec['parent']['sha256'] != PARENT_SHA or len(parent) != 3771:
        raise ValueError('Wrong parent catalog/version')
    return spec, reviews, parent


def normalized_added(spec, reviews):
    from bs4 import BeautifulSoup
    normalizer, calendars = source_modules()
    frozen = json.loads(pinned(spec['inputs'][CALENDAR]))
    calendar = calendars.market_calendar(frozen['start'], frozen['end'])
    if [str(d.date()) for d in calendar.sessions] != frozen['sessions']:
        raise ValueError('Constructed source calendar differs from pinned sessions')
    added = []
    for c in spec['candidates']:
        raw_binding = spec['inputs'][c['raw_path']]
        raw = pinned(raw_binding)
        text = ' '.join(BeautifulSoup(raw, 'html.parser').get_text(' ', strip=True).split())
        # Source approvals are independently pinned; this is an additional
        # basic content-presence check, not another claim of independent review.
        for key in ['title', 'actual_results_excerpt', 'period_excerpt']:
            if ' '.join(c['evidence'][key].split()) not in text:
                raise ValueError('Reviewed raw evidence no longer present: ' + c['event_id'] + ':' + key)
        r = normalizer.normalize_event({
            'symbol': c['symbol'], 'asset_type': 'equity', 'vendor': {},
            'official': {'actual_period_end': c['actual_period_end'],
                         'publication_at': None, 'publication_date': c['publication_date'],
                         'publication_timezone': c['publication_timezone'],
                         'source_url': c['source_url'], 'verified': True,
                         'publication_kind': 'newswire_release' if c['symbol'] == 'AVGO' else 'issuer_release'},
        }, calendar=calendar)
        if r['status'] != 'ready' or r['issues']:
            raise ValueError('Normalizer did not accept the source: ' + c['event_id'])
        for name, expected in c['expected'].items():
            if name == 'first_public_time_verified':
                continue
            if r[name] != expected:
                raise ValueError('Availability does not match independent approval: ' + c['event_id'] + ':' + name)
        r.update({
            'event_id': c['event_id'], 'fiscal_year': c['fiscal_year'], 'fiscal_quarter': c['fiscal_quarter'],
            'evidence_path': c['raw_path'], 'evidence_sha256': raw_binding['sha256'],
            'source_kind': c['source_kind'], 'source_precision': 'date_only',
            'source_retrieved_at': c['retrieved_at'], 'source_display_timestamp': c['source_display_timestamp'],
            'source_display_timestamp_used_as_first_public_time': False,
            'publication_evidence': c['evidence']['dateline'], 'period_evidence': c['evidence']['period_excerpt'],
            'source_evidence': c['evidence'],
            'publication_version': 'later_retrieved_original_issuer_or_distributed_release',
            'historical_archive_limitation': 'Retrieved in2026; historical release evidence is verified, but no contemporaneous immutable capture or exact first-public instant is certified.',
            'original_response_available': True, 'first_public_time_verified': False,
            'independent_source_review': spec['inputs'][REVIEWS[c['review_key']]],
            'single_event_source_admitted': True, 'whole_window_archive_complete': False,
            'factor_admitted': False, 'strategy_admitted': False,
        })
        added.append(r)
    return added


def manifest_for(spec, spec_sha, added_binding):
    return {
        'schema_version': 1, 'kind': 'immutable_parent_plus_added_events_overlay',
        'spec': {'path': SPEC, 'sha256': spec_sha},
        'components': [dict(spec['parent'], count=3771, role='parent_original_bytes'),
                       dict(added_binding, count=4, role='added_single_event_sources')],
        'combined_event_count': 3775, 'logical_order': 'parent original order followed by ADDED_EVENTS order',
        'parent_saved_bytes_rewritten': False, 'parent_paths_rewritten': False,
        'path_handling': 'Existing launcher may relocate old paths in memory when reading. This overlay stores no reserialized parent records and references their unchanged original bytes.',
        'whole_window_archive_complete': False, 'factor_admitted': False, 'strategy_admitted': False,
    }


def build(spec_sha):
    spec, reviews, parent = load_spec(spec_sha)
    added = normalized_added(spec, reviews)
    assert_no_conflicts(parent, added)
    if len(added) != 4:
        raise ValueError('Only four additions are permitted')
    added_binding = write_new(OUT + '/ADDED_EVENTS.json', added)
    manifest = manifest_for(spec, spec_sha, added_binding)
    manifest_binding = write_new(OUT + '/CATALOG_MANIFEST.json', manifest)
    if sha(local(PARENT).read_bytes()) != PARENT_SHA:
        raise ValueError('Parent changed during build')
    result = {
        'schema_version': 1, 'status': 'PASS_FOUR_EVENT_SOURCE_OVERLAY',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'spec': {'path': SPEC, 'sha256': spec_sha}, 'parent': spec['parent'],
        'added_events': added_binding, 'manifest': manifest_binding,
        'parent_count': 3771, 'added_count': len(added), 'combined_count': len(parent)+len(added),
        'added_event_ids': [e['event_id'] for e in added],
        'normalized_availability_matches_all_independent_expectations': True,
        'duplicate_or_conflicting_new_ids_or_keys': 0,
        'old_events_sha256_before': PARENT_SHA, 'old_events_sha256_after': sha(local(PARENT).read_bytes()),
        'old_events_bytes_unchanged': True,
        'checks': ['all input source and independent-review hashes', 'exact four fiscal-quarter identities',
                   'original parent3771 count/hash', 'all pinned calendar sessions match constructed calendar',
                   'date-only timezone end-of-day bounds', 'next session and known-by close versus independent review',
                   'duplicate event ID, fiscal-quarter, publication-date and period-end conflicts',
                   'new-only overlay manifest and parent byte preservation'],
        'limitations': ['Only four single-event sources admitted.', 'No full-window completeness or right-boundary absence certification.',
                        'No recalculation of census windows yet; no claim all source gaps resolved.',
                        'No factor or strategy admission, financial feature, price reaction or return computation.'],
        'activity': ZERO_ACTIVITY,
    }
    print(json.dumps(write_new(OUT + '/INTEGRATION.json', result)))


def verify(spec_sha):
    spec, reviews, parent = load_spec(spec_sha)
    integration = json.loads(local(OUT + '/INTEGRATION.json').read_bytes())
    added_raw = pinned(integration['added_events'])
    added = json.loads(added_raw)
    expected = normalized_added(spec, reviews)
    if added != expected or len(added) != 4:
        raise ValueError('Added records do not reproduce')
    assert_no_conflicts(parent, added)
    manifest = json.loads(pinned(integration['manifest']))
    if manifest != manifest_for(spec, spec_sha, integration['added_events']):
        raise ValueError('Overlay manifest does not reproduce')
    if integration['parent_count'] != len(parent) or integration['combined_count'] != len(parent)+len(added):
        raise ValueError('Catalog count mismatch')
    if integration['old_events_sha256_after'] != PARENT_SHA or sha(local(PARENT).read_bytes()) != PARENT_SHA:
        raise ValueError('Parent preservation check failed')
    print(json.dumps({'status': 'REPRODUCIBLE_SOURCE_OVERLAY_VERIFIED', 'parent_count': len(parent),
                      'added_count': len(added), 'combined_count': len(parent)+len(added),
                      'writes': 0, 'integration': binding(OUT + '/INTEGRATION.json')}))


class Tests(unittest.TestCase):
    def row(self):
        return dict(event_id='X_2020-01-10', symbol='X', fiscal_year=2019, fiscal_quarter=4,
                    publication_date='2020-01-10', actual_period_end='2019-12-31')
    def test_empty_no_conflicts(self):
        assert_no_conflicts([], [self.row()])
    def test_all_four_keys_rejected(self):
        for field in ['event_id', 'fiscal_quarter', 'publication_date', 'actual_period_end']:
            original, new = self.row(), self.row()
            new.update(event_id='Y_2021-02-02', fiscal_year=2021, fiscal_quarter=1,
                       publication_date='2021-02-02', actual_period_end='2021-01-31')
            if field == 'fiscal_quarter': new.update(fiscal_year=2019, fiscal_quarter=4)
            else: new[field] = original[field]
            with self.assertRaises(ValueError): assert_no_conflicts([original], [new])
    def test_duplicate_added_rejected(self):
        with self.assertRaises(ValueError): assert_no_conflicts([], [self.row(), self.row()])
    def test_other_issuer_same_quarter_allowed(self):
        r = dict(self.row(), event_id='Y_2020-01-10', symbol='Y')
        assert_no_conflicts([self.row()], [r])
    def test_missing_identity_rejected(self):
        r = self.row(); r.pop('publication_date')
        with self.assertRaises(ValueError): assert_no_conflicts([], [r])
    def test_date_only_policy(self):
        import pandas as pd
        normalizer, _ = source_modules()
        schedule = pd.DataFrame({'close': pd.to_datetime(['2020-04-16T20:00Z', '2020-04-17T20:00Z'])},
                                index=pd.to_datetime(['2020-04-16', '2020-04-17']))
        r = normalizer.normalize_event({'symbol': 'ABT', 'official': {
            'actual_period_end': '2020-03-31', 'publication_date': '2020-04-16',
            'publication_timezone': 'America/Chicago', 'verified': True, 'source_url': 'https://example.test/release',
            'publication_kind': 'issuer_release'}}, calendar=SimpleNamespace(schedule=schedule))
        self.assertEqual(r['known_by_at'], '2020-04-17T20:00:00+00:00')
        self.assertEqual(r['date_only_upper_bound_at'], '2020-04-17T04:59:59.999999999+00:00')
        self.assertFalse(r['known_by_is_actual_publication'])
    def test_bad_paths(self):
        for p in ['/tmp/x', '../x']:
            with self.assertRaises(ValueError): local(p)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ['freeze', 'build', 'verify', 'self-test']:
        modes.add_argument('--'+mode, action='store_true')
    parser.add_argument('--spec-sha')
    args = parser.parse_args()
    if args.self_test:
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
        raise SystemExit(0 if result.wasSuccessful() else 1)
    elif args.freeze: freeze()
    elif not args.spec_sha: parser.error('--build/--verify requires --spec-sha')
    elif args.build: build(args.spec_sha)
    else: verify(args.spec_sha)
