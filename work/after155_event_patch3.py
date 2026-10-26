"""Immutable third event increment: two independently reviewed CAT wire originals.

Only source/calendar metadata is consumed. No prices, financial fields or factors.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import unittest

from bs4 import BeautifulSoup
import after155_event_patch2 as prior

ROOT = prior.ROOT
SELF = 'work/after155_event_patch3.py'
OUT = 'outputs/monthly_event_patch156_batch3'
SPEC = OUT + '/PATCH_SPEC.json'
REVIEW = 'work/after155_cat_wire_independent_review.json'
SECOND = 'outputs/monthly_event_patch156_batch2/'
PARENT_MANIFEST = SECOND + 'CATALOG_MANIFEST.json'
PINS = {
    'work/after155_event_patch2.py': '96bebbdf7a753d79b3567015704b85bbe4dfea0706b04c09099b90fe81e26fae',
    REVIEW: '0dbbfb8357716ef6b142224fd82d529a4685a31e84e159ad623c7e39445a9b30',
    'work/after155_cat_wire_independent_review.py': '5bcdc32a70f7b8431a0b3094bd6215957dbfabd6d8c92f682029b4b181863eaf',
    SECOND + 'PATCH_SPEC.json': 'fcb39915b693623e467c752d526b331bee0e2d54f428df950ee0bda467658cf3',
    PARENT_MANIFEST: '04fa343d1bafc19e32fbb5fcf52f90945e81b71af9c73adfda016682d2038a62',
    SECOND + 'ADDED_EVENTS.json': 'a98fec93a5ccf97f6e42954b868b3bc0bb55d43cf3740a66cf24950240103586',
    SECOND + 'INTEGRATION.json': 'e071785472846455da33773d9d114d012e2ed2876040b03dffb0b49a64a5d53e',
    SECOND + 'CONSUMPTION_IMPACT.json': '0f9835b047da259924fb669ee6320189b0e9b13b7258f40bcc76f292cf90d8bd',
}
EXPECTED = (
    ('CAT_2025-04-30', 'CAT', 2025, 1, '2025-03-31', '2025-04-30', '2025-05-01'),
    ('CAT_2025-08-05', 'CAT', 2025, 2, '2025-06-30', '2025-08-05', '2025-08-06'),
)
NAMES = ('ADDED_EVENTS.json', 'CONSUMPTION_IMPACT.json', 'CATALOG_MANIFEST.json', 'INTEGRATION.json')
ZERO = dict(prior.ZERO)
require, read, sha, binding, checked, write = prior.require, prior.read, prior.sha, prior.binding, prior.checked, prior.write


def event_identity(r):
    return tuple(r[k] for k in ('event_id', 'symbol', 'fiscal_year', 'fiscal_quarter',
                               'official_actual_period_end', 'publication_date', 'available_session'))


def reviewed_events():
    r = read(REVIEW)
    require(r['status'] == 'pass_two_CAT_events_for_new_derived_catalog_only', 'review status')
    decision = r['admission_decision']
    require(decision['these_two_events_may_enter_new_derived_catalog'] is True, 'no independent append approval')
    for key in ('complete_event_calendar_verified', 'factor_admitted', 'strategy_admitted', 'whole_window_archive_completeness_verified'):
        require(decision[key] is False, 'review scope overstated: ' + key)
    events = deepcopy(r['events'])
    require(tuple(map(event_identity, events)) == EXPECTED, 'not exactly the two approved CAT events')
    for e in events:
        require(e['single_event_source_admissible'] is True and e['is_actual_results_release'] is True,
                'event not independently approved actual results')
        require(e['is_advance_notice'] is False and e['first_public_time_verified'] is False, 'invalid source nature')
        require(e['publication_at'] is None and e['precision'] == 'date_only', 'clock policy changed')
        require(e['publication_timezone'] == 'America/Chicago', 'issuer date zone changed')
    return events


def parent_records():
    m = read(PARENT_MANIFEST)
    expected_paths = [prior.PARENT, prior.FIRST, SECOND + 'ADDED_EVENTS.json']
    require(m['combined_event_count'] == 3779 and len(m['components']) == 3, 'wrong latest parent manifest')
    require([b['path'] for b in m['components']] == expected_paths, 'parent component order differs')
    rows = []
    for b, count in zip(m['components'], [3771, 4, 4]):
        records = json.loads(checked(b))
        require(len(records) == count == b['count'], 'parent component count differs')
        rows.extend(records)
    require(len(rows) == 3779 and len({r['event_id'] for r in rows}) == 3779, 'parent identities not unique')
    return rows


def verify_source(raw, receipt, e):
    require(receipt['http_status'] == 200, 'source response not HTTP200')
    require(receipt['path'] == e['source_path'] and receipt['sha256'] == e['source_sha256'], 'receipt source mismatch')
    require(receipt['bytes'] == len(raw) and prior.digest(raw) == e['source_sha256'], 'source bytes mismatch')
    require(receipt['requested_url'] == e['source_url'] == receipt['final_url'], 'source URL mismatch')
    require('html' in receipt['content_type'].lower(), 'non-HTML source response')
    require(receipt['request_utc'] == e['retrieved_utc'], 'retrieval time differs')
    soup = BeautifulSoup(raw, 'html.parser')
    articles = soup.select('article.news-release')
    require(len(articles) == 1, 'unique original release article required')
    article = articles[0]
    for key, ref in e['evidence'].items():
        nodes = [n for n in article.find_all(ref['tag']) if n.sourceline == ref['html_line_1based']]
        # The pinned independent reviewer explicitly selected these inner spans.
        # Parent legend spans have identical text and source line but are not its nodes.
        if key in ('date', 'location'):
            cls = {'date': 'xn-chron', 'location': 'xn-location'}[key]
            nodes = [n for n in nodes if cls in n.get('class', [])]
        matches = []
        for node in nodes:
            text = ' '.join(node.get_text(' ', strip=True).split())
            if prior.digest(text.encode()) == ref['node_text_sha256'] and ref['quote'] in text:
                matches.append(node)
        require(len(matches) == 1, 'independently reviewed DOM evidence changed: ' + key)
    return True


def closure():
    found = {}
    def add(p, h):
        p = prior.relative(p)
        require(p not in found or found[p]['sha256'] == h, 'conflicting SHA ' + p)
        require(sha(p) == h, 'input SHA differs ' + p)
        found[p] = binding(p)
    for p, h in PINS.items():
        add(p, h)
    for b in read(SECOND + 'PATCH_SPEC.json')['inputs'].values():
        checked(b); add(b['path'], b['sha256'])
    for b in read(PARENT_MANIFEST)['components']:
        checked(b); add(b['path'], b['sha256'])
    r = read(REVIEW)
    for p, h in r['verified_input_sha256'].items():
        add(p, h)
    source_spec = read('outputs/monthly_event_cat_wire156/SOURCE_SPEC.json')
    for p, h in source_spec['input_hashes'].items():
        add(p, h)
    for p in [SELF, 'run_local.sh', 'local_run.py', 'local.sandbox.sb', OUT + '/TESTS.log']:
        add(p, sha(p))
    # Bind local third-party parser code actually loaded by the evidence verifier.
    for name, module in sorted(list(sys.modules.items())):
        if name == 'bs4' or name.startswith('bs4.'):
            p = getattr(module, '__file__', None)
            if p:
                p = prior.relative(p); add(p, sha(p))
    return {p: found[p] for p in sorted(found)}


def receipt_path(e):
    return str(Path(e['source_path']).with_suffix('.receipt.json'))


def validate_sources(events, inputs):
    for e in events:
        raw = checked(inputs[e['source_path']])
        path = receipt_path(e)
        require(inputs[path]['sha256'] == e['receipt_sha256'], 'review receipt SHA differs')
        verify_source(raw, json.loads(checked(inputs[path])), e)


def freeze():
    require(not prior.local(SPEC).exists(), 'spec exists')
    inputs = closure()
    events = reviewed_events()
    validate_sources(events, inputs)
    old = parent_records()
    related = prior.collision_audit(old, events)
    prior.frozen_scope()
    s = dict(schema_version=1, kind='third_two_CAT_event_increment_spec',
        created_utc=datetime.now(timezone.utc).isoformat(), python=sys.version,
        inputs=inputs, reviewed_events=events, independent_review=inputs[REVIEW],
        parent_manifest=inputs[PARENT_MANIFEST], parent_count=3779, added_count=2, combined_count=3781,
        source_class='issuer_distributed_original_release_PRNewswire', related_period_releases=related,
        consumption_scope=inputs[prior.SCOPE + 'SLOTS.json'],
        policy=dict(composition='Unchanged original3771 + first4 + second4 + third2 referenced components',
            period_key_deduplication=False, same_period_distinct_release='retain and flag; never merge',
            original3904_slots_and99_intervals_unchanged=True, anchor_offsets=[-271, -44],
            printed_timestamp_used_as_first_publication=False, date_only_zone='America/Chicago',
            whole_window_archive_complete=False, factor_admitted=False, strategy_admitted=False), activity=ZERO)
    print(json.dumps(write(SPEC, s)))


def load_spec(expected):
    require(sha(SPEC) == expected, 'spec SHA differs')
    s = read(SPEC)
    require(s['python'] == sys.version, 'Python runtime differs')
    for b in s['inputs'].values():
        checked(b)
    require(s['reviewed_events'] == reviewed_events(), 'reviewed events changed')
    require(len(parent_records()) == s['parent_count'] == 3779, 'parent changed')
    validate_sources(s['reviewed_events'], s['inputs'])
    prior.frozen_scope()
    return s


def normalize(s):
    directory = str(ROOT / 'outputs/monthly_earnings_repair')
    if directory not in sys.path:
        sys.path.insert(0, directory)
    import event_schema
    import research_calendar
    frozen = read(prior.CALENDAR)
    cal = research_calendar.market_calendar(frozen['start'], frozen['end'])
    require([str(d.date()) for d in cal.sessions] == frozen['sessions'], 'calendar differs')
    out = []
    for e in s['reviewed_events']:
        r = event_schema.normalize_event(dict(symbol=e['symbol'], asset_type='equity', vendor={}, official=dict(
            actual_period_end=e['official_actual_period_end'], publication_date=e['publication_date'],
            publication_at=None, publication_timezone=e['publication_timezone'], source_url=e['source_url'],
            verified=True, publication_kind='issuer_release')), calendar=cal)
        require(r['status'] == 'ready' and r['issues'] == [], 'normalized event not ready')
        for k in ('symbol', 'official_actual_period_end', 'publication_date', 'publication_at', 'publication_timezone',
                  'known_by_at', 'date_only_upper_bound_at', 'available_session', 'precision', 'exchange'):
            require(r[k] == e[k], 'independent clock/identity differs: ' + k)
        r.update(event_id=e['event_id'], fiscal_year=e['fiscal_year'], fiscal_quarter=e['fiscal_quarter'],
            issuer=e['issuer'], evidence_path=e['source_path'], evidence_sha256=e['source_sha256'],
            source_receipt=s['inputs'][receipt_path(e)], source_retrieved_at=e['retrieved_utc'],
            source_kind=s['source_class'], release_kind='actual_quarterly_results', reporting_period_type='quarterly_results',
            standalone_quarterly_release_verified=True, quarter_is_period_endpoint_ordinal=False,
            period_end_is_context_inference=False, source_evidence=deepcopy(e['evidence']),
            independent_source_review=s['independent_review'], review_event_container='events',
            source_display_timestamp=e['source_display_timestamp'], source_display_timestamp_used_as_first_public_time=False,
            first_public_time_verified=False, known_by_is_actual_publication=False, source_precision='date_only',
            publication_version='later_retrieved_original_issuer_distributed_release', original_response_available=True,
            historical_archive_limitation=read(REVIEW)['vintage_limit'], single_event_source_admitted=True,
            whole_window_archive_complete=False, factor_admitted=False, strategy_admitted=False,
            relation_policy='Distinct preliminary, formal or corrective releases for one period are not deduplicated by quarter.')
        out.append(r)
    require(tuple(map(event_identity, out)) == EXPECTED, 'normalized scope differs')
    return out


def manifest(s, spec_sha, added):
    components = deepcopy(read(PARENT_MANIFEST)['components'])
    components.append(dict(added, count=2, role='third_increment_independently_reviewed_CAT'))
    return dict(schema_version=1, kind='immutable_parent_plus_three_increments', spec=dict(path=SPEC, sha256=spec_sha),
        prior_manifest=s['parent_manifest'], components=components, combined_event_count=3781,
        logical_order='Original parent order, then first, second and third increment orders',
        previous_component_bytes_rewritten=False, period_key_deduplication=False,
        whole_window_archive_complete=False, factor_admitted=False, strategy_admitted=False)


def build(spec_sha):
    s = load_spec(spec_sha)
    for name in NAMES:
        require(not prior.local(OUT + '/' + name).exists(), 'output exists ' + name)
    added = normalize(s)
    old = parent_records()
    related = prior.collision_audit(old, added)
    impact = prior.consumption(old, added)
    require(impact['changed_slots'] == len({r['slot_id'] for r in impact['changed_slot_records']}), 'duplicate impact slots')
    ab = write(OUT + '/ADDED_EVENTS.json', added)
    ib = write(OUT + '/CONSUMPTION_IMPACT.json', impact)
    mb = write(OUT + '/CATALOG_MANIFEST.json', manifest(s, spec_sha, ab))
    for b in s['inputs'].values():
        checked(b)
    report = dict(schema_version=1, status='PASS_THIRD_TWO_CAT_SOURCE_INCREMENT',
        created_utc=datetime.now(timezone.utc).isoformat(), spec=dict(path=SPEC, sha256=spec_sha),
        added_events=ab, manifest=mb, consumption_impact=ib, parent_count=3779, added_count=2, combined_count=3781,
        added_event_ids=[r['event_id'] for r in added], prior_components=deepcopy(read(PARENT_MANIFEST)['components']),
        prior_components_unchanged=True, latest_parent_collision_checked=True, related_period_releases=related,
        independently_reviewed_source_bytes_and_receipts_verified=True, independently_reviewed_dom_evidence_verified=True,
        clock_matches_independent_review=True, scope3904_unchanged=True, reporting_nature_preserved=True,
        tsla_excluded=True, whole_window_archive_complete=False, factor_admitted=False, strategy_admitted=False, activity=ZERO)
    print(json.dumps(write(OUT + '/INTEGRATION.json', report)))


def verify(spec_sha):
    s = load_spec(spec_sha)
    r = read(OUT + '/INTEGRATION.json')
    added = json.loads(checked(r['added_events']))
    require(added == normalize(s), 'normalized records differ')
    require(json.loads(checked(r['manifest'])) == manifest(s, spec_sha, r['added_events']), 'manifest differs')
    old = parent_records()
    require(prior.collision_audit(old, added) == r['related_period_releases'], 'collision result differs')
    impact = json.loads(checked(r['consumption_impact']))
    require(impact == prior.consumption(old, added), 'consumption impact differs')
    require(len(old) + len(added) == r['combined_count'] == 3781, 'count differs')
    require(r['prior_components'] == read(PARENT_MANIFEST)['components'], 'prior component bindings differ')
    print(json.dumps(dict(status='VERIFIED_THIRD_INCREMENT', integration=binding(OUT + '/INTEGRATION.json'),
        combined_count=3781, changed_slots=impact['changed_slots'], changed_intervals=impact['changed_intervals'],
        consumed_unique_before=impact['consumed_unique_before'], consumed_unique_after=impact['consumed_unique_after'],
        per_new_event_slot_ids=impact['per_new_event_slot_ids'], writes=0)))


class Tests(prior.Tests):
    # Scope-specific tests of the previous increment are superseded below.
    test_scope_exact_four = None
    test_no_cat_tsla = None
    """Retain prior collision tests and add source-receipt/DOM cases using fake HTML."""
    def test_scope_exact_two(self):
        self.assertEqual(len(EXPECTED), 2)
    def test_only_cat(self):
        self.assertEqual({r[1] for r in EXPECTED}, {'CAT'})
    def source(self):
        raw = b'<article class="news-release"><p>Fake issuer actual release</p></article>'
        e = dict(source_path='fake.html', source_sha256=prior.digest(raw), source_url='https://example.test/release',
                 retrieved_utc='2026-01-01T00:00:00Z', evidence={'dateline': dict(tag='p', html_line_1based=1,
                 quote='Fake issuer', node_text_sha256=prior.digest(b'Fake issuer actual release'))})
        receipt = dict(http_status=200, path=e['source_path'], sha256=e['source_sha256'], bytes=len(raw),
            requested_url=e['source_url'], final_url=e['source_url'], content_type='text/html', request_utc=e['retrieved_utc'])
        return raw, receipt, e
    def test_nested_reviewed_spans(self):
        for key, cls in [("date", "xn-chron"), ("location", "xn-location")]:
            with self.subTest(key=key):
                raw, r, e = self.source()
                raw = ("<article class=\"news-release\"><span><span class=\"" + cls + "\">Evidence</span></span></article>").encode()
                e["source_sha256"] = prior.digest(raw)
                r.update(sha256=prior.digest(raw), bytes=len(raw))
                e["evidence"] = {key: dict(tag="span", html_line_1based=1, quote="Evidence", node_text_sha256=prior.digest(b"Evidence"))}
                self.assertTrue(verify_source(raw, r, e))
    def test_wrong_inner_class_rejected(self):
        raw, r, e = self.source()
        raw = b"<article class=\"news-release\"><span>Evidence</span></article>"
        e["source_sha256"] = prior.digest(raw)
        r.update(sha256=prior.digest(raw), bytes=len(raw))
        e["evidence"] = {"date": dict(tag="span", html_line_1based=1, quote="Evidence", node_text_sha256=prior.digest(b"Evidence"))}
        with self.assertRaises(ValueError): verify_source(raw, r, e)
    def test_source_exact(self):
        self.assertTrue(verify_source(*self.source()))
    def test_403_rejected(self):
        raw, r, e = self.source(); r['http_status'] = 403
        with self.assertRaises(ValueError): verify_source(raw, r, e)
    def test_hash_rejected(self):
        raw, r, e = self.source(); r['sha256'] = '0' * 64
        with self.assertRaises(ValueError): verify_source(raw, r, e)
    def test_wrong_url(self):
        raw, r, e = self.source(); r['final_url'] = 'https://example.test/other'
        with self.assertRaises(ValueError): verify_source(raw, r, e)
    def test_wrong_dom_line(self):
        raw, r, e = self.source(); e['evidence']['dateline']['html_line_1based'] = 2
        with self.assertRaises(ValueError): verify_source(raw, r, e)
    def test_wrong_dom_hash(self):
        raw, r, e = self.source(); e['evidence']['dateline']['node_text_sha256'] = '0' * 64
        with self.assertRaises(ValueError): verify_source(raw, r, e)
    def test_quote_outside_node(self):
        raw, r, e = self.source(); e['evidence']['dateline']['quote'] = 'Absent text'
        with self.assertRaises(ValueError): verify_source(raw, r, e)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    for name in ('self-test', 'freeze', 'build', 'verify'):
        group.add_argument('--' + name, action='store_true')
    parser.add_argument('--spec-sha')
    args = parser.parse_args()
    if args.self_test:
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
        raise SystemExit(not result.wasSuccessful())
    elif args.freeze:
        freeze()
    elif not args.spec_sha:
        parser.error('--spec-sha required')
    elif args.build:
        build(args.spec_sha)
    else:
        verify(args.spec_sha)
