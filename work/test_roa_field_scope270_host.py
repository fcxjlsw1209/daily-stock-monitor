"""Synthetic file packages only. Never open the actual census or financial sources."""
import copy
from datetime import date
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import roa_field_scope270_host as h
from test_roa_field_scope270_v2 import fixture


def put(root, name, value, pins):
    raw = value if isinstance(value, bytes) else h.encoded(value)
    p = root / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(raw)
    ref = {'path': name, 'sha256': h.digest(raw)}
    pins[name] = ref['sha256']
    return ref


def source_event(root, i, pins):
    kind = 'apple_shape' if i < 9 or i == 31 else 'calendar'
    e, _ = fixture(kind)
    e['event_id'] = 'SYNTHETIC:' + str(i)
    e['symbol'] = 'SYN' + str(i)
    e['identity']['security_id'] = e['symbol']
    e['identity']['issuer_id'] = 'synthetic-issuer-' + str(i)
    raw = put(root, 'filings/raw/source%d.bin' % i, ('fabricated source ' + str(i)).encode(), pins)
    receipt = dict(url='https://synthetic.example/report/' + str(i), http_status=200,
                   sha256=raw['sha256'], retrieved_at='2026-10-20T00:00:00+00:00')
    rr = put(root, 'filings/raw/source%d.meta.json' % i, receipt, pins)
    source = dict(raw, receipt=rr, url=receipt['url'])
    e['source'] = source
    e['cash_quality_admission'] = dict(admitted=i < 31, reasons=[] if i < 31 else ['inherited_hold'], definition='legacy')
    if i < 31:
        e.update(prior_field_admitted=True, inherited_rejection_codes=[], accounting_review_flags=[])
    if i >= 34:
        e['inherited_rejection_codes'] = ['unknown_hold']
    e['periods']['comparison']['period_start'] = None
    for side, field in h.scope.REQUIRED:
        c = e['fields'][side][field]
        c['source'] = copy.deepcopy(source)
        c['displayed_amount'] = c['value_decimal']
        c['parsed_cell_sha256'] = ['d' * 64]
        loc = c['locator']
        loc.update(statement_title='statement', header_literals=['Year ended', c['period']['period_end']])
        if loc['format'] == 'html':
            loc.update(logical_columns=[3], physical_cell_indices_zero_based=[3])
            c.update(normalized_monetary_literal=c['displayed_amount'], comma_whitespace_normalized=False)
            if i == 11 and field == 'total_assets':
                loc['statement_segment'] = dict(table_index_zero_based=loc['table_index_zero_based'],
                                               start_row=0, end_row_exclusive=100, title=None)
        else:
            loc['page_1based'] = 3 if field == 'net_income' else 2
            loc['financial_column_index_0based'] = 0 if side == 'current' else 1
    return e


def baseline_fields(fields):
    """Fabricate the actual old/new schema difference, not equal new cells."""
    out = copy.deepcopy(fields)
    for side, field in h.scope.REQUIRED:
        cell = out[side][field]
        if cell is not None and cell['locator']['format'] == 'html':
            cell.pop('normalized_monetary_literal')
            cell.pop('comma_whitespace_normalized')
            cell['locator'].pop('statement_segment', None)
    return out


def html_row(event):
    rows = []
    for side, field in h.scope.REQUIRED:
        c, loc = event['fields'][side][field], event['fields'][side][field]['locator']
        observed = dict(value_decimal=c['value_decimal'], scale=c['scale'], currency_literal='$',
                        table_index=loc['table_index_zero_based'], row_index=loc['row_index_zero_based'],
                        physical_cell_indices=loc['physical_cell_indices_zero_based'], logical_columns=loc['logical_columns'],
                        headers=loc['header_literals'], row_label=loc['row_label'], title=loc['statement_title'],
                        parsed_cell_sha256=c['parsed_cell_sha256'])
        rows.append(dict(side=side, field=field, matched=True, status='matched', failures=[],
                         independent_candidate_count=1, independent_source=observed))
    return dict(event_id=event['event_id'], source=copy.deepcopy(event['source']), cells=rows)


def repair_row(event):
    c = event['fields']['current']['net_income']; loc = c['locator']
    o = dict(value_decimal=c['value_decimal'], scale=c['scale'],
             table_index_zero_based=loc['table_index_zero_based'], row_index_zero_based=loc['row_index_zero_based'],
             period_end=loc['period_end'], physical_cell_index_zero_based=loc['physical_cell_indices_zero_based'][0],
             raw_markup_sha256=c['parsed_cell_sha256'][0], ni_start_row=loc['row_index_zero_based'])
    return dict(event_id=event['event_id'], side='current', field='net_income', matched=True,
                current_release=event['release_date'], available_from=event['available_from'],
                source=copy.deepcopy(event['source']), period=copy.deepcopy(c['period']), independent_reference=o)


def three_row(event, calendar):
    fields = {}
    for side, field in h.scope.REQUIRED:
        c = event['fields'][side][field]; loc = c['locator']; key = side + '.' + field
        o = {k: copy.deepcopy(loc[k]) for k in (
            ('page_1based', 'financial_column_index_0based') if loc['format'] == 'pdf' else
            ('table_index_zero_based', 'row_index_zero_based', 'physical_cell_indices_zero_based'))}
        o.update(value_decimal=c['value_decimal'], scale=c['scale'], currency_literal='$', currency_iso_certified=False,
                 as_of=c['period']['period_end'], row_label=loc['row_label'], header_literals=['observed header'],
                 period_kind='duration' if field == 'net_income' else 'instant')
        fields[key] = dict(original_field_path='fields.' + key, original_cell_sha256=h.scope.canonical_sha256(c),
                           source_path=event['source']['path'], source_sha256=event['source']['sha256'],
                           matched_true=True, same_report_source_and_available_from_match=True,
                           source_amount_sign_scale_and_endpoint_match=True, observed=o)
    pr = dict(event['periods']['current'], annual_header_observed=True, opening_is_current_start_minus_one_day=True,
              opening_assets_as_of=event['periods']['comparison']['period_end'], closing_assets_as_of=event['periods']['current']['period_end'])
    return dict(event_id=event['event_id'], source=copy.deepcopy(event['source']),
                original_event_sha256=h.scope.canonical_sha256(event),
                legacy_cash_quality_admission_sha256=h.scope.canonical_sha256(event['cash_quality_admission']),
                inherited_rejection_codes_retained=copy.deepcopy(event['inherited_rejection_codes']),
                clock=dict(release_date=event['release_date'], available_from=event['available_from'], calendar_ref=calendar,
                           first_pinned_session_strictly_after_release=True, first_publication_time_certified=False),
                current_period_review=pr, fields=fields)


def package(root):
    pins, refs, docs = {}, {}, {}
    def save(role, doc):
        refs[role] = put(root, 'metadata/' + role + '.json', doc, pins)
        docs[role] = doc
    events = [source_event(root, i, pins) for i in range(44)]
    save('calendar_plan', {'schema': 'synthetic_calendar_plan'})
    sessions = sorted(set(x for e in events for x in (e['release_date'], e['available_from'])))
    sessions += ['2026-12-31']
    save('sessions', dict(sessions=sessions, count=len(sessions), first=sessions[0], last=sessions[-1], plan_sha256=refs['calendar_plan']['sha256']))
    save('accounting', {'events': [dict(event_id=e['event_id'], no_new_clearance=True) for e in events]})
    for e, ac in zip(events, docs['accounting']['events']):
        frozen = baseline_fields(e['fields'])
        if e in events[9:11]:
            frozen['current'].pop('net_income')
        e['union_provenance'] = dict(accounting_event_review=ac, accounting_review_reference=refs['accounting'],
                                     frozen_v1_existing_cells=frozen)
    raw_events = [{k: copy.deepcopy(v) for k, v in e.items() if k != 'cash_quality_admission'} for e in events]
    save('baseline_plan', {'source_hashes': dict(pins)})
    baseline = copy.deepcopy(raw_events)
    for e in baseline:
        e['fields'] = baseline_fields(e['fields'])
    for i in (9, 10):
        baseline[i]['fields']['current'].pop('net_income')
    save('baseline_data', dict(events=baseline, plan_sha256=refs['baseline_plan']['sha256']))
    save('baseline_manifest', dict(plan_sha256=refs['baseline_plan']['sha256'], output_hashes={'RAW_FACTS.json': refs['baseline_data']['sha256']}))
    save('html_review', dict(passed=True, events=[html_row(e) for e in events if e['fields']['current']['net_income']['locator']['format'] == 'html'],
                            data_sha256=refs['baseline_data']['sha256'], plan_sha256=refs['baseline_plan']['sha256'],
                            source_manifest_sha256=refs['baseline_manifest']['sha256']))
    save('html_repair_data', {'events': raw_events})
    save('html_repair_review', dict(passed=True, checked_positions=[repair_row(events[i]) for i in (9, 10)],
                                   input_hashes={refs['html_repair_data']['path']: refs['html_repair_data']['sha256']}))
    save('pdf_data', {'events': raw_events})
    visual, checks = [], []
    for e in events:
        if e['fields']['current']['net_income']['locator']['format'] != 'pdf':
            continue
        image = put(root, 'pixels/' + e['symbol'] + '.png', b'synthetic pixel fixture', pins)
        visual.append(dict(event_id=e['event_id'], source=e['source'],
                           inspected_images=[dict(image, page_1based=2), dict(image, page_1based=3)],
                           column_order=['current', 'comparison'], monetary_scale='millions',
                           period_end_dates=[e['periods'][s]['period_end'] for s in ('current', 'comparison')],
                           balance_sheet_page_1based=2, cash_flow_page_1based=3,
                           visually_observed_total_assets=[120, 120],
                           visually_observed_cash_flow_starting_net_income=[120, 120]))
        for side, field in h.scope.REQUIRED:
            c = e['fields'][side][field]
            checks.append(dict(event_id=e['event_id'], side=side, field=field, visual_match=True,
                               value_decimal=c['value_decimal'], displayed_amount=c['displayed_amount']))
    save('pdf_visual', dict(events=visual, plan_sha256=refs['baseline_plan']['sha256']))
    save('pdf_review', dict(passed=True, checks=checks, input_hashes={refs['pdf_data']['path']: refs['pdf_data']['sha256'],
                                                                  refs['pdf_visual']['path']: refs['pdf_visual']['sha256']}))
    save('union_plan', {'source_hashes': {e['source']['path']: e['source']['sha256'] for e in events}})
    save('union_data', dict(events=raw_events, plan_sha256=refs['union_plan']['sha256']))
    save('union_manifest', dict(plan_sha256=refs['union_plan']['sha256'], output_hashes={'RAW_FACTS.json': refs['union_data']['sha256']}))
    save('union_review', dict(passed=True, data_sha256=refs['union_data']['sha256'], plan_sha256=refs['union_plan']['sha256'],
                             source_manifest_sha256=refs['union_manifest']['sha256'],
                             input_hashes={refs[k]['path']: refs[k]['sha256'] for k in ('union_data', 'union_plan', 'union_manifest')}))
    old_refs = {k: refs[k] for k in ('union_data', 'union_plan', 'union_manifest', 'sessions', 'accounting')}
    old_refs['union_independent'] = refs['union_review']
    legacy_code = put(root, 'work/cash_quality_source_admission_v1.py', b'# fabricated legacy implementation\n', pins)
    old_refs['tests'] = put(root, 'metadata/legacy_tests.json', dict(passed=True, source_sha256=legacy_code['sha256']), pins)
    legacy_review = put(root, 'metadata/legacy_code_review.json', dict(passed=True, source_hashes={legacy_code['path']: legacy_code['sha256']}), pins)
    save('legacy_policy', dict(inputs=old_refs, implementation_sha256=legacy_code['sha256'], independent_code_review=legacy_review))
    save('dataset', dict(events=events, policy_sha256=refs['legacy_policy']['sha256'], method='synthetic_legacy'))
    save('legacy_admission', dict(passed=True, policy_sha256=refs['legacy_policy']['sha256'], data_sha256=refs['dataset']['sha256'],
                                 source_hashes={r['path']: r['sha256'] for r in old_refs.values()}, events=44, admitted_events=31,
                                 held_events=13, admitted_event_ids=[e['event_id'] for e in events[:31]],
                                 holds=[dict(event_id=e['event_id'], **e['cash_quality_admission']) for e in events[31:]]))
    save('legacy_review', dict(passed=True, policy_sha256=refs['legacy_policy']['sha256'], data_sha256=refs['dataset']['sha256'],
                              admitted_events=31, held_events=13))
    save('three_review', dict(passed=True, stage='independent_three_event_raw_source_cell_fact_review_not_admission',
                             original_dataset=dict(path=refs['dataset']['path'], byte_sha256=refs['dataset']['sha256']),
                             events=[three_row(e, refs['sessions']['path']) for e in events[31:34]],
                             input_sha256={r['path']: r['sha256'] for r in
                                           [refs['dataset'], refs['sessions']] +
                                           [r for e in events[31:34] for r in
                                            (e['source'], e['source']['receipt'])]}))
    codes = {}
    for name in h.CODE_FILES:
        codes[name] = put(root, name, (h.ROOT / name).read_bytes(), pins)['sha256']
    tests = dict(passed=True, synthetic_only=True, code_sha256=codes, tests=1, failures=0, errors=0,
                 terminal=dict(exit_code=0, chunk_id='fabricated-test-terminal'))
    testref = put(root, 'metadata/tests.json', tests, pins)
    policy = dict(schema=h.SCHEMA, source_only=True, output_directory=h.OUT, expected_events=44, expected_old_admitted=31,
                  inputs=refs, input_sha256=dict(pins), code_sha256=codes,
                  runtime=dict(python=sys.version, executable_sha256=h.digest(Path(sys.executable).read_bytes())), tests=testref)
    pref = put(root, 'metadata/policy.json', policy, pins)
    return policy, pref, docs, h.Evidence(root, pins)


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.p, self.ref, self.docs, self.ev = package(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_full_synthetic44_writeonce_and_preservation(self):
        before = copy.deepcopy(self.docs['dataset'])
        r = h.run(self.root, self.ref['path'], self.ref['sha256'])
        self.assertTrue(r['passed'])
        self.assertEqual((r['events'], r['admitted_events'], r['held_events']), (44, 34, 10))
        out = json.loads((self.root / h.OUT / 'DATASET.json').read_text())
        for e, original in zip(out['events'], before['events']):
            self.assertEqual({k: v for k, v in e.items() if not k.startswith('roa_field_scope270')}, original)
        self.assertEqual(self.docs['dataset'], before)
        manifest = json.loads((self.root / h.OUT / 'MANIFEST.json').read_text())
        for name, sha in manifest['output_sha256'].items():
            self.assertEqual(h.digest((self.root / h.OUT / name).read_bytes()), sha)
        with self.assertRaisesRegex(ValueError, 'Write-once'):
            h.run(self.root, self.ref['path'], self.ref['sha256'])

    def test_bound_input_tamper_rejected_before_assess(self):
        raw = self.root / self.docs['dataset']['events'][0]['source']['path']
        raw.write_bytes(b'changed')
        with patch.object(h.scope, 'assess', side_effect=AssertionError('must not run')):
            with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
                h.run(self.root, self.ref['path'], self.ref['sha256'])
        self.assertFalse((self.root / h.OUT).exists())

    def test_policy_exact_role_and_code_inventory(self):
        for group, key in (('inputs', 'three_review'), ('code_sha256', h.CODE_FILES[1])):
            p = copy.deepcopy(self.p); del p[group][key]
            with self.assertRaises(ValueError):
                h.policy_check(p, self.ev)

    def test_runtime_and_synthetic_receipt_are_required(self):
        p = copy.deepcopy(self.p); p['runtime']['python'] = 'other'
        with self.assertRaisesRegex(ValueError, 'runtime'):
            h.policy_check(p, self.ev)
        tests = self.ev.read(self.p['tests']); tests['terminal']['exit_code'] = 1
        ref = put(self.root, 'metadata/badtests.json', tests, self.ev.pins)
        p = copy.deepcopy(self.p); p['tests'] = ref
        with self.assertRaisesRegex(ValueError, 'terminal'):
            h.policy_check(p, self.ev)

    def test_full_raw_event_exact_not_merely_needed_fields(self):
        d = copy.deepcopy(self.docs)
        d['dataset']['events'][0]['unused'] = 'tamper'
        with self.assertRaises(ValueError):
            h.verify_package(self.p, d, self.ev)

    def test_legacy_admission_union_and_calendar_link_failures(self):
        changes = (
            lambda d: d['legacy_admission'].update(data_sha256='0' * 64),
            lambda d: d['union_review'].update(data_sha256='0' * 64),
            lambda d: d['union_manifest']['output_hashes'].update({'RAW_FACTS.json': '0' * 64}),
            lambda d: d['legacy_review'].update(passed=False),
            lambda d: d['sessions'].update(plan_sha256='0' * 64),
            lambda d: d['dataset']['events'].reverse(),
            lambda d: d['dataset']['events'].pop(),
            lambda d: d['accounting']['events'][0].update(no_new_clearance=False),
        )
        for change in changes:
            d = copy.deepcopy(self.docs); change(d)
            with self.assertRaises(ValueError):
                h.verify_package(self.p, d, self.ev)

    def test_embedded_review_input_not_in_policy_rejected(self):
        d = copy.deepcopy(self.docs)
        d['three_review']['input_sha256']['unbound.file'] = 'a' * 64
        with self.assertRaisesRegex(ValueError, 'omitted'):
            h.verify_package(self.p, d, self.ev)

    def test_legacy_source_code_tests_and_review_explicitly_linked(self):
        d = copy.deepcopy(self.docs)
        d['legacy_policy']['implementation_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'Conflicting policy pin'):
            h.verify_package(self.p, d, self.ev)
        ref = d['legacy_policy']['inputs']['tests']
        d = copy.deepcopy(self.docs)
        bad = put(self.root, 'metadata/legacy_badtests.json', dict(passed=True, source_sha256='0' * 64), self.ev.pins)
        d['legacy_policy']['inputs']['tests'] = bad
        with self.assertRaisesRegex(ValueError, 'implementation/tests'):
            h.verify_package(self.p, d, self.ev)

    def test_new_review_raw_or_cell_missing_never_pretends_reviewed(self):
        e = self.docs['dataset']['events'][31]
        for change in (lambda r: r['fields'].pop('comparison.total_assets'),
                       lambda r: r['fields']['current.net_income'].update(source_sha256='0' * 64),
                       lambda r: r['fields']['current.net_income'].update(matched_true=False)):
            row = copy.deepcopy(self.docs['three_review']['events'][0]); change(row)
            with self.assertRaises(ValueError):
                h.new_review_event(e, row, self.p['inputs'])
        d = copy.deepcopy(self.docs)
        del d['three_review']['input_sha256'][e['source']['path']]
        with self.assertRaisesRegex(ValueError, 'raw/receipt input binding'):
            h.verify_package(self.p, d, self.ev)

    def test_old_admitted_regression_reported_not_forced_to_pass(self):
        events, sessions, new = h.verify_package(self.p, self.docs, self.ev)
        real = h.scope.assess
        def reject_one(e, sessions, review):
            result = real(e, sessions, review)
            if e['event_id'] == 'SYNTHETIC:0':
                result['admitted'] = False
                result['reasons'] = ['synthetic explicit regression']
            return result
        with patch.object(h.scope, 'assess', side_effect=reject_one):
            _, results, bad = h.assess_census(events, sessions, new, self.docs, self.p['inputs'], self.ev)
        self.assertEqual(bad, [dict(event_id='SYNTHETIC:0', reasons=['synthetic explicit regression'])])
        self.assertFalse(results[0]['admitted'])

    def test_missing_review_retains_held_and_unknown_codes(self):
        events, sessions, new = h.verify_package(self.p, self.docs, self.ev)
        rows, results, _ = h.assess_census(events, sessions, new, self.docs, self.p['inputs'], self.ev)
        self.assertEqual(sum(not r['review_available'] for r in results), 10)
        for e in rows[34:]:
            self.assertFalse(e['roa_field_scope270']['admitted'])
            self.assertEqual(e['inherited_rejection_codes'], ['unknown_hold'])
            self.assertFalse(e['roa_field_scope270']['chain_admitted'])


class FieldMappingTests(unittest.TestCase):
    setUp = PackageTests.setUp
    tearDown = PackageTests.tearDown

    def test_html_baseline_amount_scale_physical_header_source_end(self):
        event = self.docs['dataset']['events'][12]
        self.assertEqual(h.html_baseline_cell(event, 'current', 'net_income', self.docs), 'html_review')
        changes = (lambda r: r.update(value_decimal='121'), lambda r: r.update(scale='1'),
                   lambda r: r.update(table_index=99), lambda r: r.update(row_index=99),
                   lambda r: r.update(physical_cell_indices=[99]), lambda r: r.update(headers=['wrong']),
                   lambda r: r.update(parsed_cell_sha256=['0' * 64]))
        for change in changes:
            d = copy.deepcopy(self.docs)
            row = h.one(d['html_review']['events'], event_id=event['event_id'])
            change(h.one(row['cells'], side='current', field='net_income')['independent_source'])
            with self.assertRaises(ValueError):
                h.html_baseline_cell(event, 'current', 'net_income', d)

    def test_repair_uses_exact_repaired_cell_not_baseline_flag(self):
        e = self.docs['dataset']['events'][9]
        self.assertEqual(h.html_repair_cell(e, 'current', 'net_income', self.docs), 'html_repair_review')
        for key, value in (('value_decimal', '1'), ('period_end', '2000-01-01'),
                           ('raw_markup_sha256', '0' * 64), ('physical_cell_index_zero_based', 99)):
            d = copy.deepcopy(self.docs); d['html_repair_review']['checked_positions'][0]['independent_reference'][key] = value
            with self.assertRaises(ValueError):
                h.html_repair_cell(e, 'current', 'net_income', d)

    def test_missing_legacy_key_routes_only_to_bound_repair(self):
        for i in (9, 10):
            e = self.docs['dataset']['events'][i]
            self.assertNotIn('net_income', e['union_provenance']['frozen_v1_existing_cells']['current'])
            baseline = h.one(self.docs['baseline_data']['events'], event_id=e['event_id'])
            self.assertNotIn('net_income', baseline['fields']['current'])
            proof, mapping = h.field_review(e, self.docs, self.p['inputs'], {})
            row = h.one(mapping['field_mapping'], field='current.net_income')
            self.assertEqual(row['route'], 'html_repair_review')
            self.assertIn(self.p['inputs']['html_repair_data'], proof['review_receipts'])
            self.assertIn(self.p['inputs']['html_repair_review'], proof['review_receipts'])
            changed = copy.deepcopy(e)
            changed['fields']['current']['net_income']['value_decimal'] = '121'
            with self.assertRaisesRegex(ValueError, 'Complete source cell'):
                h.field_review(changed, self.docs, self.p['inputs'], {})

    def test_baseline_route_binds_both_complete_cell_versions(self):
        e = self.docs['dataset']['events'][12]
        old = h.one(self.docs['baseline_data']['events'], event_id=e['event_id'])['fields']['current']['net_income']
        current = e['fields']['current']['net_income']
        self.assertEqual(set(current) - set(old), {'normalized_monetary_literal', 'comma_whitespace_normalized'})
        self.assertEqual(old, e['union_provenance']['frozen_v1_existing_cells']['current']['net_income'])
        self.assertEqual(h.html_baseline_cell(e, 'current', 'net_income', self.docs), 'html_review')
        proof, mapping = h.field_review(e, self.docs, self.p['inputs'], {})
        for role in ('baseline_data', 'html_repair_data', 'html_review'):
            self.assertIn(self.p['inputs'][role], proof['review_receipts'])
            self.assertIn(self.p['inputs'][role], mapping['review_sources'])
        d = copy.deepcopy(self.docs)
        h.one(d['html_repair_data']['events'], event_id=e['event_id'])['fields']['current']['net_income']['fact_admitted'] = True
        with self.assertRaisesRegex(ValueError, 'Complete source cell'):
            h.html_baseline_cell(e, 'current', 'net_income', d)

    def test_normalization_annotation_tamper_even_when_repair_matches(self):
        for key, value in (('normalized_monetary_literal', '121'), ('comma_whitespace_normalized', True),
                           ('comma_whitespace_normalized', 0)):
            d = copy.deepcopy(self.docs); e = d['dataset']['events'][12]
            e['fields']['current']['net_income'][key] = value
            h.one(d['html_repair_data']['events'], event_id=e['event_id'])['fields'] = copy.deepcopy(e['fields'])
            with self.assertRaisesRegex(ValueError, 'normalization annotations'):
                h.html_baseline_cell(e, 'current', 'net_income', d)

    def test_unknown_extension_and_changed_old_keys_remain_rejected(self):
        for change in (lambda c: c.update(unreviewed_extension=True),
                       lambda c: c.pop('normalized_monetary_literal'),
                       lambda c: c.update(currency_iso='USD'),
                       lambda c: c['locator'].update(unreviewed_locator=True)):
            d = copy.deepcopy(self.docs); e = d['dataset']['events'][12]
            change(e['fields']['current']['net_income'])
            h.one(d['html_repair_data']['events'], event_id=e['event_id'])['fields'] = copy.deepcopy(e['fields'])
            with self.assertRaises(ValueError):
                h.html_baseline_cell(e, 'current', 'net_income', d)

    def test_statement_segment_extension_bound_and_old_locator_preserved(self):
        e = self.docs['dataset']['events'][11]
        for side in ('current', 'comparison'):
            self.assertEqual(h.html_baseline_cell(e, side, 'total_assets', self.docs), 'html_review')
        for key, value in (('start_row', 999), ('end_row_exclusive', 0), ('table_index_zero_based', 999),
                           ('title', 1), ('extra', True)):
            d = copy.deepcopy(self.docs); e = d['dataset']['events'][11]
            e['fields']['current']['total_assets']['locator']['statement_segment'][key] = value
            h.one(d['html_repair_data']['events'], event_id=e['event_id'])['fields'] = copy.deepcopy(e['fields'])
            with self.assertRaisesRegex(ValueError, 'statement segment'):
                h.html_baseline_cell(e, 'current', 'total_assets', d)

    def test_baseline_provenance_tamper_is_not_hidden_by_repair_binding(self):
        e = copy.deepcopy(self.docs['dataset']['events'][12])
        e['union_provenance']['frozen_v1_existing_cells']['current']['net_income']['value_decimal'] = '121'
        with self.assertRaisesRegex(ValueError, 'Frozen baseline'):
            h.html_baseline_cell(e, 'current', 'net_income', self.docs)

    def test_pdf_visual_full_cell_binding_and_positional_evidence(self):
        e = self.docs['dataset']['events'][0]
        self.assertEqual(h.pdf_cell(e, 'current', 'net_income', self.docs), 'pdf_review')
        for key, value in (('visually_observed_cash_flow_starting_net_income', [121, 120]),
                           ('period_end_dates', ['1900-01-01', '2015-09-26']), ('cash_flow_page_1based', 99),
                           ('monetary_scale', 'thousands'), ('column_order', ['comparison', 'current'])):
            d = copy.deepcopy(self.docs); d['pdf_visual']['events'][0][key] = value
            with self.assertRaises(ValueError):
                h.pdf_cell(e, 'current', 'net_income', d)
        altered = copy.deepcopy(e); altered['fields']['current']['net_income']['locator']['line_1based'] += 1
        with self.assertRaisesRegex(ValueError, 'Complete source cell'):
            h.pdf_cell(altered, 'current', 'net_income', self.docs)

    def test_three_event_complete_hash_and_observed_physical_binding(self):
        e = self.docs['dataset']['events'][31]
        row = self.docs['three_review']['events'][0]
        h.new_review_event(e, row, self.p['inputs'])
        mutations = (lambda r: r.update(original_event_sha256='0' * 64),
                     lambda r: r['fields']['current.net_income'].update(original_cell_sha256='0' * 64),
                     lambda r: r['fields']['current.net_income']['observed'].update(value_decimal='121'),
                     lambda r: r['fields']['current.net_income']['observed'].update(page_1based=99),
                     lambda r: r['current_period_review'].update(days_inclusive=365),
                     lambda r: r['clock'].update(available_from='2000-01-01'))
        for change in mutations:
            r = copy.deepcopy(row); change(r)
            with self.assertRaises(ValueError):
                h.new_review_event(e, r, self.p['inputs'])

    def test_duplicate_or_wrong_side_review_cannot_satisfy_field(self):
        e = self.docs['dataset']['events'][12]
        d = copy.deepcopy(self.docs)
        row = h.one(d['html_review']['events'], event_id=e['event_id'])
        row['cells'].append(copy.deepcopy(row['cells'][0]))
        with self.assertRaisesRegex(ValueError, 'Exactly one'):
            h.html_baseline_cell(e, 'current', 'net_income', d)


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name); self.pins = {}
        self.event = source_event(self.root, 0, self.pins)

    def tearDown(self):
        self.temp.cleanup()

    def check(self, source=None):
        return h.verify_receipt(source or self.event['source'], h.Evidence(self.root, self.pins))

    def change_receipt(self, changes):
        source = copy.deepcopy(self.event['source'])
        receipt = json.loads((self.root / source['receipt']['path']).read_text()); receipt.update(changes)
        source['receipt'] = put(self.root, source['receipt']['path'], receipt, self.pins)
        return source

    def test_three_retrieval_formats_and_entry_key(self):
        self.assertTrue(self.check()['raw_bytes_verified'])
        source = copy.deepcopy(self.event['source'])
        old = json.loads((self.root / source['receipt']['path']).read_text())
        source['receipt'] = put(self.root, 'filings/manifest.json', {'entry': old}, self.pins)
        source['receipt_entry_key'] = 'entry'
        self.assertTrue(self.check(source)['receipt_bytes_verified'])
        source = copy.deepcopy(self.event['source'])
        old['status'] = old.pop('http_status'); old['response_sha256'] = old.pop('sha256')
        old['response_path'] = 'raw/source0.bin'
        source['receipt'] = put(self.root, source['receipt']['path'], old, self.pins)
        self.assertTrue(self.check(source)['raw_bytes_verified'])

    def test_receipt_status_url_timestamp_and_content_digest(self):
        for change in ({'http_status': 403}, {'http_status': True}, {'url': 'https://other.example/'},
                       {'retrieved_at': None}, {'retrieved_at': '2020-01-01'}, {'sha256': '0' * 64}):
            with self.subTest(change=change):
                source = self.change_receipt(change)
                with self.assertRaises((ValueError, KeyError)):
                    self.check(source)

    def test_raw_or_receipt_byte_tampering_and_absent_entry(self):
        source = copy.deepcopy(self.event['source']); source['receipt_entry_key'] = 'absent'
        with self.assertRaisesRegex(ValueError, 'Missing retrieval'):
            self.check(source)
        (self.root / source['path']).write_bytes(b'tamper')
        with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
            self.check()

    def test_path_escape_and_symlink_rejected(self):
        ev = h.Evidence(self.root, self.pins)
        for path in ('../escape', '/etc/passwd', 'CloudStorage/file'):
            with self.assertRaises(ValueError):
                ev.path(path)
        link = self.root / 'link'; link.symlink_to(self.root / self.event['source']['path'])
        with self.assertRaises(ValueError):
            ev.path('link')

    def test_duplicate_json_key_rejected(self):
        ref = put(self.root, 'duplicate.json', b'{"x":1,"x":2}', self.pins)
        with self.assertRaisesRegex(ValueError, 'Duplicate JSON'):
            h.Evidence(self.root, self.pins).read(ref)

    def test_output_child_symlink_rejected_after_root_canonicalization(self):
        with tempfile.TemporaryDirectory() as outside:
            (self.root / 'outputs').symlink_to(Path(outside).resolve())
            with self.assertRaisesRegex(ValueError, 'symlink'):
                h.write_outputs(self.root, {}, {'passed': True}, {}, {})
            self.assertEqual(list(Path(outside).iterdir()), [])

    def test_collection_relative_response_path_is_exact(self):
        source = copy.deepcopy(self.event['source'])
        receipt = json.loads((self.root / source['receipt']['path']).read_text())
        receipt['status'] = receipt.pop('http_status')
        receipt['response_sha256'] = receipt.pop('sha256')
        receipt['response_path'] = 'raw/wrong.bin'
        source['receipt'] = put(self.root, source['receipt']['path'], receipt, self.pins)
        with self.assertRaisesRegex(ValueError, 'response path mismatch'):
            self.check(source)


if __name__ == '__main__':
    unittest.main()
