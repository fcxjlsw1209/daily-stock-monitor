"""Synthetic coverage-overlay tests. Never read real bodies or build COVERAGE.json."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import after157_annual_body_coverage as c


def targets_and_events():
    targets, events = {}, {}
    for provider, (symbol, cik, host) in c.ISSUERS.items():
        for year in ('old', 'new'):
            for kind in ('page', 'body'):
                key = provider + '_' + year + '_' + kind
                target = dict(id=key, provider=provider, url='https://' + host + '/' + key,
                              body_file=key + '.html', receipt_file=key + '.receipt.json',
                              role='original_annual_body_candidate' if kind == 'body' else 'publisher_page')
                if kind == 'body':
                    event = dict(event_id=provider + '_' + year, cik=cik, form='10-K',
                                 accessionNumber=key, reportDate=year, filingDate=year,
                                 primaryDocument=key + '.htm')
                    events[event['event_id']] = event
                    target['expected_original'] = copy.deepcopy(event)
                targets[key] = target
    return targets, events


def review_fixture(provider, targets):
    own = [t for t in targets.values() if t['provider'] == provider and 'expected_original' in t]
    return dict(target_ids=[t['id'] for t in own],
                expected_event_ids=[t['expected_original']['event_id'] for t in own],
                exact_original_versions_bound=True, normal_consecutive_pair_context_verified=True,
                source_admission_changed=False, input_hashes={'fixture': '0' * 64})


class JoinTest(unittest.TestCase):
    def setUp(self):
        self.targets, self.events = targets_and_events()
        self.documents = c.documents_from_scope(self.targets, self.events)
        self.reviews = {}
        for provider in c.ISSUERS:
            review = review_fixture(provider, self.targets)
            if provider == 'microsoft':
                review['exact_original_versions_bound'] = False
                review['normal_consecutive_pair_context_verified'] = False
            self.reviews[provider] = c.validate_review(review, provider, self.targets, review['input_hashes'])
        self.rows = [dict(signal_index=i, candidate_index=j, signal_date='S%03d' % i,
                          symbol='S%02d' % j, cik='other', is_etf=j >= 32,
                          selected_event_ids=[], tier2=False, tier3=False,
                          reasons=['ORIGINAL_BLOCKER'], existing_accession_context=[])
                     for i in range(122) for j in range(49)]
        for index, provider, tier2 in [(0, 'coca_cola', True), (1, 'microsoft', True),
                                       (88 * 49, 'coca_cola', True), (89 * 49, 'coca_cola', False)]:
            self.rows[index].update(symbol=c.ISSUERS[provider][0], cik=c.ISSUERS[provider][1],
                selected_event_ids=[provider + '_new', provider + '_old'], tier2=tier2,
                existing_accession_context=[{'prior141_blockers': ['original_context_unverified']}])

    def overlay(self):
        return c.overlay_rows(self.rows, self.events, self.documents, self.reviews)

    def test_full_grid_and_old_flags_preserved_without_mutation(self):
        before = copy.deepcopy(self.rows)
        output = self.overlay()
        self.assertEqual(len(output), 5978)
        self.assertEqual(self.rows, before)
        for old, new in zip(self.rows, output):
            self.assertEqual(new['selected_event_ids'], old['selected_event_ids'])
            self.assertEqual(new['prior_tier2'], old['tier2'])
            self.assertEqual(new['prior_tier3'], old['tier3'])
            self.assertEqual(new['prior_reasons'], old['reasons'])
            self.assertEqual(new['prior_existing_accession_context'], old['existing_accession_context'])
            self.assertTrue(all(r in new['still_pending_reasons'] for r in old['reasons']))
        self.assertTrue(output[0]['body_pair_context_verified'])
        self.assertFalse(output[1]['body_pair_context_verified'])
        self.assertFalse(any(r['prior_tier3'] for r in output))

    def test_exact_split_counts_and_body_context_does_not_imply_tier2(self):
        output = self.overlay()
        stats = c.summarize(output, self.documents)
        self.assertEqual(stats['all122']['candidate_slots'], 5978)
        self.assertEqual(stats['all122']['stock_slots'], 3904)
        self.assertEqual(stats['all122']['etf_slots'], 2074)
        self.assertEqual(stats['all122']['documented_pair_slots'], 3)
        self.assertEqual(stats['all122']['tier2_with_documented_pair_slots'], 2)
        self.assertEqual(stats['first88']['documented_pair_slots'], 1)
        self.assertEqual(stats['later34']['documented_pair_slots'], 2)
        self.assertEqual(stats['all122']['both_document_match_slots'], 4)

    def test_one_document_or_same_issuer_other_year_does_not_match_pair(self):
        self.rows[0]['selected_event_ids'] = ['coca_cola_new']
        self.assertFalse(self.overlay()[0]['body_pair_context_verified'])
        self.events['other_year'] = dict(self.events['coca_cola_old'], event_id='other_year')
        self.rows[0]['selected_event_ids'] = ['coca_cola_new', 'other_year']
        self.assertFalse(self.overlay()[0]['body_pair_context_verified'])

    def test_bad_latest_is_not_replaced_and_pair_order_is_preserved(self):
        self.events['bad_latest'] = dict(self.events['coca_cola_new'], event_id='bad_latest')
        self.rows[0]['selected_event_ids'] = ['bad_latest', 'coca_cola_new']
        out = self.overlay()[0]
        self.assertEqual(out['selected_event_ids'], ['bad_latest', 'coca_cola_new'])
        self.assertFalse(out['body_pair_context_verified'])

    def test_wrong_issuer_join_rejected(self):
        self.rows[1]['selected_event_ids'] = ['coca_cola_new', 'coca_cola_old']
        with self.assertRaisesRegex(ValueError, 'wrong issuer'):
            self.overlay()

    def test_unknown_duplicate_selected_ids_rejected(self):
        for ids in (['unknown'], ['coca_cola_new', 'coca_cola_new']):
            self.rows[0]['selected_event_ids'] = ids
            with self.assertRaisesRegex(ValueError, 'selected event'):
                self.overlay()

    def test_missing_duplicate_or_reordered_grid_rejected(self):
        original = copy.deepcopy(self.rows)
        for rows in (original[:-1], [original[1], original[0]] + original[2:],
                     [original[0], original[0]] + original[2:]):
            self.rows = rows
            with self.assertRaises(ValueError):
                self.overlay()

    def test_etf_cannot_gain_filing_events(self):
        self.rows[32]['selected_event_ids'] = ['coca_cola_new']
        with self.assertRaisesRegex(ValueError, 'ETF acquired'):
            self.overlay()

    def test_event_metadata_and_cik_must_match_scope(self):
        for field in ('accessionNumber', 'filingDate', 'reportDate', 'primaryDocument', 'cik'):
            changed = copy.deepcopy(self.events)
            changed['coca_cola_old'][field] = 'wrong'
            with self.assertRaisesRegex(ValueError, 'differs from prior'):
                c.documents_from_scope(self.targets, changed)

    def test_review_must_have_true_boolean_types(self):
        review = review_fixture('coca_cola', self.targets)
        for field in ('exact_original_versions_bound', 'normal_consecutive_pair_context_verified',
                      'source_admission_changed'):
            changed = dict(review, **{field: 'false'})
            with self.assertRaisesRegex(ValueError, 'must be boolean'):
                c.validate_review(changed, 'coca_cola', self.targets, review['input_hashes'])

    def test_review_cannot_claim_changed_admission(self):
        review = review_fixture('coca_cola', self.targets)
        review['source_admission_changed'] = True
        with self.assertRaisesRegex(ValueError, 'must not change'):
            c.validate_review(review, 'coca_cola', self.targets, review['input_hashes'])

    def test_wrong_review_pair_or_other_issuer_targets_rejected(self):
        review = review_fixture('coca_cola', self.targets)
        bad = copy.deepcopy(review)
        bad['expected_event_ids'][0] = 'microsoft_old'
        with self.assertRaisesRegex(ValueError, 'event IDs mismatch'):
            c.validate_review(bad, 'coca_cola', self.targets, review['input_hashes'])
        bad = copy.deepcopy(review)
        bad['target_ids'].append('microsoft_old_body')
        with self.assertRaisesRegex(ValueError, 'targets differ'):
            c.validate_review(bad, 'coca_cola', self.targets, review['input_hashes'])

    def test_conflicting_alias_or_unbound_review_rejected(self):
        review = review_fixture('coca_cola', self.targets)
        bad = copy.deepcopy(review)
        bad['reviewed_event_ids'] = ['other']
        with self.assertRaisesRegex(ValueError, 'Conflicting review aliases'):
            c.validate_review(bad, 'coca_cola', self.targets, review['input_hashes'])
        with self.assertRaisesRegex(ValueError, 'does not bind'):
            c.validate_review(review, 'coca_cola', self.targets, {'unbound': '0' * 64})


class ClosureTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.targets, _ = targets_and_events()
        helper = self.put(c.COLLECTOR, b'synthetic collector, never executed')
        extractor = self.put(c.EXTRACTOR, b'synthetic extractor, never executed')
        pin = self.put('work/fixture.json', b'{}')
        self.scope = dict(schema=c.SCHEMA, output_dir=c.OUT, helper_sha256=helper,
                          input_hashes={'work/fixture.json': pin}, targets=list(self.targets.values()))
        binding = dict(schema=c.SCHEMA, scope_sha256=c.SCOPE_SHA, helper_sha256=helper)
        self.put_json(c.OUT + '/.collection-scope.json', binding)
        self.collection = dict(binding, all_successful=True, stopped_providers=[], targets=[])
        self.derived = dict(scope_sha256=c.SCOPE_SHA, collection_sha256=c.COLLECTION_SHA,
                            extractor_sha256=extractor, admission=False, factor_calculations=0, records=[])
        for key, target in self.targets.items():
            raw_sha = self.put(c.OUT + '/' + target['body_file'], b'synthetic raw ' + key.encode())
            text_sha = self.put(c.OUT + '/derived_text/' + key + '.txt', b'synthetic text\n')
            tbind = dict(binding, id=key, provider=target['provider'], source_url=target['url'],
                         body_file=target['body_file'])
            attempt = self.put_json(c.OUT + '/.attempt_' + key + '.json', tbind)
            receipt = dict(tbind, attempted=True, outcome='success', http_status=200, stop_provider=False,
                           body_truncated_at_limit=False, final_url=target['url'], redirect_chain=[],
                           attempt_sha256=attempt, sha256=raw_sha, bytes=len(b'synthetic raw ' + key.encode()))
            receipt_sha = self.put_json(c.OUT + '/' + target['receipt_file'], receipt)
            self.collection['targets'].append(dict(id=key, outcome='success', receipt_file=target['receipt_file'],
                                                   receipt_sha256=receipt_sha))
            self.derived['records'].append(dict(id=key, outcome='success', http_status=200, role=target['role'],
                raw_sha256=raw_sha, receipt_sha256=receipt_sha, text_file=key + '.txt', text_sha256=text_sha, lines=1))

    def put(self, name, raw):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return c.sha(raw)

    def put_json(self, name, obj):
        return self.put(name, json.dumps(obj, sort_keys=True).encode())

    def verify(self):
        return c.verify_collection(c.Reader(self.root), self.scope, self.collection, self.derived)

    def test_all_eight_synthetic_artifact_groups_verified(self):
        targets, files = self.verify()
        self.assertEqual(len(targets), 8)
        self.assertEqual(sum(len(f) for f in files.values()), 32)

    def test_raw_receipt_attempt_and_text_tampering_each_rejected(self):
        target = next(iter(self.targets.values()))
        for name in (c.OUT + '/' + target['body_file'], c.OUT + '/' + target['receipt_file'],
                     c.OUT + '/.attempt_' + target['id'] + '.json',
                     c.OUT + '/derived_text/' + target['id'] + '.txt'):
            path = self.root / name
            old = path.read_bytes()
            path.write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
                self.verify()
            path.write_bytes(old)

    def test_forged_receipt_cannot_change_original_url(self):
        target = next(iter(self.targets.values()))
        path = c.OUT + '/' + target['receipt_file']
        receipt = json.loads((self.root / path).read_bytes())
        receipt['source_url'] = 'https://www.sec.gov/forged'
        new_sha = self.put_json(path, receipt)
        self.collection['targets'][0]['receipt_sha256'] = new_sha
        self.derived['records'][0]['receipt_sha256'] = new_sha
        with self.assertRaisesRegex(ValueError, 'Bad collection receipt'):
            self.verify()

    def test_wrong_text_to_raw_binding_and_missing_manifest_member_rejected(self):
        self.derived['records'][0]['raw_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'Text/raw'):
            self.verify()
        self.derived['records'].pop()
        with self.assertRaisesRegex(ValueError, 'Target set'):
            self.verify()

    def test_reader_rejects_wrong_pin_symlink_escape_and_later_mutation(self):
        reader = c.Reader(self.root)
        with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
            reader.read('work/fixture.json', '0' * 64)
        for name in ('../escape', '/outside/file', 'work/../fixture'):
            with self.assertRaises(ValueError):
                reader.path(name)
        (self.root / 'work/link').symlink_to(self.root / 'work/fixture.json')
        with self.assertRaisesRegex(ValueError, 'Symlink'):
            reader.path('work/link')
        reader.read('work/fixture.json')
        self.put('work/fixture.json', b'changed')
        with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
            reader.reverify()


if __name__ == '__main__':
    unittest.main(verbosity=2)
