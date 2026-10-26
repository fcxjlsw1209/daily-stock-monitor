"""Synthetic-only publisher tests. Never call its main or inspect actual prices."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import record_after157_source_extension as p


def state_fixture():
    rows = [dict(directory='prior' + str(i), trials=6) for i in range(156)]
    rows.append(dict(directory='monthly_filing_association157', trials=50))
    return dict(current='monthly_filing_association157', current_status='completed_independently_verified',
                completed=rows, registered_queue=[], best_training_selected_candidate={
                    'cagr': 0.6766617372041686, 'nested': ['unchanged']},
                unrelated={'history': 'keep'}, next_step='old', updated_utc='old')


class PublisherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'migration').mkdir()
        (self.root / 'outputs/monthly_target70_research').mkdir(parents=True)
        self.state = state_fixture()
        self.put(p.lifecycle.STATE, self.state)
        self.reader = p.SourceReader(self.root)
        block = patch.object(p.walmart, 'transport', side_effect=AssertionError('No network in tests'))
        block.start()
        self.addCleanup(block.stop)

    def put(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = value if isinstance(value, bytes) else (json.dumps(value, sort_keys=True) + '\n').encode()
        path.write_bytes(raw)
        return p.files.sha(raw)

    def reference(self, name):
        self.reader.read(name)
        return dict(path=name, sha256=self.reader.hashes[name])

    def checkpoint(self):
        return dict(created_utc='first', next_step='source work only', completed_rounds=157,
                    strategy_trials=986, source_admission_changed=False, original_tier3_changed=False)

    def test_state_whitelist_and_deep_preservation(self):
        original = copy.deepcopy(self.state)
        updated = p.state_update(self.state, {'path': 'fixture', 'sha256': 'a' * 64}, 'new', 'now')
        changed = {k for k in set(updated) | set(self.state) if updated.get(k) != self.state.get(k)}
        self.assertEqual(changed, {p.REF_KEY, 'next_step', 'last_goal_turn_classification', 'updated_utc'})
        updated['best_training_selected_candidate']['nested'].append('only updated copy')
        self.assertEqual(self.state, original)

    def test_expected_completed_count_trials_incumbent_and_queue(self):
        p.validate_state(self.state)
        bad = [dict(self.state, completed=self.state['completed'][:-1]),
               dict(self.state, registered_queue=['new']), dict(self.state, current='other'),
               dict(self.state, best_training_selected_candidate={'cagr': 0.70}),
               dict(self.state, current_status='running')]
        altered = copy.deepcopy(self.state)
        altered['completed'][0]['trials'] = True
        bad.append(altered)
        for value in bad:
            with self.subTest(value=value.get('current_status')):
                with self.assertRaises(ValueError):
                    p.validate_state(value)

    def test_exact_historical_archive_survives_later_live_state(self):
        raw = (self.root / p.lifecycle.STATE).read_bytes()
        expected = p.files.sha(raw)
        archive = 'outputs/monthly_target70_research/PRE157_' + expected + '.json'
        with patch.object(p, 'HISTORICAL_SHA', expected), patch.object(p, 'ARCHIVE', archive):
            p.preserve_historical_state(self.root)
            self.assertEqual((self.root / archive).read_bytes(), raw)
            self.put(p.lifecycle.STATE, dict(self.state, updated_utc='later'))
            p.preserve_historical_state(self.root)
            self.reader.read(p.lifecycle.STATE, expected)
            self.assertNotIn(p.lifecycle.STATE, self.reader.hashes)
            self.assertEqual(self.reader.hashes[archive], expected)

    def test_missing_old_bytes_are_not_fabricated(self):
        with patch.object(p, 'HISTORICAL_SHA', 'a' * 64):
            with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
                p.preserve_historical_state(self.root)
        self.assertFalse((self.root / p.ARCHIVE).exists())

    def test_tampered_archive_rejected(self):
        self.put(p.ARCHIVE, b'not historical bytes')
        with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
            p.preserve_historical_state(self.root)

    def test_only_one_explicit_mutable_pin_can_alias(self):
        with self.assertRaisesRegex(ValueError, 'Unrecognized historical state pin'):
            self.reader.read(p.lifecycle.STATE, 'a' * 64)

    def test_checkpoint_commit_exact_backup_and_idempotent_rerun(self):
        before_bytes = (self.root / p.lifecycle.STATE).read_bytes()
        state, before = p.lifecycle.state_snapshot(self.root / p.lifecycle.STATE)
        result = p.commit_checkpoint(self.reader, state, before, self.checkpoint())
        self.assertFalse(result['already_recorded'])
        backup = self.root / 'outputs/monthly_target70_research' / ('PRE157_' + before + '.json')
        self.assertEqual(backup.read_bytes(), before_bytes)
        after_bytes = (self.root / p.lifecycle.STATE).read_bytes()
        state, before = p.lifecycle.state_snapshot(self.root / p.lifecycle.STATE)
        fresh = p.SourceReader(self.root)
        result = p.commit_checkpoint(fresh, state, before, dict(self.checkpoint(), created_utc='new time'))
        self.assertTrue(result['already_recorded'])
        self.assertEqual((self.root / p.lifecycle.STATE).read_bytes(), after_bytes)
        for key in ('completed', 'registered_queue', 'best_training_selected_candidate', 'unrelated'):
            self.assertEqual(state[key], self.state[key])

    def test_checkpoint_recovers_after_cas_conflict_without_losing_new_state(self):
        state, before = p.lifecycle.state_snapshot(self.root / p.lifecycle.STATE)
        changed = dict(state, unrelated={'new': 'preserve concurrent metadata'})
        self.put(p.lifecycle.STATE, changed)
        with self.assertRaisesRegex(ValueError, 'Concurrent continuation change'):
            p.commit_checkpoint(self.reader, state, before, self.checkpoint())
        self.assertTrue((self.root / p.CHECKPOINT).exists())
        current, new_hash = p.lifecycle.state_snapshot(self.root / p.lifecycle.STATE)
        result = p.commit_checkpoint(p.SourceReader(self.root), current, new_hash, self.checkpoint())
        self.assertFalse(result['already_recorded'])
        updated, _ = p.lifecycle.state_snapshot(self.root / p.lifecycle.STATE)
        self.assertEqual(updated['unrelated'], changed['unrelated'])

    def test_conflicting_checkpoint_never_overwritten(self):
        self.put(p.CHECKPOINT, dict(self.checkpoint(), strategy_trials=987))
        before_bytes = (self.root / p.lifecycle.STATE).read_bytes()
        state, before = p.lifecycle.state_snapshot(self.root / p.lifecycle.STATE)
        with self.assertRaisesRegex(ValueError, 'Conflicting source checkpoint'):
            p.commit_checkpoint(self.reader, state, before, self.checkpoint())
        self.assertEqual((self.root / p.lifecycle.STATE).read_bytes(), before_bytes)

    def test_existing_state_reference_never_replaced(self):
        state = dict(self.state, **{p.REF_KEY: {'path': p.CHECKPOINT, 'sha256': '0' * 64}})
        self.put(p.lifecycle.STATE, state)
        current, before = p.lifecycle.state_snapshot(self.root / p.lifecycle.STATE)
        with self.assertRaisesRegex(ValueError, 'missing source checkpoint'):
            p.commit_checkpoint(self.reader, current, before, self.checkpoint())

    def wmt_fixture(self):
        scope_name = 'work/fixture_scope.json'
        directory = 'outputs/wmt_fixture'
        helper = 'a' * 64
        targets = [dict(id='wmt' + str(i), provider='walmart', url='https://stock.walmart.com/' + str(i),
                        body_file=str(i) + '.html', receipt_file=str(i) + '.receipt.json') for i in range(6)]
        dep = 'work/fixture_dependency'
        dep_sha = self.put(dep, b'fixed')
        scope = dict(output_dir=directory, targets=targets, input_hashes={dep: dep_sha})
        scope_sha = self.put(scope_name, scope)
        binding = dict(schema=p.walmart.SCHEMA, scope_sha256=scope_sha, helper_sha256=helper)
        self.put(directory + '/.collection-scope.json', binding)
        summaries = []
        for target in targets:
            base = p.walmart.base_binding(target, scope_sha, helper)
            raw = b'synthetic body'
            raw_sha = self.put(directory + '/' + target['body_file'], raw)
            attempt_sha = self.put(directory + '/.attempt_' + target['id'] + '.json', base)
            receipt = dict(base, attempted=True, outcome='success', http_status=200, stop_provider=False,
                           body_truncated_at_limit=False, final_url=target['url'], bytes=len(raw),
                           sha256=raw_sha, attempt_sha256=attempt_sha)
            receipt_sha = self.put(directory + '/' + target['receipt_file'], receipt)
            summaries.append(dict(id=target['id'], outcome='success', receipt_file=target['receipt_file'],
                                  receipt_sha256=receipt_sha))
        collection_name = directory + '/COLLECTION.json'
        collection_sha = self.put(collection_name, dict(binding, all_successful=True,
                                                        stopped_providers=[], targets=summaries))
        return scope_name, scope, helper, {scope_name: scope_sha, collection_name: collection_sha}

    def test_wmt_calls_real_verify_existing_for_every_target(self):
        name, scope, helper, pins = self.wmt_fixture()
        with patch.dict(p.PINS, pins), patch.object(p.walmart, 'validate_scope',
                return_value=(scope, self.root / scope['output_dir'], helper)), \
                patch.object(p.walmart, 'verify_existing', wraps=p.walmart.verify_existing) as verify:
            result = p.verify_wmt_batch(self.reader, name)
        self.assertEqual(verify.call_count, 6)
        self.assertEqual(result['successful_targets'], 6)
        self.assertEqual(result['status'], 'collected_not_admitted')

    def test_wmt_tampered_body_rejected(self):
        name, scope, helper, pins = self.wmt_fixture()
        self.put(scope['output_dir'] + '/4.html', b'changed')
        with patch.dict(p.PINS, pins), patch.object(p.walmart, 'validate_scope',
                return_value=(scope, self.root / scope['output_dir'], helper)):
            with self.assertRaisesRegex(ValueError, 'body SHA mismatch'):
                p.verify_wmt_batch(self.reader, name)

    def test_optional_review_no_review_never_admits(self):
        self.assertIsNone(p.attach_wmt_review(self.reader, [], None, None))
        with self.assertRaisesRegex(ValueError, 'Both WMT'):
            p.attach_wmt_review(self.reader, [], 'work/review.json', None)

    def review_fixture(self):
        batches = []
        pins = {}
        for i in range(2):
            records = []
            for tag in ('scope', 'collection', 'raw', 'receipt'):
                name = 'work/' + tag + str(i)
                pins[name] = self.put(name, (tag + str(i)).encode())
                records.append(dict(path=name, sha256=pins[name]))
            batches.append(dict(scope=records[0], collection=records[1],
                                targets=[dict(raw=records[2], receipt=records[3])]))
        return batches, pins

    def test_conditional_review_summary_attaches_without_strategy_admission(self):
        batches, pins = self.review_fixture()
        review = dict(source_admission_changed=False, full_strategy_admission=False,
                      old_admission_flags_changed=False, input_hashes=pins,
                      summary={'first88': {'conditional_pair_slots': 7}})
        sha = self.put('work/review.json', review)
        result = p.attach_wmt_review(self.reader, batches, 'work/review.json', sha)
        self.assertEqual(result['reported_conditional_summary'], review['summary'])
        self.assertEqual(result['status'], 'review_attached_no_automatic_admission')
        self.assertFalse(result['combined_overlay_sanity_verified'])

    def combined_fixture(self, empty=False):
        originals, previous, rows = [], [], []
        # Five document matches, but only three eligible rows: one retains a
        # timing blocker, and another has no Tier2 admission.
        for i in range(122):
            for j in range(49):
                is_ko = not empty and i == 0 and j == 0
                is_wmt = not empty and i in (1, 2, 3, 88) and j == 1
                symbol = 'KO' if j == 0 else ('WMT' if j == 1 else 'COL' + str(j))
                reasons = [p.MISSING_CONTEXT] + (['TIMING_PENDING'] if is_wmt and i == 2 else [])
                original = dict(signal_index=i, candidate_index=j, symbol=symbol, is_etf=j >= 32,
                    cik='0000104169' if j == 1 else 'other', tier2=not (is_wmt and i == 3), tier3=False,
                    selected_event_ids=['wmt_a', 'wmt_b'] if is_wmt else [], reasons=reasons,
                    existing_accession_context={'admitted': False}, original_decision_cutoff='fixed')
                originals.append(original)
                previous.append(dict(original_row=copy.deepcopy(original), body_pair_context_verified=is_ko))
                matched = is_ko or is_wmt
                remaining = (['TIMING_PENDING'] if i == 2 else []) if is_wmt else ([] if is_ko else reasons)
                rows.append(dict(original_row=copy.deepcopy(original), body_pair_context_verified=matched,
                    body_context_source='KO' if is_ko else ('WMT' if is_wmt else None),
                    resolved_context_reason=[p.MISSING_CONTEXT] if matched else [],
                    remaining_row_reasons=remaining,
                    conditional_annual_pair_admitted=matched and not (is_wmt and i in (2, 3)),
                    old_admission_flags_changed=False))
        summary = {}
        for name, months, ko_count, wmt_count in [('all122', 122, 1, 2), ('first88', 88, 1, 1),
                                                ('later34', 34, 0, 1)]:
            if empty:
                ko_count = wmt_count = 0
            summary[name] = dict(all_slots=months * 49, stock_slots=months * 32,
                conditional_pair_slots=ko_count + wmt_count, by_issuer={'KO': ko_count, 'WMT': wmt_count},
                original_tier3_slots=0)
        review = dict(schema=p.WMT_OVERLAY_SCHEMA, status='SOURCE_ONLY_CONDITIONAL_OVERLAY_NO_STRATEGY_EXPERIMENT',
            source_admission_changed=False, old_admission_flags_changed=False, new_conditional_context_admission=True,
            full_strategy_admission=False, numerical_experiment_registered=False, lag_or_feature_values_computed=False,
            rows=rows, summary=summary, verified_wmt_event_pairs=[] if empty else [['wmt_a', 'wmt_b']])
        return review, dict(rows=originals), dict(rows=previous)

    def test_combined_full_grid_summary_and_zero_coverage_are_not_hardcoded(self):
        for empty in (False, True):
            review, prior, ko = self.combined_fixture(empty)
            before = copy.deepcopy((review, prior, ko))
            result = p.verify_combined_overlay(review, prior, ko)
            self.assertEqual(result, review['summary'])
            self.assertEqual(result['all122']['conditional_pair_slots'], 0 if empty else 3)
            self.assertEqual((review, prior, ko), before)

    def test_exact_combined_path_and_work_copy_both_validate_full_grid(self):
        for path in (p.WMT_OVERLAY, 'work/combined.json'):
            with self.subTest(path=path):
                batches, pins = self.review_fixture()
                review, prior, ko = self.combined_fixture()
                prior_sha = self.put(p.PRIOR, prior)
                ko_sha = self.put(p.KO, ko)
                review['input_hashes'] = dict(pins, **{p.PRIOR: prior_sha, p.KO: ko_sha})
                digest = self.put(path, review)
                with patch.object(p, 'PRIOR_SHA', prior_sha), patch.dict(p.PINS, {p.KO: ko_sha}):
                    result = p.attach_wmt_review(p.SourceReader(self.root), batches, path, digest)
                self.assertTrue(result['combined_overlay_sanity_verified'])
                self.assertEqual(result['reported_conditional_summary'], review['summary'])
                self.assertEqual(result['status'], 'review_attached_no_automatic_admission')

    def test_combined_path_requires_schema_and_no_other_output_paths_allowed(self):
        digest = self.put(p.WMT_OVERLAY, {'source_admission_changed': False})
        with self.assertRaisesRegex(ValueError, 'fixed schema'):
            p.attach_wmt_review(self.reader, [], p.WMT_OVERLAY, digest)
        for path in ('outputs/other/ADMISSION.json', p.WMT_OVERLAY + '.json',
                     'outputs/after157_walmart_context_admission_v1/other.json',
                     '/work/review.json', 'work/../outside.json'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                p.attach_wmt_review(self.reader, [], path, digest)

    def test_combined_path_symlink_and_wrong_sha_rejected(self):
        digest = self.put('work/synthetic.json', {'source_admission_changed': False})
        output = self.root / p.WMT_OVERLAY
        output.parent.mkdir(parents=True)
        output.symlink_to(self.root / 'work/synthetic.json')
        with self.assertRaisesRegex(ValueError, 'Symlink'):
            p.attach_wmt_review(self.reader, [], p.WMT_OVERLAY, digest)
        with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
            p.attach_wmt_review(self.reader, [], 'work/synthetic.json', '0' * 64)

    def test_combined_requires_exact_prior_and_ko_input_pins(self):
        batches, pins = self.review_fixture()
        review, prior, ko = self.combined_fixture()
        prior_sha, ko_sha = self.put(p.PRIOR, prior), self.put(p.KO, ko)
        for omitted in (p.PRIOR, p.KO):
            review['input_hashes'] = dict(pins, **{p.PRIOR: prior_sha, p.KO: ko_sha})
            del review['input_hashes'][omitted]
            digest = self.put(p.WMT_OVERLAY, review)
            with patch.object(p, 'PRIOR_SHA', prior_sha), patch.dict(p.PINS, {p.KO: ko_sha}):
                with self.assertRaisesRegex(ValueError, 'prior/KO pins'):
                    p.attach_wmt_review(p.SourceReader(self.root), batches, p.WMT_OVERLAY, digest)

    def test_combined_grid_and_original_values_cannot_change(self):
        review, prior, ko = self.combined_fixture()
        for mutation in ('missing', 'duplicate', 'reorder', 'nested', 'bool_as_int'):
            bad = copy.deepcopy(review)
            if mutation == 'missing':
                bad['rows'].pop()
            elif mutation == 'duplicate':
                bad['rows'][1] = copy.deepcopy(bad['rows'][0])
            elif mutation == 'reorder':
                bad['rows'][1], bad['rows'][2] = bad['rows'][2], bad['rows'][1]
            elif mutation == 'nested':
                bad['rows'][0]['original_row']['existing_accession_context']['admitted'] = True
            else:
                bad['rows'][0]['original_row']['tier3'] = 0
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                p.verify_combined_overlay(bad, prior, ko)
        prior['rows'][0]['tier3'] = True
        with self.assertRaisesRegex(ValueError, 'Original admission'):
            p.verify_combined_overlay(review, prior, ko)

    def test_combined_cannot_remove_timing_blocker_or_inflate_tier2_or_etf(self):
        review, prior, ko = self.combined_fixture()
        for index, changes in ((2 * 49 + 1, {'remaining_row_reasons': []}),
                               (3 * 49 + 1, {'conditional_annual_pair_admitted': True}),
                               (32, {'body_pair_context_verified': True, 'conditional_annual_pair_admitted': True}),
                               (49 + 1, {'old_admission_flags_changed': True}),
                               (0, {'body_context_source': 'WMT'})):
            bad = copy.deepcopy(review)
            bad['rows'][index].update(changes)
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, 'join/blocker'):
                p.verify_combined_overlay(bad, prior, ko)

    def test_combined_rejects_missing_or_positive_strategy_activity_claims(self):
        review, prior, ko = self.combined_fixture()
        for key in ('source_admission_changed', 'old_admission_flags_changed', 'full_strategy_admission',
                    'numerical_experiment_registered', 'lag_or_feature_values_computed'):
            for value in (None, True, 0):
                bad = dict(review)
                if value is None:
                    del bad[key]
                else:
                    bad[key] = value
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, 'admission/activity'):
                    p.verify_combined_overlay(bad, prior, ko)

    def test_combined_rejects_forged_summary_or_event_pairs(self):
        review, prior, ko = self.combined_fixture()
        for section, key, value in (('all122', 'conditional_pair_slots', 4),
                                   ('first88', 'by_issuer', {'KO': 0, 'WMT': 2}),
                                   ('later34', 'conditional_pair_slots', True),
                                   ('all122', 'original_tier3_slots', 1)):
            bad = copy.deepcopy(review)
            bad['summary'][section][key] = value
            with self.subTest(section=section, key=key), self.assertRaisesRegex(ValueError, 'summary'):
                p.verify_combined_overlay(bad, prior, ko)
        for pairs in ([], [['other_a', 'other_b']], [['wmt_a', 'wmt_a']],
                      [['wmt_a', 'wmt_b'], ['wmt_b', 'wmt_a']]):
            bad = dict(review, verified_wmt_event_pairs=pairs)
            with self.subTest(pairs=pairs), self.assertRaises(ValueError):
                p.verify_combined_overlay(bad, prior, ko)

    def test_unbound_or_broader_wmt_review_rejected(self):
        batches, pins = self.review_fixture()
        for fields in [dict(input_hashes={k: v for k, v in pins.items() if k != 'work/raw1'}),
                       dict(full_strategy_admission=True), dict(source_admission_changed=True)]:
            review = dict(source_admission_changed=False, input_hashes=pins)
            review.update(fields)
            sha = self.put('work/review.json', review)
            with self.assertRaises(ValueError):
                p.attach_wmt_review(p.SourceReader(self.root), batches, 'work/review.json', sha)


if __name__ == '__main__':
    unittest.main()
