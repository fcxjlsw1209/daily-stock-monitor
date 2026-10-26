"""Independent synthetic tests of the limited missing-metadata repair route."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import annual_lag158_repair_review as repair
import annual_lag158_independent_outcome as outcome
from test_annual_lag158_independent_prepare import fixture


def early_return(data, i=0, j=32):
    item = data['rows'][49*i+j]
    row = item['original_row']
    reason = repair.ETF_REASON if row['is_etf'] else repair.UNAVAILABLE_REASON
    row.update(selected_snapshot=None, cik=None, selected_event_ids=[], period_hint_ids=[],
               tier1=False, tier2=False, tier3=False, normal_period_admission=False, reasons=[reason])
    row.pop('known_amendment_event_ids', None)
    item.update(body_pair_context_verified=False, body_context_source=None,
                conditional_annual_pair_admitted=False, old_admission_flags_changed=False,
                remaining_row_reasons=[reason])
    data['source_mask'][i,j] = False
    return item


class IndependentRepair(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = fixture()

    def setUp(self):
        self.data = deepcopy(self.base)

    def build(self, n=1):
        return repair.features_oracle(**self.data, n=n)

    def test_complete_schema_matches_original_oracle_exactly(self):
        old = repair.FROZEN_ORACLE(**self.data)
        got = self.build(None)
        self.assertEqual(got['cells'], old['cells'])
        self.assertEqual(got['monthly'], old['monthly'])
        for key in repair.original.EXTRA:
            self.assertEqual(got['arrays'][key].tobytes(), old['arrays'][key].tobytes())

    def test_etf_early_return_retains_unknown_and_all_blockers(self):
        item = early_return(self.data)
        before = deepcopy(item)
        got = self.build()
        self.assertEqual(item, before)
        self.assertNotIn('known_amendment_event_ids', item['original_row'])
        cell = got['cells'][32]
        self.assertEqual(cell['reasons'], ['SOURCE_NOT_CONDITIONALLY_ADMITTED',
            'SOURCE_OR_ANNUAL_CONTEXT_NOT_VERIFIED', 'SOURCE_BLOCKER:'+repair.ETF_REASON,
            repair.ABSENT_REASON, 'OUTSIDE_FIXED_KO_WMT_SCOPE'])
        self.assertFalse(cell['available'])
        self.assertNotIn('previous_lag_days', cell)
        for key in repair.RAW:
            self.assertTrue(np.isnan(got['arrays'][key][0,32]))
        for key in repair.RANKS:
            self.assertEqual(got['arrays'][key][0,32], 0.)

    def test_no_extent_early_return_is_not_absence_of_amendments(self):
        early_return(self.data, j=2)
        got = self.build()
        self.assertIn(repair.ABSENT_REASON, got['cells'][2]['reasons'])
        self.assertFalse(got['arrays']['common_available'][0,2])
        self.assertNotIn('KNOWN_SELECTED_PERIOD_AMENDMENT', got['cells'][2]['reasons'])

    def test_explicit_null_or_other_bad_type_is_never_exception(self):
        item = early_return(self.data)
        for value in (None, '', (), {}, [None], [1], ['']):
            with self.subTest(value=value):
                item['original_row']['known_amendment_event_ids'] = value
                with self.assertRaises(ValueError): self.build()

    def test_missing_on_admitted_row_hard_fails_even_if_ineligible(self):
        row = self.data['rows'][0]['original_row']
        row.pop('known_amendment_event_ids')
        self.data['eligible'][0,0] = False
        with self.assertRaises(ValueError): self.build()

    def test_missing_on_ordinary_unadmitted_row_hard_fails(self):
        self.data['rows'][2]['original_row'].pop('known_amendment_event_ids')
        with self.assertRaises(ValueError): self.build()

    def test_only_exact_known_source_reasons_are_permitted(self):
        item = early_return(self.data)
        for reasons in ([], ['unrelated'], [repair.ETF_REASON, 'extra'], [repair.UNAVAILABLE_REASON]):
            with self.subTest(reasons=reasons):
                item['original_row']['reasons'] = reasons
                with self.assertRaises(ValueError): self.build()

    def test_all_early_return_facts_are_required(self):
        pristine = early_return(self.data)
        for key in ('selected_snapshot','cik','selected_event_ids','period_hint_ids',
                    'tier1','tier2','tier3','normal_period_admission'):
            item = deepcopy(pristine); item['original_row'].pop(key)
            self.assertFalse(repair.absent_metadata_permitted(item, False), key)
        for key in ('body_context_source','body_pair_context_verified','old_admission_flags_changed',
                    'conditional_annual_pair_admitted'):
            item = deepcopy(pristine); item.pop(key)
            self.assertFalse(repair.absent_metadata_permitted(item, False), key)

    def test_present_selected_evidence_or_true_flags_are_not_early_return(self):
        pristine = early_return(self.data)
        for key, value in (('selected_snapshot','snapshot'),('cik','0000000001'),
            ('selected_event_ids',['event']),('period_hint_ids',[[]]),('tier1',True),('tier2',True),
            ('tier3',True),('normal_period_admission',True)):
            item = deepcopy(pristine); item['original_row'][key] = value
            self.assertFalse(repair.absent_metadata_permitted(item, False), key)
        for key, value in (('body_context_source','WMT'),('body_pair_context_verified',True),
                           ('old_admission_flags_changed',True),('conditional_annual_pair_admitted',True)):
            item = deepcopy(pristine); item[key] = value
            self.assertFalse(repair.absent_metadata_permitted(item, False), key)

    def test_false_flags_reject_zero_and_missing_mask(self):
        pristine = early_return(self.data)
        for key in ('tier1','tier2','tier3','normal_period_admission'):
            item = deepcopy(pristine); item['original_row'][key] = 0
            self.assertFalse(repair.absent_metadata_permitted(item, False), key)
        self.assertFalse(repair.absent_metadata_permitted(pristine, None))
        self.assertFalse(repair.absent_metadata_permitted(pristine, True))

    def test_retains_extra_remaining_blocker_and_ineligibility(self):
        item = early_return(self.data)
        item['remaining_row_reasons'].append('KNOWN_BAD_SOURCE')
        self.data['eligible'][0,32] = False
        reasons = self.build()['cells'][32]['reasons']
        self.assertIn('SOURCE_BLOCKER:KNOWN_BAD_SOURCE', reasons)
        self.assertIn('ORIGINAL_INELIGIBLE', reasons)
        self.assertLess(reasons.index(repair.ABSENT_REASON), reasons.index('ORIGINAL_INELIGIBLE'))

    def test_missing_does_not_bypass_cutoff_validation(self):
        item = early_return(self.data)
        item['original_row']['information_cutoff_ns'] += 1
        with self.assertRaises(ValueError): self.build()

    def test_full_grid_all_etfs_missing_does_not_change_any_numeric_value(self):
        before = self.build(None)
        for i in range(122):
            for j in range(32,49): early_return(self.data, i,j)
        after = self.build(None)
        self.assertEqual(len(after['cells']), 5978)
        self.assertEqual(int(after['arrays']['common_available'].sum()), 230)
        for key in repair.original.EXTRA:
            self.assertEqual(before['arrays'][key].tobytes(), after['arrays'][key].tobytes())
        self.assertEqual(before['monthly'], after['monthly'])
        self.assertEqual(sum(repair.ABSENT_REASON in row['reasons'] for row in after['cells']), 2074)

    def test_missing_unadmitted_issuer_does_not_dilute_peer_rank(self):
        early_return(self.data, j=0)
        got = self.build()
        self.assertFalse(got['arrays']['common_available'][0,0])
        self.assertTrue(got['arrays']['common_available'][0,1])
        for key in repair.RANKS:
            self.assertTrue((got['arrays'][key][0] == 0).all())

    def test_future_rows_cannot_change_prefix(self):
        early_return(self.data)
        first = self.build(1)
        early_return(self.data, i=100)
        self.data['event_metadata'][self.data['rows'][100*49]['original_row']['selected_event_ids'][0]]['filingDate'] = '2099-01-01'
        again = self.build(1)
        self.assertEqual(first['cells'], again['cells'])
        for key in repair.original.EXTRA:
            self.assertEqual(first['arrays'][key].tobytes(), again['arrays'][key].tobytes())

    def test_all_123_prefixes_match_full_synthetic_grid(self):
        early_return(self.data)
        early_return(self.data, i=99, j=2)
        full = self.build(None)
        for n in range(123):
            prefix = self.build(n)
            self.assertEqual(prefix['cells'], full['cells'][:49*n])
            self.assertEqual(prefix['monthly'], full['monthly'][:n])
            for key in repair.original.EXTRA:
                self.assertEqual(prefix['arrays'][key].tobytes(), full['arrays'][key][:n].tobytes())

    def test_existing_nonempty_amendment_list_still_blocks(self):
        row = self.data['rows'][2]['original_row']
        row['known_amendment_event_ids'] = ['known-original-amendment']
        got = self.build()['cells'][2]
        self.assertIn('KNOWN_SELECTED_PERIOD_AMENDMENT', got['reasons'])
        self.assertNotIn(repair.ABSENT_REASON, got['reasons'])


class AuditRoute(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.amendment = dict(fixed='synthetic amendment')
        self.base = dict(passed=True, producer_modules_imported=False,
                         source_hashes={'old-source':'a'*64}, artifact_hashes={'old-artifact':'b'*64})
        self.events = []
        def verify(*args): self.events.append('verify'); return deepcopy(self.amendment)
        def augment(root, report, plan, amendment):
            self.events.append('augment')
            return dict(report, amendment_sha256=amendment, repair_artifact_hashes={'new':'c'*64})
        def check(*args): self.events.append('check')
        self.h = SimpleNamespace(verify_amendment=verify, augment_report=augment, verify_repair_report=check)

    def tearDown(self):
        self.assertIs(repair.original.features_oracle, repair.FROZEN_ORACLE)
        self.tmp.cleanup()

    def successful_audit(self, *args):
        self.events.append('audit')
        self.assertIs(repair.original.features_oracle, repair.features_oracle)
        return deepcopy(self.base)

    def test_prepare_route_binds_before_and_after_preserving_original_closures(self):
        with patch.object(repair, 'helper', return_value=self.h), patch.object(repair.original, 'audit', side_effect=self.successful_audit):
            got = repair.audit('prepare', self.root, repair.PLAN_SHA, 'd'*64)
        self.assertEqual(self.events, ['verify','audit','verify','augment','check'])
        self.assertEqual(got['source_hashes'], self.base['source_hashes'])
        self.assertEqual(got['artifact_hashes'], self.base['artifact_hashes'])

    def test_outcome_route_uses_same_repaired_independent_oracle(self):
        with patch.object(repair, 'helper', return_value=self.h), patch.object(outcome, 'audit', side_effect=self.successful_audit):
            repair.audit('outcome', self.root, repair.PLAN_SHA, 'd'*64)
        self.assertEqual(self.events, ['verify','audit','verify','augment','check'])

    def test_failure_always_restores_original_oracle_and_publishes_nothing(self):
        with patch.object(repair, 'helper', return_value=self.h), patch.object(repair.original, 'audit', side_effect=RuntimeError('synthetic failure')):
            with self.assertRaises(RuntimeError): repair.audit('prepare', self.root, repair.PLAN_SHA, 'd'*64)
        self.assertEqual(self.events, ['verify'])
        self.assertEqual(list(self.root.iterdir()), [])

    def test_nested_or_prepatched_oracle_refused(self):
        with repair.repaired_oracle():
            with self.assertRaises(ValueError):
                with repair.repaired_oracle(): pass
            self.assertIs(repair.original.features_oracle, repair.features_oracle)

    def test_changed_amendment_after_audit_rejected(self):
        counter = iter((self.amendment, {'fixed':'changed'}))
        self.h.verify_amendment = lambda *args: next(counter)
        with patch.object(repair, 'helper', return_value=self.h), patch.object(repair.original, 'audit', side_effect=self.successful_audit):
            with self.assertRaises(ValueError): repair.audit('prepare', self.root, repair.PLAN_SHA, 'd'*64)

    def test_new_provenance_cannot_expand_old_artifact_closure(self):
        self.h.augment_report = lambda root, report, plan, amend: dict(report, artifact_hashes={'new':'c'*64})
        with patch.object(repair, 'helper', return_value=self.h), patch.object(repair.original, 'audit', side_effect=self.successful_audit):
            with self.assertRaises(ValueError): repair.audit('prepare', self.root, repair.PLAN_SHA, 'd'*64)

    def test_wrong_plan_or_stage_rejected_before_helper(self):
        with patch.object(repair, 'helper') as h:
            for stage, plan in (('unknown', repair.PLAN_SHA), ('prepare','0'*64)):
                with self.assertRaises(ValueError): repair.audit(stage, self.root, plan, 'd'*64)
            h.assert_not_called()

    def test_published_review_never_reexecuted(self):
        directory = self.root/repair.OUT; directory.mkdir(parents=True)
        (directory/'INDEPENDENT_PREPARE.json').write_text('{}')
        with patch.object(repair, 'helper') as h:
            with self.assertRaises(ValueError): repair.audit('prepare', self.root, repair.PLAN_SHA, 'd'*64)
            h.assert_not_called()

    def test_producer_import_is_rejected(self):
        for name in ('annual_lag158_cell_repair','monthly_annual_lag158_repaired_runner'):
            with patch.dict(sys.modules, {name:object()}):
                with self.assertRaises(ValueError): repair.independent_imports()

    def test_publish_is_write_once(self):
        target = self.root/'receipt.json'; repair.publish(target, self.base)
        before = target.read_bytes()
        with self.assertRaises(FileExistsError): repair.publish(target, {'passed':False})
        self.assertEqual(before, target.read_bytes())


if __name__ == '__main__':
    unittest.main()
