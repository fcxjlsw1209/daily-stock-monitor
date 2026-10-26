"""Synthetic event joins for the source-only progress checkpoint."""
import copy
import unittest

from record_after157_body_progress import contextual_rows


class ContextJoinTest(unittest.TestCase):
    def setUp(self):
        self.row = dict(signal_index=0, candidate_index=1, signal_date='2018-05-31',
            symbol='KO', cik='0000021344', is_etf=False, selected_event_ids=['new', 'old'],
            tier2=False, tier3=False, reasons=['IDENTITY_PENDING'],
            existing_accession_context=[{'prior141_blockers': ['original_context_pending']}])
        self.metadata = {k: dict(cik='0000021344') for k in ('new', 'old')}
        self.pairs = {frozenset(['new', 'old'])}

    def test_document_context_does_not_promote_old_admission(self):
        original = copy.deepcopy(self.row)
        result = contextual_rows([self.row], self.pairs, self.metadata)[0]
        self.assertTrue(result['body_pair_context_verified'])
        self.assertFalse(result['prior_tier2'])
        self.assertFalse(result['prior_tier3'])
        self.assertFalse(result['source_admission_changed'])
        self.assertEqual(result['prior_reasons'], original['reasons'])
        self.assertEqual(self.row, original)
        result['prior_existing_accession_context'][0]['prior141_blockers'].append('new')
        self.assertEqual(self.row, original)

    def test_other_years_and_one_good_document_stay_pending(self):
        for ids in (['new'], ['new', 'other'], []):
            row = dict(self.row, selected_event_ids=ids)
            result = contextual_rows([row], self.pairs, self.metadata)[0]
            self.assertFalse(result['body_pair_context_verified'])
            self.assertEqual(result['selected_event_ids'], ids)

    def test_wrong_stock_or_etf_cannot_inherit_ko_context(self):
        for changes in ({'symbol': 'MSFT'}, {'is_etf': True}, {'cik': '0000789019'}):
            with self.assertRaises(ValueError):
                contextual_rows([dict(self.row, **changes)], self.pairs, self.metadata)

    def test_pair_join_preserves_original_order(self):
        row = dict(self.row, selected_event_ids=['old', 'new'], tier2=True)
        result = contextual_rows([row], self.pairs, self.metadata)[0]
        self.assertTrue(result['body_pair_context_verified'])
        self.assertEqual(result['selected_event_ids'], ['old', 'new'])


if __name__ == '__main__':
    unittest.main()
