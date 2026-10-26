"""Synthetic join tests only; no market or financial observations."""
import copy
import unittest
from after157_walmart_context_admission import join_rows, MISSING


class JoinTests(unittest.TestCase):
    def setUp(self):
        self.row = dict(symbol='WMT', cik='0000104169', is_etf=False,
                        selected_event_ids=['a', 'b'], reasons=[MISSING], tier2=True,
                        tier3=False, nested={'old_context':False})
        self.meta = {x:dict(cik='0000104169') for x in ['a','b']}
        self.pairs = {frozenset(['a','b'])}

    def run_join(self, row=None, pairs=None):
        row = self.row if row is None else row
        return join_rows([row], [dict(original_row=copy.deepcopy(row), body_pair_context_verified=False)],
                         self.pairs if pairs is None else pairs, self.meta)[0]

    def test_admission_and_original_are_separate(self):
        result = self.run_join()
        self.assertTrue(result['conditional_annual_pair_admitted'])
        self.assertEqual(result['original_row'], self.row)
        self.assertFalse(result['original_row']['tier3'])
        result['original_row']['nested']['old_context'] = True
        self.assertFalse(self.row['nested']['old_context'])

    def test_other_blocker_survives(self):
        self.row['reasons'].append('LATEST_ORIGINAL_NOT_AVAILABLE_AT_T_MINUS_42')
        result = self.run_join()
        self.assertFalse(result['conditional_annual_pair_admitted'])
        self.assertEqual(result['remaining_row_reasons'], ['LATEST_ORIGINAL_NOT_AVAILABLE_AT_T_MINUS_42'])

    def test_tier2_false_survives(self):
        self.row['tier2'] = False
        self.assertFalse(self.run_join()['conditional_annual_pair_admitted'])

    def test_no_pair_or_duplicate_ids_remain_unresolved(self):
        self.assertEqual(self.run_join(pairs=set())['remaining_row_reasons'], [MISSING])
        self.row['selected_event_ids'] = ['a','a']
        self.assertFalse(self.run_join()['conditional_annual_pair_admitted'])

    def test_wrong_issuer_rejected(self):
        for key, value in [('symbol','OTHER'),('cik','0000000001'),('is_etf',True)]:
            row = dict(self.row, **{key:value})
            with self.assertRaises(ValueError): self.run_join(row)

    def test_ko_grid_cannot_silently_drift(self):
        with self.assertRaises(ValueError):
            join_rows([self.row], [dict(original_row={}, body_pair_context_verified=False)], self.pairs, self.meta)


if __name__ == '__main__':
    unittest.main()
