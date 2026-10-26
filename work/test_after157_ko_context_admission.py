import copy
import unittest

from after157_ko_context_admission import admit_rows, MISSING_CONTEXT


class ContextAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.row = dict(symbol='KO', is_etf=False, cik='0000021344', selected_event_ids=['old', 'new'],
                        tier2=True, tier3=False, reasons=[MISSING_CONTEXT], other={'preserved': True})
        self.meta = {e: {'cik': '0000021344'} for e in ['old', 'new']}
        self.pairs = {frozenset(['old', 'new'])}

    def run_row(self, row):
        return admit_rows([row], self.pairs, self.meta)[0]

    def test_new_overlay_does_not_rewrite_any_original_field(self):
        original = copy.deepcopy(self.row)
        out = self.run_row(self.row)
        self.assertTrue(out['conditional_annual_pair_admitted'])
        self.assertFalse(out['original_row']['tier3'])
        self.assertEqual(out['original_row'], original)
        out['original_row']['other']['preserved'] = False
        self.assertEqual(self.row, original)

    def test_other_blockers_and_failed_tier2_are_not_removed(self):
        for row in [dict(self.row, tier2=False), dict(self.row, reasons=[MISSING_CONTEXT, 'AMENDMENT'])]:
            self.assertFalse(self.run_row(row)['conditional_annual_pair_admitted'])
        self.assertEqual(self.run_row(dict(self.row, reasons=[MISSING_CONTEXT, 'AMENDMENT']))['remaining_row_reasons'], ['AMENDMENT'])

    def test_incomplete_duplicate_and_unknown_pairs_are_not_admitted(self):
        for ids in [[], ['new'], ['new', 'new'], ['old', 'other']]:
            self.assertFalse(self.run_row(dict(self.row, selected_event_ids=ids))['conditional_annual_pair_admitted'])

    def test_wrong_issuer_and_etf_are_rejected(self):
        for patch in [{'symbol': 'MSFT'}, {'cik': '0000789019'}, {'is_etf': True}]:
            with self.assertRaises(ValueError):
                self.run_row(dict(self.row, **patch))


if __name__ == '__main__':
    unittest.main()
