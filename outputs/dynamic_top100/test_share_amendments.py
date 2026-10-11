import unittest
import pandas as pd
from normalize_expanded_evidence import normalize
from resolve_share_amendments import resolve, correct_context_dates


class AmendmentTests(unittest.TestCase):
    def setUp(self):
        self.raw = pd.DataFrame([
            dict(cik='0000000001', taxonomy='dei', concept='EntityCommonStockSharesOutstanding',
                 filed='2013-05-07', end='2013-04-30', val=value, accn=acc,
                 form=form, source_file='fixture')
            for value, acc, form in [(155, 'old', '10-Q'), (134, 'new', '10-Q/A')]])
        self.events = normalize(self.raw)
        self.fix = dict(cik='0000000001', taxonomy='dei', concept='EntityCommonStockSharesOutstanding',
                        filed='2013-05-07', reported_as_of='2013-04-30', original_accession='old',
                        amendment_accession='new', expected_values=[155, 134], corrected_shares=134,
                        source='https://example.org/reviewed', verification='explicit_cover_correction')

    def test_corrected_value_not_backdated_or_certified(self):
        result = resolve(self.events, self.raw, [self.fix])
        self.assertEqual(result.iloc[0].shares, 134)
        self.assertFalse(result.iloc[0].conflict)
        self.assertTrue(result.iloc[0].original_conflict)
        self.assertFalse(result.iloc[0].usable_for_rank)
        self.assertEqual(result.iloc[0].available_no_earlier_than, self.events.iloc[0].available_no_earlier_than)
        self.assertTrue(self.events.iloc[0].conflict)

    def test_unreviewed_conflict_remains(self):
        self.assertTrue(resolve(self.events, self.raw, []).iloc[0].conflict)

    def test_changed_data_or_wrong_amendment_rejected(self):
        for column, value in [('accn', 'different'), ('val', 777), ('form', '10-Q')]:
            raw = self.raw.copy()
            raw.loc[1, column] = value
            with self.assertRaises(ValueError):
                resolve(self.events, raw, [self.fix])

    def test_unreviewed_evidence_rejected(self):
        with self.assertRaises(ValueError):
            resolve(self.events, self.raw, [{**self.fix, 'verification': 'guess'}])

    def test_context_date_preserves_later_publication(self):
        fix = dict(cik='0000000001', taxonomy='dei', concept='EntityCommonStockSharesOutstanding',
                   filed='2013-05-07', accn='old', observed_end='2013-04-30',
                   corrected_end='2013-02-01', val=155, verification='original_cover_date', source='https://example.org')
        raw = correct_context_dates(self.raw, [fix])
        events = normalize(raw)
        self.assertFalse(events.conflict.any())
        self.assertEqual(set(events.reported_as_of), {'2013-02-01', '2013-04-30'})
        self.assertTrue(events.filed.eq('2013-05-07').all())
        self.assertTrue(raw.original_end.eq('2013-04-30').all())
        self.assertTrue(self.raw.end.eq('2013-04-30').all())
        with self.assertRaises(ValueError):
            correct_context_dates(self.raw, [{**fix, 'corrected_end': '2014-01-01'}])
