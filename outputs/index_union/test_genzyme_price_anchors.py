import tempfile
import unittest
from pathlib import Path

import pandas as pd

from audit_genzyme_price_anchors import compare_anchor, verify_source


class PriceAnchorTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame([
            dict(date='2011-04-04', open=2.30, close=2.35),
            dict(date='2011-04-11', open=2.45, close=2.40)])
        self.anchor = dict(date='2011-04-04', value=2.35,
                           source_price_kind='quoted_price_unspecified',
                           comparison_field='close')

    def test_unspecified_price_is_not_closing_or_opening_proof(self):
        result = compare_anchor(self.frame, self.anchor)
        self.assertTrue(result['value_matches'])
        self.assertFalse(result['price_field_corroborated'])

    def test_explicit_close_is_corroborated(self):
        self.anchor['source_price_kind'] = 'closing_price'
        self.assertTrue(compare_anchor(self.frame, self.anchor)['price_field_corroborated'])

    def test_close_cannot_prove_open_even_if_numbers_match(self):
        self.anchor.update(value=2.30, source_price_kind='closing_price', comparison_field='open')
        result = compare_anchor(self.frame, self.anchor)
        self.assertTrue(result['value_matches'])
        self.assertFalse(result['price_field_corroborated'])

    def test_different_session_is_not_replaced_by_matching_neighbor(self):
        self.anchor['date'] = '2011-04-11'
        self.assertFalse(compare_anchor(self.frame, self.anchor)['value_matches'])

    def test_missing_date_is_not_filled(self):
        self.anchor['date'] = '2011-04-05'
        self.assertFalse(compare_anchor(self.frame, self.anchor)['value_matches'])

    def test_duplicate_date_is_ambiguous(self):
        frame = pd.concat([self.frame, self.frame.iloc[:1]], ignore_index=True)
        self.assertFalse(compare_anchor(frame, self.anchor)['value_matches'])

    def test_nonfinite_price_is_not_a_match(self):
        self.frame.loc[0, 'close'] = float('nan')
        result = compare_anchor(self.frame, self.anchor)
        self.assertFalse(result['value_matches'])
        self.assertIsNone(result['vendor_value'])

    def test_changed_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'source.json'
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                verify_source(path, dict(sha256='incorrect'))


if __name__ == '__main__':
    unittest.main()
