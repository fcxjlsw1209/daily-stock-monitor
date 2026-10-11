import unittest
from reconstruct_sp_baseline import reconstruct
class BaselineTests(unittest.TestCase):
    def test_replacements_only(self):
        r=dict(deletion='A',addition='B',known_before='2009-12-01',completed_before='2009-12-31')
        self.assertEqual(reconstruct(['A','C'],[r]),['B','C'])
    def test_future_substitution_rejected(self):
        for key in ['known_before','completed_before']:
            r=dict(deletion='A',addition='B',known_before='2009-12-01',completed_before='2009-12-31')
            r[key]='2010-01-04'
            with self.assertRaises(ValueError):reconstruct(['A'],[r])
    def test_invalid_chain_rejected(self):
        r=dict(deletion='X',addition='B',known_before='2009-12-01',completed_before='2009-12-31')
        with self.assertRaises(ValueError):reconstruct(['A'],[r])
