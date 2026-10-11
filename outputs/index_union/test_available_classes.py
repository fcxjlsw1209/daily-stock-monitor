import unittest
from class_selection import select_available_classes
import test_class_selection as fixtures

class AvailableClassesTests(unittest.TestCase):
    def setUp(self):
        f=fixtures.ClassSelectionTests();f.setUp();self.f=f
    def run_case(self,availability):
        f=self.f
        return select_available_classes(f.members,f.bars,f.day,f.sessions,f.coverage,availability)
    def test_missing_class_does_not_block_observed_class(self):
        del self.f.coverage['B']
        got=self.run_case({'A':True,'B':False})
        self.assertEqual(got['selected'][0]['security_id'],'A')
        self.assertEqual(got['exclusions'][0]['security_id'],'B')
        self.assertFalse(got['exclusions'][0]['affects_existing_holdings'])
    def test_no_available_class_omits_issuer(self):
        self.assertEqual(self.run_case({'A':False,'B':False})['selected'],[])
    def test_unknown_is_not_silently_excluded(self):
        with self.assertRaises(ValueError):self.run_case({'A':True})
    def test_observed_classes_keep_strict_comparison(self):
        del self.f.coverage['B']
        with self.assertRaises(ValueError):self.run_case({'A':True,'B':True})
    def test_does_not_mutate_historical_membership(self):
        self.run_case({'A':False,'B':True})
        self.assertEqual(self.f.members[0]['security_ids'],['B','A'])

if __name__=='__main__':unittest.main()
