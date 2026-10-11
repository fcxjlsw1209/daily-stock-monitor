import unittest
from build_foreign_cap_diagnostics import known_observation,candidate_cap

class ForeignCapTests(unittest.TestCase):
    def setUp(self):
        self.r=dict(as_of='2009-12-31',filed='2010-02-26',total_common_share_count=1000,ordinary_shares_per_ads=5)
    def test_adr_is_not_common_price(self):
        self.assertEqual(candidate_cap(self.r,50),10000)
    def test_missing_ratio_fails(self):
        r=dict(self.r);r.pop('ordinary_shares_per_ads')
        with self.assertRaises(ValueError):candidate_cap(r,50)
        self.assertEqual(candidate_cap(r,50,direct_common=True),50000)
    def test_ratio_publication_not_backdated(self):
        r=dict(self.r,adr_ratio_published='2010-03-01')
        self.assertIsNone(known_observation([r],'2010-03-01'))
        self.assertEqual(known_observation([r],'2010-03-02'),r)
    def test_later_disclosure_cannot_replace_earlier(self):
        later=dict(self.r,as_of='2010-03-30',filed='2010-03-31')
        self.assertEqual(known_observation([self.r,later],'2010-03-31'),self.r)
        self.assertEqual(known_observation([self.r,later],'2010-04-01'),later)
if __name__=='__main__':unittest.main()
