import unittest
from build_registry_identity_hints import candidate_matches,name_key
class RegistryIdentityTests(unittest.TestCase):
 def test_legal_suffix_and_punctuation_only(self):
  self.assertEqual(name_key('Acme, Inc.'),name_key('ACME INCORPORATED'))
  self.assertNotEqual(name_key('Acme Holdings'),name_key('Acme'))
 def test_reused_name_remains_ambiguous(self):
  rows=[dict(cik='1',name='ACME',start_date='2000',end_date='2010'),dict(cik='2',name='ACME',start_date='2020',end_date='')]
  self.assertEqual(len(candidate_matches({'names':['Acme Inc']},{name_key('ACME'):rows})),2)
 def test_no_ticker_join_or_fuzzy_match(self):
  self.assertEqual(candidate_matches({'names':['Acme Energy'],'symbol':'ACME'},{name_key('Acme'): [dict(cik='1',name='Acme')]}),[])
 def test_duplicate_vendor_names_do_not_duplicate_sources(self):
  rows=[dict(cik='1',name='Acme',start_date='',end_date='')]
  self.assertEqual(len(candidate_matches({'names':['Acme','Acme Inc']},{name_key('Acme'):rows})),1)
