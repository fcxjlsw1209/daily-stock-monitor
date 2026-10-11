import unittest,tempfile,hashlib
from pathlib import Path
import pandas as pd
from research_scope import evidence,check_nested_sources,apply_scope,assumed_payment
class ResearchScopeTests(unittest.TestCase):
 def test_nested_tamper(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td);f=p/'a';f.write_text('valid');r={'source_file':'a','source_sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
   self.assertEqual(check_nested_sources({'child':[r]},p),1)
   f.write_text('changed')
   with self.assertRaises(ValueError):check_nested_sources({'child':[r]},p)
 def test_unknown_identity_only_removes_candidates(self):
  dates=pd.bdate_range('2015-07-01',periods=6)
  d={'eligible':pd.DataFrame(True,index=dates,columns=['GOOD','UNKNOWN','KHC']), 'close':pd.DataFrame(10.,index=dates,columns=['GOOD','UNKNOWN','KHC'])}
  out,excluded=apply_scope(d,{'GOOD':'i','KHC':'j'})
  self.assertFalse(out['eligible']['UNKNOWN'].any());self.assertTrue(out['eligible']['GOOD'].all())
  self.assertFalse(out['eligible'].loc['2015-07-02':'2015-07-06','KHC'].any());self.assertTrue(out['eligible'].loc['2015-07-07','KHC'])
  pd.testing.assert_frame_equal(d['close'],out['close']);self.assertTrue(d['eligible'].all().all())
 def test_ea_alias_ambiguity(self):
  d={'eligible':pd.DataFrame(True,index=pd.bdate_range('2026-08-03',periods=3),columns=['research_equity_ERTS'])}
  o,_=apply_scope(d,{})
  self.assertEqual(o['eligible'].iloc[:,0].tolist(),[True,False,True])
 def test_settlement_is_assumption_not_evidence(self):
  dates=pd.to_datetime(['2020-01-02','2020-01-03','2020-01-06','2020-01-07','2020-01-08'])
  events=[{'execution_session':'2020-01-02','cash_per_share':50,'payment_session':None}]
  r=assumed_payment(events,dates)
  self.assertEqual(r[0]['payment_session'],'2020-01-07');self.assertIn('not_observed',r[0]['payment_timing_basis']);self.assertIsNone(events[0]['payment_session'])
if __name__=='__main__':unittest.main()
