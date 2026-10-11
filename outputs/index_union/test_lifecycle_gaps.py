import unittest
from audit_lifecycle_gaps import classify
class LifecycleGapTests(unittest.TestCase):
 def test_quarantine_is_not_halt_proof(self):
  self.assertEqual(classify('2020-01-02',{'quarantine_from':'2020-01-02'}),'terminal_timing_or_valuation_unresolved')
 def test_last_trading_day_still_needs_price(self):
  e={'quarantine_from':'2020-01-03','last_trading_date':'2020-01-02'}
  self.assertEqual(classify('2020-01-02',e),'unresolved_quote_or_identity')
  self.assertEqual(classify('2020-01-03',e),'verified_post_trading_action_valuation_required')
 def test_absent_evidence_not_excused(self):
  self.assertEqual(classify('2020-01-02',None),'unresolved_quote_or_identity')
if __name__=='__main__':unittest.main()
