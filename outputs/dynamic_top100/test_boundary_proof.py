import unittest
from boundary_proof import prove_membership
class BoundsTests(unittest.TestCase):
 def rows(self):
  return [dict(issuer_id=s,lower_usd=lo,upper_usd=hi,session='2020-01-02',cap_basis='total_company_equity_usd',source='synthetic',source_sha256='test',bound_verified=True,available_at='2020-01-02T22:00:00Z') for s,lo,hi in [('A',100,300),('B',90,250),('C',10,80)]]
 def runproof(self,r):return prove_membership(r,['A','B','C'],'2020-01-02','2020-01-03T01:00:00Z',2)
 def test_internal_order_not_needed(self):self.assertEqual(self.runproof(self.rows())['members'],['A','B'])
 def test_boundary_overlap_blocks(self):
  r=self.rows();r[2]['upper_usd']=110;self.assertFalse(self.runproof(r)['membership_proven'])
 def test_unknown_outsider_blocks(self):
  r=self.rows();r[2].update(lower_usd=0,upper_usd=None);self.assertFalse(self.runproof(r)['membership_proven'])
 def test_missing_issuer_blocks(self):
  with self.assertRaises(ValueError):self.runproof(self.rows()[:2])
 def test_future_evidence_blocks(self):
  r=self.rows();r[0]['available_at']='2020-01-04T00:00:00Z'
  with self.assertRaises(ValueError):self.runproof(r)
 def test_unverified_bounds_block(self):
  r=self.rows();r[0]['bound_verified']=False
  with self.assertRaises(ValueError):self.runproof(r)
 def test_equal_boundary_unresolved(self):
  r=self.rows();r[2]['upper_usd']=90;self.assertFalse(self.runproof(r)['membership_proven'])
