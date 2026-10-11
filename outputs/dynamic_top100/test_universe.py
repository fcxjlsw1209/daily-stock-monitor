import unittest,tempfile
from pathlib import Path
import pandas as pd
from universe import POLICY,select,freeze,rank_factor,require_history
class PITTests(unittest.TestCase):
 def fixture(self):
  rows=pd.DataFrame([dict(issuer_id=f'I{i:03}',security_id=f'S{i:03}',symbol=f'T{i:03}',market_cap_usd=1000-i,price_date='2020-01-02',available_at='2020-01-02T21:10:00Z',sector='Technology',exchange='NASDAQ',instrument='common_stock',cap_basis='total_company_equity_usd') for i in range(101)])
  cov=dict(session='2020-01-02',session_close='2020-01-02T21:00:00Z',policy=POLICY,complete_active_roster=True,foreign_issuers_included=True,company_cap_basis_verified=True,security_identity_verified=True,sector_history_verified=True,source='synthetic test only',source_sha256='test',knowledge_basis='provider_as_reported',active_issuer_ids=list(rows.issuer_id))
  return rows,cov
 def runselect(self,r,c):return select(r,c,'2020-01-02','2020-01-03T01:00:00Z','2020-01-03T14:30:00Z')
 def test_future_revision_does_not_change_past(self):
  r,c=self.fixture();base=self.runselect(r,c);future=r.iloc[[-1]].copy();future['market_cap_usd']=999999;future['available_at']='2020-01-04T01:00:00Z'
  self.assertEqual(base,self.runselect(pd.concat([r,future]),c))
 def test_contemporaneous_revision_changes_membership(self):
  r,c=self.fixture();r.loc[100,'market_cap_usd']=9999;self.assertEqual(self.runselect(r,c)['members'][0]['issuer_id'],'I100')
 def test_missing_company_blocks_not_rank101_fill(self):
  r,c=self.fixture()
  with self.assertRaises(ValueError):self.runselect(r.iloc[1:],c)
 def test_duplicate_classes_rejected(self):
  r,c=self.fixture()
  with self.assertRaises(ValueError):self.runselect(pd.concat([r,r.iloc[[0]]]),c)
 def test_wrong_cap_basis_rejected(self):
  r,c=self.fixture();r.loc[0,'cap_basis']='ADR_float'
  with self.assertRaises(ValueError):self.runselect(r,c)
 def test_late_capture_rejected(self):
  r,c=self.fixture();c.update(knowledge_basis='contemporaneous_capture',captured_at='2020-01-04T00:00:00Z')
  with self.assertRaises(ValueError):self.runselect(r,c)
 def test_freeze_immutable(self):
  r,c=self.fixture();x=self.runselect(r,c)
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'snapshot.json';freeze(p,x);freeze(p,x)
   with self.assertRaises(ValueError):freeze(p,{**x,'universe_count':999})
 def test_rank_within_members_only(self):
  v=pd.DataFrame([[1.,2.,1000.]],columns=['A','B','C']);m=pd.DataFrame([[True,True,False]],columns=v.columns);good=m|True
  out=rank_factor(v,m,good);self.assertEqual(out.at[0,'B'],1.);self.assertTrue(pd.isna(out.at[0,'C']))
 def test_no_rank101_replacement_on_price_filter(self):
  r,c=self.fixture();x=self.runselect(r,c);self.assertNotIn('I100',[m['issuer_id'] for m in x['members']]);self.assertEqual(len(x['members']),100)
 def test_incomplete_history_refuses_backtest(self):
  with self.assertRaises(ValueError):require_history([],['2020-01-02'],{})
if __name__=='__main__':unittest.main()
