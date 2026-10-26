import unittest,json
import core as c
import test_collect
from indexed_issued import facts
class CandidateV2(test_collect.ExportAudit):
 @classmethod
 def setUpClass(cls):cls.rows=json.loads((c.HERE/'candidate_supplement_v2/events.json').read_text())
 def test_unique_dates_and_all_assigned_issuers_have_rows(self):
  keys=[(e['ticker'],e['fiscal_year'],e['fiscal_quarter'])for e in self.rows];self.assertEqual(len(keys),len(set(keys)));self.assertEqual({e['ticker']for e in self.rows},{'T','WFC','PM','HSBC'})
  base=json.loads((c.HERE/'events.json').read_text())+json.loads((c.HERE/'candidate_supplement/events.json').read_text());self.assertEqual(len(base),721);self.assertFalse(set(keys)&{(e['ticker'],e['fiscal_year'],e['fiscal_quarter'])for e in base})
  t={(e['fiscal_year'],e['fiscal_quarter'])for e in base+self.rows if e['ticker']=='T'};self.assertTrue({(n//4,n%4+1)for n in range(2010*4+3,2018*4+1)}<=t)
 def test_wfc_announcement_not_filing_header(self):
  t='Filed April 21, 2011. San Francisco. On April 20, 2011, Wells Fargo & Company (the Company) issued a press release regarding its results of operations and financial condition for the quarter ended March 31, 2011 and its 1Q11 Quarterly Supplement.'
  r=facts(t,'WFC');self.assertEqual(r['release_date'],'2011-04-20');self.assertEqual(r['fiscal_period_end_actual'],'2011-03-31')
  with self.assertRaisesRegex(ValueError,'quarterly supplement'):facts(t.replace('1Q11','2Q11'),'WFC')
 def test_pm_future_notice_or_missing_actual_end_rejected(self):
  t='New York. On April 21, 2011, Philip Morris International Inc. (the Company) issued a press release announcing its financial results for the quarter ended March 31, 2011.'
  self.assertEqual(facts(t,'PM')['fiscal_quarter'],1)
  with self.assertRaisesRegex(ValueError,'No explicit'):facts(t.replace('issued','will issue'),'PM')
  with self.assertRaisesRegex(ValueError,'No explicit'):facts(t.replace('quarter ended March 31, 2011','first quarter of 2011'),'PM')
if __name__=='__main__':unittest.main()
