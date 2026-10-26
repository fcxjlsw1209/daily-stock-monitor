import json,unittest
import core as c
import test_collect
from indexed_announcement import facts
class SupplementAudit(test_collect.ExportAudit):
 @classmethod
 def setUpClass(cls):cls.rows=json.loads((c.HERE/'candidate_supplement/events.json').read_text())
 def test_unique_dates_and_all_assigned_issuers_have_rows(self):
  keys=[(r['ticker'],r['fiscal_year'],r['fiscal_quarter'])for r in self.rows];self.assertEqual(len(keys),len(set(keys)));self.assertEqual({x['ticker']for x in self.rows},{'T','WFC','GS','VOD'})
  base=json.loads((c.HERE/'events.json').read_text());self.assertEqual(len(base),687);self.assertFalse(set(keys)&{(r['ticker'],r['fiscal_year'],r['fiscal_quarter'])for r in base})
  vod=[r for r in self.rows if r['ticker']=='VOD'];self.assertEqual(len(vod),14);self.assertEqual({(r['fiscal_year'],r['fiscal_quarter'])for r in vod},{(n//4,n%4+1)for n in range(2011*4+1,2014*4+3)})
 def test_publication_clause_beats_filing_header(self):
  text='AT&T Inc. Dallas, Texas. Filing date July 25, 2016. The registrant announced on July 21, 2016, its results of operations for the second quarter of 2016. Press release dated July 21, 2016 reporting financial results for the second quarter ended June 30, 2016.'
  self.assertEqual(facts(text)['release_date'],'2016-07-21')
  with self.assertRaisesRegex(ValueError,'No matching'):facts(text.replace('announced on','will announce on'))
 def test_no_quarter_end_from_calendar(self):
  with self.assertRaisesRegex(ValueError,'actual quarter-ended'):facts('AT&T Inc. Dallas, Texas. The registrant announced on July 21, 2016, its results of operations for the second quarter of 2016.')
if __name__=='__main__':unittest.main()
