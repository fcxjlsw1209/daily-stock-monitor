import unittest
from extract_sp_events import extract,parse_effective_date
class ExtractionTests(unittest.TestCase):
 def table(self,convention='prior to the open of trading'):
  return '<div class="wd_news_body">'+convention+'</div><table><tr><td>Effective Date</td><td>Index Name</td><td>Action</td><td>Ticker</td></tr><tr><td rowspan="2">March 23, 2026</td><td>S&amp;P 100</td><td>Addition</td><td>MU</td></tr><tr><td>S&amp;P 500</td><td>Addition</td><td>VRT</td></tr></table>'
 def test_excludes_other_indices(self):
  r=extract(self.table());self.assertEqual([x['ticker'] for x in r],['MU']);self.assertEqual(r[0]['effective_session'],'2026-03-23')
 def test_no_inferred_after_close_date(self):self.assertEqual(extract(self.table('after the close of trading')),[])
 def test_no_identity_certification(self):self.assertFalse(extract(self.table())[0]['identity_verified'])

 def test_date_without_space_after_comma(self):
  self.assertEqual(parse_effective_date('March 24,2025'),'2025-03-24')
 def test_invalid_missing_year_rejected(self):
  for value in ['NaT','nan','March 24','0001-03-24']:
   with self.assertRaises(ValueError):parse_effective_date(value)
 def test_blank_date_continues_explicit_table_group(self):
  html=self.table().replace('rowspan="2"','rowspan="1"').replace('<tr><td>S&amp;P 500','<tr><td></td><td>S&amp;P 100')
  self.assertEqual([r['effective_session'] for r in extract(html)],['2026-03-23','2026-03-23'])

 def test_publisher_month_abbreviations(self):
  self.assertEqual(parse_effective_date('Sept 21, 2026'),'2026-09-21')
  self.assertEqual(parse_effective_date('Mar. 22, 2021'),'2021-03-22')

 def test_cached_real_publisher_date_variants(self):
  from pathlib import Path
  base=Path(__file__).resolve().parent
  for file,day,adds,deletes in [
   ('e2881c8dd628690c.html','2021-03-22',{'AVGO','TMUS','LIN'},{'ALL','KMI','SLB'}),
   ('5b3b788debee1d1c.html','2025-03-24',{'PLTR','ISRG','NOW'},{'DOW','KHC','F'})]:
   with self.subTest(file=file):
    rows=extract((base/'sources/sp_releases'/file).read_text())
    self.assertEqual({r['effective_session'] for r in rows},{day})
    self.assertEqual({r['ticker'] for r in rows if r['action']=='addition'},adds)
    self.assertEqual({r['ticker'] for r in rows if r['action']=='deletion'},deletes)
