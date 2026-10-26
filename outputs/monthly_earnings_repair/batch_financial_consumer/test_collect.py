"""Offline semantic, provenance and conservative-availability audit; no network."""
import datetime as dt
import json
import re
from pathlib import Path
import sys
import tempfile
import unittest
from functools import lru_cache
from unittest.mock import patch
import core as c
from build_events import extract,reject_notice,notice_heading
sys.path.insert(0,str(c.HERE.parent))
from event_schema import normalize_event

@lru_cache(maxsize=2000)
def source_text(path):
 p=c.ROOT/path;b=p.read_bytes()
 if b.startswith(b'%PDF'):return c.clean(c.pdf_text(p))
 if p.suffix=='.xlsx':return c.workbook_text(p)
 if b.lstrip().startswith(b'{'):
  v=json.loads(b)
  if 'response'in v:return c.clean(re.sub(r'\s*\|\s*',' ',v['response']))
  if 'content'in v:return c.clean(v['title']+' '+c.text(v['content'].replace('\\r\\n',' ').replace('\\n',' ').replace('\\"','"')))
 return c.text(b)

def sample(body,ticker='NKE'):
 with tempfile.TemporaryDirectory() as td:
  root=Path(td);p=root/'release.html';p.write_text(body);url='https://issuer.example/release'
  c.dump(p.with_name(p.name+'.meta.json'),dict(url=url,status=200,sha256=c.sha(p.read_bytes()),retrieved_at='2026-10-12T22:00:00+00:00'))
  with patch.object(c,'ROOT',root),patch.object(c,'HERE',root):
   return extract(dict(url=url,evidence_path='release.html',evidence_sha256=c.sha(p.read_bytes())),ticker)

def nike():return 'NIKE reports first quarter fiscal 2012 results. BEAVERTON, September 22, 2011 - NIKE reported first quarter fiscal 2012 results. Three months ended August 31, 2011.'

class Semantics(unittest.TestCase):
 def test_non_calendar_fiscal_year(self):
  r=sample(nike());self.assertEqual((r['fiscal_year'],r['fiscal_quarter'],r['fiscal_period_end_actual']),(2012,1,'2011-08-31'))
 def test_disney_53_week_fiscal_year(self):
  r=sample('Walt Disney reports fourth quarter fiscal 2020 results. BURBANK, November 12, 2020 - Walt Disney reported fourth quarter 2020 results. Three months ended October 3, 2020.','DIS');self.assertEqual(r['fiscal_year'],2020)
 def test_modified_and_call_dates_not_release_instant(self):
  r=sample('dateModified October 3, 2026. Conference call September 23, 2011. '+nike());self.assertEqual(r['release_date'],'2011-09-22');self.assertIsNone(r['publication_timestamp'])
 def test_original_dateline_precedes_regional_republication_header(self):
  r=sample('<div class=\"wd_date\">October 2, 2011</div>'+nike());self.assertEqual(r['release_date'],'2011-09-22')
 def test_city_in_narrative_cannot_turn_table_date_into_dateline(self):
  body='<div class=\"wd_date\">September 22, 2011</div>'+nike().replace('BEAVERTON, September 22, 2011 - ','').replace('Three months ended',"BEAVERTON's new neighborhood. Three months ended")
  self.assertEqual(sample(body)['release_date'],'2011-09-22')
 def test_spaced_region_and_connecticut_datelines(self):
  r=sample('Abbott reports second quarter 2022 results. ABBOTT PARK , Ill. , July 20, 2022 - Abbott reported second quarter 2022 results. Three months ended June 30, 2022.','ABT');self.assertEqual(r['release_date'],'2022-07-20')
  r=sample('Philip Morris reports second quarter 2026 results. STAMFORD, CT --(BUSINESS WIRE)--Jul. 22, 2026 - Philip Morris reports second quarter 2026 results. Three months ended June 30, 2026.','PM');self.assertEqual(r['release_date'],'2026-07-22')
 def test_no_dateline_no_event(self):
  with self.assertRaisesRegex(ValueError,'dateline'):sample(nike().replace('BEAVERTON, September 22, 2011 - ','Conference call September 22, 2011. '))
 def test_scheduled_notice_rejected(self):
  with self.assertRaisesRegex(ValueError,'scheduled'):sample('NIKE will report first quarter results. '+nike())
 def test_logistics_notice_with_complete_dates_is_rejected(self):
  body='PepsiCo Confirms Timing and Updates Logistics for Release of First Quarter 2020 Financial Results. PURCHASE, N.Y., April 21, 2020. First quarter 2020 financial results for the quarter ended March 21, 2020.'
  with self.assertRaisesRegex(ValueError,'scheduled'):sample(body,'PEP')
 def test_actual_results_can_mention_next_quarter_schedule(self):
  r=sample(nike()+' NIKE plans to release second quarter results in December.');self.assertEqual(r['release_date'],'2011-09-22')
 def test_actual_end_required(self):
  with self.assertRaisesRegex(ValueError,'actual fiscal'):sample(nike().replace('Three months ended August 31, 2011.',''))
 def test_calendar_quarter_cannot_replace_nike_period(self):
  with self.assertRaisesRegex(ValueError,'mismatch'):sample(nike().replace('August 31','September 30').replace('September 22','October 22'))
 def test_three_column_reporting_table(self):
  r=sample('Wells Fargo reports first quarter 2016 results. SAN FRANCISCO, April 14, 2016 - Wells Fargo reported first quarter 2016 results. Quarter ended Mar 31, Dec 31, Mar 31, 2016 2015 2015.','WFC');self.assertEqual(r['fiscal_period_end_actual'],'2016-03-31')
 def test_source_stop_does_not_make_request(self):
  with tempfile.TemporaryDirectory()as td,patch.object(c,'HERE',Path(td)),patch('urllib.request.urlopen')as network:
   c.dump(Path(td)/'SOURCE_STOPS.json',{'issuer.example':{'status':403}})
   with self.assertRaisesRegex(RuntimeError,'Stopped'):c.fetch('https://issuer.example/release','NKE')
   network.assert_not_called()

class BACOriginalMirrors(unittest.TestCase):
 def test_original_announcement_statement_not_filing_header(self):
  from bac_mirror_parser import facts
  r=facts('Filed January 20, 2021. On January 19, 2021, Bank of America Corporation announced financial results for the fourth quarter and year ended December 31, 2020.', 'Bank of America Reports Q4-20 Financial Results')
  self.assertEqual(r['release_date'],'2021-01-19')
 def test_no_announcement_statement_no_event(self):
  from bac_mirror_parser import facts
  with self.assertRaisesRegex(ValueError,'No explicit'):facts('Bank of America Filed January 20, 2021. Fourth quarter ended December 31, 2020.', 'Bank of America Reports Q4-20 Financial Results')
 def test_mismatched_exhibit_rejected(self):
  from bac_mirror_parser import facts
  with self.assertRaisesRegex(ValueError,'No explicit'):facts('On January 19, 2021, Bank of America announced financial results for the fourth quarter and year ended December 31, 2020.', 'Bank of America Reports Q3-20 Financial Results')
 def test_repurchase_exhibit_not_selected(self):
  from bank_mirrors import select_press
  from bs4 import BeautifulSoup
  links=BeautifulSoup('<a href="bad">THE REPURCHASE PRESS RELEASE »</a><a href="good">EX-99.1 »</a>','html.parser').find_all('a')
  self.assertEqual(select_press(links)['href'],'good')

class ExportAudit(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.rows=json.loads((c.HERE/'events.json').read_text())
 def test_unique_dates_and_all_assigned_issuers_have_rows(self):
  keys=set();tickers=set()
  for r in self.rows:
   key=r['ticker'],r['fiscal_year'],r['fiscal_quarter'];self.assertNotIn(key,keys);keys.add(key);tickers.add(r['ticker'])
   self.assertTrue(c.START<=r['release_date']<=c.CUTOFF);self.assertLess(r['fiscal_period_end_actual'],r['release_date']);self.assertIn(r['fiscal_quarter'],[1,2,3,4])
  self.assertEqual(tickers,set('BAC HD KO MA T COST CMCSA DIS WFC C PM PEP HSBC NKE MCD PYPL ABT TMO'.split()))
 def test_raw_primary_hashes_metadata_and_evidence_class(self):
  for r in self.rows:
   with self.subTest(ticker=r['ticker'],date=r['release_date']):
    p=c.ROOT/r['evidence_path'];m=json.loads(p.with_name(p.name+'.meta.json').read_text());self.assertEqual(c.sha(p.read_bytes()),r['evidence_sha256']);self.assertEqual(m['sha256'],r['evidence_sha256']);self.assertEqual(m['retrieved_at'],r['source_retrieved_at']);self.assertIsNotNone(dt.datetime.fromisoformat(m['retrieved_at'].replace('Z','+00:00')).tzinfo)
    if r['source_class'].startswith('public_indexed'):
     self.assertTrue(r['indexed_source_only']);self.assertFalse(r['original_source_fully_cached']);self.assertIsNone(r['original_source_sha256']);self.assertFalse(m['raw_original_bytes_available'])
    else:self.assertEqual(m['status'],200);self.assertEqual(m['url'],r['source_url'])
 def test_all_fact_quotes_exist_in_cached_source(self):
  for r in self.rows:
   if r['source_class'].startswith('public_indexed'):
    response=json.loads((c.ROOT/r['evidence_path']).read_text())['response'];block=next(b for b in response.split('--------------------------------------------------------------------------------')if '('+r['source_url']+')'in b);reject_notice(dict(url=r['source_url'],notice_heading=block.strip().splitlines()[0]),c.clean(block))
   else:
    raw=(c.ROOT/r['evidence_path']).read_bytes();st=source_text(r['evidence_path']);reject_notice(dict(url=r['source_url'],notice_heading=notice_heading(raw,st)),st)
   proofs=r['release_date_evidence']+[r['fiscal_period_evidence'],r['publication_timezone_evidence']]
   if 'release_exhibit_identity_evidence'in r:proofs.append(r['release_exhibit_identity_evidence'])
   if 'quarter_package_evidence'in r:proofs.append(r['quarter_package_evidence'])
   if 'fiscal_calendar_evidence'in r:proofs.append(r['fiscal_calendar_evidence'])
   for proof in proofs:
    with self.subTest(ticker=r['ticker'],date=r['release_date'],path=proof['evidence_path']):
     p=c.ROOT/proof['evidence_path'];self.assertEqual(c.sha(p.read_bytes()),proof['sha256']);
     if proof.get('visual_verification'):
      vp=proof['visual_verification'];self.assertEqual(c.sha((c.ROOT/vp['evidence_path']).read_bytes()),vp['sha256']);self.assertIn(c.clean(proof['excerpt']),c.clean(vp['transcription']));self.assertEqual(vp['page_number'],1)
     else:self.assertIn(c.clean(re.sub(r'\s*\|\s*',' ',proof['excerpt'])),c.clean(re.sub(r'\s*\|\s*',' ',source_text(proof['evidence_path']))))
   identity=r['fiscal_identity_evidence'];self.assertEqual(c.sha((c.ROOT/identity['evidence_path']).read_bytes()),identity['sha256'])
   for k in ['quarter_excerpt','year_excerpt']:self.assertIn(c.clean(re.sub(r'\s*\|\s*',' ',identity[k])),c.clean(re.sub(r'\s*\|\s*',' ',source_text(identity['evidence_path']))))
 def test_all_rows_pass_shared_conservative_schema(self):
  for r in self.rows:
   with self.subTest(ticker=r['ticker'],date=r['release_date']):
    self.assertEqual(r['precision'],'date_only');self.assertIsNone(r['publication_timestamp']);self.assertFalse(r['first_public_time_verified'])
    n=normalize_event(dict(symbol=r['ticker'],asset_type='equity',official=dict(verified=True,publication_kind='release',actual_period_end=r['fiscal_period_end_actual'],publication_date=r['release_date'],publication_timezone=r['publication_timezone'],source_url=r['source_url'])))
    self.assertEqual(n['status'],'ready',n);self.assertIsNone(n['publication_at']);self.assertFalse(n['known_by_is_actual_publication']);self.assertGreater(n['available_session'],r['release_date'])

if __name__=='__main__':unittest.main()
