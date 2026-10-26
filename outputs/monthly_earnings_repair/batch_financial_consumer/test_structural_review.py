"""Offline independently checks original byte hashes, fact quotes and scope decisions."""
import json,unittest,subprocess,sys,re
from functools import lru_cache
import core as c
import test_collect
from build_structural_review import D,FROZEN
from event_schema import normalize_event
@lru_cache(None)
def read_original(path):
 p=c.ROOT/path;b=p.read_bytes()
 if b.startswith(b'%PDF'):
  return c.clean(subprocess.check_output([c.PDF_PYTHON,'-c','import sys;from pypdf import PdfReader;print("\\n".join(p.extract_text() or "" for p in PdfReader(sys.argv[1]).pages))',str(p)],text=True))
 return c.text(b)
class StructuralReview(test_collect.ExportAudit):
 @classmethod
 def setUpClass(cls):
  cls.rows=json.loads((D/'events.json').read_text());cls.review=json.loads((D/'DISPOSITIONS.json').read_text())['dispositions']
 def test_unique_dates_and_all_assigned_issuers_have_rows(self):
  self.assertEqual(len(self.rows),1);e=self.rows[0];self.assertEqual((e['ticker'],e['release_date'],e['fiscal_year'],e['fiscal_quarter'],e['fiscal_period_end_actual']),('TM','2010-11-05',2011,2,'2010-09-30'))
 def test_toyota_filing_date_is_not_release_date(self):
  e=self.rows[0];text=read_original(e['fiscal_identity_evidence']['evidence_path']);self.assertIn('Filing date of quarterly securities report : November 12, 2010',text);self.assertEqual(e['release_date'],'2010-11-05');self.assertEqual(e['publication_timezone'],'Asia/Tokyo')
 def test_every_disposition_quote_matches_original(self):
  for r in self.review:
   for pr in r['evidence']:
    p=c.ROOT/pr['evidence_path'];self.assertEqual(c.sha(p.read_bytes()),pr['sha256']);self.assertIn(c.clean(pr['excerpt']),read_original(pr['evidence_path']),(r['symbol'],pr['excerpt']))
 def test_future_reports_are_strictly_after_candidates(self):
  for r in self.review:
   if r['kind']!='vendor_hint_wrong_date':
    for signal in r['signal_dates']:
     self.assertLess(r['prior_release'],signal);self.assertGreater(r['next_release'],signal)
  e=self.rows[0];n=normalize_event(dict(symbol='TM',asset_type='equity',official=dict(verified=True,publication_kind='release',actual_period_end=e['fiscal_period_end_actual'],publication_date=e['release_date'],publication_timezone=e['publication_timezone'],source_url=e['source_url'])));self.assertEqual(n['available_session'],'2010-11-08')
 def test_vendor_hints_and_sales_cannot_be_admitted(self):
  cost=[r for r in self.review if r['symbol']=='COST'];self.assertEqual(len(cost),2)
  for r in cost:self.assertFalse(r['admit_vendor_hint_as_earnings']);self.assertLess(r['actual_earnings_release'],r['vendor_hint_date']);self.assertNotEqual(r['monthly_sales_period_end'],r['fiscal_period_end_actual'])
  y24=next(r for r in cost if r['vendor_hint_date']=='2024-06-06');self.assertEqual(y24['monthly_sales_release'],'2024-06-05');self.assertTrue(all(e['ticker']!='COST' for e in self.rows))
 def test_requested_scope_and_frozen_snapshots(self):
  self.assertEqual(len(self.review),8);self.assertEqual({r['symbol']for r in self.review},{'HSBC','CHL','BUD','TM','COST'})
  for name,h in FROZEN.items():self.assertEqual(c.sha((c.HERE/name).read_bytes()),h)
  meta=json.loads((c.ROOT/'outputs/monthly_architecture_restart/META.json').read_text())
  for r in self.review:
   for scope in r['scopes']:
    membership=dict(zip(meta[scope]['dates'],meta[scope]['symbols']))
    for signal in r['signal_dates']:self.assertIn(r['symbol'],membership[signal])
if __name__=='__main__':unittest.main()
