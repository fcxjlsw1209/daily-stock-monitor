import copy
import unittest
import after141_profitability_source_extract as m

class SourceContractTests(unittest.TestCase):
 def meta(self):
  raw=b'<html>original</html>';url='https://ir.amd.com/source'
  return raw,url,dict(status=200,url=url,final_url=url,sha256=m.sha(raw),bytes=len(raw),retrieved_at='2026-10-12T12:00:00+00:00',content_type='text/html;charset=UTF-8',public_unauthenticated=True)
 def test_html_receipt(self):
  raw,url,meta=self.meta();self.assertTrue(m.receipt_check(meta,raw,url,'html')['original_receipt_verified'])
 def test_hash_mismatch(self):
  raw,url,meta=self.meta()
  with self.assertRaises(ValueError):m.receipt_check(meta,raw+b' ',url,'html')
 def test_status_requires_success_integer(self):
  for status in [True,403,429,'200']:
   raw,url,meta=self.meta();meta['status']=status
   with self.subTest(status=status),self.assertRaises(ValueError):m.receipt_check(meta,raw,url,'html')
 def test_nonpublic_or_error_rejected(self):
  for field,value in [('public_unauthenticated',False),('error','failed')]:
   raw,url,meta=self.meta();meta[field]=value
   with self.assertRaises(ValueError):m.receipt_check(meta,raw,url,'html')
 def test_missing_or_naive_retrieval_rejected(self):
  for value in [None,'2026-10-12T12:00:00','wrong']:
   raw,url,meta=self.meta();meta['retrieved_at']=value
   with self.assertRaises(ValueError):m.receipt_check(meta,raw,url,'html')
 def test_wrong_type_size_or_url_rejected(self):
  for field,value in [('content_type','application/json'),('bytes',1),('final_url','https://evil.test/source')]:
   raw,url,meta=self.meta();meta[field]=value
   with self.assertRaises(ValueError):m.receipt_check(meta,raw,url,'html')
 def test_wrong_issuer_rejected(self):
  raw,url,meta=self.meta();url='https://elsewhere.test/a';meta.update(url=url,final_url=url)
  with self.assertRaises(ValueError):m.receipt_check(meta,raw,url,'html')
 def test_identity_pdf(self):
  raw=b'%PDF-original';url='https://www.apple.com/a.pdf';meta=dict(status=200,url=url,final_url=url,response_sha256=m.sha(raw),response_bytes=len(raw),retrieved_at='2026-10-12T12:00:00Z',content_type='application/pdf',public_unauthenticated=True,decoder='identity',decoded_sha256=m.sha(raw))
  self.assertTrue(m.receipt_check(meta,raw,url,'pdf')['original_receipt_verified'])
  meta['decoded_sha256']='0'*64
  with self.assertRaises(ValueError):m.receipt_check(meta,raw,url,'pdf')
 def test_calendar_strict_after(self):
  sessions=['2025-01-08','2025-01-10','2025-01-13']
  self.assertEqual(m.next_session('2025-01-08',sessions),'2025-01-10')
  self.assertEqual(m.next_session('2025-01-11',sessions),'2025-01-13')
 def test_calendar_rejects_bad_coverage(self):
  for day,sessions in [('2025-01-08',['2025-01-08']),('2025-01-07',['2025-01-08','2025-01-10']),('2025-01-08',['2025-01-10','2025-01-08']),('2025-01-08',['2025-01-08','2025-01-08','2025-01-10'])]:
   with self.assertRaises(ValueError):m.next_session(day,sessions)
 def test_local_escape_rejected(self):
  for path in ['/etc/passwd','../escape']:
   with self.assertRaises(ValueError):m.local_path(path)
 def amd_fixture(self):
  raw=b'<html><p>Advanced Micro Devices</p><p>SUNNYVALE, Calif. \xe2\x80\x94 Jan. 31, 2017</p></html>'
  spec=dict(fiscal_year=2016,fiscal_year_end='2016-12-31',release_date='2017-01-31',raw_sha256=m.sha(raw),source_url='https://ir.amd.com/original',raw_path='outputs/fake')
  event=dict(ticker='AMD',evidence_sha256=m.sha(raw),fiscal_quarter=4,fiscal_year=2016,fiscal_period_end_actual=spec['fiscal_year_end'],release_date=spec['release_date'],source_url=spec['source_url'],evidence_path=spec['raw_path'],precision='date_only',publication_timezone='America/Los_Angeles',first_public_time_verified=False,release_date_evidence=[dict(sha256=m.sha(raw),evidence_path=spec['raw_path'],source_url=spec['source_url'],excerpt='SUNNYVALE, Calif. - Jan. 31, 2017')])
  return raw,spec,event
 def test_amd_dateline_unicode_dash(self):
  raw,spec,event=self.amd_fixture();out=m.amd_publication(raw,spec,[event]);self.assertEqual(out['release_date'],'2017-01-31');self.assertFalse(out['original_event_first_public_time_verified'])
 def test_amd_no_future_date_substitution(self):
  raw,spec,event=self.amd_fixture();event['release_date']='2018-01-31'
  with self.assertRaises(ValueError):m.amd_publication(raw,spec,[event])
 def test_amd_duplicate_event_rejected(self):
  raw,spec,event=self.amd_fixture()
  with self.assertRaises(ValueError):m.amd_publication(raw,spec,[event,copy.deepcopy(event)])
 def test_amd_wrong_excerpt_rejected(self):
  raw,spec,event=self.amd_fixture();event['release_date_evidence'][0]['excerpt']='Jan. 30, 2017'
  with self.assertRaises(ValueError):m.amd_publication(raw,spec,[event])
 def apple_fixture(self):
  raw=b'<html><h1>Apple Reports Fourth Quarter Results</h1><span class="category-eyebrow__date">October 31, 2024</span><script type="application/ld+json">{"@type":"NewsArticle","datePublished":"2024-10-31"}</script><a href="/a.pdf">View PDF</a></html>'
  spec=dict(fiscal_year=2024,fiscal_year_end='2024-09-28',release_date='2024-10-31',raw_sha256='a'*64,source_url='https://www.apple.com/a.pdf')
  event=dict(source_sha256='a'*64,publication=dict(fiscal_quarter=4,release_date='2024-10-31',year_ends=[dict(end='2024-09-28',fiscal_year=2024)]),parsed=dict(statement=dict(current_end='2024-09-28')),parent_release_sha256=m.sha(raw))
  return raw,spec,event
 def test_apple_current_period_and_link(self):
  raw,spec,event=self.apple_fixture();self.assertTrue(m.apple_publication(raw,spec,'https://www.apple.com/release',{'x':event})['old_SEC_revenue_admission_not_inherited'])
 def test_apple_wrong_link_rejected(self):
  raw,spec,event=self.apple_fixture();spec['source_url']='https://www.apple.com/other.pdf'
  with self.assertRaises(ValueError):m.apple_publication(raw,spec,'https://www.apple.com/release',{'x':event})
 def test_apple_comparative_not_current_rejected(self):
  raw,spec,event=self.apple_fixture();event['parsed']['statement']['current_end']='2023-09-30'
  with self.assertRaises(ValueError):m.apple_publication(raw,spec,'https://www.apple.com/release',{'x':event})
 def test_apple_conflicting_published_date_rejected(self):
  raw,spec,event=self.apple_fixture();raw=raw.replace(b'"2024-10-31"',b'"2024-11-01"')
  with self.assertRaises(ValueError):m.apple_publication(raw,spec,'https://www.apple.com/release',{'x':event})

if __name__=='__main__':unittest.main()
