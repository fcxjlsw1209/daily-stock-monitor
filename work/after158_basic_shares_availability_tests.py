"""Pure date/identity/selection tests; no actual share or strategy values."""
from copy import deepcopy
from datetime import date,timedelta
import io
import sys
import unittest
import after158_basic_shares_availability as m

class Availability(unittest.TestCase):
 def setUp(self):
  self.days=['2024-11-27','2024-11-29','2024-12-02','2024-12-03','2024-12-04']
  self.closes={d:{'iso':d+'T'+('18'if d=='2024-11-29'else'21')+':00:00+00:00','ns':m.ns(d+'T'+('18'if d=='2024-11-29'else'21')+':00:00+00:00')}for d in self.days}
 def event(self,name='new',period='2024-06-30',release='2024-11-27'):
  e=dict(original_index=0,event_id=name,symbol='X',release_date=release,periods={'current':{'period_end':period},'comparison':{'period_end':'2023-06-30'}},identity={'security_id':'X','issuer_id':'X:Issuer','identity_valid_from':'2024-01-01','identity_valid_through':'2025-12-31'},inherited_holds={k:False for k in m.HOLD_KEYS},extraction_success=True,split_review_required=True,same_share_basis=None)
  e['inherited_holds']['inherited_rejection_codes']=[];return m.enrich([e],self.days,self.closes)[0]
 def sel(self,events,cutoff='2024-12-02',etf=False):return m.select(events,'X',cutoff,m.ns(cutoff+'T21:00:00+00:00'),etf)
 def test_two_strictly_following_sessions(self):self.assertEqual(m.known_clock('2024-11-27',self.days,self.closes)['session'],'2024-12-02')
 def test_weekend_release(self):self.assertEqual(m.known_clock('2024-11-30',self.days,self.closes)['session'],'2024-12-03')
 def test_holiday_release(self):self.assertEqual(m.known_clock('2024-11-28',self.days,self.closes)['session'],'2024-12-02')
 def test_early_close_is_actual(self):self.assertEqual(m.known_clock('2024-11-26',self.days,self.closes)['iso'],'2024-11-29T18:00:00+00:00')
 def test_no_calendar_fallback(self):
  with self.assertRaises(ValueError):m.known_clock('2024-12-03',self.days,self.closes)
 def test_known_boundary_inclusive(self):self.assertTrue(self.sel([self.event()])['metadata_alignment_pass'])
 def test_future_not_selected(self):self.assertEqual(self.sel([self.event()],cutoff='2024-11-29')['selected_event_id'],None)
 def test_bad_latest_no_older_fallback(self):
  new=self.event();new['extraction_success']=False;old=self.event('old','2024-03-31');r=self.sel([old,new]);self.assertEqual(r['selected_event_id'],'new');self.assertIn('LATEST_EXTRACTION_FAILED',r['reasons'])
 def test_future_quality_not_considered(self):
  old=self.event();future={'symbol':'X','known_at_ns':m.ns('2025-01-01T00:00:00Z')};self.assertTrue(self.sel([old,future])['metadata_alignment_pass'])
 def test_duplicate_latest_blocks(self):self.assertIn('AMBIGUOUS_LATEST_ORIGINAL_ANNUAL_EVENT',self.sel([self.event(),self.event('other')])['reasons'])
 def test_later_evidence_no_fallback(self):
  e=self.event();e['usable_at_ns']+=86400*10**9;r=self.sel([e]);self.assertIn('LATEST_NOT_YET_USABLE',r['reasons'])
 def test_exact_age_550(self):
  e=self.event(period=(date(2024,12,2)-timedelta(days=550)).isoformat());e['periods']['comparison']['period_end']='2022-01-01';self.assertTrue(self.sel([e])['metadata_alignment_pass'])
 def test_age_551_blocked(self):
  e=self.event(period=(date(2024,12,2)-timedelta(days=551)).isoformat());self.assertIn('LATEST_PERIOD_OLDER_THAN_550_DAYS',self.sel([e])['reasons'])
 def test_identity_end_inclusive(self):
  e=self.event();e['identity']['identity_valid_through']='2024-12-02';self.assertTrue(self.sel([e])['metadata_alignment_pass'])
 def test_identity_expired_blocks_latest(self):
  e=self.event();e['identity']['identity_valid_through']='2024-12-01';self.assertIn('ORIGINAL_IDENTITY_DOES_NOT_COVER_CUTOFF',self.sel([e])['reasons'])
 def test_identity_symbol_no_alias(self):
  e=self.event();e['identity']['security_id']='Y';self.assertFalse(self.sel([e])['metadata_alignment_pass'])
 def test_etf_not_matched(self):self.assertEqual(self.sel([self.event()],etf=True)['reasons'],['ETF_NOT_APPLICABLE'])
 def test_unrelated_issuer(self):
  e=self.event();e['symbol']='Y';self.assertEqual(self.sel([e])['reasons'],['ISSUER_OUTSIDE_FIXED_44_EVENT_SCOPE'])
 def test_metadata_pass_not_admission(self):
  r=self.sel([self.event()]);self.assertTrue(r['metadata_alignment_pass']);self.assertFalse(r['conditional_source_ready']);self.assertFalse(r['source_admitted']);self.assertTrue(r['split_review_required'])
 def test_mutation_free(self):
  es=[self.event()];old=deepcopy(es);self.sel(es);self.assertEqual(es,old)
 def test_financial_json_read_forbidden(self):
  with self.assertRaisesRegex(ValueError,'never parsed'):m.read('outputs/after158_basic_shares_extraction_v1/RAW_SHARES.json')
 def test_invalid_release(self):
  with self.assertRaises(ValueError):m.known_clock('2024-02-30',self.days,self.closes)
 def test_source_holds_preserved(self):
  e=self.event();e['inherited_holds']['inherited_rejection_codes']=['unknown_comparison_duration'];r=self.sel([e]);self.assertTrue(r['metadata_alignment_pass']);self.assertIn('INHERITED:unknown_comparison_duration',r['unresolved_holds'])
 def grid(self):
  days=[(date(2024,1,1)+timedelta(days=i)).isoformat()for i in range(300)if(date(2024,1,1)+timedelta(days=i)).weekday()<5]
  close={d:{'iso':d+'T21:00:00+00:00','ns':m.ns(d+'T21:00:00+00:00')}for d in days}
  etfs=['ETF'+str(i)for i in range(17)];names=['X']+['S'+str(i)for i in range(31)]+etfs
  meta={'dates':days[50:172],'symbols':[names[:]for _ in range(122)],'etfs':etfs}
  rows=[dict(signal_index=i,candidate_index=j,symbol=s,decision_date=d,decision_cutoff=close[d]['iso'],decision_cutoff_ns=close[d]['ns'],untouched_original_flags={'eligible':False,'reason':'original'})for i,d in enumerate(meta['dates'])for j,s in enumerate(names)]
  return meta,rows,days,close
 def test_all5978_original_rows_and_tminus42(self):
  meta,rows,days,close=self.grid();original=deepcopy(rows);out=m.map_grid(meta,rows,[],days,close)
  self.assertEqual(len(out),5978);self.assertEqual(rows,original);self.assertEqual([r['original_row']for r in out],original)
  for i in range(122):self.assertEqual(out[i*49]['cutoff_date'],days[i+8])
  counts=m.coverage(out);self.assertEqual(counts['all122']['stock_slots'],3904);self.assertEqual(counts['first88']['candidate_slots'],4312);self.assertEqual(counts['later34']['candidate_slots'],1666)
 def test_bad_original_close_hard_fails(self):
  meta,rows,days,close=self.grid();rows[7]['decision_cutoff_ns']+=1
  with self.assertRaisesRegex(ValueError,'changed'):m.map_grid(meta,rows,[],days,close)
 def test_short_calendar_not_calendar_day_fallback(self):
  meta,rows,days,close=self.grid();days=days[49:]
  with self.assertRaises(ValueError):m.map_grid(meta,rows,[],days,close)

if __name__=='__main__':
 if sys.argv[1:]==['--record']:
  pins=m.code_pins();stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Availability));m.verify(pins)
  m.write_once(m.OUT+'/TESTS.json',{'passed':result.wasSuccessful(),'synthetic_only':True,'tests_run':result.testsRun,'implementation_hashes':pins,'log':stream.getvalue(),'share_values_read':False});print(stream.getvalue());sys.exit(0 if result.wasSuccessful()else 1)
 else:unittest.main()
