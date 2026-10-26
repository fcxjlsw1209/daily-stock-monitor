"""Synthetic-only source-gate tests, no real financial inputs."""
from copy import deepcopy
import unittest
import after158_basic_shares_admission as m

def sample():
 p={'current':dict(period_start='2020-01-01',period_end='2020-12-31',days_inclusive=366),
    'comparison':dict(period_start='2019-01-01',period_end='2019-12-31',days_inclusive=365)}
 e=dict(event_id='AAA:2020',symbol='AAA',fiscal_year=2020,periods=p,release_date='2021-01-20',
  inherited_holds={'inherited_rejection_codes':[],'first_public_version_verified':False},identity={'security_id':'AAA'},cells={})
 for side in p:e['cells'][side]=dict(value_in_shares='1000',unit={'share_specific':True},semantic_basis='explicit_basic_weighted_average_shares',period_end=p[side]['period_end'])
 a=deepcopy(e);a['known_at']='2021-01-22T21:00:00+00:00'
 b=dict(event_id=e['event_id'],symbol='AAA',periods=p,status='unique_same_accession_annual_pair',selected_accession='one',selected_filed='2021-02-01')
 for side,key in (('current','current_selected'),('comparison','prior_selected')):
  b[key]=dict(accn='one',filed=b['selected_filed'],start=p[side]['period_start'],end=p[side]['period_end'])
 c=dict(event_id=e['event_id'],symbol='AAA',metadata_status=b['status'],selected_accession='one',selected_filed=b['selected_filed'],cells={s:dict(equal=True,native_rounding_interval_compatible=True) for s in p})
 def clock(d):return {'2021-01-20':a['known_at'],'2021-02-01':'2021-02-03T21:00:00+00:00'}[d]
 return e,a,b,c,clock
def row(e,cutoff='2021-02-28T21:00:00+00:00'):
 return dict(signal_index=1,candidate_index=0,symbol='AAA',cutoff=cutoff,availability=dict(selected_event_id=e['event_id'],reasons=[]))

class Test(unittest.TestCase):
 def test_clean_overlay(self):
  x=m.event_overlay(*sample());self.assertTrue(x['conditional_source_ready']);self.assertFalse(x['strict_source_admitted'])
 def test_later_filing_clock(self):
  x=m.event_overlay(*sample());self.assertEqual(x['usable_at'],'2021-02-03T21:00:00+00:00')
 def test_holds_block(self):
  v=sample();v[0]['inherited_holds']['inherited_rejection_codes']=['old'];v[1]['inherited_holds']=deepcopy(v[0]['inherited_holds'])
  self.assertFalse(m.event_overlay(*v)['conditional_source_ready'])
 def test_unknown_start_blocks(self):
  v=sample();v[0]['periods']['comparison']['period_start']=None;v[1]['periods']=deepcopy(v[0]['periods'])
  self.assertFalse(m.event_overlay(*v)['conditional_source_ready'])
 def test_nonconsecutive_blocks(self):
  v=sample();v[0]['periods']['current']['period_start']='2020-01-02';v[0]['periods']['current']['days_inclusive']=365;v[1]['periods']=deepcopy(v[0]['periods']);v[2]['current_selected']['start']='2020-01-02'
  self.assertFalse(m.event_overlay(*v)['conditional_source_ready'])
 def test_period_end_conflict(self):
  v=sample();v[2]['prior_selected']['end']='2019-12-30'
  with self.assertRaises(ValueError):m.event_overlay(*v)
 def test_accession_conflict(self):
  v=sample();v[3]['selected_accession']='two'
  with self.assertRaises(ValueError):m.event_overlay(*v)
 def test_mismatch_blocks(self):
  v=sample();v[3]['cells']['current']={'equal':False,'native_rounding_interval_compatible':False}
  self.assertFalse(m.event_overlay(*v)['conditional_source_ready'])
 def test_native_rounding_without_replacement(self):
  v=sample();v[3]['cells']['current']['equal']=False
  x=m.event_overlay(*v);self.assertTrue(x['conditional_source_ready']);self.assertEqual(x['current_shares'],'1000')
 def test_bad_counts_fail(self):
  for number in ('0','-1','NaN','Infinity'):
   v=sample();v[0]['cells']['current']['value_in_shares']=number
   with self.assertRaises(ValueError):m.event_overlay(*v)
 def test_diluted_fails(self):
  v=sample();v[0]['cells']['current']['semantic_basis']='diluted'
  with self.assertRaises(ValueError):m.event_overlay(*v)
 def test_nonshare_unit_fails(self):
  v=sample();v[0]['cells']['current']['unit']['share_specific']=False
  with self.assertRaises(ValueError):m.event_overlay(*v)
 def test_cell_date_conflict_fails(self):
  v=sample();v[0]['cells']['current']['period_end']='2021-01-01'
  with self.assertRaises(ValueError):m.event_overlay(*v)
 def test_original_unchanged(self):
  v=sample();before=deepcopy(v[:4]);m.event_overlay(*v);self.assertEqual(v[:4],before)
 def test_not_yet_corroborated(self):
  e=m.event_overlay(*sample());r=row(e,'2021-01-30T21:00:00+00:00')
  self.assertFalse(m.map_row(r,[e])['conditional_source_ready'])
 def test_usable_boundary(self):
  e=m.event_overlay(*sample());self.assertTrue(m.map_row(row(e,e['usable_at']),[e])['conditional_source_ready'])
 def test_initial_partial_fallback(self):
  e=m.event_overlay(*sample());r=row(e);r['signal_index']=0
  self.assertFalse(m.map_row(r,[e])['conditional_source_ready'])
 def test_bad_latest_blocks_old(self):
  e=m.event_overlay(*sample());old=deepcopy(e);old['event_id']='old';old['period_end']='2019-12-31'
  e['unresolved_reasons']=['bad'];e['conditional_source_ready']=False
  self.assertFalse(m.map_row(row(e),[old,e])['conditional_source_ready'])
 def test_old_fallback_fails(self):
  e=m.event_overlay(*sample());old=deepcopy(e);old['event_id']='old';old['period_end']='2019-12-31'
  with self.assertRaises(ValueError):m.map_row(row(old),[old,e])
 def test_bad_row_preserved(self):
  e=m.event_overlay(*sample());r=row(e);r['availability']['reasons']=['identity'];before=deepcopy(r)
  o=m.map_row(r,[e]);self.assertFalse(o['conditional_source_ready']);self.assertEqual(o['original_availability_row'],before)

if __name__=='__main__':
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Test))
 if not result.wasSuccessful():raise SystemExit(1)
 m.write(m.OUT+'/TESTS.json',dict(passed=True,synthetic_only=True,count=result.testsRun,implementation_hashes={m.SELF:m.sha(m.SELF),m.TEST:m.sha(m.TEST)}))
