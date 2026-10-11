import copy,unittest
from membership import build_union,entry_session
class MembershipTests(unittest.TestCase):
 def setUp(self):
  base=dict(verified=True,source_sha256='evidence',effective_at='2010-01-01T00:00:00Z',known_at='2009-12-31T00:00:00Z',security_ids=['A'])
  cov=dict(verified=True,evidence_sha256='coverage',**{'from':'2010-01-01T00:00:00Z','through':'2010-02-01T00:00:00Z'})
  self.d={i:dict(baseline=copy.deepcopy(base),coverage=copy.deepcopy(cov),events=[]) for i in ['NDX','SP100']}
  self.ids=[dict(security_id=s,issuer_id='issuer'+s,verified=True,source_sha256='id',known_at='2009-01-01T00:00:00Z',effective_from='2009-01-01T00:00:00Z',effective_to='2020-01-01T00:00:00Z') for s in ['A','B']]
 def run_case(self):return build_union(self.d,'2010-01-04T21:00:00Z','2010-01-04T21:20:00Z',self.ids)
 def test_union_deduplicates_issuer(self):self.assertEqual(len(self.run_case()['members']),1)
 def test_missing_index_blocks(self):
  del self.d['SP100']
  with self.assertRaises(ValueError):self.run_case()
 def test_no_current_list_fallback(self):
  self.d['NDX']['coverage']['through']='2010-01-02T00:00:00Z'
  with self.assertRaises(ValueError):self.run_case()
 def test_future_change_not_applied(self):
  self.d['NDX']['events']=[dict(event_id='x',effective_at='2010-01-05T14:30:00Z',known_at='2010-01-03T00:00:00Z',verified=True,source_sha256='x',add=['B'],remove=['A'])]
  self.assertEqual(self.run_case()['members'][0]['security_ids'],['A'])
 def test_late_evidence_blocks(self):
  self.d['NDX']['baseline']['known_at']='2010-01-05T00:00:00Z'
  with self.assertRaises(ValueError):self.run_case()
 def test_missing_identity_blocks(self):
  self.ids=[]
  with self.assertRaises(ValueError):self.run_case()
 def test_two_classes_one_company(self):
  self.d['SP100']['baseline']['security_ids']=['B'];self.ids[1]['issuer_id']='issuerA'
  self.assertEqual(self.run_case()['members'][0]['security_ids'],['A','B'])
 def test_friday_entry_waits_past_holiday(self):
  sessions=['2010-01-15','2010-01-19','2010-01-20']
  self.assertEqual(entry_session('2010-01-15T14:30:00Z','2010-01-15T23:00:00Z',sessions),'2010-01-19')
 def test_late_publication_delays_entry(self):
  self.assertEqual(entry_session('2010-01-04T14:30:00Z','2010-01-05T15:00:00Z',['2010-01-04','2010-01-05','2010-01-06']),'2010-01-06')
 def test_removed_member_not_kept_during_entry_delay(self):
  self.d['NDX']['events']=[dict(event_id='x',effective_at='2010-01-04T14:30:00Z',known_at='2010-01-04T12:00:00Z',verified=True,source_sha256='x',add=['B'],remove=['A'])]
  self.d['SP100']['baseline']['security_ids']=[]
  sessions=['2010-01-04','2010-01-05','2010-01-06']
  def run(day):return build_union(self.d,day+'T21:00:00Z',day+'T21:20:00Z',self.ids,sessions)['members']
  self.assertEqual(run('2010-01-04'),[])
  self.assertEqual(run('2010-01-05')[0]['security_ids'],['B'])
 def test_future_published_add_is_not_lookahead(self):
  self.d['NDX']['events']=[dict(event_id='x',effective_at='2010-01-04T14:30:00Z',known_at='2010-01-05T12:00:00Z',verified=True,source_sha256='x',add=['B'],remove=[])]
  result=build_union(self.d,'2010-01-04T21:00:00Z','2010-01-04T21:20:00Z',self.ids,['2010-01-04','2010-01-05','2010-01-06'])
  self.assertEqual(result['members'][0]['security_ids'],['A'])
 def test_future_published_removal_blocks_instead_of_silent_lookahead(self):
  self.d['NDX']['events']=[dict(event_id='x',effective_at='2010-01-04T14:30:00Z',known_at='2010-01-05T12:00:00Z',verified=True,source_sha256='x',add=['B'],remove=['A'])]
  with self.assertRaisesRegex(ValueError,'Removal not publicly known'):
   build_union(self.d,'2010-01-04T21:00:00Z','2010-01-05T21:20:00Z',self.ids,['2010-01-04','2010-01-05','2010-01-06'])
 def test_baseline_published_after_signal_cannot_use_runtime_cutoff(self):
  self.d['NDX']['baseline']['known_at']='2010-01-04T21:10:00Z'
  with self.assertRaises(ValueError):self.run_case()
 def test_identity_published_after_signal_cannot_use_runtime_cutoff(self):
  self.ids[0]['known_at']='2010-01-04T21:10:00Z'
  with self.assertRaises(ValueError):self.run_case()
 def test_short_lived_never_eligible_member_does_not_leak_future_removal(self):
  self.d['NDX']['events']=[
   dict(event_id='add',effective_at='2010-01-04T14:30:00Z',known_at='2010-01-05T23:00:00Z',verified=True,source_sha256='a',add=['B'],remove=[]),
   dict(event_id='remove',effective_at='2010-01-05T14:30:00Z',known_at='2010-01-05T23:00:00Z',verified=True,source_sha256='r',add=[],remove=['B'])]
  for day in ['2010-01-04','2010-01-05','2010-01-06']:
   result=build_union(self.d,day+'T21:00:00Z',day+'T21:20:00Z',self.ids,['2010-01-04','2010-01-05','2010-01-06'])
   self.assertEqual([m['issuer_id'] for m in result['members']],['issuerA'])
if __name__=='__main__':unittest.main()
