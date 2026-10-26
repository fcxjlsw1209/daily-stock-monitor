"""Synthetic union tests: no financial observations or market returns."""
import copy, tempfile, unittest
from pathlib import Path
import same_report145_union as u

def package(name):
 _,symbols,n=u.PACKAGES[name];events=[]
 for j,symbol in enumerate(sorted(symbols)):
  for k in range(n//2):
   accepted=k!=0
   events.append(dict(event_id=f'{symbol}:{k}',symbol=symbol,security_id=symbol,method=u.METHOD,
    admitted=accepted,rejection_codes=[] if accepted else ['synthetic_missing_period'],
    release_date='2020-01-01',available_from='2020-01-02',identity_valid_from='2020-01-02',
    same_report={x:{'gross_profit':{'value':'1','scale':'1'},'assets':{'value':'2','scale':'1'}} for x in ('current','comparison')}))
 counts=dict(events=n,conditionally_admitted=n-2,rejected=2,raw_values=4*n)
 data=dict(method=u.METHOD,events=events,model_admitted=False,strategy_admitted=False,counts=counts,**{x:False for x in u.CERTS})
 audit=dict(stage='conditional_source_admission',passed=True,data_sha256='f'*64,checks={x:True for x in u.SOURCE_CHECKS[name]},counts=counts,**{x:0 for x in u.COUNTERS},**{x:False for x in u.CERTS})
 return data,audit,'f'*64

def items():return {x:package(x) for x in u.PACKAGES}

class UnionTests(unittest.TestCase):
 def test_full_census(self):
  result=u.combine(items());self.assertEqual(result['counts'],dict(events=44,conditionally_admitted=40,rejected=4,raw_values=176))
 def test_exact_event_content_and_input_unchanged(self):
  a=items();before=copy.deepcopy(a);out=u.combine(a);self.assertEqual(a,before)
  expected={e['event_id']:e for x in a.values() for e in x[0]['events']}
  self.assertEqual({e['event_id']:e for e in out['events']},expected)
 def test_retains_missing_period_rejections(self):
  out=u.combine(items());self.assertEqual(sum(not e['admitted'] for e in out['events']),4)
 def test_cannot_omit_package(self):
  a=items();del a['aapl_amd']
  with self.assertRaises(ValueError):u.combine(a)
 def test_cannot_drop_event(self):
  a=items();a['aapl_amd'][0]['events'].pop()
  with self.assertRaisesRegex(ValueError,'census'):u.combine(a)
 def test_duplicates_fail(self):
  a=items();a['msft_nvda'][0]['events'][1]['event_id']=a['msft_nvda'][0]['events'][0]['event_id']
  with self.assertRaisesRegex(ValueError,'Unique'):u.combine(a)
 def test_other_issuers_fail(self):
  a=items();a['msft_nvda'][0]['events'][0]['symbol']='AAPL'
  with self.assertRaisesRegex(ValueError,'issuer'):u.combine(a)
 def test_different_measurement_fails(self):
  a=items();a['msft_nvda'][0]['method']='prior-original'
  with self.assertRaisesRegex(ValueError,'measurement'):u.combine(a)
 def test_uncertified_stage_fails(self):
  a=items();a['msft_nvda'][1]['stage']='source_preflight'
  with self.assertRaisesRegex(ValueError,'admission'):u.combine(a)
 def test_dataset_hash_fails(self):
  a=items();a['msft_nvda'][1]['data_sha256']='0'*64
  with self.assertRaisesRegex(ValueError,'binding'):u.combine(a)
 def test_missing_source_check_fails(self):
  a=items();a['aapl_amd'][1]['checks'].popitem()
  with self.assertRaisesRegex(ValueError,'checks'):u.combine(a)
 def test_false_check_fails(self):
  a=items();a['aapl_amd'][1]['checks'][u.SOURCE_CHECKS['aapl_amd'][0]]=False
  with self.assertRaisesRegex(ValueError,'checks'):u.combine(a)
 def test_certification_upgrades_fail(self):
  for which in (0,1):
   for key in u.CERTS:
    a=items();a['msft_nvda'][which][key]=True
    with self.assertRaisesRegex(ValueError,'certification'):u.combine(a)
 def test_counters_fail(self):
  for key in u.COUNTERS:
   for value in (1,False):
    a=items();a['aapl_amd'][1][key]=value
    with self.assertRaisesRegex(ValueError,'counter'):u.combine(a)
 def test_rejection_cannot_be_promoted(self):
  a=items();a['msft_nvda'][0]['events'][0]['admitted']=True
  with self.assertRaisesRegex(ValueError,'conflicts'):u.combine(a)
 def test_identity_rewrite_fails(self):
  a=items();a['msft_nvda'][0]['events'][0]['identity_valid_from']='2019-12-01'
  with self.assertRaisesRegex(ValueError,'identity'):u.combine(a)
 def test_comparison_cannot_be_missing(self):
  a=items();del a['msft_nvda'][0]['events'][0]['same_report']['comparison']
  with self.assertRaisesRegex(ValueError,'columns'):u.combine(a)
 def test_counts_cannot_lie(self):
  for which in (0,1):
   a=items();a['aapl_amd'][which]['counts']['rejected']=0
   with self.assertRaisesRegex(ValueError,'counts'):u.combine(a)
 def test_no_models_or_financial_computation(self):
  out=u.combine(items());self.assertFalse(out['model_admitted']);self.assertFalse(out['strategy_admitted'])
  self.assertEqual(out['signs_computed'],0);self.assertEqual(out['returns_computed'],0)
 def test_path_guard(self):
  root=Path('/tmp').resolve()
  with self.assertRaisesRegex(ValueError,'Cloud'):u.path_at(root,'/Users/example/OneDrive/file')
  with self.assertRaisesRegex(ValueError,'outside'):u.path_at(root,'/etc/passwd',True)
 def test_write_once(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'a.json';u.write_once(p,{'a':1})
   with self.assertRaises(FileExistsError):u.write_once(p,{'a':2})
   self.assertEqual(u.load(p),{'a':1})
if __name__=='__main__':unittest.main()
