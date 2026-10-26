"""Synthetic phase and source-identity tests; no real files or returns."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import unittest
import tempfile
import json
import basic_shares159_host as h

class Test(unittest.TestCase):
 def test_measurement_clocks(self):
  self.assertIn('corroborating_filing',h.measurement()['usable_at'])
  self.assertEqual(h.measurement()['dose'],.01)
 def test_selection_fixed(self):
  c=h.r.selection_contract();self.assertEqual(c['minimum_close_cagr_gain'],.005);self.assertTrue(c['later34_diagnostic_only'])
 def test_pin_conflict(self):
  with self.assertRaises(ValueError):h.combined({'a':'one'},{'a':'two'})
 def test_identical_pins(self):
  self.assertEqual(len(h.combined({'a':'one'},{'a':'one'})),1)
 def test_changed_input_fails(self):
  with patch.object(h,'sha',return_value='bad'):
   with self.assertRaises(ValueError):h.verify({'a':'good'})
 def test_wrong_plan_sha(self):
  with patch.object(h,'sha',return_value='bad'):
   with self.assertRaises(ValueError):h.Host('good')
 def test_runtime_rejects_before_inputs(self):
  p=dict(round=159,status='REGISTERED',arms=list(h.r.NAMES),runtime={},measurement=h.measurement(),selection_contract=h.r.selection_contract())
  with patch.object(h,'sha',return_value='s'),patch.object(h,'read',return_value=p):
   with self.assertRaises(ValueError):h.Host('s')
 def plan(self):
  return dict(round=159,status='REGISTERED',arms=list(h.r.NAMES),runtime=h.runtime(),measurement=h.measurement(),selection_contract=h.r.selection_contract(),output_directory=str(h.OUT),no_observer_writes=True,actual_features_before_registration=0,actual_orders_before_registration=0,actual_returns_before_registration=0)
 def test_valid_plan(self):h.validate_plan(self.plan())
 def test_plan_boundary_guards(self):
  for key,value in [('output_directory','elsewhere'),('no_observer_writes',False),('actual_features_before_registration',1),('actual_orders_before_registration',False),('actual_returns_before_registration',0.0)]:
   p=self.plan();p[key]=value
   with self.subTest(key=key),self.assertRaises(ValueError):h.validate_plan(p)
 def test_empty_validation_pins_rejected(self):
  with patch.object(h,'validation_paths',return_value=[Path('/test/host.py')]),patch.object(h,'sha',return_value='good'):
   with self.assertRaises(ValueError):h.validate_tests(dict(passed=True,synthetic_only=True,implementation_hashes={}))
 def test_current_validation_required(self):
  p=Path('/test/host.py')
  with patch.object(h,'validation_paths',return_value=[p]),patch.object(h,'sha',return_value='good'):
   h.validate_tests(dict(passed=True,synthetic_only=True,implementation_hashes={str(p):'good'}))
   with self.assertRaises(ValueError):h.validate_tests(dict(passed=True,synthetic_only=True,implementation_hashes={str(p):'old'}))
 def test_state_commit_exact_backup_and_cas(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d)/'state.json';original=b'{"old": 1}\n';p.write_bytes(original)
   with patch.object(h,'STATE',p):
    before=h.sha(p);h.state_commit(before,{'new':2})
    self.assertEqual((p.parent/('PRE159_'+before+'.json')).read_bytes(),original)
    self.assertEqual(json.loads(p.read_text()),{'new':2})
    with self.assertRaises(ValueError):h.state_commit(before,{'bad':3})
    self.assertEqual(json.loads(p.read_text()),{'new':2})
 def test_write_equal_never_overwrites(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d)/'proof.json';h.write_equal(p,{'a':1});h.write_equal(p,{'a':1})
   with self.assertRaises(ValueError):h.write_equal(p,{'a':2})
   self.assertEqual(json.loads(p.read_text()),{'a':1})
 def test_recovery_uses_full_plan_contract(self):
  p=self.plan();p['no_observer_writes']=False
  with self.assertRaises(ValueError):h.enqueue(p)
 def fixtures(self):
  meta={'dates':['2020-01-31']*122,'symbols':[['A']+['B'+str(j)for j in range(48)]for i in range(122)],'etfs':[]}
  events=[dict(event_id='e'+str(i),symbol='A',identity=dict(security_id='A',issuer_id='CIK1',identity_valid_from='2019-01-01',identity_valid_through='2021-01-01'))for i in range(44)]
  rows=[]
  for i in range(122):
   for j in range(49):rows.append(dict(signal_index=i,candidate_index=j,symbol=meta['symbols'][i][j],cutoff='2019-11-29T21:00:00+00:00',conditional_source_ready=j==0,selected_event_id='e0',original_availability_row={'decision_date':'2020-01-31'}))
  obj=object.__new__(h.Host);obj.inputs={'meta':Path('meta'),'source_events':Path('events'),'source_rows':Path('rows')}
  files={Path('meta'):meta,Path('events'):{'events':events},Path('rows'):rows}
  return obj,files
 def run_source(self,files,obj):
  with patch.object(h,'read',side_effect=files.__getitem__),patch.object(h.base,'validate_meta'):
   return obj.source()
 def test_exact_source_grid(self):
  obj,f=self.fixtures();out=self.run_source(f,obj);self.assertEqual(len(out[3]),122)
 def test_grid_symbol_conflict(self):
  obj,f=self.fixtures();f[Path('rows')][3]['symbol']='X'
  with self.assertRaises(ValueError):self.run_source(f,obj)
 def test_source_identity_conflict(self):
  obj,f=self.fixtures();f[Path('events')]['events'][0]['identity']['security_id']='Z'
  with self.assertRaises(ValueError):self.run_source(f,obj)
 def test_source_expired_window(self):
  obj,f=self.fixtures();f[Path('events')]['events'][0]['identity']['identity_valid_through']='2019-02-01'
  with self.assertRaises(ValueError):self.run_source(f,obj)
 def test_source_duplicate_event_ids(self):
  obj,f=self.fixtures();f[Path('events')]['events'][1]['event_id']='e0'
  with self.assertRaises(ValueError):self.run_source(f,obj)
 def test_source_cutoff_conflict(self):
  obj,f=self.fixtures();f[Path('rows')][1]['cutoff']='2019-11-28T21:00:00+00:00'
  with self.assertRaises(ValueError):self.run_source(f,obj)
 def test_source_etf_excluded(self):
  obj,f=self.fixtures();f[Path('meta')]['etfs']=['A']
  with self.assertRaises(ValueError):self.run_source(f,obj)
 def test_no_input_mutation(self):
  obj,f=self.fixtures();old=deepcopy(f);self.run_source(f,obj);self.assertEqual(old,f)

if __name__=='__main__':unittest.main(verbosity=2)
