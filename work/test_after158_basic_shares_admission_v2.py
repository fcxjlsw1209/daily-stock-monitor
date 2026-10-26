"""Narrow v2 unit-proof tests plus every frozen v1 source regression."""
import ast
from copy import deepcopy
import inspect
import unittest
import after158_basic_shares_admission_v2 as m
import test_after158_basic_shares_admission as prior

# The old synthetic cases with explicit share-specific units retain old behavior.
class OldRegression(prior.Test):
 def setUp(self):self.previous=prior.m.event_overlay;prior.m.event_overlay=m.event_overlay
 def tearDown(self):prior.m.event_overlay=self.previous


def sample():
 raw,avail,meta,bridge,clock=prior.sample()
 for obj in(raw,avail,meta,bridge):obj['event_id']='MSFT:FY2020:2021-01-20';obj['symbol']='MSFT'
 raw['source']={'sha256':'a'*64};avail['source']=deepcopy(raw['source'])
 for side,key in(('current','current_selected'),('comparison','prior_selected')):
  c=raw['cells'][side];c.update(displayed_value='1',value_decimal='1',source_sha256='a'*64)
  c['unit']={'unit':'thousands','scale':1000,'share_specific':False,'evidence':[{'text':'INCOME STATEMENTS (In thousands, except per share amounts)','locator':{'caption':True}}]}
  pointer='/facts/us-gaap/WeightedAverageNumberOfSharesOutstandingBasic/units/shares/'+str(len(key));meta[key]['json_pointer']=pointer
  bridge['cells'][side].update(original_displayed_value='1',original_share_scale='1000',original_value_in_shares='1000',companyfacts_value_in_shares='1000',original_source_sha256='a'*64,source_period_end=c['period_end'],json_pointer=pointer)
 return raw,avail,meta,bridge,clock

class Generic(unittest.TestCase):
 def test_reliable_generic_keeps_false(self):
  x=sample();before=deepcopy(x[:4]);out=m.event_overlay(*x);self.assertTrue(out['conditional_source_ready']);self.assertFalse(out['original_event']['cells']['current']['unit']['share_specific']);self.assertEqual(before,x[:4])
 def test_no_evidence_rejected(self):
  x=sample();x[0]['cells']['current']['unit']['evidence']=[]
  with self.assertRaises(ValueError):m.event_overlay(*x)
 def test_wrong_scale_rejected(self):
  x=sample();x[0]['cells']['current']['unit']['scale']=1000000
  with self.assertRaises(ValueError):m.event_overlay(*x)
 def test_wrong_normalization_rejected(self):
  x=sample();x[0]['cells']['current']['value_decimal']='2'
  with self.assertRaises(ValueError):m.event_overlay(*x)
 def test_bridge_original_string_conflict(self):
  for key,value in(('original_value_in_shares','1000.0'),('original_displayed_value','1.0'),('original_share_scale','1000000'),('original_source_sha256','b'*64),('companyfacts_value_in_shares','1001')):
   x=sample();x[3]['cells']['current'][key]=value
   with self.subTest(key=key),self.assertRaises(ValueError):m.event_overlay(*x)
 def test_rounding_only_cannot_validate_generic_units(self):
  x=sample();x[3]['cells']['current']['equal']=False;x[3]['cells']['current']['native_rounding_interval_compatible']=True
  with self.assertRaisesRegex(ValueError,'exact GAAP'):m.event_overlay(*x)
 def test_false_exact_flag_cannot_hide_numeric_mismatch(self):
  x=sample();x[3]['cells']['current']['companyfacts_value_in_shares']='999'
  with self.assertRaises(ValueError):m.event_overlay(*x)
 def test_conflicting_unit_words(self):
  x=sample();x[0]['cells']['current']['unit']['evidence'][0]['text']='In thousands (or millions)'
  with self.assertRaises(ValueError):m.event_overlay(*x)
 def test_arbitrary_word_not_unit_declaration(self):
  x=sample();x[0]['cells']['current']['unit']['evidence'][0]['text']='We discuss thousands of customers'
  with self.assertRaises(ValueError):m.event_overlay(*x)
 def test_missing_locator(self):
  x=sample();x[0]['cells']['current']['unit']['evidence'][0]['locator']={}
  with self.assertRaises(ValueError):m.event_overlay(*x)
 def test_unlisted_issuer_blocked(self):
  x=sample()
  for obj in x[:4]:obj['symbol']='OTHER'
  with self.assertRaises(ValueError):m.event_overlay(*x)
 def test_selected_context_conflict(self):
  x=sample();x[3]['cells']['current']['json_pointer']='/facts/us-gaap/Diluted/units/shares/1'
  with self.assertRaises(ValueError):m.event_overlay(*x)
 def test_old_holds_not_released(self):
  x=sample();x[0]['inherited_holds']['inherited_rejection_codes']=['excluded_accounting_comparability'];x[1]['inherited_holds']=deepcopy(x[0]['inherited_holds']);out=m.event_overlay(*x);self.assertFalse(out['conditional_source_ready']);self.assertIn('INHERITED:excluded_accounting_comparability',out['unresolved_reasons'])
 def test_event_ast_only_changes_unit_guard(self):
  first=ast.parse(inspect.getsource(m.old.event_overlay));second=ast.parse(inspect.getsource(m.event_overlay))
  def strip(tree):
   loop=next(n for n in ast.walk(tree)if isinstance(n,ast.For)and isinstance(n.target,ast.Name)and n.target.id=='side')
   loop.body=[n for n in loop.body if not(isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)and((isinstance(n.value.func,ast.Name)and n.value.func.id=='validate_unit')or any(isinstance(a,ast.Constant)and a.value=='Share-specific unit required'for a in n.value.args)))]
   return ast.dump(tree,include_attributes=False)
  self.assertEqual(strip(first),strip(second))

if __name__=='__main__':
 result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(c)for c in(OldRegression,Generic)]))
 if not result.wasSuccessful():raise SystemExit(1)
 m.write(m.OUT+'/TESTS.json',{'passed':True,'synthetic_only':True,'count':result.testsRun,'implementation_hashes':m.code_pins()})
