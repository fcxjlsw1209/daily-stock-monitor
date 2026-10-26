import unittest,copy
import numpy as np
import monthly_absolute_state135_runner as r

class Selection(unittest.TestCase):
 def rows(self):
  return [dict(name=n,training=dict(cagr=.5,sharpe=1.5,max_drawdown=-.3),early=dict(cagr=.4),recent=dict(cagr=.6))for n in r.NAMES]
 def test_tie_retains(self):self.assertEqual(r.select(self.rows())['primary'],r.NAMES[0])
 def test_only_filtered(self):
  z=self.rows();z[1]['training']['cagr']=9;self.assertEqual(r.select(z)['primary'],r.NAMES[0])
 def test_improve(self):
  z=self.rows();z[2]['training']['cagr']=.51;self.assertEqual(r.select(z)['primary'],r.NAMES[2])
 def test_each_guard(self):
  for block,key,val in [('training','sharpe',1.449),('training','max_drawdown',-.331),('early','cagr',.374),('recent','cagr',.574)]:
   z=self.rows();z[2]['training']['cagr']=.51;z[2][block][key]=val
   self.assertEqual(r.select(z)['primary'],r.NAMES[0])
 def test_invalid(self):
  for v in (np.nan,np.inf,True,'0.7',None):
   z=self.rows();z[2]['training']['cagr']=v;self.assertFalse(r.select(z)['admitted'])
 def test_no_mutate(self):
  z=self.rows();old=copy.deepcopy(z);r.select(z);self.assertEqual(z,old)
 def test_order_exact(self):
  with self.assertRaises(ValueError):r.select(self.rows()[::-1])


class ReuseContract(unittest.TestCase):
 def test_same_selection_function(self):
  self.assertIs(r.select, r.prior.select)
 def test_same_accounting_backend(self):
  self.assertIs(r.Backend, r.prior.Backend)
 def test_same_probability_score_path(self):
  self.assertIs(r.Runner.select_first88, r.prior.Runner.select_first88)
  self.assertIs(r.Runner.diagnose_later34, r.prior.Runner.diagnose_later34)
 def test_extra_inputs_named(self):
  self.assertEqual(set(r.INPUTS)-set(r.prior.INPUTS),set(r.REUSED))
 def test_wrong_reuse_path_rejected_before_read(self):
  from pathlib import Path
  with self.assertRaises(ValueError):
   r.verify_reused(Path('/tmp'),{},dict.fromkeys(r.REUSED,Path('/tmp/wrong')))

if __name__=='__main__':unittest.main()
