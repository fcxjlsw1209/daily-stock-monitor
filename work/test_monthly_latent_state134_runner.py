import unittest,copy
import numpy as np
import monthly_latent_state134_runner as r

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
if __name__=='__main__':unittest.main()
