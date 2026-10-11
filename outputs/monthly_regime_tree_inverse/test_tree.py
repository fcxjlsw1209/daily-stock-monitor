import unittest,copy
import numpy as np
import pandas as pd
import regime_tree as t

class TreeTests(unittest.TestCase):
 def toy(self):
  names=list(dict.fromkeys(k for v in t.s.MENUS.values() for k in v))+['vol126','cmf21','cmf63','signed_volume63','volume_ratio21_126','acceleration_skip42','stock_sector_skip42_126','stock_sector_skip42_252','highvol_cmf63','trend_cmf63']
  rng=np.random.default_rng(19);X=rng.uniform(-.5,.5,(20,8,len(names)))
  d=dict(names=names,X=X,eligible=np.ones((20,8),bool),order=np.tile([0,1,2],(20,1)))
  Z=pd.DataFrame({k:rng.normal(size=20) for k in t.MARKETS});XX,spec=t.a.design(d,t.s.MENUS['raw'],'legacy')
  return d,XX,spec,Z
 def test_tree_thresholds_and_weights_ignore_later_labels_and_states(self):
  d,X,spec,Z=self.toy();mask=np.arange(20)<17
  a=t.fit_tree(X,d,mask,spec,'set_order',Z,3,.6)
  d2=copy.deepcopy(d);d2['order'][17:]=[4,5,6];X2=X.copy();X2[17:]*=-999;Z2=Z.copy();Z2.iloc[17:]=999
  b=t.fit_tree(X2,d2,mask,spec,'set_order',Z2,3,.6)
  self.assertEqual(a,b)
 def test_regime_predictions_have_common_constrained_weights(self):
  d,X,spec,Z=self.toy();r=t.fit_tree(X,d,np.arange(20)<17,spec,'ordered',Z,3,.6);W,paths=t.weights_for(r,Z);A,b,_=t.e.limits(spec)
  self.assertLessEqual(r['actual_leaves'],3)
  for w in W:
   self.assertAlmostEqual(w.sum(),1.);self.assertLessEqual(np.abs(w-r['base']).sum(),.6000001);self.assertLessEqual(np.max(np.einsum('ij,j->i',A,w,optimize=False)-b),1e-7)
 @unittest.skipUnless((t.P/'FROZEN.json').exists(),'Train first')
 def test_frozen_tree_leaf_sizes_and_features(self):
  for row in t.read(t.P/'FROZEN.json')['leaders'].values():
   leaves=[]
   def visit(n):
    if 'feature' in n:
     self.assertIn(n['feature'],t.MARKETS);visit(n['left']);visit(n['right'])
    else:leaves.append(n)
   visit(row['rule']['tree']);self.assertEqual(sum(n['months'] for n in leaves),35);self.assertTrue(all(n['months']>=6 for n in leaves));self.assertEqual(row['inner_validation']['months'],18)
if __name__=='__main__':unittest.main()
