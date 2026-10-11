import unittest, copy
import numpy as np
import pandas as pd
import architecture as s

class ArchitectureTests(unittest.TestCase):
 def pack(self):
  names=list(dict.fromkeys(k for m in s.MENUS.values() for k in m))+['vol126','cmf21','cmf63','signed_volume63','volume_ratio21_126','acceleration_skip42','stock_sector_skip42_126','stock_sector_skip42_252','highvol_cmf63','trend_cmf63']
  R=np.random.default_rng(7).normal(size=(20,18,len(names)))
  return dict(names=names,X=.5*np.tanh(R),raw=R,eligible=np.ones((20,18),bool),order=np.tile([0,1,2],(20,1)),symbols=[list(map(str,range(18))) for _ in range(20)])
 def test_raw_transform_preserves_extreme_order(self):
  q=s.normalize(pd.DataFrame({'mom252_skip42':[.1,.5,1.,2.,4.]}),'absolute_tanh')
  self.assertTrue((np.diff(q.iloc[:,0])>0).all())
  self.assertTrue(np.isfinite(q).all().all())
 def test_gate_does_not_read_author_labels_or_future(self):
  d=self.pack();d2=copy.deepcopy(d);d2['order'][:]=[5,6,7];d2['raw'][17:]*=-100
  g=s.apply_gate(d,'raw','top12');h=s.apply_gate(d2,'raw','top12')
  np.testing.assert_array_equal(g['eligible'][:17],h['eligible'][:17]);self.assertTrue((g['eligible'].sum(axis=1)==12).all())
  np.testing.assert_array_equal(d['eligible'],np.ones((20,18),bool))
 def test_no_gate_is_identity_and_no_target_resurrection(self):
  d=self.pack();d['eligible'][0,0]=False;g=s.apply_gate(d,'raw','positive')
  np.testing.assert_array_equal(d['eligible'],g['eligible']);self.assertEqual(g['order'][0,0],-1)
 def test_pairwise_fit_ignores_holdout(self):
  d=self.pack();X,spec=s.a.design(d,s.MENUS['raw'],'legacy');mask=np.arange(20)<17
  before=s.fit_base(X,d,mask,spec,'set_order');d2=copy.deepcopy(d);d2['order'][17:]=[5,6,7];X2=X.copy();X2[17:]*=-20
  after=s.fit_base(X2,d2,mask,spec,'set_order');np.testing.assert_allclose(before,after,rtol=0,atol=1e-12)
  A,b,_=s.e.limits(spec);self.assertLessEqual(np.max(np.einsum('ij,j->i',A,before,optimize=False)-b),1e-7)
 def test_convex_regime_blend_preserves_constraints(self):
  d=self.pack();X,spec=s.a.design(d,s.MENUS['raw'],'legacy');r=s.fit_rule(X,d,np.arange(20)<17,spec,'set_order',np.arange(20)%2)
  A,b,_=s.e.limits(spec)
  for w in r['weights'].values():
   self.assertAlmostEqual(sum(w),1.);self.assertLessEqual(np.max(np.einsum('ij,j->i',A,w,optimize=False)-b),1e-7);self.assertLessEqual(np.abs(np.array(w)-r['base']).sum(),.3000001)
 @unittest.skipUnless((s.P/'FROZEN.json').exists(),'Run training first')
 def test_selected_rule_and_cost_convention(self):
  plan=s.read(s.P/'PLAN.json');self.assertIn('zero cost',plan['hard_rules'])
  for row in s.read(s.P/'FROZEN.json')['leaders'].values():
   self.assertEqual(row['inner_validation']['months'],18)
   A,b,_=s.e.limits(row['rule']['spec'])
   for w in row['rule']['weights'].values():self.assertLessEqual(np.max(np.einsum('ij,j->i',A,w,optimize=False)-b),1e-7)
if __name__=='__main__':unittest.main()
