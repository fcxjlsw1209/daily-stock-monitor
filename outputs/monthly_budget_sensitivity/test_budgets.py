import unittest,copy
import numpy as np
import budgets as b
class BudgetTests(unittest.TestCase):
 def spec(self):return [('long',1.,'core'),('vol',1.,'vol'),('sector',1.,'industry'),('cmf',1.,'shape'),('accel',1.,'accel')]
 def test_old_compact_allocation_excluded_only_by_invented_caps(self):
  w=np.array([.52,.219375,0,.075,.185625]);self.assertAlmostEqual(w.sum(),1.)
  A,z,_=b.limits(self.spec(),'strict');self.assertGreater(np.max(np.einsum('ij,j->i',A,w,optimize=False)-z),0)
  for mode in ['separate_acceleration','core50','dominant40']:
   A,z,_=b.limits(self.spec(),mode);self.assertLessEqual(np.max(np.einsum('ij,j->i',A,w,optimize=False)-z),1e-12)
 def test_explicit_dominant40_is_only_mode_accepting_below50_long_core(self):
  w=np.array([.4,.3,.1,.1,.1])
  for mode in ['strict','separate_acceleration','core50']:
   A,z,_=b.limits(self.spec(),mode);self.assertGreater(np.max(np.einsum('ij,j->i',A,w,optimize=False)-z),.099)
  A,z,_=b.limits(self.spec(),'dominant40');self.assertLessEqual(np.max(np.einsum('ij,j->i',A,w,optimize=False)-z),1e-12)
 def test_fit_masks_later_features_and_labels(self):
  rng=np.random.default_rng(29);X=rng.uniform(-.5,.5,(20,8,5));d=dict(eligible=np.ones((20,8),bool),order=np.tile([0,1,2],(20,1)));mask=np.arange(20)<17
  w=b.fit_base(X,d,mask,self.spec(),'set_order','separate_acceleration');dd=copy.deepcopy(d);dd['order'][17:]=[5,6,7];XX=X.copy();XX[17:]*=-100
  ww=b.fit_base(XX,dd,mask,self.spec(),'set_order','separate_acceleration');np.testing.assert_allclose(w,ww,rtol=0,atol=1e-12)
 @unittest.skipUnless((b.P/'FROZEN.json').exists(),'Train first')
 def test_frozen_weights_respect_declared_mode(self):
  for r in b.read(b.P/'FROZEN.json')['leaders'].values():
   A,z,_=b.limits(r['rule']['spec'],r['mode'])
   for w in r['rule']['weights'].values():
    self.assertAlmostEqual(sum(w),1.);self.assertLessEqual(np.max(np.einsum('ij,j->i',A,w,optimize=False)-z),1e-7)
if __name__=='__main__':unittest.main()
