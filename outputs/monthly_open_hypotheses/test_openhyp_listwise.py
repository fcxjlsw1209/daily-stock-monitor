import unittest,itertools
import numpy as np
from scipy.optimize import approx_fprime
import listwise_inverse as m

class LikelihoodTests(unittest.TestCase):
 def fixture(self):
  rng=np.random.default_rng(11);X=rng.normal(size=(4,5,3));d=dict(order=np.tile([0,1,3],(4,1)),eligible=np.ones((4,5),bool));ev=m.event_design(d,np.ones(4,bool));return X,d,ev
 def test_gradient(self):
  X,d,ev=self.fixture();w=np.array([.3,-.1,.2]);args=(X,ev,['core','risk','shape'],.01,16,.25)
  val,grad=m.objective(w,*args);fd=approx_fprime(w,lambda z:m.objective(z,*args)[0],1e-6);np.testing.assert_allclose(grad,fd,atol=2e-6)
 def test_uniform_unordered_probability(self):
  X,d,ev=self.fixture();val,_=m.objective(np.zeros(3),X,ev,['core','risk','shape'],0,1,0)
  self.assertAlmostEqual(val,np.log(10)) # 1 / choose(5,3)
 def test_set_order_invariance(self):
  X,d,ev=self.fixture();w=np.array([.1,.2,.3]);v=m.objective(w,X,ev,['core','risk','shape'],0,1,0)[0]
  d['order']=d['order'][:,::-1];v2=m.objective(w,X,m.event_design(d,np.ones(4,bool)),['core','risk','shape'],0,1,0)[0];self.assertAlmostEqual(v,v2)
 def test_missing_one_target_is_finite(self):
  X,d,ev=self.fixture();d['order'][1,2]=-1;ev=m.event_design(d,np.ones(4,bool));val,grad=m.objective(np.ones(3),X,ev,['core','risk','shape'],.01,1,0);self.assertTrue(np.isfinite(val));self.assertTrue(np.isfinite(grad).all())
 def test_future_labels_excluded(self):
  X,d,ev=self.fixture();mask=np.arange(4)<2;w=m.fit(X,d,mask,['core','risk','shape'],.1,0,1)
  d['order'][2:]=[4,2,0];np.testing.assert_allclose(w,m.fit(X,d,mask,['core','risk','shape'],.1,0,1))
if __name__=='__main__':unittest.main()
