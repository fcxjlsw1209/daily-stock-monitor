import unittest
import numpy as np
from feasibility_audit import feasible

class FeasibilityTests(unittest.TestCase):
 def test_separable_toy(self):
  X=np.array([[[3.],[2.],[1.],[0.]]]);d=dict(order=np.array([[0,1,2]]),eligible=np.ones((1,4),bool))
  self.assertTrue(feasible(X,d,['core'],np.ones(1,bool),True)['strict_common_score_separable'])
 def test_contradiction_toy(self):
  X=np.array([[[3.],[2.],[1.],[0.]],[[3.],[2.],[1.],[0.]]]);d=dict(order=np.array([[0,1,2],[3,2,1]]),eligible=np.ones((2,4),bool))
  self.assertFalse(feasible(X,d,['core'],np.ones(2,bool),False)['strict_common_score_separable'])
if __name__=='__main__':unittest.main()
