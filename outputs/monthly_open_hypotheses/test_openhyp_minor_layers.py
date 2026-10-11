import unittest
import numpy as np
import minor_layers as m

class LayerTests(unittest.TestCase):
 def test_no_forecast_effect_when_disabled(self):
  d=dict(symbols=[list('ABC')]*2,eligible=np.ones((2,3),bool));local=np.array([[.1,.8,.3],[.3,.1,.8]]);forecast=np.ones_like(local)*100
  score=m.combine(local,d,forecast,0,np.ones(2));self.assertEqual(np.argsort(-score,axis=1).tolist(),np.argsort(-local,axis=1).tolist())
 def test_weak_market_gate(self):
  d=dict(symbols=[list('ABC')]*2,eligible=np.ones((2,3),bool));local=np.tile([.1,.8,.3],(2,1));forecast=np.tile([.5,-.5,0],(2,1))
  base=m.combine(local,d,forecast,0,np.ones(2));score=m.combine(local,d,forecast,.2,np.array([0,1]))
  np.testing.assert_array_equal(base[0],score[0]);self.assertFalse(np.array_equal(base[1],score[1]))
 def test_excludes_ineligible_from_base_rank(self):
  d=dict(symbols=[list('ABC')],eligible=np.array([[True,False,True]]));local=np.array([[1,1000,2.]])
  score=m.combine(local,d,np.zeros_like(local),0,np.ones(1));self.assertEqual(score[0,1],0);self.assertGreater(score[0,2],score[0,0])
if __name__=='__main__':unittest.main()
