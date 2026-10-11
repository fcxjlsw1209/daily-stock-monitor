import unittest,copy
import numpy as np
import state_inverse as s

class StateTests(unittest.TestCase):
 def fixture(self):
  n=5;symbols=list('ABCD');rng=np.random.default_rng(1)
  X=rng.uniform(-.5,.5,(n,4,2));d=dict(symbols=[symbols]*n,eligible=np.ones((n,4),bool),order=np.tile([0,1,2],(n,1)),labels=np.tile([True,True,True,False],(n,1)))
  prices=np.array([[100.,110,90,80],[110,105,100,83],[90,120,95,87],[95,100,110,90],[100,110,90,100]])
  c=dict(dates=[f'2021-0{i+1}-28' for i in range(n)],symbols=symbols,ids={v:i for i,v in enumerate(symbols)},eligible=d['eligible'],close=prices,opening=prices*1.01,opening_dates=[f'2021-0{i+1}-29' for i in range(n)],peak=prices+2,sigma=prices*0+.1,trend=np.ones(n),highvol=np.zeros(n),periods=[dict(holdings=['A','B','C']) for _ in range(n)])
  return X,d,c
 def test_no_author_in_autonomous_rollout(self):
  X,d,c=self.fixture();w=np.arange(10)/10;first=s.rollout(c,d,X,['core','risk'],w=w)
  changed=copy.deepcopy(c)
  for p in changed['periods']:p['holdings']=['D','A','B']
  second=s.rollout(changed,d,X,['core','risk'],w=w)
  self.assertEqual(first[0]['choices'],second[0]['choices']);np.testing.assert_array_equal(first[1],second[1])
 def test_teacher_prefix_stops_at_boundary(self):
  X,d,c=self.fixture();_,S,_,_=s.rollout(c,d,X,['core','risk'],teacher_until=2,count=2)
  self.assertTrue((S[2:]==0).all())
  with self.assertRaises(AssertionError):s.rollout(c,d,X,['core','risk'],teacher_until=2,count=3)
 def test_state_starts_empty_and_uses_actual_cost(self):
  X,d,c=self.fixture();sim,S,_,_=s.rollout(c,d,X,['core','risk'],teacher_until=2,count=2)
  self.assertTrue((S[0]==0).all());self.assertAlmostEqual(S[1,0,1],.5*np.tanh(.1/.2))
 def test_future_author_labels_do_not_change_fit(self):
  X,d,c=self.fixture();mask=np.arange(5)<3
  w=s.fit_policy(X,d,c,['core','risk'],mask,.01,0,1,3)
  c['periods'][3]['holdings']=['D','C','B'];c['periods'][4]['holdings']=['D','C','B'];d['order'][3:]=[3,2,1]
  np.testing.assert_allclose(w,s.fit_policy(X,d,c,['core','risk'],mask,.01,0,1,3))
if __name__=='__main__':unittest.main()
