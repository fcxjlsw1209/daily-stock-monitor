import unittest
from unittest.mock import patch
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import exact35 as m

class ExactFitTests(unittest.TestCase):
 def test_projection_enforces_order_and_outsiders(self):
  b=np.array([.4,.3,-.2,-.1,.1]);target=[3,2,4];z=m.project_scores(b,target)
  order=np.argsort(-z)[:3].tolist();self.assertEqual(order,target)
  self.assertGreaterEqual(z[3]-z[2],m.MARGIN-1e-8);self.assertGreaterEqual(z[4]-max(z[0],z[1]),m.MARGIN-1e-8)
 def test_projection_leaves_feasible_base_unchanged(self):
  b=np.array([.3,.2,.1,0.]);np.testing.assert_allclose(m.project_scores(b,[0,1,2]),b,atol=1e-12)
 def test_serialized_tree_matches_sklearn(self):
  rng=np.random.default_rng(6);x=rng.normal(size=(70,5));y=rng.normal(size=70);m0=DecisionTreeRegressor(max_depth=5,random_state=42).fit(x,y);tree=m.serialize_tree(m0)
  q=rng.normal(size=(100,5));np.testing.assert_array_equal(m.tree_predict(tree,q),m0.predict(q))
 def test_knn_exact_training_interpolation(self):
  rng=np.random.default_rng(9);x=rng.normal(size=(30,6));y=rng.normal(size=30);pre=m.scale_for_knn(x,['core']*3+['risk']*3)
  model=dict(preprocess=pre,train_features=m.scale(x,pre).tolist(),residual=y.tolist(),neighbors=15)
  np.testing.assert_allclose(m.knn_predict(model,x),y,atol=1e-12)
 def test_knn_finite_unseen_features(self):
  x=np.array([[0.,1],[1,0],[2,2]]);pre=m.scale_for_knn(x,['core','risk']);model=dict(preprocess=pre,train_features=m.scale(x,pre).tolist(),residual=[.1,-.2,.3],neighbors=15)
  self.assertTrue(np.isfinite(m.knn_predict(model,np.array([[100.,-100]]))).all())
 def test_uniform_width_instead_of_target_injection(self):
  ranks={'d1':[dict(symbol=s) for s in [f'S{x}' for x in range(40)]],'d2':[dict(symbol=s) for s in [f'S{x}' for x in range(40)]]}
  train=[dict(start='d1',holdings=['S0','S35','ETF'])];n,a=m.choose_width(ranks,train,{'ETF'})
  self.assertEqual(n,36);self.assertEqual(len(a),5)
 def test_later_feature_changes_cannot_change_training_rows(self):
  rng=np.random.default_rng(0);X=rng.normal(size=(4,5,2));g=np.ones((4,5),bool);b=rng.normal(size=(4,5));order=np.array([[1,3,2],[2,4,0]])
  with patch.object(m,'TRAIN',2):
   first=m.training_rows(X,g,b,order);X[2:]*=1000;b[2:]*=-100;second=m.training_rows(X,g,b,order)
  for i in range(4):np.testing.assert_array_equal(first[i],second[i])
 def test_picks_use_only_scores_and_known_eligibility(self):
  score=np.array([[1,1,0,-1],[2,1,0,-1]]);base=np.array([[0,.1,0,0],[0,0,0,0]]);good=np.array([[True,True,True,False],[True,False,False,False]])
  ch=m.picks(score,base,good,[list('ABCD')]*2);self.assertEqual(ch,[[1,0,2],[]])
 def test_exact_tree_selection_on_training_only(self):
  X=np.arange(12).reshape(6,2).astype(float);y=np.array([3,2,1,1,3,2.]);base=np.zeros((2,3));good=np.ones((2,3),bool);order=np.array([[0,1,2],[1,2,0]])
  with patch.object(m,'TRAIN',2):model,trials=m.fit_tree(X,y,[],base,good,order,[list('ABC')]*2,False)
  self.assertEqual(model['config']['matching']['order_exact'],2)
  self.assertEqual(model['config']['leaves'],min(t['leaves'] for t in trials if t['matching']['order_exact']==2))

if __name__=='__main__':unittest.main()
