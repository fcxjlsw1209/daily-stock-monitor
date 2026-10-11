import unittest,copy
import numpy as np
import refine as r

class RefinementTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.m,cls.d,cls.layers,cls.x,cls.px,cls.op=r.load()
 def test_zero_change_is_exact_industry_reference(self):
  spec=r.default();actual=r.score(spec,self.d,self.layers,self.x,69)
  old=dict(layer='industry_mix',budget=.15,mode='highvol',retain=3,profit=0.)
  _,expected,_=r.t.decisions(old,self.m,self.d,self.layers)
  np.testing.assert_allclose(actual,expected,atol=1e-14)
 def test_no_industry_is_original_anchor(self):
  s=r.default();s['budget']=0
  np.testing.assert_allclose(r.score(s,self.d,self.layers,self.x,69),self.d['core']+self.d['support'],atol=1e-14)
 def test_training_scores_ignore_later_feature_changes(self):
  f=r.read(r.P/'FROZEN.json')['selected']['annual_primary']['spec'];expected=r.score(f,self.d,self.layers,self.x,35)
  x=copy.deepcopy(self.x);d=copy.deepcopy(self.d);layers=copy.deepcopy(self.layers)
  for z in x.values():z[35:]=99999
  for z in d.values():z[35:]=True if z.dtype==bool else -99999
  for obj in layers.values():obj['score'][35:]=1234;obj['active'][35:]=0
  np.testing.assert_array_equal(r.score(f,d,layers,x,35),expected)
 def test_inactive_adjustment_returns_budget(self):
  s=r.default();s.update(addon='weak_tree_2',addon_budget=.05)
  d=copy.deepcopy(self.d);d['high'][:]=False
  np.testing.assert_allclose(r.score(s,d,self.layers,self.x,69),d['core']+d['support'],atol=1e-14)
 def test_all_rules_keep_momentum_and_bounded_adjustments(self):
  rows=r.read(r.P/'TRAIN_TRIALS.json')
  for z in rows:
   s=z['spec'];self.assertLessEqual(s['core_blend'],.25);self.assertLessEqual(s['budget']+s['addon_budget'],.25)
   self.assertAlmostEqual(sum(s['industry_weights']),1.);self.assertIn(s['retain'],[3,5])
  wf=r.read(r.P/'TRAIN_WALKFORWARD.json');self.assertTrue(wf['pair_seeds_excluded'])

if __name__=='__main__':unittest.main()
