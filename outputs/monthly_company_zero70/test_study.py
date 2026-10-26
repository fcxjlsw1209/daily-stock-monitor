import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
spec=importlib.util.spec_from_file_location('zero70_driver_tests',Path(__file__).with_name('study.py'))
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)

class Driver(unittest.TestCase):
 def inputs(self):
  a=dict(baseline=np.zeros((2,5)),core=np.zeros((2,5)),core_weight=np.full((2,1),.52),eligible=np.ones((2,5),bool),state_calm10=np.zeros(2,bool),state_weak_narrow=np.zeros(2,bool),pre_score=np.array([[.3,.2,.10,.09,.085]]*2),effects=np.full((2,5),.1))
  for factor in s.FACTORS:
   a[factor]=np.zeros((2,5));a['available_'+factor]=np.zeros((2,5),bool)
  return a,dict(symbols=[list('ABCDE')]*2)
 def rows(self):
  metric=lambda c:dict(cagr=c,sharpe=1.6,max_drawdown=-.3,total_return=1)
  return [dict(spec=z,training=metric(.5 if z['id']==1 else .49),early=metric(.4),recent=metric(.6)) for z in s.configs()]
 def test_six_fixed_configs(self):
  self.assertEqual(len(s.configs()),6);self.assertEqual([c['id'] for c in s.configs()],list(range(6)))
 def test_missing_zero_exact_and_unavailable_finite_zero_increment(self):
  a,m=self.inputs();cfg=s.configs()[2]
  np.testing.assert_array_equal(s.augmented_effects(cfg,a,2),a['effects'])
  a[cfg['factor']][:]=10
  np.testing.assert_array_equal(s.augmented_effects(cfg,a,2),a['effects'])
  a['available_'+cfg['factor']][:]=True;a[cfg['factor']][:]=np.nan
  np.testing.assert_array_equal(s.augmented_effects(cfg,a,2),a['effects'])
 def test_increment_formula_and_no_mutation(self):
  a,m=self.inputs();cfg=s.configs()[2];k=cfg['factor'];a[k][0]=[-1,-.05,0,.05,1];a['available_'+k][0]=True
  before=a['effects'].copy();z=s.augmented_effects(cfg,a,2)
  np.testing.assert_allclose(z[0],before[0]+.25*np.tanh(a[k][0]/.05));np.testing.assert_array_equal(a['effects'],before)
  self.assertLessEqual(np.max(np.abs(z-before)),.25)
 def test_future_prefix_and_unknown_spec(self):
  a,m=self.inputs();cfg=s.configs()[2];before=s.augmented_effects(cfg,a,1)
  a['effects'][1]=np.nan;a[cfg['factor']][1]=1e100;a['available_'+cfg['factor']][1]=True
  np.testing.assert_array_equal(before,s.augmented_effects(cfg,a,1))
  with self.assertRaises(ValueError):s.augmented_effects(dict(cfg,id=99),a,1)
 def test_real_selector_changes_only_third_boundary_with_score_bound(self):
  a,m=self.inputs();cfg=s.configs()[2];k=cfg['factor'];a[k][:,3]=1;a['available_'+k][:,3]=True
  policy=dict(budget=.05,ceiling=5,gap=.02)
  with patch.object(s,'read',return_value=policy):
   base,score0=s.choices(s.configs()[1],a,m,2);new,score1=s.choices(cfg,a,m,2)
  self.assertEqual(base,[[0,1,2]]*2);self.assertEqual(new,[[0,1,3]]*2)
  self.assertLessEqual(np.max(np.abs(score1-score0)),.0125+1e-14)
  np.testing.assert_array_equal(score0[:,:2],score1[:,:2])
 def test_selection_uses_training_and_segment_guards(self):
  rows=self.rows();rows[2]['full_cagr']=99
  self.assertEqual(s.select(rows)['primary']['spec']['id'],1)
  rows[3]['training']['cagr']=.6
  self.assertEqual(s.select(rows)['primary']['spec']['id'],3)
  rows[3]['early']['cagr']=.1
  self.assertEqual(s.select(rows)['primary']['spec']['id'],1)
  self.assertEqual(len(s.select(rows)),8)

if __name__=='__main__':unittest.main()
