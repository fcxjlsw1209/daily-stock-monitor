import unittest
import numpy as np
import tweaks as m

class MicroTests(unittest.TestCase):
 def fixture(self):
  n=3;ss=[list('ABCD')]*n
  d=dict(core=np.tile([.1,.08,.06,.04],(n,1)),support=np.zeros((n,4)),eligible=np.ones((n,4),bool),vol=np.ones((n,4))*.02,price=np.array([[10,10,10,10],[20,10,10,10],[21,10,10,10.]],float),weak=np.array([0,1,0],bool),high=np.array([1,0,0],bool))
  l={'a':dict(score=np.tile([-.5,-.1,.2,.4],(n,1)),active=np.ones(n))};meta=dict(symbols=ss,dates=['d0','d1','d2']);spec=dict(layer='a',budget=0.,mode='global',retain=3,profit=0.)
  return meta,d,l,spec
 def test_zero_layer_keeps_anchor(self):
  a,d,l,s=self.fixture();chosen,sc,_=m.decisions(s,a,d,l,count=3)
  self.assertEqual(chosen,[[0,1,2]]*3);np.testing.assert_array_equal(sc,d['core']+d['support'])
 def test_core_is_preserved_and_support_donates_budget(self):
  a,d,l,s=self.fixture();d['support'][:]=.12;s['budget']=.10
  _,sc,_=m.decisions(s,a,d,l,count=3)
  np.testing.assert_allclose(sc,d['core']+.12*(1-.10/.48)+.1*l['a']['score'])
 def test_inactive_layer_returns_budget(self):
  a,d,l,s=self.fixture();s.update(mode='weak',budget=.1)
  _,sc,_=m.decisions(s,a,d,l,count=3);np.testing.assert_array_equal(sc[0],d['core'][0]);self.assertFalse(np.array_equal(sc[1],d['core'][1]))
 def test_profit_penalty_uses_own_prior_positions_only(self):
  a,d,l,s=self.fixture();s['profit']=.1
  ch,sc,ev=m.decisions(s,a,d,l,count=3,details=True)
  self.assertEqual(ch[0],[0,1,2]);self.assertNotIn(0,ch[1]);self.assertIn(0,ch[2])
  q={x['ticker']:x for x in ev[1]['candidates']};self.assertGreater(q['A']['profit_penalty'],0);self.assertEqual(q['D']['profit_penalty'],0)
 def test_future_features_and_prices_do_not_affect_prefix(self):
  a,d,l,s=self.fixture();s.update(profit=.05,budget=.1)
  c,sc,_=m.decisions(s,a,d,l,count=2);d['price'][2:]*=999;d['core'][2:]=999;l['a']['score'][2:]=999
  c2,sc2,_=m.decisions(s,a,d,l,count=2);self.assertEqual(c,c2);np.testing.assert_array_equal(sc,sc2)
 def test_matured_training_excludes_future_labels(self):
  dates=['2014-12-31','2019-11-01','2019-12-30','2020-01-31'];ends=['2015-01-31','2019-12-01','2020-01-31','2020-02-28']
  self.assertEqual(m.mature(dates,ends,'2019-12-31'),[0,1])
 def test_neutral_missing_rank(self):
  q=m.ranks([1,np.nan,3],[True,True,False]);self.assertEqual(q[1],0);self.assertEqual(q[2],0)
 def test_weak_tree_only_uses_its_active_budget(self):
  a,d,l,s=self.fixture();l['a']['active']=d['weak'].astype(float);s['budget']=.15
  b=m.activate(s,d,3,l['a']);np.testing.assert_array_equal(b,[0,.15,0])

if __name__=='__main__':unittest.main()
