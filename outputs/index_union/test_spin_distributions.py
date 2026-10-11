import unittest,copy
from spin_distributions import apply
class SpinTests(unittest.TestCase):
 def setUp(self):
  self.state=dict(cash=10.,fees=0.,positions={'P':dict(qty=30.,cost=300.,entry_price=10.,atr=2.,dividends=0.)})
  self.decision=dict(buys=[],sells=[dict(symbol='P')])
  self.bars={'P':dict(open=8.,close=8.,split=1.25)}
  self.event=dict(event_id='e',security_id='P',child_security_id='C',execution_session='2020-01-02',known_at='2019-12-01T00:00:00Z',terms_verified=True,source_sha256='terms',provider_adjustment=1.25,ratio=1/3,child_open=dict(price=6.,session='2020-01-02',kind='regular_way_open',source_sha256='quote'))
 def run_spin(self):return apply(self.state,self.decision,self.bars,[self.event],'2020-01-02')
 def test_receives_child_without_parent_dilution(self):
  s,d,b,a=self.run_spin();self.assertEqual(s['positions']['P']['qty'],30);self.assertEqual(s['positions']['P']['cost'],300);self.assertAlmostEqual(s['cash'],69.97);self.assertEqual(b['P']['split'],1);self.assertAlmostEqual(s['positions']['P']['dividends'],59.97);self.assertEqual(d['sells'],self.decision['sells']);self.assertEqual(self.state['cash'],10)
 def test_new_buyer_not_entitled(self):
  self.state['positions']={};self.decision['buys']=[dict(symbol='P',atr=2.,reference_close=10.)]
  s,d,b,a=self.run_spin();self.assertEqual(s['cash'],10);self.assertEqual(a,[]);self.assertEqual(d['buys'][0]['atr'],1.6)
 def test_no_repeat_cash(self):
  s,d,b,a=self.run_spin();s2,*_=apply(s,d,self.bars,[self.event],'2020-01-02');self.assertEqual(s,s2)
 def test_no_future_or_missing_quotes(self):
  self.event['child_open']['price']=float('nan')
  with self.assertRaises(ValueError):self.run_spin()
 def test_wrong_adjustment_rejected(self):
  self.bars['P']['split']=1.3
  with self.assertRaises(ValueError):self.run_spin()
if __name__=='__main__':unittest.main()
