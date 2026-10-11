import unittest,copy
import pandas as pd
import terminal_baskets
import cash_acquisitions
class TerminalBasketTests(unittest.TestCase):
 def setUp(self):
  self.dates=pd.bdate_range('2019-11-01','2019-11-30')
  self.state=dict(cash=1000.,equity=2100.,fees=1.,positions={'CELG':dict(qty=10.,cost=1001.,entry_fee=1.,dividends=2.,entry_date='2019-11-01',entry_price=100.,atr=2.),'BMY':dict(qty=3)})
  self.decision=dict(buys=[],sells=[dict(symbol='CELG',reason='exit')])
  self.event=dict(event_id='celg',security_id='CELG',execution_session='2019-11-21',effective_session='2019-11-21',known_at='2019-11-20T19:00:00-05:00',terms_verified=True,source_sha256='terms',policy='liquidate_first_tradable_open',price_basis='raw_with_explicit_actions',cash_per_share=50,payment_session=None,components=[dict(security_id='BMY',ratio=1),dict(security_id='BMYRT',ratio=1,documented_first_trade=dict(session='2019-11-21',kind='first_trade',price=2.3,source_sha256='quote'))])
 def run_event(self,e=None,prices=None):
  return terminal_baskets.apply(self.state,self.decision,[e or self.event],'2019-11-21',self.dates,prices if prices is not None else {'BMY':57})
 def test_all_components_cash_and_cost_preserved(self):
  s,d,t,a,c=self.run_event();self.assertAlmostEqual(s['cash'],1000+593*(1-.0005));self.assertEqual(s['cash_receivables'][0]['amount'],500);self.assertEqual(s['positions']['BMY'],{'qty':3});self.assertNotIn('CELG',s['positions']);self.assertAlmostEqual(t[0]['net_pnl'],1093-593*.0005+2-1001)
  self.assertIn('CELG',self.state['positions'])
 def test_missing_successor_quote_blocks_no_zero(self):
  with self.assertRaisesRegex(ValueError,'Missing component'):self.run_event(prices={})
 def test_first_trade_cannot_be_wrong_date_or_fair_value(self):
  e=copy.deepcopy(self.event);e['components'][1]['documented_first_trade']['kind']='fair_value'
  with self.assertRaises(ValueError):self.run_event(e)
 def test_no_duplicate_conversion(self):
  s,d,*_=self.run_event();s2,d2,t,a,c=terminal_baskets.apply(s,d,[self.event],'2019-11-21',self.dates,{})
  self.assertEqual(s,s2);self.assertEqual(t,[])
 def test_future_announced_terms_rejected(self):
  e=copy.deepcopy(self.event);e['known_at']='2019-11-21T10:00:00-05:00'
  with self.assertRaises(ValueError):self.run_event(e)
if __name__=='__main__':unittest.main()
