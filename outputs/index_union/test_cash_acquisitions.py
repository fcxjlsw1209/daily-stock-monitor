import unittest,copy
import pandas as pd
from cash_acquisitions import apply,finish_valuation
from union_execution import engine,execute
from test_historical_backtest import inputs
from historical_backtest import run_account

class CashAcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.dates=pd.bdate_range('2020-01-01',periods=6)
        e=engine('accel');d=dict(session=str(self.dates[1].date()),sells=[],buys=[dict(symbol='A',budget=10000.,score=.9,atr=2.)])
        self.state,*_=e.execute(e.initial_state(),d,{'A':dict(open=100.,close=100.)},self.dates)
        self.event=dict(event_id='cash1',security_id='A',kind='cash_acquisition',effective_session=str(self.dates[2].date()),cash_per_share=120.,payment_session=str(self.dates[4].date()),verified=True,source_sha256='test_receipt',price_basis='raw_with_explicit_actions')
        self.decision=dict(session=str(self.dates[2].date()),sells=[dict(symbol='A',reason='exit')],buys=[])
    def test_entitlement_is_not_spendable_cash(self):
        s,d,t,a,c=apply(self.state,self.decision,[self.event],self.dates[2],self.dates)
        self.assertEqual(s['cash'],self.state['cash']);self.assertNotIn('A',s['positions'])
        self.assertEqual(d['sells'],[]);self.assertEqual(len(t),1);self.assertEqual(t[0]['exit_fee'],0)
        after,row,*_=execute('accel',s,d,{},self.dates)
        after,row=finish_valuation(after,row,self.state['equity'])
        self.assertAlmostEqual(after['equity'],self.state['cash']+self.state['positions']['A']['qty']*120)
        self.assertEqual(row['cash_receivables'],s['cash_receivables'][0]['amount'])
        self.assertIn('A',self.state['positions'])
    def test_payment_credited_once(self):
        s,*_=apply(self.state,self.decision,[self.event],self.dates[2],self.dates)
        paid,*_=apply(s,self.decision,[self.event],self.dates[4],self.dates)
        again,*_=apply(paid,self.decision,[self.event],self.dates[4],self.dates)
        self.assertEqual(paid['cash'],again['cash']);self.assertEqual(paid['cash_receivables'],[])
        self.assertAlmostEqual(paid['cash'],self.state['cash']+self.state['positions']['A']['qty']*120)
    def test_unknown_payment_is_never_assumed_available(self):
        self.event['payment_session']=None
        s,*_=apply(self.state,self.decision,[self.event],self.dates[2],self.dates)
        s,*_=apply(s,self.decision,[self.event],self.dates[-1],self.dates)
        self.assertEqual(s['cash'],self.state['cash']);self.assertEqual(len(s['cash_receivables']),1)
    def test_stock_or_cvr_not_silently_dropped(self):
        self.event['kind']='cash_stock_cvr_acquisition'
        with self.assertRaises(ValueError):apply(self.state,self.decision,[self.event],self.dates[2],self.dates)
    def test_adjusted_share_units_rejected(self):
        self.event['price_basis']='split_adjusted'
        with self.assertRaises(ValueError):apply(self.state,self.decision,[self.event],self.dates[2],self.dates)
    def test_backtest_crosses_cash_merger_without_fake_terminal_bar(self):
        data=inputs();dates=data['close'].index
        normal=run_account('accel',data,dates[300],dates[-1]);buy=normal['orders'][0];symbol=buy['symbol']
        day=dates[dates.get_loc(pd.Timestamp(buy['date']))+1]
        data['cash_acquisitions']=[dict(self.event,security_id=symbol,effective_session=str(day.date()),payment_session=None)]
        data['eligible'].loc[day:,symbol]=False
        data['close'].loc[day:,symbol]=float('nan')
        result=run_account('accel',data,dates[300],dates[-1])
        self.assertEqual(result['status'],'complete',result.get('error'))
        self.assertTrue(any(t['reason']=='cash_acquisition_receivable' for t in result['trades']))
        self.assertEqual(len(result['state']['cash_receivables']),1)

if __name__=='__main__':unittest.main()
