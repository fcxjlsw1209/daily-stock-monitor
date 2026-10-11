import unittest
import numpy as np
import pandas as pd
from backtest_hybrid import confirmation,simulate,load,features
from backtest_dynamic import simulate as old_simulate

class HybridTests(unittest.TestCase):
    def candles(self):
        idx=pd.date_range('2026-07-01 09:30',periods=9,freq='5min',tz='America/New_York')
        a=pd.DataFrame({'Open':99.,'High':100.,'Low':98.,'Close':99.},index=idx)
        a.iloc[3]=[100,102,100,101]
        a.iloc[4]=[101,102,99.5,101]
        a.iloc[5]=[101,104,101,103]
        a.iloc[6]=[103.5,105,103,104]
        return a
    def test_separate_bars_and_next_open(self):
        r=confirmation(self.candles())
        self.assertEqual(r['status'],'confirmed')
        self.assertIn('10:00:00',r['entry_timestamp'])
        self.assertEqual(r['entry_price'],103.5)
    def test_failed_first_retest_no_retry(self):
        a=self.candles();a.iloc[4,a.columns.get_loc('Close')]=99
        self.assertEqual(confirmation(a)['status'],'failed_first_retest')
    def test_future_candles_do_not_change_signal(self):
        a=self.candles();before=confirmation(a)
        a.iloc[6,1:]=[500,1,400];a.iloc[7:]=1
        self.assertEqual(before,confirmation(a))
        self.assertEqual(before,confirmation(a.iloc[:7]))
    def test_missing_bar_rejects(self):
        self.assertEqual(confirmation(self.candles().drop(self.candles().index[4]))['status'],'missing_signal_bar')
    def test_baseline_matches_existing_engine(self):
        data=load();h=data['QQQ'];f=features(h,data['SPY']);c=f.legacy.shift(1,fill_value=False)
        ev={str(d.date()):{'status':'none'} for d in h.index}
        for cost in [5,10,20]:
            new,_,_,_=simulate(h,c,ev,'A_open_5d',cost,'2026-06-25','2026-09-18')
            old,_,_,_=old_simulate(h,f,'legacy','fixed5',cost_bps=cost,start='2026-06-25',end='2026-09-18')
            np.testing.assert_allclose(new.equity,old.equity,rtol=1e-12)
    def test_same_day_dividend_not_received_and_ledger(self):
        idx=pd.bdate_range('2026-07-01',periods=7)
        h=pd.DataFrame({'Open':100.,'Close':101.,'Dividends':1.},index=idx)
        c=pd.Series(False,index=idx);c.iloc[0]=True
        ev={str(d.date()):{'status':'confirmed','entry_price':100.,'entry_timestamp':str(d)} for d in idx}
        for mode in ['A_open_5d','B_confirm_5d','C_confirm_day']:
            d,t,o,u=simulate(h,c,ev,mode,10,str(idx[0].date()),str(idx[-1].date()))
            self.assertFalse(u);self.assertEqual(len(t),1)
            qty=t.iloc[0]['qty'];expected=0 if mode=='C_confirm_day' else qty*5
            self.assertAlmostEqual(t.dividends.sum(),expected)
            ledger=100000+expected
            for row in o.itertuples():ledger+=(-1 if row.side=='buy' else 1)*row.qty*row.price-row.fee
            self.assertAlmostEqual(ledger,d.equity.iloc[-1])
            self.assertEqual(t.sessions.iloc[0],0 if mode=='C_confirm_day' else 5)
if __name__=='__main__':unittest.main()
