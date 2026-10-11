import unittest
import pandas as pd
import numpy as np
from validate import trade,replay,score_of

class ReplayTests(unittest.TestCase):
    def test_fee_is_paid_by_sells_and_buys_and_conserves_nav(self):
        px=pd.Series({'A':10.,'B':20.});u,c,fee,orders=trade({'A':10.},0.,px,['B'],.001,pd.Timestamp('2021-01-01'))
        expected=100.*.999/1.001
        self.assertAlmostEqual(u['B']*20,expected)
        self.assertAlmostEqual(expected+fee,100.)
        self.assertEqual(c,0.);self.assertEqual(len(orders),2)
    def test_no_trade_if_units_already_equal_and_no_fee(self):
        u,c,fee,orders=trade({'A':5.,'B':2.5},0.,pd.Series({'A':10.,'B':20.}),['A','B'],.005,pd.Timestamp('2021-01-01'))
        self.assertLess(fee,1e-12);self.assertEqual(orders,[])
    def test_fixed_shares_not_daily_equal_weights(self):
        dates=pd.date_range('2021-01-01',periods=4);p=pd.DataFrame({'A':[10.,20.,10.,20.],'B':[10.,10.,10.,10.]},index=dates)
        oo=[dict(date=str(dates[0].date()),signal_date=str(dates[0].date()),symbols=['A','B'],scores=[0.,0.])]
        z,e,tt,_=replay(p,p,oo,0.,'close',dates[0],dates[-1]);self.assertTrue(np.allclose(e,[100000.,150000.,100000.,150000.]))
    def test_next_open_uses_actual_next_open_and_first_fee_not_erased(self):
        dates=pd.date_range('2021-01-01',periods=4);p=pd.DataFrame({'A':[10.,20.,20.,20.]},index=dates);op=pd.DataFrame({'A':[8.,16.,20.,20.]},index=dates)
        oo=[dict(date=str(dates[0].date()),signal_date=str(dates[0].date()),symbols=['A'],scores=[0.])]
        z,e,_,_=replay(p,op,oo,.001,'open',dates[0],dates[-1]);self.assertEqual(e.iloc[0],100000.);self.assertAlmostEqual(e.iloc[-1],100000/1.001/16*20)
    def test_missing_held_close_raises(self):
        dates=pd.date_range('2021-01-01',periods=3);p=pd.DataFrame({'A':[10.,np.nan,20.]},index=dates)
        oo=[dict(date=str(dates[0].date()),signal_date=str(dates[0].date()),symbols=['A'],scores=[0.])]
        with self.assertRaises(AssertionError):replay(p,p,oo,0.,'close',dates[0],dates[-1])
    def test_correction_preserves_core_weight(self):
        x=np.array([[[1.,2.,3.],[0.,1.,4.]]]);spec=dict(recipe=(0,1,2,.5,.35,.15),gamma=.1,correction='a');ext=np.array([[4.,2.]])
        result=score_of(spec,x,ext);expected=.5*x[:,:,0]+.28*x[:,:,1]+.12*x[:,:,2]+.1*ext
        np.testing.assert_allclose(result,expected)
if __name__=='__main__':unittest.main()
