import unittest
import numpy as np
import pandas as pd
from infer_window import replay,core_fit,FEATURES,CORE

class WindowTests(unittest.TestCase):
    def test_fixed_units_do_not_rebalance_daily(self):
        dates=pd.date_range('2022-01-01',periods=3)
        prices=pd.DataFrame({'A':[100,200,100],'B':[100,100,100],'C':[100,100,100]},index=dates)
        nav,orders=replay(prices,[{'start':str(dates[0].date()),'end':str(dates[-1].date())}],[['A','B','C']],1.)
        self.assertAlmostEqual(nav.iloc[1],4/3)
        self.assertAlmostEqual(nav.iloc[-1],1.)
        self.assertAlmostEqual(orders[0]['period_return'],0.)

    def test_months_carry_the_same_account_capital(self):
        dates=pd.date_range('2022-01-01',periods=3)
        prices=pd.DataFrame({'A':[100,200,400],'B':[100,200,400],'C':[100,200,400]},index=dates)
        events=[{'start':str(dates[0].date()),'end':str(dates[1].date())},{'start':str(dates[1].date()),'end':str(dates[2].date())}]
        nav,orders=replay(prices,events,[['A','B','C']]*2,1.)
        self.assertEqual(len(nav),3)
        self.assertAlmostEqual(nav.iloc[-1],4.)
        self.assertAlmostEqual(orders[1]['start_nav'],orders[0]['end_nav'])

    def test_pure_core_cannot_explain_reversed_identical_features(self):
        D=np.zeros((2,len(FEATURES)));D[0,:len(CORE)]=.5;D[1,:len(CORE)]=-.5
        fit=core_fit(D)
        self.assertLess(fit['margin'],0.)
        self.assertTrue((fit['w']>=0).all())
        self.assertAlmostEqual(fit['w'].sum(),1.)

if __name__=='__main__':unittest.main()
