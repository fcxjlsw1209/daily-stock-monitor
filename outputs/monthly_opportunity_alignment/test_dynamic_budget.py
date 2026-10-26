import unittest
import numpy as np
from dynamic_budget import budget_series


class DynamicTests(unittest.TestCase):
    def setUp(self):
        self.d = np.arange('2020-01', '2022-02', dtype='datetime64[M]').astype('datetime64[D]')
        self.e = (self.d.astype('datetime64[M]')+1).astype('datetime64[D]')

    def test_cold_start_and_constant_positive(self):
        b,a = budget_series(self.d,self.e,np.ones(len(self.d))*.01,.05)
        np.testing.assert_array_equal(b[:12],0.)
        np.testing.assert_allclose(b[12:],.05)
        self.assertEqual(a[12]['latest_end'],str(self.d[12]))

    def test_current_future_and_prefix_invariance(self):
        v = np.linspace(-.01,.02,len(self.d)); b,_ = budget_series(self.d,self.e,v,.05)
        for i in [5,12,20]:
            alt=v.copy();alt[i:]=1e6
            z,_=budget_series(self.d,self.e,alt,.05)
            np.testing.assert_array_equal(z[:i+1],b[:i+1])
            p,_=budget_series(self.d[:i+1],self.e[:i+1],v[:i+1],.05)
            np.testing.assert_array_equal(p,b[:i+1])

    def test_zero_negative_missing_and_bounds(self):
        for value in [0.,-.1,np.nan]:
            b,_=budget_series(self.d,self.e,np.full(len(self.d),value),.05)
            np.testing.assert_array_equal(b,0.)
        b,_=budget_series(self.d,self.e,np.tile([-.01,.03],13)[:len(self.d)],.05)
        self.assertTrue(((b>=0)&(b<=.05)).all())

    def test_unmatured_label_not_used(self):
        ends=self.e.copy();ends[11]=self.d[14]
        b,a=budget_series(self.d,ends,np.ones(len(self.d))*.01,.05)
        self.assertEqual(b[12],0.)
        self.assertEqual(a[12]['count'],11)

    def test_fixed_window_drops_old_evidence(self):
        v=np.ones(len(self.d))*.01;v[:12]=-1.
        b,_=budget_series(self.d,self.e,v,.05)
        self.assertEqual(b[12],0.)
        self.assertAlmostEqual(b[24],.05)

    def test_missing_recent_month_not_backfilled_from_old(self):
        v=np.ones(len(self.d))*.01;v[15]=np.nan
        b,_=budget_series(self.d,self.e,v,.05)
        self.assertEqual(b[20],0.)


if __name__=='__main__':unittest.main()
