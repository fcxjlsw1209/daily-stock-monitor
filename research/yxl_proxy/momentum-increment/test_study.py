import unittest
import numpy as np
import pandas as pd
from types import SimpleNamespace
from study import components,label_valid,adverse,block_interval


class Tests(unittest.TestCase):
    def test_flat_price_zero_improvement_and_weight(self):
        ix=pd.bdate_range('2020-01-01',periods=300)
        c=pd.DataFrame({'A':100.,'B':200.},index=ix)
        valid=c>0;score=c*0+.7;mom=c*0+.8
        base={'atr':c*.02,'cmf':c*0,'score':score,'base':{'valid':valid,'ranks':{'long_momentum':mom}},'masks':{'broad':valid & (np.arange(len(c))[:,None]>200)}}
        f,h,new=components(SimpleNamespace(close=c),{'adj_close':c},base)
        np.testing.assert_allclose(f.iloc[2:],0)
        # Tied percentile ranks average to .75 for two symbols.
        self.assertAlmostEqual(new['score'].iloc[-1,0],.3*.8+6/7*(.7-.3*.8)+.1*.75)
        pd.testing.assert_frame_equal(new['masks']['broad'],base['masks']['broad'])

    def test_labels_purged_at_boundary(self):
        ix=pd.bdate_range('2020-01-01',periods=20)
        v=label_valid(ix,5,ix[10])
        self.assertTrue(v.iloc[4]);self.assertFalse(v.iloc[5]);self.assertFalse(v.iloc[-1])

    def test_mae_ignores_exit_day_low(self):
        ix=pd.bdate_range('2020-01-01',periods=3)
        t=pd.DataFrame([{'symbol':'X','entry_date':str(ix[0].date()),'exit_date':str(ix[2].date()),'entry_price':100.,'exit_price':98.}])
        lows=pd.DataFrame({'X':[99.,97.,1.]},index=ix)
        self.assertAlmostEqual(adverse(t,None,{'low':lows})[0],-.03)

    def test_paired_zero_bootstrap(self):
        r=block_interval(np.zeros(80),reps=30)
        self.assertEqual(r['low95'],0);self.assertEqual(r['high95'],0)


if __name__=='__main__':unittest.main()
