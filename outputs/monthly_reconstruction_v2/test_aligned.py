import unittest
import numpy as np
import pandas as pd
from aligned_momentum import trend_quality


class AlignedFeaturesTests(unittest.TestCase):
    def test_exponential_trend_has_exact_slope_and_quality(self):
        t=np.arange(120);px=pd.DataFrame({'A':100*np.exp(.001*t)})
        quality,r2=trend_quality(px,window=30,lag=5)
        self.assertTrue(quality.iloc[:34].isna().all().all())
        np.testing.assert_allclose(quality.iloc[34:],.252,rtol=1e-8)
        np.testing.assert_allclose(r2.iloc[34:],1.,rtol=1e-8)

    def test_matches_direct_regression_ending_before_skip(self):
        rng=np.random.default_rng(11);logp=4+np.cumsum(rng.normal(.0008,.02,250))
        px=pd.DataFrame({'A':np.exp(logp)});quality,r2=trend_quality(px,window=70,lag=20)
        for i in [100,160,249]:
            y=logp[i-20-69:i-20+1];x=np.arange(70);beta=np.polyfit(x,y,1)[0];rr=np.corrcoef(x,y)[0,1]**2
            self.assertAlmostEqual(r2.at[i,'A'],rr,places=9)
            self.assertAlmostEqual(quality.at[i,'A'],beta*252*rr,places=9)

    def test_recent_skipped_changes_cannot_affect_current_score(self):
        rng=np.random.default_rng(12);px=pd.DataFrame({'A':100*np.exp(np.cumsum(rng.normal(0,.01,250)))})
        before,_=trend_quality(px,window=70,lag=20);changed=px.copy();changed.iloc[-20:]*=10
        after,_=trend_quality(changed,window=70,lag=20)
        self.assertAlmostEqual(before.iloc[-1,0],after.iloc[-1,0],places=10)


if __name__=='__main__':
    unittest.main()
