import unittest
import numpy as np
import pandas as pd
from price_units import split_adjusted_to_raw

class PriceUnitTests(unittest.TestCase):
    def sample(self):
        return pd.DataFrame(dict(open=[10.,11.,12.],high=[11.,12.,13.],low=[9.,10.,11.],close=[10.,11.,12.],volume=[100.,100.,100.],dividend=[1.,0.,0.],split=[0.,2.,0.]),index=pd.bdate_range('2020-01-01',periods=3))
    def test_split_day_excluded_from_multiplier(self):
        d=self.sample();raw=split_adjusted_to_raw(d)
        self.assertEqual(raw.raw_per_signal.tolist(),[2.,1.,1.])
        self.assertEqual(raw.open.tolist(),[20.,11.,12.])
        self.assertEqual(raw.dividend.tolist(),[2.,0.,0.])
        np.testing.assert_allclose(raw.close*raw.volume,d.close*d.volume)
    def test_reverse_split(self):
        d=self.sample();d.loc[d.index[1],'split']=.1
        self.assertEqual(split_adjusted_to_raw(d).raw_per_signal.tolist(),[.1,1.,1.])
    def test_missing_split_is_not_assumed_one(self):
        d=self.sample();d.loc[d.index[1],'split']=np.nan
        with self.assertRaises(ValueError):split_adjusted_to_raw(d)
    def test_future_split_cancels_future_adjustment_of_earlier_prices(self):
        d=self.sample();before=split_adjusted_to_raw(d)
        adjusted=d.copy()
        for col in ['open','high','low','close','dividend']:adjusted[col]/=4
        adjusted['volume']*=4
        last=adjusted.iloc[-1].copy();last['split']=4.
        adjusted.loc[adjusted.index[-1]+pd.Timedelta(days=1)]=last
        after=split_adjusted_to_raw(adjusted)
        for col in ['open','high','low','close','dividend','volume']:
            np.testing.assert_allclose(before[col],after[col].iloc[:-1])

if __name__=='__main__':unittest.main()
