import unittest
import numpy as np
import pandas as pd
from price_units import raw_to_split_adjusted,split_adjusted_to_raw
class RawUnitsTests(unittest.TestCase):
    def fixture(self):
        return pd.DataFrame(dict(open=[100.,50.,51.],high=[102.,52.,53.],low=[98.,48.,49.],close=[100.,50.,51.],volume=[10.,20.,20.],dividend=[2.,0.,1.],split=[1.,2.,1.],adj_close=[48.,49.,51.]),index=pd.date_range('2020-01-01',periods=3))
    def test_round_trip_raw_units(self):
        d=self.fixture();signal,raw=raw_to_split_adjusted(d)
        rebuilt=split_adjusted_to_raw(signal)
        for k in ['open','high','low','close','volume','dividend','split','raw_per_signal']:
            np.testing.assert_allclose(raw[k],rebuilt[k])
        self.assertEqual(signal.close.tolist(),[50.,50.,51.])
        self.assertEqual(signal.adj_close.tolist(),d.adj_close.tolist())
    def test_cash_adjustment_never_creates_shares(self):
        d=self.fixture();d['split']=1.;signal,raw=raw_to_split_adjusted(d)
        np.testing.assert_allclose(raw.raw_per_signal,1.)
        np.testing.assert_allclose(signal.close,d.close)
    def test_unknown_actions_fail(self):
        for val in [0.,-1.,np.nan]:
            d=self.fixture();d.iloc[1,d.columns.get_loc('split')]=val
            with self.assertRaises(ValueError):raw_to_split_adjusted(d)
