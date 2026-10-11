import unittest
import pandas as pd
from vendor_pool import align_snapshots
from collect_vendor_caps import parse
class VendorTest(unittest.TestCase):
 def test_no_future_or_same_day_snapshots(self):
  x=pd.DataFrame({'date':['2020-01-02','2020-01-04'],'market_cap_usd':[100.,500.]});idx=pd.date_range('2020-01-01',periods=6)
  s,_=align_snapshots(x,idx);self.assertTrue(s.iloc[:2].isna().all());self.assertEqual(s.iloc[2],100);self.assertEqual(s.iloc[3],100);self.assertEqual(s.iloc[4],500)
  x.loc[1,'market_cap_usd']=1e9;t,_=align_snapshots(x,idx);pd.testing.assert_series_equal(s.iloc[:4],t.iloc[:4])
 def test_stale_and_units(self):
  x=parse(b'data = [{"d":1577836800,"m":123}];');self.assertEqual(x.market_cap_usd.iloc[0],12300000)
  s,_=align_snapshots(x,pd.date_range('2020-03-01',periods=2));self.assertTrue(s.isna().all())
if __name__=='__main__':unittest.main()
