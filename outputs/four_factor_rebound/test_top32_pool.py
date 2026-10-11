import unittest
import pandas as pd
from top32_pool import cap_series
class CapTest(unittest.TestCase):
 def test_disclosure_and_split(self):
  idx=pd.date_range('2020-01-01',periods=6);p=pd.Series([20,20,20,10,10,10],index=idx);raw=pd.Series(1.,index=idx);splits=pd.Series([1,1,1,2,1,1],index=idx)
  r=pd.DataFrame([dict(end='2020-01-01',filed='2020-01-02',val=100),dict(end='2020-01-05',filed='2020-01-06',val=999)])
  c,u=cap_series(r,p,raw,splits)
  self.assertTrue(c.iloc[:2].isna().all());self.assertEqual(c.iloc[2],2000);self.assertEqual(c.iloc[3],2000);self.assertEqual(c.iloc[-1],2000)
  changed=r.copy();changed.loc[1,'val']=1e9
  pd.testing.assert_series_equal(c,cap_series(changed,p,raw,splits)[0])
 def test_postsplit_gaap_filing_units(self):
  idx=pd.date_range('2020-01-01',periods=6);p=pd.Series([20,20,20,10,10,10],index=idx);one=pd.Series(1.,index=idx);splits=pd.Series([1,1,1,2,1,1],index=idx)
  r=pd.DataFrame([dict(end='2020-01-01',filed='2020-01-04',val=200,concept='CommonStockSharesOutstanding')])
  c,_=cap_series(r,p,one,splits)
  self.assertEqual(c.iloc[4],2000)
 def test_conflicting_class_values_excluded(self):
  idx=pd.date_range('2020-01-01',periods=5);p=pd.Series(1.,index=idx)
  r=pd.DataFrame([dict(end='2020-01-01',filed='2020-01-02',val=x) for x in [100,200]])
  self.assertTrue(cap_series(r,p,p,p)[0].isna().all())
if __name__=='__main__':unittest.main()
