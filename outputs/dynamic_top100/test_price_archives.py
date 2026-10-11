import unittest,tempfile
from pathlib import Path
import pandas as pd
import audit_price_archives as a
class ArchiveTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.old=a.D;a.D=Path(self.tmp.name)
 def tearDown(self):a.D=self.old;self.tmp.cleanup()
 def data(self):return pd.DataFrame(dict(date=['2010-01-04','2010-01-05 00:00:00'],symbol=['X','X'],open=[10.,11.],high=[12.,12.],low=[9.,10.],close=[11.,11.],volume=[100.,100.]))
 def test_mixed_dates(self):
  d=a.audit(self.data(),'test');self.assertEqual(d.date.tolist(),['2010-01-04','2010-01-05']);self.assertFalse(d.quarantine.any())
 def test_missing_and_bad_ohlc(self):
  d=self.data();d.loc[0,'open']=None;d.loc[1,'close']=20;self.assertTrue(a.audit(d,'test').quarantine.all())
 def test_duplicate_dates(self):
  d=self.data();d.loc[1,'date']='2010-01-04';self.assertTrue(a.audit(d,'test').quarantine.all())
 def test_halted_not_executable(self):
  d=self.data();d.symbol=['TWTR','ATVI'];d.date=['2022-10-28','2023-10-13'];self.assertTrue(a.audit(d,'test').quarantine.all())
 def test_does_not_fill_or_certify(self):
  d=self.data();d.loc[1,'close']=None;r=a.audit(d,'test');self.assertTrue(pd.isna(r.loc[1,'close']));self.assertFalse(r.identity_and_adjustments_verified.any())
if __name__=='__main__':unittest.main()
