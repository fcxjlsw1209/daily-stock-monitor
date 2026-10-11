import tempfile,unittest
from pathlib import Path
import numpy as np
from extract_quantiacs import open_prices,xr

class NetCDFReaderTests(unittest.TestCase):
    def make(self,folder,calendar='proleptic_gregorian'):
        a=xr.DataArray(np.array([[[1.],[2.]]]),dims=['field','time','asset'],coords={'field':['close'],'time':np.array([-4094,0],dtype='int32'),'asset':['id']})
        a.time.attrs={'units':'days since 2019-03-19','calendar':calendar}
        f=Path(folder)/'fixture.nc';a.to_netcdf(f,engine='scipy');return f
    def test_negative_int32_days_decode_without_overflow(self):
        with tempfile.TemporaryDirectory() as folder:
            a=open_prices(self.make(folder));self.assertEqual(str(a.time.values[0])[:10],'2008-01-02');self.assertEqual(str(a.time.values[1])[:10],'2019-03-19')
    def test_unsupported_calendar_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):open_prices(self.make(folder,'360_day'))
    def test_duplicate_dates_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            f=self.make(folder);a=xr.open_dataarray(f,decode_times=False).load();a=a.assign_coords(time=np.array([0,0]));a.time.attrs={'units':'days since 2019-03-19','calendar':'proleptic_gregorian'};dest=Path(folder)/'duplicate.nc';a.to_netcdf(dest,engine='scipy')
            with self.assertRaises(ValueError):open_prices(dest)
if __name__=='__main__':unittest.main()
