import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
import cached_prices as prices

class CachedPriceTests(unittest.TestCase):
    def test_duplicate_or_unsorted_sessions_rejected(self):
        for dates in [['2020-01-02','2020-01-02'],['2020-01-03','2020-01-02']]:
            frame=pd.DataFrame({'close':[1.,2.],'volume':[3.,4.]},index=pd.to_datetime(dates))
            with self.assertRaises(ValueError):prices.validate_frame(frame)

    def test_invalid_bars_stay_missing_and_valid_prices_unchanged(self):
        frame=pd.DataFrame({'close':[10.,0.,-1.,11.],'volume':[5.,5.,5.,-1.]},index=pd.date_range('2020-01-01',periods=4))
        result=prices.validate_frame(frame)
        self.assertEqual(result.iloc[0].to_dict(),frame.iloc[0].to_dict())
        self.assertTrue(result.iloc[1:].isna().all().all())
        self.assertEqual(frame.iloc[1]['close'],0.)

    def test_csv_identity_and_bytes_must_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'monthly_2016_extension/prices';folder.mkdir(parents=True)
            p=folder/'BUD.csv';p.write_text('Date,Adj Close,Volume\n2020-01-02,10,5\n')
            m=p.with_suffix('.source.json')
            m.write_text(json.dumps({'symbol':'BUD','sha256':prices.sha(p)}))
            frame,inputs=prices.load_supplemental('BUD',tmp)
            self.assertEqual(frame.iloc[0]['close'],10.);self.assertEqual(len(inputs),2)
            p.write_text(p.read_text().replace(',10,',',99,'))
            with self.assertRaisesRegex(ValueError,'identity/hash'):prices.load_supplemental('BUD',tmp)
            m.write_text(json.dumps({'symbol':'HSBC','sha256':prices.sha(p)}))
            with self.assertRaisesRegex(ValueError,'identity/hash'):prices.load_supplemental('BUD',tmp)

    def test_chl_post_delisting_vendor_segment_never_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'dynamic_top100/additional_sources/tiingo';folder.mkdir(parents=True)
            p=folder/'CHL_prices.json';p.write_text(json.dumps([
                {'date':'2020-12-31T00:00:00Z','adjClose':10.,'adjVolume':10.},
                {'date':'2021-01-04T00:00:00Z','adjClose':999.,'adjVolume':10.}]))
            (folder/'CHL_prices.provenance.json').write_text(json.dumps({'sha256':prices.sha(p)}))
            (folder/'CHL_metadata.json').write_text(json.dumps({'ticker':'CHL','name':'China Mobile Ltd'}))
            frame,_=prices.load_supplemental('CHL',tmp)
            self.assertEqual(len(frame),1);self.assertEqual(frame.index[0],pd.Timestamp('2020-12-31'))

    def test_missing_csv_not_fabricated(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(prices.load_supplemental('BUD',tmp),(None,{}))
            self.assertEqual(prices.load_supplemental('UNKNOWN',tmp),(None,{}))

if __name__=='__main__':unittest.main()
