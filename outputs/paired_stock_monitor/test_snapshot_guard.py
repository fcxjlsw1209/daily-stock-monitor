import hashlib,json,tempfile,unittest
from pathlib import Path
import pandas as pd
from unittest.mock import patch
from snapshot_guard import prepare_snapshot,restore_rows,missing_days,FIELDS

class GapTests(unittest.TestCase):
 def setUp(self):
  self.days=pd.bdate_range('2026-09-14',periods=8)
  self.old=pd.DataFrame({c:100. for c in FIELDS},index=self.days)
  self.old['Volume']=1000.;self.old['Dividends']=0.;self.old['Stock Splits']=0.
  self.gap=self.days[-2:-1];self.current=self.old.drop(self.gap)
 def test_restore_real_row_only(self):
  fixed=restore_rows(self.current,self.old,self.gap,self.days[-1])
  pd.testing.assert_frame_equal(fixed,self.old,check_freq=False)
  self.assertEqual(len(missing_days(fixed,self.days,self.days[-1])),0)
 def test_split_or_adjustment_change_rejected(self):
  changed=self.current.copy();changed['Adj Close']=90
  with self.assertRaises(ValueError):restore_rows(changed,self.old,self.gap,self.days[-1])
  changed=self.current.copy();changed.loc[self.days[-1],'Stock Splits']=2
  with self.assertRaises(ValueError):restore_rows(changed,self.old,self.gap,self.days[-1])
 def test_dividend_rejected(self):
  changed=self.current.copy();changed.loc[self.days[-1],'Dividends']=1
  with self.assertRaises(ValueError):restore_rows(changed,self.old,self.gap,self.days[-1])
 def write(self,folder,frame):
  folder.mkdir();p=folder/'X.csv.gz';frame.to_csv(p,compression='gzip')
  (folder/'manifest.json').write_text(json.dumps({'snapshot_id':folder.name,'as_of':str(frame.index[-1].date()),'failures':[],'hashes':{'X':hashlib.sha256(p.read_bytes()).hexdigest()}}))
 def test_provenance_and_idempotence(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);self.write(root/'old',self.old.iloc[:-1]);self.write(root/'new',self.current)
   before=(root/'new/X.csv.gz').read_bytes()
   fixed=prepare_snapshot(root/'new',self.days)
   self.assertEqual(prepare_snapshot(root/'new',self.days),fixed)
   self.assertEqual(prepare_snapshot(fixed,self.days),fixed)
   self.assertEqual((root/'new/X.csv.gz').read_bytes(),before)
   provenance=json.loads((fixed/'manifest.json').read_text())['gap_recovery'][0]
   self.assertEqual(provenance['source_snapshot'],'old')
   self.assertEqual(provenance['sessions'],[str(self.gap[0].date())])
 def test_no_source_fails_closed(self):
  with tempfile.TemporaryDirectory() as d:
   folder=Path(d)/'new';self.write(folder,self.current)
   with patch('secondary_prices.recover',side_effect=ValueError('provider unavailable')),self.assertRaisesRegex(RuntimeError,'Missing trading sessions'):prepare_snapshot(folder,self.days)
 def test_corrupted_source_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);self.write(root/'old',self.old.iloc[:-1]);self.write(root/'new',self.current)
   (root/'old/X.csv.gz').write_bytes(b'corrupted')
   with patch('secondary_prices.recover',side_effect=ValueError('provider unavailable')),self.assertRaisesRegex(RuntimeError,'Missing trading sessions'):prepare_snapshot(root/'new',self.days)

 def test_secondary_fallback_wired_with_provenance(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);self.write(root/'new',self.current)
   with patch('secondary_prices.recover',return_value=(self.old,{'provider':'secondary-test'})) as fetch:
    result=prepare_snapshot(root/'new',self.days)
   self.assertEqual(fetch.call_count,1)
   self.assertEqual(json.loads((result/'manifest.json').read_text())['gap_recovery'][0]['provider'],'secondary-test')
