"""Synthetic migration/strict-input checks; no market or account files read."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd

SOURCE=Path(__file__).with_name('monthly_local67_sept30_singlepoint.py')
spec=importlib.util.spec_from_file_location('singlepoint_under_test',SOURCE)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class OverlayTests(unittest.TestCase):
    def setUp(self):
        m.SOURCES.clear()
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        (self.root/'prices').mkdir()
        self.index=pd.to_datetime(['2026-08-31','2026-09-24','2026-09-30','2026-10-01'])
        self.panel={k:pd.DataFrame({'ARKK':[10.,11.,np.nan,np.nan]},index=self.index) for k in m.FIELDS}
        self.incoming=pd.DataFrame({v:[10.,11.,12.,13.] for v in m.FIELDS.values()},index=self.index)
        self.incoming.index.name='Date'

    def tearDown(self):self.tmp.cleanup()

    def cache(self, frame=None):
        p=self.root/'prices/ARKK.csv';(self.incoming if frame is None else frame).to_csv(p)
        (self.root/'prices/ARKK.source.json').write_text(json.dumps({'symbol':'ARKK','sha256':m.sha(p)}))

    def test_tail_only_and_original_panel_unchanged(self):
        self.cache();result,audit=m.merge_overlays(self.panel,self.root,['ARKK'],m.ASOF)
        for field in m.FIELDS:
            pd.testing.assert_series_equal(result[field].ARKK.iloc[:2],self.panel[field].ARKK.iloc[:2])
            self.assertTrue(self.panel[field].ARKK.iloc[2:].isna().all())
            self.assertEqual(result[field].ARKK.iloc[2:].tolist(),[12.,13.])
        self.assertEqual(audit[0]['added_rows'],2)

    def test_small_vendor_adjusted_precision_never_replaces_history(self):
        frame=self.incoming.copy()
        frame.loc[self.index[0],'Adj Close']+=4*float(np.spacing(np.float32(10.)))
        self.cache(frame);result,_=m.merge_overlays(self.panel,self.root,['ARKK'],m.ASOF)
        self.assertEqual(result['adj'].ARKK.iloc[0],10.)

    def test_adjusted_revision_outside_precision_fails(self):
        frame=self.incoming.copy();frame.loc[self.index[0],'Adj Close']+=.01
        self.cache(frame)
        with self.assertRaisesRegex(ValueError,'adjusted history conflicts'):
            m.merge_overlays(self.panel,self.root,['ARKK'],m.ASOF)

    def test_raw_history_revision_fails(self):
        frame=self.incoming.copy();frame.loc[self.index[0],'Close']+=.01
        self.cache(frame)
        with self.assertRaisesRegex(ValueError,'raw close history differs'):
            m.merge_overlays(self.panel,self.root,['ARKK'],m.ASOF)

    def test_expected_sha_is_required(self):
        self.cache();(self.root/'prices/ARKK.csv').write_text('changed')
        with self.assertRaisesRegex(ValueError,'Input SHA mismatch'):
            m.merge_overlays(self.panel,self.root,['ARKK'],m.ASOF)

    def test_missing_sidecar_fails_explicitly(self):
        self.cache();(self.root/'prices/ARKK.source.json').unlink()
        with self.assertRaisesRegex(ValueError,'ARKK.source.json'):
            m.merge_overlays(self.panel,self.root,['ARKK'],m.ASOF)

    def test_existing_complete_frozen_data_needs_no_extra_cache(self):
        for field in m.FIELDS:self.panel[field].loc[m.ASOF,'ARKK']=12.
        result,audit=m.merge_overlays(self.panel,self.root,['ARKK'],m.ASOF)
        self.assertTrue(audit[0]['already_complete_in_original64_sources'])
        self.assertEqual(result['adj'].at[m.ASOF,'ARKK'],12.)

    def test_missing_asof_not_forward_filled(self):
        frame=self.incoming.copy();frame.loc[m.ASOF,'Close']=np.nan
        self.cache(frame)
        with self.assertRaisesRegex(ValueError,'Missing close at September30'):
            m.merge_overlays(self.panel,self.root,['ARKK'],m.ASOF)

    def test_duplicate_pool_is_rejected(self):
        (self.root/'POOL.json').write_text(json.dumps({'date':'2026-09-30','top32':[{'symbol':'ARKK'}]*32}))
        with self.assertRaisesRegex(ValueError,'Explicit uniqueTop32'):
            m.load_pool(self.root,['SPY'])

    def test_old_pool_is_rejected(self):
        (self.root/'POOL.json').write_text(json.dumps({'date':'2026-08-31','top32':[]}))
        with self.assertRaisesRegex(ValueError,'Wrong pool date'):
            m.load_pool(self.root,['SPY'])

if __name__=='__main__':unittest.main()
