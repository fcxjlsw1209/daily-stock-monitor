import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
import assemble_input_diagnostics as assembly
from research_calendar import sessions

class InputAssemblyTests(unittest.TestCase):
    def test_known_after_removal_never_readds_member(self):
        calendar=sessions('2024-01-02','2024-01-12')
        row=dict(effective_session='2024-01-03',published_at='2024-01-08T10:00:00-05:00')
        begin=assembly.add_date(row,calendar)
        self.assertEqual(begin,'2024-01-09')
        end='2024-01-05'
        self.assertFalse(((calendar>=begin)&(calendar<end)).any())
    def test_holiday_delay_uses_exchange_sessions(self):
        row=dict(effective_session='2025-01-08',published_at='2025-01-08T12:00:00-05:00')
        self.assertEqual(assembly.add_date(row,sessions('2025-01-06','2025-01-15')),'2025-01-10')
    def test_source_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'source';f.write_bytes(b'original');h=hashlib.sha256(f.read_bytes()).hexdigest()
            assembly.checked(f,h);f.write_bytes(b'changed')
            with self.assertRaises(ValueError):assembly.checked(f,h)
    def test_later_sector_report_not_used_earlier(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);f=p/'source';f.write_bytes(b'primary');h=hashlib.sha256(f.read_bytes()).hexdigest()
            docs={'HISTORICAL_INDUSTRY_OBSERVATIONS.json':{'observations':[]},
              'SP_HISTORICAL_SECTOR_OBSERVATIONS.json':{'observations':[]},
              'BIOTECH_2009_SECTOR_OBSERVATIONS.json':{'observations':[]},
              'SPDR_ANNUAL_SECTOR_OBSERVATIONS.json':{'observations':[]},
            'RYDEX_2009_SECTOR_OBSERVATIONS.json':{'observations':[]},
              'HISTORICAL_SECTOR_TRANSITIONS.json':{'records':[]},
              'SPDR_SECTOR_CANDIDATES.json':{'observations':[dict(symbol='A',sector='Technology',effective_session='2011-09-30',known_at='2011-12-02T23:59:59-05:00',source_file='source',source_sha256=h)]}}
            for name,doc in docs.items():(p/name).write_text(json.dumps(doc))
            dates=pd.to_datetime(['2011-09-30','2011-12-02','2011-12-05'])
            with patch.object(assembly,'P',p):s,_,_=assembly.sector_candidates(dates,['A'])
            self.assertTrue(s['A'].iloc[:2].isna().all());self.assertEqual(s.at[dates[-1],'A'],'Technology')
if __name__=='__main__':unittest.main()
