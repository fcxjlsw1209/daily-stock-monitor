import unittest,tempfile,json,hashlib
from pathlib import Path
from run_historical_backtest import checked_file,run_attempt,PANELS,CERTIFICATES
from test_historical_backtest import inputs

class HistoricalCLITests(unittest.TestCase):
    def test_missing_bundle_produces_blocked_record_not_performance(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);r=run_attempt(p/'missing.json',p/'attempt')
            self.assertEqual(r['status'],'blocked')
            self.assertFalse(r['historical_backtest_complete'])
            self.assertTrue((p/'attempt/RUN_STATUS.json').exists())
            self.assertFalse((p/'attempt/results.json').exists())
    def test_attempt_cannot_overwrite_existing_results(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'attempt').mkdir()
            with self.assertRaises(FileExistsError):run_attempt(p/'missing.json',p/'attempt')
    def test_changed_input_hash_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'data').write_text('new')
            with self.assertRaises(ValueError):checked_file(p,dict(file='data',sha256='old'))
    def test_full_cli_with_synthetic_certified_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data=inputs();dates=data['close'].index
            def record(path):
                return dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            panels={}
            for key in PANELS:
                path=root/(key+'.csv');data[key].to_csv(path);panels[key]=record(path)
            source=root/'synthetic_source.txt';source.write_text('Synthetic test evidence only, not market data')
            proofs={}
            for kind in CERTIFICATES:
                path=root/(kind+'_certificate.json')
                path.write_text(json.dumps(dict(kind=kind,verified=True,**{'from':str(dates[300].date()),'through':str(dates[-1].date())},sources=[record(source)])))
                proofs[kind]=record(path)
            manifest=dict(universe_version='historical_ndx_sp100_union_v3_available_data',certificates=proofs,panels=panels,
                start=str(dates[300].date()),end=str(dates[-1].date()),issuers=data['issuers'],
                signal_basis=data['signal_basis'],execution_basis=data['execution_basis'])
            path=root/'manifest.json';path.write_text(json.dumps(manifest))
            result=run_attempt(path,root/'run')
            self.assertEqual(result['status'],'complete',result)
            saved=json.loads((root/'run/results.json').read_text())
            self.assertEqual(len(saved),6)
            self.assertTrue(all(r['performance_publishable'] for r in saved.values()))

    def test_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'nested').mkdir();(p/'secret').write_text('safe_test_data')
            with self.assertRaises(ValueError):checked_file(p/'nested',dict(file='../secret',sha256=hashlib.sha256(b'safe_test_data').hexdigest()))

if __name__=='__main__':unittest.main()
