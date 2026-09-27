import unittest,tempfile,json
from pathlib import Path
from unittest.mock import patch
import run_pair

class PairTests(unittest.TestCase):
    def test_failure_replaces_old_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);(p/'COMPARISON.md').write_text('stale buy recommendations')
            with patch.object(run_pair,'P',p),patch('sys.argv',['run_pair']),patch.object(run_pair,'invoke',side_effect=RuntimeError('bad fingerprint')):
                with self.assertRaises(SystemExit):run_pair.main()
            self.assertEqual(json.loads((p/'comparison.json').read_text())['status'],'ERROR_NO_NEW_RECOMMENDATIONS')
            self.assertNotIn('stale buy recommendations',(p/'COMPARISON.md').read_text())

    def test_same_snapshot_forwarded(self):
        baseline={'snapshot_id':'immutable-test'};shadow={'snapshot_id':'immutable-test'}
        with tempfile.TemporaryDirectory() as folder,patch.object(run_pair,'P',Path(folder)),patch('sys.argv',['run_pair','--phase','morning']),patch.object(run_pair,'invoke',side_effect=[{}, {}, {}, baseline,shadow,shadow]) as call,patch.object(run_pair,'publish',return_value={'status':'ok'}):
            run_pair.main()
            self.assertEqual(call.call_count,6)
            self.assertEqual(call.call_args_list[-1].args[1],['--phase','morning','--snapshot',str(run_pair.ROOT/'work/stock_monitor_cache/immutable-test')])

    def test_mismatched_snapshot_rejected(self):
        with self.assertRaises(RuntimeError):run_pair.publish({'baseline':{'snapshot_id':'a'},'shadow':{'snapshot_id':'b'}})

if __name__=='__main__':unittest.main()
