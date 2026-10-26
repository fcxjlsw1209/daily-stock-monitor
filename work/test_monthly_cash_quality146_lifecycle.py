"""Synthetic metadata/state tests; no original market files are read."""
from pathlib import Path
from copy import deepcopy
import tempfile
import unittest
from unittest.mock import patch
import hashlib
import monthly_cash_quality146_lifecycle as m


def state():
    rows=[dict(directory="old"+str(i),trials=6) for i in range(144)]
    rows.append(dict(directory="monthly_same_report145",trials=71,
                     completion=dict(sha256=m.PREVIOUS_DONE)))
    return dict(completed=rows,registered_queue=[])


class LifecycleTests(unittest.TestCase):
    def test_snapshot_hash_and_content_use_one_read(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/"state.json";m.write(p,state())
            original=p.read_bytes()
            with patch.object(Path,"read_bytes",return_value=original) as mock:
                content,digest=m.state_snapshot(p)
            self.assertEqual(mock.call_count,1)
            self.assertEqual(content,state())
            self.assertEqual(digest,hashlib.sha256(original).hexdigest())

    def test_second_lifecycle_writer_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/"outputs/monthly_target70_research").mkdir(parents=True)
            with m.state_lock(root):
                with self.assertRaises(BlockingIOError):
                    with m.state_lock(root):pass
            with m.state_lock(root):pass

    def test_completion_survives_failed_state_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/"COMPLETION.json"
            done=dict(completed_utc="original",primary="baseline",result=.6)
            m.preserve_completion(p,done);original=p.read_bytes()
            # State commit failed after exclusive receipt write; a reverified
            # retry must preserve the original receipt and its actual timestamp.
            retry=dict(done,completed_utc="retry")
            self.assertEqual(m.preserve_completion(p,retry),done)
            self.assertEqual(p.read_bytes(),original)
            with self.assertRaises(ValueError):
                m.preserve_completion(p,dict(retry,result=.7))

    def test_predecessor_is_exact(self):
        m.predecessor(state())
        for mutate in (
            lambda x:x["completed"].pop(),
            lambda x:x["completed"][-1].update(trials=72),
            lambda x:x["completed"][-1].update(directory="wrong"),
            lambda x:x["completed"][0].update(directory=m.STUDY),
            lambda x:x["completed"][-1]["completion"].update(sha256="bad")):
            x=state();mutate(x)
            with self.assertRaises(ValueError):m.predecessor(x)

    def test_writer_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/"x.json"
            m.write(p,dict(a=1))
            with self.assertRaises(FileExistsError):m.write(p,dict(a=2))
            self.assertEqual(m.r.read(p),dict(a=1))

    def test_state_update_keeps_exact_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/"state.json";m.write(p,state())
            prior=p.read_bytes();h=m.r.sha(p)
            newer=state();newer["registered_queue"]=["frozen"]
            m.update_state(p,h,newer)
            self.assertEqual(m.r.read(p),newer)
            self.assertEqual((p.parent/("PRE146_"+h+".json")).read_bytes(),prior)

    def test_concurrent_state_change_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/"state.json";m.write(p,state())
            h=m.r.sha(p);p.write_bytes(b'{"changed": true}')
            with self.assertRaises(ValueError):m.update_state(p,h,{})
            self.assertEqual(p.read_bytes(),b'{"changed": true}')

    def test_test_manifest_requires_every_suite(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);suites=[]
            for name in ("factor","runner","prepare_review","outcome_review","lifecycle"):
                source=root/(name+".py");source.write_text("pass\n")
                log=root/(name+".log");log.write_text("Ran 2 tests in 0.01s\n\nOK\n")
                suites.append(dict(name=name,count=2,
                    source=dict(path=str(source),sha256=m.r.sha(source)),
                    log=dict(path=str(log),sha256=m.r.sha(log))))
            doc=dict(passed=True,synthetic_only=True,test_count=10,suites=suites)
            m.validate_tests(doc,m.old.Reader(root))
            for key,value in (("test_count",11),("suites",suites[:-1]),("passed",False)):
                bad=deepcopy(doc);bad[key]=value
                with self.assertRaises(ValueError):m.validate_tests(bad,m.old.Reader(root))

    def test_test_manifest_rejects_failed_or_modified_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);p=root/"x";p.write_text("Ran 2 tests in 0.01s\nFAILED\n")
            ref=dict(path=str(p),sha256=m.r.sha(p))
            doc=dict(passed=True,synthetic_only=True,test_count=2,
                     suites=[dict(name="factor",count=2,source=ref,log=ref)])
            with self.assertRaises(ValueError):m.validate_tests(doc,m.old.Reader(root))
            p.write_text("Ran 2 tests in 0.01s\nOK\n")
            with self.assertRaises(ValueError):m.validate_tests(doc,m.old.Reader(root))


if __name__=="__main__":
    unittest.main()
