"""Synthetic lifecycle/state tests; never read market arrays or run research phases."""
from pathlib import Path
from copy import deepcopy
import tempfile
import unittest
from unittest.mock import patch
import hashlib
import monthly_session_cluster148_lifecycle as m


def state():
    rows = [dict(directory="old"+str(i), trials=6) for i in range(145)]
    rows.append(dict(directory="monthly_cash_quality146", trials=69))
    rows.append(dict(directory=m.PREVIOUS, trials=4,
                     completion=dict(sha256=m.PREVIOUS_DONE)))
    return dict(completed=rows, registered_queue=[])


def source_ref(path):
    return dict(path=str(path), sha256=m.r.sha(path))


def test_manifest(root):
    suites = []
    for name, relative in m.SUITE_SOURCES.items():
        source = root/relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("# synthetic test source\n")
        log = root/(name+".log")
        log.write_text("Ran 2 tests in 0.01s\n\nOK\n")
        suites.append(dict(name=name, count=2, source=source_ref(source), log=source_ref(log)))
    return dict(passed=True, synthetic_only=True, test_count=10, suites=suites)


class LifecycleTests(unittest.TestCase):
    def test_snapshot_hash_and_content_use_one_read(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/"state.json"
            m.write(p, state())
            original = p.read_bytes()
            with patch.object(Path, "read_bytes", return_value=original) as mock:
                content, digest = m.state_snapshot(p)
            self.assertEqual(mock.call_count, 1)
            self.assertEqual(content, state())
            self.assertEqual(digest, hashlib.sha256(original).hexdigest())

    def test_second_lifecycle_writer_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"outputs/monthly_target70_research").mkdir(parents=True)
            with m.state_lock(root):
                with self.assertRaises(BlockingIOError):
                    with m.state_lock(root):
                        pass
            with m.state_lock(root):
                pass

    def test_completion_survives_failed_state_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/"COMPLETION.json"
            done = dict(completed_utc="original", primary="baseline", result=.6)
            m.preserve_completion(p, done)
            original = p.read_bytes()
            retry = dict(done, completed_utc="retry")
            self.assertEqual(m.preserve_completion(p, retry), done)
            self.assertEqual(p.read_bytes(), original)
            with self.assertRaises(ValueError):
                m.preserve_completion(p, dict(retry, result=.7))

    def test_predecessor_is_exact(self):
        m.predecessor(state())
        for mutate in (
            lambda x: x["completed"].pop(),
            lambda x: x["completed"][-1].update(trials=5),
            lambda x: x["completed"][-1].update(directory="wrong"),
            lambda x: x["completed"][0].update(directory=m.STUDY),
            lambda x: x["completed"][-1]["completion"].update(sha256="bad"),
        ):
            x = state()
            mutate(x)
            with self.assertRaises(ValueError):
                m.predecessor(x)

    def test_writer_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/"x.json"
            m.write(p, dict(a=1))
            with self.assertRaises(FileExistsError):
                m.write(p, dict(a=2))
            self.assertEqual(m.r.read(p), dict(a=1))

    def test_state_update_keeps_exact_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/"state.json"
            m.write(p, state())
            prior, h = p.read_bytes(), m.r.sha(p)
            newer = state()
            newer["registered_queue"] = ["frozen"]
            m.update_state(p, h, newer)
            self.assertEqual(m.r.read(p), newer)
            self.assertEqual((p.parent/("PRE148_"+h+".json")).read_bytes(), prior)

    def test_concurrent_state_change_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/"state.json"
            m.write(p, state())
            h = m.r.sha(p)
            p.write_bytes(b'{"changed": true}')
            with self.assertRaises(ValueError):
                m.update_state(p, h, {})
            self.assertEqual(p.read_bytes(), b'{"changed": true}')

    def test_existing_wrong_backup_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/"state.json"
            m.write(p, state())
            h = m.r.sha(p)
            (p.parent/("PRE148_"+h+".json")).write_bytes(b'{}')
            with self.assertRaises(ValueError):
                m.update_state(p, h, {})
            self.assertEqual(m.r.read(p), state())

    def test_commit_time_race_preserves_other_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/"state.json"
            m.write(p, state())
            h, real_sha = m.r.sha(p), m.r.sha
            calls = 0
            def racing_sha(path):
                nonlocal calls
                if Path(path) == p:
                    calls += 1
                    if calls == 2:
                        p.write_bytes(b'{"concurrent":true}')
                return real_sha(path)
            with patch.object(m.r, "sha", side_effect=racing_sha):
                with self.assertRaises(ValueError):
                    m.update_state(p, h, {})
            self.assertEqual(m.r.read(p), dict(concurrent=True))
            self.assertEqual(len(list(p.parent.iterdir())), 2)

    def test_test_manifest_requires_every_suite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            doc = test_manifest(root)
            m.validate_tests(doc, m.old.Reader(root))
            for key, value in (("test_count", 11), ("suites", doc["suites"][:-1]),
                               ("passed", False), ("synthetic_only", False)):
                bad = deepcopy(doc)
                bad[key] = value
                with self.assertRaises(ValueError):
                    m.validate_tests(bad, m.old.Reader(root))

    def test_test_manifest_rejects_substitute_source_or_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            doc = test_manifest(root)
            substitute = root/"other.py"
            substitute.write_text("pass\n")
            for change in (
                lambda x: x["suites"][0].update(source=source_ref(substitute)),
                lambda x: x["suites"].append(deepcopy(x["suites"][0])),
                lambda x: x["suites"][0].update(count=True),
            ):
                bad = deepcopy(doc)
                change(bad)
                with self.assertRaises(ValueError):
                    m.validate_tests(bad, m.old.Reader(root))

    def test_test_manifest_rejects_failed_modified_or_wrong_count_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            doc = test_manifest(root)
            log = Path(doc["suites"][0]["log"]["path"])
            for value in ("Ran 2 tests in 0.01s\nFAILED\n", "Ran 3 tests in 0.01s\nOK\n"):
                log.write_text(value)
                doc["suites"][0]["log"] = source_ref(log)
                with self.assertRaises(ValueError):
                    m.validate_tests(doc, m.old.Reader(root))
            log.write_text("Ran 2 tests in 0.01s\nOK\n")
            with self.assertRaises(ValueError):
                m.validate_tests(doc, m.old.Reader(root))


class BuildPlanTests(unittest.TestCase):
    """All bindings and documents below are temporary synthetic byte fixtures."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.patches = []
        def pin(obj, key, value):
            p = patch.object(obj, key, value)
            p.start()
            self.addCleanup(p.stop)
        def put(relative, value):
            path = self.root/relative
            path.parent.mkdir(parents=True, exist_ok=True)
            m.write(path, value)
            return source_ref(path)
        self.put = put
        self.tests = self.root/"TESTS.json"
        self.put("TESTS.json", test_manifest(self.root))
        raw = {}
        for i in range(64):
            ref = put("raw/"+str(i)+".json", dict(synthetic=True, i=i))
            raw[ref["path"]] = ref["sha256"]
        dependency = put("calendar_dependency.json", dict(synthetic=True))
        calendar = put(m.r.CALENDAR, dict(start="2015-01-01", end="2027-12-31",
            bindings={dependency["path"]: dependency["sha256"]}))
        pin(m.r, "CALENDAR_SHA", calendar["sha256"])
        sessions = put(m.r.SESSIONS, dict(plan_sha256=calendar["sha256"], sessions=[]))
        pin(m.r, "SESSIONS_SHA", sessions["sha256"])
        prices = put(m.r.PRICE, dict(synthetic_prices_not_an_array=True))
        pin(m.r, "PRICE_SHA", prices["sha256"])
        concept = put(m.r.CONCEPT, dict(prospective=True))
        pin(m.r, "CONCEPT_SHA", concept["sha256"])
        novelty = put("novelty.json", dict(bounded=True))
        bounded = put("bounded.json", dict(exhaustive=False))
        inputs = {key: put("prior/"+key+".json", dict(synthetic=True)) for key in m.r.INPUTS}
        inputs.update(prices=prices, sessions=sessions)
        self.prior = dict(inputs=inputs, raw_source_hashes=raw, hashes=dict(raw))
        prior_plan = put("outputs/"+m.PREVIOUS+"/PLAN.json", self.prior)
        pin(m, "PREVIOUS_PLAN", prior_plan["sha256"])
        outcome = put("outputs/"+m.PREVIOUS+"/INDEPENDENT_OUTCOME.json", dict(passed=True))
        protection = put("outputs/"+m.PREVIOUS+"/PROTECTED_AFTER.json", dict(passed=True))
        done = put("outputs/"+m.PREVIOUS+"/COMPLETION.json", dict(round=147,
            directory=m.PREVIOUS, plan_sha256=prior_plan["sha256"], trials=4,
            independent_outcome_sha256=outcome["sha256"], protected_sha256=protection["sha256"]))
        pin(m, "PREVIOUS_DONE", done["sha256"])
        self.proposal = dict(round=148, arms=list(m.r.NAMES), primary_challenger=m.r.NAMES[2],
            dose=m.r.DOSE, concept=dict(path=m.r.CONCEPT, sha256=concept["sha256"]),
            novelty=novelty, bounded_archive_novelty=bounded,
            feature_contract=m.r.feature_contract(), selection_contract=m.r.selection_contract(),
            sources=dict(prices=dict(path=m.r.PRICE, sha256=prices["sha256"]),
                sessions=dict(path=m.r.SESSIONS, sha256=sessions["sha256"]),
                calendar=dict(path=m.r.CALENDAR, sha256=calendar["sha256"]),
                predecessor=dict(path="outputs/"+m.PREVIOUS+"/COMPLETION.json", sha256=done["sha256"])))
        proposal = put(m.PROPOSAL, self.proposal)
        pin(m, "PROPOSAL_SHA", proposal["sha256"])
        runner = put("work/monthly_session_cluster148_runner.py", dict(synthetic_code=True))
        self.runner_sources = {runner["path"]: runner["sha256"]}
        pin(m.r, "source_files", lambda root: self.runner_sources)
        self.independent = {}
        for name in m.REVIEW_MODULES:
            ref = put("work/"+name+"_auditor.py", dict(synthetic_code=name))
            self.independent[name] = {ref["path"]: ref["sha256"]}
            put(m.PREREVIEWS[name], dict(passed=True, actual_factors_calculated=0,
                actual_returns_calculated=0, source_hashes=self.independent[name],
                reviewed_contract_hashes={**self.runner_sources, proposal["path"]: proposal["sha256"]}))
        pin(m, "independent_sources", lambda root: self.independent)
        self.put(m.STATE, state())

    def replace(self, relative, value):
        path = self.root/relative
        path.write_bytes(m.old.encoded(value))

    def test_build_has_three_arms_seven_inputs_and_all_pins_without_writing(self):
        plan = m.build_plan(self.root, self.tests)
        self.assertEqual(plan["round"], 148)
        self.assertEqual(plan["arms"], list(m.r.NAMES))
        self.assertEqual(plan["primary_challenger"], "low_event_clustering")
        self.assertEqual(set(plan["inputs"]), set(m.r.INPUTS))
        self.assertEqual(len(plan["inputs"]), 7)
        self.assertEqual(len(plan["raw_source_hashes"]), 64)
        self.assertEqual(self.prior["inputs"]["prices"]["path"], plan["inputs"]["prices"]["path"])
        self.assertNotIn("source_admission", plan["inputs"])
        self.assertNotIn("source_policy", plan)
        for relative in (m.PROPOSAL, m.r.CONCEPT, m.r.CALENDAR, m.r.PRICE, m.r.SESSIONS,
                         "calendar_dependency.json", "novelty.json", "bounded.json"):
            self.assertEqual(plan["hashes"][str(self.root/relative)], m.r.sha(self.root/relative))
        self.assertNotIn(str(self.root/m.STATE), plan["hashes"])
        self.assertEqual(plan["registration_precondition"]["sha256"], m.r.sha(self.root/m.STATE))
        self.assertEqual(plan["independent_prepare_source_hashes"], self.independent["prepare"])
        self.assertEqual(plan["independent_outcome_source_hashes"], self.independent["outcome"])
        self.assertFalse((self.root/"outputs"/m.STUDY/"PLAN.json").exists())
        self.assertEqual(plan["actual_feature_calculations_before_registration"], 0)
        self.assertEqual(plan["actual_returns_before_registration"], 0)

    def test_prior_closure_mutation_is_rejected(self):
        raw = next(iter(self.prior["raw_source_hashes"]))
        Path(raw).write_text("changed")
        with self.assertRaisesRegex(ValueError, "Hash mismatch"):
            m.build_plan(self.root, self.tests)

    def test_prior_completion_or_plan_mutation_is_rejected(self):
        self.replace("outputs/"+m.PREVIOUS+"/COMPLETION.json", {})
        with self.assertRaisesRegex(ValueError, "Predecessor receipt"):
            m.build_plan(self.root, self.tests)

    def test_prior_final_evidence_mutation_is_rejected(self):
        self.replace("outputs/"+m.PREVIOUS+"/PROTECTED_AFTER.json", {})
        with self.assertRaisesRegex(ValueError, "Predecessor final evidence"):
            m.build_plan(self.root, self.tests)

    def test_existing_partial_phase_output_blocks_registration(self):
        self.put("outputs/"+m.STUDY+"/DATA.npz", dict(partial=True))
        with self.assertRaisesRegex(ValueError, "precede actual outputs"):
            m.build_plan(self.root, self.tests)

    def test_pending_experiment_blocks_registration(self):
        s = state()
        s["registered_queue"] = [dict(directory="other")]
        self.replace(m.STATE, s)
        with self.assertRaisesRegex(ValueError, "Pending experiment"):
            m.build_plan(self.root, self.tests)

    def test_prereview_missing_current_contract_or_changed_source_is_rejected(self):
        path = m.PREREVIEWS["prepare"]
        review = m.r.read(self.root/path)
        review["reviewed_contract_hashes"].pop(str(self.root/m.PROPOSAL))
        self.replace(path, review)
        with self.assertRaisesRegex(ValueError, "omitted current contract"):
            m.build_plan(self.root, self.tests)

    def test_prereview_extra_unregistered_source_is_rejected(self):
        path = m.PREREVIEWS["prepare"]
        review = m.r.read(self.root/path)
        ref = self.put("unrelated_source.json", {})
        review["source_hashes"][ref["path"]] = ref["sha256"]
        self.replace(path, review)
        with self.assertRaisesRegex(ValueError, "source closure differs"):
            m.build_plan(self.root, self.tests)

    def test_prereview_actual_calculation_is_rejected(self):
        path = m.PREREVIEWS["outcome"]
        review = m.r.read(self.root/path)
        review["actual_returns_calculated"] = 1
        self.replace(path, review)
        with self.assertRaisesRegex(ValueError, "Prospective independent review"):
            m.build_plan(self.root, self.tests)

    def test_contract_changes_and_extra_financial_source_are_rejected(self):
        for mutate in (
            lambda p: p["feature_contract"].update(event_sessions=209),
            lambda p: p["selection_contract"].update(training_months=89),
            lambda p: p["sources"].update(admitted_dataset=dict(path="unused", sha256="0"*64)),
            lambda p: p["sources"]["prices"].update(path="other.npz"),
        ):
            p = deepcopy(self.proposal)
            mutate(p)
            with self.assertRaises(ValueError):
                m.hypothesis_and_inputs(m.old.Reader(self.root), p, self.prior)

    def test_named_inputs_do_not_mutate_predecessor(self):
        prior = deepcopy(self.prior)
        original = deepcopy(prior)
        m.hypothesis_and_inputs(m.old.Reader(self.root), self.proposal, prior)
        self.assertEqual(prior, original)

    def test_register_is_write_once_and_enqueue_recovery_is_idempotent(self):
        result = m.register(self.root, self.tests)
        plan_path = self.root/"outputs"/m.STUDY/"PLAN.json"
        plan_bytes = plan_path.read_bytes()
        state_bytes = (self.root/m.STATE).read_bytes()
        m.enqueue(self.root, result["plan_sha256"])
        self.assertEqual(plan_path.read_bytes(), plan_bytes)
        self.assertEqual((self.root/m.STATE).read_bytes(), state_bytes)
        with self.assertRaises(ValueError):
            m.register(self.root, self.tests)
        queued = m.r.read(self.root/m.STATE)["registered_queue"]
        self.assertEqual(queued[0]["round"], 148)
        self.assertEqual(queued[0]["directory"], m.STUDY)

    def test_enqueue_detects_changed_precondition(self):
        plan = m.build_plan(self.root, self.tests)
        self.put("outputs/"+m.STUDY+"/PLAN.json", plan)
        s = state()
        s["unrelated_new_evidence"] = True
        self.replace(m.STATE, s)
        with self.assertRaisesRegex(ValueError, "Concurrent continuation update"):
            m.enqueue(self.root, m.r.sha(self.root/"outputs"/m.STUDY/"PLAN.json"))
        self.assertTrue(m.r.read(self.root/m.STATE)["unrelated_new_evidence"])

    def test_enqueue_rejects_changed_plan_hash(self):
        self.put("outputs/"+m.STUDY+"/PLAN.json", {})
        with self.assertRaisesRegex(ValueError, "PLAN changed"):
            m.enqueue(self.root, "0"*64)



class CompletionTests(unittest.TestCase):
    """Mock only strategy execution/protection scan; exercise real receipt/state I/O."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.out = self.root/"outputs"/m.STUDY
        self.out.mkdir(parents=True)
        (self.root/m.STATE).parent.mkdir(parents=True)
        def put(name, value):
            m.write(self.out/name, value)
        self.put = put
        self.selection = dict(primary=m.r.NAMES[0], admitted=False,
            guards_against={m.r.NAMES[0]: False, m.r.NAMES[1]: False},
            next_open_guards_against={m.r.NAMES[0]: False, m.r.NAMES[1]: False},
            mechanism_confirmed=False)
        self.metric = dict(cagr=.6, sharpe=1.5, max_drawdown=-.3)
        results = {name: {mode: dict(metrics=dict(self.metric)) for mode in
                   ("same_close", "next_open_zero", "next_open_cost")} for name in m.r.NAMES}
        # An unselected diagnostic value must never become the completion result.
        results[m.r.NAMES[2]]["same_close"]["metrics"]["cagr"] = .8
        self.manifest = {str(self.root/("member_"+str(i))): "0"*64 for i in range(21444)}
        manifest_path = self.root/"manifest.json"
        m.write(manifest_path, self.manifest)
        source = self.root/"auditor.py"
        source.write_text("# synthetic\n")
        self.plan = dict(inputs=dict(protected_manifest=source_ref(manifest_path)),
            independent_outcome_source_hashes={str(source): m.r.sha(source)})
        put("PLAN.json", self.plan)
        self.plan_sha = m.r.sha(self.out/"PLAN.json")
        values = {name: {} for name in (*m.r.CACHE_FILES, "CACHE.json", "INDEPENDENT_PREPARE.json",
            "TRAIN_TRIALS.json", "TRAIN_OPEN_TRIALS.json", "WALKFORWARD.json", "VALIDATION.json",
            "DIAGNOSE_STARTED.json")}
        values.update({"RESULTS.json": results, "FROZEN.json": dict(selection=self.selection)})
        for arm in m.r.NAMES:
            for mode in ("same_close", "next_open_zero", "next_open_cost"):
                for kind in ("DAILY", "TRADES", "HOLDINGS"):
                    values[arm+"_"+mode+"_"+kind+".json"] = []
        for name, value in values.items():
            put(name, value)
        self.review = dict(passed=True, stage="evaluate", plan_sha256=self.plan_sha,
            execution_paths_verified=9, total_monthly_orders_verified=366,
            selected_name=m.r.NAMES[0], primary_cagr=.6, primary_target_reached=False,
            source_hashes=self.plan["independent_outcome_source_hashes"],
            artifact_hashes={str(p): m.r.sha(p) for p in self.out.iterdir()})
        put("INDEPENDENT_OUTCOME.json", self.review)
        s = state()
        s["registered_queue"] = [dict(directory=m.STUDY, plan=dict(sha256=self.plan_sha))]
        s["best_training_selected_candidate"] = dict(cagr=.67)
        m.write(self.root/m.STATE, s)
        self.fake_runner = patch.object(m.r, "Runner").start()
        self.addCleanup(patch.stopall)
        patch.object(m.r, "select", return_value=deepcopy(self.selection)).start()
        self.scan = patch.object(m.old, "protected_scan", return_value=21444).start()
        real_reader = m.old.Reader
        manifest = self.manifest
        class MetadataReader(real_reader):
            def verify(self, mapping, base=None):
                if mapping == manifest:
                    return dict(mapping)
                return super().verify(mapping, base)
        patch.object(m.old, "Reader", MetadataReader).start()

    def replace_review(self):
        (self.out/"INDEPENDENT_OUTCOME.json").write_bytes(m.old.encoded(self.review))

    def test_completion_increments_exactly_three_trials_and_uses_selected_arm(self):
        result = m.complete(self.root, self.plan_sha)
        s = m.r.read(self.root/m.STATE)
        done = m.r.read(self.out/"COMPLETION.json")
        self.assertEqual(len(s["completed"]), 148)
        self.assertEqual(sum(row["trials"] for row in s["completed"]), 946)
        self.assertEqual(s["registered_queue"], [])
        self.assertEqual(done["trials"], 3)
        self.assertEqual(done["independent_execution_paths"], 9)
        self.assertEqual(result["cagr"], .6)
        self.assertFalse(result["target_reached"])
        self.assertEqual(s["best_training_selected_candidate"], dict(cagr=.67))
        self.fake_runner.return_value.require_prepare_review.assert_called_once()
        self.scan.assert_called_once()

    def test_old_four_arm_counts_cannot_close_round148(self):
        for key, value in (("execution_paths_verified", 12), ("total_monthly_orders_verified", 488)):
            saved = self.review[key]
            self.review[key] = value
            self.replace_review()
            with self.assertRaisesRegex(ValueError, "Full independent outcome"):
                m.complete(self.root, self.plan_sha)
            self.review[key] = saved
        self.assertFalse((self.out/"COMPLETION.json").exists())

    def test_unreviewed_cache_or_execution_artifact_cannot_close(self):
        for name in ("DATA.npz", "INDEPENDENT_PREPARE.json", "DIAGNOSE_STARTED.json",
                     m.r.NAMES[2]+"_next_open_cost_HOLDINGS.json"):
            path = str(self.out/name)
            saved = self.review["artifact_hashes"].pop(path)
            self.replace_review()
            with self.assertRaisesRegex(ValueError, "Unreviewed outcome"):
                m.complete(self.root, self.plan_sha)
            self.review["artifact_hashes"][path] = saved
        self.assertFalse((self.out/"COMPLETION.json").exists())

    def test_review_cannot_switch_selection_to_diagnostic_winner(self):
        self.review["selected_name"] = m.r.NAMES[2]
        self.review["primary_cagr"] = .8
        self.review["primary_target_reached"] = True
        self.replace_review()
        with self.assertRaisesRegex(ValueError, "Selection mismatch"):
            m.complete(self.root, self.plan_sha)
        self.assertFalse((self.out/"COMPLETION.json").exists())

    def test_target_flag_must_match_selected_metric(self):
        self.review["primary_target_reached"] = True
        self.replace_review()
        with self.assertRaisesRegex(ValueError, "Target verification"):
            m.complete(self.root, self.plan_sha)
        self.assertFalse((self.out/"COMPLETION.json").exists())

    def test_queued_other_experiment_cannot_close(self):
        s = m.r.read(self.root/m.STATE)
        s["registered_queue"][0]["directory"] = "other"
        (self.root/m.STATE).write_bytes(m.old.encoded(s))
        with self.assertRaisesRegex(ValueError, "Registration queue mismatch"):
            m.complete(self.root, self.plan_sha)
        self.assertFalse((self.out/"COMPLETION.json").exists())


if __name__ == "__main__":
    unittest.main()
