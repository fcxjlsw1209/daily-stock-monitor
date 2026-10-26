"""Synthetic contracts: complete census, immutable source bindings, no admission."""
from copy import deepcopy
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import stock_comp_source_v1_runner as m


def fixtures():
    events = []
    for i in range(44):
        symbol = ("AAPL" if i < 10 else "AMD" if i < 20 else "MSFT" if i < 32 else "NVDA")
        event = dict(event_id="synthetic:%02d" % i, symbol=symbol, fiscal_year=2020,
                     release_date="2021-01-04", available_from="2021-01-05",
                     identity=dict(issuer_id=symbol), periods={
                         "current": dict(period_start="2020-01-01", period_end="2020-12-31", days_inclusive=366),
                         "comparison": dict(period_start="2019-01-01", period_end="2019-12-31", days_inclusive=365)},
                     source=dict(path="raw/%02d" % i, sha256="a"*64, url="https://example.test/annual",
                                 receipt=dict(path="raw/%02d.meta" % i, sha256="b"*64)),
                     inherited_rejection_codes=["prior_hold"] if i < 13 else [],
                     fields=dict(untouched_original="not parsed"), fact_admitted=False)
        events.append(event)
    raw = dict(events=events)
    inherited = deepcopy(raw)
    for i, event in enumerate(inherited["events"]):
        event["cash_quality_admission"] = dict(admitted=i >= 13,
                                             reasons=["prior_hold"] if i < 13 else [],
                                             definition="original")
    admission = dict(
        admitted_event_ids=[e["event_id"] for e in events[13:]],
        holds=[dict(event_id=e["event_id"], **e["cash_quality_admission"])
               for e in inherited["events"][:13]])
    return raw, inherited, admission, dict(sessions=["2021-01-04", "2021-01-05", "2021-01-06"])


class SourceContracts(unittest.TestCase):
    def test_full_census_holds_and_originals_preserved(self):
        f = fixtures(); original = deepcopy(f)
        specs = m.metadata_contract(*f)
        self.assertEqual(len(specs), 44)
        self.assertEqual(sum(not e["cash_quality_admission"]["admitted"] for e in specs), 13)
        self.assertEqual(f, original)

    def test_missing_or_duplicate_event_rejected(self):
        for duplicate in (False, True):
            f = fixtures()
            f[0]["events"][-1:] = [deepcopy(f[0]["events"][0])] if duplicate else []
            with self.assertRaises(ValueError): m.metadata_contract(*f)

    def test_changed_old_value_rejected(self):
        f = fixtures(); f[1]["events"][0]["fields"]["untouched_original"] = "changed"
        with self.assertRaisesRegex(ValueError, "raw event changed"): m.metadata_contract(*f)

    def test_changed_hold_reason_rejected(self):
        f = fixtures(); f[1]["events"][0]["cash_quality_admission"]["reasons"] = []
        with self.assertRaisesRegex(ValueError, "hold reasons"): m.metadata_contract(*f)

    def test_promoting_old_hold_rejected(self):
        f = fixtures(); f[1]["events"][0]["cash_quality_admission"]["admitted"] = True
        with self.assertRaisesRegex(ValueError, "hold changed"): m.metadata_contract(*f)

    def test_release_day_unavailable(self):
        f = fixtures()
        for collection in f[:2]: collection["events"][0]["available_from"] = "2021-01-04"
        with self.assertRaisesRegex(ValueError, "clock"): m.metadata_contract(*f)

    def test_period_must_precede_release(self):
        f = fixtures()
        for collection in f[:2]: collection["events"][0]["periods"]["current"]["period_end"] = "2021-01-04"
        with self.assertRaisesRegex(ValueError, "annual period"): m.metadata_contract(*f)

    def test_wrong_issuer_census_rejected(self):
        f = fixtures()
        for collection in f[:2]: collection["events"][0]["symbol"] = "NVDA"
        with self.assertRaisesRegex(ValueError, "issuer census"): m.metadata_contract(*f)

    def test_policy_disallows_calculations(self):
        p = m.policy()
        self.assertFalse(p["financial_calculations_permitted"])
        self.assertFalse(p["network_requests_permitted"])
        self.assertFalse(p["strategy_registration_permitted"])

    def test_hash_required_before_plan_io(self):
        with patch.object(m.base, "read") as read:
            with self.assertRaises(ValueError): m.load_plan(Path("/missing"), None)
            read.assert_not_called()

    def test_write_once_source_publication(self):
        with tempfile.TemporaryDirectory() as root:
            m.base.publish(root, "plan.json", dict(value=1))
            with self.assertRaises(FileExistsError): m.base.publish(root, "plan.json", dict(value=2))
            self.assertEqual(m.base.read(root, "plan.json"), dict(value=1))

    def test_tampered_file_fails_digest(self):
        with tempfile.TemporaryDirectory() as root:
            m.base.publish(root, "plan.json", dict(value=1))
            sha = m.base.sha(root, "plan.json")
            Path(root, "plan.json").write_text('{"value":2}')
            with self.assertRaisesRegex(ValueError, "SHA mismatch"): m.base.read(root, "plan.json", sha)

    def test_freeze_calls_only_metadata_inventory(self):
        specs = m.metadata_contract(*fixtures())
        with tempfile.TemporaryDirectory() as root:
            for e in specs:
                p = Path(root, e["source"]["path"]); p.parent.mkdir(exist_ok=True)
                p.write_bytes(b"synthetic")
                e["source"]["sha256"] = m.safe.digest(b"synthetic")
            with patch.object(m, "test_contract", return_value={}), \
                 patch.object(m, "source_inputs", return_value=(specs, dict(pdf_runtime={}, pdf_runtime_hashes={}), {})), \
                 patch.object(m.safe, "verify_hashes"), patch.object(m, "code_hashes", return_value={}), \
                 patch.object(m.parser, "inventory", return_value=dict(candidates=[], gaps=["synthetic"])) as inventory, \
                 patch.object(m.parser, "extract_current") as extract:
                result = m.freeze(root)
                self.assertEqual(result["financial_amounts_extracted"], 0)
                self.assertEqual(inventory.call_count, 44)
                extract.assert_not_called()

    def test_build_retains_missing_and_old_holds_without_admission(self):
        specs = m.metadata_contract(*fixtures())
        with tempfile.TemporaryDirectory() as root:
            for e in specs:
                p = Path(root, e["source"]["path"]); p.parent.mkdir(exist_ok=True)
                p.write_bytes(b"synthetic")
                e["source"]["sha256"] = m.safe.digest(b"synthetic")
                e["locator_inventory"] = dict(synthetic=True)
            def extract(raw, periods, inventory, plan):
                return dict(field=m.parser.FIELD, value=None, gaps=[dict(reason="missing")],
                            candidate_only=True, fact_admitted=False, model_admitted=False, strategy_admitted=False)
            with patch.object(m.parser, "extract_current", side_effect=extract):
                data, cov = m.products(root, dict(events=specs), "a"*64)
            self.assertEqual(data["counts"]["gap_cells"], 44)
            self.assertEqual(data["counts"]["inherited_holds"], 13)
            self.assertTrue(all(data["counts"][k] == 0 for k in m.ZERO_COUNTS))
            self.assertTrue(all(not e["fact_admitted"] for e in data["events"]))
            self.assertTrue(all(not e["new_field_admitted"] for e in cov["events"]))

    def test_verify_rejects_internally_inconsistent_audit(self):
        for changed in ("plan_sha256", "data_sha256", "coverage_sha256", "independent_field_and_visual_review_pending"):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as root:
                data, cov = dict(counts={"events":44}), dict(events=[])
                objects = {"RAW_FACTS.json":data, "COVERAGE.json":cov}
                hashes = {k:m.safe.digest(m.safe.encoded(v)) for k,v in objects.items()}
                audit = dict(passed=True, counts=data["counts"], plan_sha256="a"*64,
                             data_sha256=hashes["RAW_FACTS.json"], coverage_sha256=hashes["COVERAGE.json"],
                             independent_field_and_visual_review_pending=True,
                             all_original_events_preserved=True, all_inherited_holds_preserved=True)
                audit[changed] = False if changed.endswith("pending") else "b"*64
                objects["AUDIT.json"] = audit
                for name,obj in objects.items():
                    m.base.publish(root,m.OUT+"/"+name,obj)
                    hashes[name]=m.base.sha(root,m.OUT+"/"+name)
                m.base.publish(root,m.OUT+"/MANIFEST.json",dict(plan_sha256="a"*64,output_hashes=hashes))
                with patch.object(m,"load_plan",return_value={}), patch.object(m,"products",return_value=(data,cov)):
                    with self.assertRaisesRegex(ValueError,"Audit"):
                        m.verify(root,"a"*64)


if __name__ == "__main__":
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    log = stream.getvalue(); print(log, end="")
    if "--write-report" in sys.argv:
        m.require(not m.safe.local(m.ROOT, m.OUT + "/PLAN.json").exists(), "Plan freezes test evidence")
        path = m.safe.local(m.ROOT, m.OUT + "/TESTS.json")
        previous = [m.base.read(m.ROOT, m.OUT + "/TESTS.json")] if path.exists() else []
        report = dict(passed=result.wasSuccessful(), synthetic_only=True, tests_run=result.testsRun,
                      failures=len(result.failures), errors=len(result.errors), real_source_cells_extracted=0,
                      source_code_hashes=m.code_hashes(m.ROOT), log=log, previous_unfrozen_runs=previous)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(m.safe.encoded(report))
    raise SystemExit(0 if result.wasSuccessful() else 1)
