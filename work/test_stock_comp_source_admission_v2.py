"""Generated source contracts only; no actual financial observations."""
from copy import deepcopy
import io
import hashlib
import tempfile
from pathlib import Path
import unittest
import stock_comp_source_admission_v2 as gate


def fixture():
    source = dict(path="synthetic.html", sha256="a" * 64)
    period = dict(period_start="2024-01-01", period_end="2024-12-31", days_inclusive=366)
    loc = dict(format="html", field_scope="explicit_component_of_consolidated_indirect_operating_reconciliation",
               period_end="2024-12-31", scale="1000000", currency_literal="$",
               statement_title="consolidated statements of cash flows",
               table_index_zero_based=0, logical_columns=[1], statement_segment=0)
    old = dict(event_id="SYNTH:2024", symbol="SYNTH", fiscal_year=2024,
               release_date="2025-01-01", available_from="2025-01-02",
               identity={"security_id": "SYNTH"}, periods={"current": period},
               source=source, inherited_rejection_codes=[],
               cash_quality_admission=dict(admitted=True, reasons=[], definition="old"),
               fields={"current": {"operating_cash_flow": {
                   "scale": "1000000", "currency_literal": "$", "locator": deepcopy(loc)}}})
    candidate = {k: deepcopy(old[k]) for k in gate.src.METADATA}
    candidate.update({key: False for key in gate.FLAGS})
    candidate["stock_compensation"] = dict(
        field="stock_compensation_addback", candidate_only=True, gaps=[],
        accounting_review_flags=[dict(code="new_fields_independent_source_review_required")],
        value=dict(source=deepcopy(source), source_sha256=source["sha256"], period=deepcopy(period),
                   available_from="2025-01-02", fact_admitted=False, currency_iso=None,
                   value_decimal="50", scale="1000000", currency_literal="$", locator=loc),
        **{key: False for key in gate.FLAGS + gate.CERTS})
    return old, candidate


class Contracts(unittest.TestCase):
    def test_absolute_and_relative_local_source_closure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve()
            file=root/"source.txt";file.write_text("synthetic")
            sha=hashlib.sha256(file.read_bytes()).hexdigest()
            evidence=gate.old.Evidence(root)
            evidence.pins({"source.txt":sha,str(file):sha})
            evidence.stable()

    def test_external_cloud_traversal_and_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();file=root/"source.txt";file.write_text("synthetic")
            link=root/"link";link.symlink_to(file)
            for name in ("/etc/hosts","../source.txt",str(link),"OneDrive/source.txt"):
                with self.subTest(name=name),self.assertRaises(ValueError):
                    gate.old.Evidence(root).hash(name)

    def test_absolute_closure_digest_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();file=root/"source.txt";file.write_text("synthetic")
            with self.assertRaises(ValueError):
                gate.old.Evidence(root).pins({str(file):"0"*64})
    def test_valid_reviewed(self):
        old, candidate = fixture()
        original = deepcopy((old, candidate))
        result = gate.admitted_event(old, candidate, {old["event_id"]})
        self.assertTrue(result["stock_compensation_admission"]["admitted"])
        self.assertEqual((old, candidate), original)
        self.assertFalse(result["fact_admitted"])

    def test_old_hold_survives_complete_new_value(self):
        old, candidate = fixture()
        old["cash_quality_admission"]["admitted"] = False
        candidate["cash_quality_admission"] = deepcopy(old["cash_quality_admission"])
        result = gate.admitted_event(old, candidate, {old["event_id"]})
        self.assertEqual(result["stock_compensation_admission"]["reasons"],
                         ["preserved_original_cash_quality_hold"])

    def test_unreviewed_value_held(self):
        old, candidate = fixture()
        result = gate.admitted_event(old, candidate, set())
        self.assertFalse(result["stock_compensation_admission"]["admitted"])

    def test_missing_not_zero(self):
        old, candidate = fixture()
        candidate["stock_compensation"].update(value=None, gaps=[{"reason": "missing"}])
        result = gate.admitted_event(old, candidate, {old["event_id"]})
        self.assertFalse(result["stock_compensation_admission"]["admitted"])
        self.assertIsNone(result["stock_compensation"]["value"])

    def test_zero_and_negative_preserved(self):
        for amount in ("0", "-10"):
            old, candidate = fixture()
            candidate["stock_compensation"]["value"]["value_decimal"] = amount
            result = gate.admitted_event(old, candidate, {old["event_id"]})
            self.assertTrue(result["stock_compensation_admission"]["admitted"])
            self.assertEqual(result["stock_compensation"]["value"]["value_decimal"], amount)

    def test_metadata_changes_fail(self):
        for name in gate.src.METADATA:
            old, candidate = fixture()
            candidate[name] = "changed"
            with self.assertRaises(ValueError): gate.check_cell(old, candidate)

    def test_raw_flags_cannot_be_promoted(self):
        for layer in ("event", "result"):
            for key in gate.FLAGS:
                old, candidate = fixture()
                (candidate if layer == "event" else candidate["stock_compensation"])[key] = True
                with self.assertRaises(ValueError): gate.check_cell(old, candidate)

    def test_certification_cannot_be_promoted(self):
        for key in gate.CERTS:
            old, candidate = fixture()
            candidate["stock_compensation"][key] = True
            with self.assertRaises(ValueError): gate.check_cell(old, candidate)

    def test_value_and_gap_conflict(self):
        old, candidate = fixture()
        candidate["stock_compensation"]["gaps"] = [{"reason": "ambiguous"}]
        with self.assertRaises(ValueError): gate.check_cell(old, candidate)

    def test_unexplained_missing_fails(self):
        old, candidate = fixture()
        candidate["stock_compensation"]["value"] = None
        with self.assertRaises(ValueError): gate.check_cell(old, candidate)

    def test_value_source_period_and_time(self):
        for key in ("source", "source_sha256", "period", "available_from"):
            old, candidate = fixture()
            candidate["stock_compensation"]["value"][key] = "wrong"
            with self.assertRaises(ValueError): gate.check_cell(old, candidate)

    def test_exact_decimal_strings(self):
        for key, value in (("value_decimal", 50.0), ("scale", 1000000),
                           ("value_decimal", "NaN"), ("value_decimal", "Infinity"),
                           ("scale", "0"), ("scale", "-1")):
            old, candidate = fixture()
            candidate["stock_compensation"]["value"][key] = value
            with self.assertRaises(ValueError): gate.check_cell(old, candidate)

    def test_no_new_iso_currency(self):
        old, candidate = fixture()
        candidate["stock_compensation"]["value"]["currency_iso"] = "USD"
        with self.assertRaises(ValueError): gate.check_cell(old, candidate)

    def test_units_must_match_current_cfo(self):
        old, candidate = fixture()
        old["fields"]["current"]["operating_cash_flow"]["scale"] = "1000"
        with self.assertRaises(ValueError): gate.check_cell(old, candidate)

    def test_statement_column_and_scope_match(self):
        for key, value in (("statement_title", "balance sheet"), ("logical_columns", [2]),
                           ("statement_segment", 1), ("field_scope", "non_gaap"),
                           ("table_index_zero_based", 10)):
            old, candidate = fixture()
            candidate["stock_compensation"]["value"]["locator"][key] = value
            with self.assertRaises(ValueError): gate.check_cell(old, candidate)

    def test_optional_statement_segment_absent_on_both(self):
        old, candidate = fixture()
        del old["fields"]["current"]["operating_cash_flow"]["locator"]["statement_segment"]
        del candidate["stock_compensation"]["value"]["locator"]["statement_segment"]
        self.assertEqual(gate.check_cell(old, candidate), [])

    def test_unreviewed_new_flag_fails(self):
        old, candidate = fixture()
        candidate["stock_compensation"]["accounting_review_flags"].append({"code": "unknown"})
        with self.assertRaises(ValueError): gate.check_cell(old, candidate)

    def test_known_accounting_flags_retained(self):
        old, candidate = fixture()
        flags = candidate["stock_compensation"]["accounting_review_flags"]
        flags.append({"code": "cashflow_accounting_change_context_unreviewed"})
        result = gate.admitted_event(old, candidate, {old["event_id"]})
        self.assertEqual(result["stock_compensation"]["accounting_review_flags"], flags)

    def test_policy_does_not_license_calculations(self):
        policy = gate.policy_fields()
        self.assertFalse(policy["financial_ratios_permitted"])
        self.assertFalse(policy["strategy_registration_permitted"])
        self.assertTrue(policy["preserve_all_13_original_holds"])
        self.assertTrue(policy["invalid_latest_blocks_older_fallback"])


if __name__ == "__main__":
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    print(log.getvalue())
    if result.wasSuccessful():
        root = Path(__file__).resolve().parents[1]
        gate.src.base.publish(root, gate.OUT + "/TESTS.json", dict(
            passed=True, synthetic_only=True, tests_run=result.testsRun,
            failures=len(result.failures), errors=len(result.errors),
            real_source_cells_extracted=0, ratios_computed=0,
            source_code_hashes=gate.code_hashes(root), test_output=log.getvalue()))
    raise SystemExit(0 if result.wasSuccessful() else 1)
