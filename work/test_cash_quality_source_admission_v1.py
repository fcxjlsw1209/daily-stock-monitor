import copy
import hashlib
import io
import json
from pathlib import Path
import unittest
import cash_quality_source_admission_v1 as m


def event():
    periods = {
        "current": dict(period_start="2020-01-01", period_end="2020-12-31", days_inclusive=366),
        "comparison": dict(period_start="2019-01-01", period_end="2019-12-31", days_inclusive=365)}
    e = dict(symbol="TEST", event_id="TEST:2020", prior_field_admitted=True,
             inherited_rejection_codes=[], metadata_gaps=[], release_date="2021-02-01",
             available_from="2021-02-02", source=dict(path="synthetic", sha256="a"*64),
             identity=dict(issuer_id="test", security_id="TEST", identity_valid_from="2021-02-02",
                           identity_valid_through="2022-07-04"),
             periods=periods, fields={"current": {}, "comparison": {}})
    for side, field in m.REQUIRED:
        e["fields"][side][field] = dict(source=e["source"], available_from=e["available_from"],
            period=periods[side], currency_literal="$", currency_iso=None, value_decimal="10",
            scale="1000000", fact_admitted=False,
            locator=dict(period_end=periods[side]["period_end"],
                         ni_scope="indirect_method_consolidated_start"))
    return e


class AdmissionTests(unittest.TestCase):
    def evaluate(self, e):
        return m.assess(e, ["2021-02-01", "2021-02-02", "2021-02-03"])

    def test_valid_and_no_mutation(self):
        e = event()
        old = copy.deepcopy(e)
        self.assertTrue(self.evaluate(e)["admitted"])
        self.assertEqual(e, old)

    def test_old_hold_preserved_even_with_all_values(self):
        for key, value in (("prior_field_admitted", False),
                           ("inherited_rejection_codes", ["old_hold"])):
            e = event()
            e[key] = value
            self.assertFalse(self.evaluate(e)["admitted"])

    def test_missing_each_required_cell_blocks(self):
        for side, field in m.REQUIRED:
            e = event()
            e["fields"][side][field] = None
            self.assertFalse(self.evaluate(e)["admitted"])

    def test_unused_comparison_flow_missing_is_permitted(self):
        self.assertTrue(self.evaluate(event())["admitted"])

    def test_future_or_same_day_availability_blocks(self):
        for day in ("2021-02-01", "2021-02-03"):
            e = event()
            e["available_from"] = day
            self.assertFalse(self.evaluate(e)["admitted"])

    def test_period_duration_contiguity_and_expiry(self):
        mutations = [
            lambda e: e["periods"]["comparison"].update(days_inclusive=364),
            lambda e: e["periods"]["current"].update(period_start="2020-01-02", days_inclusive=365),
            lambda e: e["identity"].update(identity_valid_through="2022-07-05"),
            lambda e: e.update(release_date="2020-12-30")]
        for change in mutations:
            e = event()
            change(e)
            self.assertFalse(self.evaluate(e)["admitted"])

    def test_bad_cell_contract(self):
        for key, value in (("source", {}), ("available_from", "2021-02-03"),
                           ("period", {}), ("currency_literal", "EUR"),
                           ("currency_iso", "CAD"), ("value_decimal", "NaN"),
                           ("scale", "0"), ("fact_admitted", True)):
            e = event()
            e["fields"]["current"]["net_income"][key] = value
            self.assertFalse(self.evaluate(e)["admitted"])

    def test_consolidated_income_required(self):
        e = event()
        e["fields"]["current"]["net_income"]["locator"]["ni_scope"] = "continuing_only"
        self.assertFalse(self.evaluate(e)["admitted"])

    def test_signed_income_cash_allowed_assets_positive(self):
        for field in ("net_income", "operating_cash_flow"):
            e = event()
            e["fields"]["current"][field]["value_decimal"] = "-10"
            self.assertTrue(self.evaluate(e)["admitted"])
        for side in ("current", "comparison"):
            e = event()
            e["fields"][side]["total_assets"]["value_decimal"] = "0"
            self.assertFalse(self.evaluate(e)["admitted"])

    def test_metadata_gap_blocks(self):
        e = event()
        e["metadata_gaps"] = ["unknown period"]
        self.assertFalse(self.evaluate(e)["admitted"])

    def test_accounting_reference_and_records_must_match_reviewed_union(self):
        a = dict(stage="source_only_field_scoped_accounting_review", source_only=True,
                 events=[dict(event_id="test", inherited_rejection_codes=[],
                              inherited_rejections_unchanged=True)])
        reference = dict(path="accounting.json", sha256="a"*64)
        union = dict(events=[dict(event_id="test", inherited_rejection_codes=[],
            union_provenance=dict(accounting_review_reference=reference,
                                  accounting_event_review=copy.deepcopy(a["events"][0])))])
        m.validate_accounting(a, union, reference)
        altered = copy.deepcopy(a)
        altered["events"][0]["inherited_rejection_codes"] = ["changed"]
        with self.assertRaises(ValueError):
            m.validate_accounting(altered, union, reference)
        with self.assertRaises(ValueError):
            m.validate_accounting(a, union, dict(path="other.json", sha256="b"*64))

    def test_wrong_accounting_stage_rejected(self):
        with self.assertRaises(ValueError):
            m.validate_accounting(dict(stage="unreviewed"), dict(events=[]), {})


if __name__ == "__main__":
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(AdmissionTests))
    print(stream.getvalue())
    root = Path(__file__).resolve().parents[1]
    out = root / m.OUT
    out.mkdir(parents=True, exist_ok=True)
    record = dict(passed=result.wasSuccessful(), count=result.testsRun, synthetic_only=True,
                  source_sha256=m.digest(root / m.SELF),
                  tests_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out / "TESTS.json").write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
    raise SystemExit(not result.wasSuccessful())
