"""Synthetic transport checks only; no original financial amounts read."""
from copy import deepcopy
import unittest
import after149_source_batch as batch


def fixture():
    field = {"value_decimal": "10", "scale": "1000000", "fact_admitted": False,
             "displayed_amount": "10", "locator": {"row": 7}}
    return {"symbol": "CSCO", "event_id": "synthetic",
            "fact_admitted": False, "model_admitted": False,
            "strategy_admitted": False,
            "fields": {key: deepcopy(field) for key in batch.FIELDS},
            "gaps": []}


def linked_fixture():
    row = fixture()
    source = {"path": "work/synthetic.html", "sha256": "a" * 64}
    periods = {
        "current": {"period_start": "2024-07-28", "period_end": "2025-07-26", "days_inclusive": 364},
        "comparison": {"period_start": "2023-07-30", "period_end": "2024-07-27", "days_inclusive": 364},
    }
    context = {"available_session": "2025-08-14", "periods": periods}
    expected = {"symbol": "CSCO", "event_id": "synthetic", "fiscal_year": 2025,
                "source_path": source["path"], "source_sha256": source["sha256"]}
    row.update(fiscal_year=2025, source=source, available_from=context["available_session"])
    for key, field in row["fields"].items():
        side = "comparison" if key == "prior_total_assets" else "current"
        field.update(source=deepcopy(source), available_from=context["available_session"],
                     as_of=periods[side]["period_end"],
                     period=deepcopy(periods["current"]) if key == "annual_net_income" else None)
    return row, expected, context


class TransportTests(unittest.TestCase):
    def test_valid(self):
        batch.validate_record(fixture())

    def test_negative_and_zero_income_valid(self):
        for value in ("-2", "0"):
            row = fixture()
            row["fields"]["annual_net_income"]["value_decimal"] = value
            batch.validate_record(row)

    def test_zero_and_negative_assets_rejected(self):
        for field in batch.FIELDS[1:]:
            for value in ("0", "-3"):
                with self.subTest(field=field, value=value):
                    row = fixture()
                    row["fields"][field]["value_decimal"] = value
                    with self.assertRaises(ValueError):
                        batch.validate_record(row)

    def test_nonfinite_rejected(self):
        for key in ("value_decimal", "scale"):
            for value in ("NaN", "Infinity", "-Infinity"):
                row = fixture()
                row["fields"]["annual_net_income"][key] = value
                with self.assertRaises(ValueError):
                    batch.validate_record(row)

    def test_missing_with_explicit_gap_retained(self):
        row = fixture()
        row["fields"]["annual_net_income"] = None
        row["gaps"] = ["unresolved signed column"]
        batch.validate_record(row)

    def test_missing_without_gap_rejected(self):
        row = fixture()
        row["fields"]["annual_net_income"] = None
        with self.assertRaises(ValueError):
            batch.validate_record(row)

    def test_no_implicit_admission(self):
        for key in ("fact_admitted", "model_admitted", "strategy_admitted"):
            row = fixture()
            row[key] = True
            with self.assertRaises(ValueError):
                batch.validate_record(row)

    def test_no_missing_locator(self):
        row = fixture()
        row["fields"]["prior_total_assets"]["locator"] = {}
        with self.assertRaises(ValueError):
            batch.validate_record(row)

    def test_outside_root_rejected(self):
        with self.assertRaises(ValueError):
            batch.path("/tmp/unrelated_source")

    def test_linked_record(self):
        batch.validate_record(*linked_fixture())

    def test_wrong_top_level_identity(self):
        for key, value in [("symbol", "AMZN"), ("event_id", "other"), ("fiscal_year", 2024)]:
            row, expected, context = linked_fixture()
            row[key] = value
            with self.assertRaises(ValueError):
                batch.validate_record(row, expected, context)

    def test_wrong_field_source_clock_or_endpoint(self):
        for key, value in [("source", {"path": "work/other.html", "sha256": "b" * 64}),
                           ("available_from", "2025-08-13"), ("as_of", "2025-07-25"),
                           ("fact_admitted", True)]:
            row, expected, context = linked_fixture()
            row["fields"]["annual_net_income"][key] = value
            with self.assertRaises(ValueError):
                batch.validate_record(row, expected, context)

    def test_extra_field_rejected(self):
        row = fixture()
        row["fields"]["extra"] = deepcopy(row["fields"]["annual_net_income"])
        with self.assertRaises(ValueError):
            batch.validate_record(row)


if __name__ == "__main__":
    unittest.main()
