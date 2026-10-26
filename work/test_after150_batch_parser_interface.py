"""Synthetic parser-to-batch integration; no real report amounts are read."""
import hashlib
import json
from pathlib import Path
import unittest

import after150_source_batch as batch
import after150_intc_extract as intc
import after150_klac_extract as klac
import test_after150_intc_extract as intc_fixtures
import test_after150_klac_extract as klac_fixtures


class ParserBatchInterfaceTests(unittest.TestCase):
    def accept_intc(self, **kwargs):
        raw, context = intc_fixtures.fixture(**kwargs)
        output = intc.extract_document(raw, context)
        batch.validate_record(output, context)
        return output

    def accept_klac(self, **kwargs):
        raw, context = klac_fixtures.fixture(**kwargs)
        output = klac.extract_document(raw, context)
        batch.validate_record(output, context)
        return output

    def test_intc_three_fixed_years_and_title_families(self):
        for year, style in ((2017, "summary"), (2020, "plain"), (2021, "condensed")):
            with self.subTest(year=year):
                out = self.accept_intc(year=year, title_style=style)
                self.assertTrue(all(v is not None for v in out["fields"].values()))

    def test_intc_reversed_and_narrow_columns(self):
        for reverse, narrow in ((True, False), (False, True), (True, True)):
            with self.subTest(reverse=reverse, narrow=narrow):
                self.accept_intc(reverse=reverse, narrow=narrow)

    def test_intc_loss_zero_and_missing_income(self):
        for income in ("(1,234.50)", "-123", "0", "—"):
            with self.subTest(income=income):
                out = self.accept_intc(ni=income)
                self.assertEqual(out["fields"]["annual_net_income"] is None, income == "—")

    def test_intc_preserves_uncertified_derived_start(self):
        out = self.accept_intc()
        self.assertIs(out["periods"]["start_assumption_certified"], False)
        self.assertIn("annual_start_not_directly_quoted_requires_source_use_assumption", out["context_holds"])

    def test_klac_all_supported_layouts(self):
        for layout in ("wide", "compact", "compact_gap2"):
            with self.subTest(layout=layout):
                out = self.accept_klac(layout=layout)
                self.assertTrue(all(v is not None for v in out["fields"].values()))

    def test_klac_merged_numeric_cells(self):
        self.accept_klac(merged=True)

    def test_klac_loss_zero_and_missing_income(self):
        for income in ("(25)", "-25", "0", "—"):
            with self.subTest(income=income):
                out = self.accept_klac(ni=income)
                self.assertEqual(out["fields"]["annual_net_income"] is None, income == "—")

    def test_klac_pre_year_retains_three_nulls_and_hold(self):
        raw, context = klac_fixtures.fixture(year=2011)
        context["context_summary"] = dict(status="preformatted_original_requires_separate_column_geometry_review", fact_admitted=False)
        out = klac.extract_document(raw, context)
        batch.validate_record(out, context)
        self.assertTrue(all(v is None for v in out["fields"].values()))
        self.assertIn("KLAC_FY2011_preformatted_column_geometry_unverified", out["context_holds"])

    def test_klac_transition_retains_accounting_hold(self):
        out = self.accept_klac(year=2019)
        self.assertIn("KLAC_FY2019_ASC606_current_comparison_not_restated__average_assets_use_hold", out["context_holds"])

    def test_parser_outputs_fail_on_lost_source_or_hold(self):
        for parser, fixture in ((intc, intc_fixtures), (klac, klac_fixtures)):
            raw, context = fixture.fixture()
            out = parser.extract_document(raw, context)
            out["source"]["sha256"] = "0" * 64
            with self.assertRaises(Exception):
                batch.validate_record(out, context)
        raw, context = intc_fixtures.fixture()
        out = intc.extract_document(raw, context)
        out["context_holds"] = []
        with self.assertRaises(Exception):
            batch.validate_record(out, context)

    def test_metadata_only_policy_matches_full_context_catalog(self):
        root = Path(__file__).resolve().parents[1]
        context_path = root / batch.CONTEXT
        raw = context_path.read_bytes()
        context = json.loads(raw)
        policy = json.loads((root / batch.POLICY).read_text())
        batch.validate_policy(policy, context, hashlib.sha256(raw).hexdigest())
        chosen = batch.records(context)
        self.assertEqual(len(context["documents"]), 32)
        self.assertEqual(len(chosen), 19)
        self.assertEqual(len(policy["fixed_catalog_metadata"]), 32)
        self.assertEqual(set(policy["fixed_catalog_metadata"][0]), {
            "event_id", "symbol", "fiscal_year", "publication_date", "available_session",
            "official_actual_period_end", "source_path", "source_sha256",
            "html_context_reviewed", "retained_context_gap"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
