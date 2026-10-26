"""Synthetic-only regression tests; never read AMZN financial source values."""
import copy
import hashlib
import unittest
from unittest.mock import patch

import after149_amzn_extract as m


def td(text="", span=1):
    return f'<td colspan="{span}">{text}</td>'


def tr(*cells):
    return "<tr>" + "".join(cells) + "</tr>"


def group(value, dollar=False):
    # Signed amounts may span three physical cells.
    if value.startswith("(") and value.endswith(")"):
        return td("$ (" if dollar else "(") + td(value[1:-1]) + td(")")
    return td("$" if dollar else "") + td(value) + td()


def fixture(year=2021, reversed_order=False, narrow=False, ni="(1,234.50)",
            assets="8,000", prior_assets="7,000", include_dollar=True,
            unit="millions", prior_unit=None):
    ys = [year, year-1] if reversed_order else [year-1, year]
    cur_index = ys.index(year)
    # Columns 1..3,5..7 quarterly; 9..11,13..15 annual.
    hdr = [td()]
    for y in ys + ys:
        hdr += ([td(), td(str(y)), td()] if narrow else [td(str(y), 3)]) + [td()]
    cf = [
        tr(td("Consolidated Statements of Cash Flows", 17)),
        tr(td(f"(in {unit})", 17)),
        tr(td(), td("Three Months Ended December 31,", 7), td(),
           td("Twelve Months Ended December 31,", 7), td()),
        tr(*hdr),
        tr(td("OPERATING ACTIVITIES", 17)),
        tr(td("Net income (loss)"), group("999"), td(), group("888"), td(),
           *sum(([group(ni if y == year else "111"), td()] for y in ys), [])),
        tr(td("Adjustments to reconcile net income to net cash from operating activities", 17)),
        tr(td("Depreciation"), td("9", 16)),
        tr(td("Net cash provided by (used in) operating activities"), td("$" if include_dollar else "", 16)),
    ]
    bshead = [td()]
    for y in ys:
        bshead += ([td(), td(f"December 31, {y}"), td()] if narrow
                   else [td(f"December 31, {y}", 3)]) + [td()]
    bs = [
        tr(td("Consolidated Balance Sheets", 9)),
        tr(td(f"(in {prior_unit or unit}, except per share data)", 9)),
        tr(*bshead),
        tr(td("ASSETS", 9)),
        tr(td("Total assets"),
           *sum(([group(assets if y == year else prior_assets, include_dollar), td()] for y in ys), [])),
    ]
    raw = ("<html><p>Amazon.com, Inc. (NASDAQ: AMZN)</p>"
           "<table>" + "".join(cf) + "</table><table>" + "".join(bs) + "</table></html>").encode()
    c = {
        "event_id": f"AMZN_{year+1}-02-01", "fiscal_year": year,
        "publication_date": f"{year+1}-02-01", "available_session": f"{year+1}-02-02",
        "current_end": f"{year}-12-31", "comparison_end": f"{year-1}-12-31",
        "path": "synthetic/raw.html", "sha256": hashlib.sha256(raw).hexdigest(),
        "receipt": "synthetic/receipt.json",
        "cf": {"table": 0, "NI_row": 5, "CFO_row": 8,
               "current_header_columns": ([10+4*cur_index] if narrow else list(range(9+4*cur_index, 12+4*cur_index))),
               "comparison_header_columns": ([10+4*(1-cur_index)] if narrow else list(range(9+4*(1-cur_index), 12+4*(1-cur_index))))},
        "assets": {"table": 1, "row": 4,
                   "current_header_columns": ([2+4*cur_index] if narrow else list(range(1+4*cur_index, 4+4*cur_index))),
                   "comparison_header_columns": ([2+4*(1-cur_index)] if narrow else list(range(1+4*(1-cur_index), 4+4*(1-cur_index))))},
        "accounting_context": "Synthetic accounting review remains pending",
    }
    return raw, c


def changed(raw, c, old, new):
    raw = raw.replace(old.encode(), new.encode())
    c = copy.deepcopy(c)
    c["sha256"] = m.sha256(raw)
    return raw, c


class ExtractTests(unittest.TestCase):
    def out(self, **kwargs):
        return m.extract_document(*fixture(**kwargs))

    def test_three_fields_exact_signed_and_scale(self):
        out = self.out()
        self.assertEqual(set(out["fields"]), set(m.FIELDS))
        self.assertEqual(out["fields"]["annual_net_income"]["value_decimal"], "-1234.50")
        self.assertEqual(out["fields"]["current_total_assets"]["value_decimal"], "8000")
        self.assertEqual(out["fields"]["prior_total_assets"]["value_decimal"], "7000")
        self.assertTrue(out["complete_required_fields"])
        self.assertEqual(out["gaps"], [])
        self.assertEqual(out["fields"]["annual_net_income"]["scale"], "1000000")

    def test_column_order_is_not_assumed(self):
        for reverse in (True, False):
            self.assertEqual(self.out(reversed_order=reverse)["fields"]["annual_net_income"]["value_decimal"], "-1234.50")

    def test_narrow_year_anchor_preserves_split_parentheses(self):
        out = self.out(narrow=True)
        self.assertEqual(out["fields"]["annual_net_income"]["value_decimal"], "-1234.50")
        self.assertEqual(len(out["fields"]["annual_net_income"]["locator"]["physical_cells"]), 3)

    def test_cross_year_colspan_is_rejected(self):
        raw, c = fixture()
        # One amount spanning both asset year anchors cannot belong to either.
        raw, c = changed(
            raw, c,
            tr(td("Total assets"), group("7,000", True), td(), group("8,000", True), td()),
            tr(td("Total assets"), td("$ 7,000", 7), td()))
        out = m.extract_document(raw, c)
        self.assertIsNone(out["fields"]["current_total_assets"])
        self.assertIsNone(out["fields"]["prior_total_assets"])
        self.assertTrue(all(g["reason"] == "amount_cell_crosses_year_group" for g in out["gaps"]))

    def test_amount_originating_rowspan_is_rejected(self):
        physical = {
            (2, 0): {"text": "Net income", "columns": [0], "row": 2, "physical_cell": 0, "rowspan": 1},
            (2, 1): {"text": "123", "columns": [1], "row": 2, "physical_cell": 1, "rowspan": 2},
        }
        with self.assertRaisesRegex(m.FieldGap, "crosses_rows"):
            m._value(physical, physical, 2, [1], {1})

    def test_adjacent_year_parenthesis_may_not_be_borrowed(self):
        # Construct flat helper geometry with consecutive one-column years.
        headers = [
            {"text": "Twelve Months Ended December 31,", "columns": [1, 2], "row": 0, "physical_cell": 1},
            {"text": "2020", "columns": [1], "row": 1, "physical_cell": 1},
            {"text": "2021", "columns": [2], "row": 1, "physical_cell": 2},
        ]
        physical = {
            (2, 0): {"text": "Net income", "columns": [0], "row": 2, "physical_cell": 0},
            (2, 1): {"text": "(123", "columns": [1], "row": 2, "physical_cell": 1},
            (2, 2): {"text": ")", "columns": [2], "row": 2, "physical_cell": 2},
        }
        owned = m._owned_columns(headers, physical, 2, [1], 2020, True)
        with self.assertRaisesRegex(m.FieldGap, "crosses_year"):
            m._value(physical, physical, 2, [1], owned)

    def test_original_cells_retained(self):
        raw, c = fixture(ni="1,234")
        raw, c = changed(raw, c, "1,234", "1&nbsp;,&nbsp;234")
        f = m.extract_document(raw, c)["fields"]["annual_net_income"]
        self.assertEqual(f["value_decimal"], "1234")
        self.assertIn("\xa0", "".join(x["original_text"] for x in f["locator"]["physical_cells"]))

    def test_zero_is_not_missing(self):
        self.assertEqual(self.out(ni="0")["fields"]["annual_net_income"]["value_decimal"], "0")

    def test_signed_minus_is_preserved(self):
        self.assertEqual(self.out(ni="-123")["fields"]["annual_net_income"]["value_decimal"], "-123")

    def test_large_decimal_does_not_round(self):
        value = "123456789012345678901234567890123456789.123456"
        self.assertEqual(self.out(ni=f"({value})")["fields"]["annual_net_income"]["value_decimal"], "-" + value)

    def test_dash_and_blank_are_gaps_not_zero(self):
        for value in ("-", "—", "", "N/A"):
            with self.subTest(value=value):
                out = self.out(ni=value)
                self.assertIsNone(out["fields"]["annual_net_income"])
                self.assertEqual(out["gaps"][0]["field"], "annual_net_income")

    def test_ambiguous_numbers_are_not_joined(self):
        self.assertIsNone(self.out(ni="123 456")["fields"]["annual_net_income"])

    def test_conflicting_signs_rejected(self):
        self.assertIsNone(self.out(ni="(-123)")["fields"]["annual_net_income"])

    def test_raw_hash_failure_is_hard(self):
        raw, c = fixture()
        with self.assertRaises(m.SourceError):
            m.extract_document(raw+b" ", c)

    def test_identity_failure_is_hard(self):
        raw, c = fixture()
        raw, c = changed(raw, c, "NASDAQ: AMZN", "NASDAQ: OTHER")
        with self.assertRaises(m.SourceError):
            m.extract_document(raw, c)

    def test_context_event_identity_failure(self):
        raw, c = fixture()
        c["event_id"] = "MSFT_2022-02-01"
        with self.assertRaises(m.SourceError):
            m.extract_document(raw, c)

    def test_unregistered_year_failure(self):
        with self.assertRaises(m.SourceError):
            m.extract_document(*fixture(year=2015))

    def test_invalid_and_inconsistent_dates_are_full_event_gaps(self):
        for key, value in (("comparison_end", None), ("current_end", "2021-11-30"),
                           ("available_session", "2022-02-01"), ("publication_date", "2021-12-01")):
            raw, c = fixture()
            c[key] = value
            c["event_id"] = "AMZN_" + c["publication_date"]
            out = m.extract_document(raw, c)
            self.assertTrue(all(v is None for v in out["fields"].values()))
            self.assertEqual(len(out["gaps"]), 3)

    def test_quarterly_same_year_is_rejected(self):
        raw, c = fixture()
        c["cf"]["current_header_columns"] = [5, 6, 7]
        out = m.extract_document(raw, c)
        self.assertIsNone(out["fields"]["annual_net_income"])
        self.assertIn("not_exclusively_annual_column", [g["reason"] for g in out["gaps"]])

    def test_wrong_year_anchor_is_rejected(self):
        raw, c = fixture()
        c["cf"]["current_header_columns"] = c["cf"]["comparison_header_columns"]
        self.assertIsNone(m.extract_document(raw, c)["fields"]["annual_net_income"])

    def test_prior_assets_require_own_date_header(self):
        raw, c = fixture()
        raw, c = changed(raw, c, "December 31, 2020", "December 30, 2020")
        out = m.extract_document(raw, c)
        self.assertIsNone(out["fields"]["prior_total_assets"])
        # Without both dated groups, ownership cannot safely be established.
        self.assertIsNone(out["fields"]["current_total_assets"])

    def test_missing_scale_and_currency(self):
        out = self.out(include_dollar=False)
        self.assertEqual(len(out["gaps"]), 3)
        out = self.out(unit="unspecified")
        self.assertEqual(len(out["gaps"]), 3)

    def test_conflicting_table_scale_is_gap(self):
        raw, c = fixture()
        raw, c = changed(raw, c, "(in millions)", "(in millions) (in thousands)")
        self.assertIsNone(m.extract_document(raw, c)["fields"]["annual_net_income"])

    def test_cross_statement_scale_mismatch_is_hold(self):
        out = self.out(prior_unit="thousands")
        self.assertIn("cross_statement_scale_mismatch", out["context_holds"])
        self.assertTrue(out["complete_required_fields"])

    def test_currency_is_only_literal_assumption(self):
        f = self.out()["fields"]["current_total_assets"]
        self.assertEqual(f["currency_literal"], "$")
        self.assertIsNone(f["currency_iso"])

    def test_other_currency_is_gap(self):
        raw, c = fixture()
        raw, c = changed(raw, c, "(in millions)", "(in millions EUR)")
        self.assertIsNone(m.extract_document(raw, c)["fields"]["annual_net_income"])

    def test_attributed_NI_is_not_consolidated_NI(self):
        raw, c = fixture()
        raw, c = changed(raw, c, "Net income (loss)", "Net income attributable to common shareholders")
        self.assertIsNone(m.extract_document(raw, c)["fields"]["annual_net_income"])

    def test_reconciliation_anchor_required(self):
        raw, c = fixture()
        raw, c = changed(raw, c, "Adjustments to reconcile net income to net cash from operating activities", "Other items")
        self.assertIsNone(m.extract_document(raw, c)["fields"]["annual_net_income"])

    def test_duplicate_target_in_formal_table_is_gap(self):
        raw, c = fixture()
        raw, c = changed(raw, c, "Depreciation", "Net income (loss)")
        self.assertIsNone(m.extract_document(raw, c)["fields"]["annual_net_income"])

    def test_unrelated_non_gaap_duplicate_does_not_redirect(self):
        raw, c = fixture()
        raw, c = changed(raw, c, "</html>", "<table><tr><td>Non-GAAP Net income</td><td>999999</td></tr></table></html>")
        self.assertEqual(m.extract_document(raw, c)["fields"]["annual_net_income"]["value_decimal"], "-1234.50")

    def test_non_gaap_formal_header_rejected(self):
        raw, c = fixture()
        raw, c = changed(raw, c, "Consolidated Statements of Cash Flows", "Non-GAAP Consolidated Statements of Cash Flows")
        self.assertIsNone(m.extract_document(raw, c)["fields"]["annual_net_income"])

    def test_locator_mismatch_does_not_search_fallback(self):
        raw, c = fixture()
        c["assets"]["row"] = 3
        out = m.extract_document(raw, c)
        self.assertIsNone(out["fields"]["current_total_assets"])
        self.assertIsNone(out["fields"]["prior_total_assets"])

    def test_2019_restriction_survives_complete_values(self):
        out = self.out(year=2019)
        self.assertTrue(out["complete_required_fields"])
        self.assertIn("FY2019_lease_adoption_prior_period_not_retrospectively_adjusted", out["context_holds"])

    def test_nonpositive_assets_retained_with_hold(self):
        out = self.out(prior_assets="0")
        self.assertEqual(out["fields"]["prior_total_assets"]["value_decimal"], "0")
        self.assertIn("prior_total_assets_nonpositive", out["context_holds"])

    def test_never_admits_and_never_backdates_prior_assets(self):
        out = self.out()
        self.assertTrue(all(out[k] is False for k in ("fact_admitted", "model_admitted", "strategy_admitted")))
        self.assertEqual(out["fields"]["prior_total_assets"]["available_from"], "2022-02-02")
        self.assertEqual(out["fields"]["annual_net_income"]["period"],
                         {"period_start": "2021-01-01", "period_end": "2021-12-31", "days_inclusive": 365})
        self.assertEqual(out["fields"]["annual_net_income"]["as_of"], "2021-12-31")
        self.assertTrue(all(v["fact_admitted"] is False for v in out["fields"].values()))

    def test_pure_no_file_io_and_context_unchanged(self):
        raw, c = fixture()
        before = copy.deepcopy(c)
        with patch("builtins.open", side_effect=AssertionError("I/O forbidden")):
            m.extract_document(raw, c)
        self.assertEqual(c, before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
