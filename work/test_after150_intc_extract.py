"""Synthetic-only tests. Never read original INTC monetary cells."""
import copy
import unittest
from unittest.mock import patch

import after150_intc_extract as m


def td(text="", span=1):
    return f'<td colspan="{span}">{text}</td>'


def tr(*cells):
    return "<tr>"+"".join(cells)+"</tr>"


def group(value, dollar=True):
    if value.startswith("(") and value.endswith(")"):
        return td("$ (" if dollar else "(")+td(value[1:-1])+td(")")
    return td("$" if dollar else "")+td(value)+td()


def fixture(year=2021, reverse=False, narrow=False, ni="(1,234.50)",
            current_assets="8,000", prior_assets="7,000", unit="millions",
            balance_unit=None, dollar=True, title_style="condensed"):
    endpoints = {2017: ("2017-12-30", "2016-12-31", "2017-01-01"),
                 2020: ("2020-12-26", "2019-12-28", "2019-12-29"),
                 2021: ("2021-12-25", "2020-12-26", "2020-12-27")}
    current, prior, start = endpoints.get(year, (f"{year}-12-25", f"{year-1}-12-26", f"{year-1}-12-27"))
    ends = [prior, current] if reverse else [current, prior]
    current_index = ends.index(current)
    dates = {e: "Dec "+str(int(e[8:10]))+", "+e[:4] for e in ends}
    labels = [td()]
    for e in ends+ends:
        labels += ([td(), td(dates[e]), td()] if narrow else [td(dates[e], 3)])+[td()]
    income = [
        tr(td(f"(In {unit}, Except Per Share Amounts; unaudited)", 17)),
        tr(td(), td("Three Months Ended", 7), td(), td("Twelve Months Ended", 7), td()),
        tr(*labels),
        tr(td("Net revenue"), td("777", 16)),
        tr(td("Net income"), group("999", dollar), td(), group("888", dollar), td(),
           *sum(([group(ni if e == current else "111", dollar), td()] for e in ends), [])),
        tr(td("Other row"), td("42", 16)),
    ]
    bs_labels = [td()]
    for e in ends:
        bs_labels += ([td(), td(dates[e]), td()] if narrow else [td(dates[e], 3)])+[td()]
    balance = [
        tr(td(f"(In {balance_unit or unit})", 9)),
        tr(*bs_labels), tr(td("ASSETS", 9)),
        tr(td("Total assets"),
           *sum(([group(current_assets if e == current else prior_assets, dollar), td()] for e in ends), [])),
    ]
    titles = {
        "summary": ("CONSOLIDATED SUMMARY STATEMENT OF INCOME DATA", "CONSOLIDATED SUMMARY BALANCE SHEET DATA"),
        "plain": ("Consolidated Statements of Income and Other Information", "Consolidated Balance Sheets"),
        "condensed": ("Consolidated Condensed Statements of Income and Other Information", "Consolidated Condensed Balance Sheets"),
    }[title_style]
    raw = ("<html><p>Intel Corporation (Nasdaq: INTC)</p><h3>"+titles[0]+"</h3><table>"
           +"".join(income)+"</table><h3>"+titles[1]+"</h3><table>"
           +"".join(balance)+"</table></html>").encode()
    def columns(base, index):
        return [base+4*index+1] if narrow else list(range(base+4*index, base+4*index+3))
    c = {
        "symbol": "INTC", "event_id": f"INTC_{year+1}-01-25", "fiscal_year": year,
        "publication_date": f"{year+1}-01-25", "available_session": f"{year+1}-01-26",
        "official_actual_period_end": current,
        "evidence_path": "synthetic/source.html", "evidence_sha256": m.sha256(raw),
        "source_url": "https://www.intc.com/synthetic",
        "context": {"receipt": {"path": "synthetic/source.meta.json"}},
        "context_summary": {
            "current_annual_period": {"start": None, "end": current,
                                     "candidate_start_from_prior_end_plus_one": start},
            "comparison_assets_date": prior,
            "net_income_locator": {"table": 0, "row": 4, "label_cell": 0, "label": "Net income",
                                   "current_header_columns": columns(9, current_index),
                                   "comparison_header_columns": columns(9, 1-current_index)},
            "assets_locator": {"table": 1, "row": 3, "label_cell": 0, "label": "Total assets",
                               "current_header_columns": columns(1, current_index),
                               "comparison_header_columns": columns(1, 1-current_index)},
            "accounting_attention": ["synthetic accounting flag"]},
        "context_holds": ["prior_source_hold"],
    }
    return raw, c


def changed(raw, c, before, after):
    raw = raw.replace(before.encode(), after.encode())
    c = copy.deepcopy(c)
    c["evidence_sha256"] = m.sha256(raw)
    return raw, c


class ExtractTests(unittest.TestCase):
    def out(self, **kw):
        return m.extract_document(*fixture(**kw))

    def test_three_fields_exact_signed_values(self):
        out = self.out()
        self.assertEqual(set(out["fields"]), set(m.FIELDS))
        self.assertEqual([v["value_decimal"] for v in out["fields"].values()], ["-1234.50", "8000", "7000"])
        self.assertTrue(out["complete_required_fields"])
        self.assertEqual(out["gaps"], [])

    def test_all_three_title_families_and_fixed_years(self):
        for year, style in ((2017, "summary"), (2020, "plain"), (2021, "condensed")):
            self.assertTrue(self.out(year=year, title_style=style)["complete_required_fields"])

    def test_column_order_is_not_assumed(self):
        for reverse in (True, False):
            self.assertEqual(self.out(reverse=reverse)["fields"]["annual_net_income"]["value_decimal"], "-1234.50")

    def test_narrow_anchors_preserve_split_signs(self):
        cell = self.out(narrow=True)["fields"]["annual_net_income"]
        self.assertEqual(cell["value_decimal"], "-1234.50")
        self.assertEqual(len(cell["locator"]["physical_cells"]), 3)

    def test_zero_is_not_missing(self):
        self.assertEqual(self.out(ni="0")["fields"]["annual_net_income"]["value_decimal"], "0")

    def test_signed_minus_is_not_abs(self):
        self.assertEqual(self.out(ni="-123")["fields"]["annual_net_income"]["value_decimal"], "-123")

    def test_large_decimal_does_not_round(self):
        val = "123456789012345678901234567890123456789.123456789"
        self.assertEqual(self.out(ni=f"({val})")["fields"]["annual_net_income"]["value_decimal"], "-"+val)

    def test_blank_dash_and_na_are_not_zero(self):
        for val in ("", "-", "—", "N/A"):
            self.assertIsNone(self.out(ni=val)["fields"]["annual_net_income"])

    def test_multiple_numbers_and_conflicting_signs_rejected(self):
        for val in ("12 34", "(-123)", "12%", "$ $ 10"):
            self.assertIsNone(self.out(ni=val)["fields"]["annual_net_income"])

    def test_original_text_and_separator_whitespace_preserved(self):
        raw, c = fixture(ni="1,234")
        raw, c = changed(raw, c, "1,234", "1&nbsp;,&nbsp;234")
        cell = m.extract_document(raw, c)["fields"]["annual_net_income"]
        self.assertEqual(cell["value_decimal"], "1234")
        self.assertIn("\xa0", "".join(x["original_text"] for x in cell["locator"]["physical_cells"]))

    def test_hash_mismatch_fails_hard(self):
        raw, c = fixture()
        with self.assertRaises(m.SourceError):
            m.extract_document(raw+b" ", c)

    def test_wrong_issuer_or_ticker_fails_hard(self):
        for before, after in (("Intel Corporation", "Other Corporation"), ("Nasdaq: INTC", "Nasdaq: ELSE")):
            raw, c = changed(*fixture(), before, after)
            with self.assertRaises(m.SourceError):
                m.extract_document(raw, c)

    def test_fixed_scope_and_event_identity_fail_hard(self):
        for key, val in (("symbol", "AMD"), ("fiscal_year", 2018), ("event_id", "INTC_1900-01-01")):
            raw, c = fixture()
            c[key] = val
            with self.assertRaises(m.SourceError):
                m.extract_document(raw, c)

    def test_invalid_source_reference_fails_hard(self):
        for key, val in (("source_url", "https://example.com/wrong"),
                         ("source_url", "http://www.intc.com/wrong"),
                         ("source_url", "https://user:secret@www.intc.com/wrong"),
                         ("evidence_path", "")):
            raw, c = fixture()
            c[key] = val
            with self.assertRaises(m.SourceError):
                m.extract_document(raw, c)

    def test_quarterly_same_year_cannot_replace_annual(self):
        raw, c = fixture()
        c["context_summary"]["net_income_locator"]["current_header_columns"] = [1, 2, 3]
        out = m.extract_document(raw, c)
        self.assertIsNone(out["fields"]["annual_net_income"])
        self.assertIn("not_exclusively_annual_column", [g["reason"] for g in out["gaps"]])

    def test_wrong_year_anchor_is_gap(self):
        raw, c = fixture()
        spec = c["context_summary"]["net_income_locator"]
        spec["current_header_columns"] = spec["comparison_header_columns"]
        self.assertIsNone(m.extract_document(raw, c)["fields"]["annual_net_income"])

    def test_calendar_dec31_is_not_fiscal_end(self):
        raw, c = changed(*fixture(), "Dec 25, 2021", "Dec 31, 2021")
        out = m.extract_document(raw, c)
        self.assertIsNone(out["fields"]["annual_net_income"])
        self.assertIsNone(out["fields"]["current_total_assets"])

    def test_prior_assets_need_their_own_exact_date(self):
        raw, c = changed(*fixture(), "Dec 26, 2020", "Dec 25, 2020")
        self.assertIsNone(m.extract_document(raw, c)["fields"]["prior_total_assets"])

    def test_missing_and_conflicting_units_are_gaps(self):
        for unit in ("unspecified", "millions) (in thousands"):
            self.assertEqual(len(self.out(unit=unit)["gaps"]), 3)

    def test_cross_statement_scale_mismatch_is_hold(self):
        out = self.out(balance_unit="thousands")
        self.assertTrue(out["complete_required_fields"])
        self.assertIn("cross_statement_scale_mismatch", out["context_holds"])

    def test_missing_dollar_is_not_inferred_currency(self):
        self.assertEqual(len(self.out(dollar=False)["gaps"]), 3)

    def test_conflicting_currency_is_gap(self):
        raw, c = changed(*fixture(), "Except Per Share Amounts", "Except Per Share Amounts EUR")
        self.assertIsNone(m.extract_document(raw, c)["fields"]["annual_net_income"])

    def test_attributed_ni_is_not_consolidated_ni(self):
        raw, c = changed(*fixture(), "Net income", "Net income attributable to shareholders")
        self.assertIsNone(m.extract_document(raw, c)["fields"]["annual_net_income"])

    def test_duplicate_ni_within_statement_is_gap(self):
        raw, c = changed(*fixture(), "Other row", "Net income")
        self.assertIsNone(m.extract_document(raw, c)["fields"]["annual_net_income"])

    def test_non_gaap_duplicate_elsewhere_does_not_redirect(self):
        raw, c = changed(*fixture(), "</html>", "<table><tr><td>Non-GAAP Net income</td><td>99999</td></tr></table></html>")
        self.assertEqual(m.extract_document(raw, c)["fields"]["annual_net_income"]["value_decimal"], "-1234.50")

    def test_non_gaap_title_is_rejected(self):
        raw, c = changed(*fixture(), "Consolidated Condensed Statements", "Non-GAAP Consolidated Condensed Statements")
        self.assertIsNone(m.extract_document(raw, c)["fields"]["annual_net_income"])

    def test_missing_formal_title_is_gap(self):
        raw, c = changed(*fixture(), "Consolidated Condensed Statements of Income and Other Information", "Summary")
        self.assertIsNone(m.extract_document(raw, c)["fields"]["annual_net_income"])

    def test_wrong_locator_does_not_search_for_alternative(self):
        raw, c = fixture()
        c["context_summary"]["assets_locator"]["row"] = 2
        out = m.extract_document(raw, c)
        self.assertIsNone(out["fields"]["current_total_assets"])
        self.assertIsNone(out["fields"]["prior_total_assets"])

    def test_cross_year_colspan_rejected(self):
        raw, c = fixture()
        before = tr(td("Total assets"), group("8,000"), td(), group("7,000"), td())
        raw, c = changed(raw, c, before, tr(td("Total assets"), td("$ 8,000", 7), td()))
        out = m.extract_document(raw, c)
        self.assertTrue(all(out["fields"][name] is None for name in ("current_total_assets", "prior_total_assets")))

    def test_amount_rowspan_rejected(self):
        physical = {(2, 0): {"text": "Net income", "columns": [0], "row": 2, "physical_cell": 0, "rowspan": 1},
                    (2, 1): {"text": "123", "columns": [1], "row": 2, "physical_cell": 1, "rowspan": 2}}
        with self.assertRaisesRegex(m.FieldGap, "crosses_rows"):
            m._value(physical, physical, 2, [1], {1})

    def test_adjacent_year_sign_is_not_borrowed(self):
        physical = {(2, 0): {"text": "Net income", "columns": [0], "row": 2, "physical_cell": 0},
                    (2, 1): {"text": "(123", "columns": [1], "row": 2, "physical_cell": 1},
                    (2, 2): {"text": ")", "columns": [2], "row": 2, "physical_cell": 2}}
        with self.assertRaisesRegex(m.FieldGap, "crosses_year"):
            m._value(physical, physical, 2, [1], {1})

    def test_duplicate_date_headers_are_not_collapsed(self):
        raw, c = fixture()
        # An extra physical header row with identical fiscal-date anchors.
        needle = tr(td("Net revenue"), td("777", 16))
        row = tr(td(), td("Dec 25, 2021", 3), td(), td("Dec 26, 2020", 3), td(),
                 td("Dec 25, 2021", 3), td(), td("Dec 26, 2020", 3), td())
        raw, c = changed(raw, c, needle, row+needle)
        c["context_summary"]["net_income_locator"]["row"] = 5
        self.assertIsNone(m.extract_document(raw, c)["fields"]["annual_net_income"])

    def test_invalid_dates_preserve_event_as_all_null_sentinel(self):
        for mutate in (
            lambda c: c.update(available_session=c["publication_date"]),
            lambda c: c["context_summary"].update(comparison_assets_date=None),
            lambda c: c["context_summary"]["current_annual_period"].update(end="2021-02-30"),
            lambda c: c["context_summary"]["current_annual_period"].update(candidate_start_from_prior_end_plus_one="2021-01-01"),
            lambda c: c["context_summary"]["current_annual_period"].update(start="2020-12-27"),
        ):
            raw, c = fixture()
            mutate(c)
            out = m.extract_document(raw, c)
            self.assertEqual(out["event_id"], c["event_id"])
            self.assertTrue(all(v is None for v in out["fields"].values()))
            self.assertEqual(len(out["gaps"]), 3)

    def test_duration_outside_annual_bounds_is_sentinel(self):
        raw, c = fixture()
        c["context_summary"]["comparison_assets_date"] = "2020-01-01"
        c["context_summary"]["current_annual_period"]["candidate_start_from_prior_end_plus_one"] = "2020-01-02"
        self.assertFalse(m.extract_document(raw, c)["complete_required_fields"])

    def test_malformed_period_object_is_preserved_as_sentinel(self):
        raw, c = fixture()
        c["context_summary"]["current_annual_period"] = "not an object"
        out = m.extract_document(raw, c)
        self.assertEqual(len(out["gaps"]), 3)
        self.assertIn("invalid_current_annual_period", out["context_holds"])

    def test_malformed_locator_is_field_gap_not_unhandled_exception(self):
        raw, c = fixture()
        c["context_summary"]["net_income_locator"] = []
        out = m.extract_document(raw, c)
        self.assertIsNone(out["fields"]["annual_net_income"])
        self.assertEqual(out["gaps"][0]["reason"], "invalid_statement_locator")
        self.assertIsNotNone(out["fields"]["current_total_assets"])

    def test_boundary_is_explicit_assumption_not_certification(self):
        out = self.out()
        self.assertEqual(out["periods"]["current"], {"period_start": "2020-12-27", "period_end": "2021-12-25", "days_inclusive": 364})
        self.assertIsNone(out["periods"]["comparison"]["period_start"])
        self.assertFalse(out["periods"]["start_assumption_certified"])
        self.assertIn(m.BOUNDARY_HOLD, out["context_holds"])
        self.assertIn("prior_source_hold", out["context_holds"])

    def test_prior_field_never_backdated(self):
        out = self.out()
        self.assertTrue(all(c["available_from"] == "2022-01-26" for c in out["fields"].values()))
        self.assertEqual(out["fields"]["prior_total_assets"]["as_of"], "2020-12-26")
        self.assertEqual(out["fields"]["annual_net_income"]["as_of"], "2021-12-25")

    def test_negative_or_zero_assets_preserved_with_hold(self):
        for val in ("0", "-10"):
            out = self.out(prior_assets=val)
            self.assertEqual(out["fields"]["prior_total_assets"]["value_decimal"], val)
            self.assertIn("prior_total_assets_nonpositive", out["context_holds"])

    def test_no_admission_or_iso_currency_upgrade(self):
        out = self.out()
        for key in ("fact_admitted", "model_admitted", "strategy_admitted",
                    "first_public_version_verified", "complete_correction_history_verified"):
            self.assertIs(out[key], False)
        for cell in out["fields"].values():
            self.assertIs(cell["fact_admitted"], False)
            self.assertIsNone(cell["currency_iso"])
        self.assertEqual(out["accounting_review_flags"], ["synthetic accounting flag"])

    def test_pure_no_io_and_context_unchanged(self):
        raw, c = fixture()
        before = copy.deepcopy(c)
        with patch("builtins.open", side_effect=AssertionError("No file I/O")):
            m.extract_document(raw, c)
        self.assertEqual(c, before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
