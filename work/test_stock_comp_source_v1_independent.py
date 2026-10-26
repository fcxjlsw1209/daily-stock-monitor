"""Synthetic-only physical-cell and provenance checks for the independent auditor."""
from copy import deepcopy
import unittest
from bs4 import BeautifulSoup

import stock_comp_source_v1_independent as m


def fixture(literal="1,234", label="Stock-based compensation expense", annual="Years Ended"):
    text = """<table>
    <tr><td colspan="3">CONSOLIDATED STATEMENTS OF CASH FLOWS</td></tr>
    <tr><td colspan="3">(in millions) $</td></tr>
    <tr><td></td><td colspan="2">%s</td></tr>
    <tr><td></td><td>December 31, 2020</td><td>December 31, 2019</td></tr>
    <tr><td>Net income</td><td>5</td><td>4</td></tr>
    <tr><td colspan="3">Adjustments to reconcile net income to net cash from operations:</td></tr>
    <tr><td>%s</td><td>%s</td><td>321</td></tr>
    <tr><td>Net cash provided by operating activities</td><td>10</td><td>9</td></tr>
    </table>""" % (annual, label, literal)
    raw = text.encode()
    _, _, physical = m.table_cells(BeautifulSoup(raw, "html.parser").table)
    shared = dict(format="html", table_index_zero_based=0, logical_columns=[1],
                  scale="1000000", currency_literal="$", period_end="2020-12-31",
                  header_literals=[annual, "December 31, 2020"])
    ni = dict(shared, row_index_zero_based=4, ni_scope="indirect_method_consolidated_start")
    cfo = dict(shared, row_index_zero_based=7)
    loc = dict(shared, row_index_zero_based=6, row_label=label,
               physical_cell_indices_zero_based=[1], label_cell_sha256=physical[(6, 0)]["markup_sha256"],
               indirect_start_row_zero_based=4, operating_total_row_zero_based=7)
    period = dict(period_start="2020-01-01", period_end="2020-12-31", days_inclusive=366)
    old = dict(fields=dict(current=dict(net_income=dict(locator=ni), operating_cash_flow=dict(locator=cfo))),
               periods=dict(current=period))
    value = dict(locator=loc, source_sha256=m.digest(raw), scale="1000000", currency_literal="$",
                 currency_iso=None, period=deepcopy(period), displayed_amount=literal,
                 value_decimal=m.money(literal), fact_admitted=False,
                 parsed_cell_sha256=[physical[(6, 1)]["markup_sha256"]])
    return raw, value, old


class IndependentHTMLTests(unittest.TestCase):
    def test_physical_cell_and_annual_header(self):
        self.assertTrue(m.audit_html(*fixture())["verified"])

    def test_negative_component_kept_signed(self):
        args = fixture("(1,234)")
        self.assertEqual(args[1]["value_decimal"], "-1234")
        self.assertTrue(m.audit_html(*args)["signed_decimal_verified"])

    def test_comma_spaces_and_unicode_minus(self):
        self.assertEqual(m.money("\u22121 , 234"), "-1234")

    def test_no_missing_or_malformed_zero_imputation(self):
        for literal in ("", "-", "—", "NaN", "1 234", "1,23", "5%", "(-2)", "$"):
            with self.subTest(literal=literal), self.assertRaises(ValueError):
                m.money(literal)

    def test_explicit_zero_is_a_real_observation(self):
        self.assertTrue(m.audit_html(*fixture("0"))["verified"])

    def test_wrong_period_column_rejected(self):
        raw, value, old = fixture()
        value["locator"]["logical_columns"] = [2]
        with self.assertRaisesRegex(ValueError, "shared anchor"):
            m.audit_html(raw, value, old)

    def test_interim_column_rejected_even_if_anchors_agree(self):
        with self.assertRaisesRegex(ValueError, "solely annual"):
            m.audit_html(*fixture(annual="Three Months Ended"))

    def test_changed_decimal_rejected(self):
        raw, value, old = fixture()
        value["value_decimal"] = "1235"
        with self.assertRaisesRegex(ValueError, "decimal mismatch"):
            m.audit_html(raw, value, old)

    def test_changed_literal_rejected(self):
        raw, value, old = fixture()
        value["displayed_amount"] = "1234"
        with self.assertRaisesRegex(ValueError, "literal mismatch"):
            m.audit_html(raw, value, old)

    def test_source_digest_rejected(self):
        raw, value, old = fixture()
        with self.assertRaisesRegex(ValueError, "source digest"):
            m.audit_html(raw + b" ", value, old)

    def test_combined_tax_and_compensation_not_substituted(self):
        with self.assertRaisesRegex(ValueError, "physical SBC label"):
            m.audit_html(*fixture(label="Stock-based compensation and other adjustments"))

    def test_new_ISO_certification_rejected(self):
        raw, value, old = fixture()
        value["currency_iso"] = "USD"
        with self.assertRaisesRegex(ValueError, "ISO"):
            m.audit_html(raw, value, old)

    def test_scale_change_rejected(self):
        raw, value, old = fixture()
        value["scale"] = "1000"
        with self.assertRaisesRegex(ValueError, "Scale"):
            m.audit_html(raw, value, old)

    def test_parent_attributed_NI_not_substituted(self):
        raw, value, old = fixture()
        old["fields"]["current"]["net_income"]["locator"]["ni_scope"] = "parent_attributed"
        with self.assertRaisesRegex(ValueError, "NI scope"):
            m.audit_html(raw, value, old)

    def test_missing_previously_reviewed_anchor_rejected(self):
        raw, value, old = fixture()
        old["fields"]["current"]["operating_cash_flow"] = None
        with self.assertRaisesRegex(ValueError, "anchors required"):
            m.audit_html(raw, value, old)

    def test_changed_physical_markup_rejected(self):
        raw, value, old = fixture()
        value["parsed_cell_sha256"] = ["a" * 64]
        with self.assertRaisesRegex(ValueError, "markup mismatch"):
            m.audit_html(raw, value, old)

    def test_premature_admission_rejected(self):
        raw, value, old = fixture()
        value["fact_admitted"] = True
        with self.assertRaisesRegex(ValueError, "prematurely"):
            m.audit_html(raw, value, old)

    def test_explicit_frozen_sha_before_any_file_read(self):
        with self.assertRaisesRegex(ValueError, "Explicit frozen"):
            m.audit("/tmp/nonexistent-source-review", None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
