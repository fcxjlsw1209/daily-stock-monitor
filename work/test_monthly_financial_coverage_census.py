import unittest
from monthly_financial_coverage_census import (
    date_lead_covers, full_body_claim, latest_annual, scan_body, universe_rows)


class CensusTests(unittest.TestCase):
    def test_universe_is_before_ranking(self):
        meta = dict(dates=["2020-01-31", "2020-02-28"],
                    symbols=[["A", "B", "ETF"], ["C", "A", "ETF"]], etfs=["ETF"])
        self.assertEqual(universe_rows(meta),
                         {"A": meta["dates"], "B": ["2020-01-31"], "C": ["2020-02-28"]})

    def test_duplicate_candidate_rejected(self):
        with self.assertRaises(ValueError):
            universe_rows(dict(dates=["2020-01-31"], symbols=[["A", "A"]], etfs=[]))

    def test_missing_etf_rejected(self):
        with self.assertRaises(ValueError):
            universe_rows(dict(dates=["2020-01-31"], symbols=[["A"]], etfs=["ETF"]))

    def test_future_release_not_coverage(self):
        row = dict(status="ready", fiscal_quarter=4, available_session="2020-02-03",
                   official_actual_period_end="2019-12-31")
        self.assertFalse(date_lead_covers(row, "2020-01-31"))
        self.assertTrue(date_lead_covers(row, "2020-02-03"))

    def test_invalid_or_unknown_clock_not_coverage(self):
        row = dict(status="held", fiscal_quarter=4, available_session="2020-01-15",
                   official_actual_period_end="2019-12-31")
        self.assertFalse(date_lead_covers(row, "2020-01-31"))
        row.update(status="ready", available_session="unknown")
        self.assertFalse(date_lead_covers(row, "2020-01-31"))

    def test_stale_and_nonannual_not_coverage(self):
        row = dict(status="ready", fiscal_quarter=4, available_session="2018-01-01",
                   official_actual_period_end="2017-12-31")
        self.assertFalse(date_lead_covers(row, "2020-01-31"))
        row.update(fiscal_quarter=3, available_session="2020-01-01")
        self.assertFalse(date_lead_covers(row, "2020-01-31"))

    def test_staleness_uses_period_end_not_late_publication(self):
        row = dict(status="ready", fiscal_quarter=4, available_session="2020-01-15",
                   official_actual_period_end="2017-12-31")
        self.assertFalse(date_lead_covers(row, "2020-01-31"))

    def test_pdf_not_claimed_as_scanned(self):
        got = scan_body(b"%PDF-1.4 Total assets statements of cash flows")
        self.assertFalse(got["text_scanned"])
        self.assertEqual(got["labels"], {})

    def test_index_and_observation_not_full_body(self):
        for kind in ("issuer_original_release_public_index_excerpt",
                     "public_index_observation_of_original_issuer_release",
                     "issuer_original_release_web_pdf_text_observation"):
            self.assertFalse(full_body_claim(kind))
        self.assertTrue(full_body_claim("issuer_original_release_full_bytes"))

    def test_latest_invalid_does_not_fall_back(self):
        old = dict(available_session="2019-01-01", status="ready")
        new = dict(available_session="2020-01-01", status="held")
        self.assertIs(latest_annual([old, new], "2020-01-31")[0], new)

    def test_latest_tie_is_ambiguous(self):
        row = dict(available_session="2020-01-01")
        latest, status, unknown = latest_annual([row, dict(row)], "2020-01-31")
        self.assertIsNone(latest)
        self.assertEqual(status, "ambiguous_latest_annual_event")
        self.assertEqual(unknown, 0)

    def test_unknown_clock_retained(self):
        latest, status, unknown = latest_annual([{}], "2020-01-31")
        self.assertIsNone(latest)
        self.assertEqual(unknown, 1)

    def test_unknown_clock_blocks_old_but_not_if_publication_future(self):
        old = dict(available_session="2019-01-01", publication_date="2018-12-31")
        unknown = dict(publication_date="2020-01-15")
        latest, reason, count = latest_annual([old, unknown], "2020-01-31")
        self.assertIsNone(latest)
        self.assertEqual(reason, "unknown_clock_barrier")
        unknown["publication_date"] = "2020-02-15"
        self.assertIs(latest_annual([old, unknown], "2020-01-31")[0], old)

    def test_script_and_style_do_not_supply_evidence(self):
        got = scan_body(b"<script>Total assets</script><style>Net income</style><p>Hello</p>")
        self.assertFalse(any(got["labels"].values()))

    def test_presence_checks_not_value_extraction(self):
        got = scan_body(b"""<h2>Consolidated Balance Sheets</h2>
        <h2>Consolidated statements of cash flows</h2><table>
        <tr><td>Total assets</td><td>10,000</td></tr>
        <tr><td>Net income</td><td>1,000</td></tr>
        <tr><td>Net cash provided by operating activities</td><td>900</td></tr>
        <tr><td>Stock-based compensation</td><td>100</td></tr></table>
        <p>Years ended December 31</p>""")
        self.assertTrue(all(got["labels"].values()))
        self.assertNotIn("100", str(got))
        self.assertIn("labels_only", got["status"])


if __name__ == "__main__":
    unittest.main()
