"""Offline evidence admission tests. Does not request market or earnings data."""
import importlib.util
import json
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("jpmorgan_collection", Path(__file__).with_name("collect.py"))
c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)


class ParserTests(unittest.TestCase):
    def test_original_dateline_date_only_not_clock(self):
        text = "JPMORGAN CHASE REPORTS FIRST-QUARTER 2020 RESULTS. New York, April 14, 2020 - actual results. at 8:30 a.m. Eastern"
        result = c.parse_release(text, 2020, 1, [], {})
        self.assertEqual(result["release_date"], "2020-04-14")
        self.assertNotIn("publication_timestamp", result)

    def test_today_explicit_date_allowed(self):
        text = "New York, NY 10017 FIRST-QUARTER 2020 JPMorgan Chase will host a conference call today, April 14, 2020, at 8:30 a.m. ET"
        result = c.parse_release(text, 2020, 1, [], {})
        self.assertEqual(result["release_date"], "2020-04-14")

    def test_same_pdf_official_event_needs_original_today_context(self):
        text = "New York, NY 10017 FIRST-QUARTER 2020 JPMorgan Chase will host a conference call today at 8:30 a.m. ET"
        events = [dict(date="Tue Apr 14 00:00:00 EDT 2020")]
        result = c.parse_release(text, 2020, 1, events, dict(evidence_path="test.json", sha256="a"*64))
        self.assertEqual(result["release_date"], "2020-04-14")
        self.assertEqual(len(result["release_date_evidence"]), 2)
        with self.assertRaisesRegex(ValueError, "today context"):
            c.parse_release(text.replace("today", "tomorrow"), 2020, 1, events, {})

    def test_event_conflicts_remain_pending_without_explicit_body_date(self):
        text = "New York, NY 10017 FIRST-QUARTER 2020 conference call today at 8:30 a.m."
        with self.assertRaisesRegex(ValueError, "unambiguous"):
            c.parse_release(text, 2020, 1, [dict(date="Tue Apr 14 00:00:00 EDT 2020"),dict(date="Wed Apr 15 00:00:00 EDT 2020")], {})

    def test_replay_date_or_schedule_alone_cannot_supply_release_date(self):
        text = "New York, NY 10017 FIRST-QUARTER 2020. A replay will be available on April 14, 2020. Results scheduled to be released April 14, 2020."
        with self.assertRaisesRegex(ValueError, "today context"):
            c.parse_release(text, 2020, 1, [], {})

    def test_identity_timezone_and_real_period_evidence_required(self):
        with self.assertRaisesRegex(ValueError, "fiscal year"):
            c.parse_release("New York, April 14, 2020 FIRST-QUARTER 2019", 2020, 1, [], {})
        with self.assertRaisesRegex(ValueError, "timezone"):
            c.parse_release("FIRST-QUARTER 2020 will host conference call today, April 14, 2020", 2020, 1, [], {})
        self.assertIsNone(c.find_period("FIRST-QUARTER 2020", 2020, 1))
        self.assertEqual(c.find_period("three months ended March 31, 2020", 2020, 1)["date"], "2020-03-31")

    def test_cached_events_hashes_identity_and_date_invariants(self):
        file = c.HERE / "events.json"
        if not file.exists():
            self.skipTest("Collection not finished")
        events = json.loads(file.read_text())
        self.assertGreaterEqual(len(events), 40)
        keys = set()
        for row in events:
            key = (row["fiscal_year"], row["fiscal_quarter"])
            self.assertNotIn(key, keys); keys.add(key)
            self.assertEqual(row["ticker"], "JPM")
            self.assertEqual(row["precision"], "date_only")
            self.assertIsNone(row["publication_timestamp"])
            self.assertFalse(row["first_public_time_verified"])
            self.assertLess(row["fiscal_period_end_actual"], row["release_date"])
            self.assertLessEqual(row["release_date"], c.CUTOFF)
            self.assertTrue(row["source_url"].startswith(c.ORIGIN + "/"))
            self.assertEqual(c.sha((c.ROOT/row["evidence_path"]).read_bytes()), row["evidence_sha256"])
            for proof in [row["fiscal_period_evidence"], row["official_archive_evidence"]] + row["release_date_evidence"]:
                self.assertEqual(c.sha((c.ROOT/proof["evidence_path"]).read_bytes()), proof["sha256"])
            for proof in row.get("additional_evidence", []):
                self.assertTrue(proof["role"])
                self.assertEqual(c.sha((c.ROOT/proof["evidence_path"]).read_bytes()), proof["sha256"])


if __name__ == "__main__":
    unittest.main()
