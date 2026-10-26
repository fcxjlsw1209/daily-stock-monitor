import json
import unittest

from collect import parse_html


class ParseTests(unittest.TestCase):
    def body(self, pub="2016-01-26Z", period="December 26, 2015", fiscal=True):
        js = {"@type": "NewsArticle", "datePublished": pub,
              "dateModified": "2026-10-12T17:00:00Z", "headline": "Earnings"}
        statement = "fiscal 2016 first quarter ended " + period if fiscal else "estimated future financial results"
        return ('<script type="application/ld+json">' + json.dumps(js) + '</script><p>CUPERTINO, California — ' + statement +
                '</p><p>conference call 2:00 p.m. PDT</p>').encode()

    def target(self):
        return {"reported_date": "2016-01-26", "fiscal_period_end": "2015-12-31",
                "report_time_raw": "post-market", "in_study_window": True}

    def parse(self, **kwargs):
        return parse_html(self.body(**kwargs), self.target(), "https://www.apple.com/a", "raw/a.html")

    def test_actual_fiscal_date_not_provider_calendar_label(self):
        e = self.parse()
        self.assertEqual(e["fiscal_period_end_actual"], "2015-12-26")
        self.assertEqual(e["vendor_fiscal_period_end_label"], "2015-12-31")
        self.assertFalse(e["vendor_period_label_matches_actual"])
        self.assertEqual((e["fiscal_year"], e["fiscal_quarter"]), (2016, 1))

    def test_date_only_does_not_manufacture_midnight_or_call_time(self):
        e = self.parse()
        self.assertIsNone(e["publication_timestamp"])
        self.assertEqual(e["precision"], "date_only")
        self.assertFalse(e["first_public_time_verified"])
        self.assertEqual(e["publication_timezone"], "America/Los_Angeles")
        self.assertEqual(e["publication_timezone_evidence"]["observed_dateline"], "CUPERTINO, California")
        self.assertFalse(e["publication_timezone_evidence"]["cms_timezone_observed"])
        self.assertFalse(e["publication_timezone_evidence"]["exact_publication_time_inferred"])

    def test_missing_dateline_requires_explicit_timezone_review(self):
        b = self.body().replace(b"CUPERTINO, California", b"Unknown location")
        with self.assertRaisesRegex(ValueError, "timezone policy"):
            parse_html(b, self.target(), "https://www.apple.com/a", "raw/a.html")

    def test_wrong_official_date_rejected(self):
        with self.assertRaises(ValueError):
            self.parse(pub="2016-01-27Z")

    def test_timestamp_new_precision_requires_review(self):
        with self.assertRaises(ValueError):
            self.parse(pub="2016-01-26T17:00:00Z")

    def test_absent_fiscal_body_rejected(self):
        with self.assertRaises(ValueError):
            self.parse(fiscal=False)


if __name__ == "__main__":
    unittest.main()
