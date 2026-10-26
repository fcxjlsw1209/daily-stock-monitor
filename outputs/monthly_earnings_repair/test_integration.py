"""Offline integration checks; temporary fixtures only, no network requests."""
import contextlib
import datetime as dt
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import exchange_calendars as xc
import pandas as pd

P = Path(__file__).resolve().parent
sys.path.insert(0, str(P))


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = load("earnings_build_offline_test", P / "build.py")
msft = load("earnings_msft_offline_test", P / "microsoft/collect.py")
sec = load("earnings_sec_offline_test", P / "sec_inventory.py")


class MicrosoftParseTests(unittest.TestCase):
    def test_dateline_whitespace_and_real_fiscal_identity(self):
        body = """<h1>Earnings Release FY24 Q3</h1>
        <p>REDMOND,\n Wash. — April   25, 2024 — Microsoft announced results
        for the quarter ended\n March 31, 2024.</p>
        <p>Conference call April 25, 2024 at 2:30 PM.</p>"""
        result = msft.parse(body, 2024, 3, "https://example.com/verified-release", Path("fixture.html"), "abc")
        self.assertEqual(result["release_date"], "2024-04-25")
        self.assertEqual(result["fiscal_period_end_actual"], "2024-03-31")
        self.assertIsNone(result["publication_timestamp"])
        self.assertEqual(result["precision"], "date_only")
        self.assertFalse(result["conference_call_time_used"])

    def test_wrong_title_and_wrong_period_rejected_separately(self):
        wrong_title = "Earnings Release FY24 Q2 REDMOND, Wash. — October 24, 2023 — quarter ended September 30, 2023."
        with self.assertRaises(ValueError):
            msft.parse(wrong_title, 2024, 1, "https://example.com/release", Path("fixture"), "abc")
        # A comparison-period phrase or redirected page must not be accepted
        # merely because its visible FY/Q title matches the expected template.
        wrong_period = "Earnings Release FY24 Q1 REDMOND, Wash. — October 24, 2023 — quarter ended June 30, 2023."
        with self.assertRaises(ValueError):
            msft.parse(wrong_period, 2024, 1, "https://example.com/release", Path("fixture"), "abc")

    def test_call_only_and_page_metadata_do_not_supply_dateline(self):
        body = '<h1>Earnings Release FY24 Q3</h1><meta name="dateModified" content="2024-04-25">Conference call April 25, 2024. Quarter ended March 31, 2024.'
        with self.assertRaises(ValueError):
            msft.parse(body, 2024, 3, "https://example.com/release", Path("fixture"), "abc")

    def test_retry_after_seconds_http_date_and_cross_day_stop(self):
        now = dt.datetime(2026, 10, 12, 22, tzinfo=dt.timezone.utc)
        until = msft.retry_not_before("172800", now)
        self.assertEqual(until, "2026-10-14T22:00:00+00:00")
        self.assertEqual(msft.retry_not_before("Wed, 14 Oct 2026 22:00:00 GMT", now), until)
        state = dict(date="2026-10-12", retry_not_before=until)
        self.assertTrue(msft.is_stopped(state, now + dt.timedelta(days=1)))
        self.assertFalse(msft.is_stopped(state, now + dt.timedelta(days=3)))
        utc_next_la_today = dt.datetime(2026, 10, 13, 1, tzinfo=dt.timezone.utc)
        self.assertTrue(msft.is_stopped(dict(date="2026-10-12"), utc_next_la_today))
        self.assertTrue(msft.is_stopped(dict(date="2026-10-13"), utc_next_la_today))
        self.assertIsNone(msft.retry_not_before("invalid", now))


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calendar = xc.get_calendar("XNYS", start="2010-01-01", end="2030-12-31")

    def make_row(self, folder, day="2024-04-25", period="2024-03-31"):
        evidence = folder / "release.html"
        evidence.write_text("offline issuer publication evidence fixture")
        return dict(ticker="MSFT", release_date=day, publication_timestamp=None,
            precision="date_only", publication_timezone="America/Los_Angeles",
            fiscal_period_end_actual=period, fiscal_year=2024, fiscal_quarter=3,
            source_url="https://example.com/issuer/release", evidence_path=str(evidence),
            sha256=hashlib.sha256(evidence.read_bytes()).hexdigest())

    def test_adapter_source_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp);row = self.make_row(folder)
            row["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "checksum"):
                build.adapt(row, folder, self.calendar, {})

    def test_correction_is_hash_bound_and_does_not_mutate_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);row=self.make_row(folder,period='2024-03-30')
            src=folder/'events.json';src.write_text(json.dumps([row]));original=src.read_bytes()
            replacement=dict(row,fiscal_period_end_actual='2024-03-31')
            cf=folder/'CORRECTIONS.json'
            correction=dict(ticker=row['ticker'],release_date=row['release_date'],fiscal_year=row['fiscal_year'],fiscal_quarter=row['fiscal_quarter'],
                old_end='2024-03-30',verified_end='2024-03-31',reason='Explicit issuer fiscal table',
                independent_evidence=dict(evidence_path=row['evidence_path'],sha256=row['sha256']),corrected_event=replacement)
            manifest=dict(original_input=str(src),original_input_sha256=build.sha(src),corrections=[correction]);cf.write_text(json.dumps(manifest))
            result=build.apply_corrections([row],src,[(cf,manifest)])
            self.assertEqual(src.read_bytes(),original)
            self.assertEqual(row['fiscal_period_end_actual'],'2024-03-30')
            self.assertEqual(result[0]['fiscal_period_end_actual'],'2024-03-31')
            event=build.adapt(result[0],folder,self.calendar,{})
            self.assertEqual(event['source_correction']['old_fiscal_period_end'],'2024-03-30')
            with self.assertRaisesRegex(ValueError,'source checksum'):
                build.apply_corrections([row],src,[(cf,dict(manifest,original_input_sha256='0'*64))])
            changed=dict(correction,old_end='2024-03-29')
            with self.assertRaisesRegex(ValueError,'prior value'):
                build.apply_corrections([row],src,[(cf,dict(manifest,corrections=[changed]))])

    def test_secondary_publication_evidence_must_match_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); row = self.make_row(folder)
            extra = folder/'official_event_calendar.html'
            extra.write_text('Official event date linked to this exact earnings PDF')
            row['additional_evidence'] = [dict(evidence_path=str(extra), sha256='wrong', role='date')]
            with self.assertRaisesRegex(ValueError, 'Additional publication evidence'):
                build.adapt(row, folder, self.calendar, {})
            row['additional_evidence'][0]['sha256'] = hashlib.sha256(extra.read_bytes()).hexdigest()
            result = build.adapt(row, folder, self.calendar, {})
            self.assertEqual(result['additional_evidence'], row['additional_evidence'])

    def test_exact_timestamp_requires_separate_reference_and_matching_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp);row = self.make_row(folder)
            row.update(publication_timestamp="2024-04-25T13:00:00-07:00", precision="exact")
            with self.assertRaisesRegex(ValueError, "timestamp evidence"):
                build.adapt(row, folder, self.calendar, {})
            timestamp_file = folder / "timestamp_evidence.html"
            timestamp_file.write_text("Original release publication: April25,2024 13:00PDT")
            row["timestamp_evidence"] = dict(evidence_path=str(timestamp_file), sha256="0" * 64)
            with self.assertRaisesRegex(ValueError, "timestamp evidence checksum"):
                build.adapt(row, folder, self.calendar, {})
            row["timestamp_evidence"]["sha256"] = hashlib.sha256(timestamp_file.read_bytes()).hexdigest()
            result = build.adapt(row, folder, self.calendar, {})
            self.assertEqual(result["status"], "ready")
            self.assertEqual(result["precision"], "exact")
            self.assertEqual(result["known_by_at"], "2024-04-25T20:00:00+00:00")
            self.assertEqual(result["timestamp_evidence"], row["timestamp_evidence"])

    def test_missing_source_timezone_remains_pending_not_headquarters_assumption(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp);row = self.make_row(folder)
            row.pop("publication_timezone")
            event = build.adapt(row, folder, self.calendar, {})
            self.assertEqual(event["status"], "pending_review")
            self.assertIsNone(event["known_by_at"])

    def test_adapter_retains_original_announcing_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); row = self.make_row(folder)
            row.update(ticker='research_equity_FB', historical_ticker='FB',
                issuer_name_at_release='Facebook, Inc.', issuer_identity='Same Class A issuer')
            event = build.adapt(row, folder, self.calendar, {})
            self.assertEqual(event['symbol'], 'research_equity_FB')
            self.assertEqual(event['historical_ticker'], 'FB')
            self.assertEqual(event['issuer_name_at_release'], 'Facebook, Inc.')
            self.assertEqual(build.PRICE_SYMBOLS[event['symbol']], 'META')

    def test_adapter_keeps_original_evidence_limitations(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); row = self.make_row(folder)
            row.update(evidence_limitation='Indexed excerpt, not original PDF bytes',
                publication_version='corrected_current_archive', first_public_time_verified=False,
                period_basis='Separately verified official 10-Q cover')
            event = build.adapt(row, folder, self.calendar, {})
            for key in ('evidence_limitation', 'publication_version', 'first_public_time_verified', 'period_basis'):
                self.assertEqual(event[key], row[key])

    def fixture_tree(self, root, duplicate=False, pending_count=0):
        outputs = root / "outputs"; repair = outputs / "monthly_earnings_repair"
        source = repair / "microsoft";source.mkdir(parents=True)
        row = self.make_row(source)
        if duplicate:
            events = [row, dict(row)]
        else:
            events = []
            for i in range(pending_count):
                year, quarter = 2015 + i // 4, 1 + i % 4
                events.append(dict(row, release_date="%04d-%02d-15" % (year, quarter * 3),
                    fiscal_year=year, fiscal_quarter=quarter, fiscal_period_end_actual=None))
        (source / "events.json").write_text(json.dumps(events))
        provider = outputs / "monthly_event_confirmation/raw";provider.mkdir(parents=True)
        (provider / "AAPL.json").write_text(json.dumps(dict(quarterlyEarnings=[])))
        metadata = outputs / "monthly_architecture_restart";metadata.mkdir(parents=True)
        (metadata / "META.json").write_text(json.dumps(dict(monthly=dict(dates=["2026-09-24"],
            symbols=[["MSFT", "UNKN", "QQQ"]], etfs=["QQQ"]))))
        (repair / "PROTECTED.json").write_text("{}")
        (repair / "SEC_REVIEW_QUEUE.json").write_text(json.dumps(dict(candidate_filings=0, issuers_with_candidates=0)))
        (repair / "PLAN.json").write_text(json.dumps(dict(source_stops=[])))
        return outputs, repair

    def test_conflicting_duplicate_release_dates_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            outputs, repair = self.fixture_tree(Path(tmp), duplicate=True)
            with mock.patch.object(build, "O", outputs), mock.patch.object(build, "P", repair), mock.patch.object(build, "SOURCES", ["microsoft"]):
                with self.assertRaisesRegex(ValueError, "duplicate"):
                    build.main()

    def test_duplicate_vendor_release_dates_require_reconciliation(self):
        with tempfile.TemporaryDirectory() as tmp:
            outputs, repair = self.fixture_tree(Path(tmp))
            provider_file = outputs / "monthly_event_confirmation/raw/AAPL.json"
            provider_file.write_text(json.dumps(dict(quarterlyEarnings=[
                dict(reportedDate="2024-05-02", fiscalDateEnding="2024-03-31"),
                dict(reportedDate="2024-05-02", fiscalDateEnding="2024-03-30")])) )
            with mock.patch.object(build, "O", outputs), mock.patch.object(build, "P", repair), mock.patch.object(build, "SOURCES", ["microsoft"]):
                with self.assertRaisesRegex(ValueError, "Duplicate provider"):
                    build.main()

    def test_pending_metadata_not_counted_as_verified_history_and_etf_is_na(self):
        with tempfile.TemporaryDirectory() as tmp:
            outputs, repair = self.fixture_tree(Path(tmp), pending_count=40)
            with mock.patch.object(build, "O", outputs), mock.patch.object(build, "P", repair), mock.patch.object(build, "SOURCES", ["microsoft"]), contextlib.redirect_stdout(io.StringIO()):
                build.main()
            coverage = {r["symbol"]: r for r in json.loads((repair / "COVERAGE.json").read_text())}
            self.assertEqual(coverage["MSFT"]["verified_events"], 0)
            self.assertNotEqual(coverage["MSFT"]["status"], "verified_history_collected")
            self.assertEqual(coverage["UNKN"]["verified_events"], 0)
            states = {r["symbol"]: r for r in json.loads((repair / "MONTHLY_AVAILABILITY.json").read_text())}
            self.assertEqual(states["QQQ"]["status"], "not_applicable_etf")
            self.assertEqual(states["UNKN"]["status"], "missing_event_history")
            self.assertIsNone(states["UNKN"]["event_id"])
            self.assertEqual(states["MSFT"]["status"], "missing_event_history")

    def test_quarter_gaps_and_duplicates_are_explicit(self):
        rows = [dict(fiscal_year=2023, fiscal_quarter=q) for q in [1, 3, 3, 4]]
        result = build.quarter_gaps(rows)
        self.assertEqual(result["duplicates"], 1)
        self.assertEqual(result["missing_between_observed_endpoints"], ["2023Q2"])

    def monthly_fixture(self):
        dates = self.calendar.sessions_in_range('2024-04-01', '2024-05-31')
        prices = pd.DataFrame({'close': [100. + i for i in range(len(dates))],
                               'volume': 1000.}, index=dates)
        event = build.normalize_event(dict(symbol='MSFT', asset_type='equity',
            official=dict(actual_period_end='2024-03-31', publication_date='2024-04-29',
                publication_timezone='America/Los_Angeles', publication_kind='release',
                verified=True, source_url='https://example.com/release')), self.calendar)
        event['event_id'] = 'MSFT_2024-04-29'
        membership = dict(signal_date='2024-04-30', symbol='MSFT',
            status='verified_event_available', event_id=event['event_id'])
        return event, membership, prices

    def test_monthly_immature_event_does_not_use_future_bars_or_old_event(self):
        event, membership, prices = self.monthly_fixture()
        old_event = dict(event, event_id='MSFT_earlier', known_by_at='2024-04-01T20:00:00+00:00')
        a = build.monthly_reaction_panel([membership], [old_event, event],
            {'MSFT': prices}, self.calendar)[0]
        altered = prices.copy(); altered.loc[altered.index > '2024-04-30', 'close'] = 99999.
        b = build.monthly_reaction_panel([membership], [old_event, event],
            {'MSFT': altered}, self.calendar)[0]
        self.assertEqual(a, b)
        self.assertEqual(a['event_id'], event['event_id'])
        for result in a['reactions'].values():
            self.assertEqual(result['status'], 'waiting_for_closed_window')
            self.assertIsNone(result['post_known_close_to_close_return'])

    def test_monthly_unknown_stale_and_etf_have_no_numeric_reaction(self):
        rows = [dict(signal_date='2024-04-30', symbol='UNKNOWN', event_id=None,
            status=status) for status in ('missing_event_history', 'stale_event', 'not_applicable_etf')]
        result = build.monthly_reaction_panel(rows, [], {}, self.calendar)
        self.assertEqual([r['status'] for r in result], [r['status'] for r in rows])
        self.assertTrue(all(r['reactions'] == {} for r in result))

    def test_monthly_mature_window_and_missing_interior_bar(self):
        event, membership, prices = self.monthly_fixture()
        membership['signal_date'] = '2024-05-31'
        a = build.monthly_reaction_panel([membership], [event], {'MSFT': prices}, self.calendar)[0]
        self.assertTrue(all(v['status'] == 'ready' for v in a['reactions'].values()))
        self.assertTrue(all(pd.Timestamp(v['end_close_at']) <= pd.Timestamp(a['signal_asof'])
                            for v in a['reactions'].values()))
        prices = prices.drop(pd.Timestamp('2024-05-01'))
        b = build.monthly_reaction_panel([membership], [event], {'MSFT': prices}, self.calendar)[0]
        self.assertTrue(all(v['status'] == 'missing_price_data' for v in b['reactions'].values()))
        self.assertTrue(all(v['post_known_close_to_close_return'] is None for v in b['reactions'].values()))

    def test_monthly_event_identity_or_future_availability_rejected(self):
        event, membership, prices = self.monthly_fixture()
        membership['signal_date'] = '2024-04-26'
        with self.assertRaisesRegex(ValueError, 'identity/availability'):
            build.monthly_reaction_panel([membership], [event], {'MSFT': prices}, self.calendar)
        membership.update(signal_date='2024-05-31', symbol='OTHER')
        with self.assertRaisesRegex(ValueError, 'identity/availability'):
            build.monthly_reaction_panel([membership], [event], {'OTHER': prices}, self.calendar)

    def test_sec_provenance_hash_mismatch_never_admitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "0000000001_submissions.json"
            f.write_text("{}")
            f.with_suffix(".provenance.json").write_text(json.dumps(dict(
                source="https://data.sec.gov/submissions/CIK0000000001.json", sha256="0" * 64)))
            with self.assertRaisesRegex(ValueError, "checksum"):
                sec.source(f)


if __name__ == "__main__":
    unittest.main()
