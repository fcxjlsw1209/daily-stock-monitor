"""Synthetic input-adapter checks only; never call it with real research inputs."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import monthly_peer_earnings139_inputs as adapter


class Calendar:
    sessions = pd.bdate_range("2020-01-01", periods=40)
    def session_close(self, day):
        return pd.Timestamp(day).tz_localize("UTC") + pd.Timedelta(hours=16)


def fixture():
    calendar = Calendar()
    day = str(calendar.sessions[-1].date())
    close = calendar.session_close(day)
    old = calendar.session_close(calendar.sessions[2]).isoformat()
    new = (close - pd.Timedelta(hours=1)).isoformat()
    meta = dict(dates=[day], symbols=[["A", "B", "ETF"]], etfs=["ETF"])
    events = [dict(event_id="a_old", symbol="A", status="ready", known_by_at=old),
              dict(event_id="a_new", symbol="A", status="ready", known_by_at=new),
              dict(event_id="b_old", symbol="B", status="ready", known_by_at=old)]
    rows = []
    for symbol, event_id, known in (("A", "a_new", new), ("B", "b_old", old), ("ETF", None, None)):
        reactions = {} if symbol == "ETF" else {"10": dict(known_by_at=known,
            status="immature" if symbol == "A" else "ready",
            anchor_session=day if symbol == "A" else str(calendar.sessions[2].date()))}
        rows.append(dict(signal_date=day, symbol=symbol, signal_asof=close.isoformat(),
                         status="not_applicable_etf" if symbol == "ETF" else "verified_event_available",
                         event_id=event_id, reactions=reactions))
    return meta, deepcopy(rows), events, {"rows": rows}, calendar


class LatestKnownTests(unittest.TestCase):
    def test_new_immature_event_is_retained(self):
        meta, availability, events, document, calendar = fixture()
        got = adapter.validate_latest(meta, availability, events, document, calendar)
        self.assertEqual(got[meta["dates"][0], "A"]["event_id"], "a_new")
        self.assertEqual(got[meta["dates"][0], "A"]["reactions"]["10"]["status"], "immature")

    def test_older_mature_event_cannot_replace_latest_known(self):
        meta, availability, events, document, calendar = fixture()
        for rows in (availability, document["rows"]):
            rows[0]["event_id"] = "a_old"
            rows[0]["reactions"]["10"].update(status="ready", known_by_at=events[0]["known_by_at"])
        with self.assertRaisesRegex(ValueError, "latest known"):
            adapter.validate_latest(meta, availability, events, document, calendar)

    def test_future_event_is_not_known(self):
        meta, availability, events, document, calendar = fixture()
        events.append(dict(event_id="future", symbol="A", status="ready",
                           known_by_at=(calendar.session_close(meta["dates"][0])+pd.Timedelta(seconds=1)).isoformat()))
        adapter.validate_latest(meta, availability, events, document, calendar)

    def test_nonready_event_cannot_carry_knowledge_timestamp(self):
        data = fixture()
        data[2][1]["status"] = "pending"
        with self.assertRaisesRegex(ValueError, "Unready event has a knowledge timestamp"):
            adapter.validate_latest(*data)
        data = fixture()
        data[2].append(dict(event_id="pending", symbol="A", status="pending", known_by_at=None))
        adapter.validate_latest(*data)

    def test_wrong_or_naive_signal_close_is_rejected(self):
        for value in ("2020-02-25T15:59:59+00:00", "2020-02-25T16:00:00"):
            data = fixture()
            data[3]["rows"][0]["signal_asof"] = value
            with self.assertRaisesRegex(ValueError, "signal close"):
                adapter.validate_latest(*data)

    def test_etf_cannot_have_event_or_reaction(self):
        for field, value in (("event_id", "a_new"), ("reactions", {"10": {}}), ("status", "verified_event_available")):
            meta, availability, events, document, calendar = fixture()
            document["rows"][2][field] = value
            availability[2][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "ETF"):
                adapter.validate_latest(meta, availability, events, document, calendar)

    def test_duplicate_candidates_rows_availability_and_event_ids_fail(self):
        for which in ("meta", "rows", "availability", "events"):
            meta, availability, events, document, calendar = fixture()
            if which == "meta": meta["symbols"][0].append("A")
            if which == "rows": document["rows"].append(deepcopy(document["rows"][0]))
            if which == "availability": availability.append(deepcopy(availability[0]))
            if which == "events": events.append(deepcopy(events[0]))
            with self.subTest(which=which), self.assertRaisesRegex(ValueError, "Duplicate"):
                adapter.validate_latest(meta, availability, events, document, calendar)

    def test_ambiguous_latest_timestamp_and_naive_known_time_fail(self):
        data = fixture()
        extra = dict(data[2][1], event_id="same_timestamp")
        data[2].append(extra)
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            adapter.validate_latest(*data)
        data = fixture()
        data[2][1]["known_by_at"] = "2020-02-25T15:00:00"
        with self.assertRaisesRegex(ValueError, "timezone"):
            adapter.validate_latest(*data)

    def test_reaction_knowledge_must_match_latest_event(self):
        data = fixture()
        data[3]["rows"][0]["reactions"]["10"]["known_by_at"] = data[2][0]["known_by_at"]
        with self.assertRaisesRegex(ValueError, "knowledge mismatch"):
            adapter.validate_latest(*data)


class IssuerTests(unittest.TestCase):
    def test_aliases_and_shared_issuer_identity(self):
        rows = [dict(symbol="META", url="issuer/meta"), dict(symbol="BRK-B", url="issuer/berkshire"),
                dict(symbol="GOOG", url="issuer/alphabet"), dict(symbol="GOOGL", url="issuer/alphabet")]
        result = adapter.issuer_map(rows)
        self.assertEqual(result["research_equity_FB"], "issuer/meta")
        self.assertEqual(result["BRK.B"], "issuer/berkshire")
        self.assertEqual(result["GOOG"], result["GOOGL"])

    def test_conflicting_alias_identity_and_invalid_metadata_fail(self):
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            adapter.issuer_map([dict(symbol="META", url="one"), dict(symbol="research_equity_FB", url="two")])
        for row in (dict(symbol="A", url=""), dict(symbol=None, url="issuer"), dict(symbol="A", url=None)):
            with self.assertRaisesRegex(ValueError, "Invalid"):
                adapter.issuer_map([row])


class LoadTests(unittest.TestCase):
    def setUp(self):
        self.meta, self.availability, self.events, self.document, self.calendar = fixture()
        self.official = [str(day.date()) for day in self.calendar.sessions]
        self.event = dict(event10=np.array([[np.nan, .2, np.nan]]),
                          matched_recent10=np.array([[np.nan, .1, np.nan]]),
                          valid10=np.array([[False, True, False]]))
        self.data = {
            adapter.REPAIR + "/SCOPED_READINESS.json": dict(scoped_post_known_drift_data_ready=True,
                unresolved_count=0, unresolved=[], invalid_candidate_price_windows=[], protected_files_changed=0,
                inputs={"EVENTS.json": "digest"}),
            adapter.REPAIR + "/MONTHLY_PANEL_AUDIT.json": dict(input_sha256={"MONTHLY_FEATURES.json": "digest"}),
            adapter.REPAIR + "/MONTHLY_FEATURES.json": self.document,
            adapter.REPAIR + "/MONTHLY_AVAILABILITY.json": self.availability,
            adapter.REPAIR + "/EVENTS.json": self.events,
            adapter.CAPS: dict(records=[dict(symbol="A", url="issuer/A"), dict(symbol="B", url="issuer/B")]),
        }
        self.reads = []
        self.reader = SimpleNamespace(json=self.read, sha=lambda path: "digest", path=lambda path: Path("/synthetic")/path)
        self.validator = SimpleNamespace(_calendar=lambda: self.calendar,
                                         build_features=lambda *args: (self.event, {"synthetic": True}))
        spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda module: None))
        for name, value in (("spec_from_file_location", spec), ("module_from_spec", self.validator)):
            patcher = patch.object(adapter.importlib.util, name, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def read(self, path):
        self.reads.append(path)
        return self.data[path]

    def load(self, official=None):
        return adapter.load("/synthetic", self.reader, self.meta, dict(adj="synthetic", sectors="synthetic"),
                            self.official if official is None else official,
                            lambda *args: {"A": "XLK", "B": "XLK", "ETF": "BAD_ETF_MAPPING"})

    def test_preserves_frozen_arrays_latest_identity_masks_anchors_and_issuer(self):
        result, provenance = self.load()
        for name in ("event10", "matched_recent10", "valid10"):
            self.assertIs(result[name], self.event[name])
        self.assertEqual(result["event_ids"].tolist(), [["a_new", "b_old", None]])
        self.assertIsNone(result["anchor_sessions"][0, 0])
        self.assertEqual(result["anchor_sessions"][0, 1], str(self.calendar.sessions[2].date()))
        self.assertEqual(result["sector_benchmarks"].tolist(), [["XLK", "XLK", None]])
        self.assertEqual(result["issuer_keys"].tolist(), [["issuer/A", "issuer/B", None]])
        self.assertFalse(provenance["raw_event_prices_reconstructed"])
        self.assertTrue(provenance["latest_known_event_independently_checked"])

    def test_unknown_issuer_stays_missing_with_provenance(self):
        self.data[adapter.CAPS]["records"] = [dict(symbol="A", url="issuer/A")]
        result, provenance = self.load()
        self.assertIsNone(result["issuer_keys"][0, 1])
        self.assertEqual(provenance["unknown_issuer_symbols"], ["B"])

    def test_official_calendar_and_source_binding_mismatch_fail(self):
        with self.assertRaisesRegex(ValueError, "calendar differs"):
            self.load(self.official[:5] + self.official[6:])
        self.reader.sha = lambda path: "wrong"
        with self.assertRaisesRegex(ValueError, "Readiness binding"):
            self.load()

    def test_failed_readiness_stops_before_feature_inputs(self):
        self.data[adapter.REPAIR + "/SCOPED_READINESS.json"]["protected_files_changed"] = 1
        with self.assertRaisesRegex(ValueError, "readiness failed"):
            self.load()
        self.assertEqual(self.reads, [adapter.REPAIR + "/SCOPED_READINESS.json"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
