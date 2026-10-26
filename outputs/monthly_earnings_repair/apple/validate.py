"""Validate saved Apple evidence only; performs no network requests."""
import datetime as dt
import json
from pathlib import Path

from collect import HERE, ROOT, VENDOR, dump, parse_html, sha


def main():
    targets = [x for x in json.loads(VENDOR.read_text()) if x["reported_date"] >= "2015-01-01"]
    target_map = {x["reported_date"]: x for x in targets}
    events = json.loads((HERE / "events.json").read_text())
    assert len(events) == len({x["release_date"] for x in events}), "duplicate release"
    errors = []
    fiscal_ids = []
    for e in events:
        try:
            p = ROOT / e["evidence_path"]
            body = p.read_bytes(); meta = json.loads(p.with_suffix(".meta.json").read_text())
            assert sha(body) == e["evidence_sha256"] == meta["sha256"], "checksum changed"
            assert meta["http_status"] == 200
            parsed = parse_html(body, target_map[e["release_date"]], e["source_url"], e["evidence_path"])
            assert parsed == e, "stored record differs from reparse"
            assert e["source_url"].startswith("https://www.apple.com/newsroom/"), "not US issuer source"
            assert e["fiscal_period_end_actual"] < e["release_date"], "fiscal end after release"
            assert e["publication_timestamp"] is None and e["precision"] == "date_only"
            assert e["publication_timezone"] == "America/Los_Angeles"
            assert e["publication_timezone_evidence"]["kind"] == "issuer_local_calendar_date_policy"
            assert e["publication_timezone_evidence"]["cms_timezone_observed"] is False
            assert e["publication_timezone_evidence"]["exact_publication_time_inferred"] is False
            fiscal_ids.append(e["fiscal_year"] * 4 + e["fiscal_quarter"] - 1)
        except (AssertionError, ValueError, KeyError) as ex:
            errors.append({"release_date": e["release_date"], "error": str(ex)})
    expected_fiscal_ids = list(range(2015 * 4, 2026 * 4 + 3))
    missing = sorted(set(target_map) - {x["release_date"] for x in events})
    report = {
        "checked_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": "Apple US Newsroom original earnings releases",
        "coverage_start": "2015-01-27", "coverage_end": "2026-07-30",
        "expected_provider_targets": len(targets), "verified_events": len(events),
        "expected_in_study_events": sum(x["in_study_window"] for x in targets),
        "verified_in_study_events": sum(x["in_study_window"] for x in events),
        "warmup_2015_events": sum(not x["in_study_window"] for x in events),
        "date_only_events": sum(x["precision"] == "date_only" for x in events),
        "explicit_publication_timezone_events": sum(x.get("publication_timezone") == "America/Los_Angeles" for x in events),
        "publication_timezone_policy": "America/Los_Angeles from each original article's Cupertino California dateline as an explicit conservative issuer-local-date research mapping; not observed CMS timezone or exact release time.",
        "exact_publication_timestamp_events": sum(x["publication_timestamp"] is not None for x in events),
        "provider_fiscal_label_corrections": sum(not x["vendor_period_label_matches_actual"] for x in events),
        "fiscal_quarters_contiguous_FY2015Q1_through_FY2026Q3": sorted(fiscal_ids) == expected_fiscal_ids,
        "missing_release_dates": missing, "validation_errors": errors,
        "all_target_dates_and_fiscal_periods_verified": not missing and not errors and sorted(fiscal_ids) == expected_fiscal_ids,
        "available_time_policy": "Date-only publication must not be used earlier than the next US trading session; no inference of release minute, midnight UTC, or conference-call time.",
        "not_claimed": ["full 73-equity historical universe readiness", "point-in-time consensus estimates", "verified first-release timestamps", "backtested earnings factor"],
        "provider_normalized_sha256": sha(VENDOR.read_bytes()),
        "events_sha256": sha((HERE / "events.json").read_bytes()),
        "network_requests_from_validator": 0,
    }
    dump(HERE / "VALIDATION.json", report)
    print(json.dumps(report, indent=2))
    if missing or errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
