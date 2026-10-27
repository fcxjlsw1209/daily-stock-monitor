"""Fabricated source metadata only; no raw filings, datasets or write-once output."""
import ast
import copy
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import unittest

import roa_field_scope270_v2 as m


def fixture(kind="calendar"):
    # Dates illustrate calendar/52-week shapes. All amounts and refs are invented.
    choices = {
        "calendar": ("SYN", "2020-01-01", "2020-12-31", "2021-02-01", "2021-02-02", "html"),
        "apple_shape": ("AAPL", "2015-09-27", "2016-09-24", "2016-10-25", "2016-10-26", "pdf"),
        "microsoft_shape": ("MSFT", "2014-07-01", "2015-06-30", "2015-07-21", "2015-07-22", "html"),
        "amd_shape": ("AMD", "2014-12-28", "2015-12-26", "2016-01-19", "2016-01-20", "html"),
    }
    symbol, start, end, release, available, fmt = choices[kind]
    prior_end = (date.fromisoformat(start) - timedelta(days=1)).isoformat()
    current = dict(period_start=start, period_end=end,
                   days_inclusive=(date.fromisoformat(end) - date.fromisoformat(start)).days + 1)
    comparison = dict(period_start=None, period_end=prior_end, days_inclusive=None)
    e = dict(event_id=symbol + ":fabricated", symbol=symbol,
             prior_field_admitted=False,
             inherited_rejection_codes=list(m.INAPPLICABLE_CODES), metadata_gaps=[],
             accounting_review_flags=[dict(code=m.REVIEWABLE_FLAG)],
             release_date=release, available_from=available,
             identity=dict(issuer_id="synthetic-issuer-" + symbol, security_id=symbol,
                           identity_valid_from=available,
                           identity_valid_through=(date.fromisoformat(end) + timedelta(days=550)).isoformat()),
             periods=dict(current=current, comparison=comparison),
             source=dict(path="synthetic/" + symbol + ".raw", sha256="a" * 64,
                         receipt=dict(path="synthetic/retrieval.json", sha256="c" * 64),
                         url="https://synthetic.example/" + symbol + "/release"),
             fields=dict(current={}, comparison={}), fact_admitted=False,
             model_admitted=False, strategy_admitted=False)
    for side, field in m.REQUIRED:
        period = e["periods"][side]
        locator = dict(format=fmt, period_end=period["period_end"],
                       row_label="Net income" if field == "net_income" else "Total assets",
                       scale="1000000", currency_literal="$",
                       ni_scope="indirect_method_consolidated_start" if field == "net_income" else None)
        if fmt == "html":
            locator.update(table_index_zero_based=1, row_index_zero_based=2,
                           physical_cell_indices_zero_based=[3, 4])
        else:
            locator.update(page_1based=2, line_1based=7, financial_column_index_0based=0)
        e["fields"][side][field] = dict(source=copy.deepcopy(e["source"]),
            available_from=available, period=copy.deepcopy(period), locator=locator,
            value_decimal="-7" if kind == "amd_shape" and field == "net_income" else "120",
            scale="1000000", currency_literal="$", currency_iso=None, fact_admitted=False)
    sessions = [release, available, (date.fromisoformat(available) + timedelta(days=1)).isoformat()]
    return e, sessions


def review(event):
    pins = {}
    for side, field in m.REQUIRED:
        cell = event.get("fields", {}).get(side, {}).get(field)
        # A nonexistent field cannot acquire a valid binding, even if null is hashed.
        pins[side + "." + field] = m.canonical_sha256(cell)
    return dict(schema=m.REVIEW_SCHEMA, event_id=event["event_id"],
                event_sha256=m.canonical_sha256(event), raw_source={key: event["source"][key] for key in ("path", "sha256")},
                cell_sha256=pins, review_receipts=[dict(path="synthetic/independent-review.json", sha256="b" * 64)],
                independently_reviewed=True)


def assessed(e, sessions):
    return m.assess(e, sessions, review(e))


class EvidenceTests(unittest.TestCase):
    def test_canonical_digest_literal(self):
        value = {"z": [None, True, "é"], "a": 1}
        self.assertEqual(m.canonical_sha256(value), hashlib.sha256(
            '{"a":1,"z":[null,true,"é"]}'.encode("utf-8")).hexdigest())

    def test_non_json_or_nonfinite_digest_rejected(self):
        for value in ({1: "x"}, {"a": float("nan")}, {"a": float("inf")}, (1, 2), object()):
            with self.subTest(value=repr(value)), self.assertRaises(ValueError):
                m.canonical_sha256(value)

    def test_three_source_shapes_and_no_issuer_exception(self):
        for kind in ("apple_shape", "microsoft_shape", "amd_shape"):
            e, sessions = fixture(kind)
            self.assertTrue(assessed(e, sessions)["admitted"], kind)
            e["symbol"] = "UNSEEN"
            e["identity"]["security_id"] = "UNSEEN"
            self.assertTrue(assessed(e, sessions)["admitted"], kind)

    def test_no_mutation_of_any_input(self):
        e, sessions = fixture()
        r = review(e)
        before = copy.deepcopy((e, sessions, r))
        result = m.assess(e, sessions, r)
        self.assertTrue(result["admitted"])
        self.assertEqual((e, sessions, r), before)
        result["raw_source"]["path"] = "modified output"
        result["review_receipts"][0]["path"] = "modified output"
        self.assertEqual((e, sessions, r), before)

    def test_output_scope_does_not_certify_files_currency_or_chain(self):
        e, sessions = fixture()
        result = assessed(e, sessions)
        self.assertEqual(tuple(result), m.OUTPUT_KEYS)
        self.assertTrue(result["admitted"] and result["source_only"] and result["evidence_binding_passed"])
        for key in ("strict_pit", "source_bytes_verified", "review_receipt_bytes_verified",
                    "currency_iso_certified", "first_public_version_verified",
                    "complete_correction_history_verified", "ratios_computed", "chain_admitted",
                    "existing_admissions_modified"):
            self.assertIs(result[key], False, key)
        self.assertEqual(result["scope_inapplicable_inherited_codes"], list(m.INAPPLICABLE_CODES))
        self.assertEqual(result["scope_satisfied_accounting_flags"], [m.REVIEWABLE_FLAG])
        json.dumps(result, allow_nan=False)

    def test_full_event_tamper_including_unused_body_rejected(self):
        changes = [lambda e: e.update(unrelated="new"),
                   lambda e: e["periods"]["comparison"].update(period_start="1900-01-01"),
                   lambda e: e["fields"]["comparison"].update(net_income={"anything": "new"}),
                   lambda e: e["identity"].update(issuer_id="changed")]
        for change in changes:
            e, sessions = fixture(); r = review(e); change(e)
            result = m.assess(e, sessions, r)
            self.assertFalse(result["admitted"])
            self.assertFalse(result["evidence_binding_passed"])

    def test_each_needed_cell_tamper_rejected_even_if_event_digest_refreshed(self):
        for side, name in m.REQUIRED:
            for key, value in (("value_decimal", "121"), ("scale", "1000"),
                               ("currency_iso", "USD"), ("locator", {"period_end": "2020-12-31"})):
                e, sessions = fixture(); r = review(e)
                e["fields"][side][name][key] = value
                r["event_sha256"] = m.canonical_sha256(e)
                self.assertFalse(m.assess(e, sessions, r)["admitted"])

    def test_review_schema_and_all_bindings_required(self):
        mutations = [lambda r: r.update(extra=True), lambda r: r.pop("event_sha256"),
                     lambda r: r.update(schema="wrong"), lambda r: r.update(event_id="other"),
                     lambda r: r.update(independently_reviewed=1),
                     lambda r: r.update(event_sha256="A" * 64),
                     lambda r: r["raw_source"].update(sha256="c" * 64),
                     lambda r: r["raw_source"].update(path="other.raw"),
                     lambda r: r["cell_sha256"].pop("current.net_income"),
                     lambda r: r["cell_sha256"].update(unused="a" * 64),
                     lambda r: r.update(review_receipts=[]),
                     lambda r: r["review_receipts"][0].update(sha256="no"),
                     lambda r: r["review_receipts"][0].update(path=""),
                     lambda r: r["review_receipts"].append(copy.deepcopy(r["review_receipts"][0]))]
        for change in mutations:
            e, sessions = fixture(); r = review(e); change(r)
            self.assertFalse(m.assess(e, sessions, r)["admitted"])

    def test_delete_unused_flows_with_new_event_binding_same_decision(self):
        e, sessions = fixture()
        for side in ("current", "comparison"):
            e["fields"][side]["operating_cash_flow"] = {"unread_amount": "not-a-number"}
        e["fields"]["comparison"]["net_income"] = {"unread_amount": "Infinity"}
        old_review = review(e)
        before = m.assess(e, sessions, old_review)
        del e["fields"]["current"]["operating_cash_flow"]
        del e["fields"]["comparison"]["operating_cash_flow"]
        del e["fields"]["comparison"]["net_income"]
        self.assertFalse(m.assess(e, sessions, old_review)["admitted"])
        after = assessed(e, sessions)
        self.assertTrue(before["admitted"] and after["admitted"])
        self.assertEqual(before["required_cell_sha256"], after["required_cell_sha256"])

    def test_delete_each_needed_cell_rejected(self):
        for side, name in m.REQUIRED:
            e, sessions = fixture(); del e["fields"][side][name]
            self.assertFalse(assessed(e, sessions)["admitted"])


class ScopeTests(unittest.TestCase):
    def test_already_admitted_source_without_legacy_holds(self):
        e, sessions = fixture()
        e.update(prior_field_admitted=True, inherited_rejection_codes=[], accounting_review_flags=[])
        self.assertTrue(assessed(e, sessions)["admitted"])

    def test_prior_false_without_explicit_allowed_reason_held(self):
        e, sessions = fixture(); e["inherited_rejection_codes"] = []
        self.assertFalse(assessed(e, sessions)["admitted"])

    def test_unknown_legacy_and_accounting_hold_never_cleared(self):
        for code in ("excluded_accounting_comparability", "missing_current_net_income", "unknown", ""):
            e, sessions = fixture(); e["inherited_rejection_codes"].append(code)
            self.assertFalse(assessed(e, sessions)["admitted"])
        e, sessions = fixture()
        e["accounting_review_flags"].append(dict(code="unreviewed_basis_change"))
        self.assertFalse(assessed(e, sessions)["admitted"])

    def test_reviewable_flag_requires_binding(self):
        e, sessions = fixture(); r = review(e); r["event_sha256"] = "0" * 64
        result = m.assess(e, sessions, r)
        self.assertFalse(result["admitted"])
        self.assertEqual(result["scope_satisfied_accounting_flags"], [])
        self.assertEqual(result["scope_inapplicable_inherited_codes"], [])

    def test_malformed_accounting_flags_held(self):
        for flags in (None, "", ["unknown"], [{}], [dict(code=True)]):
            e, sessions = fixture(); e["accounting_review_flags"] = flags
            self.assertFalse(assessed(e, sessions)["admitted"])

    def test_missing_or_malformed_legacy_metadata_held(self):
        for key in ("prior_field_admitted", "inherited_rejection_codes", "metadata_gaps"):
            e, sessions = fixture(); del e[key]
            self.assertFalse(assessed(e, sessions)["admitted"])
        for key, value in (("prior_field_admitted", 1), ("inherited_rejection_codes", None),
                           ("metadata_gaps", [dict(reason="unknown")])):
            e, sessions = fixture(); e[key] = value
            self.assertFalse(assessed(e, sessions)["admitted"])

    def test_all_metadata_gaps_hold(self):
        e, sessions = fixture(); e["metadata_gaps"] = ["unknown comparison duration"]
        self.assertFalse(assessed(e, sessions)["admitted"])

    def test_legacy_codes_preserved_with_partial_unknown(self):
        e, sessions = fixture(); e["inherited_rejection_codes"].append("unknown")
        before = copy.deepcopy(e)
        result = assessed(e, sessions)
        self.assertFalse(result["admitted"])
        self.assertEqual(e, before)

    def test_comparison_flow_start_and_duration_not_required(self):
        e, sessions = fixture()
        e["periods"]["comparison"]["period_start"] = "uninterpreted-flow-date"
        e["periods"]["comparison"]["days_inclusive"] = "unknown"
        self.assertTrue(assessed(e, sessions)["admitted"])

    def test_missing_comparison_flow_start_key_is_not_invented(self):
        e, sessions = fixture()
        del e["periods"]["comparison"]["period_start"]
        e["fields"]["comparison"]["total_assets"]["period"]["period_start"] = None
        self.assertTrue(assessed(e, sessions)["admitted"])
        self.assertNotIn("period_start", e["periods"]["comparison"])

    def test_raw_admission_flags_not_rewritten_or_accepted(self):
        for name in ("fact_admitted", "model_admitted", "strategy_admitted"):
            e, sessions = fixture(); e[name] = True
            self.assertFalse(assessed(e, sessions)["admitted"])


class CellClockTests(unittest.TestCase):
    def test_negative_or_zero_net_income_allowed(self):
        for value in ("-660", "-0", "0"):
            e, sessions = fixture(); e["fields"]["current"]["net_income"]["value_decimal"] = value
            self.assertTrue(assessed(e, sessions)["admitted"])

    def test_asset_values_strictly_positive(self):
        for side in ("current", "comparison"):
            for value in ("0", "-0", "-1", "NaN", "Infinity"):
                e, sessions = fixture(); e["fields"][side]["total_assets"]["value_decimal"] = value
                self.assertFalse(assessed(e, sessions)["admitted"])

    def test_different_positive_scales_allowed_without_rescaling_or_ratio(self):
        e, sessions = fixture()
        for (side, name), scale in zip(m.REQUIRED, ("1", "1000", 1000000)):
            cell = e["fields"][side][name]; cell["scale"] = scale; cell["locator"]["scale"] = scale
        before = copy.deepcopy(e)
        result = assessed(e, sessions)
        self.assertTrue(result["admitted"])
        self.assertEqual(e, before)
        self.assertIs(result["ratios_computed"], False)

    def test_nonfinite_bad_numeric_types_and_scale_rejected(self):
        for key, value in (("value_decimal", "sNaN"), ("value_decimal", "no"),
                           ("value_decimal", True), ("value_decimal", 1.5),
                           ("scale", "-1"), ("scale", "0"), ("scale", "Infinity"), ("scale", True)):
            e, sessions = fixture(); e["fields"]["current"]["net_income"][key] = value
            self.assertFalse(assessed(e, sessions)["admitted"])

    def test_conditional_literal_currency_not_iso_certification(self):
        for iso in (None, "USD"):
            e, sessions = fixture()
            for side, name in m.REQUIRED:
                e["fields"][side][name]["currency_iso"] = iso
            result = assessed(e, sessions)
            self.assertTrue(result["admitted"])
            self.assertFalse(result["currency_iso_certified"])
        for key, value in (("currency_literal", "EUR"), ("currency_iso", "CAD")):
            e, sessions = fixture(); e["fields"]["current"]["net_income"][key] = value
            self.assertFalse(assessed(e, sessions)["admitted"])
        e, sessions = fixture(); e["fields"]["current"]["net_income"]["currency_iso"] = "USD"
        self.assertFalse(assessed(e, sessions)["admitted"])

    def test_explicit_as_of_agrees_for_all_required_cells(self):
        for side, name in m.REQUIRED:
            for value in (None, "match", "2020-06-30", "2020-6-30", 1):
                e, sessions = fixture(); cell = e["fields"][side][name]
                cell["as_of"] = cell["period"]["period_end"] if value == "match" else value
                self.assertEqual(assessed(e, sessions)["admitted"], value in (None, "match"))

    def test_cell_source_clock_period_and_locator_mutation_with_new_review_still_held(self):
        for side, name in m.REQUIRED:
            changes = [lambda c: c["source"].update(sha256="d" * 64),
                       lambda c: c.update(available_from="2021-02-03"),
                       lambda c: c["period"].update(period_end="2018-12-31"),
                       lambda c: c["locator"].update(period_end="2018-12-31"),
                       lambda c: c["locator"].update(scale="10"),
                       lambda c: c.update(fact_admitted=True)]
            for change in changes:
                e, sessions = fixture(); change(e["fields"][side][name])
                self.assertFalse(assessed(e, sessions)["admitted"])

    def test_ni_period_and_scope_exact(self):
        for change in (lambda c: c["period"].update(days_inclusive=365),
                       lambda c: c["locator"].update(ni_scope="parent_only")):
            e, sessions = fixture(); change(e["fields"]["current"]["net_income"])
            self.assertFalse(assessed(e, sessions)["admitted"])

    def test_current_duration_range_and_complete_fields(self):
        for duration in (329, 330, 400, 401):
            e, sessions = fixture()
            e["periods"]["current"]["period_start"] = (date(2020, 12, 31) - timedelta(days=duration - 1)).isoformat()
            e["periods"]["current"]["days_inclusive"] = duration
            prior = (date.fromisoformat(e["periods"]["current"]["period_start"]) - timedelta(days=1)).isoformat()
            e["periods"]["comparison"]["period_end"] = prior
            for side, name in m.REQUIRED:
                cell = e["fields"][side][name]; cell["period"] = copy.deepcopy(e["periods"][side])
                cell["locator"]["period_end"] = cell["period"]["period_end"]
            self.assertEqual(assessed(e, sessions)["admitted"], duration in (330, 400))
        for key in ("period_start", "period_end", "days_inclusive"):
            e, sessions = fixture(); del e["periods"]["current"][key]
            self.assertFalse(assessed(e, sessions)["admitted"])

    def test_current_or_opening_asset_instant_mismatch(self):
        for side in ("current", "comparison"):
            e, sessions = fixture(); c = e["fields"][side]["total_assets"]
            c["period"]["period_end"] = c["locator"]["period_end"] = "2020-06-30"
            self.assertFalse(assessed(e, sessions)["admitted"])
        e, sessions = fixture(); e["periods"]["comparison"]["period_end"] = "2019-12-30"
        self.assertFalse(assessed(e, sessions)["admitted"])

    def test_missing_or_malformed_physical_locator(self):
        for kind in ("calendar", "apple_shape"):
            changes = [lambda c: c.update(locator={}), lambda c: c["locator"].update(format="text"),
                       lambda c: c["locator"].update(row_label=""),
                       lambda c: c["locator"].update(currency_literal="CAD")]
            changes.append(lambda c: c["locator"].update(physical_cell_indices_zero_based=[True])
                           if kind == "calendar" else c["locator"].update(page_1based=0))
            for change in changes:
                e, sessions = fixture(kind); change(e["fields"]["current"]["net_income"])
                self.assertFalse(assessed(e, sessions)["admitted"])

    def test_calendar_must_bracket_release_and_availability(self):
        e, sessions = fixture()
        for bad in ([], [sessions[1]], [sessions[0]], sessions[::-1], sessions + [sessions[-1]],
                    [sessions[0], "2021-02-03"], ["2021-2-1", sessions[1]]):
            self.assertFalse(assessed(e, bad)["admitted"])
        for value in ("2021-02-01", "2021-02-03", None):
            e, sessions = fixture(); e["available_from"] = value
            self.assertFalse(assessed(e, sessions)["admitted"])

    def test_missing_release_clock_or_identity_fields_held(self):
        for key in ("release_date", "available_from", "source", "identity"):
            e, sessions = fixture(); r = review(e); del e[key]
            self.assertFalse(m.assess(e, sessions, r)["admitted"])
        for key, value in (("security_id", "other"), ("issuer_id", ""),
                           ("identity_valid_from", "2021-02-03"), ("identity_valid_through", "2022-07-05")):
            e, sessions = fixture(); e["identity"][key] = value
            self.assertFalse(assessed(e, sessions)["admitted"])

    def test_release_before_annual_end_held(self):
        e, _ = fixture(); e["release_date"] = "2020-12-30"; e["available_from"] = "2020-12-31"
        e["identity"]["identity_valid_from"] = e["available_from"]
        for side, name in m.REQUIRED:
            e["fields"][side][name]["available_from"] = e["available_from"]
        self.assertFalse(assessed(e, ["2020-12-30", "2020-12-31"])["admitted"])

    def test_pure_module_has_no_io_or_ratio_execution(self):
        tree = ast.parse(Path(m.__file__).read_text())
        imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        imports.update(alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names)
        self.assertEqual(imports, {"bisect", "datetime", "decimal", "hashlib", "json", "math", "urllib.parse"})
        calls = [node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
        self.assertFalse({"open", "eval", "exec", "compile"}.intersection(calls))
        self.assertFalse(any(isinstance(node, (ast.Div, ast.FloorDiv)) for node in ast.walk(tree)))


class RichSourceTests(unittest.TestCase):
    @staticmethod
    def sync_source(event):
        for side, name in m.REQUIRED:
            event["fields"][side][name]["source"] = copy.deepcopy(event["source"])

    def test_both_real_metadata_shapes_null_and_string_entry(self):
        for entry in ("absent", None, "document-key"):
            e, sessions = fixture()
            if entry != "absent":
                e["source"]["receipt_entry_key"] = entry
            self.sync_source(e)
            r = review(e)
            result = m.assess(e, sessions, r)
            self.assertTrue(result["admitted"])
            self.assertEqual(set(result["raw_source"]), {"path", "sha256"})
            self.assertEqual(result["event_sha256"], m.canonical_sha256(e))
            for side, name in m.REQUIRED:
                self.assertEqual(result["required_cell_sha256"][side + "." + name],
                                 m.canonical_sha256(e["fields"][side][name]))

    def test_rich_source_mandatory_keys_exact_allowlist(self):
        for missing in ("path", "sha256", "receipt", "url"):
            e, sessions = fixture(); r = review(e)
            del e["source"][missing]
            self.sync_source(e)
            r["event_sha256"] = m.canonical_sha256(e)
            r["cell_sha256"] = {side + "." + name: m.canonical_sha256(e["fields"][side][name])
                                for side, name in m.REQUIRED}
            self.assertFalse(m.assess(e, sessions, r)["admitted"])
        e, sessions = fixture(); e["source"]["unreviewed"] = True
        self.sync_source(e)
        self.assertFalse(assessed(e, sessions)["admitted"])

    def test_nested_receipt_is_strict_reference(self):
        for receipt in (None, {}, {"path": "", "sha256": "c" * 64},
                        {"path": "r", "sha256": "wrong"},
                        {"path": "r", "sha256": "c" * 64, "passed": True}):
            e, sessions = fixture(); e["source"]["receipt"] = receipt
            self.sync_source(e)
            self.assertFalse(assessed(e, sessions)["admitted"])

    def test_entry_key_types_and_empty_values_rejected(self):
        for entry in ("", " ", 0, False, [], {}):
            e, sessions = fixture(); e["source"]["receipt_entry_key"] = entry
            self.sync_source(e)
            self.assertFalse(assessed(e, sessions)["admitted"])

    def test_https_host_credentials_fragment_and_parser_errors(self):
        invalid = (None, 1, "", "http://synthetic.example/r", "file:///r", "https:///r",
                   "https://", "https://user@synthetic.example/r",
                   "https://user:pass@synthetic.example/r", "https://synthetic.example/r#part",
                   "https://synthetic.example/r#", "https://synthetic.example:bad/r",
                   "https://synthetic.example:65536/r", "https://[broken/r",
                   " https://synthetic.example/r", "https://synthetic.example/\nr",
                   "https://synthetic.example/\x00r")
        for url in invalid:
            with self.subTest(url=url):
                e, sessions = fixture(); e["source"]["url"] = url
                self.sync_source(e)
                self.assertFalse(assessed(e, sessions)["admitted"])

    def test_url_no_issuer_allowlist_or_authentication_claim(self):
        for url in ("https://another.example/report?document=1", "https://[::1]:443/report"):
            e, sessions = fixture(); e["source"]["url"] = url
            self.sync_source(e)
            result = assessed(e, sessions)
            self.assertTrue(result["admitted"])
            self.assertFalse(result["source_bytes_verified"])
            self.assertFalse(result["review_receipt_bytes_verified"])

    def test_source_metadata_tamper_requires_fresh_complete_event_binding(self):
        changes = (lambda s: s.update(url="https://another.example/release"),
                   lambda s: s["receipt"].update(path="different-retrieval.json"),
                   lambda s: s["receipt"].update(sha256="d" * 64),
                   lambda s: s.update(receipt_entry_key="different-document"))
        for change in changes:
            e, sessions = fixture(); r = review(e)
            change(e["source"]); self.sync_source(e)
            self.assertFalse(m.assess(e, sessions, r)["evidence_binding_passed"])
            # Refreshing only the event digest cannot hide changed cell source metadata.
            r["event_sha256"] = m.canonical_sha256(e)
            self.assertFalse(m.assess(e, sessions, r)["evidence_binding_passed"])
            self.assertTrue(assessed(e, sessions)["admitted"])

    def test_each_required_cell_must_match_complete_event_source(self):
        changes = (lambda s: s.update(url="https://another.example/release"),
                   lambda s: s["receipt"].update(path="different-retrieval.json"),
                   lambda s: s["receipt"].update(sha256="d" * 64),
                   lambda s: s.update(receipt_entry_key=None),
                   lambda s: s.update(receipt_entry_key="other"))
        for side, name in m.REQUIRED:
            for change in changes:
                e, sessions = fixture(); change(e["fields"][side][name]["source"])
                # Even a newly bound independent declaration does not waive same-report equality.
                result = assessed(e, sessions)
                self.assertTrue(result["evidence_binding_passed"])
                self.assertFalse(result["admitted"])

    def test_review_raw_source_remains_exact_two_key_projection(self):
        e, sessions = fixture(); r = review(e)
        r["raw_source"] = copy.deepcopy(e["source"])
        self.assertFalse(m.assess(e, sessions, r)["admitted"])
        for key in ("path", "sha256"):
            e, sessions = fixture(); r = review(e)
            r["raw_source"][key] = "other" if key == "path" else "f" * 64
            self.assertFalse(m.assess(e, sessions, r)["admitted"])

    def test_frozen_v1_files_unchanged(self):
        directory = Path(__file__).parent
        for name, expected in (
            ("roa_field_scope270.py", "28567702366a549e04b7fb1235fd95b047867cf3adb32b5045b20a02e373abd5"),
            ("test_roa_field_scope270.py", "cabf0ad37dca05142edafc883a9a312ed5a6c1063490bbcce74735f724d377c1")):
            self.assertEqual(hashlib.sha256((directory / name).read_bytes()).hexdigest(), expected)


if __name__ == "__main__":
    unittest.main()
