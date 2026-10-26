"""Independent source-gate synthetics, without importing the producer."""
from copy import deepcopy
import unittest

import stock_comp_source_admission_v1_independent as m


def sample(eid="SYNTH:2024", symbol="SYNTH", held=False):
    source = dict(path="synthetic.html", sha256="a" * 64)
    period = dict(period_start="2024-01-01", period_end="2024-12-31", days_inclusive=366)
    loc = dict(format="html", field_scope="explicit_component_of_consolidated_indirect_operating_reconciliation",
               period_end=period["period_end"], scale="1000000", currency_literal="$",
               statement_title="consolidated statements of cash flows", table_index_zero_based=1,
               logical_columns=[2], row_label="Stock-based compensation expense")
    prior = dict(event_id=eid, symbol=symbol, fiscal_year=2024, release_date="2025-01-06",
                 available_from="2025-01-07", identity=dict(security_id=symbol),
                 source=source, periods=dict(current=period), inherited_rejection_codes=["old"] if held else [],
                 cash_quality_admission=dict(admitted=not held, reasons=["old"] if held else [], definition="original"),
                 fields=dict(current=dict(operating_cash_flow=dict(scale="1000000", currency_literal="$",
                                                                   locator=deepcopy(loc)))))
    candidate = {key: deepcopy(prior[key]) for key in m.META}
    candidate.update({key: False for key in m.RAW_FLAGS})
    result = dict(field="stock_compensation_addback", candidate_only=True, gaps=[],
                  accounting_review_flags=[dict(code="new_fields_independent_source_review_required")],
                  value=dict(source=deepcopy(source), source_sha256=source["sha256"], period=deepcopy(period),
                             available_from=prior["available_from"], fact_admitted=False, currency_iso=None,
                             value_decimal="25", scale="1000000", currency_literal="$", locator=loc))
    result.update({key: False for key in m.RAW_FLAGS + m.CERTS})
    candidate["stock_compensation"] = result
    return prior, candidate


def population():
    originals, raw, patches, merged, outputs = [], [], [], {}, []
    ids = sorted(m.PATCH_IDS) + ["SYNTH:%02d" % i for i in range(41)]
    for i, eid in enumerate(ids):
        prior, candidate = sample(eid, "AAPL" if i < 10 else "SYNTH", held=i >= 31)
        originals.append(prior)
        originals[-1]["identity"] = deepcopy(candidate["identity"])
        merged[eid] = deepcopy(candidate)
        out = deepcopy(candidate)
        out["stock_compensation_admission"] = dict(admitted=i < 31,
            reasons=[] if i < 31 else ["preserved_original_cash_quality_hold"], definition=m.DEFINITION)
        outputs.append(out)
        if i < 3:
            patches.append(dict(event_id=eid, result=deepcopy(candidate["stock_compensation"])))
            candidate["stock_compensation"].update(
                value=None, gaps=[dict(reason="invalid_or_ambiguous_current_monetary_cell", error_type="ValueError")])
        raw.append(candidate)
    return originals, raw, patches, merged, outputs


class SourceGateTests(unittest.TestCase):
    def test_valid_source_only_decision(self):
        prior, candidate = sample()
        self.assertEqual(m.decision(prior, candidate, {prior["event_id"]}), [])

    def test_old_hold_kept_despite_complete_cell(self):
        prior, candidate = sample(held=True)
        self.assertEqual(m.decision(prior, candidate, {prior["event_id"]}),
                         ["preserved_original_cash_quality_hold"])

    def test_missing_review_narrows_availability(self):
        prior, candidate = sample()
        self.assertEqual(m.decision(prior, candidate, set()), ["independent_source_value_review_missing"])

    def test_missing_value_is_not_zero(self):
        prior, candidate = sample()
        candidate["stock_compensation"].update(value=None, gaps=[dict(reason="missing")])
        self.assertEqual(m.decision(prior, candidate, set()), ["stock_compensation_value_missing"])

    def test_old_hold_and_missing_value_both_recorded(self):
        prior, candidate = sample(held=True)
        candidate["stock_compensation"].update(value=None, gaps=[dict(reason="missing")])
        self.assertEqual(m.decision(prior, candidate, set()),
                         ["preserved_original_cash_quality_hold", "stock_compensation_value_missing"])

    def test_signed_negative_needs_retained_review_flag(self):
        prior, candidate = sample()
        candidate["stock_compensation"]["value"]["value_decimal"] = "-25"
        with self.assertRaisesRegex(ValueError, "review flag"):
            m.decision(prior, candidate, {prior["event_id"]})
        candidate["stock_compensation"]["accounting_review_flags"].append(
            dict(code="negative_stock_compensation_addback_requires_review"))
        self.assertEqual(m.decision(prior, candidate, {prior["event_id"]}), [])
        self.assertEqual(candidate["stock_compensation"]["value"]["value_decimal"], "-25")

    def test_explicit_zero_preserved(self):
        prior, candidate = sample()
        candidate["stock_compensation"]["value"]["value_decimal"] = "0"
        self.assertEqual(m.decision(prior, candidate, {prior["event_id"]}), [])

    def test_bad_numeric_literals_rejected(self):
        for value in ("NaN", "inf", "-", "", "2,000", 0.5):
            prior, candidate = sample()
            candidate["stock_compensation"]["value"]["value_decimal"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                m.decision(prior, candidate, {prior["event_id"]})

    def test_optional_segment_absent_passes_but_changed_fails(self):
        prior, candidate = sample()
        self.assertEqual(m.decision(prior, candidate, {prior["event_id"]}), [])
        candidate["stock_compensation"]["value"]["locator"]["statement_segment"] = {"start_row": 3}
        with self.assertRaisesRegex(ValueError, "segment"):
            m.decision(prior, candidate, {prior["event_id"]})

    def test_same_period_scale_and_currency_required(self):
        for key, value in (("scale", "1000"), ("currency_literal", "EUR"),
                           ("period", {}), ("available_from", "2025-01-06"), ("source", {})):
            prior, candidate = sample()
            candidate["stock_compensation"]["value"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                m.decision(prior, candidate, {prior["event_id"]})

    def test_all_metadata_preserved(self):
        for key in m.META:
            prior, candidate = sample()
            candidate[key] = "changed"
            with self.subTest(key=key), self.assertRaises(ValueError):
                m.decision(prior, candidate, {prior["event_id"]})

    def test_raw_admission_and_certifications_cannot_change(self):
        for key in m.RAW_FLAGS + m.CERTS:
            prior, candidate = sample()
            candidate["stock_compensation"][key] = True
            with self.subTest(key=key), self.assertRaises(ValueError):
                m.decision(prior, candidate, {prior["event_id"]})

    def test_combined_or_net_tax_label_not_eligible(self):
        for label in ("Stock-based compensation and other", "Stock-based compensation, net of tax",
                      "Cash paid for shares", "Non-GAAP Stock-based compensation"):
            prior, candidate = sample()
            candidate["stock_compensation"]["value"]["locator"]["row_label"] = label
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, "Standalone"):
                m.decision(prior, candidate, {prior["event_id"]})

    def test_unknown_accounting_flag_rejected(self):
        prior, candidate = sample()
        candidate["stock_compensation"]["accounting_review_flags"].append(dict(code="new_undefined_issue"))
        with self.assertRaisesRegex(ValueError, "policy disposition"):
            m.decision(prior, candidate, {prior["event_id"]})

    def test_complete_patch_is_exact_and_nonmutating(self):
        _, raw, patches, merged, _ = population()
        before = deepcopy((raw, patches))
        self.assertEqual(m.patched_candidates(raw, patches), merged)
        self.assertEqual((raw, patches), before)

    def test_patch_cannot_overwrite_existing_value(self):
        _, raw, patches, _, _ = population()
        raw[0]["stock_compensation"] = deepcopy(patches[0]["result"])
        with self.assertRaisesRegex(ValueError, "failed cells"):
            m.patched_candidates(raw, patches)

    def test_wrong_patch_event_or_duplicate_rejected(self):
        for kind in ("wrong", "duplicate", "missing"):
            _, raw, patches, _, _ = population()
            if kind == "wrong":
                patches[0]["event_id"] = "different"
            elif kind == "duplicate":
                patches[-1] = deepcopy(patches[0])
            else:
                patches.pop()
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                m.patched_candidates(raw, patches)

    def test_patch_metadata_injection_rejected(self):
        _, raw, patches, _, _ = population()
        patches[0]["available_from"] = "1900-01-01"
        with self.assertRaisesRegex(ValueError, "metadata injection"):
            m.patched_candidates(raw, patches)

    def test_exact_population_decisions(self):
        prior, _, _, merged, outputs = population()
        admitted, held = m.verify_output_events(prior, merged, outputs, set(merged))
        self.assertEqual((len(admitted), len(held)), (31, 13))

    def test_output_cannot_modify_source_cell(self):
        prior, _, _, merged, outputs = population()
        outputs[0]["stock_compensation"]["value"]["value_decimal"] = "500"
        with self.assertRaisesRegex(ValueError, "candidate or accounting"):
            m.verify_output_events(prior, merged, outputs, set(merged))

    def test_output_cannot_promote_old_hold(self):
        prior, _, _, merged, outputs = population()
        outputs[-1]["stock_compensation_admission"].update(admitted=True, reasons=[])
        with self.assertRaisesRegex(ValueError, "eligibility decision"):
            m.verify_output_events(prior, merged, outputs, set(merged))

    def test_reordered_population_rejected(self):
        prior, _, _, merged, outputs = population()
        outputs.reverse()
        with self.assertRaisesRegex(ValueError, "ordered"):
            m.verify_output_events(prior, merged, outputs, set(merged))

    def test_removed_accounting_flag_rejected(self):
        prior, _, _, merged, outputs = population()
        outputs[0]["stock_compensation"]["accounting_review_flags"] = []
        with self.assertRaisesRegex(ValueError, "candidate or accounting"):
            m.verify_output_events(prior, merged, outputs, set(merged))

    def test_explicit_policy_digest_required_before_io(self):
        with self.assertRaisesRegex(ValueError, "Explicit source policy"):
            m.verify("/tmp/no-such-admission", None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
