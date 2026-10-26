import copy
import unittest

import after150_source_batch as b


def context():
    return dict(symbol="INTC", event_id="INTC_test", fiscal_year=2020,
                evidence_path="work/synthetic.html", evidence_sha256="a"*64,
                publication_date="2021-01-21",
                available_session="2021-01-22", official_actual_period_end="2020-12-26",
                context_summary=dict(comparison_assets_date="2019-12-28"))


def value():
    c = context()
    src = dict(path=c["evidence_path"], sha256=c["evidence_sha256"])
    fields = {}
    for name in b.FIELDS:
        fields[name] = dict(
            value_decimal="10", displayed_amount="10", scale=1000000,
            source=src.copy(), available_from=c["available_session"],
            currency_literal="$", currency_iso=None,
            locator=dict(table_index=2, row_index=3), fact_admitted=False)
    fields["annual_net_income"]["period"] = dict(
        period_start="2019-12-29", period_end="2020-12-26", days_inclusive=364)
    fields["current_total_assets"]["as_of"] = "2020-12-26"
    fields["prior_total_assets"]["as_of"] = "2019-12-28"
    return dict(symbol=c["symbol"], event_id=c["event_id"], fiscal_year=c["fiscal_year"],
                source=src, available_from=c["available_session"], fields=fields,
                publication_date=c["publication_date"],
                periods=dict(current=copy.deepcopy(fields["annual_net_income"]["period"]),
                             comparison=dict(period_end="2019-12-28"),
                             start_basis="derived_same_report_prior_annual_end_plus_one_not_directly_quoted",
                             start_assumption_certified=False),
                context_holds=["annual_start_not_directly_quoted_requires_source_use_assumption"],
                fact_admitted=False, model_admitted=False, strategy_admitted=False, gaps=[])


class BatchTests(unittest.TestCase):
    def test_valid_transport(self):
        b.validate_record(value(), context())

    def test_signed_loss_allowed(self):
        v = value()
        v["fields"]["annual_net_income"]["value_decimal"] = "-10"
        b.validate_record(v, context())

    def test_missing_field_requires_gap(self):
        v = value()
        v["fields"]["prior_total_assets"] = None
        with self.assertRaises(ValueError):
            b.validate_record(v, context())
        v["gaps"] = [dict(field="prior_total_assets", reason="unresolved source geometry")]
        b.validate_record(v, context())

    def test_reject_nonfinite_and_nonpositive(self):
        for field, numeric in [
            ("annual_net_income", "NaN"), ("current_total_assets", "0"),
            ("prior_total_assets", "-1")]:
            with self.subTest(field=field, numeric=numeric):
                v = value()
                v["fields"][field]["value_decimal"] = numeric
                with self.assertRaises(ValueError):
                    b.validate_record(v, context())

    def test_reject_bad_scale(self):
        for numeric in ["NaN", "Infinity", "0", "-1"]:
            v = value()
            v["fields"]["annual_net_income"]["scale"] = numeric
            with self.assertRaises(ValueError):
                b.validate_record(v, context())

    def test_no_premature_admission(self):
        for key in ("fact_admitted", "model_admitted", "strategy_admitted"):
            v = value()
            v[key] = True
            with self.assertRaises(ValueError):
                b.validate_record(v, context())

    def test_field_premature_admission(self):
        v = value()
        v["fields"]["current_total_assets"]["fact_admitted"] = True
        with self.assertRaises(ValueError):
            b.validate_record(v, context())

    def test_identity_and_source_protected(self):
        for key, replacement in [("symbol", "KLAC"), ("event_id", "other"),
                                 ("fiscal_year", 2019)]:
            v = value()
            v[key] = replacement
            with self.assertRaises(ValueError):
                b.validate_record(v, context())
        v = value()
        v["source"]["sha256"] = "b"*64
        with self.assertRaises(ValueError):
            b.validate_record(v, context())

    def test_field_source_and_clock_protected(self):
        for key, replacement in [("available_from", "2020-12-26"),
                                 ("source", dict(path="work/other.html", sha256="a"*64))]:
            v = value()
            v["fields"]["annual_net_income"][key] = replacement
            with self.assertRaises(ValueError):
                b.validate_record(v, context())

    def test_wrong_year_or_length_rejected(self):
        for change in [dict(days_inclusive=365), dict(period_start="2020-09-27"),
                       dict(period_end="2019-12-28")]:
            v = value()
            v["fields"]["annual_net_income"]["period"].update(change)
            with self.assertRaises(ValueError):
                b.validate_record(v, context())

    def test_current_asset_endpoint(self):
        v = value()
        v["fields"]["current_total_assets"]["as_of"] = "2019-12-28"
        with self.assertRaises(ValueError):
            b.validate_record(v, context())

    def test_prior_asset_endpoint(self):
        v = value()
        v["fields"]["prior_total_assets"]["as_of"] = "2019-12-31"
        with self.assertRaises(ValueError):
            b.validate_record(v, context())

    def test_publication_and_currency(self):
        v = value()
        v["publication_date"] = "2021-01-20"
        with self.assertRaises(ValueError):
            b.validate_record(v, context())
        v = value()
        v["fields"]["annual_net_income"]["currency_iso"] = "USD"
        with self.assertRaises(ValueError):
            b.validate_record(v, context())

    def test_boundary_metadata_preserved(self):
        for key, replacement in [("start_basis", "quoted"), ("start_assumption_certified", True)]:
            v = value()
            v["periods"][key] = replacement
            with self.assertRaises(ValueError):
                b.validate_record(v, context())
        v = value()
        v["context_holds"] = []
        with self.assertRaises(ValueError):
            b.validate_record(v, context())

    def test_null_not_covered_by_other_fields_gap(self):
        v = value()
        v["fields"]["prior_total_assets"] = None
        v["gaps"] = [dict(field="annual_net_income", reason="irrelevant")]
        with self.assertRaises(ValueError):
            b.validate_record(v, context())

    def test_policy_catalog_exact(self):
        ctx = dict(documents=[context()])
        policy = dict(source_scope=dict(source_context=dict(path=b.CONTEXT, sha256="c"*64)),
                      fixed_catalog_metadata=b.catalog_metadata(ctx), required_fields=list(b.FIELDS))
        b.validate_policy(policy, ctx, "c"*64)
        policy["fixed_catalog_metadata"][0]["fiscal_year"] = 2019
        with self.assertRaises(ValueError):
            b.validate_policy(policy, ctx, "c"*64)

    def test_catalog_scope_and_explicit_missing(self):
        docs = [dict(symbol=s, event_id=f"{s}_{i}",
                     html_context_reviewed=(s == "KLAC" or i < 3))
                for s in ("INTC", "KLAC") for i in range(16)]
        self.assertEqual(len(b.records(dict(documents=docs))), 19)
        with self.assertRaises(ValueError):
            b.records(dict(documents=docs[:-1]))
        extra = copy.deepcopy(docs)
        extra[3]["html_context_reviewed"] = True
        with self.assertRaises(ValueError):
            b.records(dict(documents=extra))
        duplicate = copy.deepcopy(docs)
        duplicate[1]["event_id"] = duplicate[0]["event_id"]
        with self.assertRaises(ValueError):
            b.records(dict(documents=duplicate))


if __name__ == "__main__":
    unittest.main()
