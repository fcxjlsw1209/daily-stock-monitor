"""Transport and provenance guards; all amounts are synthetic."""
import copy
import unittest
import after151_source_batch as b


def context():
    return dict(symbol="KO", event_id="KO_test", fiscal_year=2020,
                source=dict(path="work/synthetic.html", sha256="a"*64, url="https://example.invalid",
                            receipt_path="work/synthetic.meta.json", receipt_sha256="b"*64),
                publication_date="2021-02-10", available_session="2021-02-11",
                official_actual_period_end="2020-12-31", context_status=b.READY,
                context_holds=["uncertified_currency", "unreviewed_geometry"],
                periods=dict(proposed_current_start="2020-01-01",
                             current_end="2020-12-31", prior_end="2019-12-31"))


def value():
    c = context()
    period = dict(period_start="2020-01-01", period_end="2020-12-31", days_inclusive=366)
    fields = {f: dict(value_decimal="10", scale=1000000, displayed_amount="10",
                     locator=dict(table_index=0, row_index=2),
                     source=copy.deepcopy(c["source"]), available_from=c["available_session"],
                     as_of="2019-12-31" if f == "prior_total_assets" else "2020-12-31",
                     period=copy.deepcopy(period) if f == "annual_net_income" else None,
                     currency_literal="$", currency_iso=None, fact_admitted=False)
              for f in b.FIELDS}
    return dict(symbol=c["symbol"], event_id=c["event_id"], fiscal_year=c["fiscal_year"],
                source=copy.deepcopy(c["source"]), context_holds=c["context_holds"][:],
                publication_date=c["publication_date"], available_from=c["available_session"],
                fields=fields, gaps=[], fact_admitted=False, model_admitted=False,
                strategy_admitted=False,
                periods=dict(current=period, comparison=dict(period_end="2019-12-31"),
                             start_basis="derived_same_report_prior_annual_end_plus_one_not_directly_quoted",
                             start_assumption_certified=False))


class TransportTests(unittest.TestCase):
    def test_valid_and_signed_income(self):
        for amount in ("10", "-10", "0"):
            v = value()
            v["fields"]["annual_net_income"]["value_decimal"] = amount
            b.validate_record(v, context())

    def test_null_needs_specific_cause(self):
        v = value()
        v["fields"]["prior_total_assets"] = None
        with self.assertRaises(ValueError):
            b.validate_record(v, context())
        v["gaps"] = [dict(field="annual_net_income", reason="unrelated")]
        with self.assertRaises(ValueError):
            b.validate_record(v, context())
        v["gaps"] = [dict(field="prior_total_assets", reason="geometry unresolved")]
        b.validate_record(v, context())

    def test_identity_clock_source_and_hold_guards(self):
        for key, val in dict(symbol="LRCX", event_id="other", fiscal_year=2019,
                             publication_date="2020-12-31", available_from="2020-12-31",
                             context_holds=[], source={}).items():
            v = value()
            v[key] = val
            with self.subTest(key=key), self.assertRaises(ValueError):
                b.validate_record(v, context())

    def test_admission_guard(self):
        for key in ("fact_admitted", "model_admitted", "strategy_admitted"):
            v = value()
            v[key] = True
            with self.assertRaises(ValueError):
                b.validate_record(v, context())
        v = value()
        v["fields"]["annual_net_income"]["fact_admitted"] = True
        with self.assertRaises(ValueError):
            b.validate_record(v, context())

    def test_receipt_preserved(self):
        for field in b.FIELDS:
            v = value()
            v["fields"][field]["source"]["receipt_sha256"] = "c"*64
            with self.assertRaises(ValueError):
                b.validate_record(v, context())

    def test_nonfinite_invalid_assets_and_scale(self):
        for field, key, val in [
            ("annual_net_income", "value_decimal", "NaN"),
            ("current_total_assets", "value_decimal", "0"),
            ("prior_total_assets", "value_decimal", "-1"),
            ("annual_net_income", "scale", "Infinity"),
            ("annual_net_income", "scale", "0")]:
            v = value()
            v["fields"][field][key] = val
            with self.assertRaises(ValueError):
                b.validate_record(v, context())

    def test_field_evidence_and_currency(self):
        for key, val in dict(source={}, available_from="2020-12-31", currency_iso="USD",
                             currency_literal="EUR", displayed_amount="", locator={},
                             as_of="2019-12-31").items():
            v = value()
            v["fields"]["annual_net_income"][key] = val
            with self.assertRaises(ValueError):
                b.validate_record(v, context())

    def test_asset_endpoints_and_period(self):
        for f in ("current_total_assets", "prior_total_assets"):
            for key, val in (("as_of", "2020-03-31"), ("period", {"bad": True})):
                v = value()
                v["fields"][f][key] = val
                with self.assertRaises(ValueError):
                    b.validate_record(v, context())

    def test_annual_derivation(self):
        for key, val in (("start_basis", "quoted"), ("start_assumption_certified", True)):
            v = value()
            v["periods"][key] = val
            with self.assertRaises(ValueError):
                b.validate_record(v, context())
        v = value()
        v["periods"]["current"]["days_inclusive"] = 365
        with self.assertRaises(ValueError):
            b.validate_record(v, context())

    def test_catalog_population_and_order(self):
        docs = []
        for s in ("KO", "LRCX"):
            for y in range(2010, 2026):
                selected = (s == "KO" and 2011 <= y <= 2016) or (s == "LRCX" and 2013 <= y <= 2016)
                docs.append(dict(symbol=s, event_id=f"{s}_{y}", fiscal_year=y,
                                 context_status=b.READY if selected else "unprocessed"))
        self.assertEqual(len(b.records(dict(events=docs))), 10)
        for changed in (docs[:-1], docs[::-1], docs+[docs[0]]):
            with self.assertRaises(ValueError):
                b.records(dict(events=changed))
        bad = copy.deepcopy(docs)
        bad[0]["context_status"] = b.READY
        with self.assertRaises(ValueError):
            b.records(dict(events=bad))


if __name__ == "__main__":
    unittest.main()
