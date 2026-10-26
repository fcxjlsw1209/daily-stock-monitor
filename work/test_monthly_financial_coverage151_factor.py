"""Synthetic tests only; no original source or strategy results are loaded."""
import unittest
from datetime import date, timedelta
from fractions import Fraction

import numpy as np
import monthly_financial_coverage151_factor as f


def event(event_id="x", release="2020-02-03", available="2020-02-04"):
    source = dict(path="synthetic.html", sha256="a"*64)
    period = dict(period_start="2019-01-01", period_end="2019-12-31", days_inclusive=365)
    fields = {}
    for name, value in zip(f.FIELDS, ("10", "120", "80")):
        fields[name] = dict(source=source.copy(), available_from=available,
                            currency_literal="$", currency_iso=None,
                            as_of="2018-12-31" if name == "prior_total_assets" else "2019-12-31",
                            period=period.copy() if name == "annual_net_income" else None,
                            value_decimal=value, scale="1000000")
    return dict(event_id=event_id, symbol="INTC", source=source,
                release_date=release, available_from=available, fields=fields,
                identity=dict(issuer_id="synthetic-issuer", security_id="INTC",
                              identity_valid_from=available,
                              identity_valid_through=(date(2019, 12, 31)+timedelta(days=550)).isoformat()),
                ni_assets_admission=dict(admitted=True, reasons=[], definition=f.DEFINITION))


def catalog(e):
    return dict({k: e[k] for k in ("event_id", "symbol", "release_date", "available_from")},
                clock_review=dict(verified=True, first_pinned_session=e["available_from"]))


class CoverageTests(unittest.TestCase):
    def test_exact_ratio(self):
        result = f.evaluate(event(), "2020-02-04")
        self.assertTrue(result["available"])
        self.assertEqual(result["earnings_bounded"], f.bounded(Fraction(1, 10)))

    def test_signed_income(self):
        for value in ("-10", "0"):
            e = event()
            e["fields"]["annual_net_income"]["value_decimal"] = value
            result = f.evaluate(e, "2020-03-01")
            self.assertTrue(result["available"])
            self.assertEqual(result["earnings_bounded"], f.bounded(Fraction(int(value), 100)))

    def test_unprocessed_latest_blocks_old(self):
        old, new = event(), event("y", "2020-03-02", "2020-03-03")
        result = f.latest([catalog(old), catalog(new)], {"x": old}, "INTC", "2020-03-03")
        self.assertEqual(result["reason"], "latest_annual_event_unprocessed")

    def test_held_latest_blocks_old(self):
        old, new = event(), event("y", "2020-03-02", "2020-03-03")
        new["ni_assets_admission"].update(admitted=False, reasons=["accounting_transition"])
        result = f.latest([catalog(old), catalog(new)], {"x": old, "y": new},
                          "INTC", "2020-03-03")
        self.assertEqual(result["reason"], "latest_event_held")

    def test_future_event_does_not_block_old(self):
        old, new = event(), event("y", "2020-03-02", "2020-03-03")
        result = f.latest([catalog(old), catalog(new)], {"x": old, "y": new},
                          "INTC", "2020-03-02")
        self.assertEqual(result["event_id"], "x")
        self.assertTrue(result["available"])

    def test_ambiguous_latest(self):
        a, b = event(), event("y")
        result = f.latest([catalog(a), catalog(b)], {"x": a, "y": b}, "INTC", "2020-03-03")
        self.assertEqual(result["reason"], "ambiguous_latest_annual_event")

    def test_unverified_clock_blocks_older_value(self):
        old, new = event(), event("y", "2020-03-02", "2020-03-03")
        row = catalog(new)
        row["clock_review"] = dict(verified=False, first_pinned_session=None)
        row["available_from"] = None
        result = f.latest([catalog(old), row], {"x": old, "y": new}, "INTC", "2020-03-03")
        self.assertEqual(result["reason"], "latest_annual_event_clock_unverified")

    def test_unverified_same_day_not_yet_known(self):
        old, new = event(), event("y", "2020-03-02", "2020-03-03")
        row = catalog(new)
        row["clock_review"] = dict(verified=False, first_pinned_session=None)
        result = f.latest([catalog(old), row], {"x": old}, "INTC", "2020-03-02")
        self.assertEqual(result["event_id"], "x")
        self.assertTrue(result["available"])

    def test_verified_clock_mismatch_rejected(self):
        e = event()
        row = catalog(e)
        row["clock_review"]["first_pinned_session"] = "2020-02-05"
        with self.assertRaisesRegex(ValueError, "clock mismatch"):
            f.latest([row], {"x": e}, "INTC", "2020-03-03")

    def test_stale_report(self):
        self.assertEqual(f.evaluate(event(), "2021-07-04")["reason"], "stale_latest_event")

    def test_mixed_scale_normalization(self):
        e = event()
        e["fields"]["annual_net_income"].update(scale="1000", value_decimal="10000")
        self.assertEqual(f.evaluate(e, "2020-03-01")["earnings_bounded"],
                         f.evaluate(event(), "2020-03-01")["earnings_bounded"])

    def test_invalid_fields(self):
        cases = [("prior_total_assets", "as_of", "2018-12-30"),
                 ("current_total_assets", "value_decimal", "0"),
                 ("current_total_assets", "value_decimal", "-1"),
                 ("annual_net_income", "scale", "-1"),
                 ("annual_net_income", "value_decimal", "NaN"),
                 ("annual_net_income", "available_from", "2020-02-03"),
                 ("annual_net_income", "currency_literal", "EUR"),
                 ("annual_net_income", "currency_iso", "USD")]
        for name, field, value in cases:
            with self.subTest(name=name, field=field, value=value):
                e = event()
                e["fields"][name][field] = value
                self.assertFalse(f.evaluate(e, "2020-03-01")["available"])

    def test_annual_duration(self):
        e = event()
        e["fields"]["annual_net_income"]["period"]["period_start"] = "2019-10-01"
        self.assertEqual(f.evaluate(e, "2020-03-01")["reason"], "invalid_annual_period")

    def test_no_early_availability(self):
        self.assertFalse(f.evaluate(event(), "2020-02-03")["available"])

    def test_identity_boundaries(self):
        for key, value in [("issuer_id", ""), ("security_id", "KLAC"),
                           ("identity_valid_from", "2020-02-03"),
                           ("identity_valid_through", "2025-01-01")]:
            e = event()
            e["identity"][key] = value
            self.assertFalse(f.evaluate(e, "2020-02-04")["available"])

    def test_catalog_release_date_mismatch(self):
        e = event()
        row = catalog(e)
        row["release_date"] = "2020-02-02"
        with self.assertRaisesRegex(ValueError, "release date mismatch"):
            f.latest([row], {"x": e}, "INTC", "2020-02-04")

    def test_preserve_old_and_missing_cells(self):
        names = ["AAPL", "INTC"] + ["S"+str(i) for i in range(30)] + ["E"+str(i) for i in range(17)]
        meta = dict(dates=["2020-02-03", "2020-02-04"],
                    symbols=[names.copy(), names.copy()], etfs=names[-17:])
        old = np.full((2, 49), np.nan)
        old[:, 0] = .125
        before = old.tobytes()
        e = event()
        data = dict(schema_version=1, method=f.METHOD, events=[e], latest_event_catalog=[catalog(e)])
        result = f.build(meta, old, old.copy(), data)
        self.assertEqual(old.tobytes(), before)
        self.assertEqual(result["arrays"]["earnings_original"].tobytes(), before)
        self.assertTrue(np.isnan(result["arrays"]["earnings_expanded"][0, 1]))
        self.assertAlmostEqual(result["arrays"]["earnings_expanded"][1, 1], 1/11)
        self.assertEqual(result["audit"]["added_cells"], 1)
        shortened = f.build(dict(meta, dates=meta["dates"][:1], symbols=meta["symbols"][:1]), old[:1], old[:1].copy(), data)
        self.assertEqual(shortened["arrays"]["earnings_expanded"].tobytes(),
                         result["arrays"]["earnings_expanded"][:1].tobytes())

    def test_batch1_values_and_nan_payload_are_preserved(self):
        names=["AAPL","AMZN","INTC"]+["S"+str(i) for i in range(29)]+["E"+str(i) for i in range(17)]
        meta=dict(dates=["2020-02-04"],symbols=[names],etfs=names[-17:])
        original=np.full((1,49),np.nan);original[0,0]=.1
        prior=original.copy();prior[0,1]=-.2
        e=event();data=dict(schema_version=1,method=f.METHOD,events=[e],latest_event_catalog=[catalog(e)])
        result=f.build(meta,original,prior,data)
        self.assertEqual(result["arrays"]["earnings_batch1"].tobytes(),prior.tobytes())
        self.assertEqual(result["arrays"]["earnings_expanded"][0,:2].tobytes(),prior[0,:2].tobytes())
        self.assertTrue(result["arrays"]["expanded_available"][0,2])
        prior[0,0]=.11
        with self.assertRaisesRegex(ValueError,"nonbatch1"):
            f.build(meta,original,prior,data)

    def test_prior150_cannot_contain_new_issuer(self):
        names=["INTC"]+["S"+str(i) for i in range(31)]+["E"+str(i) for i in range(17)]
        meta=dict(dates=["2020-02-04"],symbols=[names],etfs=names[-17:])
        original=np.full((1,49),np.nan);prior=original.copy();prior[0,0]=.1
        with self.assertRaisesRegex(ValueError,"Prior150 scope"):
            f.build(meta,original,prior,dict(schema_version=1,method=f.METHOD,events=[],latest_event_catalog=[]))

    def test_duplicate_catalog_rejected(self):
        names=["INTC"]+["S"+str(i) for i in range(31)]+["E"+str(i) for i in range(17)]
        meta=dict(dates=["2020-02-04"],symbols=[names],etfs=names[-17:]);a=np.full((1,49),np.nan)
        e=event()
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            f.build(meta,a,a.copy(),dict(schema_version=1,method=f.METHOD,events=[e],latest_event_catalog=[catalog(e),catalog(e)]))

    def test_original_new_symbol_leak_rejected(self):
        names = ["INTC"] + ["S"+str(i) for i in range(31)] + ["E"+str(i) for i in range(17)]
        meta = dict(dates=["2020-02-04"], symbols=[names], etfs=names[-17:])
        old = np.full((1, 49), np.nan)
        old[0, 0] = .1
        with self.assertRaises(ValueError):
            f.build(meta, old, old.copy(), dict(schema_version=1, method=f.METHOD,
                                   events=[], latest_event_catalog=[]))


if __name__ == "__main__":
    unittest.main()
