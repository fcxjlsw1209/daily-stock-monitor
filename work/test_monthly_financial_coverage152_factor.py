"""Synthetic tests only; no original source or strategy results are loaded."""
import unittest
from datetime import date, timedelta
from fractions import Fraction

import numpy as np
import monthly_financial_coverage152_factor as f


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
    return dict(event_id=event_id, symbol="KO", source=source,
                release_date=release, available_from=available, fields=fields,
                identity=dict(issuer_id="synthetic-issuer", security_id="KO",
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
        result = f.latest([catalog(old), catalog(new)], {"x": old}, "KO", "2020-03-03")
        self.assertEqual(result["reason"], "latest_annual_event_unprocessed")

    def test_held_latest_blocks_old(self):
        old, new = event(), event("y", "2020-03-02", "2020-03-03")
        new["ni_assets_admission"].update(admitted=False, reasons=["accounting_transition"])
        result = f.latest([catalog(old), catalog(new)], {"x": old, "y": new},
                          "KO", "2020-03-03")
        self.assertEqual(result["reason"], "latest_event_held")

    def test_future_event_does_not_block_old(self):
        old, new = event(), event("y", "2020-03-02", "2020-03-03")
        result = f.latest([catalog(old), catalog(new)], {"x": old, "y": new},
                          "KO", "2020-03-02")
        self.assertEqual(result["event_id"], "x")
        self.assertTrue(result["available"])

    def test_ambiguous_latest(self):
        a, b = event(), event("y")
        result = f.latest([catalog(a), catalog(b)], {"x": a, "y": b}, "KO", "2020-03-03")
        self.assertEqual(result["reason"], "ambiguous_latest_annual_event")

    def test_unverified_clock_blocks_older_value(self):
        old, new = event(), event("y", "2020-03-02", "2020-03-03")
        row = catalog(new)
        row["clock_review"] = dict(verified=False, first_pinned_session=None)
        row["available_from"] = None
        result = f.latest([catalog(old), row], {"x": old, "y": new}, "KO", "2020-03-03")
        self.assertEqual(result["reason"], "latest_annual_event_clock_unverified")

    def test_unverified_same_day_not_yet_known(self):
        old, new = event(), event("y", "2020-03-02", "2020-03-03")
        row = catalog(new)
        row["clock_review"] = dict(verified=False, first_pinned_session=None)
        result = f.latest([catalog(old), row], {"x": old}, "KO", "2020-03-02")
        self.assertEqual(result["event_id"], "x")
        self.assertTrue(result["available"])

    def test_verified_clock_mismatch_rejected(self):
        e = event()
        row = catalog(e)
        row["clock_review"]["first_pinned_session"] = "2020-02-05"
        with self.assertRaisesRegex(ValueError, "clock mismatch"):
            f.latest([row], {"x": e}, "KO", "2020-03-03")

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
        for key, value in [("issuer_id", ""), ("security_id", "LRCX"),
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
            f.latest([row], {"x": e}, "KO", "2020-02-04")

    def test_preserve_old_and_missing_cells(self):
        names = ["AAPL", "KO"] + ["S"+str(i) for i in range(30)] + ["E"+str(i) for i in range(17)]
        meta = dict(dates=["2020-02-03", "2020-02-04"],
                    symbols=[names.copy(), names.copy()], etfs=names[-17:])
        old = np.full((2, 49), np.nan)
        old[:, 0] = .125
        before = old.tobytes()
        e = event()
        data = dict(schema_version=1, method=f.METHOD, events=[e], latest_event_catalog=[catalog(e)])
        result = f.build(meta, old, old.copy(), old.copy(), data)
        self.assertEqual(old.tobytes(), before)
        self.assertEqual(result["arrays"]["earnings_original"].tobytes(), before)
        self.assertTrue(np.isnan(result["arrays"]["earnings_expanded"][0, 1]))
        self.assertAlmostEqual(result["arrays"]["earnings_expanded"][1, 1], 1/11)
        self.assertEqual(result["audit"]["added_cells"], 1)
        shortened = f.build(dict(meta, dates=meta["dates"][:1], symbols=meta["symbols"][:1]), old[:1], old[:1].copy(), old[:1].copy(), data)
        self.assertEqual(shortened["arrays"]["earnings_expanded"].tobytes(),
                         result["arrays"]["earnings_expanded"][:1].tobytes())

    def test_batch1_values_and_nan_payload_are_preserved(self):
        names=["AAPL","AMZN","KO"]+["S"+str(i) for i in range(29)]+["E"+str(i) for i in range(17)]
        meta=dict(dates=["2020-02-04"],symbols=[names],etfs=names[-17:])
        original=np.full((1,49),np.nan);original[0,0]=.1
        prior=original.copy();prior[0,1]=-.2
        e=event();data=dict(schema_version=1,method=f.METHOD,events=[e],latest_event_catalog=[catalog(e)])
        result=f.build(meta,original,prior,prior.copy(),data)
        self.assertEqual(result["arrays"]["earnings_batch1"].tobytes(),prior.tobytes())
        self.assertEqual(result["arrays"]["earnings_expanded"][0,:2].tobytes(),prior[0,:2].tobytes())
        self.assertTrue(result["arrays"]["expanded_available"][0,2])
        prior[0,0]=.11
        with self.assertRaisesRegex(ValueError,"nonbatch1"):
            f.build(meta,original,prior,prior.copy(),data)

    def test_prior150_cannot_contain_new_issuer(self):
        names=["KO"]+["S"+str(i) for i in range(31)]+["E"+str(i) for i in range(17)]
        meta=dict(dates=["2020-02-04"],symbols=[names],etfs=names[-17:])
        original=np.full((1,49),np.nan);prior=original.copy();prior[0,0]=.1
        with self.assertRaisesRegex(ValueError,"Prior150 scope"):
            f.build(meta,original,prior,prior.copy(),dict(schema_version=1,method=f.METHOD,events=[],latest_event_catalog=[]))

    def test_duplicate_catalog_rejected(self):
        names=["KO"]+["S"+str(i) for i in range(31)]+["E"+str(i) for i in range(17)]
        meta=dict(dates=["2020-02-04"],symbols=[names],etfs=names[-17:]);a=np.full((1,49),np.nan)
        e=event()
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            f.build(meta,a,a.copy(),a.copy(),dict(schema_version=1,method=f.METHOD,events=[e],latest_event_catalog=[catalog(e),catalog(e)]))

    def test_original_new_symbol_leak_rejected(self):
        names = ["KO"] + ["S"+str(i) for i in range(31)] + ["E"+str(i) for i in range(17)]
        meta = dict(dates=["2020-02-04"], symbols=[names], etfs=names[-17:])
        old = np.full((1, 49), np.nan)
        old[0, 0] = .1
        with self.assertRaises(ValueError):
            f.build(meta, old, old.copy(), old.copy(), dict(schema_version=1, method=f.METHOD,
                                   events=[], latest_event_catalog=[]))



class CumulativeControlTests(unittest.TestCase):
    def fixture(self):
        names=['AAPL','AMZN','INTC','KO','LRCX']+['S'+str(i) for i in range(27)]+['E'+str(i) for i in range(17)]
        meta=dict(dates=['2020-02-04','2020-03-03','2020-04-03'],symbols=[names.copy() for _ in range(3)],etfs=names[-17:])
        original=np.full((3,49),np.nan);original[:,0]=[-0.,.1,.2]
        # Preserve noncanonical NaN payloads, not just NaN equality.
        original.view(np.uint64)[:,5]=np.uint64(0x7ff8000000000011)
        batch1=original.copy();batch1[:,1]=[-.2,0.,.3]
        batch2=batch1.copy();batch2[:,2]=[.2,-.3,.4]
        e=event();data=dict(schema_version=1,method=f.METHOD,events=[e],latest_event_catalog=[catalog(e)])
        return meta,original,batch1,batch2,data

    def test_three_prior_controls_and_all_nonnew_cells_byte_exact(self):
        meta,*arrays,data=self.fixture();before=[a.tobytes() for a in arrays]
        result=f.build(meta,*arrays,data)
        for key,a in zip(('earnings_original','earnings_batch1','earnings_batch2'),arrays):
            self.assertEqual(result['arrays'][key].tobytes(),a.tobytes())
        expanded=result['arrays']['earnings_expanded']
        for j,symbol in enumerate(meta['symbols'][0]):
            if symbol not in f.NEW_SYMBOLS:self.assertEqual(expanded[:,j].tobytes(),arrays[2][:,j].tobytes())
        self.assertEqual(before,[a.tobytes() for a in arrays])
        self.assertEqual(len(result['arrays']),8)
        for prefix in ('original','batch1','batch2','expanded'):
            np.testing.assert_array_equal(result['arrays'][prefix+'_available'],np.isfinite(result['arrays']['earnings_'+prefix]))

    def test_prior151_may_only_add_batch2(self):
        for column in (0,1,3):
            meta,original,batch1,batch2,data=self.fixture();batch2[0,column]=.9
            with self.subTest(column=column),self.assertRaises(ValueError):f.build(meta,original,batch1,batch2,data)

    def test_old_source_events_are_not_recomputed(self):
        meta,*arrays,data=self.fixture()
        result=f.build(meta,*arrays,data)
        for symbol in ('AMZN','CSCO','INTC','KLAC'):
            e=event(symbol);e['symbol']=symbol;e['fields']={};data['events'].append(e)
            data['latest_event_catalog'].append(catalog(e))
        other=f.build(meta,*arrays,data)
        for key in result['arrays']:self.assertEqual(result['arrays'][key].tobytes(),other['arrays'][key].tobytes())

    def test_all_prefixes_and_future_held_event(self):
        meta,*arrays,data=self.fixture();future=event('future','2020-03-02','2020-03-03')
        future['ni_assets_admission'].update(admitted=False,reasons=['accounting_comparability_unresolved'])
        data['events'].append(future);data['latest_event_catalog'].append(catalog(future))
        result=f.build(meta,*arrays,data)
        self.assertTrue(result['arrays']['expanded_available'][0,3])
        self.assertFalse(result['arrays']['expanded_available'][1:,3].any())
        for n in range(4):
            prefix=f.build(dict(meta,dates=meta['dates'][:n],symbols=meta['symbols'][:n]),*[a[:n] for a in arrays],data)
            for key,value in result['arrays'].items():self.assertEqual(prefix['arrays'][key].tobytes(),value[:n].tobytes())

    def test_frozen_null_new_event_cannot_be_replaced(self):
        meta,*arrays,data=self.fixture();e=data['events'][0]
        e['fields']=dict.fromkeys(f.FIELDS);e['ni_assets_admission'].update(admitted=False,reasons=['missing_annual_net_income'])
        result=f.build(meta,*arrays,data)
        self.assertEqual(result['arrays']['earnings_expanded'].tobytes(),arrays[2].tobytes())
        self.assertEqual(result['audit']['added_cells'],0)

    def test_lrcx_as_well_as_ko_is_supported(self):
        e=event();e['symbol']='LRCX';e['identity']['security_id']='LRCX'
        self.assertTrue(f.evaluate(e,'2020-03-03')['available'])

    def test_large_exact_numbers_near_cancel_do_not_round_early(self):
        e=event();e['fields']['annual_net_income']['value_decimal']='0.000000000000000000001'
        e['fields']['current_total_assets']['value_decimal']='100000000000000000000.1'
        e['fields']['prior_total_assets']['value_decimal']='100000000000000000000.3'
        expected=Fraction('0.000000000000000000001')/((Fraction('100000000000000000000.1')+Fraction('100000000000000000000.3'))/2)
        self.assertEqual(f.evaluate(e,'2020-03-03')['earnings_bounded'],f.bounded(expected))

    def test_duplicate_event_and_unexpected_issuer_rejected(self):
        for kind in ('duplicate','unexpected'):
            meta,*arrays,data=self.fixture()
            if kind=='duplicate':data['events'].append(data['events'][0])
            else:data['events'][0]['symbol']='UNDECLARED'
            with self.subTest(kind=kind),self.assertRaises(ValueError):f.build(meta,*arrays,data)

    def test_control_dtype_shape_and_infinity_rejected(self):
        for kind in ('dtype','shape','infinity'):
            meta,original,batch1,batch2,data=self.fixture()
            if kind=='dtype':batch2=batch2.astype(np.float32)
            elif kind=='shape':batch2=batch2[:,:48]
            else:batch2[0,2]=np.inf
            with self.subTest(kind=kind),self.assertRaises(ValueError):f.build(meta,original,batch1,batch2,data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
