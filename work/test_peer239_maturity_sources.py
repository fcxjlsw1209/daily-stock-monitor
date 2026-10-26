"""Pure metadata tests: no source loader, prices, NPZ, model or strategy calls."""
from copy import deepcopy
from datetime import date, timedelta
import calendar
import unittest

import peer239_maturity_sources as producer
import peer239_maturity_independent as independent


def endpoint(symbol, day, role, status="valid", reason=None):
    return dict(symbol=symbol,date=day,role=role,status=status,reason=reason)


def make_fixture(dates, names=None, etfs=None, new_symbol=None):
    names=list(names or ["A","B","C","NEW","ETF"])
    etfs=list(etfs if etfs is not None else ["ETF"])
    meta=dict(dates=list(dates),symbols=[names[:] for _ in dates],etfs=etfs)
    events=[];records=[]
    for q,day in enumerate(dates):
        partial=day=="2016-08-17"
        stock=[j for j,s in enumerate(names) if s not in etfs]
        d=date.fromisoformat(day)
        start=date(d.year-1,d.month,min(d.day,calendar.monthrange(d.year-1,d.month)[1])).isoformat()
        end=(d-timedelta(days=61)).isoformat()
        source_dates=[] if partial else [start,end]
        positions=[] if partial else [q*21,q*21+210]
        events.append(dict(query_index=q,date=day,initial_partial=partial,stock_indices=stock,
                           source_positions=positions[:],source_dates=source_dates[:]))
        for j in stock:
            symbol=names[j];sector="XLK";gap=[]
            points=[] if partial else [endpoint(symbol,start,"security_start"),endpoint(symbol,end,"security_end")]
            if symbol==new_symbol and not partial:
                sector=None;gap.append("missing_sector")
                for point in points:
                    if point["date"]<"2020-06-08":
                        point.update(status="missing",reason="missing_value")
                        gap.append("endpoint:"+point["role"]+":missing_value")
            records.append(dict(query_index=q,column_index=j,date=day,symbol=symbol,
                issuer_key="issuer:"+symbol,original_sector=sector,sector=sector,mapping_origin="synthetic",
                mapping_repair=None,mapping_evidence=[],peer_indices=[],peer_issuer_count=0,
                structural_peer_support=False,initial_partial=partial,source_positions=positions[:],
                source_dates=source_dates[:],endpoints=points,gap_reasons=gap))
    source=dict(schema="peer239_source_report_v1",status="SOURCE_GAPS",admitted=False,
                events=events,records=records,gaps=[deepcopy(r) for r in records if r["gap_reasons"]],
                counts=dict(events=len(dates),stock_cells=len(records)),source_only=True,
                strict_pit=False,momentum_computed=False,rankings_computed=False,
                orders_computed=False,returns_computed=False,model_fits=0)
    rules=[]
    if new_symbol:
        rules=[dict(symbol=new_symbol,issuer_key="issuer:"+new_symbol,not_before="2020-06-08",
                    available_from="2020-06-09",kind="exchange_new_listing_lower_bound",evidence_id="synthetic:new")]
    return source,meta,dict(schema="peer239_history_bound_policy_v1",securities=rules)


def fixture():
    return make_fixture(["2016-08-17","2020-06-30","2020-07-31","2020-08-31"],new_symbol="NEW")


def physical_prefix(args,n):
    source,meta,policy=deepcopy(args)
    source["events"]=source["events"][:n]
    source["records"]=[r for r in source["records"] if r["query_index"]<n]
    source["gaps"]=[r for r in source["gaps"] if r["query_index"]<n]
    source["counts"]=dict(events=n,stock_cells=len(source["records"]))
    meta["dates"]=meta["dates"][:n];meta["symbols"]=meta["symbols"][:n]
    return source,meta,policy


def record(source,q=1,symbol="NEW"):
    return next(r for r in source["records"] if r["query_index"]==q and r["symbol"]==symbol)


def change_endpoint(row,index,status,reason):
    point=row["endpoints"][index]
    row["gap_reasons"]=[g for g in row["gap_reasons"] if not g.startswith("endpoint:"+point["role"]+":")]
    point.update(status=status,reason=reason)
    if status!="valid":row["gap_reasons"].append("endpoint:"+point["role"]+":"+str(reason))


class MaturityTests(unittest.TestCase):
    def cross(self,args,limit=None):
        before=deepcopy(args)
        out=producer.project(*args,limit=limit)
        expected=independent.inspect(*args,limit=limit)
        independent.exact(out,expected)
        receipt=independent.check_sources(*args,out,limit=limit)
        self.assertTrue(all(receipt[k] is True for k in independent.FLAGS))
        self.assertEqual(args,before)
        self.assertFalse(receipt["actual_price_values_loaded"])
        return out

    def test_source_gap_projection_preserves_inventory_and_original_gaps(self):
        args=fixture();out=self.cross(args)
        self.assertEqual(out["counts"],dict(events=4,stock_cells=16,mature_cells=9,warmup_cells=3,
            partial_cells=4,supported_cells=9,deferred_sector_cells=3,deferred_endpoints=5,gap_cells=0))
        self.assertTrue(out["admitted"])
        self.assertFalse(args[0]["admitted"])
        row=record(out)
        self.assertEqual(row["original_gap_reasons"],record(args[0])["gap_reasons"])
        self.assertEqual(row["history_status"],"official_pre_inception")
        self.assertEqual(row["peer_indices"],[])

    def test_all_small_logical_physical_prefixes_and_zero(self):
        args=fixture()
        for n in range(5):
            logical=self.cross(args,n);physical=self.cross(physical_prefix(args,n))
            independent.exact(logical,physical)

    def test_full_122_by_32_inventory_five_prefixes(self):
        dates=["2016-08-17"]
        y,m=2016,8
        for _ in range(121):
            dates.append(date(y,m,calendar.monthrange(y,m)[1]).isoformat())
            y,m=(y+1,1) if m==12 else (y,m+1)
        stocks=["S%02d"%i for i in range(32)];etfs=["ETF%02d"%i for i in range(17)]
        args=make_fixture(dates,stocks+etfs,etfs)
        full=self.cross(args)
        self.assertEqual(full["counts"]["stock_cells"],3904)
        self.assertEqual(full["counts"]["mature_cells"],3872)
        self.assertEqual(full["counts"]["partial_cells"],32)
        for n in (0,1,53,88,122):
            independent.exact(self.cross(args,n),self.cross(physical_prefix(args,n)))

    def test_availability_boundary_exact_day_and_future_not_retroactive(self):
        args=fixture();args[2]["securities"][0]["available_from"]="2020-06-30"
        self.assertEqual(record(self.cross(args))["history_status"],"official_pre_inception")
        args[2]["securities"][0]["available_from"]="2020-07-01"
        out=self.cross(args)
        self.assertEqual(record(out)["history_status"],"observed_full_window")
        self.assertIsNone(record(out)["history_bound"])
        self.assertFalse(out["admitted"])
        self.assertEqual(record(out,2)["history_status"],"official_pre_inception")

    def test_future_evidence_never_changes_earlier_prefix(self):
        args=fixture();args[2]["securities"][0]["available_from"]="2021-01-01"
        no_policy=deepcopy(args);no_policy[2]["securities"]=[]
        independent.exact(self.cross(args),self.cross(no_policy))

    def test_start_equal_bound_is_mature_not_deferred(self):
        args=fixture();bound=args[2]["securities"][0]
        bound["not_before"]=record(args[0])["source_dates"][0]
        out=self.cross(args);row=record(out)
        self.assertEqual(row["history_status"],"observed_full_window")
        self.assertEqual(row["deferred_gap_reasons"],[])
        self.assertIn("mature_endpoint_not_valid:security_start",row["gap_reasons"])

    def test_end_equal_bound_is_post_bound_and_must_be_valid(self):
        args=fixture();row=record(args[0]);end=row["source_dates"][1]
        args[2]["securities"][0]["not_before"]=end
        out=self.cross(args)
        self.assertIn("post_bound_endpoint_missing:security_end",record(out)["gap_reasons"])
        self.assertNotIn("endpoint:security_end:missing_value",record(out)["deferred_gap_reasons"])

    def test_valid_old_reused_ticker_prices_are_contamination_not_warmup_fill(self):
        args=fixture();change_endpoint(record(args[0]),0,"valid",None)
        out=self.cross(args)
        self.assertFalse(out["admitted"])
        self.assertIn("pre_inception_price_or_unclassified_missing:security_start",record(out)["gap_reasons"])

    def test_invalid_pre_bound_endpoints_never_deferred(self):
        for reason in ("invalid_boolean","invalid_nonreal","nonfinite","nonpositive"):
            with self.subTest(reason=reason):
                args=fixture();change_endpoint(record(args[0]),0,"invalid",reason)
                row=record(self.cross(args))
                self.assertIn("endpoint:security_start:"+reason,row["gap_reasons"])
                self.assertIn("pre_inception_price_or_unclassified_missing:security_start",row["gap_reasons"])

    def test_all_three_genuine_missing_reasons_defer(self):
        for reason in ("missing_value","missing_column","missing_row"):
            with self.subTest(reason=reason):
                args=fixture();change_endpoint(record(args[0]),0,"missing",reason)
                out=self.cross(args)
                self.assertTrue(out["admitted"])
                self.assertIn("endpoint:security_start:"+reason,record(out)["deferred_gap_reasons"])

    def test_unknown_missing_reason_and_unrecorded_gap_cannot_be_excused(self):
        args=fixture();change_endpoint(record(args[0]),0,"missing","unclassified")
        self.assertFalse(self.cross(args)["admitted"])
        args=fixture();record(args[0])["gap_reasons"].remove("endpoint:security_start:missing_value")
        self.assertIn("pre_inception_price_or_unclassified_missing:security_start",record(self.cross(args))["gap_reasons"])

    def test_post_bound_missing_stays_a_gap_while_pre_bound_deferred(self):
        args=fixture();change_endpoint(record(args[0],3),1,"missing","missing_value")
        row=record(self.cross(args),3)
        self.assertIn("endpoint:security_start:missing_value",row["deferred_gap_reasons"])
        self.assertIn("post_bound_endpoint_missing:security_end",row["gap_reasons"])
        self.assertIn("endpoint:security_end:missing_value",row["gap_reasons"])

    def test_mature_missing_and_sector_gaps_never_deferred(self):
        args=fixture();row=record(args[0],1,"A")
        row["sector"]=None;row["gap_reasons"].append("missing_sector")
        change_endpoint(row,0,"missing","missing_row")
        out=self.cross(args);row=record(out,1,"A")
        self.assertEqual(row["deferred_gap_reasons"],[])
        self.assertIn("missing_sector",row["gap_reasons"])
        self.assertIn("mature_endpoint_not_valid:security_start",row["gap_reasons"])

    def test_wrong_security_or_issuer_policy_does_not_waive_gaps(self):
        for key,value in (("symbol","WRONG"),("issuer_key","issuer:WRONG")):
            args=fixture();args[2]["securities"][0][key]=value
            out=self.cross(args)
            self.assertEqual(out["counts"]["warmup_cells"],0)
            self.assertFalse(out["admitted"])

    def test_missing_identity_stays_a_gap(self):
        args=fixture();record(args[0])["issuer_key"]=None
        row=record(self.cross(args))
        self.assertIn("missing_issuer",row["gap_reasons"])
        self.assertEqual(row["history_status"],"observed_full_window")

    def test_duplicate_identity_not_hidden_and_peer_count_is_distinct_issuers(self):
        args=fixture();record(args[0],1,"B")["issuer_key"]="issuer:A"
        out=self.cross(args);a=record(out,1,"A");b=record(out,1,"B");c=record(out,1,"C")
        self.assertIn("duplicate_issuer",a["gap_reasons"])
        self.assertIn("duplicate_issuer",b["gap_reasons"])
        self.assertEqual(a["peer_indices"],[2])
        self.assertEqual(c["peer_indices"],[0,1])
        self.assertEqual(c["peer_issuer_count"],1)
        self.assertFalse(c["structural_peer_support"])

    def test_unrelated_original_gap_never_disappears(self):
        args=fixture();record(args[0])["gap_reasons"].append("independent_original_identity_warning")
        self.assertIn("independent_original_identity_warning",record(self.cross(args))["gap_reasons"])

    def test_mature_peer_domain_before_eligibility_and_same_supported_mask(self):
        args=fixture();out=self.cross(args)
        altered=deepcopy(args);altered[1]["eligible"]=[[False]*5 for _ in range(4)]
        independent.exact(out,self.cross(altered))
        for symbol in ("A","B","C"):
            self.assertTrue(record(out,1,symbol)["structural_peer_support"])
            self.assertEqual(record(out,1,symbol)["peer_issuer_count"],2)
        self.assertFalse(record(out)["structural_peer_support"])

    def test_peer_sector_boundary_and_one_peer_structural_not_source_gap(self):
        args=make_fixture(["2020-06-30"],names=["A","B","C"],etfs=[])
        record(args[0],0,"C")["sector"]="XLF"
        out=self.cross(args)
        self.assertTrue(out["admitted"])
        self.assertEqual(out["counts"]["supported_cells"],0)
        self.assertEqual(record(out,0,"A")["peer_indices"],[1])

    def test_partial_has_no_source_or_history_bound_or_peers(self):
        args=fixture();row=record(self.cross(args),0)
        self.assertEqual(row["history_status"],"initial_partial")
        self.assertIsNone(row["history_bound"])
        self.assertEqual(row["endpoints"],[])
        self.assertEqual(row["peer_indices"],[])
        args[0]["records"][0]["endpoints"]=[endpoint("A","2015-01-01","security_start")]
        self.assert_both_invalid(args)

    def test_adr_start_kind_and_nonblank_evidence_identity(self):
        args=fixture();args[2]["securities"][0]["kind"]="issuer_actual_adr_start"
        self.assertTrue(self.cross(args)["admitted"])
        for key in ("symbol","issuer_key","evidence_id"):
            bad=deepcopy(args);bad[2]["securities"][0][key]=""
            self.assert_both_invalid(bad)

    def assert_both_invalid(self,args,limit=None):
        for fn in (producer.project,independent.inspect):
            with self.assertRaises((ValueError,KeyError,TypeError,IndexError)):
                fn(*args,limit=limit)

    def test_bad_policy_schema_duplicate_bounds_and_prepublication_kind(self):
        for operation in (lambda p:p.update(schema="wrong"),
                          lambda p:p["securities"].append(deepcopy(p["securities"][0])),
                          lambda p:p["securities"][0].update(kind="guessed_ipo"),
                          lambda p:p["securities"][0].update(available_from="2020-06-07"),
                          lambda p:p["securities"][0].update(not_before="2020-6-08"),
                          lambda p:p["securities"][0].update(extra=True)):
            args=fixture();operation(args[2]);self.assert_both_invalid(args)

    def test_strict_source_only_and_fit_zero_types(self):
        for key,value in (("source_only",1),("momentum_computed",True),("rankings_computed",True),
                          ("orders_computed",True),("returns_computed",True),("model_fits",False),
                          ("model_fits",0.0),("model_fits",1)):
            args=fixture();args[0][key]=value;self.assert_both_invalid(args)

    def test_full_inventory_and_event_cell_endpoint_identity_failures(self):
        operations=(lambda a:a[0]["records"].pop(4),
                    lambda a:a[0]["events"][1].update(date="2020-06-29"),
                    lambda a:record(a[0]).update(symbol="WRONG"),
                    lambda a:record(a[0]).update(source_dates=["1999-01-01","1999-02-01"]),
                    lambda a:record(a[0])["endpoints"][0].update(symbol="WRONG"),
                    lambda a:record(a[0])["endpoints"][0].update(date="1999-01-01"))
        for operation in operations:
            args=fixture();operation(args);self.assert_both_invalid(args)

    def test_strict_prefix_limits_and_future_record_poison_is_not_consumed(self):
        args=fixture()
        for n in (-1,5,True,1.0):self.assert_both_invalid(args,n)
        expected=self.cross(args,2)
        args[0]["events"][3]["date"]="poison"
        record(args[0],3)["endpoints"]=[{"poison":True}]
        args[1]["symbols"][3]=["poison"]
        independent.exact(expected,self.cross(args,2))

    def test_receipt_exact_type_content_and_counter_corruption_rejected(self):
        args=fixture();out=self.cross(args)
        for operation in (lambda x:x["counts"].update(warmup_cells=3.0),
                          lambda x:x["counts"].update(gap_cells=1),
                          lambda x:x.update(admitted=1),
                          lambda x:record(x)["deferred_gap_reasons"].pop(),
                          lambda x:record(x,1,"A").update(peer_indices=[]),
                          lambda x:x.update(extra="unreviewed")):
            bad=deepcopy(out);operation(bad)
            with self.assertRaises(ValueError):independent.check_sources(*args,bad)

    def test_returned_report_is_detached_from_inputs(self):
        args=fixture();before=deepcopy(args);out=self.cross(args)
        record(out)["history_bound"]["evidence_id"]="changed"
        record(out)["original_gap_reasons"].append("changed")
        out["events"][1]["source_dates"].append("changed")
        self.assertEqual(args,before)

    def test_no_raw_or_first_vintage_certification_claim(self):
        args=fixture();out=self.cross(args);receipt=independent.check_sources(*args,out)
        self.assertFalse(receipt["original_source_certification_checked_by_this_api"])
        self.assertFalse(receipt["official_evidence_hashes_checked_by_this_api"])
        self.assertTrue(receipt["source_certification_and_evidence_hashes_are_host_responsibility"])
        self.assertFalse(receipt["strict_pit"])
        self.assertFalse(receipt["producer_modules_imported"])
        for key in ("momentum_computed","rankings_computed","orders_computed","returns_computed"):
            self.assertIs(out[key],False)


if __name__=="__main__":unittest.main(verbosity=2)
