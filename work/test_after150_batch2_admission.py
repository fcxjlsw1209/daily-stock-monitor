"""Pure/synthetic batch2 admission tests. No actual source values are read."""
from copy import deepcopy
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import after150_batch2_admission as m


def sessions():
    a, b = date(2015, 1, 1), date(2028, 1, 1)
    return [a+timedelta(days=i) for i in range((b-a).days)
            if (a+timedelta(days=i)).weekday() < 5]


def event(symbol="INTC", year=2021, missing=False):
    end = date(year, 12, 25) if symbol == "INTC" else date(year, 6, 30)
    prior = end-timedelta(days=364) if symbol == "INTC" else date(year-1, 6, 30)
    start = prior+timedelta(days=1)
    release = date(year+1, 1, 25).isoformat() if symbol == "INTC" else date(year, 7, 25).isoformat()
    available = m.first_session(release, sessions()) or (m.day(release)+timedelta(days=1)).isoformat()
    source = dict(path="source.html", sha256="a"*64, url="https://example.test/source", receipt="receipt.json")
    period = dict(period_start=str(start), period_end=str(end), days_inclusive=(end-start).days+1)
    fields = {}
    for name in m.FIELDS:
        fields[name] = dict(displayed_amount="(10)" if name == "annual_net_income" else "100",
            value_decimal="-10" if name == "annual_net_income" else "100", scale="1000000",
            currency_literal="$", currency_iso=None, source=deepcopy(source), available_from=available,
            locator={"row": 3}, as_of=str(prior if name == "prior_total_assets" else end),
            period=deepcopy(period) if name == "annual_net_income" else None, fact_admitted=False)
    if missing:
        fields = dict.fromkeys(m.FIELDS)
    context = dict(symbol=symbol, fiscal_year=year, event_id=symbol+"_"+release,
        publication_date=release, available_session=available, official_actual_period_end=str(end),
        evidence_path=source["path"], evidence_sha256=source["sha256"], source_url=source["url"],
        context={"receipt": {"path": "receipt.json", "sha256": "b"*64}},
        context_summary={"comparison_assets_date": str(prior), "current_annual_period":{
            "start": None if symbol == "INTC" else str(start), "end": str(end),
            "candidate_start_from_prior_end_plus_one": str(start)}})
    periods = dict(current=period, comparison=dict(period_start=None, period_end=str(prior), days_inclusive=None))
    if symbol == "INTC":
        periods.update(start_basis="derived_same_report_prior_annual_end_plus_one_not_directly_quoted",
                       start_assumption_certified=False)
    holds = [m.INTC_HOLD] if symbol == "INTC" else []
    if symbol == "KLAC" and year == 2019:
        holds.append("ASC606_transition_context")
    if missing:
        holds.append("PRE_geometry")
    item = dict(symbol=symbol, event_id=context["event_id"], fiscal_year=year,
        publication_date=release, available_from=available, source=source, fields=fields,
        periods=periods, context_holds=holds, accounting_review_flags=["keep this note"],
        fact_admitted=False, model_admitted=False, strategy_admitted=False, untouched={"keep": [1, 2]})
    return item, (symbol, "0000050863" if symbol == "INTC" else "0000319201", context)


def rowreview(e):
    row = dict(event_id=e["event_id"], symbol=e["symbol"], fiscal_year=e["fiscal_year"], passed=True,
        metadata_verified=True, identity_verified=True,
        periods_verified=any(e["fields"].values()), currency_verified=any(e["fields"].values()),
        clock_verified=m.first_session(e["publication_date"],sessions()) == e["available_from"],
        retained_context_holds=deepcopy(e["context_holds"]), source=deepcopy(e["source"]),
        available_from=e["available_from"], release_date=e["publication_date"], fields={})
    for name, cell in e["fields"].items():
        row["fields"][name] = (dict(is_null=True, null_reason_verified=True,
            null_classification="original_report_present_unparsed_PRE_column_geometry") if cell is None else
            dict(is_null=False, verified=True, source_value_present=True, value_and_sign_verified=True,
                 units_verified=True, period_verified=True, source_locator_verified=True))
    if e["symbol"] == "INTC":
        row.update(boundary_assumption_verified=True, assumption_id=m.ASSUMPTION, start_is_source_quoted=False,
            derived_start=e["periods"]["current"]["period_start"],
            boundary_evidence={f: dict(table=1, date=e["fields"][f]["as_of"], headers=[{"text":"synthetic date"}])
                               for f in m.FIELDS})
    return row


def assessed(symbol="INTC", year=2021):
    e, c = event(symbol, year, missing=symbol == "KLAC" and year == 2011)
    return m.assess_event(e, c, rowreview(e), sessions())


def catalogrow(e):
    return dict(event_id=e["event_id"], symbol=e["symbol"], fiscal_year=e["fiscal_year"], fiscal_quarter=4,
        publication_date=e["publication_date"], available_session=e["available_from"],
        official_actual_period_end=e["periods"]["current"]["period_end"])


def write(root, path, value):
    p = Path(root)/path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, sort_keys=True)+"\n")
    return m.digest(p)


def fixture(root):
    root = Path(root)
    source_sha = write(root, "source.html", {"synthetic": "no real financial source"})
    receipt_sha = write(root, "receipt.json", {})
    docs, selected, records, catalog = [], [], [], []
    for symbol, years in (("INTC", range(2010, 2026)), ("KLAC", range(2011, 2027))):
        for year in years:
            e, (_, _, ctx) = event(symbol, year, missing=symbol == "KLAC" and year == 2011)
            e["source"]["sha256"] = source_sha
            for cell in e["fields"].values():
                if cell:
                    cell["source"]["sha256"] = source_sha
            ctx["evidence_sha256"] = source_sha
            ctx["context"]["receipt"]["sha256"] = receipt_sha
            row = catalogrow(e)
            catalog.append(row)
            chosen = symbol == "KLAC" or year in (2017, 2020, 2021)
            ctx.update(row, html_context_reviewed=chosen)
            docs.append(ctx)
            if chosen:
                selected.append(e)
                records.append(dict(event_id=e["event_id"], source_path="source.html",
                                    source_sha256=source_sha, receipt_path="receipt.json"))
    context = dict(documents=docs, identity_leads={s:dict(cik=c) for s,c in
        (("INTC","0000050863"),("KLAC","0000319201"))})
    context_sha = write(root, m.CONTEXT, context)
    cref = dict(path=m.CONTEXT, sha256=context_sha)
    policy = dict(schema_version=1, policy_id="intc_klac_annual_ni_same_report_assets_batch2_v1",
        stage="prospective_source_policy_not_strategy_registration", definition=m.DEFINITION,
        required_fields=list(m.FIELDS), source_scope=dict(symbols=["INTC","KLAC"],
            catalog_events=32, html_context_reviewed_events=19, no_automatic_admission=True, source_context=cref),
        periods={"INTC_52_53_week_assumption":dict(enabled=True,not_a_quoted_fact=True,assumption_id=m.ASSUMPTION)},
        known_time={"maximum_age_days":550}, accounting={"KLAC_FY2011":"Hold PRE", "KLAC_FY2019":"Hold ASC606"},
        preservation={k:False for k in ("first_public_version_verified","complete_correction_history_verified",
                                      "certified_daily_security_master")},
        input_hashes={m.CONTEXT:context_sha})
    policy["fixed_catalog_metadata"] = [dict(event_id=x["event_id"],symbol=x["symbol"],fiscal_year=x["fiscal_year"],
        publication_date=x["publication_date"],available_session=x["available_session"],
        official_actual_period_end=x["official_actual_period_end"],source_path=x["evidence_path"],
        source_sha256=x["evidence_sha256"],html_context_reviewed=x["html_context_reviewed"],
        retained_context_gap=x.get("gap")) for x in docs]
    policy_sha = write(root, m.POLICY, policy)
    names = ["INTC","KLAC"]+["STOCK%02d"%i for i in range(71)]
    etfs = ["ETF%02d"%i for i in range(17)]
    dates = ["%04d-%02d-01"%(2016+(7+i)//12,(7+i)%12+1) for i in range(122)]
    panels = [[names[(i+j)%73] for j in range(32)]+etfs for i in range(122)]
    slots = [dict(session=d,symbol=s) for d,p in zip(dates,panels) for s in p if s not in etfs]
    write(root,m.CENSUS,dict(stocks=[dict(symbol=s) for s in names],annual_documents=catalog,
                            monthly_candidate_coverage=list(reversed(slots))))
    write(root,m.META,dict(dates=dates,symbols=panels,etfs=etfs))
    write(root,m.SESSIONS,dict(sessions=[str(s) for s in sessions()]))
    original = dict(events=[dict(event_id=f"old{i}",symbol=["AAPL","AMD","MSFT","NVDA"][i%4],
                                stock_compensation_admission={"admitted":i<31}) for i in range(44)])
    old_sha = write(root,m.OLD_DATA,original)
    old_adm_sha = write(root,m.OLD_ADMISSION,dict(passed=True,data_sha256=old_sha,events=44,admitted_events=31,held_events=13))
    old_ref = dict(path=m.OLD_DATA,sha256=old_sha,admission_path=m.OLD_ADMISSION,
        admission_sha256=old_adm_sha,**m.old.old_population(original),preserved_without_modification=True)
    prior_events = [dict(event_id=f"batch1_{i}",symbol="AMZN" if i<10 else "CSCO",
                        ni_assets_admission={"admitted":i<17}) for i in range(26)]
    prior_data = dict(method=m.METHOD,events=prior_events,latest_event_catalog=[{"id":i} for i in range(32)],
                      original_four_issuer_reference=old_ref)
    prior_inputs = {m.OLD_DATA:old_sha,m.OLD_ADMISSION:old_adm_sha}
    outputs = {"DATASET.json":write(root,m.BATCH1+"/DATASET.json",prior_data),
        "ADMISSION.json":write(root,m.BATCH1+"/ADMISSION.json",dict(passed=True,
            counts=dict(events=26,admitted=17,held=9),input_hashes=prior_inputs)),
        "COVERAGE.json":write(root,m.BATCH1+"/COVERAGE.json",{"unchanged":True})}
    write(root,m.BATCH1+"/MANIFEST.json",dict(input_hashes=prior_inputs,outputs=outputs))
    for path in ("code.py","work/after150_batch2_admission.py","work/test_after150_batch2_admission.py",
                 "work/after149_batch1_admission.py",
                 "work/after150_batch2_independent_review.py","work/test_after150_batch2_independent_review.py"):
        write(root,path,{"synthetic":"stub"})
    inputs = {p:m.digest(root/p) for p in
        (m.POLICY,m.CONTEXT,m.CENSUS,m.META,m.SESSIONS,"source.html","receipt.json",
         m.BATCH1+"/DATASET.json",m.BATCH1+"/ADMISSION.json")}
    plan = dict(inputs=inputs,code={"code.py":m.digest(root/"code.py")},catalog_ref=cref,
        context=m.CONTEXT,policy=m.POLICY,records=records,catalog_events=32,catalog_ids=[d["event_id"] for d in docs])
    plan_sha = write(root,m.BATCH+"/PLAN.json",plan)
    extraction = dict(plan_sha256=plan_sha,catalog_ref=cref,catalog_ids=plan["catalog_ids"],
        status="source_values_extracted_pending_independent_review",events=selected,
        new_financial_admissions=0,new_strategy_trials=0,ratios_computed=0,returns_computed=0)
    extraction_sha = write(root,m.BATCH+"/EXTRACTIONS.json",extraction)
    review = dict(passed=True,plan_sha256=plan_sha,extractions_sha256=extraction_sha,
        event_reviews=[rowreview(e) for e in selected],input_hashes=dict(inputs,**{
            m.BATCH+"/PLAN.json":plan_sha,m.BATCH+"/EXTRACTIONS.json":extraction_sha}))
    review.update(stage="independent_source_value_and_null_classification_review",
        financial_calculations=0,factor_calculations=0,returns_computed=0,new_admissions=0,
        source_hashes={p:m.digest(root/p) for p in
            ("work/after150_batch2_independent_review.py","work/test_after150_batch2_independent_review.py")})
    review["input_hashes"].update(review["source_hashes"])
    review_sha = write(root,m.REVIEW,review)
    return plan_sha,extraction_sha,review_sha,policy_sha


class SourceTests(unittest.TestCase):
    def test_signed_zero_or_negative_ni_allowed(self):
        for value in ("0","-10","1.25"):
            e,c=event();e["fields"]["annual_net_income"]["value_decimal"]=value
            self.assertTrue(m.assess_event(e,c,rowreview(e),sessions())["ni_assets_admission"]["admitted"])

    def test_intc_boundary_extra_flag_required(self):
        e,c=event();review=rowreview(e);review["boundary_assumption_verified"]=False
        self.assertFalse(m.assess_event(e,c,review,sessions())["ni_assets_admission"]["admitted"])

    def test_intc_original_hold_remains_despite_conditional_overlay(self):
        out=assessed()
        self.assertTrue(out["ni_assets_admission"]["admitted"])
        self.assertIn(m.INTC_HOLD,out["context_holds"])
        self.assertFalse(out["fact_admitted"])

    def test_intc_365_day_not_52_week(self):
        e,c=event()
        e["fields"]["annual_net_income"]["period"]["days_inclusive"]=365
        self.assertFalse(m.assess_event(e,c,rowreview(e),sessions())["ni_assets_admission"]["admitted"])

    def test_nonquoted_derivation_required(self):
        e,c=event();e["periods"]["start_assumption_certified"]=True
        self.assertFalse(m.assess_event(e,c,rowreview(e),sessions())["ni_assets_admission"]["admitted"])

    def test_klac2011_all_null_retained(self):
        out=assessed("KLAC",2011)
        self.assertTrue(all(x is None for x in out["fields"].values()))
        self.assertIn("KLAC_FY2011_PRE_geometry_unparsed",out["ni_assets_admission"]["reasons"])

    def test_klac2011_filling_null_fails_hard(self):
        e,c=event("KLAC",2011)
        with self.assertRaisesRegex(ValueError,"null"):
            m.assess_event(e,c,rowreview(e),sessions())

    def test_klac2019_keeps_transition_hold(self):
        out=assessed("KLAC",2019)
        self.assertIn("KLAC_FY2019_ASC606_transition_unresolved",out["ni_assets_admission"]["reasons"])

    def test_other_klac_year_not_blanket_rejected(self):
        self.assertTrue(assessed("KLAC",2021)["ni_assets_admission"]["admitted"])

    def test_extra_unresolved_hold_is_not_discarded(self):
        e,c=event();e["context_holds"].append("new_problem")
        self.assertIn("retained_context_hold:new_problem",
            m.assess_event(e,c,rowreview(e),sessions())["ni_assets_admission"]["reasons"])

    def test_invalid_assets_held_without_correction(self):
        for value in ("0","-1","NaN","Infinity"):
            e,c=event();e["fields"]["current_total_assets"]["value_decimal"]=value
            out=m.assess_event(e,c,rowreview(e),sessions())
            self.assertFalse(out["ni_assets_admission"]["admitted"])
            self.assertEqual(out["fields"]["current_total_assets"]["value_decimal"],value)

    def test_per_field_scales_retained_no_rescaling(self):
        e,c=event();e["fields"]["annual_net_income"]["scale"]="1000"
        self.assertTrue(m.assess_event(e,c,rowreview(e),sessions())["ni_assets_admission"]["admitted"])

    def test_nonpositive_scale_held(self):
        e,c=event();e["fields"]["annual_net_income"]["scale"]="0"
        self.assertFalse(m.assess_event(e,c,rowreview(e),sessions())["ni_assets_admission"]["admitted"])

    def test_iso_upgrade_not_silently_admitted(self):
        e,c=event();e["fields"]["annual_net_income"]["currency_iso"]="USD"
        self.assertFalse(m.assess_event(e,c,rowreview(e),sessions())["ni_assets_admission"]["admitted"])

    def test_source_and_availability_mismatch_held(self):
        for key,value in (("source",{}),("available_from","1900-01-01"),("as_of","2021-12-31")):
            e,c=event();e["fields"]["annual_net_income"][key]=value
            self.assertFalse(m.assess_event(e,c,rowreview(e),sessions())["ni_assets_admission"]["admitted"])

    def test_early_calendar_remains_held(self):
        self.assertIn("unverified_exchange_calendar",assessed("KLAC",2012)["ni_assets_admission"]["reasons"])

    def test_wrong_next_session_held(self):
        e,c=event();e["available_from"]="2022-01-28";c[2]["available_session"]=e["available_from"]
        for f in e["fields"].values():f["available_from"]=e["available_from"]
        self.assertFalse(m.assess_event(e,c,rowreview(e),sessions())["ni_assets_admission"]["admitted"])

    def test_source_identity_mismatch_fails(self):
        e,c=event();e["source"]["sha256"]="c"*64
        with self.assertRaisesRegex(ValueError,"Source/context"):
            m.assess_event(e,c,rowreview(e),sessions())

    def test_pure_no_mutation_and_all_metadata_retained(self):
        e,c=event();before=deepcopy(e)
        out=m.assess_event(e,c,rowreview(e),sessions())
        self.assertEqual(e,before)
        self.assertEqual({k:out[k] for k in e},before)

    def test_identity_window_is_not_security_master(self):
        out=assessed()
        self.assertEqual(out["identity"]["identity_valid_through"],str(date(2021,12,25)+timedelta(days=550)))
        self.assertFalse(out["identity"]["certified_daily_security_master"])

    def test_latest_invalid_blocks_older(self):
        old=assessed("KLAC",2018);new=assessed("KLAC",2019)
        rows=[]
        for e in (old,new):
            row=catalogrow(e);row["clock_review"]={"verified":True};rows.append(row)
        out=m.select_latest(rows,{e["event_id"]:e for e in (old,new)},"KLAC","2019-08-01")
        self.assertEqual(out["reason"],"latest_annual_event_held")

    def test_latest_unprocessed_blocks_old(self):
        e=assessed();row=catalogrow(e);row["clock_review"]={"verified":True}
        newer=deepcopy(row);newer.update(event_id="new",publication_date="2022-02-01",available_session="2022-02-02")
        self.assertEqual(m.select_latest([row,newer],{e["event_id"]:e},"INTC","2022-03-01")["reason"],"latest_annual_event_unprocessed")

    def test_unknown_latest_clock_blocks_old(self):
        e=assessed();row=catalogrow(e);row["clock_review"]={"verified":True}
        newer=deepcopy(row);newer.update(event_id="new",publication_date="2022-02-01",clock_review={"verified":False})
        self.assertEqual(m.select_latest([row,newer],{e["event_id"]:e},"INTC","2022-03-01")["reason"],"latest_annual_clock_unverified")

    def test_550_boundary_inclusive_then_stale(self):
        e=assessed();row=catalogrow(e);row["clock_review"]={"verified":True}
        for age, expected in ((550,True),(551,False)):
            out=m.select_latest([row],{e["event_id"]:e},"INTC",str(date(2021,12,25)+timedelta(days=age)))
            self.assertIs(out["available"],expected)

    def test_review_requires_each_nonnull_field_flag(self):
        e,_=event();r=rowreview(e)
        for flag in ("verified","source_value_present","value_and_sign_verified","units_verified","period_verified","source_locator_verified"):
            review=dict(passed=True,plan_sha256="p",extractions_sha256="e",event_reviews=[deepcopy(r)])
            review["event_reviews"][0]["fields"]["annual_net_income"][flag]=False
            with self.assertRaises(ValueError):
                m.verify_independent_review(review,[e],"p","e")

    def test_independent_clock_disagreement_is_rejected(self):
        e,c=event();review=rowreview(e);review["clock_verified"]=False
        with self.assertRaisesRegex(ValueError,"Independent clock"):
            m.assess_event(e,c,review,sessions())

    def test_review_missing_boundary_locator_rejected(self):

        e,_=event();r=rowreview(e);r["boundary_evidence"]["prior_total_assets"]["headers"]=[]
        with self.assertRaises(ValueError):
            m.verify_independent_review(dict(passed=True,plan_sha256="p",extractions_sha256="e",event_reviews=[r]),[e],"p","e")

    def test_review_null_classification_required(self):
        e,_=event("KLAC",2011,True);r=rowreview(e);r["fields"]["annual_net_income"].pop("null_classification")
        with self.assertRaises(ValueError):
            m.verify_independent_review(dict(passed=True,plan_sha256="p",extractions_sha256="e",event_reviews=[r]),[e],"p","e")

    def test_review_event_order_population_no_drops(self):
        e,_=event();r=rowreview(e)
        with self.assertRaises(ValueError):
            m.verify_independent_review(dict(passed=True,plan_sha256="p",extractions_sha256="e",event_reviews=[r,r]),[e],"p","e")


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.hashes=fixture(self.root)

    def build(self):
        return m.build(self.root,*self.hashes)

    def test_complete_synthetic_package_preserves_all_populations(self):
        objects,_=self.build();data=objects["DATASET.json"]
        self.assertEqual(len(data["events"]),19)
        self.assertEqual(len(data["latest_event_catalog"]),32)
        self.assertEqual(data["prior_batch1_reference"]["events"],26)
        self.assertEqual(data["original_four_issuer_reference"]["events"],44)
        self.assertEqual(objects["COVERAGE.json"]["counts"]["stock_slots"],3904)
        self.assertEqual(objects["ADMISSION.json"]["actual_factors_calculated"],0)

    def test_all_raw_events_preserved_except_four_overlay_keys(self):
        objects,_=self.build()
        originals=json.loads((self.root/m.BATCH/"EXTRACTIONS.json").read_text())["events"]
        for raw,out in zip(originals,objects["DATASET.json"]["events"]):
            self.assertEqual(raw,{k:v for k,v in out.items() if k not in ("identity","ni_assets_admission","source_review","clock_review")})

    def test_changed_original_hash_rejected(self):
        (self.root/"source.html").write_text("tampered")
        with self.assertRaises(ValueError):self.build()

    def test_wrong_explicit_sha_rejected(self):
        args=list(self.hashes);args[1]="0"*64
        with self.assertRaises(ValueError):m.build(self.root,*args)

    def test_batch1_changed_bytes_rejected(self):
        (self.root/m.BATCH1/"DATASET.json").write_text("{}")
        with self.assertRaises(ValueError):self.build()

    def test_original44_changed_bytes_rejected(self):
        (self.root/m.OLD_DATA).write_text("{}")
        with self.assertRaises(ValueError):self.build()

    def test_prior_dataset_requires_explicit_new_plan_pin(self):
        plan=json.loads((self.root/m.BATCH/"PLAN.json").read_text())
        del plan["inputs"][m.BATCH1+"/DATASET.json"]
        new_sha=write(self.root,m.BATCH+"/PLAN.json",plan)
        with self.assertRaisesRegex(ValueError,"Mandatory binding"):
            m.build(self.root,new_sha,*self.hashes[1:])

    def test_reviewer_source_closure_is_required(self):
        review=json.loads((self.root/m.REVIEW).read_text())
        review["source_hashes"]={}
        sha=write(self.root,m.REVIEW,review)
        with self.assertRaisesRegex(ValueError,"Nonempty"):
            m.build(self.root,self.hashes[0],self.hashes[1],sha,self.hashes[3])

    def test_source_review_cannot_report_prior_admission(self):
        review=json.loads((self.root/m.REVIEW).read_text())
        review["new_admissions"]=True
        sha=write(self.root,m.REVIEW,review)
        with self.assertRaisesRegex(ValueError,"source-only"):
            m.build(self.root,self.hashes[0],self.hashes[1],sha,self.hashes[3])

    def test_publish_is_write_once_and_manifest_hashes_outputs(self):

        objects,evidence=self.build()
        manifest=m.publish(self.root,objects,evidence)
        for name,h in manifest["outputs"].items():self.assertEqual(m.digest(self.root/m.OUT/name),h)
        with self.assertRaisesRegex(ValueError,"Write-once"):m.publish(self.root,objects,evidence)

    def test_change_after_build_blocks_publish(self):
        objects,evidence=self.build()
        (self.root/"source.html").write_text("changed")
        with self.assertRaises(ValueError):m.publish(self.root,objects,evidence)
        self.assertFalse((self.root/m.OUT/"MANIFEST.json").exists())

    def test_missing_or_duplicate_stock_slot_fails(self):
        data=json.loads((self.root/m.CENSUS).read_text())
        meta=json.loads((self.root/m.META).read_text())
        data["monthly_candidate_coverage"][1]=data["monthly_candidate_coverage"][0]
        with self.assertRaises(ValueError):m.coverage(meta,data,[],[])

    def test_absolute_local_path_accepted_escape_rejected(self):
        e=m.Evidence(self.root)
        e.pin(str(self.root/"source.html"))
        with self.assertRaises(ValueError):e.path(self.root/".."/"escape")


if __name__=="__main__":
    unittest.main(verbosity=2)
