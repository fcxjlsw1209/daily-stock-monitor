"""Synthetic-only tests for source preflight. No market/issuer data is loaded."""
import argparse
import copy
from datetime import date, timedelta
import tempfile
import unittest
from unittest import mock
import after142_same_report_preflight as m

def sessions():
 start=date(2015,1,1)
 return [(start+timedelta(days=i)).isoformat() for i in range(4800)]

def fixture():
 calendar=sessions()
 d={m.SESSIONS:{"sessions":calendar},m.AAPL:{"rows":[]},m.AAPL_REVIEW:{"rows":[],"numeric_source_passed":True},m.IDENTITY:{"financial_event_mappings":[],"sources":[{"source_id":"dtn2014-16","publication_date_in_visible_heading":"2014-06-04"},{"source_id":"dtn2019-04","publication_date_in_visible_heading":"2019-03-12"}],"conclusion":{"conditional_research_identity_mapping_sufficient":True}},m.AMD:{"rows":[]},m.AMD_REVIEW:{"events":[],"review_completed":True,"checks":{"all_40_amounts_equal_candidates":True}},m.AMD_AUDIT:{"rows":[]}}
 for symbol,years in m.EXPECTED.items():
  for year in years:
   end=f"{year}-12-31";prior=f"{year-1}-12-31";start=f"{year}-01-01";previous_start=f"{year-1}-01-01"
   release=f"{year+1}-01-15";available=f"{year+1}-01-16"
   source={"path":f"synthetic/{symbol}{year}.html","sha256":m.sha(f"{symbol}{year}".encode())}
   def fact(kind,when,amount):
    return {"date":when,"amount_decimal":amount,"displayed_amount":amount,"scale":1000000,"currency_literal":"$","measure_label":"Total assets" if kind=="assets" else "Gross profit"}
   cg=fact("gross",end,"100");ca=fact("assets",end,"300");pg=fact("gross",prior,"90");pa=fact("assets",prior,"250")
   days=(date(year,12,31)-date(year,1,1)).days+1
   prior_days=(date(year-1,12,31)-date(year-1,1,1)).days+1
   cp={"start":start,"end":end,"days_inclusive":days,"derivation":"synthetic same-source fiscal boundary"}
   if symbol=="AAPL":
    original={"symbol":symbol,"fiscal_year":year,"source_path":source["path"],"source_sha256":source["sha256"],"provenance":{"raw_path":source["path"],"raw_sha256":source["sha256"]},"fact_admitted":False,"strategy_admitted":False,"historical_comparison_event_created":False,"publication":{"release_date":release},"availability":{"release_date":release,"candidate_first_usable_session":available,"first_public_version_verified":False},"all_four_cells_available_session":available,"annual_period":cp,"comparison_annual_period":{"start":None if year==2016 else previous_start,"end":prior,"days_inclusive":None if year==2016 else prior_days,"reason":"synthetic earlier boundary"},"gross_amount":cg,"total_assets":ca,"comparison_gross_amount":pg,"comparison_total_assets":pa}
    if year!=2016:
     old=d[m.AAPL]["rows"][-1]
     original["comparison_annual_period"].update(duration_source=old["source_path"],duration_source_sha256=old["source_sha256"])
    numeric={"fiscal_year":year,"source_path":source["path"],"source_sha256":source["sha256"],"release_date":release,"all_four_cells_first_usable_session":available,"current_end":end,"comparison_end":prior,"current_duration_days":days,"comparative_duration_days":None if year==2016 else prior_days,"gross_current":"100","assets_current":"300","gross_comparison":"90","assets_comparison":"250"}
    identity={"fiscal_year":year,"financial_source_path":source["path"],"financial_source_sha256":source["sha256"],"release_date":release,"period_end":end,"identity_valid_from_inclusive":available,"identity_valid_to_inclusive":(date(year,12,31)+timedelta(days=550)).isoformat(),"available_identity_anchors":["dtn2014-16"]+(["dtn2019-04"] if year>=2019 else []),"primary_equity_mapping_anchor":"dtn2014-16","conditionally_sufficient_for_mapping":True,"certified_daily_security_identity":False,"fact_or_strategy_admitted":False}
    d[m.AAPL]["rows"].append(original);d[m.AAPL_REVIEW]["rows"].append(numeric);d[m.IDENTITY]["financial_event_mappings"].append(identity)
   else:
    fullsource=dict(source,url="https://ir.amd.com/synthetic")
    for f in (cg,ca,pg,pa):f.update(source_raw_sha256=source["sha256"],available_from=available,currency_iso=None)
    original={"symbol":symbol,"fiscal_year":year,"source":fullsource,"fact_admitted":False,"strategy_admitted":False,"historical_first_version_verified":False,"publication_anchor":{"release_date":release,"ticker_literal":"AMD (NASDAQ:AMD)","issuer_name":"Advanced Micro Devices, Inc.","continuous_security_identity_verified":False},"available_from":available,"current_period":cp,"comparison_period":{"start":None,"end":prior,"duration_label":"Year Ended"},"current":{"gross_amount":cg,"total_assets":ca},"comparison":{"gross_amount":pg,"total_assets":pa},"scope_review":{"eligible_for_candidate_review":year not in (2018,2025)}}
    review={"fiscal_year":year,"source":fullsource,"available_from":available,"current_period":cp,"comparison_period_review":{"start":previous_start,"end":prior,"days_inclusive":prior_days},"independent_raw_amount_observations":{"current_gross":"100","current_assets":"300","comparison_gross":"90","comparison_assets":"250"}}
    d[m.AMD]["rows"].append(original);d[m.AMD_REVIEW]["events"].append(review);d[m.AMD_AUDIT]["rows"].append({"fiscal_year":year,"source_sha256":source["sha256"],"passed":True})
 return d

class SourcePreflightTests(unittest.TestCase):
 def setUp(self):self.docs=fixture()
 def test_twenty_originals_sixteen_conditional_four_sentinels(self):
  out=m.merged(self.docs)
  self.assertEqual(out["counts"]["events"],20)
  self.assertEqual(out["counts"]["conditionally_eligible"],16)
  rejected=[(e["symbol"],e["fiscal_year"]) for e in out["events"] if e["rejection_codes"]]
  self.assertEqual(rejected,[("AAPL",2016),("AMD",2015),("AMD",2018),("AMD",2025)])
  self.assertFalse(any(e[k] for e in out["events"] for k in ("fact_admitted","model_admitted","strategy_admitted","first_public_version_verified")))
 def test_no_input_mutation_and_exact_original_record(self):
  before=copy.deepcopy(self.docs);out=m.merged(self.docs)
  self.assertEqual(self.docs,before)
  self.assertEqual([e["original_record"] for e in out["events"]],self.docs[m.AAPL]["rows"]+self.docs[m.AMD]["rows"])
 def test_all_four_keep_current_publication(self):
  for e in m.merged(self.docs)["events"]:
   for group in e["financials"].values():
    for f in group.values():self.assertEqual(f["available_from"],e["available_from"])
 def test_no_phantom_2019_amd(self):
  self.assertNotIn(2019,[e["fiscal_year"] for e in m.merged(self.docs)["events"] if e["symbol"]=="AMD"])
 def test_current_year_mismatch(self):
  self.docs[m.AAPL]["rows"][1]["fiscal_year"]=2028
  with self.assertRaises(ValueError):m.merged(self.docs)
 def test_comparative_cell_wrong_year(self):
  self.docs[m.AAPL]["rows"][1]["comparison_gross_amount"]["date"]="2015-12-31"
  with self.assertRaisesRegex(ValueError,"year/date"):m.merged(self.docs)
 def test_mixed_source_cell(self):
  self.docs[m.AMD]["rows"][1]["comparison"]["gross_amount"]["source_raw_sha256"]="0"*64
  with self.assertRaisesRegex(ValueError,"Mixed source"):m.merged(self.docs)
 def test_review_source_mismatch(self):
  self.docs[m.IDENTITY]["financial_event_mappings"][0]["financial_source_sha256"]="0"*64
  with self.assertRaisesRegex(ValueError,"source mismatch"):m.merged(self.docs)
 def test_review_value_mismatch(self):
  self.docs[m.AAPL_REVIEW]["rows"][1]["gross_current"]="999"
  with self.assertRaisesRegex(ValueError,"numeric review"):m.merged(self.docs)
 def test_duplicate_event_year(self):
  self.docs[m.AMD]["rows"].append(copy.deepcopy(self.docs[m.AMD]["rows"][0]))
  with self.assertRaisesRegex(ValueError,"Duplicate"):m.merged(self.docs)
 def test_unknown_top_or_nested_fields(self):
  digest=m.schema_digest(self.docs)
  for mutate in (lambda d:d.update(unknown=True),lambda d:d[m.AAPL]["rows"][0]["gross_amount"].update(future_override=True)):
   altered=copy.deepcopy(self.docs);mutate(altered)
   with self.assertRaisesRegex(ValueError,"Unknown field"):m.check_schema(altered,digest)
 def test_changed_scalar_type(self):
  altered=copy.deepcopy(self.docs);altered[m.AAPL]["rows"][0]["gross_amount"]["amount_decimal"]=100
  with self.assertRaises(ValueError):m.check_schema(altered,m.schema_digest(self.docs))
 def test_unknown_normalized_event_field(self):
  e=m.merged(self.docs)["events"][0];e["maybe_admitted"]=True
  with self.assertRaisesRegex(ValueError,"Unknown"):m.validate_event(e,self.docs[m.SESSIONS]["sessions"])
 def test_missing_duration_stays_raw_but_ineligible(self):
  e=m.merged(self.docs)["events"][0]
  self.assertIsNone(e["comparison_period"]["start"])
  self.assertEqual(e["financials"]["comparison"]["gross_profit"]["amount_decimal"],"90")
  self.assertIn("unknown_comparison_duration",e["rejection_codes"])
 def test_missing_duration_cannot_be_marked_eligible(self):
  e=m.merged(self.docs)["events"][0];e["eligible_under_listed_research_assumptions"]=True
  with self.assertRaisesRegex(ValueError,"Eligibility"):m.validate_event(e,self.docs[m.SESSIONS]["sessions"])
 def test_future_boundary_rejected(self):
  r=self.docs[m.AAPL]["rows"][0];r["all_four_cells_available_session"]="2020-01-01"
  with self.assertRaises(ValueError):m.merged(self.docs)
 def test_direct_future_boundary_rejected(self):
  with self.assertRaisesRegex(ValueError,"Future"):m.boundary_period({"end":"2020-12-31"},{"start":"2020-01-01","end":"2020-12-31","source":{},"available_from":"2022-01-01"},"2021-01-01")
 def test_amd_2020_uses_excluded_2018_date_only(self):
  out=m.merged(self.docs)
  e=next(e for e in out["events"] if (e["symbol"],e["fiscal_year"])==("AMD",2020))
  self.assertEqual(e["comparison_period"]["start"],"2019-01-01")
  self.assertTrue(e["comparison_period"]["monetary_exclusion_of_boundary_source_unchanged"])
  self.assertTrue(e["eligible_under_listed_research_assumptions"])
  old=next(e for e in out["events"] if (e["symbol"],e["fiscal_year"])==("AMD",2018))
  self.assertFalse(old["eligible_under_listed_research_assumptions"])
 def test_2019_identity_never_backfills(self):
  self.docs[m.IDENTITY]["financial_event_mappings"][0]["available_identity_anchors"].append("dtn2019-04")
  with self.assertRaisesRegex(ValueError,"Future"):m.merged(self.docs)
 def test_identity_expiry_is_own_end_plus_550(self):
  for e in m.merged(self.docs)["events"]:
   self.assertEqual(e["identity"]["valid_through"],(m.day(e["current_period"]["end"])+timedelta(days=550)).isoformat())
 def test_invalid_latest_blocks_previous(self):
  out=m.merged(self.docs)["events"]
  result=m.latest_event_metadata(out,"AMD","2026-02-01")
  self.assertIn("AMD:FY2025",result["event_id"])
  self.assertFalse(result["eligible_under_listed_research_assumptions"])
  self.assertIn("excluded_accounting_comparability",result["rejection_codes"])
 def test_expired_latest_not_fallback(self):
  result=m.latest_event_metadata(m.merged(self.docs)["events"],"AAPL","2028-01-01")
  self.assertIn("AAPL:FY2025",result["event_id"])
  self.assertIn("identity_expired",result["rejection_codes"])
 def test_no_available_event(self):
  self.assertIsNone(m.latest_event_metadata(m.merged(self.docs)["events"],"AAPL","2015-01-01"))
 def test_late_publication_rejection_sentinel(self):
  base=m.merged(self.docs)["events"][1]
  release="2020-01-15"
  e=m.make_event(base["symbol"],base["fiscal_year"],release,base["source"],base["provenance"],base["current_period"],base["comparison_period"],{"current":{"gross_profit":base["financials"]["current"]["gross_profit"]["original_cell"],"total_assets":base["financials"]["current"]["total_assets"]["original_cell"]},"comparison":{"gross_profit":base["financials"]["comparison"]["gross_profit"]["original_cell"],"total_assets":base["financials"]["comparison"]["total_assets"]["original_cell"]}},base["identity"],[],base["original_record"],base["reviews"],self.docs[m.SESSIONS]["sessions"])
  self.assertFalse(e["eligible_under_listed_research_assumptions"])
  self.assertIn("publication_after_identity_expiry",e["rejection_codes"])
 def test_nonfinite_bool_or_numeric_amount_rejected(self):
  for bad in ("NaN","Infinity","1e3",True,100):
   with self.subTest(bad=bad),self.assertRaises(ValueError):m.decimal_amount(bad)
 def test_zero_and_negative_gross_preserved(self):
  e=m.merged(self.docs)["events"][1]
  raw=copy.deepcopy(e["financials"]["current"]["gross_profit"]["original_cell"])
  for value in ("0","-12.50"):
   raw.update(amount_decimal=value,displayed_amount=value)
   f=m.monetary_fact(raw,raw["date"],e["source"],e["available_from"],"gross_profit")
   self.assertEqual(f["amount_decimal"],value)
 def test_nonpositive_assets_rejected(self):
  r=self.docs[m.AAPL]["rows"][1]["total_assets"];r.update(amount_decimal="0",displayed_amount="0")
  self.docs[m.AAPL_REVIEW]["rows"][1]["assets_current"]="0"
  with self.assertRaisesRegex(ValueError,"Nonpositive"):m.merged(self.docs)
 def test_units_and_iso_do_not_silently_upgrade(self):
  for change in ({"scale":True},{"scale":1000},{"currency_literal":"EUR"},{"ISO_currency_verified":True}):
   docs=copy.deepcopy(self.docs);docs[m.AAPL]["rows"][1]["gross_amount"].update(change)
   with self.subTest(change=change),self.assertRaises(ValueError):m.merged(docs)
 def test_different_display_amount_rejected(self):
  self.docs[m.AAPL]["rows"][1]["gross_amount"]["displayed_amount"]="101"
  with self.assertRaisesRegex(ValueError,"Display/value"):m.merged(self.docs)
 def test_json_duplicate_nonfinite_fail(self):
  for raw in (b'{"a":1,"a":2}',b'{"a":NaN}',b'{"a":Infinity}'):
   with self.assertRaises(ValueError):m.decode(raw)
 def test_calendar_outside_or_duplicate_fails(self):
  for cal in ([],["2020-01-01"]*2,["2020-02-01","2020-01-01"],["2022-01-01","2022-01-02"]):
   with self.assertRaises(ValueError):m.next_session("2020-01-01",cal)
 def test_strict_next_session(self):
  self.assertEqual(m.next_session("2020-01-01",["2020-01-01","2020-01-03"]),"2020-01-03")
 def test_binding_conflict_and_escape_fail(self):
  b={};m.bind(b,"work/example","a"*64)
  with self.assertRaises(ValueError):m.bind(b,"work/example","b"*64)
  for path in ("../escape","/absolute"):
   with self.assertRaises(ValueError):m.local(path)
 def test_receipt_variants_and_invalid_timestamp(self):
  raw=b"synthetic";s=m.sha(raw)
  meta={"response_sha256":s,"http_status":200,"public_unauthenticated":True,"retrieved_utc":"2026-01-01T00:00:00+00:00","headers":{"Content-Type":"text/html"},"response_bytes":len(raw),"url":"https://example.invalid/a","final_url":"https://example.invalid/a"}
  m.receipt_check(raw,meta,s,"text/html")
  for patch in ({"retrieved_utc":"bad"},{"retrieved_utc":"2026-01-01T00:00:00"},{"http_status":403},{"response_bytes":1},{"final_url":"https://else.invalid/a"}):
   with self.subTest(patch=patch),self.assertRaises(ValueError):m.receipt_check(raw,dict(meta,**patch),s,"text/html")
 def test_normalized_output_tampering_rejected(self):
  base=m.merged(self.docs)["events"][1]
  for field,value in (("amount_decimal","999"),("displayed_amount","999"),("scale",1000),("currency_literal","USD"),("currency_iso","USD")):
   e=copy.deepcopy(base);e["financials"]["current"]["gross_profit"][field]=value
   with self.subTest(field=field),self.assertRaisesRegex(ValueError,"Normalized"):m.validate_event(e,self.docs[m.SESSIONS]["sessions"])
 def test_receipt_expected_source_url_required(self):
  raw=b"synthetic";s=m.sha(raw)
  meta={"sha256":s,"status":200,"public_unauthenticated":True,"retrieved_at":"2026-01-01T00:00:00Z","content_type":"text/html","bytes":len(raw),"url":"https://example.invalid/a","final_url":"https://example.invalid/a"}
  with self.assertRaisesRegex(ValueError,"source URL"):m.receipt_check(raw,meta,s,"text/html","https://example.invalid/b")
 def test_existing_output_not_overwritten(self):
  with tempfile.TemporaryDirectory() as folder,mock.patch.object(m,"ROOT",m.Path(folder).resolve()):
   m.write_new(m.OUT+"/synthetic.json",{"a":1})
   with self.assertRaises(FileExistsError):m.write_new(m.OUT+"/synthetic.json",{"a":2})
   self.assertEqual(m.decode(m.read(m.OUT+"/synthetic.json")),{"a":1})
 def test_plan_drift_blocks_transform(self):
  plan={"input_hashes":{"wrong":"x"},"runtime":m.sys.version,"assumptions":m.ASSUMPTIONS}
  with mock.patch.object(m,"frozen",return_value=plan),mock.patch.object(m,"input_bundle",return_value=({},{})),mock.patch.object(m,"merged") as transform:
   with self.assertRaisesRegex(ValueError,"plan drift"):m.execute("a"*64)
   transform.assert_not_called()

if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("--record",action="store_true");args=parser.parse_args()
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SourcePreflightTests))
 if args.record and result.wasSuccessful():
  m.write_new(m.OUT+"/TESTS.json",{"schema_version":1,"passed":True,"synthetic_only":True,"tests_run":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),"source_hashes":{p:m.sha(m.read(p)) for p in (m.SELF,m.TEST)},"financial_computations":0,"actual_source_transforms":0})
 raise SystemExit(0 if result.wasSuccessful() else 1)

