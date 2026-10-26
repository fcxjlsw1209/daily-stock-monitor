"""Frozen, offline same-report SOURCE PREFLIGHT; no ratios, signals or admission.

--freeze-plan reads schemas/provenance and freezes code plus source hashes first.
--build requires that exact plan digest. --verify reproduces metadata only.
"""
import argparse
import bisect
import copy
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import uuid

ROOT = Path(__file__).resolve().parent.parent
SELF = "work/after142_same_report_preflight.py"
TEST = "work/after142_same_report_preflight_tests.py"
OUT = "outputs/monthly_same_report_preflight_v1"
AMD = "outputs/monthly_amd_comparative_source_v1/CANDIDATES.json"
AMD_AUDIT = "outputs/monthly_amd_comparative_source_v1/AUDIT.json"
AMD_MANIFEST = "outputs/monthly_amd_comparative_source_v1/MANIFEST.json"
AMD_REVIEW = "work/after142_amd_comparative_independent_review.json"
AAPL = "outputs/monthly_aapl_comparative_source_v1/COMPARATIVE_FACTS.json"
AAPL_REVIEW = "work/after142_aapl_comparative_independent_review.json"
IDENTITY = "work/after142_aapl_refetched_identity_review.json"
SESSIONS = "outputs/monthly_forward_guidance_preflight_v1/SESSIONS.json"
TOP = {
 AMD:"5ff80e9efddc9b5f1203f0299415065b6218349dc6c69587f845af3f71ffe38e",
 AMD_AUDIT:"6af7b52a46fa471b7b3b3b94eeb46c46665e7a4f704a534a96ef9cad6621cff4",
 AMD_MANIFEST:"7226e7bc91e7f2c26f90f74fa0214b8feee5c872271d059e7def055d596fd616",
 AMD_REVIEW:"e688932e950fa90aa6f23239bb80e7095bf9cc9f9f2dfbf63f72cfafdfd83c9f",
 AAPL:"0f8a3173b2876605f7f0d0088d405403ce5668c614eb35983dc2368caec70b2a",
 AAPL_REVIEW:"8711c2209b39e06717d211e7005f355c019939b50b225702cfe224ef623872ec",
 IDENTITY:"70fa81285e1ef7af6265000c6bc811f6d19ce761d997d67f04fc47808d43ddfa",
 SESSIONS:"c05c15f19bc1f7a3428e456a71199d64f05799ef3c432fa45f7bc485d0ff3cf4",
}
EXPECTED = {"AAPL":list(range(2016,2026)), "AMD":[2015,2016,2017,2018,2020,2021,2022,2023,2024,2025]}
ELIGIBLE = {"AAPL":list(range(2017,2026)), "AMD":[2016,2017,2020,2021,2022,2023,2024]}
ASSUMPTIONS = {
 "status":"Source preflight only. Conditional eligibility is not fact/model/strategy admission.",
 "vintage":"Assume currently retrieved official original-dated pages preserve original content. First-public-version and complete correction/reissue history are NOT certified.",
 "currency":"Same-report dollar labels are assumed to share the same unit. Preserve original decimal strings and scales; ISO USD is NOT certified; no conversion or normalization.",
 "AMD_identity":"Each event uses only its own original dated AMD (NASDAQ:AMD) and Advanced Micro Devices, Inc. anchor.",
 "AAPL_identity":"NEW assumption distinct from 141: prior 2014 NasdaqTrader equity mapping plus current Apple Inc. statement/dated parent release continuity. 2019 confirmation is used only after its own next-session availability. No debt CUSIP or debt-market evidence becomes equity identification.",
 "identity_window":"From this current release next pinned session through this current fiscal period-end +550 calendar days inclusive. No future event determines an earlier window; this is NOT a certified daily security master.",
 "period_boundaries":"Consecutive fiscal years assumed contiguous. Current start is the day after explicit same-report prior fiscal end. Prior starts use only earlier-public boundary evidence, including AMD FY2018 date-only evidence for the FY2020 comparison. Derived starts are NOT direct quotes.",
 "duration":"Both annual periods must have 330..400 inclusive days. AAPL2016 and AMD2015 prior starts remain unknown. No quarter summing, 365-day assumption or annualization.",
 "accounting":"As-reported consolidated monetary gross profit/margin and year-end total assets; not margin percentages, non-GAAP, organic or constant-business figures. FY2016 AMD presentation notes and Xilinx scope remain disclosed.",
 "invalid_events":"Retain all 20 current-release events and all four raw values. Select latest available event before eligibility; invalid/expired latest events cannot fall back to earlier eligible events. AMD2018/2025 exclusions remain.",
 "missing":"No AMD FY2019 publication is invented from FY2020 comparative columns. All four values are unavailable until the CURRENT publication next-session date.",
 "calendar":"Existing pinned XNYS research sessions with prior closure policy; not independent daily official certification.",
 "stop":"No network or CloudStorage. SEC 403 stop for 2026-10-19 America/Los_Angeles unchanged.",
}

def sha(raw):
 return hashlib.sha256(raw).hexdigest()

def encode(value):
 return (json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()

def fail_constant(value):
 raise ValueError("Non-finite JSON constant: "+value)

def object_pairs(pairs):
 out={}
 for key,val in pairs:
  if key in out: raise ValueError("Duplicate JSON key: "+key)
  out[key]=val
 return out

def decode(raw):
 return json.loads(raw,parse_constant=fail_constant,object_pairs_hook=object_pairs)

def check(ok, message):
 if not ok: raise ValueError(message)

def exact(obj, required):
 check(type(obj) is dict and set(obj)==set(required), "Unknown or missing fields")

def day(value):
 check(type(value) is str, "Date must be string")
 try: result=date.fromisoformat(value)
 except (TypeError,ValueError): raise ValueError("Invalid date")
 check(result.isoformat()==value,"Noncanonical date")
 return result

def local(path):
 check(type(path) is str and path and not Path(path).is_absolute() and ".." not in Path(path).parts,"Local relative path required")
 p=ROOT/path
 check(p.resolve()==p,"Symlink path prohibited")
 return p

def read(path, expected=None):
 raw=local(path).read_bytes()
 if expected is not None: check(sha(raw)==expected,"SHA mismatch: "+path)
 return raw

def frozen(path, expected):
 return decode(read(path,expected))

def shape(value):
 if type(value) is dict: return {k:shape(v) for k,v in sorted(value.items())}
 if type(value) is list: return [shape(v) for v in value]
 if value is None:return "null"
 if type(value) is bool:return "bool"
 if type(value) is str:return "string"
 if type(value) is int:return "integer"
 if type(value) is float:
  check(Decimal(str(value)).is_finite(),"Non-finite number")
  return "float"
 raise ValueError("Unsupported JSON type")

def schema_digest(value):
 return sha(encode(shape(value)))

def check_schema(value, expected_digest):
 check(schema_digest(value)==expected_digest,"Unknown field, missing field or changed schema")

def bind(mapping,path,digest):
 check(bool(re.fullmatch("[0-9a-f]{64}",digest)),"Invalid SHA")
 local(path)
 check(path not in mapping or mapping[path]==digest,"Conflicting source binding")
 mapping[path]=digest

def reference(mapping,ref):
 exact(ref,("path","sha256"))
 bind(mapping,ref["path"],ref["sha256"])

def next_session(release,sessions):
 day(release)
 check(type(sessions) is list and sessions and sessions==sorted(set(sessions)),"Invalid sessions")
 for d in sessions:day(d)
 check(sessions[0]<=release<sessions[-1],"Publication outside pinned calendar")
 i=bisect.bisect_right(sessions,release)
 return sessions[i]

def period(start,end,reason,evidence):
 day(end)
 if start is None:
  return {"start":None,"end":end,"days_inclusive":None,"derived":True,"start_is_direct_quote":False,"derivation":reason,"evidence":copy.deepcopy(evidence)}
 days=(day(end)-day(start)).days+1
 check(330<=days<=400,"Annual duration outside 330..400 days")
 return {"start":start,"end":end,"days_inclusive":days,"derived":True,"start_is_direct_quote":False,"derivation":reason,"evidence":copy.deepcopy(evidence)}

def boundary_period(current,prior,available,from_review=None):
 """Use earlier original date evidence only, never earlier monetary values."""
 check(prior["available_from"]<available,"Future or contemporaneous boundary evidence")
 check(prior["end"]==current["end"] or current.get("date_only_bridge") is True,"Mismatched comparison year")
 start=(day(prior["end"])+timedelta(days=1)).isoformat() if current.get("date_only_bridge") else prior["start"]
 out=period(start,current["end"],"Earlier public fiscal-end boundary; consecutive annual periods assumed contiguous", [prior["source"]])
 if from_review:
  for key in ("start","end","days_inclusive"):check(out[key]==from_review[key],"Independent period review mismatch")
 return out

def decimal_amount(value):
 check(type(value) is str and bool(re.fullmatch(r"-?\d+(?:\.\d+)?",value)),"Exact finite decimal string required")
 d=Decimal(value);check(d.is_finite(),"Non-finite amount")
 return d

def monetary_fact(raw,end,source,available,kind):
 check(raw["date"]==end,"Wrong year/date in source cell")
 check(type(raw["scale"]) is int and raw["scale"]==1000000,"Unexpected source scale")
 check(raw["currency_literal"]=="$","Unexpected currency")
 value=decimal_amount(raw["amount_decimal"])
 if kind=="total_assets":check(value>0,"Nonpositive assets")
 check(raw["measure_label"].lower() in (("total assets",) if kind=="total_assets" else ("gross margin","gross profit")),"Unexpected monetary label")
 display=raw["displayed_amount"].replace("$","").replace(",","").replace(" ","")
 if display.startswith("(") and display.endswith(")"):display="-"+display[1:-1]
 check(decimal_amount(display)==value,"Display/value mismatch")
 check(raw.get("currency_iso") is None and raw.get("ISO_currency_verified",False) is False,"ISO certification upgrade")
 if "source_raw_sha256" in raw:check(raw["source_raw_sha256"]==source["sha256"],"Mixed source")
 if "available_from" in raw:check(raw["available_from"]==available,"Backdated comparative")
 return {"amount_decimal":raw["amount_decimal"],"displayed_amount":raw["displayed_amount"],"scale":raw["scale"],"currency_literal":"$","currency_iso":None,"date":end,"available_from":available,"source":copy.deepcopy(source),"original_cell":copy.deepcopy(raw)}

EVENT_KEYS=("event_id","symbol","fiscal_year","release_date","available_from","source","provenance","current_period","comparison_period","financials","identity","rejection_codes","eligible_under_listed_research_assumptions","source_only","fact_admitted","model_admitted","strategy_admitted","first_public_version_verified","original_record","reviews")

def validate_event(e,sessions):
 exact(e,EVENT_KEYS)
 check(e["symbol"] in EXPECTED and type(e["fiscal_year"]) is int,"Invalid issuer/year")
 check(e["event_id"]==e["symbol"]+":FY"+str(e["fiscal_year"])+":"+e["release_date"],"Event ID mismatch")
 check(day(e["current_period"]["end"]).year==e["fiscal_year"],"Mismatched fiscal year")
 check(day(e["comparison_period"]["end"]).year==e["fiscal_year"]-1,"Mismatched prior year")
 check(day(e["release_date"])>day(e["current_period"]["end"]),"Publication must follow fiscal end")
 check(e["available_from"]==next_session(e["release_date"],sessions),"Wrong publication availability")
 check(e["identity"]["valid_from"]==e["available_from"],"Identity begins before event")
 check(e["identity"]["valid_through"]==(day(e["current_period"]["end"])+timedelta(days=550)).isoformat(),"Identity expiry mismatch")
 for name in ("fact_admitted","model_admitted","strategy_admitted","first_public_version_verified"):
  check(e[name] is False,"Admission/version upgrade prohibited")
 check(e["source_only"] is True,"Not source-only")
 check(type(e["rejection_codes"]) is list,"Rejection codes must be list")
 check(e["eligible_under_listed_research_assumptions"] is (not bool(e["rejection_codes"])),"Eligibility mismatch")
 exact(e["financials"],("current","comparison"))
 for group in ("current","comparison"):
  exact(e["financials"][group],("gross_profit","total_assets"))
  p=e[group+"_period"]
  for kind,f in e["financials"][group].items():
   check(f["source"]==e["source"] and f["available_from"]==e["available_from"],"Mixed source or availability")
   check(f["date"]==p["end"],"Mismatched cell period")
   check(f==monetary_fact(f["original_cell"],p["end"],e["source"],e["available_from"],kind),"Normalized amount/cell mismatch")
  if p["start"] is None:
   check(group=="comparison" and "unknown_comparison_duration" in e["rejection_codes"],"Missing duration not rejected")
  else:
   check(p["days_inclusive"]==(day(p["end"])-day(p["start"])).days+1 and 330<=p["days_inclusive"]<=400,"Period duration mismatch")
 check(e["current_period"]["start"]==(day(e["comparison_period"]["end"])+timedelta(days=1)).isoformat(),"Noncontiguous current year")
 if e["symbol"]=="AMD" and e["fiscal_year"] in (2018,2025):
  check("excluded_accounting_comparability" in e["rejection_codes"],"Excluded latest event lost")
 if e["available_from"]>e["identity"]["valid_through"]:check("publication_after_identity_expiry" in e["rejection_codes"],"Late publication not rejected")
 return e

def make_event(symbol,year,release,source,provenance,current,comparison,cells,identity,rejections,original,reviews,sessions):
 available=next_session(release,sessions)
 identity=copy.deepcopy(identity)
 identity.update(valid_from=available,valid_through=(day(current["end"])+timedelta(days=550)).isoformat(),certified_daily_security_master=False)
 reasons=list(rejections)
 if comparison["start"] is None:reasons.append("unknown_comparison_duration")
 if available>identity["valid_through"]:reasons.append("publication_after_identity_expiry")
 financials={}
 for group,p in (("current",current),("comparison",comparison)):
  financials[group]={k:monetary_fact(cells[group][k],p["end"],source,available,k) for k in ("gross_profit","total_assets")}
 e={"event_id":symbol+":FY"+str(year)+":"+release,"symbol":symbol,"fiscal_year":year,"release_date":release,"available_from":available,"source":copy.deepcopy(source),"provenance":copy.deepcopy(provenance),"current_period":current,"comparison_period":comparison,"financials":financials,"identity":identity,"rejection_codes":reasons,"eligible_under_listed_research_assumptions":not reasons,"source_only":True,"fact_admitted":False,"model_admitted":False,"strategy_admitted":False,"first_public_version_verified":False,"original_record":copy.deepcopy(original),"reviews":reviews}
 return validate_event(e,sessions)

def by_year(rows):
 check(type(rows) is list,"Rows required")
 result={}
 for r in rows:
  y=r["fiscal_year"];check(type(y) is int and y not in result,"Duplicate/invalid year")
  result[y]=r
 return result

def latest_event_metadata(events,symbol,asof):
 """Metadata-only sentinel test API. Does not rank securities or compute factors."""
 day(asof)
 known=[e for e in events if e["symbol"]==symbol and e["available_from"]<=asof]
 if not known:return None
 latest=max(known,key=lambda e:e["available_from"])
 check(sum(e["available_from"]==latest["available_from"] for e in known)==1,"Ambiguous latest event")
 reasons=list(latest["rejection_codes"])
 if asof>latest["identity"]["valid_through"]:reasons.append("identity_expired")
 return {"event_id":latest["event_id"],"eligible_under_listed_research_assumptions":not reasons,"rejection_codes":reasons}

def aapl_events(docs):
 rows=by_year(docs[AAPL]["rows"]);review=by_year(docs[AAPL_REVIEW]["rows"])
 identities=by_year(docs[IDENTITY]["financial_event_mappings"])
 anchors={s["source_id"]:s for s in docs[IDENTITY]["sources"]}
 check(sorted(rows)==EXPECTED["AAPL"] and set(rows)==set(review)==set(identities),"AAPL coverage mismatch")
 check(docs[AAPL_REVIEW]["numeric_source_passed"] is True and docs[IDENTITY]["conclusion"]["conditional_research_identity_mapping_sufficient"] is True,"AAPL review not passed")
 sessions=docs[SESSIONS]["sessions"];out=[]
 for year,r in sorted(rows.items()):
  v=review[year];ident=identities[year];p=r["provenance"]
  source={"path":r["source_path"],"sha256":r["source_sha256"]}
  check(r["symbol"]=="AAPL" and r["fact_admitted"] is False and r["strategy_admitted"] is False and r["historical_comparison_event_created"] is False,"AAPL scope changed")
  for other in ({"path":v["source_path"],"sha256":v["source_sha256"]},{"path":ident["financial_source_path"],"sha256":ident["financial_source_sha256"]},{"path":p["raw_path"],"sha256":p["raw_sha256"]}):check(other==source,"AAPL review/source mismatch")
  release=r["publication"]["release_date"];available=next_session(release,sessions)
  check(release==v["release_date"]==ident["release_date"]==r["availability"]["release_date"],"Publication mismatch")
  check(available==v["all_four_cells_first_usable_session"]==r["all_four_cells_available_session"]==r["availability"]["candidate_first_usable_session"]==ident["identity_valid_from_inclusive"],"AAPL availability mismatch")
  check(r["availability"]["first_public_version_verified"] is False,"First-version upgrade")
  cp=r["annual_period"];current=period(cp["start"],cp["end"],cp["derivation"],[source])
  check(current["end"]==v["current_end"]==ident["period_end"] and current["days_inclusive"]==v["current_duration_days"]==cp["days_inclusive"],"AAPL current period mismatch")
  pr=r["comparison_annual_period"];check(pr["end"]==v["comparison_end"],"AAPL prior end mismatch")
  if year==2016:
   check(pr["start"] is None and pr["days_inclusive"] is None and v["comparative_duration_days"] is None,"First AAPL prior duration invented")
   comparison=period(None,pr["end"],pr["reason"],[source])
  else:
   old=rows[year-1];oldsource={"path":old["source_path"],"sha256":old["source_sha256"]}
   check(pr["duration_source"]==oldsource["path"] and pr["duration_source_sha256"]==oldsource["sha256"],"AAPL boundary source mismatch")
   comparison=boundary_period(pr,{"start":old["annual_period"]["start"],"end":old["annual_period"]["end"],"available_from":old["all_four_cells_available_session"],"source":oldsource},available)
   check(pr["start"]==comparison["start"] and pr["days_inclusive"]==comparison["days_inclusive"]==v["comparative_duration_days"],"AAPL comparison duration mismatch")
  selected=[]
  for name in ident["available_identity_anchors"]:
   a=anchors[name];pub=a["publication_date_in_visible_heading"]
   check(day(pub)<day(release),"Future AAPL identity anchor")
   if name=="dtn2019-04":check(available>=next_session(pub,sessions),"Future 2019 confirmation")
   selected.append(copy.deepcopy(a))
  check(ident["primary_equity_mapping_anchor"]=="dtn2014-16" and any(a["source_id"]=="dtn2014-16" for a in selected),"Missing primary equity anchor")
  check(ident["conditionally_sufficient_for_mapping"] is True and ident["certified_daily_security_identity"] is False and ident["fact_or_strategy_admitted"] is False,"Identity review scope mismatch")
  check(ident["identity_valid_to_inclusive"]==(day(current["end"])+timedelta(days=550)).isoformat(),"AAPL reviewed expiry mismatch")
  identity={"kind":"prior_equity_anchor_plus_current_issuer_continuity","current_issuer_name":"Apple Inc.","anchors":selected,"original_event_mapping":copy.deepcopy(ident),"new_assumption_distinct_from_141":True}
  cells={"current":{"gross_profit":r["gross_amount"],"total_assets":r["total_assets"]},"comparison":{"gross_profit":r["comparison_gross_amount"],"total_assets":r["comparison_total_assets"]}}
  for actual,expected in ((r["gross_amount"],v["gross_current"]),(r["total_assets"],v["assets_current"]),(r["comparison_gross_amount"],v["gross_comparison"]),(r["comparison_total_assets"],v["assets_comparison"])):check(actual["amount_decimal"]==expected,"AAPL numeric review mismatch")
  out.append(make_event("AAPL",year,release,source,p,current,comparison,cells,identity,[],r,[AAPL_REVIEW,IDENTITY],sessions))
 return out

def amd_events(docs):
 rows=by_year(docs[AMD]["rows"]);review=by_year(docs[AMD_REVIEW]["events"]);audit=by_year(docs[AMD_AUDIT]["rows"])
 check(sorted(rows)==EXPECTED["AMD"] and set(rows)==set(review)==set(audit),"AMD coverage mismatch")
 check(docs[AMD_REVIEW]["review_completed"] is True and docs[AMD_REVIEW]["checks"]["all_40_amounts_equal_candidates"] is True,"AMD review missing")
 sessions=docs[SESSIONS]["sessions"];out=[]
 for year,r in sorted(rows.items()):
  v=review[year];a=audit[year];s=r["source"];source={"path":s["path"],"sha256":s["sha256"]}
  check(r["symbol"]=="AMD" and r["fact_admitted"] is False and r["strategy_admitted"] is False and r["historical_first_version_verified"] is False,"AMD scope changed")
  check(v["source"]==s and a["source_sha256"]==source["sha256"] and a["passed"] is True,"AMD source review mismatch")
  release=r["publication_anchor"]["release_date"];available=next_session(release,sessions)
  check(available==r["available_from"]==v["available_from"],"AMD availability mismatch")
  cp=r["current_period"];current=period(cp["start"],cp["end"],cp["derivation"],[source])
  check(current["days_inclusive"]==cp["days_inclusive"] and current["end"]==v["current_period"]["end"],"AMD current period mismatch")
  pr=r["comparison_period"];check(pr["start"] is None and pr["duration_label"]=="Year Ended","AMD raw prior schema changed")
  if year==2015:
   comparison=period(None,pr["end"],"Prior-prior year boundary unavailable; do not assume 365 days",[source])
  else:
   old=rows[2018 if year==2020 else year-1];oldsource={"path":old["source"]["path"],"sha256":old["source"]["sha256"]}
   target={"end":pr["end"],"date_only_bridge":year==2020}
   prior={"start":old["current_period"]["start"],"end":old["current_period"]["end"],"available_from":old["available_from"],"source":oldsource}
   reviewed=v["comparison_period_review"] if year not in (2018,2025) else None
   comparison=boundary_period(target,prior,available,reviewed)
   comparison["monetary_exclusion_of_boundary_source_unchanged"]=year==2020
  anchor=r["publication_anchor"]
  check(anchor["ticker_literal"]=="AMD (NASDAQ:AMD)" and anchor["issuer_name"]=="Advanced Micro Devices, Inc." and anchor["continuous_security_identity_verified"] is False,"AMD identity mismatch")
  identity={"kind":"own_original_dated_issuer_ticker_anchor","current_issuer_name":anchor["issuer_name"],"anchors":[dict(copy.deepcopy(anchor),source=source)],"new_assumption_distinct_from_141":False}
  excluded=year in (2018,2025)
  check(r["scope_review"]["eligible_for_candidate_review"] is (not excluded),"AMD exclusions changed")
  reasons=["excluded_accounting_comparability"] if excluded else []
  cells={"current":{"gross_profit":r["current"]["gross_amount"],"total_assets":r["current"]["total_assets"]},"comparison":{"gross_profit":r["comparison"]["gross_amount"],"total_assets":r["comparison"]["total_assets"]}}
  obs=v["independent_raw_amount_observations"]
  for actual,key in ((cells["current"]["gross_profit"],"current_gross"),(cells["current"]["total_assets"],"current_assets"),(cells["comparison"]["gross_profit"],"comparison_gross"),(cells["comparison"]["total_assets"],"comparison_assets")):check(actual["amount_decimal"]==obs[key],"AMD numeric review mismatch")
  out.append(make_event("AMD",year,release,source,s,current,comparison,cells,identity,reasons,r,[AMD_REVIEW,AMD_AUDIT],sessions))
 return out

def merged(docs):
 events=aapl_events(docs)+amd_events(docs)
 check(len(events)==20 and len({e["event_id"] for e in events})==20,"Event count/duplicate")
 for symbol in EXPECTED:
  got=[e["fiscal_year"] for e in events if e["symbol"]==symbol and e["eligible_under_listed_research_assumptions"]]
  check(got==ELIGIBLE[symbol],"Unexpected conditional coverage")
 return {"schema_version":1,"status":"SOURCE_PREFLIGHT_NOT_FACT_MODEL_OR_STRATEGY_ADMISSION","assumptions":ASSUMPTIONS,"events":events,"missing_publications":[{"symbol":"AMD","fiscal_year":2019,"event_created":False,"reason":"Original release missing; later comparative columns do not create earlier availability"}],"counts":{"events":20,"raw_monetary_values":80,"conditionally_eligible":16,"rejection_sentinels":4,"fact_admissions":0,"model_admissions":0,"strategy_admissions":0},"financial_computations":0,"network_requests":0}

def receipt_check(raw,meta,expected,content,expected_url=None):
 digest=meta.get("response_sha256",meta.get("sha256"))
 check(digest==expected==sha(raw),"Receipt/source SHA mismatch")
 check(type(meta.get("status",meta.get("http_status"))) is int and meta.get("status",meta.get("http_status"))==200,"Receipt HTTP failure")
 check(meta.get("public_unauthenticated") is True,"Receipt public origin absent")
 stamp=meta.get("retrieved_at",meta.get("retrieved_utc"))
 check(type(stamp) is str,"Retrieval timestamp absent")
 try:t=datetime.fromisoformat(stamp.replace("Z","+00:00"))
 except ValueError:raise ValueError("Malformed retrieval timestamp")
 check(t.tzinfo is not None,"Timezone missing")
 ctype=meta.get("content_type",meta.get("headers",{}).get("Content-Type",""))
 check(content in ctype,"Receipt content type mismatch")
 size=meta.get("response_bytes",meta.get("bytes",meta.get("size")))
 check(type(size) is int and size==len(raw),"Receipt size mismatch")
 url=meta.get("url",meta.get("source_url"))
 check(type(url) is str and url.startswith("https://") and meta.get("final_url")==url,"Receipt URL mismatch")
 if expected_url is not None:check(url==expected_url,"Receipt source URL mismatch")

def input_bundle():
 docs={p:frozen(p,s) for p,s in TOP.items()};bindings=dict(TOP)
 # Bind prior source audit dependencies without importing or executing them.
 for p,s in docs[AMD_MANIFEST]["input_sha256"].items():bind(bindings,p,s)
 for p,s in docs[AMD_REVIEW]["source_hashes"].items():bind(bindings,p,s)
 for r in docs[AAPL]["rows"]:
  p=r["provenance"];bind(bindings,p["raw_path"],p["raw_sha256"])
  for k in ("receipt","parent_release","parent_receipt"):reference(bindings,p[k])
 for a in docs[IDENTITY]["sources"]:
  bind(bindings,a["raw_path"],a["raw_sha256"]);bind(bindings,a["receipt_path"],a["receipt_sha256"])
 for suffix,key in (("PLAN.json","producer_plan_sha256"),("RECEIPT.json","producer_receipt_sha256")):
  bind(bindings,"outputs/monthly_aapl_comparative_source_v1/"+suffix,docs[AAPL_REVIEW][key])
 for p in (SELF,TEST,"run_local.sh","local_run.py","local.sandbox.sb",OUT+"/TESTS.json"):bind(bindings,p,sha(read(p)))
 for p,s in bindings.items():read(p,s)
 # Validate only the receipt types actually consumed by this metadata preflight.
 for r in docs[AAPL]["rows"]:
  p=r["provenance"]
  receipt_check(read(p["raw_path"]),decode(read(p["receipt"]["path"])),p["raw_sha256"],"application/pdf",p["source_url"])
  receipt_check(read(p["parent_release"]["path"]),decode(read(p["parent_receipt"]["path"])),p["parent_release"]["sha256"],"text/html")
 for r in docs[AMD]["rows"]:
  p=r["source"];receipt_check(read(p["path"]),decode(read(p["receipt"]["path"])),p["sha256"],"text/html",p["url"])
 for p in docs[IDENTITY]["sources"]:receipt_check(read(p["raw_path"]),decode(read(p["receipt_path"])),p["raw_sha256"],"text/html",p["url"])
 tests=decode(read(OUT+"/TESTS.json"))
 check(tests["passed"] is True and tests["synthetic_only"] is True and tests["failures"]==0 and tests["errors"]==0,"Tests must pass")
 check(tests["source_hashes"]=={p:bindings[p] for p in (SELF,TEST)},"Stale synthetic test record")
 return docs,bindings

def write_new(path,value):
 p=local(path);check(str(p).startswith(str(local(OUT))+"/"),"Output outside owned directory")
 p.parent.mkdir(parents=True,exist_ok=True)
 raw=encode(value);tmp=p.with_name("."+p.name+"."+uuid.uuid4().hex+".tmp")
 try:
  with tmp.open("xb") as f:f.write(raw);f.flush();os.fsync(f.fileno())
  os.link(str(tmp),str(p)) # Atomic exclusive publication; never replace old bytes.
 finally:
  if tmp.exists():tmp.unlink()
 return sha(raw)

def freeze_plan():
 check(not local(OUT+"/PLAN.json").exists(),"PLAN already frozen")
 docs,bindings=input_bundle()
 plan={"schema_version":1,"stage":"source_preflight_plan","created_utc":datetime.now(timezone.utc).isoformat(),"runtime":sys.version,"input_hashes":bindings,"document_schema_sha256":{p:schema_digest(v) for p,v in docs.items()},"assumptions":ASSUMPTIONS,"expected_years":EXPECTED,"expected_conditional_years":ELIGIBLE,"expected_events":20,"raw_values":80,"financial_computations_permitted":False,"admission_permitted":False,"outputs":["EVENTS.json","SOURCE_AUDIT.json","MANIFEST.json"],"schema_policy":"Every top-level consumed document is bound by exact bytes and recursive key/type/list structure. Unknown fields fail; no silent dropping. Original event records are preserved.","upstream_runtime_scope":"Original extraction/review artifacts are hashed; this preflight consumes their reviewed data and raw provenance. It does not reload upstream external PDF runtime libraries."}
 digest=write_new(OUT+"/PLAN.json",plan)
 print(json.dumps({"stage":"plan_frozen_before_transform","plan_sha256":digest,"bound_files":len(bindings)}))

def execute(plan_sha,verify=False):
 check(bool(re.fullmatch("[0-9a-f]{64}",plan_sha or "")),"Explicit plan SHA required")
 plan=frozen(OUT+"/PLAN.json",plan_sha);docs,bindings=input_bundle()
 check(plan["input_hashes"]==bindings and plan["runtime"]==sys.version and plan["assumptions"]==ASSUMPTIONS,"Frozen plan drift")
 for p,v in docs.items():check_schema(v,plan["document_schema_sha256"][p])
 data=merged(docs)
 audit={"schema_version":1,"stage":"source_preflight","plan_sha256":plan_sha,"passed":True,"meaning":"All source-preflight invariants passed; no financial fact/model/strategy admission.","counts":data["counts"],"conditional_years":ELIGIBLE,"rejections":[{"event_id":e["event_id"],"reasons":e["rejection_codes"]} for e in data["events"] if e["rejection_codes"]],"checks":{"all_twenty_original_records_preserved":True,"all_eighty_amount_strings_and_scales_preserved":True,"current_publication_availability_for_all_four_values":True,"causal_prior_period_boundary_sources_only":True,"causal_identity_anchors_only":True,"invalid_latest_retained":True,"all_frozen_source_bytes_rechecked":True},"input_hashes":bindings,"financial_computations":0,"network_requests":0}
 for p,s in bindings.items():read(p,s)
 payloads={"EVENTS.json":data,"SOURCE_AUDIT.json":audit}
 if verify:
  manifest=decode(read(OUT+"/MANIFEST.json"))
  check(manifest["plan_sha256"]==plan_sha and manifest["input_hashes"]==bindings,"Manifest binding mismatch")
  for name,obj in payloads.items():
   raw=read(OUT+"/"+name,manifest["output_hashes"][name]);check(decode(raw)==obj,"Output replay mismatch")
  print(json.dumps({"status":"EXACT_SOURCE_PREFLIGHT_REPRODUCED","events":20,"conditional":16,"rejected":4}));return
 check(not any(local(OUT+"/"+n).exists() for n in (*payloads,"MANIFEST.json")),"Refusing existing output overwrite")
 outputs={n:write_new(OUT+"/"+n,v) for n,v in payloads.items()}
 write_new(OUT+"/MANIFEST.json",{"schema_version":1,"stage":"source_preflight_complete_not_admission","created_utc":datetime.now(timezone.utc).isoformat(),"plan_sha256":plan_sha,"input_hashes":bindings,"output_hashes":outputs,"counts":data["counts"],"source_only":True,"financial_computations":0})
 print(json.dumps({"status":"SOURCE_PREFLIGHT_COMPLETE_NOT_ADMISSION","output_hashes":outputs,"events":20,"conditional":16,"rejected":4}))

if __name__=="__main__":
 parser=argparse.ArgumentParser(description=__doc__)
 mode=parser.add_mutually_exclusive_group(required=True)
 mode.add_argument("--freeze-plan",action="store_true");mode.add_argument("--build",action="store_true");mode.add_argument("--verify",action="store_true")
 parser.add_argument("--plan-sha256")
 args=parser.parse_args()
 if args.freeze_plan:freeze_plan()
 else:execute(args.plan_sha256,args.verify)

