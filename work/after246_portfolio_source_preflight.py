"""Prospective247 source-clock/identity metadata only; run via run_local.sh."""
from pathlib import Path
from datetime import date,datetime,timezone
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'work/after246_portfolio_source_preflight.json'
DRAFT='work/monthly_portfolio_path247_DRAFT_01.json'
DRAFT_SHA='c6b43b3af5c2df6d3ae514273dcaa8b18c88618bb9825d9e5b18f3a8e40766ae'
PINS={}
def require(ok,message):
    if not ok:raise ValueError(message)
def path(name):
    p=Path(name);p=p if p.is_absolute() else ROOT/p
    require(p.is_relative_to(ROOT) and p.resolve().is_relative_to(ROOT) and 'CloudStorage' not in str(p),'Local source only')
    return p
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def bind(name,expected=None):
    p=path(name);h=sha(p);require(expected is None or h==expected,'Changed immutable source: '+str(p))
    key=str(p.relative_to(ROOT));require(key not in PINS or PINS[key]==h,'Conflicting source pin')
    PINS[key]=h;return h
def read(name,expected=None):
    bind(name,expected);return json.loads(path(name).read_text())
def iso(s):
    require(type(s) is str and date.fromisoformat(s).isoformat()==s,'Canonical ISO date');return s
require(not OUT.exists(),'New preflight write once')
draft=read(DRAFT,DRAFT_SHA)
require(draft['schema']=='monthly_portfolio_path247_prospective_draft_v1' and draft['status']=='PROSPECTIVE_DEFINITION_DRAFT_NOT_REGISTERED_NOT_EXECUTED','Prospective draft scope')
require(all(draft[k] is False for k in ('actual_new_features','actual_new_orders','actual_new_returns','central_or_observer_changed')) and draft['new_trials_registered']==0,'No draft actual work')
for name,h in draft['evidence_sha256'].items():bind(name,h)
prior_name='work/endpoint_median245_SOURCE_REVIEW.json'
prior=read(prior_name,'972c0975c91e6e95141941c54aa13ad82882be048975589876bee09b85880ab9')
require(prior['passed'] is True and prior['metadata_only'] is True and prior['strict_pit'] is False and prior['prices_loaded'] is False and prior['npz_decoded'] is False,'Prior metadata-only scope')
bind(prior['implementation']['path'],prior['implementation']['sha256'])
local='outputs/monthly_local_linear70/'
plan=read(local+'PLAN.json','0016175aee1df43b734010d80bd543b881c57036a7f8c6ab73428340203afc96')
original=read(local+'INPUT_HASHES.json','5ac2b639fa8c944688787eba47ea1b420bfbe9c6e1e051a3d827d8b2b32b264d')
raw=plan['raw_source_hashes'];require(len(raw)==64 and raw==prior['raw_source_hashes'],'Exact original64 immutable map')
for name,h in raw.items():require(original.get(name)==h,'Original registry');bind(name,h)
meta=read(local+'META.json','3b46968f486e88fa170ac01f120dd5ef74ebd5a56243c6991c5a4fea506076f4')
require(len(meta['dates'])==len(meta['ends'])==len(meta['symbols'])==122,'Original122 axis')
require(meta['dates']==sorted(set(meta['dates'])) and meta['dates'][0]=='2016-08-17','Query chronology')
require(meta['ends'][:-1]==meta['dates'][1:] and meta['ends'][-1]=='2026-09-24','Original evaluation metadata only')
require(len(meta['etfs'])==len(set(meta['etfs']))==17,'Original17ETF metadata')
for names in meta['symbols']:
    require(len(names)==49 and names==sorted(set(names)),'Original49 unique sorted securities')
    require(sum(s not in meta['etfs'] for s in names)==32,'Original32 directstock +17ETF')
for d in meta['dates']+meta['ends']:iso(d)
cache=read(local+'CACHE_HASHES.json','e3b440f8e2f2669d535cb92d33a2302be7ceec4f54792ff4078a7d976356460b')
for name,h in cache.items():bind(local+name,h)
cal_name='outputs/monthly_paired_target143/CALENDAR.json'
cal=read(cal_name,'8ee4da69fba84db1edee37a7e0f5af5d3ab16c6257e5d37cddbd531f01404622')
for name,h in cal['provenance']['source_hashes'].items():bind(name,h)
sessions=cal['sessions'];require(sessions==sorted(set(sessions)),'Unique official calendar')
positions={iso(d):i for i,d in enumerate(sessions)};windows=[]
for i,d in enumerate(meta['dates']):
    require(d in positions,'Query on official calendar');t=positions[d]
    require(t>=294 and t+1<len(sessions),'Official source/witness coverage')
    partial=d=='2016-08-17'
    require(not partial or i==0,'Named initial partial only')
    require(partial or sessions[t+1][:7]!=d[:7],'Every regular query official monthend')
    ds=sessions[t-294:t-41];require(len(ds)==253 and positions[ds[-1]]-positions[ds[0]]==252,'Exact253 sessions/252 intervals')
    old=prior['calendar']['windows'][i]
    require(old['date']==d and old['price_end']==ds[-1] and positions[ds[0]]-positions[old['price_start']]==20,'Window20 sessions shorter than245, unchanged skip42 end')
    windows.append(dict(query_index=i,date=d,initial_partial=partial,query_session_position=t,price_start=ds[0],first_return_date=ds[1],price_end=ds[-1],source_start_position=t-294,source_end_position=t-42,prices=253,return_intervals=252,next_official_session=sessions[t+1],source_dates_sha256=hashlib.sha256(json.dumps(ds,separators=(',',':')).encode()).hexdigest()))
require(len(windows)==122 and sum(x['initial_partial'] for x in windows)==1,'122 query clocks including one bypass')
policy=read('work/peer239_maturity_policy.json','3bbcbcbe141fb0d2d96cd64ca74f2ee1e9450e12c1f2594c7cb84fb98bf64862')
evidence=read('work/peer239_listing_gap_evidence.json',policy['evidence']['sha256'])
bounds=prior['listing_bounds'];require(bounds==policy['securities'] and len(bounds)==2,'Exact reviewed security lower bounds')
byid={x['id']:x for x in evidence['primary_evidence']};bound_evidence=[]
for b in bounds:
    e=byid[b['evidence_id']];require(iso(e['publication_date'])<iso(b['available_from']),'Declared next-day availability')
    bind(e['raw_path'],e['raw_sha256'])
    for key,hkey in (('retrieval_receipt','retrieval_sha256'),('read_text_path','text_sha256')):
        if key in e:bind(e[key],e[hkey])
    bound_evidence.append(dict(bound=b,primary_evidence=e,clock_certified=False))
# Existing239 records authenticate these identities; only their file bytes are hashed.
for name in ('outputs/monthly_peer239_source_audit/REPORT.json','outputs/monthly_peer239_maturity_audit/REPORT.json','outputs/monthly_peer239_maturity_audit/COMPLETION.json'):
    bind(name,prior['input_sha256'][name])
identities=[];seen=set()
for old in prior['listing_identity_cells']:
    i,j=old['query_index'],old['column_index'];b=next(b for b in bounds if b['symbol']==old['symbol'])
    require((i,j) not in seen and meta['dates'][i]==old['date'] and meta['symbols'][i][j]==old['symbol'] and old['issuer_key']==b['issuer_key'],'Exact listing query/column/symbol/issuer');seen.add((i,j))
    w=windows[i]
    identities.append(dict(query_index=i,column_index=j,date=w['date'],symbol=old['symbol'],issuer_key=old['issuer_key'],new_price_start=w['price_start'],new_price_end=w['price_end'],bound_visible=b['available_from']<=w['date'],start_precedes_bound=w['price_start']<b['not_before'],in_actual_shortlist='UNKNOWN_NOT_COMPUTED',actual_source_available='UNKNOWN_NOT_COMPUTED'))
expected={(i,j) for i,names in enumerate(meta['symbols']) for j,s in enumerate(names) if s in {b['symbol'] for b in bounds}}
require(seen==expected and len(identities)==5,'All five identity occurrences without E/shortlist knowledge')
for name in ('outputs/monthly_stability_monitor/inputs.py','outputs/monthly_local_linear70/study.py','outputs/monthly_architecture_restart/study.py','work/monthly_risk_consensus219_host.py'):
    bind(name,prior['input_sha256'][name])
implementation=bind(Path(__file__))
report=dict(schema='after246_portfolio_source_preflight_v1',created_utc=datetime.now(timezone.utc).isoformat(),passed=True,status='METADATA_VERIFIED_ACTUAL_SHORTLIST_AND253_SUPPORT_UNKNOWN',source_only=True,metadata_only=True,source_admission_metadata_only=True,draft_path=DRAFT,draft_sha256=DRAFT_SHA,draft_not_strategy_registration=True,
 input_sha256=dict(sorted(PINS.items())),draft_evidence_sha256=draft['evidence_sha256'],raw_source_hashes=raw,raw_source_count=64,raw_source_hashes_verified=True,raw_source_map_matches_original=True,metadata_axes_verified=True,calendar_clock_verified=True,source_window_recomputed_253=True,listing_identity_metadata_verified=True,
 metadata=dict(events=122,formal_monthends=121,named_initial_partial='2016-08-17',width=49,direct_stocks_per_event=32,etfs_per_event=17,distinct_symbols=len({s for row in meta['symbols'] for s in row}),earliest_query=meta['dates'][0],latest_query=meta['dates'][-1],end_date_metadata=meta['ends'][-1],original_etfs=meta['etfs'],original_META_sha256=PINS[local+'META.json']),
 calendar=dict(path=cal_name,sha256=PINS[cal_name],official_start=sessions[0],official_end=sessions[-1],session_count=len(sessions),source_clock='t-294..t-42 inclusive',prices=253,return_intervals=252,windows=windows,source_dates_hash_encoding='SHA256 of UTF-8 json.dumps(official253ISOdates,separators=(comma,colon))'),
 listing_bounds=bounds,listing_bound_evidence=bound_evidence,listing_identity_cells=identities,
 source_scope=dict(extra_sources=[],VOX_used=False,sector_mapping_used=False,SPY_benchmark_required=False,SPY_if_shortlisted='Ordinary original focal ETF with identical253-price source rule',original64_auxiliary_files='Whole original raw map preserved by hash; existing sector/auxiliary files in that map are not new portfolio-statistic dependencies',cache_archive='Original cache files byte-hashed only; no array or score/order contents decoded'),
 previous_source=dict(path=prior_name,sha256=PINS[prior_name],prior_prices=273,new_prices=253,removed_earlier_sessions=20,source_mask_reused=False,listing_identity_only_reused=True,original239_support_mask_reused=False),
 future_registered_checks=[
  'Define shortlist using original P/E first and preserve it before any price-validity test. This preflight has not read E/P/F/orders and cannot establish actualshortlist membership or support.',
  'For each required shortlist name inspect all253 positive finite real nonboolean adjusted closes on exact official t-294..t-42 sessions. No fill, row compression, rank6 replacement, substitution, date-dependent trimming or future execution filter. ETFs follow the same rule; noSPY/sector benchmark gate.',
  'Any missing required shortlisted source triggers common exact fallback across all experimental arms under the final registered contract; do not remove an unavailable name and silently alter the domain.',
  'If one of the five reviewed issuer occurrences becomes required, verify exact query/column/date/symbol/issuer/start294/end42. Visible not_before crossed by a falsely complete source must hardfail identity contamination, not be hidden by fallback or another missing name. Future evidence is not appliedearly; the five identities alone do not prove allothers.',
  'After registration independently verify descriptive fixed-unit path algebra, same numerator/commonfeasible domain, joint/diagonal variance semantics and zero-risk commonfallback. None of those statistics or feasibility constraints has been executed here.',
  'Future host must verify original27 arrays and exact incumbent orders, sourcehashes/runtime/frozen context, source and decision prefixes, and originalprotected archives. All5 metadata identity cells are inventory evidence, not a claim that they are shortlisted.'
 ],
 limitations=[
  'Fresh253-session clock/hash/identity preflight only. No actual source completeness, shortlist, scores, orders, portfolio statistics or returns computed. Prior273 support cannot substitute for new shortlist-specific admission.',
  'NoVOX or sector mapping is needed for this univariate basket-price hypothesis. This does not imply the originalLocal67 baseline had no other data dependencies; they remain frozen in its inherited closure.',
  'OnlySPCX/SKHY five explicit current-universe issuer occurrences have newlisting bounds. Complete othersecurity/ETF inception, alias/corporate-action continuity and original2016leading-gap splice remain inherited assumptions.',
  'Next-calendar-day availability is a declared research assumption; no certified first-publication byte vintage or exhaustive correction history. SPCX is the newUSissue, not oldsame-tickerETF; SKHY is NasdaqADR, not Koreanordinary.',
  'EstimatedhistoricalTop32/theme17 and adjusted historical prices remain retrospective; repeated sample use is not pristineOOS. This preflight is not final247context, strategy registration, sourceadmission, author replication or promise of70percent.',
  '246 prepare/results were not read, polled or used. The full21444 archive was not redundantly rehashed; futurehost remains responsible for registered protection checks.'
 ],implementation=dict(path=str(Path(__file__).relative_to(ROOT)),sha256=implementation),
 actual_shortlist_computed=False,actual_support_computed=False,actual_source_coverage_recomputed=False,actual_new_features_computed=False,actual_model_fits_computed=False,actual_orders_computed=False,actual_returns_computed=False,npz_decoded=False,prices_loaded=False,loader_executed=False,sector_mapping_computed=False,registered_strategy=False,central_modified=False,observers_modified=False,read_246_actual_outputs=False,network_used=False,strict_pit=False)
with OUT.open('x') as f:json.dump(report,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
print(json.dumps(dict(path=str(OUT),sha256=sha(OUT),helper_sha256=implementation,input_pins=len(PINS),raw_source_files=64,events=122,source_prices=253,identity_cells=len(identities),first_window=windows[0],last_window=windows[-1],actual_shortlist='UNKNOWN',actual_support='UNKNOWN')))
