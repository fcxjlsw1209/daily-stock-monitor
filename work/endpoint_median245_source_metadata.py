"""Round245 metadata/hash-only source-clock review; run through run_local.sh."""
from pathlib import Path
from datetime import date, datetime, timezone
import hashlib, json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'work/endpoint_median245_SOURCE_REVIEW.json'
CONTEXT='work/monthly_endpoint_median245_context.json'
CONTEXT_SHA='45b89115124079ecbcb9b875a22e69675384736f4b180b6d3769fce285f9e570'
COVERAGE='work/after244_endpoint_coverage.json'
COVERAGE_SHA='1bc72f208c4e321db5077a8b04ee98ef34d8e44b5d5149d5f974615994f26ce7'
PINS={}
def require(ok,msg):
    if not ok: raise ValueError(msg)
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
    p=path(name);h=sha(p);require(expected is None or expected==h,'SHA mismatch '+str(p))
    key=str(p.relative_to(ROOT));require(key not in PINS or PINS[key]==h,'Conflicting pin')
    PINS[key]=h;return h
def read(name,expected=None):
    bind(name,expected);return json.loads(path(name).read_text())
def iso(s):
    require(type(s) is str and date.fromisoformat(s).isoformat()==s,'Canonical date');return s

require(not OUT.exists(),'Write-once source review')
c=read(CONTEXT,CONTEXT_SHA);coverage=read(COVERAGE,COVERAGE_SHA)
require(c['schema']=='monthly_endpoint_median245_context_v1' and c['integration']['budget']==.025,'Fixed context')
for name,h in c['input_sha256'].items():bind(name,h)
for name,h in coverage['input_sha256'].items():bind(name,h)
local='outputs/monthly_local_linear70/'
p=read(local+'PLAN.json','0016175aee1df43b734010d80bd543b881c57036a7f8c6ab73428340203afc96')
original=read(local+'INPUT_HASHES.json','5ac2b639fa8c944688787eba47ea1b420bfbe9c6e1e051a3d827d8b2b32b264d')
prior=read('work/risk_consensus219_SOURCE_REVIEW.json','cbf98df0568cdfb67260e84b8126c72cf3b981ceee94ade50a34074d8cb01b5b')
raw=p['raw_source_hashes'];require(len(raw)==64 and raw==prior['raw_source_hashes'],'Original64 lineage')
for name,h in raw.items():
    require(original.get(name)==h,'Original source registry');bind(name,h)
meta=read(local+'META.json','3b46968f486e88fa170ac01f120dd5ef74ebd5a56243c6991c5a4fea506076f4')
require(len(meta['dates'])==len(meta['ends'])==len(meta['symbols'])==122,'Original122 axis')
require(meta['dates']==sorted(set(meta['dates'])) and meta['dates'][0]=='2016-08-17','Original unique ordered dates')
require(len(set(meta['etfs']))==len(meta['etfs'])==17,'Original17ETF')
require(all(len(s)==len(set(s))==49 for s in meta['symbols']),'Original49 query names')
for s in meta['dates']+meta['ends']:iso(s)
cache=read(local+'CACHE_HASHES.json')
for name,h in cache.items():bind(local+name,h)
cal=read('outputs/monthly_paired_target143/CALENDAR.json','8ee4da69fba84db1edee37a7e0f5af5d3ab16c6257e5d37cddbd531f01404622')
for name,h in cal['provenance']['source_hashes'].items():bind(name,h)
sessions=cal['sessions'];require(sessions==sorted(set(sessions)),'Official session ordering')
positions={iso(d):i for i,d in enumerate(sessions)};windows=[]
for day in meta['dates']:
    require(day in positions,'Query in official calendar');t=positions[day]
    require(t>=314 and t+1<len(sessions),'Source calendar range')
    require(day=='2016-08-17' or day[:7]!=sessions[t+1][:7],'Formal month-end')
    window=sessions[t-314:t-41];require(len(window)==273,'Full273 official sessions')
    windows.append(dict(date=day,price_start=window[0],first_return_date=window[1],price_end=window[-1]))
require([(x['date'],x['price_end']) for x in windows]==[(x['date'],x['price_end']) for x in prior['calendar']['windows']],'Same query and skip42 end, newly extended start')
policy=read('work/peer239_maturity_policy.json','3bbcbcbe141fb0d2d96cd64ca74f2ee1e9450e12c1f2594c7cb84fb98bf64862')
evidence=read('work/peer239_listing_gap_evidence.json',policy['evidence']['sha256'])
source=read('outputs/monthly_peer239_source_audit/REPORT.json','dd7c7f39ae529fc62ec15c82e7f736fe6f5667c76a10bfb4e5b0ef1fc5148b7c')
mature=read('outputs/monthly_peer239_maturity_audit/REPORT.json','5138025ced5531b6d3447cc9bb9835e886b3f45157b6b4e03a54e888fcc97f7c')
close=read('outputs/monthly_peer239_maturity_audit/COMPLETION.json','fdeaf5afdd7a2d2823d0ee66bcc00a13b4d767c88b1eded2f9b575f1f17b2dc8')
require(close['source_only'] is True and close['strict_pit'] is False,'Prior source scope')
for field,name in (('plan_sha256','SOURCE_PLAN.json'),('report_sha256','REPORT.json'),('independent_review_sha256','INDEPENDENT_REVIEW.json')):
    bind('outputs/monthly_peer239_maturity_audit/'+name,close[field])
ind=read('outputs/monthly_peer239_maturity_audit/INDEPENDENT_REVIEW.json')
require(ind['passed'] is True and [x['prefix'] for x in ind['physical_prefixes']]==[0,1,53,88,122],'Prior metadata independent receipt')
require(policy['rules']['source_dates']=='official t-252 and t-42','Do not reuse short source mask')
expected=[dict(symbol='SPCX',issuer_key='https://companiesmarketcap.com/spacex/marketcap/',not_before='2026-06-08',available_from='2026-06-09',kind='exchange_new_listing_lower_bound',evidence_id='spcx_nasdaq_new_issue'),dict(symbol='SKHY',issuer_key='https://companiesmarketcap.com/sk-hynix/marketcap/',not_before='2026-07-10',available_from='2026-07-11',kind='issuer_actual_adr_start',evidence_id='skhy_adr_listing')]
require(policy['securities']==expected,'Exact reviewed bounds')
primary={x['id']:x for x in evidence['primary_evidence']};listing_evidence=[]
for b in expected:
    e=primary[b['evidence_id']];require(iso(e['publication_date'])<b['available_from'],'Evidence publication clock')
    bind(e['raw_path'],e['raw_sha256'])
    for key,hkey in (('retrieval_receipt','retrieval_sha256'),('read_text_path','text_sha256')):
        if key in e:bind(e[key],e[hkey])
    listing_evidence.append(dict(bound=b,primary_evidence=e,clock_certification='Next-calendar-day availability is inherited declared assumption; first-publication byte vintage is not certified.'))
record_map={(r['query_index'],r['column_index']):r for r in source['records']}
mature_map={(r['query_index'],r['column_index']):r for r in mature['records']}
identity_cells=[]
for b in expected:
    cells=[]
    require(b['symbol'] not in meta['etfs'],'Bound instrument is a direct stock in original META')
    for i,names in enumerate(meta['symbols']):
        for j,symbol in enumerate(names):
            if symbol!=b['symbol']:continue
            r=record_map[(i,j)];m=mature_map[(i,j)]
            require(r['date']==m['date']==meta['dates'][i] and r['symbol']==m['symbol']==symbol and r['issuer_key']==m['issuer_key']==b['issuer_key'],'Full symbol/issuer/query identity')
            cells.append(dict(query_index=i,column_index=j,date=meta['dates'][i],symbol=symbol,issuer_key=b['issuer_key'],new_price_start=windows[i]['price_start'],new_price_end=windows[i]['price_end'],bound_visible=b['available_from']<=meta['dates'][i],start_precedes_bound=windows[i]['price_start']<b['not_before']))
    require(cells,'Reviewed bound has META identity evidence');identity_cells.extend(cells)
for name in ('work/peer239_maturity_sources.py','work/peer239_maturity_independent.py','work/peer239_maturity_host.py','work/peer239_source_contract.json','outputs/monthly_local_linear70/study.py','outputs/monthly_core_timing70_research/core_adapter.py','outputs/monthly_path70_research/incumbent_adapter.py','outputs/monthly_daily_episodes/episodes.py'):
    bind(name)
lineage=prior['source_lineage_reference'];bind(lineage['path'],lineage['sha256'])


prior244=read('work/up_participation244_SOURCE_REVIEW.json','358aa881cce62c3b90f8657148e83d770f662c484e2c8cf0f0821d09d57f5119')
require(prior244['raw_source_hashes']==raw and prior244['listing_bounds']==expected,'Original lineage/bounds unchanged')
identity_keys=('query_index','column_index','date','symbol','issuer_key')
require([{k:x[k] for k in identity_keys} for x in identity_cells]==[{k:x[k] for k in identity_keys} for x in prior244['listing_identity_cells']],'Same five complete issuer identities')
definition=read('work/after244_endpoint_definition_review.json','a4dcd72f6d79510b8c74d5fe379998a2abbdfe2219a39bad66b1cefc95f8db6c')
plan244=read('outputs/monthly_up_participation244/PLAN.json','1776f9471e431fb72c3729216d38df9f9a078fc09fd1a993c38f4a806395c48a')
for name in ('outputs/monthly_stability_monitor/inputs.py','outputs/monthly_state_rank_research/state_rank.py','outputs/monthly_architecture_restart/study.py','work/monthly_risk_consensus219_host.py'):
    bind(name)
pair_clocks=[]
for row in windows:
    t=positions[row['date']];pairs=[]
    for k in range(21):
        start=t-314+k;end=start+252
        require(end<=t-42 and end-start==252,'Exactly252 intervals per annual pair')
        pairs.append(dict(k=k,start_date=sessions[start],end_date=sessions[end],start_offset=314-k,end_offset=62-k))
    require(pairs[0]['start_date']==row['price_start'] and pairs[-1]['end_date']==row['price_end'],'Exact273 union')
    pair_clocks.append(dict(date=row['date'],pairs=pairs))
implementation=bind(Path(__file__))
report=dict(schema='endpoint_median245_metadata_source_review_v1',created_utc=datetime.now(timezone.utc).isoformat(),passed=True,
 source_only=True,metadata_only=True,source_admission_metadata_only=True,context_sha256=CONTEXT_SHA,coverage_sha256=COVERAGE_SHA,
 definition_review_sha256=PINS['work/after244_endpoint_definition_review.json'],input_sha256=dict(sorted(PINS.items())),evidence_sha256=dict(c['input_sha256']),raw_source_hashes=raw,raw_source_count=64,
 raw_source_hashes_verified=True,raw_source_map_matches_original=True,metadata_axes_verified=True,calendar_clock_verified=True,source_window_recomputed_273=True,listing_identity_metadata_verified=True,
 listing_bounds=expected,listing_bound_evidence=listing_evidence,listing_identity_cells=identity_cells,
 metadata=dict(events=122,width=49,etf_count=17,stock_count_per_event=32,unique_candidate_symbols=len({s for row in meta['symbols'] for s in row}),known_listing_bound_securities=2,known_listing_identity_cells=len(identity_cells)),
 calendar=dict(path='outputs/monthly_paired_target143/CALENDAR.json',sha256=PINS['outputs/monthly_paired_target143/CALENDAR.json'],source_clock='t-314..t-42 inclusive',prices=273,adjacent_intervals=272,annual_return_intervals=252,overlapping_windows=21,windows=windows,pair_clocks=pair_clocks,calendar_start=sessions[0],calendar_end=sessions[-1],session_count=len(sessions)),
 benchmark=dict(required=False,common_benchmark_mask=False,SPY_treated_as_original_focal_ETF=True,extra_benchmark_sources=[]),
 prior_source_comparison=dict(prior_review='work/up_participation244_SOURCE_REVIEW.json',prior_prices=253,new_prices=273,extra_earlier_sessions=20,identity_and_raw_map_reused=True,price_mask_reused=False,prior_support_count_not_reused=True),
 inherited_source_closure=dict(plan='outputs/monthly_up_participation244/PLAN.json',input_pins=len(plan244['input_sha256']),scope='Parent245 registration/gate inherit244 complete immutable input closure. This metadata review binds prior PLAN bytes and verifies its own listed direct raw/cache/source pins; no full21444 archive rescan.'),
 inherited_maturity_scope=dict(report='outputs/monthly_peer239_maturity_audit/REPORT.json',certified_window='Only t-252/t-42 endpoint status and stock issuer metadata under original conditional policy',mask_reused=False,certifies_new_full273_support=False,certifies_etf_support=False,not_used_for_factor_rank_or_eligibility=True),
 host_required_checks_after_registration=[
  'Recompute persecurity all273 exact official-session adjusted prices t-314..t-42. Missing/interior/invalid source yields common unavailable; no date compression, fill, endpoint-only validity or inherited244/242 mask.',
  'All49 original query securities, including ETFs andSPY as an ordinary focal asset, use the same univariate source rule. NoSPY benchmark gate, variance minimum, up/down group condition, or originaleligibility filter enters source validity or ranking.',
  'For k0..20 in oldest-first273 array validate252-return pair P[252+k]/P[k]-1. Exact pairclock union starts t-314, nott-315; there is no unused earlier-price requirement. Retain all273 interior-price checks even though only42 endpoint observations enter ratios.',
  'Exact binary64 rational gains, median sortedindex10, arithmeticmean sum/21, and common fullpool average-pct ranks require independent new245 feature/count/audit verification. Zero/negative/flat/tied gains remain source-valid; no rawfloat rounding before rank.',
  'Match exact query/column/date/symbol/fullissuer/start314/end42 listing proof, including consumedprefixes. Unexpected/missing/duplicate identities fail. For visiblebound and source_start<not_before, own-source_available must be false or fail identitycontamination. Equality atbound allowed; futureevidence not appliedearly.',
  'No covert masking of valid contaminatedprices, oldSPCX ETF substitution, SKHYforeignordinary splice, changes to originalE, or source repairs. Only five knownidentity occurrences have reviewedlowerbounds; unknown completeness elsewhere must be disclosed.',
  'Verify original27 arrays, source64 and exact originalorders, .025 fundedaux formula including extra.01 extension, namedpartialno-op, fullpre/post/order audits, one originalepisode perarm and all123prefixes. Actual support remains unknown untilregisteredprepare.',
 ],
 limitations=[
  'This is fresh calendar/identity/hash verification only; no actual273-price coverage or newfactor/order/return computed. 244 had253 support,239 had twoendpoint support: neither certifies this longer source window.',
  '21 annual observations overlap heavily; they are not independent samples. This source review does not certify estimator quality, novelty or prospective returns.',
  'OnlySPCX andSKHY explicit issuer-bound dates are checked. Other71 directstock identities and17ETFs lack new exhaustiveinception/securitymaster certification.',
  'SPCX is the exchange-described newissue lowerbound, not realized firsttrade or oldsame-ticker ETF. SKHY is the NasdaqADR, not the priorKoreanordinary. No laterconfirmation may be borrowedearlier.',
  'Next-calendar-day evidenceavailability is an inherited researchassumption; original firstpublication bytes and completecorrectionhistory are not certified.',
  'EstimatedTop32/theme17, retrospective adjustedprices/aliases/corporateactions and original2016leadinggap splice remain declared assumptions. StrictPIT isfalse; immutable hashes do not repair lineage.',
  'Raw/cache files are read only as bytes for hashing; JSON identities and official-calendar metadata are inspected. No loader, numericalstrategy orNPZ decoder is executed.',
 ],
 source_lineage_reference=lineage,implementation=dict(path=str(Path(__file__).relative_to(ROOT)),sha256=implementation),
 actual_support_computed=False,actual_source_coverage_recomputed=False,actual_new_features_computed=False,actual_new_orders_computed=False,actual_new_returns_computed=False,actual_model_fits_computed=False,actual_returns_computed=False,npz_decoded=False,source_npz_decoded=False,prices_loaded=False,loader_executed=False,original_sources_modified=False,observers_modified=False,strict_pit=False,host_not_pinned_to_avoid_circular_binding=True)
with OUT.open('x') as f:json.dump(report,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps(dict(path=str(OUT),sha256=sha(OUT),implementation_sha256=implementation,input_pins=len(PINS),raw_source_files=64,metadata_events=122,listing_identity_cells=len(identity_cells),source_prices=273,annual_windows=21,inherited_input_pins=len(plan244['input_sha256']))))
