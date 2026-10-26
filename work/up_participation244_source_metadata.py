"""Round244 metadata/hash-only source-clock review; run through run_local.sh."""
from pathlib import Path
from datetime import date, datetime, timezone
import hashlib, json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'work/up_participation244_SOURCE_REVIEW.json'
CONTEXT='work/monthly_up_participation244_context.json'
CONTEXT_SHA='27761991e543d95a44006255c537160fd924ad858f2e5767c89a4b5e99ee3fe8'
COVERAGE='work/after243_up_participation_coverage.json'
COVERAGE_SHA='7bc311bf33ff35d5bf486bbb779875184c5f3f045f87794d9509979c06b117ed'
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
require(c['schema']=='monthly_up_participation244_context_v1' and c['integration']['budget']==.025,'Fixed context')
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
    require(t>=294 and t+1<len(sessions),'Source calendar range')
    require(day=='2016-08-17' or day[:7]!=sessions[t+1][:7],'Formal month-end')
    window=sessions[t-294:t-41];require(len(window)==253,'Full253 official sessions')
    windows.append(dict(date=day,price_start=window[0],first_return_date=window[1],price_end=window[-1]))
require(windows==prior['calendar']['windows'],'Source clock matches frozen219')
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

old242=read('work/block_sharpe242_SOURCE_REVIEW.json','8fa09aed161426fa3281f8134ddc62f75bbe51e3bff75c93efe3734ee0e7a840')
require(old242['raw_source_hashes']==raw and old242['calendar']['windows']==windows and old242['listing_bounds']==expected and old242['listing_identity_cells']==identity_cells,'Same-window historical metadata parity')
plan242=read('outputs/monthly_block_sharpe242/PLAN.json','f00bb5aa01bbf90514d19a95c41ed53cce6704eaab1954cd842442c218fded57')
plan243=read('outputs/monthly_core_conditioned243/PLAN.json',c['input_sha256']['outputs/monthly_core_conditioned243/PLAN.json'])
for n in ('outputs/monthly_stability_monitor/inputs.py','outputs/monthly_state_rank_research/state_rank.py','outputs/monthly_architecture_restart/study.py','work/monthly_risk_consensus219_host.py'):
    bind(n)
lineage_data=read(lineage['path'],lineage['sha256'])
spy=next(x for x in read('outputs/four_factor_rebound/ETF_SOURCES.json') if x['symbol']=='SPY')
spy_path='outputs/index_union/sources/yahoo_prices/SPY.csv'
spy_sha='9fbc17731dfadfb74382d076a281dcc935f1971f74f8d08cfe860726b69edff5'
require(path(spy['file'])==path(spy_path) and spy['sha256']==spy_sha,'Pinned benchmark identity')
require('SPY' in meta['etfs'] and all('SPY' in names for names in meta['symbols']),'Benchmark retained in original universe')
spy_lineage=lineage_data['source_lineage']['SPY']
require(len(spy_lineage)==1 and spy_lineage[0]['operation']=='overwrite_full_column' and path(spy_lineage[0]['path'])==path(spy_path) and spy_lineage[0]['sha256']==spy_sha,'Exact benchmark overwrite lineage')
for mf in ('outputs/monthly_reconstruction_v2/ETF_MANIFEST.json','outputs/monthly_reconstruction_v2/foreign/ETF_MANIFEST.json'):
    require(not any(x['symbol']=='SPY' for x in read(mf)),'No later ETF-manifest override')
require(not (ROOT/'outputs/monthly_2016_extension/prices/SPY.csv').exists(),'No SPY leading-gap overlay')
bind(spy_path,spy_sha)
implementation=bind(Path(__file__))
report=dict(schema='up_participation244_metadata_source_review_v1',created_utc=datetime.now(timezone.utc).isoformat(),passed=True,
    source_only=True,metadata_only=True,source_admission_metadata_only=True,context_sha256=CONTEXT_SHA,coverage_sha256=COVERAGE_SHA,
    input_sha256=dict(sorted(PINS.items())),evidence_sha256=dict(c['input_sha256']),raw_source_hashes=raw,raw_source_count=64,
    raw_source_hashes_verified=True,raw_source_map_matches_original=True,metadata_axes_verified=True,calendar_clock_verified=True,
    source_window_matches219=True,source_window_matches242=True,listing_identity_metadata_verified=True,
    listing_bounds=expected,listing_bound_evidence=listing_evidence,listing_identity_cells=identity_cells,
    security_source_distinct_from_benchmark=True,common_mask_not_inherited_from242=True,spy_source_metadata_verified=True,
    metadata=dict(events=122,width=49,etf_count=17,stock_count_per_event=32,unique_candidate_symbols=len({s for row in meta['symbols'] for s in row}),known_listing_bound_securities=2,known_listing_identity_cells=len(identity_cells)),
    calendar=dict(path='outputs/monthly_paired_target143/CALENDAR.json',sha256=PINS['outputs/monthly_paired_target143/CALENDAR.json'],source_clock='t-294..t-42 inclusive',prices=253,daily_intervals=252,windows=windows),
    benchmark=dict(symbol='SPY',original_etf_member=True,present_in_all122_query_rows=True,path=spy_path,sha256=spy_sha,field='Adj Close',operation='overwrite_full_column',manifest='outputs/four_factor_rebound/ETF_SOURCES.json',lineage=spy_lineage,loader='outputs/monthly_stability_monitor/inputs.py',preserves_all_adjusted_columns=True,stored_price_units='Inherited vendor adjusted-close units; no new normalization or corporate-action/PIT certification'),
    inherited_source_closures=dict(round242_plan='outputs/monthly_block_sharpe242/PLAN.json',round242_input_pins=len(plan242['input_sha256']),round243_plan='outputs/monthly_core_conditioned243/PLAN.json',round243_input_pins=len(plan243['input_sha256']),scope='Parent244 registration and gate inherit243 full immutable inputs. This receipt binds prior PLAN bytes and freshly checks only its listed direct metadata/raw/cache pins; it does not rehash the full21444 protected archive.'),
    inherited_maturity_scope=dict(report='outputs/monthly_peer239_maturity_audit/REPORT.json',certified_window='Only t-252/t-42 endpoint status and stock issuer metadata under its own declared policy',mask_reused=False,certifies_new_full253_support=False,certifies_etf_support=False,not_used_for_new_factor_rank_or_eligibility=True),
    host_required_checks_after_registration=[
        'Validate exact query dates, ordered symbols and official t-294/t-42 boundaries against immutable META/calendar, including consumed physical prefixes. Retain SPY benchmark even though it belongs to ETF17 and appears as its own focal security.',
        'Validate all253 own-security prices independently before any benchmark/common-availability decision. source_available must mean own-security full253 completeness only; benchmark failure or fewer40 up days must not suppress this source flag or listing proof.',
        'Use exact symbol AND whole-issuer identity proven by listing_identity_cells. Missing, duplicate, unexpected or changed proof tuples(query_index,column_index,date,symbol,issuer,start294,end42) fail; do not join by ticker alone.',
        'For every known bound visible at query, start294 before not_before must have own-security source_available false; if true, hardfail identity contamination even when benchmark is missing or final targets unavailable. Equality at bound is allowed. Do not apply future evidence early, mask contaminated valid prices, fill prices, splice alternate histories or modify original E.',
        'Validate all253 SPY prices independently. Both candidate/control share complete own-and-SPY support and at least40 exact up-day comparisons. Do not inherit any242 sample-volatility mask, down-day minimum, sector mapping, peer eligibility or source-support count.',
        'Full support count/sign/rational-win and tie comparisons, probabilities and common target mask require new independent244 checks after registration. This metadata receipt proves no actual244 price availability, up-day count, ratio, target, rank, score, order or return.',
        'Unavailable targets preserve original pre and inventory. Valid zero remains a funded zero target, including SPY self when support exists. Initial named2016-08-17 remains the fixed no-op integration exception; source metadata still follows its exact clock.',
        'Parent host must bind all original27 cached arrays and immutable inputs, exact original incidence, one episode per arm, and independent full/pre/post score audit. Only fixed first88 zero-cost paths may select; costs remain diagnostic.'
    ],
    limitations=[
        'Only the two explicit known listed-security bounds are cross-checked here; no exhaustive inception/security-master certification for other71 direct stocks or17ETFs.',
        'Correct instrument ticker is SPCX, not SPXC. Its bound is exchange new-issue lower bound, not realized first trading. Earlier same-ticker ETF history is not this issuer; June/July must not borrow later August actual-trade confirmation.',
        'SKHY is the Nasdaq ADR, not the Korean ordinary share. Depositary/CUSIP/ratio history is not newly certified; foreign-history splice is unauthorized.',
        'Estimated historicalTop32/theme17, retrospective adjusted prices/corporate actions, aliases and original2016 leading-gap splice remain research assumptions; immutable hashes do not establish strict PIT.',
        'Next-calendar-day availability of listing evidence is an inherited declared convention; original first-publication content versions and complete revision history remain uncertified.',
        'Exact binary64-rational comparisons certify order of stored values, not vendor precision, tradability or original publication vintages. Market-up conditional win share is not a predictive probability or proof of alpha.',
        'This source-only receipt hashes raw/cache bytes and examines JSON/source code only. It does not read actual price values or NPZ arrays, execute loaders, or infer actual244 common support from prior242/239 results.',
        'No frozen source, prior mask, observer or registry is repaired or promoted. Any new source gap must be reported by the registered244 calculation; no covert substitutes or parameter changes.'
    ],
    source_lineage_reference=lineage,implementation=dict(path=str(Path(__file__).relative_to(ROOT)),sha256=implementation),
    actual_support_computed=False,actual_source_coverage_recomputed=False,actual_new_features_computed=False,actual_new_orders_computed=False,actual_new_returns_computed=False,
    actual_model_fits_computed=False,actual_returns_computed=False,npz_decoded=False,source_npz_decoded=False,prices_loaded=False,loader_executed=False,
    original_sources_modified=False,observers_modified=False,strict_pit=False,host_not_pinned_to_avoid_circular_binding=True)
with OUT.open('x') as f:json.dump(report,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps(dict(path=str(OUT),sha256=sha(OUT),implementation_sha256=implementation,input_pins=len(PINS),raw_source_files=64,metadata_events=122,listing_identity_cells=len(identity_cells),inherited_closures=report['inherited_source_closures'])))
