"""Round242 metadata/hash-only source-clock review; run through run_local.sh."""
from pathlib import Path
from datetime import date, datetime, timezone
import hashlib, json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'work/block_sharpe242_SOURCE_REVIEW.json'
CONTEXT='work/monthly_block_sharpe242_context.json'
CONTEXT_SHA='246528339472f8ffec8bebe09abd123eb20b9d1a3d9111bcf34ade2f21eab05d'
COVERAGE='work/after241_block_sharpe_coverage.json'
COVERAGE_SHA='12742466919f990aea7b6ea418dfab7f9b93951cf862bb291b3e8dc74c415db9'
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
require(c['schema']=='monthly_block_sharpe242_context_v1' and c['integration']['blend']==.5,'Fixed context')
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
implementation=bind(Path(__file__))
report=dict(schema='block_sharpe242_metadata_source_review_v1',created_utc=datetime.now(timezone.utc).isoformat(),passed=True,
    source_only=True,metadata_only=True,source_admission_metadata_only=True,prior_attempt=dict(exit_code=1,reason='Initial context SHA gate failed before inputs because root added prospective underflow clarification before registration; no output existed',tool_chunk='3b6a12'),context_sha256=CONTEXT_SHA,coverage_sha256=COVERAGE_SHA,
    input_sha256=dict(sorted(PINS.items())),evidence_sha256=dict(c['input_sha256']),raw_source_hashes=raw,raw_source_count=64,
    raw_source_hashes_verified=True,raw_source_map_matches_original=True,metadata_axes_verified=True,calendar_clock_verified=True,
    source_window_matches219=True,listing_identity_metadata_verified=True,listing_bounds=expected,listing_bound_evidence=listing_evidence,listing_identity_cells=identity_cells,
    metadata=dict(events=122,width=49,etf_count=17,stock_count_per_event=32,unique_candidate_symbols=len({s for row in meta['symbols'] for s in row}),known_listing_bound_securities=2,known_listing_identity_cells=len(identity_cells)),
    calendar=dict(path='outputs/monthly_paired_target143/CALENDAR.json',sha256=PINS['outputs/monthly_paired_target143/CALENDAR.json'],source_clock='t-294..t-42 inclusive',prices=253,daily_intervals=252,block_intervals=12,block_length_sessions=21,windows=windows),
    inherited_maturity_scope=dict(report='outputs/monthly_peer239_maturity_audit/REPORT.json',certified_window='Only t-252/t-42 endpoint status and stock issuer metadata under its own declared policy',mask_reused=False,certifies_new_full253_support=False,certifies_etf_support=False,not_used_for_new_factor_rank_or_eligibility=True),
    host_required_checks_after_registration=[
        'After new full253 feature oracle and before any orders, validate query dates, ordered symbols and exact official t-294/t-42 boundaries against immutable META/calendar. Recheck consumed physical prefixes identically.',
        'Use exact symbol AND whole-issuer identity proven by listing_identity_cells. Original META/239 source-report pins must remain unchanged; an unexpected bounded-symbol occurrence or changed issuer is a hard failure, not a ticker-only remap.',
        'For each visible bound available_from<=query, if t-294 date<not_before then source_available must be false. If the complete source window is valid, hard fail identity contamination even if either SD is zero and final feature available is false.',
        'Equality at not_before is not excluded. No future bound is applied early. Bounds do not authorize filling, alternate security histories, changing original E, modifying old source reports, or special-case masking valid input.',
        'All253 prices, including block-interior dates, must be independently validated; both daily and block positive sample SD share one feature mask. Source-valid and feature-available must be counted separately.',
        'Do not import239 peer support or its t-252 maturity mask into242. All49 original names remain in metadata; source unavailability yields both new targets unavailable and original core retained. New source coverage must be disclosed, not inferred from this receipt.'
    ],
    limitations=[
        'Only the two explicit known listed-security bounds are newly cross-checked here. No exhaustive inception/security-master certification for the other71 direct-stock identities or17ETFs.',
        'SPCX bound is the exchange-described new issue lower bound, not realized first trading. June/July must not borrow the later August actual-trading confirmation; earlier SPCX ETF history is not this issuer.',
        'SKHY is the Nasdaq ADR, not the earlier Korean ordinary listing; depositary/CUSIP/ratio history is not newly certified. No foreign-history splice.',
        'Inherited estimated Top32 and theme17, retrospective adjusted prices, aliases and original2016 leading-gap source splice remain declared research assumptions; immutable bytes do not establish strict PIT.',
        'Next-calendar-day public availability is an explicit research convention. Original first-publication content versions and complete revisions are not certified.',
        'This receipt hashes raw/cache bytes without decoding prices or NPZ and verifies JSON identities/calendars only. It does not establish actual242 source support, variance, rank, order or return correctness.',
        'No original baseline or prior historical mask is retroactively repaired. Source unknowns stay explicit; final host/synthetic/independent audits remain required.'
    ],
    source_lineage_reference=lineage,implementation=dict(path=str(Path(__file__).relative_to(ROOT)),sha256=implementation),
    actual_support_computed=False,actual_source_coverage_recomputed=False,actual_new_features_computed=False,actual_new_orders_computed=False,actual_new_returns_computed=False,
    actual_model_fits_computed=False,actual_returns_computed=False,npz_decoded=False,source_npz_decoded=False,prices_loaded=False,loader_executed=False,
    original_sources_modified=False,observers_modified=False,strict_pit=False,host_not_pinned_to_avoid_circular_binding=True)
with OUT.open('x') as f:json.dump(report,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps(dict(path=str(OUT),sha256=sha(OUT),implementation_sha256=implementation,input_pins=len(PINS),raw_source_files=64,metadata_events=122,listing_identity_cells=len(identity_cells))))
