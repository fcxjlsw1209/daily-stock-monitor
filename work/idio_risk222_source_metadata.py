"""Round222 write-once SHA/source-metadata review. Use the offline wrapper."""
from pathlib import Path
from datetime import datetime, timezone, date
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'work/idio_risk222_SOURCE_REVIEW.json'
CONTEXT=ROOT/'work/monthly_idio_risk222_context.json'
import argparse
_parser=argparse.ArgumentParser()
_parser.add_argument('--context-sha256',required=True)
_args=_parser.parse_args()
CONTEXT_SHA=_args.context_sha256
if CONTEXT_SHA!='68bc4e9362a9c4489d2a7725961ff84f262e5ae844ddb80ff46c8122879b65a9':
    raise ValueError('Final222 context digest required')
if len(CONTEXT_SHA)!=64 or any(c not in '0123456789abcdef' for c in CONTEXT_SHA):
    raise ValueError('Explicit lowercase context SHA required')
COVERAGE=ROOT/'work/after221_idio_risk_coverage.json'
COVERAGE_SHA='bb6f530a6b3d8d650dd419c96dba86b832a720a1d6523cd543e8f923eb1f9f82'
LOCAL=ROOT/'outputs/monthly_local_linear70'
CALENDAR=ROOT/'outputs/monthly_paired_target143/CALENDAR.json'
CALENDAR_SHA='8ee4da69fba84db1edee37a7e0f5af5d3ab16c6257e5d37cddbd531f01404622'
PINS={}

def require(ok,message):
    if not ok:raise ValueError(message)

def local(name):
    p=Path(name)
    p=p if p.is_absolute() else ROOT/p
    require(p.is_relative_to(ROOT) and 'CloudStorage' not in str(p),'Nonlocal source: '+str(p))
    require(p.resolve().is_relative_to(ROOT),'Symlink escape: '+str(p))
    return p

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def bind(name,expected=None):
    p=local(name);actual=sha(p)
    require(expected is None or expected==actual,'Digest mismatch: '+str(p))
    require(str(p) not in PINS or PINS[str(p)]==actual,'Conflicting digest')
    PINS[str(p)]=actual
    return actual

def read(name,expected=None):
    p=local(name);bind(p,expected)
    return json.loads(p.read_text())

def iso(value):
    require(type(value) is str and date.fromisoformat(value).isoformat()==value,'Canonical ISO date required')
    return date.fromisoformat(value)

require(not OUT.exists(),'Write-once source review already exists')
context=read(CONTEXT,CONTEXT_SHA)
require(context['round']==222 and context['name']=='monthly_idio_risk222','Wrong round context')
require(context['formation']['start_offset']==-294 and context['formation']['end_offset']==-42 and context['formation']['return_intervals']==252,'Formation contract')
require(context['integration']['blend']==.5,'Fixed half-core contract')
require(context['formation']['risk_start_offset']==-63 and context['formation']['risk_end_offset']==0 and context['formation']['risk_return_intervals']==63 and context['formation']['market_symbol']=='SPY','Fixed current63 market-risk clock')
require(context['formation']['new_predictive_model_fits']==0 and context['formation']['parameter_search'] is False,'No new predictive training or parameter search')
require(context['arms']==['incumbent_local67','idiosyncratic_core','total_volatility_control'],'Fixed paired arms')
evidence={}
for name,expected in context['evidence_sha256'].items():evidence[name]=bind(name,expected)
coverage=read(COVERAGE,COVERAGE_SHA)
require(coverage['family_novelty'] is False and coverage['npz_decoded'] is False,'Limited novelty contract')
for key in ('new_features_computed','new_orders_computed','new_returns_computed','new_predictive_models_fit','new_trials_registered'):
    require(type(coverage[key]) is int and coverage[key]==0,'Coverage computation boundary')
for name,expected in coverage['input_sha256'].items():bind(name,expected)
plan=read(LOCAL/'PLAN.json','0016175aee1df43b734010d80bd543b881c57036a7f8c6ab73428340203afc96')
original=read(LOCAL/'INPUT_HASHES.json','5ac2b639fa8c944688787eba47ea1b420bfbe9c6e1e051a3d827d8b2b32b264d')
prior=read('work/risk_consensus219_SOURCE_REVIEW.json','cbf98df0568cdfb67260e84b8126c72cf3b981ceee94ade50a34074d8cb01b5b')
raw=plan['raw_source_hashes']
require(len(raw)==64 and raw==prior['raw_source_hashes'],'Original64 lineage map')
for name,expected in raw.items():
    require(original.get(name)==expected,'Source absent from original registered map')
    bind(name,expected)
meta=read(LOCAL/'META.json','3b46968f486e88fa170ac01f120dd5ef74ebd5a56243c6991c5a4fea506076f4')
require(len(meta['dates'])==len(meta['ends'])==len(meta['symbols'])==122,'Original event axes')
require(meta['dates']==sorted(set(meta['dates'])),'Distinct increasing dates')
require(meta['dates'][0]=='2016-08-17' and meta['dates'][-1]=='2026-08-31' and meta['ends'][-1]=='2026-09-24','Original date endpoints')
require(len(meta['etfs'])==len(set(meta['etfs']))==17,'Original ETF axis')
require(all(len(row)==49 and len(set(row))==49 for row in meta['symbols']),'Original candidate axes')
require(all(type(symbol) is str and symbol and symbol.strip()==symbol for row in meta['symbols'] for symbol in row),'Canonical security IDs')
for value in meta['dates']+meta['ends']:iso(value)
require(all(end>start for start,end in zip(meta['dates'],meta['ends'])),'Outcome end after decision')
cache=read(LOCAL/'CACHE_HASHES.json','e3b440f8e2f2669d535cb92d33a2302be7ceec4f54792ff4078a7d976356460b')
for name,expected in cache.items():bind(LOCAL/name,expected)
bind(LOCAL/'local_linear_ORDERS.json','c0e9286cdfae5e586a83ee0c412447a607ca21d1a5d6bbe47d40bd494b9d947e')
calendar=read(CALENDAR,CALENDAR_SHA)
for name,expected in calendar['provenance']['source_hashes'].items():bind(name,expected)
sessions=calendar['sessions']
require(type(sessions) is list and sessions and sessions==sorted(set(sessions)),'Complete unique calendar')
require(calendar['start']=='2008-01-01' and calendar['end']=='2035-12-31','Full143 calendar horizon')
for value in sessions:require(iso(value).weekday()<5,'Official calendar weekday')
positions={day:index for index,day in enumerate(sessions)}
windows=[]
risk_windows=[]
for day in meta['dates']:
    require(day in positions,'Decision not in pinned calendar')
    t=positions[day]
    require(t>=294 and t+1<len(sessions),'Calendar history/next-session coverage')
    if day!='2016-08-17':require(day[:7]!=sessions[t+1][:7],'Formal decision not month end')
    formation=sessions[t-294:t-41]
    require(len(formation)==253 and formation[-1]<day,'Exact causal formation date count')
    windows.append(dict(date=day,price_start=formation[0],first_return_date=formation[1],price_end=formation[-1]))
    risk=sessions[t-63:t+1]
    require(len(risk)==64 and risk[-1]==day,'Current63 risk clock')
    require(len(set(formation)|set(risk))==295,'Contiguous stock support union')
    risk_windows.append(dict(date=day,stock_price_start=risk[0],market_price_start=risk[0],first_return_date=risk[1],stock_price_end=risk[-1],market_price_end=risk[-1],stock_prices=64,market_prices=64,simple_returns_each=63))
require(windows==prior['calendar']['windows'],'Same253 calendar windows differ from219')
require(prior['source_window_matches199'] is True,'Prior199 provenance window statement')
lineage=prior['source_lineage_reference']
bind(lineage['path'],lineage['sha256'])
require(lineage['sha256']=='d12cdcfc032404e297f1d95a2753785175db9892b71c0e8b49f241470bfad911','Source lineage reference')
static_files=('work/monthly_idio_risk222_kernel.py','work/monthly_tail_moment221_kernel.py','work/monthly_risk_consensus219_kernel.py','outputs/monthly_local_linear70/study.py','outputs/monthly_core_timing70_research/core_adapter.py','outputs/monthly_path70_research/incumbent_adapter.py','outputs/monthly_daily_episodes/episodes.py')
for name in static_files:evidence[name]=bind(name)
# Read source text only, never import the source loader, kernel or strategy.
kernel=(ROOT/static_files[0]).read_text()
require('PRICE_COUNT, RISK_PRICE_COUNT, RISK_RETURNS, SKIP = 253, 64, 63, 42' in kernel,'Source kernel window constants')
for token in ("MARKET_SYMBOL = 'SPY'",'momentum = finite(p[-1]/p[0]-1.)','stock_mean = finite(math.fsum(stock_returns)/RISK_RETURNS)','market_mean = finite(math.fsum(market_returns)/RISK_RETURNS)','beta = finite(cross/market_ss)','residuals = [finite(b-finite(beta*a)) for a, b in zip(x, y)]','math.sqrt(sum_squares/62.)','raw_candidate=finite(momentum/idio_risk), raw_control=finite(momentum/total_risk)','if market_ss == 0.:','if total_risk == 0.:','if idio_risk == 0.:','descriptive_beta_fits=beta_fits','predictive_model_fits=0, new_model_fits=0'):
    require(token in kernel,'Fixed source kernel formula token: '+token)
core=(ROOT/'outputs/monthly_core_timing70_research/core_adapter.py').read_text()
for token in ('delta[changed] = blend * (cw[changed] * rank[changed] - core[changed])','original.update(baseline=baseline.copy(), core=core.copy(),','return path_adapter.score_and_choose('):
    require(token in core,'Original frozen core integration token')
# Verify SPY's stored adjusted-close provenance using manifests, not price rows.
loader=ROOT/'outputs/monthly_stability_monitor/inputs.py'
loader_sha=bind(loader)
require(raw.get(str(loader))==loader_sha,'Loader not in original64 source map')
loader_text=loader.read_text()
for token in ("('adj','Adj Close')","panel[k][entry['symbol']]=z[col].reindex(panel[k].index)","for s,cache in [('CHL','CHL'),('PTR','PTRCY')]"):
    require(token in loader_text,'Frozen adjusted-close load path changed')
market_entries=[]
for rel in ('outputs/four_factor_rebound/ETF_SOURCES.json','outputs/monthly_reconstruction_v2/ETF_MANIFEST.json','outputs/monthly_reconstruction_v2/foreign/ETF_MANIFEST.json'):
    rows=read(rel)
    for entry in rows:
        if entry['symbol']=='SPY':
            require(raw.get(entry['file'])==entry['sha256'],'Market raw pin absent from original64')
            bind(entry['file'],entry['sha256'])
            market_entries.append(dict(manifest=str(ROOT/rel),manifest_sha256=PINS[str(ROOT/rel)],path=entry['file'],sha256=entry['sha256'],column='Adj Close'))
require(len(market_entries)==1,'Expected one unambiguous frozen SPY manifest entry')
require(market_entries[0]['sha256']=='9fbc17731dfadfb74382d076a281dcc935f1971f74f8d08cfe860726b69edff5','Market original digest')
# Presence checks only: later loader branches may not override SPY.
for folder in ('monthly_published44_reconstruction','monthly_top35_extension','monthly_2016_extension'):
    require(not (ROOT/'outputs'/folder/'prices/SPY.csv').exists(),'Unexpected later SPY price override')
implementation=bind(Path(__file__))
evidence[str(Path(__file__).relative_to(ROOT))]=implementation
metadata=dict(events=122,width=49,etf_count=17,unique_candidate_symbols=len({s for row in meta['symbols'] for s in row}),first_query=meta['dates'][0],last_query=meta['dates'][-1],first_outcome=meta['dates'][0],last_outcome=meta['ends'][-1])
report=dict(schema='idio_risk222_source_review_v1',created_utc=datetime.now(timezone.utc).isoformat(),passed=True,
    context_sha256=CONTEXT_SHA,coverage_sha256=COVERAGE_SHA,
    scope='Streaming local-byte SHA checks and JSON/source-code metadata only. No imports of source strategy or loader, no numeric array decoding, no price/return/factor arithmetic. Binary cache bytes are hashed without decoding.',
    input_sha256=dict(sorted(PINS.items())),evidence_sha256=dict(sorted(evidence.items())),
    raw_source_hashes=raw,raw_source_count=len(raw),raw_source_map_matches_original=True,raw_source_hashes_verified=True,
    source_only=True,source_admission_metadata_only=True,metadata_axes_verified=True,calendar_clock_verified=True,source_window_matches199=True,integration_source_verified=True,current63_market_risk_window_checked=True,
    metadata=metadata,calendar=dict(path=str(CALENDAR),sha256=CALENDAR_SHA,session_count=len(sessions),first_session=sessions[0],last_session=sessions[-1],same_253_session_windows_as199=True,same_windows_as219=True,full_calendar_checked=True,windows=windows,risk_windows=risk_windows,current63_uses_full_original_calendar=True),
    market_source=dict(symbol='SPY',entries=market_entries,loader=str(loader),loader_sha256=loader_sha,adjusted_close_provenance_checked=True,later_override_absent=True,price_rows_read=False),
    source_lineage_reference=dict(lineage,scope='Frozen199 provenance reference only. Current raw bytes independently rehashed; no new price/identity/PIT reconstruction or actual feature-support claim.'),
    prior_source_review=dict(path=str(ROOT/'work/risk_consensus219_SOURCE_REVIEW.json'),sha256=PINS[str(ROOT/'work/risk_consensus219_SOURCE_REVIEW.json')],current_loader_executed=False,current_raw_bytes_reverified=True,scope='Prior metadata lineage and matching253-session windows; no reuse of old positive-risk support.'),
    integration=dict(core_adapter='outputs/monthly_core_timing70_research/core_adapter.py',delta='0.5*(cw*T-C)',baseline_core_pre_score_copies=True,original_support_passed=True,episode_calls=1,calm_and_initial='Host supplies NaN; producer computes all dates',live_observers_untouched=True,numerical_integration_recomputed=False,host_not_pinned_to_avoid_review_hash_cycle=True,host_requires_separate_final_code_review=True),
    numerical_definition=dict(cumulative_momentum=True,momentum_prices=253,stock_risk_prices=64,market_risk_prices=64,risk_returns_each=63,variance_denominator=62,current_window_centered_ols=True,residual_numerator=False,residuals_recentered=False,descriptive_beta_fits_are_counted=True,new_predictive_models=0,common_zero_risk_unavailable=True,negative_and_zero_momentum_not_filtered=True,no_clipping_or_epsilon=True,explicit_products_fsum_sqrt=True,arithmetic_source_text_only=True),
    limitations=['Current63 idiosyncratic risk was already implemented as idio63; denominator-only cumulative-M integration is a narrow comparison, not a new family. Exact exhaustive historical coverage is UNKNOWN.',
        'No new source-vintage or strict PIT certification; adjusted prices and historical membership are inherited retrospective proxies.',
        'Byte identity does not prove original publication clock, economic identity or predictive validity.',
        'This metadata review does not establish actual222 formation/stock/SPY price support, positive common risk availability, feature/rank/order/return correctness.',
        'Skip42 applies to cumulative momentum only; both paired risks use the original current63 risk clock. Degenerate shared support can change total-risk control ranks relative to incumbent.',
        'Final host code and synthetic/independent execution require their own pinned checks; this receipt does not execute the host.'],
    implementation=dict(path=str(Path(__file__)),sha256=implementation),
    actual_source_coverage_recomputed=False,actual_new_feature=False,actual_new_order=False,actual_new_returns=False,
    actual_new_features_computed=False,actual_new_orders_computed=False,actual_new_returns_computed=False,
    npz_decoded=False,original_sources_modified=False,observers_accessed=False,strict_pit=False)
with OUT.open('x') as stream:
    json.dump(report,stream,indent=2,sort_keys=True);stream.write('\n')
print(json.dumps(dict(path=str(OUT),sha256=sha(OUT),verified_files=len(PINS),raw_source_files=len(raw),calendar_sessions=len(sessions),metadata_events=len(windows),source_only=True,implementation_sha256=implementation)))
