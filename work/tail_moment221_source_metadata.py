"""Round221 write-once SHA/source-metadata review. Use the offline wrapper."""
from pathlib import Path
from datetime import datetime, timezone, date
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'work/tail_moment221_SOURCE_REVIEW.json'
CONTEXT=ROOT/'work/monthly_tail_moment221_context.json'
CONTEXT_SHA='191d495345708e5609de41907ce672d548b0885adac3ef1251a21531d50bc316'
COVERAGE=ROOT/'work/after220_tail_moment_coverage.json'
COVERAGE_SHA='f290095f3ead5abc79bf5d4e08a23d466ae7d385d23b0fafff92b04c60e5845a'
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
require(context['round']==221 and context['name']=='monthly_tail_moment221','Wrong round context')
require(context['formation']['start_offset']==-294 and context['formation']['end_offset']==-42 and context['formation']['return_intervals']==252,'Formation contract')
require(context['integration']['blend']==.5,'Fixed half-core contract')
require(context['formation']['new_predictive_model_fits']==0 and context['formation']['parameter_search'] is False,'No new fits/search contract')
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
for day in meta['dates']:
    require(day in positions,'Decision not in pinned calendar')
    t=positions[day]
    require(t>=294 and t+1<len(sessions),'Calendar history/next-session coverage')
    if day!='2016-08-17':require(day[:7]!=sessions[t+1][:7],'Formal decision not month end')
    formation=sessions[t-294:t-41]
    require(len(formation)==253 and formation[-1]<day,'Exact causal formation date count')
    windows.append(dict(date=day,price_start=formation[0],first_return_date=formation[1],price_end=formation[-1]))
require(windows==prior['calendar']['windows'],'Same253 calendar windows differ from219')
require(prior['source_window_matches199'] is True,'Prior199 provenance window statement')
lineage=prior['source_lineage_reference']
bind(lineage['path'],lineage['sha256'])
require(lineage['sha256']=='d12cdcfc032404e297f1d95a2753785175db9892b71c0e8b49f241470bfad911','Source lineage reference')
static_files=('work/monthly_tail_moment221_kernel.py','work/monthly_risk_consensus219_kernel.py','outputs/monthly_local_linear70/study.py','outputs/monthly_core_timing70_research/core_adapter.py','outputs/monthly_path70_research/incumbent_adapter.py','outputs/monthly_daily_episodes/episodes.py')
for name in static_files:evidence[name]=bind(name)
# Read source text only, never import the source loader, kernel or strategy.
kernel=(ROOT/static_files[0]).read_text()
require('PRICE_COUNT, SKIP, WINDOW_RETURNS = 253, 42, 252' in kernel,'Source kernel window constants')
for token in ('momentum=finite(p[-1]/p[0]-1.)','squares=[finite(x*x) for x in losses]','cubes=[finite((x*x)*x) for x in losses]','d2=finite(math.sqrt(s2/WINDOW_RETURNS))','d3=finite(np.cbrt(s3/WINDOW_RETURNS))','raw_candidate=finite(momentum/d3)','raw_control=finite(momentum/d2)'):
    require(token in kernel,'Fixed source kernel formula token')
core=(ROOT/'outputs/monthly_core_timing70_research/core_adapter.py').read_text()
for token in ('delta[changed] = blend * (cw[changed] * rank[changed] - core[changed])','original.update(baseline=baseline.copy(), core=core.copy(),','return path_adapter.score_and_choose('):
    require(token in core,'Original frozen core integration token')
implementation=bind(Path(__file__))
evidence[str(Path(__file__).relative_to(ROOT))]=implementation
metadata=dict(events=122,width=49,etf_count=17,unique_candidate_symbols=len({s for row in meta['symbols'] for s in row}),first_query=meta['dates'][0],last_query=meta['dates'][-1],first_outcome=meta['dates'][0],last_outcome=meta['ends'][-1])
report=dict(schema='tail_moment221_source_review_v1',created_utc=datetime.now(timezone.utc).isoformat(),passed=True,
    context_sha256=CONTEXT_SHA,coverage_sha256=COVERAGE_SHA,
    scope='Streaming local-byte SHA checks and JSON/source-code metadata only. No imports of source strategy or loader, no numeric array decoding, no price/return/factor arithmetic. Binary cache bytes are hashed without decoding.',
    input_sha256=dict(sorted(PINS.items())),evidence_sha256=dict(sorted(evidence.items())),
    raw_source_hashes=raw,raw_source_count=len(raw),raw_source_map_matches_original=True,raw_source_hashes_verified=True,
    source_only=True,source_admission_metadata_only=True,metadata_axes_verified=True,calendar_clock_verified=True,source_window_matches199=True,integration_source_verified=True,
    metadata=metadata,calendar=dict(path=str(CALENDAR),sha256=CALENDAR_SHA,session_count=len(sessions),first_session=sessions[0],last_session=sessions[-1],same_253_session_windows_as199=True,same_windows_as219=True,full_calendar_checked=True,windows=windows),
    source_lineage_reference=dict(lineage,scope='Frozen199 provenance reference only. Current raw bytes independently rehashed; no new price/identity/PIT reconstruction or actual feature-support claim.'),
    prior_source_review=dict(path=str(ROOT/'work/risk_consensus219_SOURCE_REVIEW.json'),sha256=PINS[str(ROOT/'work/risk_consensus219_SOURCE_REVIEW.json')],current_loader_executed=False,current_raw_bytes_reverified=True,scope='Prior metadata lineage and matching253-session windows; no reuse of old positive-risk support.'),
    integration=dict(core_adapter='outputs/monthly_core_timing70_research/core_adapter.py',delta='0.5*(cw*T-C)',baseline_core_pre_score_copies=True,original_support_passed=True,episode_calls=1,calm_and_initial='Host supplies NaN; producer computes all dates',live_observers_untouched=True,numerical_integration_recomputed=False,host_not_pinned_to_avoid_review_hash_cycle=True,host_requires_separate_final_code_review=True),
    numerical_definition=dict(cumulative_momentum=True,loss_orders=[3,2],all_days_denominator=252,source_prices=253,common_no_negative_returns_unavailable=True,negative_and_zero_momentum_not_filtered=True,no_negative_only_mean=True,no_clipping_or_epsilon=True,deterministic_explicit_products_fsum_sqrt_cbrt=True,arithmetic_source_text_only=True),
    limitations=['Risk-adjusted/downside momentum is an explored family; exact exhaustive historical coverage is UNKNOWN.',
        'No new source-vintage or strict PIT certification; adjusted prices and historical membership are inherited retrospective proxies.',
        'Byte identity does not prove original publication clock, economic identity or predictive validity.',
        'This metadata review does not establish actual221 price support, downside-positive common availability, feature/rank/order/return correctness.',
        'The unchanged half-core keeps current63 risk and all frozen auxiliary clocks; skip42 applies to the new numerator and denominator.',
        'Final host code and synthetic/independent execution require their own pinned checks; this receipt does not execute the host.'],
    implementation=dict(path=str(Path(__file__)),sha256=implementation),
    actual_source_coverage_recomputed=False,actual_new_feature=False,actual_new_order=False,actual_new_returns=False,
    actual_new_features_computed=False,actual_new_orders_computed=False,actual_new_returns_computed=False,
    npz_decoded=False,original_sources_modified=False,observers_accessed=False,strict_pit=False)
with OUT.open('x') as stream:
    json.dump(report,stream,indent=2,sort_keys=True);stream.write('\n')
print(json.dumps(dict(path=str(OUT),sha256=sha(OUT),verified_files=len(PINS),raw_source_files=len(raw),calendar_sessions=len(sessions),metadata_events=len(windows),source_only=True,implementation_sha256=implementation)))
