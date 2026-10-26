"""Round223 final source metadata receipt; never decode arrays or prices."""
from pathlib import Path
from datetime import datetime,timezone,date
import hashlib,json,argparse
R=Path(__file__).resolve().parents[1]
CONTEXT=R/'work/monthly_risk_reliability223_context.json'
CONTEXT_SHA='38c0dd8b6771f43e2b58ec46b985f2f3e5fb6a12cf209a5ba179768746860a70'
COVERAGE=R/'work/after222_risk_forecast_coverage.json'
COVERAGE_SHA='f5a494616cbe54aa2d71184c4b2c4e33edc00e5a4e8ddc37340306d2d6561213'
OUT=R/'work/risk_reliability223_SOURCE_REVIEW.json'
PINS={}
def require(x,m):
    if not x:raise ValueError(m)
def local(p):
    p=Path(p);p=p if p.is_absolute() else R/p
    require(p.is_relative_to(R) and p.resolve().is_relative_to(R) and 'CloudStorage' not in str(p),'Nonlocal path');return p
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def bind(p,h=None):
    p=local(p);v=sha(p);require(h is None or v==h,'Source digest differs: '+str(p));require(str(p) not in PINS or PINS[str(p)]==v,'Source changed during review');PINS[str(p)]=v;return v
def read(p,h=None):bind(p,h);return json.loads(local(p).read_text())
parser=argparse.ArgumentParser();parser.add_argument('--context-sha256',required=True);args=parser.parse_args()
require(args.context_sha256==CONTEXT_SHA and not OUT.exists(),'Final context or write-once receipt')
context=read(CONTEXT,CONTEXT_SHA);coverage=read(COVERAGE,COVERAGE_SHA)
require(context['round']==223 and context['name']=='monthly_risk_reliability223','Final223 identity')
require(context['formation']['history_months']==60 and context['formation']['min_months']==18 and context['formation']['forecast_returns']==63,'Fixed history/maturity')
require(context['integration']['budget']==.01 and context['arms']==['incumbent_local67','risk_forecast_reliability','previous_month_reliability_control'],'Fixed paired auxiliary experiment')
require(context['identity_and_source']['strict_pit'] is False,'Conditional source limitation')
evidence={}
for p,h in context['evidence_sha256'].items():evidence[p]=bind(p,h)
for p,h in coverage['input_sha256'].items():bind(p,h)
for k in ('new_features_computed','new_orders_computed','new_returns_computed','new_predictive_models_fit','new_trials_registered'):require(type(coverage[k]) is int and coverage[k]==0,'Coverage computation boundary')
require(coverage['npz_decoded'] is False and coverage['family_novelty'] is False,'Coverage scope')
plan=read('outputs/monthly_local_linear70/PLAN.json','0016175aee1df43b734010d80bd543b881c57036a7f8c6ab73428340203afc96')
original=read('outputs/monthly_local_linear70/INPUT_HASHES.json','5ac2b639fa8c944688787eba47ea1b420bfbe9c6e1e051a3d827d8b2b32b264d')
raw=plan['raw_source_hashes'];require(raw==coverage['raw_source_hashes'] and len(raw)==64,'Original64 source map')
for p,h in raw.items():require(original.get(p)==h,'Missing original source');bind(p,h)
meta=read('outputs/monthly_local_linear70/META.json','3b46968f486e88fa170ac01f120dd5ef74ebd5a56243c6991c5a4fea506076f4')
require(len(meta['dates'])==len(meta['ends'])==len(meta['symbols'])==122 and all(len(r)==len(set(r))==49 for r in meta['symbols']),'Original monthly axes')
require(meta['dates']==sorted(set(meta['dates'])) and meta['dates'][0]=='2016-08-17' and meta['dates'][-1]=='2026-08-31' and meta['ends'][-1]=='2026-09-24','Original temporal axis')
require(len(meta['etfs'])==len(set(meta['etfs']))==17,'Original ETF axis')
cache=read('outputs/monthly_local_linear70/CACHE_HASHES.json','e3b440f8e2f2669d535cb92d33a2302be7ceec4f54792ff4078a7d976356460b')
for p,h in cache.items():bind(R/'outputs/monthly_local_linear70'/p,h)
bind('outputs/monthly_local_linear70/local_linear_ORDERS.json','c0e9286cdfae5e586a83ee0c412447a607ca21d1a5d6bbe47d40bd494b9d947e')
cal=read('outputs/monthly_paired_target143/CALENDAR.json','8ee4da69fba84db1edee37a7e0f5af5d3ab16c6257e5d37cddbd531f01404622')
for p,h in cal['provenance']['source_hashes'].items():bind(p,h)
ss=cal['sessions'];require(ss==sorted(set(ss)) and len(ss)==7039 and cal['start']=='2008-01-01' and cal['end']=='2035-12-31','Full original calendar')
for d in ss:require(date.fromisoformat(d).isoformat()==d,'Canonical calendar')
pos={d:i for i,d in enumerate(ss)};months={}
for d in ss:months.setdefault(d[:7],[]).append(d)
queries=[]
for d in meta['dates']:
    require(d in pos,'Query absent calendar');t=pos[d]
    require(t>=294 and t+1<len(ss) and (d=='2016-08-17' or d[:7]!=ss[t+1][:7]),'Month-end and original core clock')
    formation=ss[t-294:t-41];require(len(formation)==253,'Frozen core skip42 clock')
    queries.append(dict(date=d,year=int(d[:4]),annual_cutoff=str(int(d[:4])-1)+'-12-31',core_price_start=formation[0],core_price_end=formation[-1],first_partial=d=='2016-08-17'))
annual=[]
for y in sorted({int(d[:4]) for d in meta['dates']}):
    pairs=[]
    for hy in range(y-5,y):
        for hm in range(1,13):
            month=f'{hy:04d}-{hm:02d}';ds=months[month];first,last=pos[ds[0]],pos[ds[-1]];priorend=ss[first-1];prev=months[priorend[:7]];prevfirst=pos[prev[0]]
            require(first>=64 and prevfirst>=1 and last+1<len(ss),'Complete source-window calendar bounds')
            require(ss[first-1][:7]!=month and ss[last+1][:7]!=month and ss[prevfirst-1][:7]!=priorend[:7],'Verified full natural months')
            require(ds[-1]<=f'{y-1:04d}-12-31' and priorend<ds[0],'Annual maturity')
            pairs.append(dict(realized_month=month,forecast_date=priorend,forecast63_price_start=ss[first-64],forecast63_price_end=priorend,forecast63_prices=64,previous_month=priorend[:7],previous_month_price_start=ss[prevfirst-1],previous_month_price_end=priorend,previous_month_returns=len(prev),realized_price_start=priorend,realized_first_return_date=ds[0],realized_price_end=ds[-1],realized_returns=len(ds)))
    require(len(pairs)==60,'Exactly60 calendar histories')
    annual.append(dict(query_year=y,cutoff=f'{y-1:04d}-12-31',first_realized_month=pairs[0]['realized_month'],last_realized_month=pairs[-1]['realized_month'],windows=pairs))
require(annual==coverage['calendar']['annual_windows'],'Independent metadata source windows differ')
ids=sorted({s for row in meta['symbols'] for s in row})
require(ids==[r['security_id'] for r in coverage['identity_policy']['records']] and len(ids)==90,'Exact canonical identity review axis')
require(coverage['identity_policy']['requires_169_identity_metadata'] is False and coverage['identity_policy']['known_explicit_quarantine_overlap']==[],'Identity scope changed')
require(context['identity_and_source']['foreign_no_base_issuer_key']==coverage['metadata']['missing_base_issuer_non_etfs'],'Foreign identity limits mismatch')
static=['work/monthly_risk_reliability223_kernel.py','outputs/monthly_path70_research/incumbent_adapter.py','outputs/monthly_local_linear70/study.py','outputs/monthly_daily_episodes/episodes.py','outputs/monthly_stability_monitor/inputs.py']
for p in static:evidence[p]=bind(p)
k=(R/static[0]).read_text()
for token in ('HISTORY_MONTHS, MIN_MONTHS, FORECAST_RETURNS = 60, 18, 63','returns = [finite(prices[i+1]/prices[i]-1.)','total = finite(math.fsum(_square(r) for r in returns))','result = finite(total/len(returns))','denominator = finite(realized+forecast)','difference = finite(realized-forecast)','ratio = finite(difference/denominator)','if d_forecast == 0. or d_prior == 0.:','return -result'):
    require(token in k,'Fixed source arithmetic declaration: '+token)
a=(R/static[1]).read_text()
for token in ('support = (baseline-core)/(1-cw)','delta[changed] = budget*(factor[changed]-support[changed])','def score_and_choose('):require(token in a,'Frozen auxiliary integration contract')
loader=(R/static[4]).read_text()
for token in ('panel[k][entry[\'symbol\']]=z[col].reindex(panel[k].index)','gap=a.isna()&(a.index<first)','panel[\'adj\'].loc[gap,s]=z.loc[gap,\'Adj Close\']*ratio',"actual=lambda s:{'META':'research_equity_FB','BRK-B':'BRK.B'}.get(s,s)"):
    require(token in loader,'Frozen source-loader continuity/splice contract')
implementation=bind(Path(__file__));evidence[str(Path(__file__).relative_to(R))]=implementation
report=dict(schema='risk_reliability223_source_review_v1',created_utc=datetime.now(timezone.utc).isoformat(),passed=True,context_sha256=CONTEXT_SHA,coverage_sha256=COVERAGE_SHA,scope='Local byte digests, JSON identity/date metadata and static code only; no strategy imports, price rows, numeric arrays, actual monthly support, factors or portfolios.',input_sha256=dict(sorted(PINS.items())),evidence_sha256=dict(sorted(evidence.items())),raw_source_hashes=raw,raw_source_count=64,source_admission_metadata_only=True,metadata_axes_verified=True,calendar_clock_verified=True,raw_source_hashes_verified=True,raw_source_map_matches_original=True,integration_source_verified=True,annual_cutoff_calendar_verified=True,canonical_identity_scope_reviewed=True,metadata=dict(events=122,width=49,unique_canonical_ids=90,etf_count=17,first_query=meta['dates'][0],last_query=meta['dates'][-1],last_outcome=meta['ends'][-1]),calendar=dict(path='outputs/monthly_paired_target143/CALENDAR.json',sha256=PINS[str(R/'outputs/monthly_paired_target143/CALENDAR.json')],session_count=len(ss),queries=queries,annual_windows=annual,calendar_windows_only_not_actual_source_support=True),identity=coverage['identity_policy'],integration=dict(function='Frozen path_adapter.score_and_choose',budget=.01,formula='.01*(target_rank-original_support)',core_unchanged=True,initial_partial_targets_missing=True,all_states=True,episode_calls=1,no_rerank=True,static_source_verified=True,numerical_integration_recomputed=False,host_not_pinned_to_avoid_cycle=True,host_requires_separate_final_review=True),limitations=context['limitations']+['A same canonical source column is the declared research identity; no permanent-issuer or full ticker-reuse certificate is asserted.','Calendar maturity and byte integrity do not establish actual63-price/monthly support or18 common pairs; preparation must verify these after registration.','Static new-kernel review is not numerical execution; independent oracle/tests/host have separate admission gates.'],actual_new_features_computed=False,actual_new_orders_computed=False,actual_new_returns_computed=False,npz_decoded=False,actual_source_coverage_recomputed=False,original_sources_modified=False,observers_accessed=False,strict_pit=False,implementation=dict(path=str(Path(__file__)),sha256=implementation))
with OUT.open('x') as f:json.dump(report,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps(dict(path=str(OUT),sha256=sha(OUT),implementation_sha256=implementation,files=len(PINS),annual_years=len(annual),source_only=True)))
