"""Revalidate unchanged295-price coverage evidence; no new price arithmetic."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
CTX=ROOT/'work/monthly_adjacent_risk205_context.json'
CTX_SHA='60f0a5a6e1e8bd81f80272c5150c7a6baba8154a92a2c2b02e5a7281844d6b3c'
COV=ROOT/'work/after204_bipower_coverage.json'
COV_SHA='4eff269796a894a688cb3a5c16cc1d677376bac0956220c89b00e4bc6799f5a0'
PRIOR=ROOT/'work/median201_SOURCE_REVIEW.json'
PRIOR_SHA='4ca67a75465bb05f8b255272a185c436047c89fefe2193d939ec5ee9ae1e810d'
OUT=ROOT/'work/adjacent205_SOURCE_REVIEW.json'
CENTRAL=ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
def require(x,msg):
    if not x:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
    require(not OUT.exists() and not (ROOT/'outputs/monthly_adjacent_risk205').exists(),'Preregisration source review writeonce')
    pins={}
    def pin(path,digest=None):
        p=Path(path);p=(p if p.is_absolute() else ROOT/p).resolve()
        require(p.is_relative_to(ROOT) and p!=CENTRAL,'Nonlocal or mutable source')
        h=sha(p);require(digest is None or h==digest,'Changed source '+str(p));require(str(p) not in pins or pins[str(p)]==h,'Conflicting source');pins[str(p)]=h;return h
    for p,h in ((CTX,CTX_SHA),(COV,COV_SHA),(PRIOR,PRIOR_SHA)):pin(p,h)
    pin(__file__);ctx,cov,prior=read(CTX),read(COV),read(PRIOR)
    require(ctx['round']==205 and ctx['pre_registration_actual_new_features']==ctx['pre_registration_actual_orders']==ctx['pre_registration_actual_returns']==0,'Contextidentity/prospective')
    require(ctx['formation']['price_count']==295 and ctx['formation']['start_offset']==-294 and ctx['formation']['end_offset']==0 and ctx['formation']['return_count']==63 and ctx['formation']['adjacent_count']==62,'Sourceclock')
    require(prior['passed'] is True and prior['source_only'] is True and prior['actual_source_coverage_computed'] is True and prior['unresolved_blockers']==[],'Original source evidence')
    for k in ('actual_features_computed','actual_orders_computed','actual_returns_computed','actual_signal_returns_computed','actual_dispersion_computed','actual_ranks_computed','actual_deltas_computed'):
        require(prior[k] is False,'Prior source scope '+k)
    for mapping in (prior['source_sha256'],ctx['evidence_sha256'],cov['source_sha256']):
        for p,h in mapping.items():pin(p,h)
    local=ROOT/'outputs/monthly_local_linear70';pin(local/'PLAN.json');pin(local/'META.json')
    raw=read(local/'PLAN.json')['raw_source_hashes'];require(raw==prior['raw_source_hashes'] and len(raw)==64,'Originalraw64')
    for p,h in raw.items():pin(p,h)
    calendar_path=ROOT/'outputs/monthly_paired_target143/CALENDAR.json';pin(calendar_path);pin(calendar_path.parent/'PLAN.json')
    calendar=read(calendar_path);require(calendar['provenance']==read(calendar_path.parent/'PLAN.json')['calendar'],'Calendar source binding')
    for p,h in calendar['provenance']['source_hashes'].items():pin(p,h)
    cal=calendar['sessions'];require(cal==sorted(set(cal)),'Uniqueofficialcalendar')
    meta=read(local/'META.json');require(len(meta['dates'])==len(meta['symbols'])==122 and len(meta['etfs'])==17,'Original122x49')
    coverage=prior['source_window_coverage'];require(coverage['price_count']==295 and len(coverage['by_query'])==122,'Original sourcewindow')
    require(coverage['complete_positive_price_cells']+coverage['incomplete_price_cells']==5978 and coverage['field_counts']['observations']==5978*295,'Storedsourcecoveragecounts')
    for i,(day,symbols,row) in enumerate(zip(meta['dates'],meta['symbols'],coverage['by_query'])):
        t=cal.index(day);dates=cal[t-294:t+1]
        require(t>=294 and len(dates)==295 and len(symbols)==len(set(symbols))==49 and set(meta['etfs']).issubset(symbols),'Query domain')
        require(row['date']==day and row['start']==dates[0] and row['end']==dates[-1] and row['long_end']==dates[252] and row['risk_price_start']==dates[231] and row['risk_price_end']==dates[294],'Inheritedclock differs')
        require(row['source_sessions']==295 and row['risk_source_prices']==64 and row['risk_return_intervals']==63 and row['candidates']==49,'Inherited dimensions')
        require((i==0 and day=='2016-08-17') or cal[t+1][:7]!=day[:7],'Query is not monthend')
    state=read(CENTRAL);require(state['current']=='monthly_aux_rank204' and state['registered_queue']==[] and len(state['completed'])==202 and sum(x['trials'] for x in state['completed'])==1123,'Review before205registration')
    result=dict(schema_version=1,created_utc=datetime.now(timezone.utc).isoformat(),passed=True,source_only=True,conditional_research_sources_ready=True,unresolved_blockers=[],context_sha256=CTX_SHA,coverage_sha256=COV_SHA,review_code_sha256=sha(__file__),prior_source_review_sha256=PRIOR_SHA,raw_source_hashes=raw,source_sha256=pins,raw_source_parity=prior['raw_source_parity'],axes=prior['axes'],source_window_coverage=coverage,source_lineage=prior['source_lineage'],alias_rules=prior['alias_rules'],normalization_facts=prior['normalization_facts'],source_coverage_reused=True,unchanged_source_bytes_reverified=True,actual_source_coverage_computed=False,actual_features_computed=False,actual_signal_returns_computed=False,actual_adjacent_products_computed=False,actual_squared_returns_computed=False,actual_risk_estimates_computed=False,actual_dispersion_computed=False,actual_ranks_computed=False,actual_deltas_computed=False,actual_orders_computed=False,actual_returns_computed=False,actual_portfolio_returns_computed=False,new_model_fits=0,production_files_modified=False,strategy_registered_by_reviewer=False,observers_accessed=False,strict_pit=False,strict_pit_vintage=False,strict_pit_membership=False,observed_central_state=dict(completed_rounds=202,cumulative_trials=1123,registered_queue=[]),limitations=['Fresh hash verification reuses the exact prior295-price coverage; no new price-cell arithmetic was performed.','Same-sourcecoverage does not prove positivity or finite arithmetic of adjacent/RMS denominators; registered preparation checks those.','HistoricalproxyTop32/theme17 and retrospectivevendorversions remain conditionalresearch,notcertifiedstrictPIT or pristineOOS.'])
    for p,h in pins.items():require(sha(p)==h,'Changed during source review')
    with OUT.open('x') as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps(dict(passed=True,path=str(OUT),sha256=sha(OUT),source_pins=len(pins),inherited_complete_price_cells=coverage['complete_positive_price_cells'],inherited_incomplete_price_cells=coverage['incomplete_price_cells'],new_source_coverage_computed=False)))
if __name__=='__main__':main()
