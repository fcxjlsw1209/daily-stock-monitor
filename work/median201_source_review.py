"""Round201 original adjusted-price source audit; no factors or strategy replay."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,importlib.util,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
O=ROOT/'outputs'
CONTEXT=ROOT/'work/monthly_median_risk201_context.json'
CONTEXT_SHA='4e937ed4474675b64069b410ef790697474ea2bfe03585e0f8f97d30afec9701'
COVERAGE=ROOT/'work/after200_median_risk_coverage.json'
PRIOR=ROOT/'work/cmf200_SOURCE_REVIEW.json'
PRIOR_SHA='9742ef91bba7617a95fc738911d167e6abb82b94d95a73f59455253bca815ee7'
OUT=ROOT/'work/median201_SOURCE_REVIEW.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def require(ok,msg):
    if not ok:raise ValueError(msg)
def counts(v):
    x=np.asarray(v,dtype=float);f=np.isfinite(x)
    return dict(observations=int(x.size),finite_positive=int((f&(x>0)).sum()),zero=int((f&(x==0)).sum()),negative=int((f&(x<0)).sum()),nan=int(np.isnan(x).sum()),infinite=int(np.isinf(x).sum()))
def window(c,d):
    t=c.index(d);require(t>=294,'Need295officialprices')
    out=c[t-294:t+1];require(len(out)==295 and out[-1]==d,'Sourcewindow')
    return out

def main():
    require(not OUT.exists(),'Write-once source receipt')
    require(counts([1.,0.,-1.,np.nan,np.inf,-np.inf])==dict(observations=6,finite_positive=1,zero=1,negative=1,nan=1,infinite=2),'Synthetic count partition')
    require(window(list(range(400)),350)==list(range(56,351)),'Synthetic295window')
    try:window(list(range(400)),293)
    except ValueError:pass
    else:raise ValueError('Shortwindowmustfail')
    pins={}
    def pin(p,expected=None):
        p=Path(p).resolve();require(p.is_relative_to(ROOT),'Nonlocal source')
        h=sha(p);require(expected is None or h==expected,'Changed source '+str(p));require(str(p) not in pins or pins[str(p)]==h,'Conflictingpin');pins[str(p)]=h;return h
    pin(__file__);pin(CONTEXT,CONTEXT_SHA);pin(PRIOR,PRIOR_SHA);pin(COVERAGE)
    ctx,prior,coverage=read(CONTEXT),read(PRIOR),read(COVERAGE)
    require(ctx['round']==201 and ctx['name']=='monthly_median_risk201','Contextidentity')
    require(ctx['pre_registration_actual_new_features']==ctx['pre_registration_actual_orders']==ctx['pre_registration_actual_returns']==0,'Prospectiveonly')
    require(prior['passed'] is True and prior['raw_source_parity']['exact_source_map_equal'],'Previous source certification')
    require(coverage['status']=='METADATA_ONLY_COVERAGE_NOT_REGISTRATION' and coverage['new_features_computed']==0 and coverage['new_orders_computed']==0 and coverage['new_returns_computed']==0,'Coverage evidence')
    for p,h in prior['source_sha256'].items():pin(p,h)
    for p,h in ctx['evidence_sha256'].items():pin(ROOT/p,h)
    for p,h in coverage.get('evidence_sha256',{}).items():pin(Path(p) if Path(p).is_absolute() else ROOT/p,h)
    local=O/'monthly_local_linear70';pin(local/'PLAN.json');pin(local/'META.json')
    raw=read(local/'PLAN.json')['raw_source_hashes'];require(raw==prior['raw_source_hashes'] and len(raw)==64,'Original64closure')
    for p,h in raw.items():pin(p,h)
    meta=read(local/'META.json');cp=O/'monthly_paired_target143/CALENDAR.json';pin(cp);pin(cp.parent/'PLAN.json')
    calrec=read(cp);cal=calrec['sessions'];require(cal==sorted(set(cal)),'Officialcalendar')
    require(calrec['provenance']==read(cp.parent/'PLAN.json')['calendar'],'Calendarprovenance')
    for p,h in calrec['provenance']['source_hashes'].items():pin(p,h)
    require(len(meta['dates'])==len(meta['symbols'])==122 and len(meta['etfs'])==len(set(meta['etfs']))==17,'Queryaxes')
    loader=O/'monthly_stability_monitor/inputs.py';require(str(loader) in raw,'Frozenloaderbinding')
    spec=importlib.util.spec_from_file_location('median201_original_source_loader',loader);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    panel,info=mod.load_panel(meta['etfs']);used=dict(info['source_hashes']);used[str(loader)]=sha(loader);require(used==raw,'Originalsourcemapchanged')
    a=panel['adj'];require(a.index.is_unique and a.index.is_monotonic_increasing and a.columns.is_unique,'Priceaxes')
    total={k:0 for k in counts([])};complete=0;events=[]
    for i,(day,names) in enumerate(zip(meta['dates'],meta['symbols'])):
        require(len(names)==len(set(names))==49 and set(meta['etfs']).issubset(names),'49candidateidentities')
        t=cal.index(day);require((i==0 and day=='2016-08-17') or cal[t+1][:7]!=day[:7],'Monthendquery')
        dates=window(cal,day);idx=pd.to_datetime(dates);require(a.loc[idx[0]:idx[-1]].index.equals(idx),'Exactofficial295axis')
        require(set(names).issubset(a.columns),'Missingcandidatecolumn')
        block=a.loc[idx,names].to_numpy();q=counts(block)
        good=np.all(np.isfinite(block)&(block>0),axis=0);n=int(good.sum());complete+=n
        for k,v in q.items():total[k]+=v
        events.append(dict(date=day,start=dates[0],end=dates[-1],long_end=dates[252],risk_price_start=dates[231],risk_price_end=dates[294],source_sessions=295,risk_source_prices=64,risk_return_intervals=63,candidates=49,complete_positive_price_cells=n,incomplete_price_symbols=[s for s,g in zip(names,good) if not g],counts=q))
    require(total['observations']==122*49*295,'Sourceobservationaccounting')
    central=read(O/'monthly_target70_research/CONTINUATION_STATUS.json')
    require(central['current']=='monthly_cmf_state200' and central['registered_queue']==[] and len(central['completed'])==198 and sum(r['trials'] for r in central['completed'])==1111,'Reviewmustprecede201registration')
    result=dict(schema_version=1,passed=True,source_only=True,conditional_research_sources_ready=True,unresolved_blockers=[],created_utc=datetime.now(timezone.utc).isoformat(),context_sha256=CONTEXT_SHA,review_code_sha256=sha(__file__),coverage_sha256=sha(COVERAGE),raw_source_hashes=raw,source_sha256=pins,raw_source_parity=dict(passed=True,exact_source_map_equal=True,count=64,current_loader_sha256=sha(loader)),axes=dict(monthly_queries=122,symbols_per_query=49,etf_count=17,first_query=meta['dates'][0],last_query=meta['dates'][-1]),source_window_coverage=dict(price_count=295,complete_positive_price_cells=complete,incomplete_price_cells=122*49-complete,field_counts=total,by_query=events),source_lineage=prior['source_lineage'],alias_rules=prior['alias_rules'],normalization_facts=prior['normalization_facts'],integration_source_verification=dict(numerical_integration_verified=False,core='Only noncalm original halfcore; current63risk and skip42 long numerator; calm and initialpartial unchanged'),strict_pit=False,strict_pit_vintage=False,strict_pit_membership=False,actual_source_coverage_computed=True,actual_features_computed=False,actual_orders_computed=False,actual_returns_computed=False,actual_portfolio_returns_computed=False,actual_signal_returns_computed=False,actual_dispersion_computed=False,actual_ranks_computed=False,actual_deltas_computed=False,new_model_fits=0,production_files_modified=False,strategy_registered_by_reviewer=False,observers_accessed=False,observed_central_state=dict(completed_rounds=198,cumulative_trials=1111,registered_queue=[]),limitations=['Sourcecoverage does not prove positive MAD denominators or finite arithmetic; those belong to registered preparation.','Historical proxyTop32+theme17 and retrospective vendor/action versions are conditional research, not certified strictPIT or pristineOOS.','Source audit imports only frozen original panel loader; no scoring/model/strategy builder and no NPZ feature arrays.'],synthetic_source_checks_passed=3)
    for p,h in pins.items():require(sha(p)==h,'Sourcechangedmidreview')
    with OUT.open('x') as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps(dict(passed=True,path=str(OUT),sha256=sha(OUT),source_pins=len(pins),complete_price_cells=complete,incomplete_price_cells=122*49-complete)))
if __name__=='__main__':main()
