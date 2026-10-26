"""Prospective current64-price/63-volume source validation; no derived downflags/features."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import hashlib,json,math,numbers,importlib.util
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
CTX=ROOT/'work/monthly_down_volume207_context.json'
CTX_SHA='223ac747a847b9d908cea27bf7596f544c783d0247ce56c255b24029e0712516'
PRIOR=ROOT/'work/volume195_SOURCE_REVIEW.json'
OUT=ROOT/'work/downvolume207_SOURCE_REVIEW.json'
CENTRAL=ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def require(ok,msg):
    if not ok:raise ValueError(msg)
def quality(values):
    counts=dict(observations=0,finite_positive=0,zero=0,negative=0,nan=0,infinite=0,nonnumeric_or_boolean=0)
    for v in values:
        counts['observations']+=1
        if isinstance(v,(bool,np.bool_)) or not isinstance(v,numbers.Real):counts['nonnumeric_or_boolean']+=1;continue
        v=float(v)
        if math.isnan(v):counts['nan']+=1
        elif not math.isfinite(v):counts['infinite']+=1
        elif v>0:counts['finite_positive']+=1
        elif v==0:counts['zero']+=1
        else:counts['negative']+=1
    require(sum(v for k,v in counts.items() if k!='observations')==counts['observations'],'Quality partition')
    return counts
def window(cal,day,volume=False):
    t=cal.index(day);offset=62 if volume else 63;require(t>=63,'Insufficientofficialsessions')
    return cal[t-offset:t+1]
def main():
    require(not OUT.exists() and not (ROOT/'outputs/monthly_down_volume207').exists(),'Writeonce pre-registration review')
    require(quality([1.,0.,-1.,np.nan,np.inf,True,'1'])==dict(observations=7,finite_positive=1,zero=1,negative=1,nan=1,infinite=1,nonnumeric_or_boolean=2),'Qualitysynthetic')
    require(window(list(range(500)),400)==list(range(337,401)) and window(list(range(500)),400,True)==list(range(338,401)),'Windowsynthetic')
    pins={}
    def pin(path,digest=None):
        p=Path(path);p=(p if p.is_absolute() else ROOT/p).resolve();require(p.is_relative_to(ROOT) and p!=CENTRAL,'Nonlocal/mutable source')
        h=sha(p);require(digest is None or h==digest,'Changedsource '+str(p));require(str(p) not in pins or pins[str(p)]==h,'Pinconflict');pins[str(p)]=h
    pin(__file__);pin(CTX,CTX_SHA)
    ctx=read(CTX);require(ctx['round']==207 and ctx['pre_registration_actual_new_features']==ctx['pre_registration_actual_orders']==ctx['pre_registration_actual_returns']==0,'Prospectivecontext')
    for p,h in ctx['evidence_sha256'].items():pin(p,h)
    prior=read(PRIOR);require(prior['passed'] is True and prior['actual_features_computed'] is False,'Priorvolume evidence')
    for p,h in prior['source_sha256'].items():pin(p,h)
    local=ROOT/'outputs/monthly_local_linear70';pin(local/'PLAN.json');pin(local/'META.json')
    raw=read(local/'PLAN.json')['raw_source_hashes'];require(raw==prior['raw_source_hashes'] and len(raw)==64,'Raw64lineage')
    for p,h in raw.items():pin(p,h)
    cp=ROOT/'outputs/monthly_paired_target143/CALENDAR.json';pin(cp);pin(cp.parent/'PLAN.json');calrec=read(cp);cal=calrec['sessions']
    require(cal==sorted(set(cal)) and calrec['provenance']==read(cp.parent/'PLAN.json')['calendar'],'Calendarprovenance')
    for p,h in calrec['provenance']['source_hashes'].items():pin(p,h)
    meta=read(local/'META.json');require(len(meta['dates'])==len(meta['symbols'])==122 and len(meta['etfs'])==17,'Originalaxes')
    loader=ROOT/'outputs/monthly_stability_monitor/inputs.py';require(str(loader) in raw,'Frozeninputloader')
    spec=importlib.util.spec_from_file_location('downvolume207_inputloader',loader);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    panel,info=mod.load_panel(meta['etfs']);actual=dict(info['source_hashes']);actual[str(loader)]=sha(loader);require(actual==raw,'Loadedrawsourceparity')
    adj,vol=panel['adj'],panel['volume'];require(adj.index.equals(vol.index) and adj.columns.equals(vol.columns),'Alignedpricevolume')
    for f in (adj,vol):require(f.index.is_unique and f.index.is_monotonic_increasing and f.columns.is_unique,'Sourceaxes')
    events=[];complete=0;reasons=Counter()
    for i,(day,symbols) in enumerate(zip(meta['dates'],meta['symbols'])):
        require(len(symbols)==len(set(symbols))==49 and set(meta['etfs']).issubset(symbols),'Original49')
        t=cal.index(day);require((i==0 and day=='2016-08-17') or cal[t+1][:7]!=day[:7],'Monthend')
        pdays,vdays=window(cal,day),window(cal,day,True);pi,vi=pd.to_datetime(pdays),pd.to_datetime(vdays)
        pmiss=pi.difference(adj.index).strftime('%Y-%m-%d').tolist();vmiss=vi.difference(vol.index).strftime('%Y-%m-%d').tolist()
        cells=[];good=0
        for s in symbols:
            pq=quality(adj.loc[pi,s].to_numpy()) if s in adj.columns and not pmiss else None
            vq=quality(vol.loc[vi,s].to_numpy()) if s in vol.columns and not vmiss else None
            ok=pq is not None and vq is not None and pq['finite_positive']==64 and vq['finite_positive']==63
            reason='available' if ok else 'invalid_or_missing_source';good+=int(ok);reasons[reason]+=1
            cells.append(dict(symbol=s,available=ok,reason=reason,price_quality=pq,volume_quality=vq))
        complete+=good;events.append(dict(date=day,price_start=pdays[0],price_end=pdays[-1],volume_start=vdays[0],volume_end=vdays[-1],price_count=64,volume_count=63,candidates=49,common_available_cells=good,missing_official_price_sessions=pmiss,missing_official_volume_sessions=vmiss,cells=cells))
    state=read(CENTRAL);require(state['current']=='monthly_identity_transfer206' and not state['registered_queue'] and len(state['completed'])==204 and sum(r['trials'] for r in state['completed'])==1129,'Centralbefore207')
    result=dict(schema_version=1,created_utc=datetime.now(timezone.utc).isoformat(),passed=True,source_only=True,conditional_research_sources_ready=True,unresolved_blockers=[],context_sha256=CTX_SHA,review_code_sha256=sha(__file__),raw_source_hashes=raw,source_sha256=pins,raw_source_parity=dict(passed=True,exact_source_map_equal=True,count=64),axes=dict(monthly_queries=122,symbols_per_query=49,etf_count=17,first_query=meta['dates'][0],last_query=meta['dates'][-1]),source_window_coverage=dict(price_count=64,volume_count=63,end_offset=0,checked_cells=5978,common_available_cells=complete,unavailable_cells=5978-complete,accepted_source_price_observations=64*complete,accepted_source_volume_observations=63*complete,reason_counts=dict(reasons),by_query=events),source_lineage=prior['source_lineage'],alias_rules=prior['alias_rules'],normalization_facts=prior['normalization_facts'],volume_provenance=dict(path=str(PRIOR),sha256=sha(PRIOR),note='Stored source volume may be raw or provider/split normalized by original loader; no new adjustment. Not consistent turnover or signedflow; within-window scale changes remain limitation.'),actual_source_coverage_computed=True,actual_features_computed=False,actual_orders_computed=False,actual_returns_computed=False,actual_portfolio_returns_computed=False,actual_down_flags_computed=False,actual_dispersion_computed=False,actual_ranks_computed=False,actual_deltas_computed=False,new_model_fits=0,production_files_modified=False,strategy_registered_by_reviewer=False,observers_accessed=False,strict_pit=False,strict_pit_vintage=False,strict_pit_membership=False,observed_central_state=dict(completed_rounds=204,cumulative_trials=1129,registered_queue=[]),limitations=ctx['limitations']+['Source support audit does not establish >=2down availability; no pricecomparisons, volume moments, feature arrays or strategies computed.','Native in prospective context denotes original stored series, not a certification that every volume source has identical split-adjustment conventions.'],synthetic_source_checks_passed=2)
    for p,h in pins.items():require(sha(p)==h,'Changedduringreview')
    with OUT.open('x') as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps(dict(passed=True,path=str(OUT),sha256=sha(OUT),source_pins=len(pins),complete_source_cells=complete,unavailable_cells=5978-complete,new_actual_features=0)))
if __name__=='__main__':main()
