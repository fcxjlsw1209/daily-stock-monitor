"""Round203 source coverage only: no gains, rolling medians, activity flags, ranks or portfolios."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import ast, hashlib, importlib.util, json, math, numbers
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
O=ROOT/'outputs'
CONTEXT=ROOT/'work/monthly_volume_pair203_context.json'
CONTEXT_SHA='b448260b4feb624d18c4813a5d0c671056e45ee2f6ffe62b1f15c73aece17806'
COVERAGE=ROOT/'work/after202_direction_coverage.json'
PRIOR=ROOT/'work/breadth202_SOURCE_REVIEW.json'
PRIOR_SHA='7ff43af545a29b76b8becd1f0d65b8b560baf723d484ce52d80921285b24dd5d'
OUT=ROOT/'work/volume203_SOURCE_REVIEW.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def require(ok,msg):
    if not ok:raise ValueError(msg)
def quality(values):
    q=dict(observations=0,finite_positive=0,zero=0,negative=0,nan=0,infinite=0,nonnumeric_or_boolean=0)
    for v in values:
        q['observations']+=1
        if isinstance(v,(bool,np.bool_)) or not isinstance(v,numbers.Real):q['nonnumeric_or_boolean']+=1;continue
        v=float(v)
        if math.isnan(v):q['nan']+=1
        elif not math.isfinite(v):q['infinite']+=1
        elif v>0:q['finite_positive']+=1
        elif v==0:q['zero']+=1
        else:q['negative']+=1
    require(sum(v for k,v in q.items() if k!='observations')==q['observations'],'Quality partition')
    return q

def window(calendar,day,volume=False):
    t=calendar.index(day);earliest=356 if volume else 294;count=315 if volume else 253
    require(t>=earliest,'Insufficient official sessions before query')
    dates=calendar[t-earliest:t-42+1]
    require(len(dates)==count and dates[-1]==calendar[t-42],'Exact skip42 window')
    return dates

def mapper_from_frozen_ast(path):
    tree=ast.parse(Path(path).read_text());selected=[]
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'SECTOR','FOREIGN'} for t in node.targets):
            ast.literal_eval(node.value);selected.append(node)
        elif isinstance(node,ast.FunctionDef) and node.name=='benchmark_map':selected.append(node)
    names=[n.name if isinstance(n,ast.FunctionDef) else n.targets[0].id for n in selected]
    require(set(names)=={'SECTOR','FOREIGN','benchmark_map'} and len(names)==3,'Exact three mapping AST nodes')
    ns={'pd':pd};exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),ns)
    return ns['benchmark_map'],dict(sector=ns['SECTOR'],foreign=ns['FOREIGN'],nodes=names,
        method='Only original literal SECTOR, FOREIGN assignments and original benchmark_map AST; no other module imports or code executed')

def main():
    require(not OUT.exists(),'Write-once source receipt')
    q=quality([1.,0.,-1.,float('nan'),float('inf'),True,'1',np.bool_(False)])
    require(q==dict(observations=8,finite_positive=1,zero=1,negative=1,nan=1,infinite=1,nonnumeric_or_boolean=3),'Synthetic quality')
    require(window(list(range(500)),400)==list(range(106,359)),'Synthetic price window')
    require(window(list(range(500)),400,True)==list(range(44,359)),'Synthetic volume window')
    require(window(list(range(500)),400,True)[63]==window(list(range(500)),400)[1],'Price/volume current alignment')
    try:window(list(range(500)),355,True)
    except ValueError:pass
    else:raise ValueError('Short volume window must fail')
    pins={}
    def pin(p,expected=None):
        p=Path(p).resolve();require(p.is_relative_to(ROOT),'Nonlocal source')
        h=sha(p);require(expected is None or h==expected,'Changed source '+str(p))
        require(str(p) not in pins or pins[str(p)]==h,'Conflicting pin');pins[str(p)]=h;return h
    pin(__file__);pin(CONTEXT,CONTEXT_SHA);pin(PRIOR,PRIOR_SHA);pin(COVERAGE)
    ctx,prior,coverage=read(CONTEXT),read(PRIOR),read(COVERAGE)
    require(ctx['round']==203 and ctx['name']=='monthly_volume_pair203','Context identity')
    require(ctx['pre_registration_actual_new_features']==ctx['pre_registration_actual_orders']==ctx['pre_registration_actual_returns']==0,'Prospective only')
    require(prior['passed'] is True and prior['raw_source_parity']['exact_source_map_equal'],'Prior source certificate')
    require(coverage['status']=='METADATA_ONLY_UNREGISTERED_DIRECTION_REVIEW','Coverage metadata')
    for k in ['market_panels_loaded','actual_new_features_computed','models_fitted','orders_computed','portfolio_returns_computed','new_trials']:
        require(coverage['scope'][k]==0,'Direction review boundary '+k)
    for p,h in prior['source_sha256'].items():pin(p,h)
    for p,h in ctx['evidence_sha256'].items():pin(ROOT/p,h)
    for p,h in coverage['evidence_sha256'].items():
        if p.endswith('/CONTINUATION_STATUS.json'):require(sha(ROOT/p)==h,'Coverage central state changed')
        else:pin(ROOT/p,h)
    vp=ROOT/'work/volume195_SOURCE_REVIEW.json';volume_review=read(vp)
    require(volume_review['passed'] is True,'Volume source evidence not passed')
    for p,h in volume_review['source_sha256'].items():pin(p,h)
    local=O/'monthly_local_linear70';pin(local/'PLAN.json');pin(local/'META.json')
    raw=read(local/'PLAN.json')['raw_source_hashes'];require(raw==prior['raw_source_hashes']==volume_review['raw_source_hashes'] and len(raw)==64,'Original raw64 closure')
    for p,h in raw.items():pin(p,h)
    p192=read(O/'monthly_sector_recurrence192/PLAN.json');require(p192['raw_source_hashes']==raw,'192 source parity')
    require(p192['sector_source_binding']==prior['sector_source_binding'],'Original sector binding')
    meta=read(local/'META.json');cp=O/'monthly_paired_target143/CALENDAR.json';pin(cp);pin(cp.parent/'PLAN.json')
    calrec=read(cp);cal=calrec['sessions'];require(cal==sorted(set(cal)),'Official calendar')
    require(calrec['provenance']==read(cp.parent/'PLAN.json')['calendar'],'Calendar provenance')
    for p,h in calrec['provenance']['source_hashes'].items():pin(p,h)
    require(len(meta['dates'])==len(meta['symbols'])==122 and len(meta['etfs'])==len(set(meta['etfs']))==17,'Query axes')
    mappingpath=O/'monthly_strict_relative/strict_study.py';pin(mappingpath,ctx['evidence_sha256'][str(mappingpath.relative_to(ROOT))])
    mapper,mapping_metadata=mapper_from_frozen_ast(mappingpath)
    probe=pd.DataFrame({'ZZ':['Technology']},index=pd.to_datetime(['2020-01-02']))
    require(mapper('2020-01-02',['ZZ','TSM','SPY','UNKNOWN'],probe,['SPY'])=={'ZZ':'XLK','TSM':'XLK','SPY':None,'UNKNOWN':None},'Synthetic frozen mapping')
    loader=O/'monthly_stability_monitor/inputs.py';require(str(loader) in raw,'Frozen loader binding')
    spec=importlib.util.spec_from_file_location('volume203_original_source_loader',loader);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    panel,info=mod.load_panel(meta['etfs']);used=dict(info['source_hashes']);used[str(loader)]=sha(loader)
    require(used==raw,'Original source map changed');adj,volume,sectors=panel['adj'],panel['volume'],panel['sectors']
    for frame in [adj,volume,sectors]:require(frame.index.is_unique and frame.index.is_monotonic_increasing and frame.columns.is_unique,'Source axes')
    require(adj.index.equals(volume.index) and adj.columns.equals(volume.columns),'Price-volume axes differ')
    events=[];reasons=Counter();accepted_prices=accepted_volume=complete=0
    for i,(day,names) in enumerate(zip(meta['dates'],meta['symbols'])):
        require(len(names)==len(set(names))==49 and set(meta['etfs']).issubset(names),'Original49 identities')
        t=cal.index(day);require((i==0 and day=='2016-08-17') or cal[t+1][:7]!=day[:7],'Monthend query')
        dates=window(cal,day);vdates=window(cal,day,True);idx=pd.to_datetime(dates);vidx=pd.to_datetime(vdates)
        missing=idx.difference(adj.index).strftime('%Y-%m-%d').tolist();vmissing=vidx.difference(volume.index).strftime('%Y-%m-%d').tolist()
        mapping=mapper(day,names,sectors.loc[:pd.Timestamp(day)],meta['etfs']);require(set(mapping)==set(names),'Mapping identity')
        needed=set(names)|{v for v in mapping.values() if v is not None}
        qc={s:quality(adj.loc[idx,s].to_numpy()) for s in sorted(needed) if s in adj.columns and not missing}
        vc={s:quality(volume.loc[vidx,s].to_numpy()) for s in names if s in volume.columns and not vmissing}
        cells=[]
        for symbol in names:
            benchmark=mapping[symbol];bad=[];reason='available'
            if symbol in meta['etfs']:require(benchmark is None,'ETF neutral');reason='etf_neutral'
            elif benchmark is None:reason='missing_benchmark'
            elif missing or vmissing:reason='missing_official_session'
            else:
                require(benchmark in mapping_metadata['sector'].values() and benchmark!=symbol,'Own benchmark identity')
                for s in [symbol,benchmark]:
                    if s not in qc:bad.append(dict(symbol=s,field='adj',reason='missing_column'))
                    elif qc[s]['finite_positive']!=253:bad.append(dict(symbol=s,field='adj',reason='invalid_price',counts=qc[s]))
                if symbol not in vc:bad.append(dict(symbol=symbol,field='volume',reason='missing_column'))
                elif vc[symbol]['finite_positive']!=315:bad.append(dict(symbol=symbol,field='volume',reason='invalid_volume',counts=vc[symbol]))
                if bad:reason='invalid_consumed_source'
            available=reason=='available'
            if available:complete+=1;accepted_prices+=506;accepted_volume+=315
            reasons[reason]+=1;cells.append(dict(symbol=symbol,benchmark=benchmark,available=available,reason=reason,invalid_sources=bad))
        events.append(dict(query=day,price_start=dates[0],price_end=dates[-1],volume_start=vdates[0],volume_end=vdates[-1],price_count=253,volume_count=315,endpoint_dates=[dates[k] for k in range(0,253,21)],missing_official_sessions=missing,missing_volume_sessions=vmissing,benchmarks=mapping,source_quality=qc,volume_quality=vc,cells=cells))
    require(sum(reasons.values())==122*49,'5978 source cells')
    central=read(O/'monthly_target70_research/CONTINUATION_STATUS.json')
    require(central['current']=='monthly_breadth_pair202' and central['registered_queue']==[] and len(central['completed'])==200 and sum(r['trials'] for r in central['completed'])==1117,'Pre203 central state')
    result=dict(schema_version=1,passed=True,source_only=True,conditional_research_sources_ready=True,unresolved_blockers=[],created_utc=datetime.now(timezone.utc).isoformat(),context_sha256=CONTEXT_SHA,review_code_sha256=sha(__file__),coverage_sha256=sha(COVERAGE),raw_source_hashes=raw,source_sha256=pins,raw_source_parity=dict(passed=True,exact_source_map_equal=True,count=64,current_loader_sha256=sha(loader)),mapping_source=dict(path=str(mappingpath),sha256=sha(mappingpath),**mapping_metadata),sector_source_binding=p192['sector_source_binding'],axes=dict(monthly_queries=122,symbols_per_query=49,etf_count=17,first_query=meta['dates'][0],last_query=meta['dates'][-1]),source_window_coverage=dict(price_count=253,volume_count=315,skip_sessions=42,checked_cells=5978,common_available_cells=complete,unavailable_cells=5978-complete,reason_counts=dict(reasons),accepted_source_price_observations=accepted_prices,accepted_source_volume_observations=accepted_volume,observation_basis='peraccepted complete cell source consumption not unique raw records; rejected partial reads excluded',by_query=events),source_lineage=prior['source_lineage'],alias_rules=prior['alias_rules'],normalization_facts=prior['normalization_facts'],volume_provenance=volume_review,strict_pit=False,strict_pit_vintage=False,strict_pit_membership=False,actual_source_coverage_computed=True,actual_features_computed=False,actual_orders_computed=False,actual_returns_computed=False,actual_portfolio_returns_computed=False,actual_signal_returns_computed=False,actual_medians_computed=False,actual_high_flags_computed=False,actual_activity_blocks_computed=False,actual_covariance_computed=False,actual_ranks_computed=False,actual_deltas_computed=False,new_model_fits=0,production_files_modified=False,strategy_registered_by_reviewer=False,observers_accessed=False,observed_central_state=dict(completed_rounds=200,cumulative_trials=1117,registered_queue=[]),limitations=ctx['limitations']+['Fullsource availability audit only; no derived gains, medians, flags, covariance, ranks or performance.','Imports only frozen original inputloader and isolated mappingAST; no scoring/model/strategy builder orNPZ featurearrays.'],synthetic_source_checks_passed=6)
    for p,h in pins.items():require(sha(p)==h,'Source changed midreview')
    with OUT.open('x') as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps(dict(passed=True,path=str(OUT),sha256=sha(OUT),source_pins=len(pins),common_available_cells=complete,reason_counts=dict(reasons),accepted_source_price_observations=accepted_prices,accepted_source_volume_observations=accepted_volume)))
if __name__=='__main__':main()
