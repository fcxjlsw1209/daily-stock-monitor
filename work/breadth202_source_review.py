"""Round202 source coverage only: no block gains, breadth, ranks or portfolios."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import ast, hashlib, importlib.util, json, math, numbers
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
O=ROOT/'outputs'
CONTEXT=ROOT/'work/monthly_breadth_pair202_context.json'
CONTEXT_SHA='0e7eb021c65a22b843c285f83525ce9ff0bc5e908e2a5a0a4cf0b1fd66952eaf'
COVERAGE=ROOT/'work/after201_breadth_pair_coverage.json'
PRIOR=ROOT/'work/median201_SOURCE_REVIEW.json'
PRIOR_SHA='4ca67a75465bb05f8b255272a185c436047c89fefe2193d939ec5ee9ae1e810d'
OUT=ROOT/'work/breadth202_SOURCE_REVIEW.json'
BASKET=['XLB','XLE','XLF','XLI','XLK','XLP','XLU','XLV','XLY']
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

def window(calendar,day):
    t=calendar.index(day);require(t>=294,'Need 294 sessions before query')
    dates=calendar[t-294:t-42+1]
    require(len(dates)==253 and dates[-1]==calendar[t-42],'Exact skip42 window')
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
    require(window(list(range(400)),350)==list(range(56,309)),'Synthetic window')
    try:window(list(range(400)),293)
    except ValueError:pass
    else:raise ValueError('Short window must fail')
    pins={}
    def pin(p,expected=None):
        p=Path(p).resolve();require(p.is_relative_to(ROOT),'Nonlocal source')
        h=sha(p);require(expected is None or h==expected,'Changed source '+str(p))
        require(str(p) not in pins or pins[str(p)]==h,'Conflicting pin');pins[str(p)]=h;return h
    pin(__file__);pin(CONTEXT,CONTEXT_SHA);pin(PRIOR,PRIOR_SHA);pin(COVERAGE)
    ctx,prior,coverage=read(CONTEXT),read(PRIOR),read(COVERAGE)
    require(ctx['round']==202 and ctx['name']=='monthly_breadth_pair202','Context identity')
    require(ctx['pre_registration_actual_new_features']==ctx['pre_registration_actual_orders']==ctx['pre_registration_actual_returns']==0,'Prospective only')
    require(prior['passed'] is True and prior['raw_source_parity']['exact_source_map_equal'],'Prior source certificate')
    require(coverage['status']=='METADATA_ONLY_COVERAGE_NOT_REGISTERED' and all(coverage[k]==0 for k in ['actual_new_features_computed','actual_new_orders_computed','actual_new_returns_computed','strategy_trials_added']),'Coverage metadata')
    require(ctx['formation']['breadth_etfs']==BASKET,'Frozen breadth identities')
    for p,h in prior['source_sha256'].items():pin(p,h)
    for p,h in ctx['evidence_sha256'].items():pin(ROOT/p,h)
    # The coverage central hash describes the unchanged pre-registration state;
    # recheck it here, but do not include mutable central state in frozen sources.
    for p,h in coverage['input_sha256'].items():
        if p.endswith('/CONTINUATION_STATUS.json'):require(sha(ROOT/p)==h,'Coverage central state changed')
        else:pin(ROOT/p,h)
    local=O/'monthly_local_linear70';pin(local/'PLAN.json');pin(local/'META.json')
    raw=read(local/'PLAN.json')['raw_source_hashes'];require(raw==prior['raw_source_hashes'] and len(raw)==64,'Original raw64 closure')
    for p,h in raw.items():pin(p,h)
    p192=read(O/'monthly_sector_recurrence192/PLAN.json')
    for name in ['monthly_sector_dimension176','monthly_sector_transition189']:
        other=read(O/name/'PLAN.json');require(other['raw_source_hashes']==raw,'Prior sector raw64 parity')
        for field in ['mapping_source','sectors_source','manifest_sha256','sources','price_column','loader']:
            require(other['sector_source_binding'][field]==p192['sector_source_binding'][field],'Prior sector binding parity: '+field)
        if name=='monthly_sector_dimension176':
            require(other['sector_source_binding']['dimension_etfs']==BASKET and other['sector_source_binding']['relative_only_etfs']==['XLC','XLRE'],'176 nine plus two partition')
        else:require(other['sector_source_binding']['own_sector_etfs']==p192['sector_source_binding']['own_sector_etfs'],'189 own-sector domain')
    require(p192['raw_source_hashes']==raw,'192 source parity')
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
    spec=importlib.util.spec_from_file_location('breadth202_original_source_loader',loader);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    panel,info=mod.load_panel(meta['etfs']);used=dict(info['source_hashes']);used[str(loader)]=sha(loader)
    require(used==raw,'Original source map changed');adj,sectors=panel['adj'],panel['sectors']
    for frame in [adj,sectors]:require(frame.index.is_unique and frame.index.is_monotonic_increasing and frame.columns.is_unique,'Source axes')
    require(set(BASKET).issubset(adj.columns),'Missing fixed breadth ETF columns')
    events=[];reasons=Counter();basket_counts=Counter();accepted_prices=0;complete=0
    for i,(day,names) in enumerate(zip(meta['dates'],meta['symbols'])):
        require(len(names)==len(set(names))==49 and set(meta['etfs']).issubset(names),'Original49 identities')
        t=cal.index(day);require((i==0 and day=='2016-08-17') or cal[t+1][:7]!=day[:7],'Monthend query')
        dates=window(cal,day);idx=pd.to_datetime(dates)
        missing=idx.difference(adj.index).strftime('%Y-%m-%d').tolist()
        mapping=mapper(day,names,sectors.loc[:pd.Timestamp(day)],meta['etfs'])
        require(set(mapping)==set(names),'Mapping identity')
        needed=set(BASKET)|set(names)|{v for v in mapping.values() if v is not None}
        qc={s:quality(adj.loc[idx,s].to_numpy()) for s in sorted(needed) if s in adj.columns and not missing}
        cells=[]
        for symbol in names:
            benchmark=mapping[symbol];basket=[s for s in BASKET if s!=benchmark];consumed=[];bad=[]
            reason='available'
            if symbol in meta['etfs']:require(benchmark is None,'ETF neutral');reason='etf_neutral'
            elif benchmark is None:reason='missing_benchmark'
            elif missing:reason='missing_official_session'
            else:
                require(benchmark in mapping_metadata['sector'].values(),'Unrecognized own benchmark')
                consumed=[symbol,benchmark]+basket
                require(len(set(consumed))==len(consumed) and len(basket) in [8,9],'Consumed identities')
                for s in consumed:
                    if s not in qc:bad.append(dict(symbol=s,reason='missing_column'))
                    elif qc[s]['finite_positive']!=253:bad.append(dict(symbol=s,reason='invalid_price',counts=qc[s]))
                if bad:reason='invalid_consumed_source'
            available=reason=='available'
            if available:
                complete+=1;basket_counts[str(len(basket))]+=1;accepted_prices+=253*len(consumed)
            reasons[reason]+=1
            cells.append(dict(symbol=symbol,benchmark=benchmark,breadth_etfs=basket if benchmark and symbol not in meta['etfs'] else [],consumed=consumed,available=available,reason=reason,invalid_sources=bad))
        events.append(dict(query=day,price_start=dates[0],price_end=dates[-1],price_count=253,endpoint_dates=[dates[k] for k in range(0,253,21)],missing_official_sessions=missing,benchmarks=mapping,source_quality=qc,cells=cells))
    require(sum(reasons.values())==122*49,'5978 source cells')
    central=read(O/'monthly_target70_research/CONTINUATION_STATUS.json')
    require(central['current']=='monthly_median_risk201' and central['registered_queue']==[] and len(central['completed'])==199 and sum(r['trials'] for r in central['completed'])==1114,'Pre202 central state')
    result=dict(schema_version=1,passed=True,source_only=True,conditional_research_sources_ready=True,unresolved_blockers=[],created_utc=datetime.now(timezone.utc).isoformat(),context_sha256=CONTEXT_SHA,review_code_sha256=sha(__file__),coverage_sha256=sha(COVERAGE),raw_source_hashes=raw,source_sha256=pins,raw_source_parity=dict(passed=True,exact_source_map_equal=True,count=64,current_loader_sha256=sha(loader)),mapping_source=dict(path=str(mappingpath),sha256=sha(mappingpath),**mapping_metadata),sector_source_binding=p192['sector_source_binding'],axes=dict(monthly_queries=122,symbols_per_query=49,etf_count=17,first_query=meta['dates'][0],last_query=meta['dates'][-1]),source_window_coverage=dict(price_count=253,skip_sessions=42,checked_cells=5978,common_available_cells=complete,unavailable_cells=5978-complete,reason_counts=dict(reasons),available_cells_by_basket_size=dict(basket_counts),accepted_source_price_observations=accepted_prices,observation_basis='per accepted complete cell consumption, not unique source records; rejected partial reads excluded',by_query=events),source_lineage=prior['source_lineage'],alias_rules=prior['alias_rules'],normalization_facts=prior['normalization_facts'],integration_source_verification=dict(numerical_integration_verified=False,core='Original27 arrays unchanged; .01 centered-rank auxiliary only before original episode; initialpartial0, calm active'),strict_pit=False,strict_pit_vintage=False,strict_pit_membership=False,actual_source_coverage_computed=True,actual_features_computed=False,actual_orders_computed=False,actual_returns_computed=False,actual_portfolio_returns_computed=False,actual_signal_returns_computed=False,actual_breadth_computed=False,actual_covariance_computed=False,actual_ranks_computed=False,actual_deltas_computed=False,new_model_fits=0,production_files_modified=False,strategy_registered_by_reviewer=False,observers_accessed=False,observed_central_state=dict(completed_rounds=199,cumulative_trials=1114,registered_queue=[]),limitations=['This admission checks full source availability only, not derived gains, breadth, covariance, ranks or strategy performance.','Historical proxyTop32+theme17 and retrospective vendor/action/classification versions are conditional research, not certified strictPIT or pristineOOS.','Source audit imports only frozen original panel loader and isolated frozen mapping AST; no scoring/model/strategy builder or NPZ feature arrays.'],synthetic_source_checks_passed=4)
    for p,h in pins.items():require(sha(p)==h,'Source changed midreview')
    with OUT.open('x') as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps(dict(passed=True,path=str(OUT),sha256=sha(OUT),source_pins=len(pins),common_available_cells=complete,reason_counts=dict(reasons),accepted_source_price_observations=accepted_prices)))
if __name__=='__main__':main()
