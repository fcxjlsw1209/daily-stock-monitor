"""262 supplied-source total versus beta-one tracking-risk leadership.

No source loading, registration, fitted models or return replay. Source-only
statistics precede their independent gate; new ranks and funded scores follow it.
"""
from copy import deepcopy
from numbers import Real
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
FROZEN={
 'work/monthly_portfolio_path247_kernel.py':'18c7726ae701bb3557636bfa69a0c0dc32d516286f3bb1cd27f50157d5380092',
 'work/monthly_peer239_kernel.py':'9d0b9cc7d39243a6d9ca3117a71b97c93c2e1be4adb67757ff5cfaa42f0da419',
 'work/monthly_relative_gate235_kernel.py':'bab4a2b9794ef00a63d43422a51cef167daffbb11b31dc0452cfcd2e74a6af8d',
 'work/ownsector238_repair_source_policy.json':'bb743a4ca456c43219436f5e39f0387b3b721e65d2344012de699ba79d6ec7de',
 'work/peer239_maturity_policy.json':'3bbcbcbe141fb0d2d96cd64ca74f2ee1e9450e12c1f2594c7cb84fb98bf64862',
 'work/monthly_peer_path252_v2_LISTING_POLICY.json':'919a698dafe5fafc7b0516ac7fb613d1ff5292e0a831f049dc134b8864ca1e91',
 'work/tracking_risk262_listing_policy.json':'dbb8821baa54ddee961a6d06d30bd478b5ff5b18beae3f899cd92466b4ebae92'}
for _p,_h in FROZEN.items():
    if hashlib.sha256((ROOT/_p).read_bytes()).hexdigest()!=_h:raise ValueError('Frozen dependency changed: '+_p)
def _module(path,name):
    spec=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
old=_module('work/monthly_portfolio_path247_kernel.py','tracking262_original247')
prior=_module('work/monthly_peer239_kernel.py','tracking262_axes239')
REPAIR=json.loads((ROOT/'work/ownsector238_repair_source_policy.json').read_text())
BOUNDS=json.loads((ROOT/'work/peer239_maturity_policy.json').read_text())['securities']
LATE=json.loads((ROOT/'work/monthly_peer_path252_v2_LISTING_POLICY.json').read_text())
EARLY=json.loads((ROOT/'work/tracking_risk262_listing_policy.json').read_text())
ARMS=('incumbent_local67','sector_total_risk_control','sector_tracking_risk_candidate')
INITIAL_PARTIAL='2016-08-17'
SOURCE_SCHEMA='tracking_risk262_source_contract_v1'
STAT_SCHEMA='monthly_tracking_risk262_statistics_v1'
AUDIT_SCHEMA='monthly_tracking_risk262_decisions_v1'
MAPPING_KEYS=('query_index','column_index','date','symbol','issuer_key','original_sector','sector','mapping_origin','mapping_repair','mapping_evidence')
CLOCK_KEYS=('query_index','date','initial_partial','numeric_evaluation_required','symbols','stock_indices','query_position','next_official_session','numerator_positions','numerator_dates','risk_positions','risk_dates','required_positions','required_dates')
EVENT_KEYS=('query_index','date','symbols','initial_partial','query_position','next_official_session','clock_sessions','numerator_positions','numerator_dates','risk_positions','risk_dates','required_positions','required_dates','stock_indices','mature_indices','supported_indices','cells','total_target','tracking_target')
CELL_KEYS=('column_index','symbol','issuer_key','original_sector','sector','mapping_origin','mapping_repair','mapping_evidence','history_bound','path_history_bound','history_status','own_observations','sector_observations','statistics')
OBS_KEYS=('date','value','status','reason')
STAT_KEYS=('schema','own_momentum','sector_momentum','relative_momentum','own_returns','sector_returns','tracking_returns','total_risk','tracking_risk','total_ratio','tracking_ratio','supported')
RISK_KEYS=('mean','centered','squares','square_sum','variance','sd','constant')
COUNT_FIELDS=('events','symbols_per_event','eligible_cells','initial_partial_events','stock_cells','mature_cells','warmup_cells','unmapped_warmup_cells','supported_cells','zero_total_cells','zero_tracking_cells','own_price_observations','sector_price_observations','stock_return_observations','sector_return_observations','tracking_return_observations','total_risk_statistics','tracking_risk_statistics','total_nonzero_deltas','tracking_nonzero_deltas','total_changed_holdings','tracking_changed_holdings','candidate_control_changed_holdings','original_episode_calls','new_model_fits','predictive_model_fits')
DECISION_EVENT_KEYS=('query_index','date','initial_partial','eligible_indices','supported_indices','original_support','total_deltas','tracking_deltas','choices','calm','weak','core_weight','pre_order')
need=old.require

def dependency_files():
    return tuple(dict.fromkeys([*(ROOT/p for p in FROZEN),*old.dependency_files()]))

def _exact(a,b,label):
    need(type(a) is type(b),label+' type')
    if isinstance(b,dict):
        need(set(a)==set(b),label+' keys')
        for k in b:_exact(a[k],b[k],label+'.'+k)
    elif isinstance(b,list):
        need(len(a)==len(b),label+' length')
        for x,y in zip(a,b):_exact(x,y,label)
    elif isinstance(b,float):need(a.hex()==b.hex(),label+' binary64')
    else:need(a==b,label+' value')

def _finite(x,label):
    need(isinstance(x,Real) and not isinstance(x,(bool,np.bool_)),label+' real nonbool')
    try:y=float(x)
    except (TypeError,ValueError,OverflowError) as exc:raise ValueError(label+' binary64') from exc
    need(math.isfinite(y),label+' finite')
    need(not(y==0. and x!=0),label+' nonzero conversion underflow');return y

def _positive(x,label):
    y=_finite(x,label);need(y>0.,label+' positive');return y

def _arith(x,label):
    if not math.isfinite(x):raise ArithmeticError('Nonfinite '+label)
    return x

def _sum(xs,label):
    try:y=math.fsum(xs)
    except (OverflowError,ValueError) as exc:raise ArithmeticError(label) from exc
    return _arith(y,label)

def _div(a,b,label):
    need(b!=0.,label+' nonzero denominator')
    y=_arith(a/b,label)
    if a!=0. and y==0.:raise ArithmeticError('Underflow '+label)
    return y

def _square(x,label):
    y=_arith(x*x,label)
    if x!=0. and y==0.:raise ArithmeticError('Underflow '+label)
    return y

def risk63(values):
    need(isinstance(values,(list,tuple,np.ndarray)) and len(values)==63,'Exactly63 returns')
    xs=[_finite(x,'return') for x in values];constant=all(x==xs[0] for x in xs)
    if constant:mean=xs[0];centered=[0.]*63;squares=[0.]*63;ss=var=sd=0.
    else:
        mean=_div(_sum(xs,'mean sum'),63.,'mean');centered=[_arith(x-mean,'centered return') for x in xs]
        squares=[_square(x,'centered square') for x in centered];ss=_sum(squares,'square sum');need(ss>0.,'Nonconstant positive square sum')
        var=_div(ss,62.,'sample variance');sd=_arith(math.sqrt(var),'sample SD');need(sd>0.,'Positive sample SD')
    result=dict(mean=mean,centered=centered,squares=squares,square_sum=ss,variance=var,sd=sd,constant=constant)
    need(tuple(result)==RISK_KEYS,'Risk schema');return result

def measure(own65,sector65):
    need(isinstance(own65,(list,tuple,np.ndarray)) and isinstance(sector65,(list,tuple,np.ndarray)) and len(own65)==len(sector65)==65,'Exactly65 required prices per instrument')
    own=[_positive(x,'own price') for x in own65];sector=[_positive(x,'sector price') for x in sector65]
    # required order: t-252, then t-63..t; t-42 is offset22.
    mi=_arith(_div(own[22],own[0],'own gross')-1.,'own momentum');mb=_arith(_div(sector[22],sector[0],'sector gross')-1.,'sector momentum')
    gi=_arith(1.+mi,'own recomposed gross');gb=_arith(1.+mb,'sector recomposed gross');need(gi>0. and gb>0.,'Positive recomposed gross')
    rel=_arith(_div(gi,gb,'relative gross')-1.,'relative momentum')
    ri=[_arith(_div(own[k],own[k-1],'own daily gross')-1.,'own return') for k in range(2,65)]
    rb=[_arith(_div(sector[k],sector[k-1],'sector daily gross')-1.,'sector return') for k in range(2,65)]
    tracking=[_arith(x-y,'tracking return') for x,y in zip(ri,rb)]
    total=risk63(ri);track=risk63(tracking);supported=total['sd']>0. and track['sd']>0.
    rt=_div(rel,total['sd'],'total risk ratio') if supported else None
    rr=_div(rel,track['sd'],'tracking risk ratio') if supported else None
    out=dict(schema=STAT_SCHEMA,own_momentum=mi,sector_momentum=mb,relative_momentum=rel,own_returns=ri,sector_returns=rb,tracking_returns=tracking,total_risk=total,tracking_risk=track,total_ratio=rt,tracking_ratio=rr,supported=supported)
    need(tuple(out)==STAT_KEYS,'Statistics schema');return out

def rank(value,domain):
    need(bool(domain),'Nonempty rank domain');less=sum(x<value for x in domain);equal=sum(x==value for x in domain)
    return (float(2*less+equal+1)/2.0)/float(len(domain))-.5

def _n(arrays,n):
    shape=np.shape(arrays['baseline']);need(len(shape)==2 and shape[1]>0,'Original axes')
    n=shape[0] if n is None else n
    need(type(n) is int and 0<=n<=shape[0],'Canonical prefix0..N');return n

def _meta(meta,n):
    return {k:deepcopy(meta[k][:n]) for k in ('dates','ends','symbols')}|{'etfs':deepcopy(meta['etfs'])}

def _gate(callback,args,label):
    if callback is None:return dict(passed=True,synthetic_no_callback=True)
    r=callback(*args);need(isinstance(r,dict) and r.get('passed') is True,label+' passed receipt required');return deepcopy(r)

def _clock(q,day,names,t,partial,sessions,etfs):
    ep=[] if partial else [t-252,t-42];rp=[] if partial else list(range(t-63,t+1));req=sorted(set(ep+rp))
    ds=lambda ps:[str(sessions[k].date()) for k in ps]
    return dict(query_index=q,date=str(day.date()),initial_partial=partial,numeric_evaluation_required=not partial,symbols=deepcopy(names),stock_indices=[j for j,s in enumerate(names) if s not in etfs],query_position=t,next_official_session=str(sessions[t+1].date()),numerator_positions=ep,numerator_dates=ds(ep),risk_positions=rp,risk_dates=ds(rp),required_positions=req,required_dates=ds(req))

def _mapping(r,q,j,day,names):
    need(isinstance(r,dict) and set(r)==set(MAPPING_KEYS),'Exact mapping projection')
    need(type(r['query_index']) is int and type(r['column_index']) is int and r['query_index']==q and r['column_index']==j and r['date']==day and r['symbol']==names[j],'Mapping axes/identity')
    need(prior.ident(r['issuer_key']) and r['mapping_origin'] in ('query_date_sector_panel','frozen_FOREIGN_override'),'Whole issuer/mapping origin')
    original=r['original_sector'];need(original is None or original in prior.SECTORS,'Original sector identity')
    if original=='XLC':expected=('VOX','full_period_communication_proxy',None)
    elif original is not None:expected=(original,'unchanged',None)
    else:
        rule=next((x for x in REPAIR['missing_mapping_rules'] if x['symbol']==r['symbol']),None)
        if rule and rule['valid_from']<=day and (rule['valid_until_exclusive'] is None or day<rule['valid_until_exclusive']):expected=(rule['benchmark'],'dated_business_proxy',rule['source_ids'])
        else:expected=(None,'unresolved',None)
    _exact([r['sector'],r['mapping_repair'],r['mapping_evidence']],list(expected),'Frozen mapping rule')
    if r['symbol'] in {b['symbol'] for b in BOUNDS}:need(r['issuer_key']==next(b['issuer_key'] for b in BOUNDS if b['symbol']==r['symbol']),'Known ticker whole issuer mismatch')
    return deepcopy(r)

def _bounds(row,partial):
    if partial:return None,None
    b=next((x for x in BOUNDS if x['symbol']==row['symbol'] and x['issuer_key']==row['issuer_key'] and x['available_from']<=row['date']),None);p=b
    for policy in (LATE,EARLY):
        x=policy['path_bound_override']
        if row['symbol']==x['symbol'] and row['issuer_key']==x['issuer_key'] and x['available_from']<=row['date']:
            _exact(b,policy['original_history_bound'],'Original lower bound preserved');p=x
    return deepcopy(b),deepcopy(p)

def _slot(frame,symbol,day):
    if symbol not in frame.columns:return None,'missing','missing_column'
    if day not in frame.index:return None,'missing','missing_row'
    x=frame.at[day,symbol]
    if x is None or x is pd.NA:return None,'missing','missing_value'
    need(isinstance(x,Real) and not isinstance(x,(bool,np.bool_)),'Malformed price '+symbol+'/'+str(day.date()))
    try:y=float(x)
    except (TypeError,ValueError,OverflowError) as exc:raise ValueError('Malformed price conversion') from exc
    if math.isnan(y):return None,'missing','missing_value'
    need(math.isfinite(y) and y>0.,'Nonpositive/nonfinite price '+symbol+'/'+str(day.date()))
    return y,'valid',None

def _observations(frame,symbol,dates,bound):
    out=[]
    for date in dates:
        value,status,reason=_slot(frame,symbol,pd.Timestamp(date));before=bound is not None and date<bound['not_before']
        need(status==('missing' if before else 'valid'),('Positive prebound identity conflict ' if before else 'Missing required source ')+symbol+'/'+date)
        out.append(dict(date=date,value=value,status=status,reason=reason))
    return out

def _check_obs(observations,dates,bound):
    need(isinstance(observations,list) and len(observations)==len(dates),'Full cached required source')
    for o,d in zip(observations,dates):
        need(isinstance(o,dict) and set(o)==set(OBS_KEYS) and o['date']==d,'Cached source clock/schema')
        before=bound is not None and d<bound['not_before']
        if before:need(o['value'] is None and o['status']=='missing' and o['reason'] in prior.MISSING_REASONS,'Prebound missing only')
        else:need(type(o['value']) is float and _positive(o['value'],'cached required price')>0. and o['status']=='valid' and o['reason'] is None,'Complete positive source')

def _cell(row,clock,own,sector):
    partial=clock['initial_partial'];b,p=_bounds(row,partial)
    status='initial_partial' if partial else 'official_pre_inception' if b is not None and clock['numerator_dates'][0]<b['not_before'] else 'observed_full_window'
    need(row['sector'] is not None or status in ('initial_partial','official_pre_inception'),'Mature mapping missing')
    _check_obs(own,clock['required_dates'],p)
    _check_obs(sector,clock['required_dates'] if row['sector'] is not None else [],None)
    stat=None
    if status=='observed_full_window':stat=measure([o['value'] for o in own],[o['value'] for o in sector])
    out={k:deepcopy(row[k]) for k in CELL_KEYS[:8]};out.update(history_bound=b,path_history_bound=p,history_status=status,own_observations=own,sector_observations=sector,statistics=stat)
    need(tuple(out)==CELL_KEYS,'Cell schema');return out

def _event(clock,cells,sessions):
    t=clock['query_position'];out={k:deepcopy(clock[k]) for k in EVENT_KEYS[:6] if k in clock}
    out['clock_sessions']=[] if clock['initial_partial'] else [str(x.date()) for x in sessions[t-252:t+2]]
    for k in EVENT_KEYS[7:14]:out[k]=deepcopy(clock[k])
    out.update(mature_indices=[c['column_index'] for c in cells if c['history_status']=='observed_full_window'],supported_indices=[c['column_index'] for c in cells if c['statistics'] is not None and c['statistics']['supported']],cells=cells,total_target=None,tracking_target=None)
    need(tuple(out)==EVENT_KEYS,'Event schema');return out

def build_features(panel,meta,calendar,arrays,review,n=None):
    frame,sessions,queries,etfs,width=prior._input(panel,meta,calendar,arrays,n);n=len(queries)
    if queries:
        consumed_index=frame.index[frame.index<=queries[-1][0]]
        need(consumed_index.isin(sessions).all(),'Consumed source rows must be official sessions')
    need(isinstance(review,dict) and review.get('schema')==SOURCE_SCHEMA and review.get('passed') is True and review.get('metadata_only') is True and review.get('source_admitted') is False,'Prospective metadata-only source contract')
    need(review.get('strict_pit') is False and isinstance(review.get('clock_events'),list) and len(review['clock_events'])>=n,'Source clocks')
    _exact(review['listing_bounds'],BOUNDS,'Original listing policy');_exact(review['early_listing_policy']['policy'],EARLY,'Early actual-trade policy')
    _exact(review['listing_policy252_v2']['policy'],LATE,'Late policy preserved')
    by={q:[] for q in range(n)}
    for row in review['mapping_records']:
        need(isinstance(row,dict) and type(row.get('query_index')) is int and row['query_index']>=0,'Mapping index')
        if row['query_index']<n:by[row['query_index']].append(row)
    result=[]
    for q,(day,names,t,partial) in enumerate(queries):
        clock=_clock(q,day,names,t,partial,sessions,etfs);_exact(review['clock_events'][q],clock,'Official source clock')
        need([r.get('column_index') for r in by[q]]==clock['stock_indices'],'All direct stocks beforeE')
        rows=[_mapping(r,q,j,str(day.date()),names) for j,r in zip(clock['stock_indices'],by[q])]
        need(len({r['issuer_key'] for r in rows})==len(rows),'Duplicate whole issuer')
        cells=[]
        for r in rows:
            _,bound=_bounds(r,partial)
            own=_observations(frame,r['symbol'],clock['required_dates'],bound)
            sector=_observations(frame,r['sector'],clock['required_dates'],None) if r['sector'] is not None else []
            # Retrospective factor inception is a contradiction check, never a PIT clock.
            for b in review.get('factor_listing_bounds',[]):
                if r['sector']==b['symbol'] and clock['required_dates']:need(clock['required_dates'][0]>=b['inception_date'],'Factor existed before required date')
            cells.append(_cell(r,clock,own,sector))
        result.append(_event(clock,cells,sessions))
    return result

def _cached(features,meta,n):
    need(isinstance(features,list) and len(features)>=n,'Cached features prefix');out=[];prior_query=None
    etfs=set(meta['etfs'])
    for q,f in enumerate(features[:n]):
        need(isinstance(f,dict) and set(f)==set(EVENT_KEYS),'Cached event schema')
        names=meta['symbols'][q];d=meta['dates'][q];partial=d==INITIAL_PARTIAL;t=f['query_position']
        need(f['query_index']==q and f['date']==d and f['symbols']==names and type(f['initial_partial']) is bool and f['initial_partial']==partial and type(t) is int,'Cached original identity')
        need(prior_query is None or d>prior_query,'Cached chronological queries');prior_query=d
        if partial:
            need(q==0 and f['clock_sessions']==[],'Partial no new source clock')
            clock=dict(query_index=q,date=d,initial_partial=True,numeric_evaluation_required=False,symbols=deepcopy(names),stock_indices=[j for j,s in enumerate(names) if s not in etfs],query_position=t,next_official_session=f['next_official_session'],numerator_positions=[],numerator_dates=[],risk_positions=[],risk_dates=[],required_positions=[],required_dates=[])
            sessions=pd.DatetimeIndex([])
        else:
            cs=f['clock_sessions'];sessions=prior.axes._calendar(cs);need(len(cs)==254 and cs[252]==d and cs[-1]==f['next_official_session'] and cs[-1][:7]!=d[:7] and t>=252,'Cached complete official clock')
            ep=[t-252,t-42];rp=list(range(t-63,t+1));req=[t-252]+rp
            clock=dict(query_index=q,date=d,initial_partial=False,numeric_evaluation_required=True,symbols=deepcopy(names),stock_indices=[j for j,s in enumerate(names) if s not in etfs],query_position=t,next_official_session=cs[-1],numerator_positions=ep,numerator_dates=[cs[0],cs[210]],risk_positions=rp,risk_dates=cs[189:253],required_positions=req,required_dates=[cs[0]]+cs[189:253])
        for k in CLOCK_KEYS:
            if k in f:_exact(f[k],clock[k],'Cached clock '+k)
        need(isinstance(f['cells'],list) and [c.get('column_index') for c in f['cells']]==clock['stock_indices'],'Cached complete inventory')
        cells=[]
        for c in f['cells']:
            need(isinstance(c,dict) and set(c)==set(CELL_KEYS),'Cached cell schema');j=c['column_index']
            r=dict(query_index=q,date=d,**{k:deepcopy(c[k]) for k in CELL_KEYS[:8]});_mapping(r,q,j,d,names)
            rebuilt=_cell(r,clock,deepcopy(c['own_observations']),deepcopy(c['sector_observations']));_exact(c,rebuilt,'Cached source/statistics');cells.append(rebuilt)
        need(len({c['issuer_key'] for c in cells})==len(cells),'Cached unique issuer')
        rebuilt=deepcopy(f);rebuilt.update(cells=cells,mature_indices=[c['column_index'] for c in cells if c['history_status']=='observed_full_window'],supported_indices=[c['column_index'] for c in cells if c['statistics'] is not None and c['statistics']['supported']],total_target=None,tracking_target=None)
        for k in ('mature_indices','supported_indices'):_exact(f[k],rebuilt[k],'Cached common domain')
        # Shared instruments must have exactly the same dates/prices across roles.
        seen={}
        for c in cells:
            for symbol,obs in ((c['symbol'],c['own_observations']),(c['sector'],c['sector_observations'])):
                if symbol is not None and obs:
                    if symbol in seen:_exact(obs,seen[symbol],'Shared source path')
                    else:seen[symbol]=obs
        out.append(rebuilt)
    return out

def _ranked(features):
    out=deepcopy(features)
    for f in out:
        width=len(f['symbols']);total=[None]*width;tracking=[None]*width
        cells={c['column_index']:c for c in f['cells']};domain=f['supported_indices']
        dt=[cells[j]['statistics']['total_ratio'] for j in domain];dr=[cells[j]['statistics']['tracking_ratio'] for j in domain]
        for j in domain:
            total[j]=rank(cells[j]['statistics']['total_ratio'],dt);tracking[j]=rank(cells[j]['statistics']['tracking_ratio'],dr)
        f.update(total_target=total,tracking_target=tracking)
    return out

def _incumbent(study,arrays,meta,n,validator,expected):
    bridge=old._bridge(study,arrays,meta,n);orders=study.rp.make_orders(bridge['choices'],bridge['post'],_meta(meta,n))
    if expected is not None:need(isinstance(expected,list) and len(expected)>=n,'Expected incumbent prefix');_exact(orders,expected[:n],'Expected incumbent orders')
    gate=_gate(validator,(deepcopy(orders),bridge['post'].copy(),bridge['pre'].copy()),'Incumbent')
    return bridge,orders,gate

def _select(study,arrays,meta,bridge,features,n,orders0,ig,sg):
    width=bridge['width'];cm=_meta(meta,n);ca={k:np.asarray(v)[:n].copy() for k,v in arrays.items()};ca['core_weight']=bridge['cw'].copy()
    orders={ARMS[0]:orders0};scores={ARMS[0]:bridge['post'].copy()};pres={ARMS[0]:bridge['pre'].copy()};choices={ARMS[0]:deepcopy(bridge['choices'])}
    support=(ca['baseline']-ca['core'])/(1.-bridge['cw']);deltas={}
    for arm,key in zip(ARMS[1:],('total_target','tracking_target')):
        target=np.full((n,width),np.nan)
        for i,f in enumerate(features):
            for j,x in enumerate(f[key]):
                if x is not None:target[i,j]=x
        delta=np.zeros((n,width));mask=np.isfinite(target);delta[mask]=.05*(target[mask]-support[mask]);expected=bridge['pre'].copy();nz=delta!=0.;expected[nz]+=delta[nz]
        # Existing239 complete coefficient check, including Local67 extension.
        active=(bridge['calm'] & ~bridge['weak'])[:,None] & ca['extension_available'] & ca['forecast_available'] & bridge['eligible'] & np.isfinite(ca['rank_log_excess'])
        cap=np.minimum(.05,np.maximum(0.,.10-(.025*bridge['calm']+.05*bridge['weak'])))
        for i in range(n):
            for j in range(width):
                weights=[float(bridge['cw'][i,j]),.025*float(bridge['calm'][i]),.05*float(bridge['weak'][i]),.01*float(active[i,j]),float(cap[i]),.05*float(mask[i,j])]
                remaining=1.-weights[0]
                for w in weights[1:]:remaining-=w
                need(remaining>=.33-1e-12 and abs(math.fsum(weights+[remaining])-1.)<=1e-12,'Original complete239 budget including extension')
        calls=[]
        def boundary(changed,good,names,effects,original,policy):
            need(not calls and np.array_equal(changed,expected) and changed.tobytes()==expected.tobytes(),'One funded original episode and exact pre')
            need(np.array_equal(good,bridge['eligible']) and np.array_equal(effects,ca['effects']) and names==cm['symbols'] and policy==old.POLICY,'Original E/effects/policy')
            calls.append(True);pres[arm]=changed.copy();return study.ep.select_boundary(changed,good,names,effects,original,policy)
        chosen,score=old.adapter.score_and_choose('incumbent66',.05,target,ca,cm,n,pre_score=bridge['pre'].copy(),effects=ca['effects'],episode_policy=old.POLICY.copy(),boundary_selector=boundary)
        need(len(calls)==1,'Exactly one episode per arm');scores[arm]=score.copy();choices[arm]=chosen;orders[arm]=study.rp.make_orders(chosen,score,cm);deltas[arm]=delta
    counts=dict.fromkeys(COUNT_FIELDS,0);counts.update(events=n,symbols_per_event=width,original_episode_calls=3);events=[]
    for i,f in enumerate(features):
        eligible=np.flatnonzero(bridge['eligible'][i]).tolist();cells=f['cells'];stats=[c['statistics'] for c in cells if c['statistics'] is not None]
        counts['eligible_cells']+=len(eligible);counts['initial_partial_events']+=int(f['initial_partial']);counts['stock_cells']+=len(cells);counts['mature_cells']+=len(f['mature_indices']);counts['supported_cells']+=len(f['supported_indices'])
        counts['warmup_cells']+=sum(c['history_status']=='official_pre_inception' for c in cells);counts['unmapped_warmup_cells']+=sum(c['history_status']=='official_pre_inception' and c['sector'] is None for c in cells)
        counts['own_price_observations']+=sum(len(c['own_observations']) for c in cells);counts['sector_price_observations']+=sum(len(c['sector_observations']) for c in cells)
        for k in ('stock_return_observations','sector_return_observations','tracking_return_observations'):counts[k]+=63*len(stats)
        counts['total_risk_statistics']+=len(stats);counts['tracking_risk_statistics']+=len(stats)
        counts['zero_total_cells']+=sum(s['total_risk']['sd']==0. for s in stats);counts['zero_tracking_cells']+=sum(s['tracking_risk']['sd']==0. for s in stats)
        for arm,key in zip(ARMS[1:],('total','tracking')):
            counts[key+'_nonzero_deltas']+=int(np.count_nonzero(deltas[arm][i]));counts[key+'_changed_holdings']+=int(set(choices[arm][i])!=set(choices[ARMS[0]][i]))
        counts['candidate_control_changed_holdings']+=int(set(choices[ARMS[1]][i])!=set(choices[ARMS[2]][i]))
        e=dict(query_index=i,date=f['date'],initial_partial=f['initial_partial'],eligible_indices=eligible,supported_indices=f['supported_indices'].copy(),original_support=support[i].tolist(),total_deltas=deltas[ARMS[1]][i].tolist(),tracking_deltas=deltas[ARMS[2]][i].tolist(),choices={arm:[int(j) for j in choices[arm][i]] for arm in ARMS},calm=bool(bridge['calm'][i]),weak=bool(bridge['weak'][i]),core_weight=float(bridge['cw'][i,0]),pre_order=sorted(eligible,key=lambda j:(-bridge['pre'][i,j],cm['symbols'][i][j])))
        need(tuple(e)==DECISION_EVENT_KEYS,'Decision schema');events.append(e)
    return orders,scores,pres,dict(schema=AUDIT_SCHEMA,counts=counts,events=events,incumbent_gate=ig,source_gate=sg)

def prepare(study,arrays,meta,panel,calendar,review,n=None,*,incumbent_validator=None,expected_incumbent_orders=None,source_validator=None):
    n=_n(arrays,n);before=old._snapshot(arrays);bm=_meta(meta,n)
    bridge,orders,ig=_incumbent(study,arrays,meta,n,incumbent_validator,expected_incumbent_orders)
    need(old._snapshot(arrays)==before and _meta(meta,n)==bm,'Incumbent gate mutated originals')
    features=build_features(panel,meta,calendar,arrays,review,n)
    sg=_gate(source_validator,(deepcopy(features),),'Source')
    need(old._snapshot(arrays)==before and _meta(meta,n)==bm,'Source gate mutated originals')
    features=_ranked(features);out=_select(study,arrays,meta,bridge,features,n,orders,ig,sg)
    need(old._snapshot(arrays)==before and _meta(meta,n)==bm,'Original inputs mutated')
    return (features,)+out

def decisions(study,arrays,meta,features,n=None,*,incumbent_validator=None,expected_incumbent_orders=None,source_validator=None):
    n=_n(arrays,n);before=old._snapshot(arrays);bm=_meta(meta,n)
    bridge,orders,ig=_incumbent(study,arrays,meta,n,incumbent_validator,expected_incumbent_orders)
    need(old._snapshot(arrays)==before and _meta(meta,n)==bm,'Incumbent gate mutated originals')
    consumed=_cached(features,meta,n);sg=_gate(source_validator,(deepcopy(consumed),),'Source')
    need(old._snapshot(arrays)==before and _meta(meta,n)==bm,'Source gate mutated originals')
    ranked=_ranked(consumed)
    for actual,wanted in zip(features[:n],ranked):_exact(actual,wanted,'Cached full features')
    out=_select(study,arrays,meta,bridge,ranked,n,orders,ig,sg)
    need(old._snapshot(arrays)==before and _meta(meta,n)==bm,'Original inputs mutated')
    return out


def unranked_prefix(features,n):
    need(type(n) is int and isinstance(features,list) and 0<=n<=len(features),'Feature prefix')
    result=deepcopy(features[:n])
    for f in result:f['total_target']=None;f['tracking_target']=None
    return result
