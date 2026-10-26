"""264 original-U0 gain-age endpoint versus continuous survival full noncalm core.

Only supplied inputs. No price loading, model fitting, registration or replay.
"""
from copy import deepcopy
from numbers import Real
from pathlib import Path
import hashlib
import importlib.util
import math
import numpy as np
import pandas as pd
import gain_survival264_sources as source

ROOT=Path(__file__).resolve().parents[1]
FROZEN={'work/monthly_portfolio_path247_kernel.py':'18c7726ae701bb3557636bfa69a0c0dc32d516286f3bb1cd27f50157d5380092','outputs/monthly_core_timing70_research/core_adapter.py':'b1c45a7e9fc671675b461e6f016eda4e17a7885c1ff74630cadddf9ce8d1544a','outputs/monthly_path70_research/incumbent_adapter.py':'5619bfc3512a0aa057e79042256a379360d6baa1bc0c448f2f301b84d6ec3e2b'}
for _p,_h in FROZEN.items():
    if hashlib.sha256((ROOT/_p).read_bytes()).hexdigest()!=_h:raise ValueError('Frozen dependency changed: '+_p)
def _module(path,name):
    spec=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
old=_module('work/monthly_portfolio_path247_kernel.py','gain264_original247')
core_adapter=_module('outputs/monthly_core_timing70_research/core_adapter.py','gain264_core_adapter')
BOUNDS=source.read_pin('work/peer239_maturity_policy.json')['securities']
ARMS=('incumbent_local67','endpoint_gain_age_control','continuous_gain_survival_candidate')
INITIAL_PARTIAL=source.INITIAL_PARTIAL
SOURCE_SCHEMA=source.SOURCE_SCHEMA
STAT_SCHEMA='monthly_gain_survival264_statistics_v1'
AUDIT_SCHEMA='monthly_gain_survival264_decisions_v1'
EVENT_KEYS=('query_index','date','symbols','initial_partial','calm','weak','core_weight','query_position','next_official_session','clock_sessions','original_price_origin','u0_indices','active_indices','numerator_positions','numerator_dates','current_positions','current_dates','formation_positions','formation_dates','required_positions','required_dates','cells','endpoint_identity','survival_identity','branch','endpoint_target','survival_target')
CELL_KEYS=('column_index','symbol','original_finite','original_raw','original_rank','frozen_core','rebuilt_core','core_abs_error','original_sigma63','history_bound','observations','statistics')
OBS_KEYS=('date','value')
STAT_KEYS=('schema','positive_anchor_indices','positive_gains','potential_ages','survival_ages','first_breach_indices','endpoint_retained','gain_sum','survival_weighted_terms','endpoint_weighted_terms','survival_weighted_sum','endpoint_weighted_sum','normalizer','survival_quality','endpoint_quality','original_raw','survival_raw','endpoint_raw','nonpositive_original_preserved','no_positive_gains','survival_comparisons')
COUNT_FIELDS=('events','symbols_per_event','eligible_cells','initial_partial_events','calm_events','active_events','original_core_cells','original_finite_cells','required_cells','price_observations','gain_return_observations','positive_gain_anchors','survival_comparisons','endpoint_comparisons','gain_statistics','quality_ratio_evaluations','no_positive_gain_cells','nonpositive_original_cells','positive_raw_multiplications','empty_domain_events','available_events','endpoint_identity_events','survival_identity_events','endpoint_ranked_cells','survival_ranked_cells','endpoint_nonzero_deltas','survival_nonzero_deltas','endpoint_changed_holdings','survival_changed_holdings','candidate_control_changed_holdings','original_episode_calls','new_model_fits','predictive_model_fits')
DECISION_EVENT_KEYS=('query_index','date','initial_partial','calm','weak','core_weight','branch','endpoint_identity','survival_identity','eligible_indices','u0_indices','original_support','endpoint_deltas','survival_deltas','choices','pre_order')
need=old.require

def dependency_files():
    return tuple(dict.fromkeys([*(ROOT/p for p in FROZEN),ROOT/'work/gain_survival264_sources.py',*source.dependency_files(),*old.dependency_files(),ROOT/'outputs/monthly_stability_research/stability.py',ROOT/'outputs/monthly_quality_research/explore.py']))

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
    need(math.isfinite(y),label+' finite');need(not(y==0. and x!=0),label+' nonzero conversion underflow');return y

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
    need(b!=0.,label+' nonzero denominator');y=_arith(a/b,label)
    if a!=0. and y==0.:raise ArithmeticError('Underflow '+label)
    return y

def _mul(a,b,label):
    y=_arith(a*b,label)
    if a!=0. and b!=0. and y==0.:raise ArithmeticError('Underflow '+label)
    return y

def measure(prices253,original_raw):
    """Positive simple gains, later-closes-only first breach, and a matched endpoint."""
    need(isinstance(prices253,(list,tuple,np.ndarray)) and len(prices253)==253,'Exactly253 required prices')
    p=[_positive(x,'required price') for x in prices253];raw=_finite(original_raw,'Original raw')
    indices=[];gains=[];ages=[];survived=[];breaches=[];retained=[];st=[];et=[];comparisons=0
    for i in range(1,253):
        gross=_div(p[i],p[i-1],'gain gross');g=_arith(gross-1.,'simple gain')
        if g<=0.:continue
        age=252-i;first=None
        for j in range(i+1,253):
            comparisons+=1
            if p[j]<p[i]:first=j;break
        duration=age if first is None else first-i-1;keep=p[252]>=p[i]
        need(0<=duration<=age<=251,'Integer gain durations')
        indices.append(i);gains.append(g);ages.append(age);survived.append(duration);breaches.append(first);retained.append(keep)
        st.append(_mul(g,float(duration),'survival weighted gain'));et.append(_mul(g,float(age if keep else 0),'endpoint weighted gain'))
    w=_sum(gains,'positive gain sum');ss=_sum(st,'survival weighted sum');es=_sum(et,'endpoint weighted sum')
    if gains:
        normalizer=_mul(252.,w,'gain duration normalizer');qs=_div(ss,normalizer,'survival quality');qe=_div(es,normalizer,'endpoint quality')
        # Integer durations prove the real-valued 251/252 bound. Canonical
        # products/sums/division can round one ULP above that rounded fraction.
        need(0.<=qs<=1. and 0.<=qe<=1.,'Gain quality finite unit range')
    else:normalizer=0.;qs=qe=1.
    preserve=raw<=0.;sr=raw if preserve else _mul(raw,qs,'survival transformed raw');er=raw if preserve else _mul(raw,qe,'endpoint transformed raw')
    result=dict(schema=STAT_SCHEMA,positive_anchor_indices=indices,positive_gains=gains,potential_ages=ages,survival_ages=survived,first_breach_indices=breaches,endpoint_retained=retained,gain_sum=w,survival_weighted_terms=st,endpoint_weighted_terms=et,survival_weighted_sum=ss,endpoint_weighted_sum=es,normalizer=normalizer,survival_quality=qs,endpoint_quality=qe,original_raw=raw,survival_raw=sr,endpoint_raw=er,nonpositive_original_preserved=preserve,no_positive_gains=not gains,survival_comparisons=comparisons)
    need(tuple(result)==STAT_KEYS,'Statistic schema');return result

def rank(value,domain):
    need(bool(domain),'Nonempty rank domain');less=sum(x<value for x in domain);equal=sum(x==value for x in domain)
    return (float(2*less+equal+1)/2.)/float(len(domain))-.5

def _n(arrays,n):
    shape=np.shape(arrays['baseline']);need(len(shape)==2 and shape[1]>0,'Original axes');n=shape[0] if n is None else n;need(type(n) is int and 0<=n<=shape[0],'Canonical prefix0..N');return n

def _meta(meta,n):return {k:deepcopy(meta[k][:n]) for k in ('dates','ends','symbols')}|{'etfs':deepcopy(meta['etfs'])}

def _gate(callback,args,label):
    if callback is None:return dict(passed=True,synthetic_no_callback=True)
    r=callback(*args);need(isinstance(r,dict) and r.get('passed') is True,label+' passed receipt required');return deepcopy(r)

def _state(arrays,n):
    total,width=np.shape(arrays['baseline']);out={}
    for key in ('state_calm10','state_weak_narrow'):
        a=np.asarray(arrays[key]);need(a.shape==(total,) and a.dtype.kind=='b','Boolean state');out[key]=a[:n].copy()
    raw=np.asarray(arrays['core_weight']);need(raw.shape in ((total,),(total,1),(total,width)),'Core weight axes');cw=np.asarray(raw[:n],float);cw=cw[:,None] if cw.ndim==1 else cw;cw=np.broadcast_to(cw,(n,width)).copy();need(np.array_equal(cw,np.broadcast_to(np.where(out['state_calm10'],.5,.52)[:,None],(n,width))),'Original weights')
    return out['state_calm10'],out['state_weak_narrow'],cw

def _bound(symbol,day):return deepcopy(next((b for b in BOUNDS if b['symbol']==symbol and b['available_from']<=day),None))

def _setup(panel,meta,calendar,arrays,review,n):
    need(isinstance(panel,dict) and isinstance(panel.get('adj'),pd.DataFrame),'Adjusted source frame');frame=panel['adj']
    need(isinstance(frame.index,pd.DatetimeIndex) and frame.index.tz is None and frame.index.is_unique and frame.index.is_monotonic_increasing and not frame.index.hasnans and all(x==x.normalize() for x in frame.index),'Source date axes')
    need(frame.columns.is_unique and all(type(s) is str and s for s in frame.columns),'Source symbol axes')
    ss=source.calendar_days(calendar);width=np.shape(arrays['baseline'])[1]
    need(isinstance(review,dict) and review.get('schema')==SOURCE_SCHEMA and review.get('passed') is True and review.get('metadata_only') is True and review.get('source_admitted') is False and review.get('strict_pit') is False and review.get('panel_first_session')==source.ORIGIN,'Metadata-only source contract')
    _exact(review['listing_bounds'],BOUNDS,'Original listing bounds');need(isinstance(review['clock_events'],list) and len(review['clock_events'])>=n,'Metadata clocks')
    clocks=[]
    for q in range(n):
        names=meta['symbols'][q];need(isinstance(names,list) and len(names)==width and len(set(names))==width and all(type(s) is str and s for s in names),'Original names');day=old._iso(meta['dates'][q],'query');need(q==0 or day>meta['dates'][q-1],'Chronological query');need(day!=INITIAL_PARTIAL or q==0,'Named partial first')
        c=source.clock(q,day,names,ss);_exact(review['clock_events'][q],c,'Metadata official clock');clocks.append(c)
    got=[]
    for x in review['identity_cells']:
        need(isinstance(x,dict) and type(x.get('query_index')) is int and x['query_index']>=0,'Identity query index')
        if x['query_index']<n:got.append(deepcopy(x))
    _exact(got,source.identities(meta,BOUNDS,n),'Complete consumed whole issuer identities')
    return frame,ss,clocks

def _old_number(x):
    if x is None or x is pd.NA:return np.nan
    need(isinstance(x,Real) and not isinstance(x,(bool,np.bool_)),'Original history real nonbool')
    try:return float(x)
    except (ValueError,OverflowError,TypeError) as exc:raise ValueError('Original history binary64') from exc

def _old_data(frame,meta,sessions,clocks):
    formal=[c for c in clocks if not c['initial_partial']]
    if not formal:return None
    end=formal[-1]['date'];past=frame.loc[:end];need(source.ORIGIN in sessions,'Original official origin');start=sessions.index(source.ORIGIN);last=formal[-1]['query_position']
    need(past.index.strftime('%Y-%m-%d').tolist()==sessions[start:last+1],'Complete original official history from2008')
    names=sorted({s for c in formal for s in c['symbols']});need(set(names)<=set(past.columns),'Original source missing column');px=pd.DataFrame(index=past.index)
    for s in names:px[s]=np.asarray([_old_number(x) for x in past[s]],float)
    with np.errstate(all='ignore'):
        sigma=px.pct_change(fill_method=None).rolling(63).std();raw210=px.shift(42)/px.shift(252)-1.;raw252=(px.shift(42)/px.shift(294)-1.)/sigma.replace(0,np.nan)
    return sigma,raw210,raw252

def _old_cells(data,clock,arrays,q,calm,cw):
    if clock['initial_partial']:return []
    sigma,r210,r252=data;names=clock['symbols'];day=clock['date'];raw=(r210 if calm else r252).loc[day,names].to_numpy(float);raw[~np.isfinite(raw)]=np.nan
    ranked=pd.Series(raw).rank(pct=True).fillna(.5).to_numpy()-.5;core=np.asarray(arrays['core'])[q];out=[]
    for j,s in enumerate(names):
        frozen=_finite(core[j],'Original core');rebuilt=float(cw[j]*ranked[j]);err=abs(rebuilt-frozen);need(err<=1e-12,'Original weighted core parity');sv=float(sigma.at[pd.Timestamp(day),s]);sv=sv if math.isfinite(sv) else None
        out.append(dict(column_index=j,symbol=s,original_finite=bool(np.isfinite(raw[j])),original_raw=float(raw[j]) if np.isfinite(raw[j]) else None,original_rank=float(ranked[j]),frozen_core=frozen,rebuilt_core=rebuilt,core_abs_error=err,original_sigma63=sv,history_bound=_bound(s,day),observations=[],statistics=None))
    return out

def _scan(frame,symbol,dates,bound):
    out=[]
    for d in dates:
        need(symbol in frame.columns and pd.Timestamp(d) in frame.index,'Missing required source axis '+symbol+'/'+d);v=_positive(frame.at[pd.Timestamp(d),symbol],'Required '+symbol+'/'+d)
        need(bound is None or d>=bound['not_before'],'Positive prelisting required source '+symbol+'/'+d);out.append(dict(date=d,value=v))
    return out

def _finish(cells,partial,calm,u0):
    active=[] if partial or calm else u0.copy()
    branch='initial_partial' if partial else 'calm' if calm else 'empty_domain' if not active else 'available'
    identities=[branch=='available' and all(cells[j]['statistics'][key+'_raw'].hex()==cells[j]['original_raw'].hex() for j in active) for key in ('endpoint','survival')]
    return active,*identities,branch

def build_features(panel,meta,calendar,arrays,review,n=None):
    n=_n(arrays,n);calm,weak,cw=_state(arrays,n);frame,sessions,clocks=_setup(panel,meta,calendar,arrays,review,n);data=_old_data(frame,meta,sessions,clocks);out=[]
    for q,c in enumerate(clocks):
        partial=c['initial_partial'];cells=_old_cells(data,c,arrays,q,bool(calm[q]),cw[q]);u0=[j for j,x in enumerate(cells) if x['original_finite']];active=[] if partial or calm[q] else u0.copy()
        # Complete source scan across the full domain before any new gain arithmetic.
        for j in active:cells[j]['observations']=_scan(frame,cells[j]['symbol'],c['required_dates'],cells[j]['history_bound'])
        for j in active:cells[j]['statistics']=measure([x['value'] for x in cells[j]['observations']],cells[j]['original_raw'])
        active,ei,si,branch=_finish(cells,partial,bool(calm[q]),u0);t=c['query_position']
        e=dict(query_index=q,date=c['date'],symbols=deepcopy(c['symbols']),initial_partial=partial,calm=bool(calm[q]),weak=bool(weak[q]),core_weight=float(cw[q,0]),query_position=t,next_official_session=c['next_official_session'],clock_sessions=[] if partial else sessions[t-294:t+2],original_price_origin=source.ORIGIN,u0_indices=u0,active_indices=active)
        for key in ('numerator_positions','numerator_dates','current_positions','current_dates','formation_positions','formation_dates','required_positions','required_dates'):e[key]=deepcopy(c[key])
        e.update(cells=cells,endpoint_identity=ei,survival_identity=si,branch=branch,endpoint_target=None,survival_target=None);need(tuple(e)==EVENT_KEYS,'Event schema');out.append(e)
    return out

def _cached(features,meta,arrays,n):
    need(isinstance(features,list) and len(features)>=n,'Cached feature prefix');calm,weak,cw=_state(arrays,n);out=[];width=np.shape(arrays['baseline'])[1]
    for q,f in enumerate(features[:n]):
        need(isinstance(f,dict) and set(f)==set(EVENT_KEYS),'Cached event schema');day=meta['dates'][q];partial=day==INITIAL_PARTIAL;t=f['query_position'];need(type(t) is int and t>=0,'Cached official position')
        for key,v in dict(query_index=q,date=day,symbols=meta['symbols'][q],initial_partial=partial,calm=bool(calm[q]),weak=bool(weak[q]),core_weight=float(cw[q,0]),original_price_origin=source.ORIGIN).items():_exact(f[key],v,'Cached identity/state '+key)
        if partial:
            need(q==0 and f['clock_sessions']==[] and f['cells']==[],'Partial no new consumption');clock={k:[] for k in ('numerator_positions','numerator_dates','current_positions','current_dates','formation_positions','formation_dates','required_positions','required_dates')}
        else:
            cs=source.calendar_days(f['clock_sessions']);need(len(cs)==296 and cs[294]==day and cs[295]==f['next_official_session'] and cs[295][:7]!=day[:7] and t>=294,'Cached official296 clock')
            ep=[t-294,t-42];cp=list(range(t-63,t+1));fp=list(range(t-294,t-41));ds=lambda ps:[cs[p-(t-294)] for p in ps];clock=dict(numerator_positions=ep,numerator_dates=ds(ep),current_positions=cp,current_dates=ds(cp),formation_positions=fp,formation_dates=ds(fp),required_positions=fp,required_dates=ds(fp))
        for k,v in clock.items():_exact(f[k],v,'Cached clock '+k)
        need(isinstance(f['cells'],list) and len(f['cells'])==(0 if partial else width),'Cached full49 evidence');cells=deepcopy(f['cells']);u0=[]
        for j,c in enumerate(cells):
            need(isinstance(c,dict) and set(c)==set(CELL_KEYS),'Cached cell schema');need(c['column_index']==j and c['symbol']==meta['symbols'][q][j] and type(c['original_finite']) is bool,'Cached cell axes')
            if c['original_finite']:_finite(c['original_raw'],'Cached original raw');u0.append(j)
            else:need(c['original_raw'] is None,'Unavailable original raw')
            sv=c['original_sigma63'];need(sv is None or _finite(sv,'Original sigma')>=0.,'Original sigma evidence');_exact(c['history_bound'],_bound(c['symbol'],day),'Cached bound')
        vals=[c['original_raw'] if c['original_finite'] else np.nan for c in cells];ranks=pd.Series(vals,dtype=float).rank(pct=True).fillna(.5).to_numpy()-.5
        active=[] if partial or calm[q] else u0.copy()
        for j,c in enumerate(cells):
            r=float(ranks[j]);frozen=_finite(np.asarray(arrays['core'])[q,j],'Original core');rebuilt=float(cw[q,j]*r);error=abs(rebuilt-frozen);need(error<=1e-12,'Cached original core parity')
            for k,v in dict(original_rank=r,frozen_core=frozen,rebuilt_core=rebuilt,core_abs_error=error).items():_exact(c[k],v,'Cached original '+k)
            obs=c['observations'];need(isinstance(obs,list) and len(obs)==(253 if j in active else 0),'Cached required253 domain')
            for o,d in zip(obs,clock['required_dates']):
                need(isinstance(o,dict) and set(o)==set(OBS_KEYS) and o['date']==d and type(o['value']) is float,'Cached observations');_positive(o['value'],'Cached required price');need(c['history_bound'] is None or d>=c['history_bound']['not_before'],'Cached prelisting conflict')
            if j not in active:need(c['statistics'] is None,'No new statistics outside activeU0')
        # Same all-source-before-arithmetic order as physical preparation.
        for j in active:cells[j]['statistics']=measure([o['value'] for o in cells[j]['observations']],cells[j]['original_raw'])
        active,ei,si,branch=_finish(cells,partial,bool(calm[q]),u0)
        for j in active:_exact(f['cells'][j]['statistics'],cells[j]['statistics'],'Cached canonical statistic')
        rebuilt=deepcopy(f);rebuilt.update(cells=cells,u0_indices=u0,active_indices=active,endpoint_identity=ei,survival_identity=si,branch=branch,endpoint_target=None,survival_target=None)
        for k in ('u0_indices','active_indices','endpoint_identity','survival_identity','branch'):_exact(f[k],rebuilt[k],'Cached domain/branch '+k)
        out.append(rebuilt)
    return out

def _ranked(features):
    out=deepcopy(features)
    for f in out:
        for key in ('endpoint','survival'):
            target=[None]*len(f['symbols'])
            if f['branch']=='available':
                domain=f['active_indices'];cells=f['cells'];values=[cells[j]['statistics'][key+'_raw'] for j in domain]
                for j in domain:target[j]=rank(cells[j]['statistics'][key+'_raw'],values)
            f[key+'_target']=target
    return out

def _incumbent(study,arrays,meta,n,validator,expected):
    bridge=old._bridge(study,arrays,meta,n);orders=study.rp.make_orders(bridge['choices'],bridge['post'],_meta(meta,n))
    if expected is not None:need(isinstance(expected,list) and len(expected)>=n,'Expected incumbent prefix');_exact(orders,expected[:n],'Expected incumbent orders')
    gate=_gate(validator,(deepcopy(orders),bridge['post'].copy(),bridge['pre'].copy()),'Incumbent');return bridge,orders,gate

def _select(study,arrays,meta,bridge,features,n,orders0,ig,sg):
    width=bridge['width'];cm=_meta(meta,n);ca={k:np.asarray(v)[:n].copy() for k,v in arrays.items()};ca['core_weight']=bridge['cw'].copy();support=(ca['baseline']-ca['core'])/(1.-bridge['cw'])
    orders={ARMS[0]:orders0};scores={ARMS[0]:bridge['post'].copy()};pres={ARMS[0]:bridge['pre'].copy()};choices={ARMS[0]:deepcopy(bridge['choices'])};deltas={}
    for arm,key in zip(ARMS[1:],('endpoint_target','survival_target')):
        target=np.full((n,width),np.nan)
        for i,f in enumerate(features):
            if f[key.replace('_target','_identity')]:continue
            for j,x in enumerate(f[key]):
                if x is not None:target[i,j]=x
        delta=np.zeros((n,width));mask=np.isfinite(target);delta[mask]=bridge['cw'][mask]*target[mask]-bridge['core'][mask];expected=bridge['pre'].copy();nz=delta!=0.;expected[nz]+=delta[nz];calls=[]
        def boundary(changed,good,names,effects,original,policy):
            need(not calls and changed.tobytes()==expected.tobytes(),'One fullcore original episode exact pre');need(np.array_equal(good,bridge['eligible']) and np.array_equal(effects,ca['effects']) and names==cm['symbols'] and policy==old.POLICY,'Original E/effects/policy')
            for k in ('baseline','core','core_weight'):need(np.array_equal(original[k],ca[k]),'Original support '+k)
            calls.append(True);pres[arm]=changed.copy();return study.ep.select_boundary(changed,good,names,effects,original,policy)
        chosen,score=core_adapter.score_and_choose('incumbent66',1.,target,ca,cm,n,pre_score=bridge['pre'].copy(),effects=ca['effects'],episode_policy=old.POLICY.copy(),boundary_selector=boundary)
        need(len(calls)==1,'Exactly one episode per arm')
        copied=[]
        for i,f in enumerate(features):
            if f['branch']!='available' or f[key.replace('_target','_identity')]:
                pres[arm][i]=bridge['pre'][i];score[i]=bridge['post'][i];chosen[i]=deepcopy(bridge['choices'][i]);copied.append(i)
        scores[arm]=score.copy();choices[arm]=chosen;orders[arm]=study.rp.make_orders(chosen,score,cm)
        for i in copied:orders[arm][i]=deepcopy(orders0[i])
        deltas[arm]=delta
    counts=dict.fromkeys(COUNT_FIELDS,0);counts.update(events=n,symbols_per_event=width,original_episode_calls=3);events=[]
    for i,f in enumerate(features):
        eligible=np.flatnonzero(bridge['eligible'][i]).tolist();cells=f['cells'];active=f['active_indices'];branch=f['branch'];partial=f['initial_partial']
        counts['eligible_cells']+=len(eligible);counts['initial_partial_events']+=int(partial);counts['calm_events']+=int(not partial and f['calm']);counts['active_events']+=int(not partial and not f['calm']);counts['original_core_cells']+=len(cells);counts['original_finite_cells']+=len(f['u0_indices']);counts['required_cells']+=len(active);counts['price_observations']+=253*len(active);counts['gain_return_observations']+=252*len(active);counts['gain_statistics']+=len(active)
        for j in active:
            stat=cells[j]['statistics'];anchors=len(stat['positive_gains']);counts['positive_gain_anchors']+=anchors;counts['survival_comparisons']+=stat['survival_comparisons'];counts['endpoint_comparisons']+=anchors;counts['quality_ratio_evaluations']+=2*int(anchors>0);counts['no_positive_gain_cells']+=int(stat['no_positive_gains']);counts['nonpositive_original_cells']+=int(stat['nonpositive_original_preserved']);counts['positive_raw_multiplications']+=2*int(not stat['nonpositive_original_preserved'])
        counts['empty_domain_events']+=int(branch=='empty_domain');counts['available_events']+=int(branch=='available')
        for arm,key in zip(ARMS[1:],('endpoint','survival')):
            counts[key+'_identity_events']+=int(f[key+'_identity']);counts[key+'_ranked_cells']+=len(active) if branch=='available' else 0;counts[key+'_nonzero_deltas']+=int(np.count_nonzero(deltas[arm][i]));counts[key+'_changed_holdings']+=int(set(choices[arm][i])!=set(choices[ARMS[0]][i]))
        counts['candidate_control_changed_holdings']+=int(set(choices[ARMS[1]][i])!=set(choices[ARMS[2]][i]))
        e=dict(query_index=i,date=f['date'],initial_partial=partial,calm=f['calm'],weak=f['weak'],core_weight=f['core_weight'],branch=branch,endpoint_identity=f['endpoint_identity'],survival_identity=f['survival_identity'],eligible_indices=eligible,u0_indices=f['u0_indices'].copy(),original_support=support[i].tolist(),endpoint_deltas=deltas[ARMS[1]][i].tolist(),survival_deltas=deltas[ARMS[2]][i].tolist(),choices={arm:[int(j) for j in choices[arm][i]] for arm in ARMS},pre_order=sorted(eligible,key=lambda j:(-bridge['pre'][i,j],cm['symbols'][i][j])))
        need(tuple(e)==DECISION_EVENT_KEYS,'Decision schema');events.append(e)
    return orders,scores,pres,dict(schema=AUDIT_SCHEMA,counts=counts,events=events,incumbent_gate=ig,source_gate=sg)

def prepare(study,arrays,meta,panel,calendar,review,n=None,*,incumbent_validator=None,expected_incumbent_orders=None,source_validator=None):
    n=_n(arrays,n);before=old._snapshot(arrays);bm=_meta(meta,n);bridge,orders,ig=_incumbent(study,arrays,meta,n,incumbent_validator,expected_incumbent_orders);need(old._snapshot(arrays)==before and _meta(meta,n)==bm,'Incumbent gate mutated originals')
    features=build_features(panel,meta,calendar,arrays,review,n);sg=_gate(source_validator,(deepcopy(features),),'Source');need(old._snapshot(arrays)==before and _meta(meta,n)==bm,'Source gate mutated originals');features=_ranked(features);out=_select(study,arrays,meta,bridge,features,n,orders,ig,sg);need(old._snapshot(arrays)==before and _meta(meta,n)==bm,'Original inputs mutated');return (features,)+out

def decisions(study,arrays,meta,features,n=None,*,incumbent_validator=None,expected_incumbent_orders=None,source_validator=None):
    n=_n(arrays,n);before=old._snapshot(arrays);bm=_meta(meta,n);bridge,orders,ig=_incumbent(study,arrays,meta,n,incumbent_validator,expected_incumbent_orders);need(old._snapshot(arrays)==before and _meta(meta,n)==bm,'Incumbent gate mutated originals')
    consumed=_cached(features,meta,arrays,n);sg=_gate(source_validator,(deepcopy(consumed),),'Source');need(old._snapshot(arrays)==before and _meta(meta,n)==bm,'Source gate mutated originals');ranked=_ranked(consumed)
    for actual,wanted in zip(features[:n],ranked):_exact(actual,wanted,'Cached full features')
    out=_select(study,arrays,meta,bridge,ranked,n,orders,ig,sg);need(old._snapshot(arrays)==before and _meta(meta,n)==bm,'Original inputs mutated');return out

def unranked_prefix(features,n):
    need(type(n) is int and isinstance(features,list) and 0<=n<=len(features),'Feature prefix');out=deepcopy(features[:n])
    for f in out:f['endpoint_target']=None;f['survival_target']=None
    return out
