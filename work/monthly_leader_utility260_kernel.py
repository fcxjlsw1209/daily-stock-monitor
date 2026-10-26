"""Pure paired clipped-label utility router; caller supplies all source data.

No loader, registry, original-model refit, or performance calculation. Canonical
annual arithmetic and every new decision follow the prospective 260 contract.
"""
from copy import deepcopy
from datetime import date
import hashlib
import importlib.util
from numbers import Real
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
BASE_PATH=ROOT/'work/monthly_local_distance256_kernel.py'
BASE_SHA='97ada679902942f31e4881354fcf95b1fccd2119f77c09760904b8c328364e30'
if hashlib.sha256(BASE_PATH.read_bytes()).hexdigest()!=BASE_SHA:raise ValueError('Changed frozen original bridge')
_spec=importlib.util.spec_from_file_location('leader260_original_bridge',BASE_PATH)
base=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(base)
bridge,require=base.bridge,base.require
INITIAL_PARTIAL='2016-08-17'
ARMS=('incumbent_local67','unconditional_leader_utility','state_leader_utility')
POLICY=dict(budget=.05,ceiling=5,gap=.02)
COLUMNS=('spy_vol63','spy_mom252_skip42','xly_xlp_relative126')
INPUT_KEYS=('schema','n','width','training_row_ids','dates','ends','symbols','offsets','plain','eligible','common_good','labels','event_dates','event_symbols','event_plain')
TARGET_KEYS=('schema','dates','ends','valid','values','records','counts')
TARGET_RECORD_KEYS=('month_index','signal_date','label_end','row_indices','domain_indices','raw_choices','risk_choices','selected_union','raw_only','risk_only','k','valid','reason','value')
TARGET_COUNT_FIELDS=('months','source_rows','domain_rows','cash_months','identical_set_months','unavailable_months','valid_months','selected_label_observations','nonzero_targets')
SOURCE_KEYS=('schema','columns','training','queries','counts')
SOURCE_EVENT_KEYS=('date','initial_partial','spy_start','relative_start','price_end','spy_price_count','xly_price_count','xlp_price_count','values')
SOURCE_COUNT_FIELDS=('training_events','query_events','partial_events','state_vectors','price_observations')
ROUTER_KEYS=('schema','n','event_dates','active','intercept_prediction','state_prediction','intercept_p','state_p','audit','control_gate')
BUNDLE_KEYS=ROUTER_KEYS
MODEL_AUDIT_KEYS=('schema','annual','counts','market_columns','original_models_refitted')
ANNUAL_KEYS=('year','cutoff','lower_bound','training_indices','query_indices','status','weights','weight_sum','mean_x','sd_x','scale_x','mean_d','target_sd','gram','rhs','beta','fit_performed')
MODEL_COUNT_FIELDS=('years','training_months','valid_target_months','annual_training_month_observations','intercept_estimates','ridge_solves','active_query_events','cold_query_events','partial_events','original_model_fits')
AUDIT_KEYS=('schema','counts','events','incumbent_gate')
COUNT_FIELDS=('events','symbols_per_event','partial_events','active_events','cold_events','eligible_cells','intercept_changed_delta_cells','state_changed_delta_cells','intercept_changed_holdings_events','state_changed_holdings_events','state_vs_intercept_changed_holdings_events','incumbent_episode_calls','intercept_episode_calls','state_episode_calls','new_model_fits')
EVENT_KEYS=('query_index','date','partial','active','eligible_indices','intercept_p','state_p','intercept_delta','state_delta','incumbent_choices','intercept_choices','state_choices')


def dependency_files():
    return tuple(dict.fromkeys((BASE_PATH,)+tuple(base.dependency_files())))


def _schema(name):return 'monthly_leader_utility260_'+name+'_v1'

def _days(values,name):return base._days(values,name)

def _gate(callback,*args):
    if callback is None:return dict(passed=True,scope='synthetic_no_callback')
    require(callable(callback),'Gate callable')
    result=callback(*args)
    require(type(result) is dict and result.get('passed') is True,'Independent gate failed')
    return deepcopy(result)


def router_inputs(train,combined_meta,arrays,n=None):
    require(type(train) is dict and type(combined_meta) is dict,'Input mappings')
    m,tm=combined_meta['monthly'],combined_meta['training']
    shape=np.shape(arrays['baseline']);require(len(shape)==2 and shape[1]>0,'Original shape')
    total,width=shape;n=total if n is None else n
    require(type(n) is int and 0<=n<=total,'Query prefix')
    require(all(isinstance(m.get(k),list) and len(m[k])==total for k in ('dates','ends','symbols')),'Monthly axes')
    qdates=_days(m['dates'][:n],'Query dates');qends=_days(m['ends'][:n],'Query ends')
    require(np.all(qends>qdates) and (n<2 or np.all(qdates[1:]>qdates[:-1])),'Query chronology')
    event_symbols=deepcopy(m['symbols'][:n])
    require(all(isinstance(s,list) and len(s)==width and len(set(s))==width and all(type(x) is str and x for x in s) for s in event_symbols),'Query identities')
    require(all(isinstance(tm.get(k),list) for k in ('dates','ends','symbols')) and len(tm['dates'])==len(tm['ends'])==len(tm['symbols']),'Training metadata axes')
    td=_days(tm['dates'],'Training metadata dates');te=_days(tm['ends'],'Training metadata ends')
    offsets=np.asarray(train['offsets'])
    require(offsets.shape==(len(td)+1,) and offsets.dtype.kind in 'iu' and offsets[0]==0 and np.all(np.diff(offsets)>0),'Training offset headers')
    sizes=np.diff(offsets).astype(int)
    require(len(td)<2 or np.all(td[1:]>td[:-1]),'Training chronology')
    require(len(set(td.astype('datetime64[M]').tolist()))==len(td),'One training signal per month')
    require(np.all(te>td) and np.all(te.astype('datetime64[M]')==td.astype('datetime64[M]')+1),'Training next-month maturity')
    offsets=np.r_[0,np.cumsum(sizes)];rows=int(offsets[-1])
    require(np.array_equal(np.asarray(train['offsets']),offsets),'Training offsets parity')
    require(np.array_equal(_days(train['dates'],'Training dates'),np.repeat(td,sizes)) and np.array_equal(_days(train['ends'],'Training ends'),np.repeat(te,sizes)),'Training date identity')
    for key,expected in (('train_plain',(rows,10)),('train_eligible',(rows,)),('common_good',(rows,)),('target_simple_excess',(rows,)),('event_plain',(total*width,10)),('event_dates',(total*width,))):
        require(np.shape(train[key])==expected,'Source axes '+key)
    require(np.array_equal(_days(np.asarray(train['event_dates'])[:n*width],'Event dates'),np.repeat(qdates,width)),'Query date parity')
    require(np.shape(arrays['eligible'])==shape and np.asarray(arrays['eligible']).dtype==np.dtype(bool),'Original eligibility')
    ordinary=qdates[qdates!=np.datetime64(INITIAL_PARTIAL)]
    cutoff=np.datetime64('%04d-12-31'%(int(str(ordinary[-1])[:4])-1)) if len(ordinary) else None
    month_keep=te<=cutoff if cutoff is not None else np.zeros(len(td),bool)
    require(not np.any(month_keep[1:] & ~month_keep[:-1]),'Training maturity prefix')
    count=int(month_keep.sum());stop=int(offsets[count]);ids=np.arange(stop,dtype=int)
    # Cut before numeric conversion. Future targets may be arbitrary objects.
    plain=np.asarray(np.asarray(train['train_plain'])[:stop],dtype=float).copy()
    eligible=base._bool(np.asarray(train['train_eligible'])[:stop],stop,'Training eligibility')
    good=base._bool(np.asarray(train['common_good'])[:stop],stop,'Common validity')
    labels=np.asarray(train['target_simple_excess'])[:stop].copy()
    require(all(isinstance(s,list) and len(s)==sizes[i] and len(set(s))==len(s) for i,s in enumerate(tm['symbols'][:count])),'Consumed training identity widths')
    symbols=np.asarray([str(s) for group in tm['symbols'][:count] for s in group],dtype=str)
    require(all(type(s) is str and s and s.strip()==s for group in tm['symbols'][:count] for s in group),'Training symbols')
    query=np.asarray(train['event_plain'])[:n*width].copy()
    return dict(schema=_schema('inputs'),n=n,width=width,training_row_ids=ids,
        dates=np.repeat(td[:count],sizes[:count]),ends=np.repeat(te[:count],sizes[:count]),
        symbols=symbols,offsets=offsets[:count+1].copy(),plain=plain,eligible=eligible,
        common_good=good,labels=labels,event_dates=qdates,event_symbols=event_symbols,event_plain=query)


def build_expert_targets(inp):
    require(type(inp) is dict and set(inp)==set(INPUT_KEYS) and inp['schema']==_schema('inputs'),'Input schema')
    X=np.asarray(inp['plain']);rows=len(inp['dates'])
    require(X.shape==(rows,10) and X.dtype==np.dtype(float),'Training plain10')
    E=base._bool(inp['eligible'],rows,'Teacher eligibility');good=base._bool(inp['common_good'],rows,'Teacher common validity')
    labels=np.asarray(inp['labels']);require(labels.shape==(rows,),'Teacher label shape')
    offsets=np.asarray(inp['offsets']);require(offsets.ndim==1 and len(offsets)>0 and offsets[0]==0 and offsets[-1]==rows and (len(offsets)==1 or np.all(np.diff(offsets)>0)),'Teacher offsets')
    records=[];dates=[];ends=[];values=[];valids=[];counts=dict.fromkeys(TARGET_COUNT_FIELDS,0)
    for i,(start,stop) in enumerate(zip(offsets[:-1],offsets[1:])):
        ids=list(range(int(start),int(stop)));names=[str(x) for x in inp['symbols'][start:stop]]
        require(len(names)==len(set(names)),'Unique monthly identities')
        domain=[j for j in ids if E[j] and np.isfinite(X[j]).all()]
        k=min(3,len(domain)) if len(domain)>=2 else 0
        raw=sorted(domain,key=lambda j:(-X[j,0],str(inp['symbols'][j])))[:k]
        risk=sorted(domain,key=lambda j:(-X[j,1],str(inp['symbols'][j])))[:k]
        byname=lambda js:sorted(js,key=lambda j:str(inp['symbols'][j]))
        union=byname(set(raw)|set(risk));positive=byname(set(raw)-set(risk));negative=byname(set(risk)-set(raw))
        ok=True;v=0.;reason='cash';checked={}
        if k:
            for j in union:
                x=labels[j]
                require(isinstance(x,Real) and not isinstance(x,(bool,np.bool_)),'Selected label real nonboolean')
                try:value=float(x)
                except (OverflowError,ValueError,TypeError):value=np.nan
                checked[j]=value
                if not good[j] or not np.isfinite(value):ok=False
            counts['selected_label_observations']+=len(union)
            if not ok:v=np.nan;reason='selected_label_unavailable'
            elif set(raw)==set(risk):reason='identical_sets'
            else:
                with np.errstate(over='raise',invalid='raise'):
                    v=float((np.sum(np.asarray([checked[j] for j in positive],float))-np.sum(np.asarray([checked[j] for j in negative],float)))/k)
                require(np.isfinite(v),'Finite paired target');reason='available'
        signal=str(inp['dates'][start]);end=str(inp['ends'][start])
        require(all(str(inp['dates'][j])==signal and str(inp['ends'][j])==end for j in ids),'Monthly source identity')
        row=dict(month_index=i,signal_date=signal,label_end=end,
            row_indices=[int(inp['training_row_ids'][j]) for j in ids],domain_indices=[int(inp['training_row_ids'][j]) for j in domain],
            raw_choices=[str(inp['symbols'][j]) for j in raw],risk_choices=[str(inp['symbols'][j]) for j in risk],
            selected_union=[str(inp['symbols'][j]) for j in union],raw_only=[str(inp['symbols'][j]) for j in positive],risk_only=[str(inp['symbols'][j]) for j in negative],
            k=k,valid=bool(ok),reason=reason,value=float(v) if ok else None)
        records.append(row);dates.append(signal);ends.append(end);values.append(v);valids.append(ok)
        counts['domain_rows']+=len(domain);counts['cash_months']+=int(reason=='cash');counts['identical_set_months']+=int(reason=='identical_sets')
        counts['unavailable_months']+=int(not ok);counts['valid_months']+=int(ok);counts['nonzero_targets']+=int(ok and v!=0.)
    counts.update(months=len(records),source_rows=rows)
    return dict(schema=_schema('targets'),dates=np.asarray(dates,dtype='datetime64[D]'),ends=np.asarray(ends,dtype='datetime64[D]'),
        valid=np.asarray(valids,bool),values=np.asarray(values,float),records=records,counts=counts)


def build_market_inputs(adj,calendar,training_signal_dates,query_dates):
    require(isinstance(adj,pd.DataFrame),'Adjusted price frame')
    require(isinstance(adj.index,pd.DatetimeIndex) and adj.index.tz is None and adj.index.is_unique and adj.index.is_monotonic_increasing and adj.columns.is_unique,'Source axes')
    require(np.all(adj.index==adj.index.normalize()),'Daily source dates')
    cal=_days(calendar,'Official calendar');require(len(cal)>0 and np.all(cal[1:]>cal[:-1]),'Official calendar order')
    train_days=_days(training_signal_dates,'Training state dates');query_days=_days(query_dates,'Query state dates')
    pos={str(d):i for i,d in enumerate(cal)}
    def state(day,allow_partial):
        s=str(day);partial=allow_partial and s==INITIAL_PARTIAL
        if partial:return np.zeros(3),dict(date=s,initial_partial=True,spy_start=None,relative_start=None,price_end=None,spy_price_count=0,xly_price_count=0,xlp_price_count=0,values=[0.,0.,0.])
        require(s in pos and pos[s]>=252,'Complete official state history')
        t=pos[s]
        def prices(symbol,start):
            require(symbol in adj.columns,'Missing state column '+symbol)
            days=pd.DatetimeIndex(cal[start:t+1]);require(days.isin(adj.index).all(),'Missing official state session '+symbol)
            raw=adj.loc[days,symbol].to_numpy();out=np.empty(len(raw),float)
            for j,x in enumerate(raw):
                require(isinstance(x,Real) and not isinstance(x,(bool,np.bool_)),'State price real nonboolean '+symbol)
                try:out[j]=float(x)
                except (OverflowError,ValueError,TypeError) as exc:raise ValueError('State price binary64') from exc
                require(np.isfinite(out[j]) and out[j]>0.,'Positive finite state price '+symbol)
            return out
        spy=prices('SPY',t-252);xly=prices('XLY',t-126);xlp=prices('XLP',t-126)
        with np.errstate(over='raise',invalid='raise',divide='raise',under='ignore'):
            returns=spy[-63:]/spy[-64:-1]-1.
            v=np.asarray([np.std(returns,ddof=1)*np.sqrt(252.),spy[-43]/spy[0]-1.,(xly[-1]/xly[0])/(xlp[-1]/xlp[0])-1.],float)
        require(np.isfinite(v).all(),'Finite market states')
        return v,dict(date=s,initial_partial=False,spy_start=str(cal[t-252]),relative_start=str(cal[t-126]),price_end=s,spy_price_count=253,xly_price_count=127,xlp_price_count=127,values=v.tolist())
    tx=[];qx=[];tr=[];qr=[]
    for d in train_days:v,r=state(d,False);tx.append(v);tr.append(r)
    for d in query_days:v,r=state(d,True);qx.append(v);qr.append(r)
    partial=sum(r['initial_partial'] for r in qr);vectors=len(tr)+len(qr)-partial
    counts=dict(training_events=len(tr),query_events=len(qr),partial_events=partial,state_vectors=vectors,price_observations=507*vectors)
    return np.asarray(tx,float).reshape(-1,3),np.asarray(qx,float).reshape(-1,3),dict(schema=_schema('market_source'),columns=list(COLUMNS),training=tr,queries=qr,counts=counts)


def fit_router(target_bundle,train_X,query_X,event_dates,*,control_validator=None):
    receipt=_gate(control_validator)
    require(type(target_bundle) is dict and set(target_bundle)==set(TARGET_KEYS) and target_bundle['schema']==_schema('targets'),'Target schema')
    dates=_days(target_bundle['dates'],'Target dates');ends=_days(target_bundle['ends'],'Target ends');T=len(dates)
    require(len(ends)==T and (T<2 or np.all(dates[1:]>dates[:-1])) and np.all(ends>dates),'Target chronology')
    require(len(set(dates.astype('datetime64[M]').tolist()))==T,'Distinct target months')
    valid=base._bool(target_bundle['valid'],T,'Target valid');values=np.asarray(target_bundle['values'])
    require(values.shape==(T,) and np.shape(train_X)==(T,3),'Training router axes')
    events=_days(event_dates,'Router dates');n=len(events);require(np.shape(query_X)==(n,3),'Query state axes')
    require(n<2 or np.all(events[1:]>events[:-1]),'Router query order')
    partial=events==np.datetime64(INITIAL_PARTIAL);active=np.zeros(n,bool)
    pred0=np.zeros(n);pred1=np.zeros(n);p0=np.full(n,.5);p1=np.full(n,.5)
    years=sorted(set(int(str(d)[:4]) for d in events[~partial]));annual=[]
    counts=dict.fromkeys(MODEL_COUNT_FIELDS,0);counts.update(years=len(years),training_months=T,valid_target_months=int(valid.sum()),partial_events=int(partial.sum()))
    for year in years:
        cutoff=np.datetime64('%04d-12-31'%(year-1));lower=np.datetime64('%04d-12-31'%(year-6))
        ids=np.flatnonzero((dates>lower)&(dates<=cutoff)&(ends<=cutoff)&valid)
        qids=np.flatnonzero(np.asarray([int(str(d)[:4])==year for d in events])&~partial)
        # Both train and query numeric bodies are consumed only after date masks.
        X=np.asarray(np.asarray(train_X)[ids],float);d=np.asarray(values[ids],float);Q=np.asarray(np.asarray(query_X)[qids],float)
        require(np.isfinite(X).all() and np.isfinite(d).all() and np.isfinite(Q).all(),'Finite required router states/targets')
        row=dict(year=year,cutoff=str(cutoff),lower_bound=str(lower),training_indices=ids.tolist(),query_indices=qids.tolist(),status='insufficient_months',weights=[],weight_sum=0.,mean_x=None,sd_x=None,scale_x=None,mean_d=None,target_sd=None,gram=None,rhs=None,beta=None,fit_performed=False)
        counts['annual_training_month_observations']+=len(ids)
        if len(ids)<18:annual.append(row);continue
        age=(cutoff.astype('datetime64[M]')-dates[ids].astype('datetime64[M]')).astype(int)
        u=np.exp2(-age/12.);W=u.copy();W*=len(ids)/np.sum(u);mass=float(np.sum(W))
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            mx=np.einsum('n,ni->i',W,X,optimize=False)/mass
            centered=X-mx
            sx=np.sqrt(np.einsum('ni,n,ni->i',centered,W,centered,optimize=False)/mass)
            scale=np.where(sx<=1e-12,1.,sx)
            md=float(np.einsum('n,n->',W,d,optimize=False)/mass)
            residual=d-md
            sd=float(np.sqrt(np.einsum('n,n,n->',residual,W,residual,optimize=False)/mass))
        require(np.isfinite(mx).all() and np.isfinite(sx).all() and np.isfinite(md) and np.isfinite(sd) and np.isfinite(W).all() and np.all(W>0.),'Finite weighted moments')
        row.update(weights=W.tolist(),weight_sum=mass,mean_x=mx.tolist(),sd_x=sx.tolist(),scale_x=scale.tolist(),mean_d=md,target_sd=sd,status='constant_target')
        if sd<=1e-12:annual.append(row);continue
        Z=(X-mx)/scale;Zq=(Q-mx)/scale
        with np.errstate(over='raise',invalid='raise'):
            gram=np.einsum('ni,n,nj->ij',Z,W,Z,optimize=False)+10.*np.eye(3)
            rhs=np.einsum('ni,n,n->i',Z,W,residual,optimize=False)
        require(np.isfinite(gram).all() and np.isfinite(rhs).all(),'Finite Ridge system')
        beta=np.linalg.solve(gram,rhs)
        require(np.shape(beta)==(3,) and np.isfinite(beta).all(),'Finite Ridge solution')
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            predictions=md+np.einsum('ij,j->i',Zq,beta,optimize=False)
            probabilities0=.5+.25*np.tanh(md/sd)
            probabilities1=.5+.25*np.tanh(predictions/sd)
        require(np.isfinite(predictions).all() and np.isfinite(probabilities1).all() and np.all((probabilities1>=.25)&(probabilities1<=.75)),'Bounded route')
        active[qids]=True;pred0[qids]=md;pred1[qids]=predictions;p0[qids]=probabilities0;p1[qids]=probabilities1
        row.update(status='fitted',gram=gram.tolist(),rhs=rhs.tolist(),beta=beta.tolist(),fit_performed=True)
        counts['intercept_estimates']+=1;counts['ridge_solves']+=1;annual.append(row)
    counts.update(active_query_events=int(active.sum()),cold_query_events=int((~active&~partial).sum()))
    audit=dict(schema=_schema('model_audit'),annual=annual,counts=counts,market_columns=list(COLUMNS),original_models_refitted=False)
    return dict(schema=_schema('router'),n=n,event_dates=events,active=active,intercept_prediction=pred0,state_prediction=pred1,intercept_p=p0,state_p=p1,audit=audit,control_gate=receipt)


def _select(study,cache,pre):
    source=dict(baseline=cache['base'],core=cache['core'],core_weight=cache['cw'],eligible=cache['good'],state_calm10=cache['calm'],state_weak_narrow=cache['weak'])
    chosen,post=study.ep.select_boundary(pre.copy(),cache['good'].copy(),deepcopy(cache['meta']['symbols']),cache['effects'].copy(),{k:v.copy() for k,v in source.items()},POLICY.copy())
    post=bridge._matrix(post,pre.shape,len(pre),'Final scores');choices=[[int(j) for j in row] for row in chosen]
    return study.rp.make_orders(choices,post,cache['meta']),post,choices


def original_decisions(study,arrays,meta,n=None):
    n=np.shape(arrays['baseline'])[0] if n is None else n
    cache=bridge._cache(study,arrays,meta,n);pre=cache['pre'].copy()
    orders,post,_=_select(study,cache,pre)
    return orders,post,pre


def decisions(study,arrays,meta,router_bundle,event_plain,n=None,*,incumbent_validator=None,expected_incumbent_orders=None):
    total,width=np.shape(arrays['baseline']);n=total if n is None else n
    require(type(n) is int and 0<=n<=total,'Decision prefix')
    cache=bridge._cache(study,arrays,meta,n)
    inc_orders,inc_post,inc_choices=_select(study,cache,cache['pre'])
    if expected_incumbent_orders is not None:require(inc_orders==expected_incumbent_orders,'Frozen incumbent orders changed')
    receipt=_gate(incumbent_validator,deepcopy(inc_orders),inc_post.copy(),cache['pre'].copy())
    require(type(router_bundle) is dict and set(router_bundle)==set(ROUTER_KEYS) and router_bundle['schema']==_schema('router'),'Router schema')
    bn=router_bundle['n'];require(type(bn) is int and bn>=n,'Router prefix')
    require(np.shape(router_bundle['event_dates'])==(bn,) and np.array_equal(_days(np.asarray(router_bundle['event_dates'])[:n],'Consumed router dates'),_days(meta['dates'][:n],'Decision dates')),'Router dates')
    require(np.shape(router_bundle['active'])==(bn,) and np.asarray(router_bundle['active']).dtype==np.dtype(bool),'Router active layout')
    active=np.asarray(router_bundle['active'])[:n].copy()
    ps={}
    for key in ('intercept_prediction','state_prediction','intercept_p','state_p'):
        raw=np.asarray(router_bundle[key]);require(raw.shape==(bn,),'Router vector '+key)
        x=np.asarray(raw[:n],float);require(np.isfinite(x).all(),'Finite consumed '+key)
        if key.endswith('_p'):require(np.all((x>=.25)&(x<=.75)),'Bounded p');ps[key]=x
    plain=np.asarray(event_plain);require(plain.ndim==2 and plain.shape[1]==10 and len(plain)>=n*width and len(plain)%width==0,'Event plain layout')
    R=np.asarray(plain[:n*width],float).reshape(n,width,10)
    partial=np.asarray([d==INITIAL_PARTIAL for d in meta['dates'][:n]],bool)
    require(not np.any(active&partial),'Partial cannot activate router')
    for i in range(n):
        if partial[i]:continue
        E=cache['good'][i]
        require(np.isfinite(R[i,E,:2]).all() and np.isfinite(cache['core'][i,E]).all() and np.isfinite(cache['cw'][i,E]).all() and np.all(cache['cw'][i,E]>=.5),'Original eligible core/rank validity')
        require(np.all(np.abs(R[i,E,:2])<=.5),'Original centered percentile ranks')
    orders={ARMS[0]:inc_orders};scores={ARMS[0]:inc_post};pres={ARMS[0]:cache['pre'].copy()};choices={ARMS[0]:inc_choices};deltas={}
    for arm,key in zip(ARMS[1:],('intercept_p','state_p')):
        delta=np.zeros((n,width));pre=cache['pre'].copy()
        for i in np.flatnonzero(active):
            E=cache['good'][i];r0=R[i,E,0];r1=R[i,E,1];p=ps[key][i]
            with np.errstate(over='raise',invalid='raise',divide='raise'):
                blend=p*r0+(1.-p)*r1;equal=r0==r1;blend[equal]=r0[equal]
                delta[i,E]=.10*(blend-cache['core'][i,E]/cache['cw'][i,E])
        require(np.isfinite(delta).all(),'Finite core delta')
        changed=delta!=0.;pre[changed]+=delta[changed]
        orders[arm],scores[arm],choices[arm]=_select(study,cache,pre);pres[arm]=pre;deltas[arm]=delta
    a,b=ARMS[1:];events=[]
    for i,day in enumerate(meta['dates'][:n]):
        events.append(dict(query_index=i,date=day,partial=bool(partial[i]),active=bool(active[i]),eligible_indices=np.flatnonzero(cache['good'][i]).tolist(),intercept_p=float(ps['intercept_p'][i]),state_p=float(ps['state_p'][i]),intercept_delta=deltas[a][i].tolist(),state_delta=deltas[b][i].tolist(),incumbent_choices=choices[ARMS[0]][i],intercept_choices=choices[a][i],state_choices=choices[b][i]))
    counts=dict.fromkeys(COUNT_FIELDS,0)
    counts.update(events=n,symbols_per_event=width,partial_events=int(partial.sum()),active_events=int(active.sum()),cold_events=int((~active&~partial).sum()),eligible_cells=int(cache['good'][~partial].sum()),intercept_changed_delta_cells=int(np.count_nonzero(deltas[a])),state_changed_delta_cells=int(np.count_nonzero(deltas[b])),intercept_changed_holdings_events=sum(set(x)!=set(y) for x,y in zip(inc_choices,choices[a])),state_changed_holdings_events=sum(set(x)!=set(y) for x,y in zip(inc_choices,choices[b])),state_vs_intercept_changed_holdings_events=sum(set(x)!=set(y) for x,y in zip(choices[a],choices[b])),incumbent_episode_calls=1,intercept_episode_calls=1,state_episode_calls=1)
    return orders,scores,pres,dict(schema=_schema('decisions'),counts=counts,events=events,incumbent_gate=receipt)
