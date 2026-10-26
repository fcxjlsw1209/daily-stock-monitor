"""Independent260 clipped-leader utility/source/model/selection verification.

Only frozen independent control/accounting helpers are imported. New teacher,
market-clock and router computations are separately reconstructed. Canonical
binary64 decisions and augmented-SVD evidence have distinct scopes and counts.
"""
import copy
import hashlib
import importlib.util
import numbers
import re
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
PREVIOUS=ROOT/'work/monthly_joint_identity259_independent.py'
PREVIOUS_SHA='217407ac3f1ec71c6898f245d7ba0ddea8e138609f115f75e3d7d13323762c2d'
def require(ok,message):
    if not ok:raise ValueError(message)
require(hashlib.sha256(PREVIOUS.read_bytes()).hexdigest()==PREVIOUS_SHA,'Frozen259 independent bytes')
_spec=importlib.util.spec_from_file_location('leader260_frozen259_independent',PREVIOUS)
previous=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(previous)
exact_array,record_check=previous.exact_array,previous.record_check
continuous=previous.continuous
original_base,accounting=previous.original_base,previous.accounting
accounts,bridge,numeric=previous.accounts,previous.bridge,previous.numeric
PATHS,ACCOUNTING_SHA=previous.PATHS,previous.ACCOUNTING_SHA
ARMS=('incumbent_local67','unconditional_leader_utility','state_leader_utility')
INITIAL='2016-08-17'
TARGETS=('simple_excess','log_excess','vol_scaled','loss_utility')
INPUT_KEYS=('schema','n','width','training_row_ids','dates','ends','symbols','offsets','plain','eligible','common_good','labels','event_dates','event_symbols','event_plain')
INPUT_FLAGS=('full_input_audit_checked','no_future_training_values','cached_simple_labels_preserved','common_good_preserved','unused_targets_not_consumed','original_training_identity_checked')
TARGET_COUNT_FIELDS=('months','source_rows','domain_rows','cash_months','identical_set_months','unavailable_months','valid_months','selected_label_observations','nonzero_targets')
SOURCE_COUNT_FIELDS=('training_events','query_events','partial_events','state_vectors','price_observations')
MODEL_COUNT_FIELDS=('years','training_months','valid_target_months','annual_training_month_observations','intercept_estimates','ridge_solves','active_query_events','cold_query_events','partial_events','original_model_fits')
COUNT_FIELDS=('events','symbols_per_event','partial_events','active_events','cold_events','eligible_cells','intercept_changed_delta_cells','state_changed_delta_cells','intercept_changed_holdings_events','state_changed_holdings_events','state_vs_intercept_changed_holdings_events','incumbent_episode_calls','intercept_episode_calls','state_episode_calls','new_model_fits')
MARKET_COLUMNS=['spy_vol63','spy_mom252_skip42','xly_xlp_relative126']
CONTINUOUS_ATOL=CONTINUOUS_RTOL=2e-10
INCUMBENT_FLAGS=previous.INCUMBENT_FLAGS
INCUMBENT_GATE_FLAGS=INCUMBENT_FLAGS
TARGET_FLAGS=('teacher_domain_before_label_checks','selected_union_labels_checked','same_set_cash_exact_zero','no_label_based_replacement','full_teacher_audit_checked')
SOURCE_FLAGS=('official_market_windows_checked','market_states_independently_recomputed','no_future_market_values','full_source_audit_checked')
MODEL_FLAGS=('common_paired_month_cohort_checked','weighted_scaler_target_sd_checked','canonical_router_checked','independent_augmented_svd','full_model_audit_checked','actual_fit_counts_checked','cold_incumbent_bypass_checked')
SELECTION_FLAGS=('original_core_budget_preserved','original_eligibility_unchanged','ten_percentage_point_delta_checked','three_arms_single_episode_checked','partial_cold_exact_copy','full_decision_audit_checked')

def dependency_files():return tuple(sorted(set(previous.dependency_files())|{PREVIOUS,Path(__file__).resolve()}))

def _finite(x,name):
    require(np.isfinite(x).all(),'Nonfinite '+name)
    return x

def _number(x,name,finite=False):
    require(not isinstance(x,(bool,np.bool_)) and isinstance(x,numbers.Real),name+' real nonbool')
    try:v=float(x)
    except (TypeError,ValueError,OverflowError) as exc:raise ValueError(name+' binary64') from exc
    if finite:require(np.isfinite(v),name+' finite')
    return v

def _floats(values,name,finite=False):
    a=np.asarray(values);out=np.empty(a.shape,float)
    for ix in np.ndindex(a.shape):out[ix]=_number(a[ix],name,finite)
    return out

def _days(value,name):
    raw=np.asarray(value)
    require(raw.ndim==1,name+' date vector')
    for v in raw:
        if isinstance(v,(str,np.str_)):
            require(re.fullmatch(r'\d{4}-\d{2}-\d{2}',str(v)) is not None,name+' canonical ISO day')
        elif isinstance(v,np.datetime64):
            require(not np.isnat(v) and v==v.astype('datetime64[D]'),name+' whole day')
        elif isinstance(v,pd.Timestamp):
            require(v.tz is None and v==v.normalize(),name+' unzoned whole day')
        else:raise ValueError(name+' explicit whole-day date')
    try:a=np.asarray(raw,dtype='datetime64[D]')
    except (TypeError,ValueError) as exc:raise ValueError(name+' dates') from exc
    require(a.ndim==1 and not np.isnat(a).any(),name+' dates')
    return a

def _bools(value,size,name):
    a=np.asarray(value);require(a.shape==(size,) and a.dtype==np.dtype(bool),name+' bool vector')
    return a

def _receipt(n,flags,**kw):
    result=dict(passed=True,checked_events=n,producer_modules_imported=False,computes_returns=False)
    result.update({key:True for key in flags});result.update(kw);return result

def verify_incumbent(arrays,meta,orders,scores,pre_scores,n=None):
    proof=previous.verify_incumbent_selection(arrays,meta,orders,scores,pre_scores,n)
    return dict(proof,old_model_solves=0,conditional_ridge_solves=0,independent_canonical_solves=0,independent_svd_solves=0,original_episode_calls=1)
verify_incumbent_selection=verify_incumbent

def _inc_receipt(n):
    return dict(passed=True,checked_events=n,ordered_choices_checked=n,full_scores_supplied=True,
      full_pre_scores_supplied=True,independent_original_pre_checked=True,exact_incumbent_copy_checked=True,
      original_eligibility_unchanged=True,single_original_episode=True,numerical_tolerance=0.,
      producer_modules_imported=False,computes_returns=False,candidate_selection_computed=False,
      old_model_solves=0,conditional_ridge_solves=0,independent_canonical_solves=0,independent_svd_solves=0,original_episode_calls=1)

def _gate(gate,n):
    require(type(gate) is dict and gate.get('passed') is True,'Passed original gate')
    if gate.get('scope')=='synthetic_no_callback':record_check(gate,dict(passed=True,scope='synthetic_no_callback'),'Synthetic gate')
    else:record_check(gate,_inc_receipt(n),'Exact original gate scope')

def _zero_work():
    return dict(old_model_solves=0,conditional_ridge_solves=0,independent_canonical_solves=0,independent_svd_solves=0,original_episode_calls=0)

def _ids(values,size,label):
    a=np.asarray(values);require(a.shape==(size,),label+' axis')
    require(all(isinstance(v,(str,np.str_)) and v and str(v).strip()==v for v in a),label+' identities')
    return a.astype(str)

def check_tree(actual,expected,label):
    if isinstance(expected,np.ndarray):exact_array(actual,expected,label)
    elif type(expected) is dict:
        require(type(actual) is dict and set(actual)==set(expected),label+' keys')
        for key,value in expected.items():check_tree(actual[key],value,label+'/'+key)
    else:record_check(actual,expected,label)

def router_inputs(train,meta,arrays,n=None):
    require(type(meta) is dict and 'monthly' in meta and 'training' in meta,'Combined input metadata')
    m,tm=meta['monthly'],meta['training'];n=len(m['dates']) if n is None else n
    require(type(n) is int and 0<=n<=len(m['dates']),'Consumed query count')
    E=np.asarray(arrays['eligible']);require(E.ndim==2 and E.dtype==np.dtype(bool) and E.shape[0]>=n and E.shape[1]>0,'Original eligibility axes')
    width=E.shape[1];qd=_days(m['dates'][:n],'query dates');qe=_days(m['ends'][:n],'query ends')
    require(qd.shape==qe.shape and np.all(qe>qd) and np.all(qd[1:]>qd[:-1]),'Query dates strictly increasing and ends future')
    query_symbols=[]
    for row in m['symbols'][:n]:
        ss=_ids(row,width,'Query inventory');require(len(set(ss))==width,'Unique query symbols');query_symbols.extend(ss)
    td=_days(train['dates'],'training flat dates');te=_days(train['ends'],'training label ends')
    require(td.shape==te.shape and np.all(te>td) and np.all(te.astype('datetime64[M]')==td.astype('datetime64[M]')+1),'Next-month label end axes')
    offsets=np.asarray(train['offsets']);require(offsets.ndim==1 and offsets.dtype.kind in 'iu' and len(offsets)==len(tm['dates'])+1,'Complete monthly offsets')
    require(len(offsets)>0 and offsets[0]==0 and offsets[-1]==len(td) and np.all(offsets[1:]>offsets[:-1]),'Positive monthly spans')
    md=_days(tm['dates'],'training monthly dates');me=_days(tm['ends'],'training monthly ends')
    require(md.shape==me.shape and len(tm['symbols'])==len(md) and np.all(md[1:]>md[:-1]) and len(np.unique(md.astype('datetime64[M]')))==len(md),'Training header chronology')
    for i,(start,stop) in enumerate(zip(offsets[:-1],offsets[1:])):
        require(np.all(td[start:stop]==md[i]) and np.all(te[start:stop]==me[i]),'Flattened training date/end identity')
    formal=qd[qd!=np.datetime64(INITIAL)]
    keep=np.zeros(len(td),bool)
    if len(formal):
        year=int(formal[-1].astype('datetime64[Y]').astype(int))+1970
        keep=te<=np.datetime64('%04d-12-31'%(year-1))
    require(not np.any(keep[1:]&~keep[:-1]),'Maturity must be a row prefix')
    ids=np.flatnonzero(keep).astype(np.int64);names=[];out_offsets=[0]
    for i,(start,stop) in enumerate(zip(offsets[:-1],offsets[1:])):
        if not keep[start:stop].any():continue
        require(keep[start:stop].all(),'Whole-month maturity')
        ss=_ids(tm['symbols'][i],int(stop-start),'Training symbols');require(len(set(ss))==len(ss),'Unique training symbols')
        names.extend(ss);out_offsets.append(out_offsets[-1]+len(ss))
    plain=np.asarray(train['train_plain']);eligible=np.asarray(train['train_eligible']);good=np.asarray(train['common_good']);labels=np.asarray(train['target_simple_excess'])
    require(plain.shape==(len(td),10),'Original plain10 shape')
    require(eligible.shape==good.shape==labels.shape==(len(td),) and eligible.dtype==good.dtype==np.dtype(bool),'Original labels/Boolean masks axes')
    event_dates=_days(np.asarray(train['event_dates'])[:n*width],'event dates')
    require(np.array_equal(event_dates,np.repeat(qd,width)),'Query cached date join')
    ep=np.asarray(train['event_plain']);require(ep.ndim==2 and ep.shape[0]>=n*width and ep.shape[1]==10,'Query plain10 axes')
    # No future symbol or numeric body is dereferenced; no target except simple_excess is consumed.
    return dict(schema='monthly_leader_utility260_inputs_v1',n=n,width=width,training_row_ids=ids,
      dates=td[ids].copy(),ends=te[ids].copy(),symbols=np.asarray(names,str),offsets=np.asarray(out_offsets,np.int64),
      plain=_floats(plain[ids],'consumed plain10'),eligible=eligible[ids].copy(),common_good=good[ids].copy(),labels=labels[ids].copy(),
      event_dates=qd,event_symbols=[[str(v) for v in query_symbols[i*width:(i+1)*width]] for i in range(n)],event_plain=ep[:n*width].copy())

def verify_inputs(train,meta,arrays,inp,n=None):
    wanted=router_inputs(train,meta,arrays,n);check_tree(inp,wanted,'Full consumed inputs')
    return _receipt(wanted['n'],INPUT_FLAGS,counts=dict(training_rows=len(wanted['dates']),training_months=len(wanted['offsets'])-1,
      query_rows=wanted['n']*wanted['width']),full_input_audit_supplied=True,exact_discrete=True,numerical_tolerance=0.,**_zero_work())

def targets_oracle(inp):
    require(type(inp) is dict and set(inp)==set(INPUT_KEYS) and inp['schema']=='monthly_leader_utility260_inputs_v1','Complete teacher input schema')
    d=_days(inp['dates'],'teacher dates');end=_days(inp['ends'],'teacher ends');size=len(d)
    names=_ids(inp['symbols'],size,'Teacher symbols');ids=np.asarray(inp['training_row_ids'])
    require(ids.shape==(size,) and ids.dtype.kind in 'iu' and np.all(ids[1:]>ids[:-1]),'Original teacher row IDs')
    X=_floats(inp['plain'],'teacher plain');require(X.shape==(size,10),'Teacher plain10')
    E=_bools(inp['eligible'],size,'Teacher eligibility');good=_bools(inp['common_good'],size,'Teacher common good');labels=np.asarray(inp['labels'])
    require(labels.shape==(size,),'Teacher label axis')
    offsets=np.asarray(inp['offsets']);require(offsets.ndim==1 and offsets.dtype.kind in 'iu' and len(offsets)>0 and offsets[0]==0 and offsets[-1]==size and np.all(np.diff(offsets)>0),'Teacher offsets')
    count={key:0 for key in TARGET_COUNT_FIELDS};count['source_rows']=size
    records=[];dates=[];ends=[];valid=[];values=[]
    for i,(start,stop) in enumerate(zip(offsets[:-1],offsets[1:])):
        rows=np.arange(start,stop);require(np.all(d[rows]==d[start]) and np.all(end[rows]==end[start]) and end[start]>d[start],'Paired monthly identity')
        require(len(set(names[rows]))==len(rows),'Teacher duplicate identity')
        domain=[j for j in rows if E[j] and np.isfinite(X[j]).all()]
        count['domain_rows']+=len(domain);count['months']+=1
        selected=[]
        for column in (0,1):selected.append([] if len(domain)<2 else sorted(domain,key=lambda j:(-float(X[j,column]),str(names[j])))[:3])
        raw,risk=selected;k=len(raw);require(k==len(risk),'Matched basket sizes')
        union=sorted(set(raw)|set(risk),key=lambda j:str(names[j]));positive=sorted(set(raw)-set(risk),key=lambda j:str(names[j]));negative=sorted(set(risk)-set(raw),key=lambda j:str(names[j]))
        available=True;value=0.;reason='cash';seen={}
        if k:
            for j in union:
                val=_number(labels[j],'Selected cached label');seen[j]=val
                if not good[j] or not np.isfinite(val):available=False
            count['selected_label_observations']+=len(union)
            if not available:reason='selected_label_unavailable';value=None;count['unavailable_months']+=1
            elif not positive:reason='identical_sets';count['identical_set_months']+=1
            else:
                reason='available';value=float((np.sum(np.asarray([seen[j] for j in positive],dtype=np.float64))-np.sum(np.asarray([seen[j] for j in negative],dtype=np.float64)))/k)
                require(np.isfinite(value),'Finite paired teacher difference')
        else:count['cash_months']+=1
        count['valid_months']+=int(available);count['nonzero_targets']+=int(available and value!=0.)
        records.append(dict(month_index=i,signal_date=str(d[start]),label_end=str(end[start]),row_indices=ids[rows].tolist(),
          domain_indices=ids[domain].tolist(),raw_choices=[str(names[j]) for j in raw],risk_choices=[str(names[j]) for j in risk],
          selected_union=[str(names[j]) for j in union],raw_only=[str(names[j]) for j in positive],risk_only=[str(names[j]) for j in negative],
          k=k,valid=bool(available),reason=reason,value=value))
        dates.append(d[start]);ends.append(end[start]);valid.append(available);values.append(np.nan if value is None else value)
    return dict(schema='monthly_leader_utility260_targets_v1',dates=np.asarray(dates,dtype='datetime64[D]'),ends=np.asarray(ends,dtype='datetime64[D]'),
      valid=np.asarray(valid,bool),values=np.asarray(values,float),records=records,counts=count)
build_expert_targets=targets_oracle

def verify_targets(inp,target_bundle):
    expected=targets_oracle(inp);check_tree(target_bundle,expected,'Full paired teacher')
    return _receipt(inp['n'],TARGET_FLAGS,counts=expected['counts'],full_target_audit_supplied=True,numerical_tolerance=0.,exact_discrete=True,**_zero_work())

def _calendar(calendar):
    raw=calendar['sessions'] if type(calendar) is dict else calendar
    d=_days(raw,'official sessions');require(len(d)>0 and np.all(d[1:]>d[:-1]),'Strict official session order')
    return list(pd.DatetimeIndex(d))

def market_oracle(adj,calendar,training_signal_dates,query_dates):
    require(type(adj) is pd.DataFrame and isinstance(adj.index,pd.DatetimeIndex),'Adjusted source dataframe')
    require(adj.index.tz is None and adj.index.is_unique and adj.index.is_monotonic_increasing and adj.index.equals(adj.index.normalize()),'Unique sorted whole-day source axis')
    require(adj.columns.is_unique,'Unique source symbols')
    sessions=_calendar(calendar);pos={d:i for i,d in enumerate(sessions)}
    td,qd=_days(training_signal_dates,'training market dates'),_days(query_dates,'query market dates')
    require(np.all(td[1:]>td[:-1]) and np.all(qd[1:]>qd[:-1]),'Strict signal chronology')
    c={k:0 for k in SOURCE_COUNT_FIELDS};c.update(training_events=len(td),query_events=len(qd))
    result=[];evidence=[]
    for source_kind,dates in enumerate((td,qd)):
        X=np.empty((len(dates),3));rows=[]
        for i,date in enumerate(dates):
            day=pd.Timestamp(date);partial=source_kind==1 and str(date)==INITIAL
            if partial:
                X[i]=0.;start=relative=end=None;counts=(0,0,0);c['partial_events']+=1
            else:
                X[i]=_market_state(adj,sessions,pos,day);t=pos[day]
                start=str(sessions[t-252].date());relative=str(sessions[t-126].date());end=str(date);counts=(253,127,127)
                c['state_vectors']+=1;c['price_observations']+=507
            rows.append(dict(date=str(date),initial_partial=partial,spy_start=start,relative_start=relative,price_end=end,
              spy_price_count=counts[0],xly_price_count=counts[1],xlp_price_count=counts[2],values=X[i].tolist()))
        result.append(X);evidence.append(rows)
    return result[0],result[1],dict(schema='monthly_leader_utility260_market_source_v1',columns=MARKET_COLUMNS.copy(),training=evidence[0],queries=evidence[1],counts=c)

def verify_market_inputs(adj,calendar,training_signal_dates,query_dates,train_X,query_X,source_audit):
    t,q,a=market_oracle(adj,calendar,training_signal_dates,query_dates)
    exact_array(train_X,t,'Exact historical states');exact_array(query_X,q,'Exact query states');record_check(source_audit,a,'Market audit')
    return _receipt(len(q),SOURCE_FLAGS,counts=a['counts'],source_audit=a,full_source_audit_supplied=True,
      continuous_atol=CONTINUOUS_ATOL,continuous_rtol=CONTINUOUS_RTOL,rank_tolerance=0.,exact_discrete=True,**_zero_work())
verify_source=verify_market_inputs

def _market_state(adj,sessions,positions,day):
    require(day in positions,'Signal absent from official calendar');t=positions[day]
    require(t>=252,'Insufficient official state history')
    values={}
    for symbol,length in (('SPY',253),('XLY',127),('XLP',127)):
        require(symbol in adj.columns,'Missing market symbol '+symbol)
        dates=sessions[t-length+1:t+1]
        require(all(d in adj.index for d in dates),'Missing official market price row')
        v=_floats(adj.loc[dates,symbol].to_numpy(),symbol,True)
        require(np.all(v>0),'Positive market prices');values[symbol]=v
    spy=values['SPY'];returns=spy[-63:]/spy[-64:-1]-1.
    state=np.array([np.std(returns,ddof=1)*np.sqrt(252.),spy[-43]/spy[0]-1.,
      (values['XLY'][-1]/values['XLY'][0])/(values['XLP'][-1]/values['XLP'][0])-1.])
    return _finite(state,'market state')

def ridge_oracle(X,d,dates,cutoff,queries,*,verify_svd=True):
    """One canonical prospective system plus a separately executed SVD proof."""
    X=_floats(X,'states',True);d=_floats(d,'leader targets',True);Q=_floats(queries,'query states',True)
    require(X.shape==(len(d),3) and Q.ndim==2 and Q.shape[1]==3 and len(d)>0,'Three-state system axes')
    days=_days(dates,'fit signal dates');require(len(days)==len(d) and len(np.unique(days.astype('datetime64[M]')))==len(days),'One row per fit month')
    age=np.datetime64(cutoff,'M').astype(int)-days.astype('datetime64[M]').astype(int)
    require(np.all(age>=0),'Past monthly ages');u=np.exp2(-age/12.);w=u.copy();w*=len(w)/np.sum(u);_finite(w,'weights')
    require(np.all(w>0),'Positive weights');mass=float(w.sum())
    mu=np.einsum('n,ni->i',w,X,optimize=False)/mass
    dx=X-mu;sd=np.sqrt(np.einsum('ni,n,ni->i',dx,w,dx,optimize=False)/mass)
    scale=np.where(sd<=1e-12,1.,sd)
    a=float(np.einsum('n,n->',w,d,optimize=False)/mass);center=d-a
    target_sd=float(np.sqrt(np.einsum('n,n,n->',center,w,center,optimize=False)/mass))
    out=dict(weights=w,weight_sum=mass,state_mean=mu,state_raw_scale=sd,state_scale=scale,target_mean=a,target_sd=target_sd,
      beta=None,gram=None,rhs=None,prediction=None,svd_solves=0,canonical_solves=0)
    for name in ('state_mean','state_raw_scale','state_scale','target_mean','target_sd'):_finite(out[name],name)
    if len(d)<18 or target_sd<=1e-12:return out
    Z=(X-mu)/scale;Zq=(Q-mu)/scale
    gram=np.einsum('ni,n,nj->ij',Z,w,Z,optimize=False)+10.*np.eye(3)
    rhs=np.einsum('ni,n,n->i',Z,w,center,optimize=False)
    _finite(gram,'gram');_finite(rhs,'rhs')
    with threadpool_limits(limits=1):beta=np.linalg.solve(gram,rhs)
    pred=a+np.einsum('ij,j->i',Zq,beta,optimize=False)
    _finite(beta,'slopes');_finite(pred,'prediction')
    if verify_svd:
        with threadpool_limits(limits=1):svd=numeric.least_squares(Z,center,w,np.full(3,10.))
        continuous(svd,beta,'Independent SVD slopes')
        continuous(a+np.einsum('ij,j->i',Zq,svd,optimize=False),pred,'Independent SVD prediction')
        continuous(np.einsum('ij,j->i',gram,svd,optimize=False),rhs,'Independent normal residual')
    out.update(beta=beta,gram=gram,rhs=rhs,prediction=pred,svd_solves=int(verify_svd),canonical_solves=1)
    return out

def router_oracle(target,train_X,query_X,event_dates,*,verify_svd=True,control_gate=None):
    require(type(target) is dict and target.get('schema')=='monthly_leader_utility260_targets_v1','Teacher bundle for annual fit')
    dates=_days(target['dates'],'teacher signals');ends=_days(target['ends'],'teacher maturity');qd=_days(event_dates,'router queries');n=len(qd)
    gate=dict(passed=True,scope='synthetic_no_callback') if control_gate is None else copy.deepcopy(control_gate);_gate(gate,n)
    require(dates.shape==ends.shape and np.all(dates[1:]>dates[:-1]) and np.all(ends>dates),'Chronological paired months')
    valid=_bools(target['valid'],len(dates),'Target availability');ys=np.asarray(target['values']);require(ys.shape==(len(dates),),'Target values axis')
    X=_floats(train_X,'historical states',True);Q=_floats(query_X,'query states',True)
    require(X.shape==(len(dates),3) and Q.shape==(n,3),'Router three-state axes')
    require(np.all(qd[1:]>qd[:-1]),'Increasing router queries')
    partial=qd==np.datetime64(INITIAL);require(np.all(Q[partial]==0.),'Partial market placeholder zero')
    active=np.zeros(n,bool);ip=np.zeros(n);sp=np.zeros(n);iw=np.full(n,.5);sw=np.full(n,.5)
    annual=[];counts={k:0 for k in MODEL_COUNT_FIELDS};counts.update(training_months=len(dates),valid_target_months=int(valid.sum()),partial_events=int(partial.sum()))
    actual=dict(independent_canonical_solves=0,independent_svd_solves=0)
    years=qd.astype('datetime64[Y]').astype(int)+1970
    for year0 in np.unique(years[~partial]):
        year=int(year0);cutoff=np.datetime64('%04d-12-31'%(year-1));lower=np.datetime64('%04d-12-31'%(year-6))
        ids=np.flatnonzero(valid&(dates>lower)&(dates<=cutoff)&(ends<=cutoff));qi=np.flatnonzero((years==year)&~partial)
        require(len(np.unique(dates[ids].astype('datetime64[M]')))==len(ids),'Distinct valid signal months')
        row=dict(year=year,cutoff=str(cutoff),lower_bound=str(lower),training_indices=ids.tolist(),query_indices=qi.tolist(),status='insufficient_months',
          weights=[],weight_sum=0.,mean_x=None,sd_x=None,scale_x=None,mean_d=None,target_sd=None,gram=None,rhs=None,beta=None,fit_performed=False)
        counts['years']+=1;counts['annual_training_month_observations']+=len(ids)
        if len(ids)>=18:
            fit=ridge_oracle(X[ids],ys[ids],dates[ids],cutoff,Q[qi],verify_svd=verify_svd)
            row.update(weights=fit['weights'].tolist(),weight_sum=fit['weight_sum'],mean_x=fit['state_mean'].tolist(),sd_x=fit['state_raw_scale'].tolist(),
              scale_x=fit['state_scale'].tolist(),mean_d=fit['target_mean'],target_sd=fit['target_sd'],status='constant_target')
            if fit['canonical_solves']:
                row.update(status='fitted',gram=fit['gram'].tolist(),rhs=fit['rhs'].tolist(),beta=fit['beta'].tolist(),fit_performed=True)
                active[qi]=True;ip[qi]=fit['target_mean'];sp[qi]=fit['prediction']
                iw[qi]=.5+.25*np.tanh(ip[qi]/fit['target_sd']);sw[qi]=.5+.25*np.tanh(sp[qi]/fit['target_sd'])
                counts['intercept_estimates']+=1;counts['ridge_solves']+=1
                actual['independent_canonical_solves']+=fit['canonical_solves'];actual['independent_svd_solves']+=fit['svd_solves']
        annual.append(row)
    counts.update(active_query_events=int(active.sum()),cold_query_events=int((~partial&~active).sum()))
    for x in (ip,sp,iw,sw):_finite(x,'Router predictions and weights')
    bundle=dict(schema='monthly_leader_utility260_router_v1',n=n,event_dates=qd,active=active,intercept_prediction=ip,state_prediction=sp,
      intercept_p=iw,state_p=sw,audit=dict(schema='monthly_leader_utility260_model_audit_v1',annual=annual,counts=counts,
      market_columns=MARKET_COLUMNS.copy(),original_models_refitted=False),control_gate=gate)
    return bundle,actual

def verify_models(target_bundle,train_X,query_X,event_dates,bundle):
    require(type(bundle) is dict and 'control_gate' in bundle,'Full router model bundle')
    expected,actual=router_oracle(target_bundle,train_X,query_X,event_dates,control_gate=bundle['control_gate'])
    check_tree(bundle,expected,'Full independently rebuilt canonical router')
    return _receipt(expected['n'],MODEL_FLAGS,counts=expected['audit']['counts'],full_model_audit_supplied=True,
      full_bundle_supplied=True,continuous_atol=CONTINUOUS_ATOL,continuous_rtol=CONTINUOUS_RTOL,rank_tolerance=0.,exact_discrete=True,
      old_model_solves=0,conditional_ridge_solves=actual['independent_canonical_solves'],intercept_estimates=expected['audit']['counts']['intercept_estimates'],
      original_episode_calls=0,**actual)

def _router_prefix(bundle,dates,n):
    require(type(bundle) is dict and bundle.get('schema')=='monthly_leader_utility260_router_v1' and type(bundle.get('n')) is int and bundle['n']>=n,'Router prefix layout')
    require(np.array_equal(_days(np.asarray(bundle['event_dates'])[:n],'Saved forecast query dates'),dates),'Saved query identity')
    out={}
    for key in ('active','intercept_prediction','state_prediction','intercept_p','state_p'):
        a=np.asarray(bundle[key]);require(a.shape==(bundle['n'],),'Forecast vector axis')
        out[key]=_bools(a[:n],n,'Active router mask') if key=='active' else _floats(a[:n],key,True)
    partial=dates==np.datetime64(INITIAL);require(not out['active'][partial].any(),'Partial inactive')
    for stem in ('intercept','state'):
        p=out[stem+'_p'];require(np.all((p>=.25)&(p<=.75)),'Bounded router allocation')
        require(np.all(p[~out['active']]==.5) and np.all(out[stem+'_prediction'][~out['active']]==0.),'Cold placeholders')
    return out

def selection_oracle(arrays,meta,bundle,event_plain,n=None):
    n=len(meta['dates']) if n is None else n;base=original_base(arrays,meta,n);width=base['width']
    dates=_days(meta['dates'][:n],'Selection dates');forecast=_router_prefix(bundle,dates,n)
    raw=np.asarray(event_plain);require(raw.ndim==2 and raw.shape[0]>=n*width and raw.shape[1]==10,'Consumed event plain10')
    Q=_floats(raw[:n*width],'Consumed event plain10').reshape(n,width,10)
    deltas={arm:np.zeros((n,width)) for arm in ARMS[1:]};pres={arm:base['P'].copy() for arm in ARMS}
    for i in range(n):
        if str(dates[i])==INITIAL:continue
        ids=np.flatnonzero(base['E'][i]);r0=Q[i,ids,0];r1=Q[i,ids,1];core=base['core'][i,ids];cw=float(base['cw'][i])
        require(np.isfinite(r0).all() and np.isfinite(r1).all() and np.isfinite(core).all() and np.isfinite(cw) and cw>=.5,'Finite eligible original core/ranks')
        require(np.all(np.abs(r0)<=.5) and np.all(np.abs(r1)<=.5),'Original bounded rank units')
        if not forecast['active'][i]:continue
        for arm,key in zip(ARMS[1:],('intercept_p','state_p')):
            p=float(forecast[key][i]);blend=p*r0+(1.-p)*r1;equal=r0==r1;blend[equal]=r0[equal]
            delta=.10*(blend-core/cw);_finite(delta,'Core redistribution')
            deltas[arm][i,ids]=delta;nonzero=delta!=0.;pres[arm][i,ids[nonzero]]+=delta[nonzero]
    scores={};choices={}
    for arm in ARMS:scores[arm],choices[arm]=accounts.episode(pres[arm],base)
    count={k:0 for k in COUNT_FIELDS};count.update(events=n,symbols_per_event=width,incumbent_episode_calls=1,intercept_episode_calls=1,state_episode_calls=1)
    events=[]
    for i in range(n):
        partial=str(dates[i])==INITIAL;active=bool(forecast['active'][i]);di=deltas[ARMS[1]][i];ds=deltas[ARMS[2]][i]
        count['partial_events']+=int(partial);count['active_events']+=int(active);count['cold_events']+=int(not partial and not active)
        count['eligible_cells']+=int(base['E'][i].sum()) if not partial else 0
        count['intercept_changed_delta_cells']+=int(np.count_nonzero(di));count['state_changed_delta_cells']+=int(np.count_nonzero(ds))
        count['intercept_changed_holdings_events']+=int(set(choices[ARMS[1]][i])!=set(choices[ARMS[0]][i]))
        count['state_changed_holdings_events']+=int(set(choices[ARMS[2]][i])!=set(choices[ARMS[0]][i]))
        count['state_vs_intercept_changed_holdings_events']+=int(set(choices[ARMS[2]][i])!=set(choices[ARMS[1]][i]))
        events.append(dict(query_index=i,date=str(dates[i]),partial=partial,active=active,eligible_indices=np.flatnonzero(base['E'][i]).tolist(),
          intercept_p=float(forecast['intercept_p'][i]),state_p=float(forecast['state_p'][i]),intercept_delta=di.tolist(),state_delta=ds.tolist(),
          incumbent_choices=choices[ARMS[0]][i],intercept_choices=choices[ARMS[1]][i],state_choices=choices[ARMS[2]][i]))
    return scores,choices,pres,dict(schema='monthly_leader_utility260_decisions_v1',counts=count,events=events,incumbent_gate=dict(passed=True,scope='synthetic_no_callback'))

def orders_from_choices(scores,choices,meta):
    return {arm:[dict(date=meta['dates'][i],end=meta['ends'][i],signal_date=meta['dates'][i],symbols=[meta['symbols'][i][j] for j in ids],
      scores=[float(scores[arm][i,j]) for j in ids]) for i,ids in enumerate(choices[arm])] for arm in ARMS}

def verify_selection(arrays,meta,bundle,event_plain,orders,*,scores,pre_scores,audit,n=None):
    n=len(meta['dates']) if n is None else n
    require(type(audit) is dict and 'incumbent_gate' in audit,'Complete decision audit');_gate(audit['incumbent_gate'],n)
    expected,choices,pres,wanted=selection_oracle(arrays,meta,bundle,event_plain,n)
    record_check(orders,orders_from_choices(expected,choices,meta),'Exact ordered portfolio')
    for actual,target,label in ((scores,expected,'Final scores'),(pre_scores,pres,'Pre scores')):
        require(type(actual) is dict and set(actual)==set(ARMS),'Three '+label+' arms')
        for arm in ARMS:exact_array(actual[arm],target[arm],label+'/'+arm)
    wanted['incumbent_gate']=copy.deepcopy(audit['incumbent_gate']);record_check(audit,wanted,'Complete source/decision audit')
    work=_zero_work();work['original_episode_calls']=3
    return _receipt(n,SELECTION_FLAGS,counts=wanted['counts'],ordered_choices_checked=3*n,full_scores_supplied=True,
      full_pre_scores_supplied=True,full_audit_supplied=True,full_bundle_supplied=True,numerical_tolerance=0.,**work)

def audit_outcomes(px,opening,orders,exports,results,meta):
    require(set(orders)==set(exports)==set(results)==set(ARMS),'Three260 accounting arms')
    rename=lambda d:{prior:d[new] for new,prior in zip(ARMS,previous.ARMS)}
    report=previous.audit_outcomes(px,opening,rename(orders),rename(exports),rename(results),meta)
    return dict(report,paths={arm:report['paths'][old] for arm,old in zip(ARMS,previous.ARMS)})
