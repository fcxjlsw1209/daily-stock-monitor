"""Independent273 clock/cohort/Ridge mathematics on caller-supplied inputs.

No producer, source loader, host, portfolio selector or return engine is imported.
Canonical normal equations are checked exactly; an independently assembled
augmented SVD is a numerical cross-check only, never a rank/decision tolerance.
"""
from datetime import date
import hashlib
import math
import numpy as np
from threadpoolctl import threadpool_limits

TARGETS=('simple_excess','log_excess','vol_scaled','loss_utility')
INITIAL='2016-08-17'
SVD_ATOL=SVD_RTOL=2e-10
MODEL_FLAGS=('original_query_identity_checked','maturity_before_numeric_body','five_year_exclusive_lower_bound_checked',
 'all_four_targets_common_cohort_checked','month_balanced_query_clock_weights_checked','canonical_ridge_exact',
 'independent_augmented_svd_checked','original_control_raw_available_rank_exact','candidate_active_domain_checked',
 'local_and_other_targets_not_fitted','source_inputs_unchanged')
DELTA_FLAGS=('original_control_exact','full_original_U_rank_checked','paired_ready_preserved','local_basis_preserved',
 'weak_only','partial_nonweak_notready_exact','nonconsumed_bytes_preserved','exact_single_pooled_delta')
WORK_FIELDS=('training_rows_scanned','canonical_solves','svd_solves','canonical_prediction_rows','svd_prediction_rows',
 'rank_evaluations','delta_evaluations','output_cells_compared','original_local_fits','original_other_target_fits')
RECORD_KEYS=('kind','query_indices','cutoff','lower_bound','training_row_ids','query_row_ids','training_months','status',
 'signal_dates','label_end_dates','symbols','weights','weights_sha256','mean','scale','beta','intercept','gram','rhs','raw_predictions')
COUNT_FIELDS=('events','width','training_rows','valid_training_rows','annual_groups','annual_ridge_fits',
 'monthly_ridge_fits','annual_prediction_rows','monthly_prediction_rows','actual_model_fits','local_model_fits','other_target_model_fits')


def dependency_files():return ()


def _work():return dict.fromkeys(WORK_FIELDS,0)


def _receipt(kind,n,width,work,flags,**extra):
    return dict(schema='monthly_matured_clock273_independent_'+kind+'_v1',passed=True,events=n,width=width,
        flags={k:True for k in flags},work=work,canonical_tolerance=0.,svd_atol=SVD_ATOL,svd_rtol=SVD_RTOL,
        producer_modules_imported=False,source_files_authenticated=False,labels_reconstructed=False,
        computes_returns=False,**extra)


def require(ok,message):
    if not ok:raise ValueError(message)


def exact_array(actual,expected,label):
    require(isinstance(actual,np.ndarray) and isinstance(expected,np.ndarray),label+' ndarray')
    require(actual.shape==expected.shape and actual.dtype==expected.dtype,label+' shape/dtype')
    require(actual.tobytes()==expected.tobytes(),label+' exact bytes')


def exact(actual,expected,label='record'):
    require(type(actual) is type(expected),label+' type')
    if isinstance(expected,np.ndarray):exact_array(actual,expected,label)
    elif type(expected) is dict:
        require(set(actual)==set(expected),label+' keys')
        for k,v in expected.items():exact(actual[k],v,label+'.'+k)
    elif type(expected) in (list,tuple):
        require(len(actual)==len(expected),label+' length')
        for i,(a,b) in enumerate(zip(actual,expected)):exact(a,b,label+'/'+str(i))
    elif type(expected) is float:require(np.float64(actual).tobytes()==np.float64(expected).tobytes(),label+' float bits')
    else:require(actual==expected,label+' value')


def _days(values,label):
    require(isinstance(values,(list,tuple,np.ndarray)),label+' date sequence')
    out=[]
    for value in values:
        text=str(value);parsed=date.fromisoformat(text)
        require(parsed.isoformat()==text,label+' canonical date');out.append(text)
    return np.asarray(out,dtype='datetime64[D]')


def lower_bound(cutoff):
    value=date.fromisoformat(str(cutoff))
    try:result=value.replace(year=value.year-5)
    except ValueError:result=value.replace(year=value.year-5,day=28)
    return np.datetime64(result.isoformat(),'D')


def month_weights(dates,cutoff):
    months,ix,counts=np.unique(dates.astype('datetime64[M]'),return_inverse=True,return_counts=True)
    age=cutoff.astype('datetime64[M]').astype(int)-months.astype(int)
    require((age>=0).all(),'Nonnegative mature weight age')
    weights=1./counts[ix].astype(float);weights*=np.exp2(-age[ix]/12.)
    total=weights.sum();require(math.isfinite(float(total)) and total>0,'Positive finite weight mass')
    return weights*(len(weights)/total)


def rank(values):
    values=[float(x) for x in values];require(all(math.isfinite(x) for x in values),'Finite rank inputs')
    n=len(values)
    return np.asarray([((float(2*sum(y<x for y in values)+sum(y==x for y in values)+1)/2.)/float(n))-.5 for x in values],dtype=np.float64)


def _array(value,shape,label,dtype):
    require(isinstance(value,np.ndarray) and value.shape==shape and value.dtype==np.dtype(dtype),label+' shape/dtype')
    return value


def _header(train,meta,arrays,n,models=True):
    m=meta['monthly'];n=len(m['dates']) if n is None else n
    require(type(n) is int and 0<=n<=len(m['dates']),'Native prefix count')
    E=arrays['eligible'];require(isinstance(E,np.ndarray) and E.ndim==2 and E.dtype==np.dtype(bool) and len(E)>=n and E.shape[1]>0,'Original E')
    width=E.shape[1];dates=_days(m['dates'][:n],'Query dates');require(np.all(dates[1:]>dates[:-1]),'Strict query order')
    require(not any(str(x)==INITIAL for x in dates[1:]),'Initial partial position')
    ends=_days(m['ends'][:n],'Query ends');require(len(ends)==n and np.all(ends>dates),'Query end chronology')
    require(len(m['symbols'][:n])==n,'Query symbol rows');symbols=[]
    for ss in m['symbols'][:n]:
        require(type(ss) is list and len(ss)==width and len(set(ss))==width and all(type(s) is str and s for s in ss),'Query symbols')
        symbols.extend(ss)
    formal=dates!=np.datetime64(INITIAL);a={}
    for key in ('raw_source_log','rank_log_excess','eligible','forecast_available'):
        value=arrays[key];_array(value,E.shape,key,bool if key in ('eligible','forecast_available') else 'float64');a[key]=value[:n].copy()
    for key in ('state_weak_narrow','paired_month_ready'):
        value=arrays[key];_array(value,(len(E),),key,bool);a[key]=value[:n].copy()
    for key in ('model_available','raw_local_linear','raw_local_constant'):
        value=arrays[key];_array(value,E.shape,key,bool if key=='model_available' else 'float64');a[key]=value[:n].copy()
    use=a['eligible']&a['forecast_available']
    require(not np.isinf(a['raw_source_log']).any() and np.isfinite(a['raw_source_log'][use]).all(),'Original raw support')
    require(np.isfinite(a['rank_log_excess']).all() and (np.abs(a['rank_log_excess'])<=.5+1e-12).all(),'Original rank support')
    require(not np.isinf(a['raw_local_linear']).any() and not np.isinf(a['raw_local_constant']).any(),'Local raw support')
    ready=use.any(axis=1)&np.all(~use|(a['model_available']&np.isfinite(a['raw_local_linear'])&np.isfinite(a['raw_local_constant'])),axis=1)
    exact_array(a['paired_month_ready'],ready,'Original whole-row readiness')
    active=formal&a['state_weak_narrow']&ready&use.any(axis=1)
    result=dict(n=n,width=width,dates=dates,symbols=symbols,formal=formal,arrays=a,use=use,active=active)
    if not models or not formal.any():return result
    tm=meta['training'];td=_days(tm['dates'],'Train month signals');te=_days(tm['ends'],'Train month ends')
    require(len(td)==len(te)==len(tm['symbols']) and np.all(td[1:]>td[:-1]) and np.all(te[1:]>=te[:-1]) and np.all(te>td),'Training clocks')
    require(np.all(te.astype('datetime64[M]')==td.astype('datetime64[M]')+1),'Natural-next-month labels')
    offsets=train['offsets'];require(isinstance(offsets,np.ndarray) and offsets.dtype.kind in 'iu' and offsets.shape==(len(td)+1,) and offsets[0]==0 and np.all(np.diff(offsets)>0),'Whole-month offsets')
    rows=int(offsets[-1]);require(train['dates'].shape==train['ends'].shape==(rows,),'Flat training clock shapes')
    # Metadata determines the prefix before any financial value is accessed.
    cutoffs=[np.datetime64(str(x)[:4]+'-01-01')-np.timedelta64(1,'D') for x in dates[formal]]
    cutoffs.extend(dates[active]);max_cutoff=max(cutoffs)
    mature=np.flatnonzero(te<=max_cutoff);require(np.array_equal(mature,np.arange(len(mature))),'Mature contiguous training prefix')
    stop=int(offsets[len(mature)]);names=[]
    for k in mature:
        lo,hi=int(offsets[k]),int(offsets[k+1]);ss=tm['symbols'][k]
        require(type(ss) is list and len(ss)==hi-lo and len(set(ss))==len(ss) and all(type(s) is str and s for s in ss),'Mature security identities')
        require(np.all(train['dates'][lo:hi]==td[k]) and np.all(train['ends'][lo:hi]==te[k]),'Flat/monthly clock identity');names.extend(ss)
    def body(key,shape,dtype='float64'):
        raw=train[key];require(isinstance(raw,np.ndarray) and raw.shape==shape,key+' static shape')
        small=raw[:stop];require(small.dtype==np.dtype(dtype),key+' dtype');return small.copy()
    X=body('train_plain',(rows,10));good=body('common_good',(rows,),bool)
    targets={k:body('target_'+k,(rows,)) for k in TARGETS}
    result.update(td=train['dates'][:stop].copy(),te=train['ends'][:stop].copy(),names=names,X=X,good=good,targets=targets,stop=stop,max_cutoff=max_cutoff,training_months=len(mature))
    q=train['event_plain'];require(isinstance(q,np.ndarray) and q.ndim==2 and q.shape[0]>=n*width and q.shape[1]==10,'Query feature shape')
    rows_q=np.flatnonzero(np.repeat(formal,width));Q=np.zeros((n*width,10));selected=q[rows_q];require(selected.dtype==np.dtype('float64'),'Selected query dtype');Q[rows_q]=selected
    qdates=_days(train['event_dates'][:n*width],'Flat query dates');require(np.array_equal(qdates,np.repeat(dates,width)),'Flat query identity')
    result['Q']=Q
    return result


def _usable(inp):
    good=inp['good']&np.isfinite(inp['X']).all(axis=1)
    for target in inp['targets'].values():good &= np.isfinite(target)
    months=inp['td'].astype('datetime64[M]')
    for month in np.unique(months):
        ix=months==month
        if np.count_nonzero(good&ix)<2:good[ix]=False
    return good


def _fit_record(inp,kind,queries,cutoff,work):
    valid=inp['valid'];lower=lower_bound(cutoff)
    ids=np.flatnonzero(valid&(inp['td']>lower)&(inp['te']<=cutoff));months=inp['td'][ids].astype('datetime64[M]')
    count=len(np.unique(months));w=inp['width'];qids=np.asarray([i*w+j for i in queries for j in np.flatnonzero(inp['use'][i])],dtype=int)
    status='cold_start' if count<18 else 'fit' if len(qids) else 'no_prediction_rows'
    require(kind!='monthly' or status=='fit','Active monthly clock requires at least18 mature months and original U')
    row=dict(kind=kind,query_indices=list(queries),cutoff=str(cutoff),lower_bound=str(lower),training_row_ids=ids.tolist(),query_row_ids=qids.tolist(),
        training_months=count,status=status,signal_dates=[str(x) for x in inp['td'][ids]],label_end_dates=[str(x) for x in inp['te'][ids]],
        symbols=[inp['names'][i] for i in ids],weights=[],weights_sha256=None,
        mean=None,scale=None,beta=None,intercept=None,gram=None,rhs=None,raw_predictions=[])
    if kind=='annual':
        for i in queries:
            expected=inp['arrays']['eligible'][i]&np.isfinite(inp['Q'][i*w:(i+1)*w]).all(axis=1) if count>=18 else np.zeros(w,bool)
            exact_array(inp['arrays']['forecast_available'][i],expected,'Cohort-derived original availability')
    if status=='fit':
        X=inp['X'][ids];Q=inp['Q'][qids];require((np.abs(X)<=.5+1e-12).all() and np.isfinite(Q).all() and (np.abs(Q)<=.5+1e-12).all(),'Bounded selected plain10')
        weights=month_weights(inp['td'][ids],cutoff);numeric,pred=ridge_fit(X,inp['targets']['log_excess'][ids],Q,weights)
        row.update(numeric,weights=weights.tolist(),weights_sha256=hashlib.sha256(weights.tobytes()).hexdigest())
        work['canonical_solves']+=1;work['svd_solves']+=1;work['canonical_prediction_rows']+=len(qids);work['svd_prediction_rows']+=len(qids)
    else:pred=np.empty(0)
    return row,qids,pred


def ridge_fit(X,y,Q,weights):
    """Separate normal-equation and explicitly augmented SVD implementations."""
    require(X.ndim==Q.ndim==2 and X.shape[1]==Q.shape[1]==10 and y.shape==weights.shape==(len(X),),'Ridge dimensions')
    require(all(np.isfinite(x).all() for x in (X,y,Q,weights)) and (weights>0).all(),'Finite Ridge inputs')
    with np.errstate(over='raise',invalid='raise',divide='raise'),threadpool_limits(limits=1):
        mean=np.average(X,axis=0,weights=weights);centered=X-mean
        scale=np.sqrt(np.average(centered*centered,axis=0,weights=weights));scale=np.where(scale>1e-12,scale,1.)
        Z=centered/scale;intercept=float(np.average(y,weights=weights))
        gram=np.einsum('ni,n,nj->ij',Z,weights,Z,optimize=False);gram+=10.*np.eye(10)
        rhs=np.einsum('ni,n,n->i',Z,weights,y-intercept,optimize=False)
        beta=np.linalg.solve(gram,rhs);query=(Q-mean)/scale
        predictions=np.einsum('ni,i->n',query,beta,optimize=False)+intercept
        root=np.sqrt(weights)
        augmented=np.concatenate((Z*root[:,None],math.sqrt(10.)*np.eye(10)),axis=0)
        response=np.concatenate(((y-intercept)*root,np.zeros(10)))
        U,s,Vt=np.linalg.svd(augmented,full_matrices=False)
        require(np.isfinite(s).all() and (s>0).all(),'Full-rank augmented system')
        projected=np.einsum('ni,n->i',U,response,optimize=False)/s
        svd_beta=np.einsum('ij,i->j',Vt,projected,optimize=False)
        svd_prediction=np.einsum('ni,i->n',query,svd_beta,optimize=False)+intercept
    for name,a in [('mean',mean),('scale',scale),('beta',beta),('gram',gram),('rhs',rhs),('raw',predictions)]:require(np.isfinite(a).all(),'Finite '+name)
    require(math.isfinite(intercept),'Finite intercept')
    require(np.allclose(svd_beta,beta,atol=SVD_ATOL,rtol=SVD_RTOL),'Independent SVD coefficients')
    require(np.allclose(svd_prediction,predictions,atol=SVD_ATOL,rtol=SVD_RTOL),'Independent SVD predictions')
    return dict(mean=mean.tolist(),scale=scale.tolist(),beta=beta.tolist(),intercept=intercept,gram=gram.tolist(),rhs=rhs.tolist(),raw_predictions=predictions.tolist()),predictions


def _ranks(raw,use,fallback,work):
    result=fallback.copy()
    for i in range(len(result)):
        ids=np.flatnonzero(use[i])
        if len(ids):
            result[i,ids]=rank(raw[i,ids]);work['rank_evaluations']+=len(ids)
    return result


def _expected(inp,full,work):
    """Reconstruct records from dates and source values, never supplied records."""
    n,w=inp['n'],inp['width'];a=inp['arrays'];use=inp['use']
    raw=a['raw_source_log'].copy();available=a['forecast_available'].copy();old_rank=a['rank_log_excess']
    records=[];counts=dict.fromkeys(COUNT_FIELDS,0);counts.update(events=n,width=w)
    if inp['formal'].any():
        inp['valid']=_usable(inp)
        counts.update(training_rows=inp['stop'],valid_training_rows=int(inp['valid'].sum()))
        years=sorted(set(str(d)[:4] for d in inp['dates'][inp['formal']]))
        counts['annual_groups']=len(years)
        for year in years:
            queries=[i for i,d in enumerate(inp['dates']) if inp['formal'][i] and str(d).startswith(year)]
            row,ids,pred=_fit_record(inp,'annual',queries,np.datetime64(str(int(year)-1)+'-12-31'),work)
            if row['status']=='fit':
                exact_array(pred,raw.reshape(-1)[ids],'Reconstructed annual source raw')
                raw.reshape(-1)[ids]=pred;counts['annual_ridge_fits']+=1;counts['annual_prediction_rows']+=len(ids)
            records.append(row)
    ranks=_ranks(raw,use,old_rank,work);exact_array(ranks,old_rank,'Original complete U percentile ranks')
    control=dict(raw=raw,available=available,rank=ranks,audit=dict(fits=records))
    candidate=None
    if full:
        candidate_raw=raw.copy();monthly=[]
        for i in np.flatnonzero(inp['active']):
            row,ids,pred=_fit_record(inp,'monthly',[int(i)],inp['dates'][i],work)
            candidate_raw.reshape(-1)[ids]=pred;monthly.append(row)
            counts['monthly_ridge_fits']+=1;counts['monthly_prediction_rows']+=len(ids)
        candidate=dict(raw=candidate_raw,available=available.copy(),rank=_ranks(candidate_raw,use,old_rank,work),audit=dict(fits=monthly))
    counts['actual_model_fits']=counts['annual_ridge_fits']+counts['monthly_ridge_fits']
    targets=inp.get('targets',{k:np.empty(0,np.float64) for k in TARGETS})
    good=inp.get('good',np.empty(0,bool))
    labels=dict(target_keys=list(TARGETS),target_sha256={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in targets.items()},
        common_good_sha256=hashlib.sha256(good.tobytes()).hexdigest(),labels_unchanged=True,common_good_unchanged=True,
        original_row_order_preserved=True,recentered=False,raw_target_reconstructed=False)
    scope=dict(events=n,width=w,consumed_training_rows=inp.get('stop',0),training_months=inp.get('training_months',0),
        latest_needed_cutoff=str(inp['max_cutoff']) if 'max_cutoff' in inp else None,
        future_training_numeric_accessed=False,future_queries_accessed=False)
    audit=dict(schema='monthly_matured_clock273_model_audit_v1',counts=counts,annual_control_exact=True,
        original_ranking_population=True,paired_ready_preserved=True,local_models_reused=True,other_targets_not_fitted=True,
        full_clock_change=True,strict_pit=False,computes_returns=False,control_gate=None,label_preservation=labels,source_scope=scope)
    return dict(n=n,width=w,control=control,candidate=candidate,audit=audit)


def _control_receipt(inp,counts):
    work=_work();fits=counts['annual_ridge_fits'];rows=counts['annual_prediction_rows']
    work.update(training_rows_scanned=2*counts['training_rows'],canonical_solves=fits,svd_solves=fits,
        canonical_prediction_rows=rows,svd_prediction_rows=rows,rank_evaluations=int(inp['use'].sum()),output_cells_compared=3*inp['n']*inp['width'])
    return _receipt('control',inp['n'],inp['width'],work,MODEL_FLAGS)


def _verify_models(train,meta,arrays,bundle,n,full):
    require(type(bundle) is dict and set(bundle)=={'n','width','control','candidate','audit'},'Exact model bundle schema')
    inp=_header(train,meta,arrays,n)
    require(type(bundle['n']) is int and bundle['n']==inp['n'] and type(bundle['width']) is int and bundle['width']==inp['width'],'Whole physical model scope')
    work=_work();expected=_expected(inp,full,work)
    if full:
        # This validates the saved callback receipt's scope. Runtime sequencing is
        # enforced by the producer callback/host; this check cannot prove timing.
        exact(bundle['audit']['control_gate'],_control_receipt(inp,expected['audit']['counts']),'Annual prerequisite receipt')
        expected['audit']['control_gate']=bundle['audit']['control_gate']
    exact(bundle,expected,'Independent reconstructed model bundle')
    after=_header(train,meta,arrays,n)
    before={k:v for k,v in inp.items() if k!='valid'}
    exact(after,before,'Consumed source preservation')
    work['training_rows_scanned']=2*inp.get('stop',0)
    work['output_cells_compared']=(6 if full else 3)*inp['n']*inp['width']
    # Scanned rows count the two input-copy/integrity passes; canonical/SVD fit
    # rows recur per actual solve and are exposed in each exact fit record.
    return _receipt('models' if full else 'control',inp['n'],inp['width'],work,MODEL_FLAGS)


def verify_control_models(train,combined_meta,arrays,control_view,n=None):
    """Annual calibration callback: candidate and current gate must be None."""
    return _verify_models(train,combined_meta,arrays,control_view,n,False)


def verify_models(train,combined_meta,arrays,bundle,n=None):
    """Re-fit both clocks independently, with exact canonical and SVD checks."""
    return _verify_models(train,combined_meta,arrays,bundle,n,True)


def verify_delta(arrays,monthly_meta,bundle,actual_rank,actual_delta,actual_audit,n=None):
    """Validate a consumed prefix only; fit-record bodies are deliberately opaque.

    This checks transformation mathematics, not the provenance of model fits.
    Callers must separately bind a successful full/physical verify_models proof.
    """
    inp=_header(None,dict(monthly=monthly_meta),arrays,n,models=False)
    n,w=inp['n'],inp['width'];use=inp['use'];a=inp['arrays'];work=_work()
    require(type(bundle) is dict and set(bundle)=={'n','width','control','candidate','audit'},'Bundle schema')
    require(type(bundle['n']) is int and bundle['n']>=n and type(bundle['width']) is int and bundle['width']==w,'Cached prefix scope')
    active=use&inp['active'][:,None]
    for role in ('control','candidate'):
        f=bundle[role];require(type(f) is dict and set(f)=={'raw','available','rank','audit'},'Forecast schema')
        raw=_array(f['raw'],(bundle['n'],w),role+' raw','float64')[:n]
        available=_array(f['available'],(bundle['n'],w),role+' availability',bool)[:n]
        actual=_array(f['rank'],(bundle['n'],w),role+' rank','float64')[:n]
        exact_array(available,a['forecast_available'],role+' original availability')
        require(not np.isinf(raw).any() and np.isfinite(raw[use]).all(),'Complete finite forecast domain')
        computed=_ranks(raw,use,a['rank_log_excess'],work);exact_array(actual,computed,role+' complete U ranks')
        if role=='control':
            exact_array(raw,a['raw_source_log'],'Original raw exact');exact_array(actual,a['rank_log_excess'],'Original rank exact')
        else:
            exact_array(raw[~active],a['raw_source_log'][~active],'Inactive candidate raw bytes')
            exact_array(actual[~active],a['rank_log_excess'][~active],'Inactive candidate rank bytes')
            candidate=computed
    delta=np.zeros((n,w),np.float64)
    for i,j in zip(*np.nonzero(active)):
        delta[i,j]=.0375*(float(candidate[i,j])-float(a['rank_log_excess'][i,j]));work['delta_evaluations']+=1
    audit=dict(schema='monthly_matured_clock273_rank_delta_v1',events=n,width=w,ranking_cells=int(use.sum()),
        active_indices=[np.flatnonzero(row).tolist() for row in active],changed_cells=int(np.count_nonzero(delta)),
        original_ranking_population=True,paired_ready_preserved=True,local_basis_preserved=True,weak_only=True,
        partial_exact=True,calm_slot_preserved=True,no_post_rerank=True,computes_returns=False)
    exact_array(actual_rank,candidate,'Supplied candidate ranks');exact_array(actual_delta,delta,'Exact pooled delta')
    exact(actual_audit,audit,'Rank delta audit')
    work['output_cells_compared']=8*n*w
    return _receipt('delta',n,w,work,DELTA_FLAGS,model_fit_provenance_checked=False,original_pre_score_addition_checked=False)
