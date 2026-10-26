"""Independent rational labels, augmented weighted Ridge, ranks and accounting.

Pure synthetic/caller-supplied arrays only. No production190 imports, data loading,
model selection or replay entry point. Historical label reconstruction is distinct
from portfolio-return calculation. Frozen helpers load lazily after SHA checks.
"""
from collections import Counter
from fractions import Fraction
from pathlib import Path
import hashlib
import importlib.util
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
SOURCE_HELPER = ROOT/'work/monthly_endpoint155_prepare_review.py'
SOURCE_SHA = '0e594355beecc84a705517c8cd2774214e4300b5248e0c772a0306e73c38174f'
OUTCOME_HELPER = ROOT/'work/price_distribution171_independent.py'
OUTCOME_SHA = '94e2a984affad28ae1ba62de245519ffd9f4763c1a6ab3a5cd95dea02e0a1841'
ACCOUNTING = ROOT/'work/ptr169_results_independent.py'
ACCOUNTING_SHA = '6264571ad6ddacbc408449180c74a32b37659757c9bdfc5ce0e2672467607d59'
ARMS = ('incumbent_local67','severe_relative_loss_ridge','relative_loss_ridge')
TARGETS = ('candidate_target','control_target')
EXTRA = ('candidate_prediction','control_prediction','candidate_rank','control_rank','common_available')
LABELS = ('dates','ends','symbols',*TARGETS,'base_good','good')
DOSE = .01
ATOL = 1e-12


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(array): return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _module(path, expected, name):
    require(sha(path)==expected,'Frozen helper changed: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def frozen_source_oracle(): return _module(SOURCE_HELPER,SOURCE_SHA,'tail190_source_oracle')
def accounting(): return _module(ACCOUNTING,ACCOUNTING_SHA,'tail190_accounting')


def dependency_files():
    # The caller's PLAN must bind the full returned inherited source closure.
    inherited=frozen_source_oracle().source_files(ROOT)
    return tuple(sorted(set([Path(__file__).resolve(),OUTCOME_HELPER,ACCOUNTING,*inherited])))


def audit_outcomes(px, opening, orders, exports, results, meta):
    require(sha(ACCOUNTING)==ACCOUNTING_SHA,'Frozen accounting changed')
    helper=_module(OUTCOME_HELPER,OUTCOME_SHA,'tail190_isolated_outcomes')
    require(helper.ARMS==('incumbent_local67','price_activity_distribution','price_only_distribution'),'Outcome helper arms')
    helper.ARMS=ARMS
    result=helper.audit_outcomes(px,opening,orders,exports,results,meta)
    require(result['checked_paths']==9 and result['checked_slices']==45 and set(result['paths'])==set(ARMS),'Nine paths/five slices')
    result['outcome_helper_sha256']=OUTCOME_SHA
    return result


def close(actual,expected,name,atol=ATOL):
    a,b=np.asarray(actual,float),np.asarray(expected,float)
    require(a.shape==b.shape,name+': shape')
    require(not np.isinf(a).any() and not np.isinf(b).any(),name+': infinity')
    require(np.array_equal(np.isnan(a),np.isnan(b)),name+': missing mask')
    finite=np.isfinite(b)
    require(np.all(np.abs(a[finite]-b[finite])<=atol),name+': numerical difference')


def record_check(actual,expected,name='audit'):
    if isinstance(expected,dict):
        require(isinstance(actual,dict),name+': mapping')
        for key,value in expected.items():
            require(key in actual,name+': missing '+key)
            record_check(actual[key],value,name+'/'+key)
    elif isinstance(expected,list):
        require(isinstance(actual,list) and len(actual)==len(expected),name+': list')
        for i,(a,b) in enumerate(zip(actual,expected)): record_check(a,b,name+'/'+str(i))
    elif isinstance(expected,float):
        require(type(actual) in (float,int),name+': numeric scalar');close(actual,expected,name)
    else: require(type(actual) is type(expected) and actual==expected,name+': value/type')


def days(values,name):
    a=np.asarray(values)
    require(a.ndim==1,name+': vector')
    if not a.size:return np.array([],dtype='datetime64[D]')
    require(a.dtype.kind not in 'biufc',name+': nonnumeric dates')
    d=pd.DatetimeIndex(a)
    require(not d.hasnans and d.tz is None and d.equals(d.normalize()),name+': midnight dates')
    return d.to_numpy(dtype='datetime64[D]').copy()


def boolean(values,shape,name):
    a=np.asarray(values);require(a.shape==shape and a.dtype.kind=='b',name+': exact boolean shape')
    return a.copy()


def numeric(values,shape,name):
    a=np.asarray(values);require(a.shape==shape and a.dtype.kind in 'fiu',name+': real numeric shape')
    return a.astype(float,copy=True)


def calendar_map(sessions):
    axis=days(sessions,'sessions')
    require(len(axis)>0 and np.all(axis[1:]>axis[:-1]),'Ordered unique nonempty calendar')
    last={}
    for day in axis:last[day.astype('datetime64[M]')]=day
    return axis,last


def endpoint_schedule(sessions,signals,expected=None):
    _,last=calendar_map(sessions);signals=days(signals,'signals');out=[]
    for d in signals:
        m=d.astype('datetime64[M]')
        require(last.get(m)==d,'Official training month end required')
        require(m+1 in last,'Calendar must schedule next natural month')
        out.append(last[m+1])
    out=np.array(out,dtype='datetime64[D]')
    if expected is not None:require(np.array_equal(out,days(expected,'declared ends')),'Official next-month ends mismatch')
    return out


def flatten_meta(meta,name):
    require(isinstance(meta,dict) and all(k in meta for k in ('dates','ends','symbols')),name+': explicit metadata')
    d,e=days(meta['dates'],name+' dates'),days(meta['ends'],name+' ends')
    require(e.shape==d.shape and np.all(e>d),name+': end axes')
    require(not len(d) or np.all(d[1:]>d[:-1]),name+': unique ordered signals')
    pools=meta['symbols'];require(isinstance(pools,(list,tuple)) and len(pools)==len(d),name+': pools')
    widths=[];symbols=[]
    for row in pools:
        require(isinstance(row,(list,tuple,np.ndarray)),name+': pool vector')
        require(all(isinstance(x,(str,np.str_)) and x and x.strip()==x for x in row),name+': symbols')
        require(len(row)==len(set(row)),name+': duplicate symbol')
        widths.append(len(row));symbols.extend(row)
    return d,e,np.array(widths,int),np.array(symbols,str)


def exact_targets(stock_start,stock_end,spy_start,spy_end):
    values=(stock_start,stock_end,spy_start,spy_end)
    require(all(np.isfinite(v) and v>0 for v in values),'Four positive finite endpoints')
    s0,s1,m0,m1=(Fraction.from_float(float(v)) for v in values)
    relative=(s1/s0)/(m1/m0)
    return float(relative<Fraction(9,10)),float(relative<Fraction(1,1))


def labels_oracle(adj,sessions,dates,symbols,common_good,eligible,finite_features,expected_ends=None):
    require(isinstance(adj,pd.DataFrame) and isinstance(adj.index,pd.DatetimeIndex),'Price frame')
    axis=days(adj.index,'price dates')
    require(adj.columns.is_unique and (not len(axis) or np.all(axis[1:]>axis[:-1])),'Unique ordered price axes')
    d=days(dates,'signals');e=endpoint_schedule(sessions,d,expected_ends);n=len(d)
    s=np.asarray(symbols)
    require(s.shape==(n,) and all(isinstance(x,(str,np.str_)) and x for x in s),'Training symbols')
    require(len(set(zip(d.astype(str),s)))==n and (not n or np.all(d[1:]>=d[:-1])),'Unique ordered row identities')
    base=boolean(common_good,(n,),'common') & boolean(eligible,(n,),'eligible') & boolean(finite_features,(n,),'features')
    y,z=np.full(n,np.nan),np.full(n,np.nan)
    for i in np.flatnonzero(base):
        start,end=pd.Timestamp(d[i]),pd.Timestamp(e[i])
        if 'SPY' not in adj.columns or s[i] not in adj.columns or start not in adj.index or end not in adj.index:continue
        cells=(adj.at[start,s[i]],adj.at[end,s[i]],adj.at[start,'SPY'],adj.at[end,'SPY'])
        if any(np.asarray(v).ndim or np.asarray(v).dtype.kind not in 'fiu' for v in cells):continue
        cells=tuple(float(v) for v in cells)
        if not all(np.isfinite(v) and v>0 for v in cells):continue
        y[i],z[i]=exact_targets(*cells)
    good=base & np.isfinite(y) & np.isfinite(z);months=[]
    for day in np.unique(d):
        rows=d==day;count=int(good[rows].sum())
        if count<2:good[rows]=False
        months.append(dict(signal_date=str(day),label_end=str(e[np.flatnonzero(rows)[0]]),base_rows=int(base[rows].sum()),endpoint_rows=count,good_rows=int(good[rows].sum()),status='complete' if count>=2 else 'fewer_than_two_shared_rows'))
    require(np.all(y[np.isfinite(y)]<=z[np.isfinite(y)]),'Severe labels subset')
    label=dict(dates=d,ends=e,symbols=s.astype(str,copy=True),candidate_target=y,control_target=z,base_good=base,good=good)
    audit=dict(targets=list(TARGETS),rows=n,base_rows=int(base.sum()),good_rows=int(good.sum()),months=months,clipped=False,centered=False,
        relative_wealth_thresholds=dict(candidate=[9,10],control=[1,1]),exact_binary_float_rational=True,candidate_is_subset=True,four_positive_finite_endpoints=True,
        maturity='next official calendar month-end close',inherited_common_good_retained=True,no_endpoint_shortening=True,calendar_completeness_certified_by_this_function=False)
    return label,audit


def weights_oracle(dates,cutoff):
    d=days(dates,'weight dates');cutoff=np.datetime64(cutoff,'D')
    require(len(d)>0 and np.all(d<=cutoff),'Mature nonempty weights')
    months=d.astype('datetime64[M]');counts=Counter(months.astype(str))
    ages=cutoff.astype('datetime64[M]').astype(int)-months.astype(int)
    raw=np.array([1./counts[str(m)] for m in months])*np.exp2(-ages/12.)
    return raw*(len(raw)/raw.sum())


def fit_oracle(x,y,w):
    x,y,w=np.asarray(x,float),np.asarray(y,float),np.asarray(w,float)
    require(x.ndim==2 and x.shape[1]==10 and y.shape==w.shape==(len(x),),'Plain10 fit axes')
    require(len(x)>0 and np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(w).all() and (w>0).all(),'Finite fit inputs')
    mean=np.average(x,axis=0,weights=w)
    scale=np.sqrt(np.average((x-mean)**2,axis=0,weights=w));scale=np.where(scale>1e-12,scale,1.)
    intercept=float(np.average(y,weights=w))
    design=np.vstack(((x-mean)/scale*np.sqrt(w[:,None]),np.sqrt(10.)*np.eye(10)))
    target=np.r_[(y-intercept)*np.sqrt(w),np.zeros(10)]
    beta=np.linalg.lstsq(design,target,rcond=None)[0]
    require(all(np.isfinite(v).all() for v in (mean,scale,beta,np.asarray(intercept))),'Finite least-squares solution')
    return mean,scale,beta,intercept


def annual_oracle(labels,x,query_dates,query_x,query_good,query_monthend):
    require(set(labels)==set(LABELS),'Exact seven label arrays')
    d,e=days(labels['dates'],'training dates'),days(labels['ends'],'training ends')
    qd=days(query_dates,'queries');n,q=len(d),len(qd)
    x=numeric(x,(n,10),'train plain10');qx=numeric(query_x,(q,10),'query plain10')
    require(e.shape==(n,) and np.all(e>d) and np.all(e.astype('datetime64[M]')==d.astype('datetime64[M]')+1),'Next-month maturity')
    ys={k:numeric(labels[k],(n,),k) for k in TARGETS}
    good=boolean(labels['good'],(n,),'training good') & np.isfinite(x).all(axis=1)
    good &= np.isfinite(ys[TARGETS[0]]) & np.isfinite(ys[TARGETS[1]])
    require(all(np.isin(y[good],[0.,1.]).all() for y in ys.values()),'Two binary targets')
    require(np.all(ys[TARGETS[0]][good]<=ys[TARGETS[1]][good]),'Severe event subset')
    for day in np.unique(d):
        if np.count_nonzero(good & (d==day))<2:good[d==day]=False
    valid=boolean(query_good,(q,),'query good') & boolean(query_monthend,(q,),'query month-end') & np.isfinite(qx).all(axis=1)
    years=qd.astype('datetime64[Y]').astype(int)+1970
    pred={k:np.full(q,np.nan) for k in TARGETS};available=np.zeros(q,bool);annual=[];fits=0
    for year in np.unique(years):
        cutoff=np.datetime64('%04d-12-31'%(year-1),'D');lower=np.datetime64('%04d-12-31'%(year-6),'D')
        train=good & (d>lower) & (e<=cutoff);ids=np.flatnonzero(train);query=(years==year)&valid
        months=len(np.unique(d[train].astype('datetime64[M]')))
        status='cold_start' if months<18 else ('fit' if query.any() else 'no_queries')
        row=dict(year=int(year),cutoff=str(cutoff),lookback_exclusive_start=str(lower),training_rows=len(ids),training_months=months,prediction_rows=int(query.sum()),
            latest_label_end=str(e[train].max()) if len(ids) else None,training_indices=ids.tolist(),training_indices_sha256=digest(ids),training_dates_sha256=digest(d[train]),
            training_symbols_sha256=digest(labels['symbols'][train]),status=status,target_counts={k:int(ys[k][train].sum()) for k in TARGETS},
            target_prevalence={k:float(ys[k][train].mean()) if len(ids) else None for k in TARGETS})
        if status=='fit':
            require(np.all(np.abs(x[train])<=.5+1e-12) and np.all(np.abs(qx[query])<=.5+1e-12),'Frozen centered plain10 bounds')
            w=weights_oracle(d[train],cutoff)
            models={k:fit_oracle(x[train],ys[k][train],w) for k in TARGETS}
            for k,model in models.items():pred[k][query]=np.sum((qx[query]-model[0])/model[1]*model[2],axis=1)+model[3]
            require(all(np.isfinite(v[query]).all() for v in pred.values()),'Finite forecasts')
            available[query]=True;fits+=2
            row.update(weight_sum=float(w.sum()),weights_sha256=digest(w),scaler_mean_sha256=digest(models[TARGETS[0]][0]),scaler_scale_sha256=digest(models[TARGETS[0]][1]),
                target_sha256={k:digest(ys[k][train]) for k in TARGETS},weighted_target_prevalence={k:float(np.average(ys[k][train],weights=w)) for k in TARGETS},shared_rows_weights_scalers=True,model_fits=2)
        else:row['model_fits']=0
        annual.append(row)
    audit=dict(alpha=10.,half_life_months=12.,lookback_years=5,minimum_mature_months=18,new_pooled_model_fits=fits,local_model_fits=0,
        shared_training_mask_sha256=digest(good),training_only_standardization=True,calibrated_probabilities=False,target_clipping=False,target_centering=False,annual=annual)
    return dict(candidate_prediction=pred[TARGETS[0]],control_prediction=pred[TARGETS[1]],common_available=available),audit


def rank_oracle(values):
    values=np.asarray(values,float);n=len(values)
    require(values.ndim==1 and np.isfinite(values).all(),'Finite rank domain')
    if n<=1:return np.zeros(n)
    return np.array([(np.count_nonzero(values<v)+(np.count_nonzero(values==v)-1)/2)/(n-1)-.5 for v in values])


def independent_build(adj_close,sessions,trainmeta,meta,train_plain,event_plain,common_good,train_eligible,eligibility):
    td,te,tw,ts=flatten_meta(trainmeta,'training');qd,qe,qw,qs=flatten_meta(meta,'query')
    eligible=np.asarray(eligibility)
    require(eligible.ndim==2 and eligible.dtype.kind=='b' and eligible.shape[0]==len(qd),'Query boolean grid')
    require(np.all(qw==eligible.shape[1]),'Constant query width')
    d,e=np.repeat(td,tw),np.repeat(te,tw)
    x=numeric(train_plain,(len(d),10),'training plain10')
    calendar,last=calendar_map(sessions);members=set(calendar.tolist());monthend=[]
    for i,day in enumerate(qd):
        require(day.tolist() in members,'Query official session')
        valid=last[day.astype('datetime64[M]')]==day
        require(valid or (i==0 and str(day)=='2016-08-17'),'Only initial query partial allowed')
        monthend.append(valid)
    label,la=labels_oracle(adj_close,calendar,d,ts,common_good,train_eligible,np.isfinite(x).all(axis=1),e)
    with threadpool_limits(limits=1):
        forecast,ma=annual_oracle(label,x,np.repeat(qd,qw),event_plain,eligible.ravel(),np.repeat(np.array(monthend,bool),qw))
    shape=eligible.shape
    raw={k:forecast[k].reshape(shape) for k in EXTRA[:2]}
    mask=forecast['common_available'].reshape(shape) & eligible & np.isfinite(raw[EXTRA[0]]) & np.isfinite(raw[EXTRA[1]])
    values=dict(**raw,candidate_rank=np.zeros(shape),control_rank=np.zeros(shape),common_available=mask)
    for i in range(shape[0]):
        for prediction,rank in zip(EXTRA[:2],EXTRA[2:4]):values[rank][i,mask[i]]=rank_oracle(-values[prediction][i,mask[i]])
    ra=dict(common_counts=mask.sum(axis=1).astype(int).tolist(),population='signal-time eligible with both finite forecasts',
        mapping='average_rank0/(n-1)-0.5; both use negative prediction; n<=1 => 0',missing_correction=0,max_absolute_rank=.5,no_holdings_or_returns_computed=True,
        non_monthend_fallback_indices=np.flatnonzero(~np.array(monthend,bool)).tolist(),query_evaluation_ends_not_used_for_labels=True)
    return dict(arrays=values,labels=label,label_audit=la,model_audit=ma,rank_audit=ra)


def verify_build(adj_close,sessions,trainmeta,meta,train_plain,event_plain,common_good,train_eligible,eligibility,built):
    expected=independent_build(adj_close,sessions,trainmeta,meta,train_plain,event_plain,common_good,train_eligible,eligibility)
    require(set(built)==set(expected),'Exact build sections')
    for section in ('arrays','labels'):
        require(set(built[section])==set(expected[section]),'Exact '+section+' keys')
        for key,wanted in expected[section].items():
            actual=np.asarray(built[section][key]);require(actual.shape==wanted.shape and actual.dtype==wanted.dtype,section+'/'+key+': dtype/shape')
            if key in EXTRA[:2]:close(actual,wanted,key)
            else:np.testing.assert_array_equal(actual,wanted,err_msg=section+'/'+key)
    for section in ('label_audit','model_audit','rank_audit'):record_check(built[section],expected[section],section)
    arrays=expected['arrays'];shape=arrays['common_available'].shape;fits=expected['model_audit']['new_pooled_model_fits']
    errors=[float(np.max(np.abs(built['arrays'][k][np.isfinite(arrays[k])]-arrays[k][np.isfinite(arrays[k])]))) for k in EXTRA[:2] if np.isfinite(arrays[k]).any()]
    return dict(passed=True,audit_checked=True,independent_exact_rational_labels=True,endpoint_calendar_checked=True,shared_maturity_rows_weights_scalers_verified=True,
        annual_augmented_least_squares_verified=True,event_prevalence_checked=True,shared_query_mask_and_centered_ranks_verified=True,
        no_query_future_outcomes=True,initial_partial_fallback_checked=True,producer_modules_imported=False,events=shape[0],symbols_per_event=shape[1],
        training_rows=len(expected['labels']['dates']),good_training_rows=int(expected['labels']['good'].sum()),available_cells=int(arrays['common_available'].sum()),
        unavailable_cells=int((~arrays['common_available']).sum()),new_pooled_model_fits=fits,independent_model_fits=fits,local_model_fits=0,
        label_arrays_checked=7,forecast_arrays_checked=5,annual_records_checked=len(expected['model_audit']['annual']),max_forecast_error=max(errors,default=0.),
        computes_portfolio_returns=False,limitations=['Binary Ridge ranks are not calibrated probabilities.','Inherited common_good and historical data vintages remain.'])


def independent_deltas(values,dose=DOSE):
    require(dose==DOSE,'Fixed dose .01')
    require(set(values)==set(EXTRA),'Five forecast arrays')
    mask=np.asarray(values['common_available']);require(mask.ndim==2 and mask.dtype.kind=='b','Common boolean grid')
    a,b=(numeric(values[k],mask.shape,k) for k in EXTRA[:2])
    require(np.array_equal(np.isfinite(a),mask) and np.array_equal(np.isfinite(b),mask),'Shared finite predictions')
    require(not np.isinf(a).any() and not np.isinf(b).any(),'No infinite forecasts')
    result={}
    for key,prediction,rankkey in zip(('candidate','control'),(a,b),EXTRA[2:4]):
        ranks=numeric(values[rankkey],mask.shape,rankkey);expected=np.zeros(mask.shape)
        for i in range(mask.shape[0]):expected[i,mask[i]]=rank_oracle(-prediction[i,mask[i]])
        np.testing.assert_array_equal(ranks,expected)
        delta=dose*ranks
        if len(delta):delta[0]=0.
        result[key]=delta
    return result


def verify_deltas(values,deltas,dose=DOSE):
    expected=independent_deltas(values,dose)
    require(set(deltas)==set(expected),'Two deltas')
    for key,value in expected.items():
        actual=np.asarray(deltas[key]);require(actual.dtype==np.float64,'Float64 deltas')
        np.testing.assert_array_equal(actual,value)
    return dict(passed=True,common_query_population_checked=True,negative_prediction_ranks_checked=True,average_ties_checked=True,initial_delta_zero=True,
        common_available_only=True,no_new_eligibility_filter=True,missing_exact_zero=True,dose=DOSE,max_absolute_correction=.005,
        changed_cells={k:int(np.count_nonzero(v)) for k,v in expected.items()},computes_portfolio_returns=False)


def verify_selection(original,meta,deltas,scores,choices):
    require(set(deltas)=={'candidate','control'} and set(scores)==set(choices)==set(ARMS),'Three arms')
    helper=frozen_source_oracle().old.original_helper();x=helper.original_prefix(original,meta,None)
    errors=[]
    for arm,key in zip(ARMS,(None,'candidate','control')):
        pre=x['local'].copy()
        if key is not None:
            d=numeric(deltas[key],pre.shape,key);require(np.isfinite(d).all() and np.all(np.abs(d)<=.005),'Bounded deltas')
            require(not np.any(d[~x['eligible']]) and (not len(d) or not np.any(d[0])),'Ineligible/initial deltas zero')
            nonzero=d!=0.;pre[nonzero]+=d[nonzero]
        choice,score,details=helper.episode(pre,x)
        require(choices[arm]==choice,'Independent ordered choices '+arm)
        close(scores[arm],score,'Scores '+arm)
        require(all(set(v['top2'])<=set(v['selected']) if len(v['selected'])>=2 else True for v in details),'Original top-two lock')
        finite=np.isfinite(score)
        errors.append(float(np.max(np.abs(np.asarray(scores[arm])[finite]-score[finite]))) if finite.any() else 0.)
    return dict(passed=True,scores_and_choices_checked=True,one_original_episode_checked=True,initial_partial_fallback_checked=True,ordered_choices_checked=len(meta['dates'])*3,one_episode_per_arm=True,original_support_unchanged=True,
        independent_original_local67_pre=True,max_score_error=max(errors,default=0.),portfolio_returns_calculated=False)
