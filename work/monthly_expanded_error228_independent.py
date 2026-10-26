"""Independent risk-only source adapter checker; no model/target computation.

Reconstructs detached arrays and historical-use semantics from caller inputs.
Caller snapshots before/after producer execution remain necessary: a post-hoc
checker cannot prove that a producer did not mutate inputs before this call.
"""
from datetime import date
from pathlib import Path
import copy
import math
import numpy as np

TARGETS=('simple_excess','log_excess','vol_scaled','loss_utility')
RAW=('source_log','local_constant','local_linear')
COUNT_FIELDS=('training_rows','query_rows','original_forecast_rows','expanded_forecast_rows',
 'added_forecast_rows','original_label_rows','added_label_rows','extended_label_rows',
 'original_history_rows','expanded_history_rows')
FLAGS=('passed','independent_adapter_rebuild','exact_training_identity','exact_query_identity',
 'source227_metadata_exact','original_train_fields_preserved','old_forecasts_byte_exact',
 'old_labels_byte_exact','original_common_good_preserved','only_ineligible_labels_added',
 'expanded_masks_checked','historical_use_or_expansion','finite_used_values_checked','outputs_detached',
 'input_bytes_unchanged_by_checker','dtype_shape_bytes_checked','audit_checked','no_numerical_recomputation')
INITIAL='2016-08-17'


def dependency_files():return (Path(__file__).resolve(),)


def require(ok,message):
    if not ok:raise ValueError(message)


def exact(a,b,label):
    if isinstance(b,np.ndarray):
        require(isinstance(a,np.ndarray) and a.dtype==b.dtype and a.shape==b.shape,label+': dtype/shape')
        require(a.tobytes(order='C')==b.tobytes(order='C'),label+': bytes')
    elif isinstance(b,dict):
        require(isinstance(a,dict) and set(a)==set(b),label+': keys')
        for key in b:exact(a[key],b[key],label+'/'+str(key))
    elif isinstance(b,(list,tuple)):
        require(type(a) is type(b) and len(a)==len(b),label+': sequence')
        for i,(x,y) in enumerate(zip(a,b)):exact(x,y,label+'/'+str(i))
    elif isinstance(b,np.generic):
        require(type(a) is type(b) and a.tobytes()==b.tobytes(),label+': scalar bytes')
    else:
        require(type(a) is type(b) and a==b,label+': scalar')


def array(value,shape,kind,label):
    require(isinstance(value,np.ndarray) and value.shape==shape and value.dtype.kind in kind,label+': array schema')
    require(value.dtype.kind!='O',label+': object array unsupported')
    return value


def iso(v):
    require(type(v) is str,'Canonical ISO date required')
    try:parsed=date.fromisoformat(v)
    except (TypeError,ValueError) as e:raise ValueError('Invalid ISO date') from e
    require(parsed.isoformat()==v,'Canonical ISO date required');return parsed


def axis(meta,width=None,query=False):
    require(isinstance(meta,dict) and {'dates','ends','symbols'}<=set(meta),'Explicit identity metadata required')
    ds,es,ss=(meta[k] for k in ('dates','ends','symbols'))
    require(all(type(v) is list for v in (ds,es,ss)) and len(ds)==len(es)==len(ss),'Metadata axes')
    flat=[];offsets=[0];last=None
    for i,(d,e,names) in enumerate(zip(ds,es,ss)):
        day,end=iso(d),iso(e);m=day.year*12+day.month;em=end.year*12+end.month
        partial=query and i==0 and d==INITIAL and e=='2016-08-31'
        require(end>day and (em==m+1 or partial),'Natural-month label end')
        if i:
            initial_pair=query and i==1 and ds[0]==INITIAL and d=='2016-08-31'
            require(d>ds[i-1] and (m>last or initial_pair),'Unique increasing signal months')
        require(type(names) is list and (width is None or len(names)==width),'Identity width')
        require(all(type(s) is str and s and s==s.strip() for s in names) and len(names)==len(set(names)),'Exact unique security IDs')
        flat.extend((d,s,e) for s in names);offsets.append(len(flat));last=m
    require(len(flat)==len(set(flat)),'Unique date/security/end identities')
    return flat,offsets


def raw_array(source,key,q,available,label):
    a=array(source[key],(q,),'f',label)
    require(not np.isinf(a).any() and np.array_equal(np.isfinite(a),available),label+': finite/available domain')
    return a


def arrays_in(value):
    if isinstance(value,np.ndarray):return [value]
    if isinstance(value,dict):return [x for v in value.values() for x in arrays_in(v)]
    if isinstance(value,(list,tuple)):return [x for v in value for x in arrays_in(v)]
    return []


def rebuild(original_train,training_meta,query_meta,old206,new227,labels227,original_use,*,source_meta227):
    """Independent identity and byte reconstruction, with no forecast arithmetic."""
    require(isinstance(original_train,dict),'Original training dictionary required')
    require(isinstance(original_use,np.ndarray) and original_use.dtype.kind=='b' and original_use.ndim==2 and original_use.shape[1]>0,'Original U bool matrix')
    n,w=original_use.shape;q=n*w
    qaxis,_=axis(query_meta,w,True);tax,offsets=axis(training_meta);t=len(tax)
    require(len(qaxis)==q,'Query identity count')
    require(isinstance(source_meta227,dict) and {'monthly','training'}<=set(source_meta227),'Source227 metadata required')
    exact(source_meta227['monthly'],query_meta,'Source227 monthly metadata')
    exact(source_meta227['training'],training_meta,'Source227 training metadata')
    needed={'offsets','dates','ends','train_plain','common_good','train_eligible'}|{'target_'+k for k in TARGETS}
    require(needed<=set(original_train),'Original training fields')
    for k,v in original_train.items():require(isinstance(v,np.ndarray) and v.dtype.kind!='O','Original training array field '+k)
    off=array(original_train['offsets'],(len(offsets),),'iu','Original offsets')
    require(off.tolist()==offsets,'Training offsets/identities')
    for name,pos in (('dates',0),('ends',2)):
        a=array(original_train[name],(t,),'UMS','Original '+name)
        require([str(x) for x in a]==[r[pos] for r in tax],'Flat training '+name+' alignment')
    oldg=array(original_train['common_good'],(t,),'b','Original common_good')
    olde=array(original_train['train_eligible'],(t,),'b','Original train_eligible')
    X=array(original_train['train_plain'],(t,10),'fiu','Original plain10')
    require(not np.any(oldg & ~olde),'Original common labels must be eligible')
    require(isinstance(old206,dict) and set(old206)=={'raw_'+k for k in (*RAW,'candidate','control')}|{'available','local_available'},'Original206 exact7 schema')
    require(isinstance(new227,dict) and set(new227)=={'raw_'+k for k in RAW}|{'available','local_available','expansion_allowed'},'Source227 exact6 schema')
    olda=array(old206['available'],(q,),'b','Original available');newa=array(new227['available'],(q,),'b','Expanded available')
    oldl=array(old206['local_available'],(q,),'b','Original local availability');newl=array(new227['local_available'],(q,),'b','Expanded local availability')
    allow=array(new227['expansion_allowed'],(q,),'b','Expansion allowed')
    require(not np.any(olda & ~newa),'Cannot remove original available forecasts')
    addedq=newa & ~olda
    require(not np.any(addedq & ~allow),'Added queries require admitted warmup')
    require(not np.any(oldl & ~olda) and not np.any(newl & ~newa),'Local masks subset available')
    exact(newl[~addedq],oldl[~addedq],'Original local mask bytes')
    require(not np.any(original_use.reshape(-1)&~olda),'Original U must lie within original availability')
    for k in (*RAW,'candidate','control'):raw_array(old206,'raw_'+k,q,olda,'Original raw '+k)
    for k in RAW:
        a=raw_array(new227,'raw_'+k,q,newa,'Expanded raw '+k)
        require(a.dtype==old206['raw_'+k].dtype,'Original forecast dtype preserved')
        exact(a[~addedq],old206['raw_'+k][~addedq],'Original forecast bytes '+k)
    label_keys={'target_'+k for k in TARGETS}|{'common_good','added','vol63','dates','ends','offsets'}
    require(isinstance(labels227,dict) and set(labels227)==label_keys,'Source227 exact10 label schema')
    for k in ('dates','ends','offsets'):exact(labels227[k],original_train[k],'Label identity '+k)
    good=array(labels227['common_good'],(t,),'b','Expanded common_good')
    added=array(labels227['added'],(t,),'b','Added label mask')
    array(labels227['vol63'],(t,),'fiu','Admitted source volatility')
    require(not np.any(added & olde) and not np.any(added & oldg),'Only previously ineligible labels may be added')
    require(np.array_equal(good,oldg|added),'Original common_good OR added only')
    require(np.isfinite(X[good]).all(),'Finite common plain10')
    train={k:v.copy() for k,v in original_train.items()}
    train['common_good']=good.copy()
    for name in TARGETS:
        k='target_'+name;old=array(original_train[k],(t,),'f','Original target '+name)
        new=array(labels227[k],(t,),'f','Expanded target '+name)
        require(new.dtype==old.dtype,'Target dtype preserved')
        exact(new[~added],old[~added],'Original target bytes '+name)
        require(np.isfinite(old[oldg]).all() and np.isfinite(new[good]).all(),'Finite admitted target '+name)
        train[k]=new.copy()
    source={'raw_local_linear':new227['raw_local_linear'].copy(),'available':newa.copy()}
    use=original_use.copy();use|=addedq.reshape((n,w))
    counts=dict(training_rows=t,query_rows=q,original_forecast_rows=int(olda.sum()),expanded_forecast_rows=int(newa.sum()),
        added_forecast_rows=int(addedq.sum()),original_label_rows=int(oldg.sum()),added_label_rows=int(added.sum()),
        extended_label_rows=int(good.sum()),original_history_rows=int(original_use.sum()),expanded_history_rows=int(use.sum()))
    audit=dict(method='expanded_same_query_error_history_adapter',train_output_role='risk_history_only_never_Ridge_training',
        historical_use_rule='original_use OR newly_available227',original_identity_exact=True,old_forecasts_exact=True,
        old_labels_exact=True,original_common_good_preserved=True,added_labels_only_ineligible=True,historical_use_or_expansion=True,
        outputs_detached=True,source_forecasts_refitted=False,labels_recomputed=False,numerical_risk_estimates_computed=False,
        features_recomputed=False,portfolio_returns_computed=False,counts=counts)
    return dict(train=train,source=source,historical_use=use,audit=audit)


def check_adapter(original_train,training_meta,query_meta,old206,new227,labels227,original_use,adapted,*,source_meta227):
    inputs=(original_train,training_meta,query_meta,old206,new227,labels227,original_use,source_meta227)
    before=copy.deepcopy(inputs)
    expected=rebuild(*inputs[:-1],source_meta227=source_meta227)
    exact(adapted,expected,'Expanded risk-only adapter')
    input_arrays=arrays_in(inputs)
    for out in arrays_in(adapted):
        require(all(not np.shares_memory(out,src) for src in input_arrays),'Outputs must be detached from every input')
    exact(inputs,before,'Checker input preservation')
    receipt=dict(**{k:True for k in FLAGS},counts=expected['audit']['counts'],numerical_tolerance=0.,
        producer_modules_imported=False,computes_returns=False,computes_predictions=False,computes_labels=False,
        computes_risk_estimates=False,new_model_fits=0,
        mutation_scope='checker inputs unchanged; producer mutation requires host before/after snapshots')
    require_receipt(receipt,expected['audit'])
    return receipt


def require_receipt(check,adapter_audit):
    require(isinstance(check,dict) and isinstance(adapter_audit,dict),'Adapter receipt dictionaries required')
    for k in FLAGS:require(check.get(k) is True,'Missing/false independent adapter flag '+k)
    require(adapter_audit.get('method')=='expanded_same_query_error_history_adapter'
        and adapter_audit.get('train_output_role')=='risk_history_only_never_Ridge_training'
        and adapter_audit.get('historical_use_rule')=='original_use OR newly_available227','Risk-only adapter provenance')
    for k in ('original_identity_exact','old_forecasts_exact','old_labels_exact','original_common_good_preserved',
              'added_labels_only_ineligible','historical_use_or_expansion','outputs_detached'):
        require(adapter_audit.get(k) is True,'Adapter audit scope '+k)
    for k in ('source_forecasts_refitted','labels_recomputed','numerical_risk_estimates_computed','features_recomputed','portfolio_returns_computed'):
        require(adapter_audit.get(k) is False,'Adapter audit calculation scope '+k)
    counts=adapter_audit.get('counts')
    require(isinstance(counts,dict) and set(counts)==set(COUNT_FIELDS),'Adapter count schema')
    require(all(type(v) is int and v>=0 for v in counts.values()),'Nonnegative exact integer counts')
    require(counts['original_forecast_rows']+counts['added_forecast_rows']==counts['expanded_forecast_rows']<=counts['query_rows'],'Forecast count identity')
    require(counts['original_label_rows']+counts['added_label_rows']==counts['extended_label_rows']<=counts['training_rows'],'Label count identity')
    require(counts['original_history_rows']+counts['added_forecast_rows']==counts['expanded_history_rows']<=counts['expanded_forecast_rows'],'Expanded history count identity')
    exact(check.get('counts'),counts,'Independent adapter counts')
    require(type(check.get('numerical_tolerance')) is float and check['numerical_tolerance']==0.,'Exact adapter tolerance')
    for k in ('producer_modules_imported','computes_returns','computes_predictions','computes_labels','computes_risk_estimates'):
        require(check.get(k) is False,'Independent scope '+k)
    require(type(check.get('new_model_fits')) is int and check['new_model_fits']==0,'No new model fits')
    require(check.get('mutation_scope')=='checker inputs unchanged; producer mutation requires host before/after snapshots','Honest mutation scope')
    return True
