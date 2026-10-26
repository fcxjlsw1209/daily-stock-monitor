"""Pure caller-supplied frozen261 prefix comparisons; no data loads or fits.

TRUST CONTRACT: before either call the host verifies all frozen261 manifest/SHA
bindings and independently verifies the exact current bundle and source layout.
The required receipt is checked for stage/scope/flags/count consistency here;
it is not authenticated merely because a caller can supply a Python dictionary.
In particular annual_cohorts and original_u must come from that independently
verified source, never from an unverified candidate's status. The helper derives
prefix query IDs and fit status itself and never uses current status to skip a
reference model. No actual caller may substitute synthetic/no-callback receipts.

Reference mapping: forecasts/model_audit/labels/scores/pre_scores/orders/meta are
already decoded, pinned261 artifacts. New-tree structural validation/projection
reuses the producer's pure validator; its import verifies code pins only.
"""
from copy import deepcopy
import math
import hashlib
import numbers
import numpy as np

INITIAL_PARTIAL='2016-08-17'
REFERENCE_KEYS=('forecasts','model_audit','labels','scores','pre_scores','orders','meta')
FORECAST_KEYS=tuple(role+'_'+key for role in ('control','candidate') for key in ('raw','available','rank'))
LAYOUT_KEYS=('n','width','monthly','original_u','annual_cohorts')
COHORT_KEYS=('year','cutoff','lower_bound','training_row_ids','training_months')
RIDGE_KEYS=('year','cutoff','lower_bound','training_row_ids','query_row_ids','training_months','status','weights','mean','scale','beta','intercept','gram','rhs','raw_predictions')
TREE_KEYS=('year','cutoff','training_row_ids','query_row_ids','status','tree','raw_predictions','manual_predictions')
ARMS=('incumbent_local67','matched_pooled_ridge','weak_pooled_tree')
CALIBRATION_FLAGS=('original_training_identity_checked','annual_prior_december_cutoffs_checked','five_year_mature_cohort_checked','all_four_target_common_cohort_checked','canonical_matched_ridge_checked','independent_augmented_svd','original_model_cache_checked','original_query_rank_domain_checked','full_calibration_bundle_checked','no_future_training_values')
CONTROL_FLAGS=CALIBRATION_FLAGS+('all_tree_parameters_checked','independent_tree_input_pipeline','same_library_unconstrained_refit_checked','serialized_nodes_bins_exact','manual_unconstrained_predictions_exact','monotone_models_not_computed')
COUNT_FIELDS=('forecast_numeric_cells_compared','availability_cells_compared','label_values_compared','label_row_ids_compared','annual_metadata_rows_compared','annual_training_row_ids_compared','annual_query_row_ids_compared','ridge_models_compared','tree_models_compared','query_prediction_values_compared','score_numeric_cells_compared','ordered_events_compared','calibration_incumbent_score_cells_compared','calibration_incumbent_ordered_events_compared','verification_binding_checks','model_fits','returns_computed')

def need(ok,message):
    if not ok:raise ValueError(message)

def _keys(value,keys,label):need(type(value) is dict and set(value)==set(keys),label+' exact keys')
def _int(value,label,minimum=0):need(type(value) is int and value>=minimum,label+' native integer');return value

def _equal(actual,expected,label):
    """Typed JSON equality, including signed zero; no tolerance or coercion."""
    need(type(actual) is type(expected),label+' type')
    if type(actual) is float:need(actual.hex()==expected.hex(),label+' exact float bits')
    elif type(actual) is dict:
        need(set(actual)==set(expected),label+' keys')
        for key in expected:_equal(actual[key],expected[key],label+'/'+str(key))
    elif type(actual) in (list,tuple):
        need(len(actual)==len(expected),label+' length')
        for i,(a,b) in enumerate(zip(actual,expected)):_equal(a,b,label+'/'+str(i))
    elif actual is None or type(actual) in (str,int,bool):need(actual==expected,label+' value')
    else:raise ValueError(label+' unsupported structural type')

def _array_prefix(value,shape,stop,kind,label):
    # Inspect only ndarray structure before slicing. Never coerce the full future.
    need(isinstance(value,np.ndarray) and value.shape==shape,label+' static shape')
    part=value[:stop]
    if kind=='float':
        need(value.dtype in (np.dtype('float64'),np.dtype(object)),label+' binary64 dtype')
        if part.dtype==np.dtype(object):
            for v in part.flat:need(isinstance(v,numbers.Real) and not isinstance(v,(bool,np.bool_)),label+' real nonbool scalar')
            part=np.asarray(part,dtype=np.float64)
    elif kind=='bool':need(value.dtype==np.dtype(bool),label+' Boolean dtype')
    else:need(value.dtype.kind in 'iu' and value.dtype!=np.dtype(bool),label+' integer dtype')
    return part.copy()

def _array_equal(actual,expected,label):
    need(actual.shape==expected.shape and actual.dtype==expected.dtype,label+' shape/dtype')
    need(actual.tobytes()==expected.tobytes(),label+' exact bytes')

def _ids(values,label):
    need(type(values) is list and all(type(v) is int and v>=0 for v in values),label+' native IDs')
    need(all(a<b for a,b in zip(values,values[1:])),label+' sorted unique IDs');return list(values)

def _monthly(meta,n,w,label):
    need(type(meta) is dict and {'dates','ends','symbols','etfs'}<=set(meta),label+' metadata keys')
    total=len(meta['dates']);need(total>=n,label+' prefix available')
    need(all(type(meta[k]) is list and len(meta[k])==total for k in ('dates','ends','symbols')),label+' metadata shape')
    dates=meta['dates'][:n];ends=meta['ends'][:n];symbols=meta['symbols'][:n]
    need(all(type(d) is str and len(d)==10 for d in dates+ends),label+' day strings')
    dd=np.asarray(dates,dtype='datetime64[D]');ee=np.asarray(ends,dtype='datetime64[D]')
    need(not np.isnat(dd).any() and not np.isnat(ee).any() and np.all(ee>dd) and np.all(dd[1:]>dd[:-1]),label+' ordered clocks')
    need(all(type(row) is list and len(row)==w and all(type(s) is str for s in row) and len(set(row))==w for row in symbols),label+' unique symbols')
    need(type(meta['etfs']) is list and all(type(x) is str for x in meta['etfs']),label+' ETF identities')
    return total,dict(dates=list(dates),ends=list(ends),symbols=deepcopy(symbols),etfs=list(meta['etfs']))

def _annual_index(rows,years,label):
    need(type(rows) is list,label+' annual list');selected={};seen=set()
    for row in rows:
        need(type(row) is dict and type(row.get('year')) is int,label+' year header')
        year=row['year'];need(year not in seen,label+' unique year');seen.add(year)
        if year in years:selected[year]=row
    need(set(selected)==set(years),label+' consumed years present');return selected

def _layout(layout):
    _keys(layout,LAYOUT_KEYS,'Verified layout');n=_int(layout['n'],'events');w=_int(layout['width'],'width',1)
    total,meta=_monthly(layout['monthly'],n,w,'Verified source')
    U=_array_prefix(layout['original_u'],(total,w),n,'bool','Verified original U')
    years=sorted({int(d[:4]) for d in meta['dates'] if d!=INITIAL_PARTIAL})
    # No ordinary year: do not retrieve or inspect any annual numeric/body fields.
    source={} if not years else _annual_index(layout['annual_cohorts'],years,'Verified cohorts')
    out=[]
    for year in years:
        row=source[year];_keys(row,COHORT_KEYS,'Verified cohort')
        ids=_ids(row['training_row_ids'],'Cohort training');nm=_int(row['training_months'],'Training months')
        need(nm<=len(ids)//2,'At least two valid rows per training month')
        cutoff='%04d-12-31'%(year-1);lower='%04d-12-31'%(year-6)
        need(row['cutoff']==cutoff and row['lower_bound']==lower,'Original five-year annual bounds')
        qi=[i*w+j for i,d in enumerate(meta['dates']) if d!=INITIAL_PARTIAL and int(d[:4])==year for j in range(w) if U[i,j]]
        status='cold_start' if nm<18 else ('fit' if qi else 'no_prediction_rows')
        out.append(dict(year=year,cutoff=cutoff,lower_bound=lower,training_row_ids=ids,query_row_ids=qi,training_months=nm,status=status))
    return n,w,meta,out

def _receipt(receipt,current,n,stage):
    need(type(receipt) is dict and receipt.get('passed') is True,'Independent model receipt required')
    need('synthetic_no_callback' not in receipt and 'synthetic' not in receipt,'No synthetic receipt at historical reference gate')
    need(type(receipt.get('checked_events')) is int and receipt['checked_events']==n,'Independent receipt prefix scope')
    for flag in (CALIBRATION_FLAGS if stage=='calibration' else CONTROL_FLAGS):need(receipt.get(flag) is True,'Independent receipt flag:'+flag)
    for flag in ('full_bundle_supplied','full_calibration_supplied','full_label_preservation_supplied','full_model_audit_supplied'):need(receipt.get(flag) is True,'Independent receipt complete:'+flag)
    need(receipt.get('full_control_supplied') is (stage=='control') and receipt.get('full_candidate_supplied') is False and receipt.get('candidate_models_computed') is False,'Independent pre-candidate stage')
    need(receipt.get('computes_returns') is False and receipt.get('producer_modules_imported') is False,'Independent no-return provenance')
    _equal(receipt.get('counts'),current['audit']['counts'],'Independent current counts')
    _equal(receipt.get('producer_equivalent_counts'),current['audit']['counts'],'Independent equivalent counts')

class _VerifiedBinding:
    __slots__=('current','layout','receipt','stage','digest')
    def __init__(self,current,layout,receipt,stage,digest):
        object.__setattr__(self,'current',current);object.__setattr__(self,'layout',layout)
        object.__setattr__(self,'receipt',receipt);object.__setattr__(self,'stage',stage);object.__setattr__(self,'digest',digest)
    def __setattr__(self,*args):raise AttributeError('Immutable verification binding')

_ISSUED=[]

def _snapshot(current,layout,receipt,stage):
    n,w,meta,annual=_layout(layout);U=layout['original_u'][:n]
    h=hashlib.sha256()
    def visit(value):
        h.update(type(value).__name__.encode()+b':')
        if isinstance(value,np.ndarray):
            need(value.dtype.kind!='O','Verified current numeric arrays, no object bodies')
            h.update(value.dtype.str.encode());h.update(repr(value.shape).encode());h.update(value.tobytes())
        elif type(value) is dict:
            for key in sorted(value):visit(key);visit(value[key])
        elif type(value) in (list,tuple):
            h.update(str(len(value)).encode())
            for x in value:visit(x)
        elif type(value) is float:h.update(value.hex().encode())
        elif value is None or type(value) in (str,int,bool):h.update(repr(value).encode())
        else:raise ValueError('Unsupported verified snapshot type')
    visit((stage,n,w,meta,annual,U,current,receipt));return h.hexdigest()

def bind_verified_inputs(current_bundle,layout,independent_receipt,*,stage):
    """Call immediately after the host's independent gate, on those exact inputs.

    This is a mutation binding, not independent authentication or revalidation.
    The host must first validate the complete actual receipt and source layout.
    No reference body is accessed. Bindings cannot be copied/serialized/replayed.
    """
    need(stage in ('calibration','control'),'Explicit verified stage')
    n,w,_,_=_layout(layout);need(current_bundle.get('n')==n and current_bundle.get('width')==w,'Verified binding scope')
    _receipt(independent_receipt,current_bundle,n,stage)
    result=_VerifiedBinding(current_bundle,layout,independent_receipt,stage,_snapshot(current_bundle,layout,independent_receipt,stage))
    _ISSUED.append(result);return result

def _verify_binding(binding,current,layout,receipt,stage):
    need(type(binding) is _VerifiedBinding and any(binding is x for x in _ISSUED),'Locally issued verification binding')
    need(binding.current is current and binding.layout is layout and binding.receipt is receipt and binding.stage==stage,'Exact verified objects and stage')
    need(binding.digest==_snapshot(current,layout,receipt,stage),'Verified input changed after independent gate')

def _prepare(reference,current,layout,receipt,stage):
    _keys(reference,REFERENCE_KEYS,'Pinned261 reference');n,w,meta,annual=_layout(layout)
    need(type(current) is dict and current.get('schema')=='monthly_monotone_tree266_models_v1' and type(current.get('n')) is int and current['n']==n and type(current.get('width')) is int and current['width']==w,'Independently verified current exact prefix')
    need(current.get('monotone_candidate') is None,'Historical gate precedes candidate fits')
    need((current.get('unconstrained_control') is None)==(stage=='calibration'),'Exact historical model stage')
    _receipt(receipt,current,n,stage)
    total,old_meta=_monthly(reference['meta'],n,w,'Pinned261');_equal(meta,old_meta,'Pinned261 metadata prefix')
    _keys(reference['forecasts'],FORECAST_KEYS,'Pinned261 forecasts')
    _keys(reference['labels'],('training_row_ids','label_values'),'Pinned261 labels')
    need(type(reference['model_audit']) is dict and set(reference['model_audit'])=={'model','labels','control','candidate'},'Pinned261 model audit keys')
    counts=dict.fromkeys(COUNT_FIELDS,0)
    return n,w,total,meta,annual,counts

def _forecasts(reference,current,newrole,oldrole,n,w,total,counts):
    f=current[newrole];need(type(f) is dict and set(f)=={'raw','available','rank','audit'},'Current forecast schema')
    for key,kind in (('raw','float'),('available','bool'),('rank','float')):
        shape=(n,w) if key=='rank' else (n*w,);oldshape=(total,w) if key=='rank' else (total*w,);stop=n if key=='rank' else n*w
        a=_array_prefix(f[key],shape,stop,kind,'Current '+key);b=_array_prefix(reference['forecasts'][oldrole+'_'+key],oldshape,stop,kind,'261 '+key)
        _array_equal(a,b,'261 '+newrole+'/'+key)
        counts['availability_cells_compared' if key=='available' else 'forecast_numeric_cells_compared']+=a.size

def _labels(reference,current,annual,counts):
    if not annual:return
    ids=current['training_row_ids'];values=current['label_values']
    need(isinstance(ids,np.ndarray) and ids.ndim==1,'Current training ID axes');size=len(ids)
    a=_array_prefix(ids,(size,),size,'int','Current training IDs')
    old=reference['labels']['training_row_ids'];need(isinstance(old,np.ndarray) and old.ndim==1 and len(old)>=size,'261 label ID axes')
    b=_array_prefix(old,old.shape,size,'int','261 training IDs');_array_equal(a,b,'261 training IDs')
    a=_array_prefix(values,(size,),size,'float','Current labels');old=reference['labels']['label_values']
    b=_array_prefix(old,(len(reference['labels']['training_row_ids']),),size,'float','261 labels');_array_equal(a,b,'261 log-excess labels')
    counts['label_values_compared']+=size;counts['label_row_ids_compared']+=size

def _project_query(row,qi,label):
    oldids=_ids(row['query_row_ids'],label+' full query');positions={v:i for i,v in enumerate(oldids)}
    need(all(v in positions for v in qi),label+' prefix query subset')
    # Header list may be inspected whole; numerical prediction lists are indexed
    # only at the consumed IDs, never copied/coerced as a full-year array.
    result={}
    for key in ('raw_predictions','manual_predictions'):
        if key not in row:continue
        value=row[key];need(type(value) is list and len(value)==len(oldids),label+' '+key+' header')
        result[key]=[value[positions[q]] for q in qi]
        need(all(type(x) is float and math.isfinite(x) for x in result[key]),label+' finite prediction scalar')
    return result

def _annual(reference,current,newrole,oldrole,annual,counts):
    if not annual:return
    years=[r['year'] for r in annual];new=current[newrole]['audit'];old=reference['model_audit'][oldrole]
    suffix='ridge_audit' if newrole=='calibration_ridge' else 'tree_audit'
    need(type(new) is dict and set(new)=={'schema','annual'} and new['schema']=='monthly_monotone_tree266_'+suffix+'_v1','Current annual schema')
    need(type(old) is dict and set(old)=={'schema','annual'} and old['schema']=='monthly_weak_tree261_'+suffix+'_v1','261 annual schema')
    aa=_annual_index(new['annual'],years,'Current');bb=_annual_index(old['annual'],years,'261')
    need(len(new['annual'])==len(years),'Current only consumed annuals')
    ridge=newrole=='calibration_ridge';keys=RIDGE_KEYS if ridge else TREE_KEYS
    for layout in annual:
        year=layout['year'];a=aa[year];b=bb[year];_keys(a,keys,'Current annual');_keys(b,keys,'261 annual')
        for key in ('year','cutoff','training_row_ids')+ (('lower_bound','training_months') if ridge else ()):
            _equal(a[key],layout[key],'Current annual '+key);_equal(b[key],layout[key],'261 annual '+key)
        _equal(a['query_row_ids'],layout['query_row_ids'],'Current query IDs');_equal(a['status'],layout['status'],'Current prefix status')
        counts['annual_metadata_rows_compared']+=1;counts['annual_training_row_ids_compared']+=len(layout['training_row_ids']);counts['annual_query_row_ids_compared']+=len(layout['query_row_ids'])
        if layout['status']!='fit':
            need(a['raw_predictions']==[],'No prefix predictions')
            if ridge:
                need(a['weights']==[] and all(a[k] is None for k in ('mean','scale','beta','intercept','gram','rhs')),'No prefix Ridge fit')
                # Source cohort supports training even when prefix has no query;
                # do not touch full-year coefficient/weight/prediction bodies.
                need(b['status']==('cold_start' if layout['status']=='cold_start' else 'no_prediction_rows') or (layout['status']=='no_prediction_rows' and b['status']=='fit'),'261 supported nonfit projection')
            else:
                need(a['tree'] is None and a['manual_predictions']==[],'No prefix tree fit')
                need(b['status']==layout['status'] or (layout['status']=='no_prediction_rows' and b['status']=='fit'),'261 supported tree projection')
            continue
        need(b['status']=='fit','261 fitted year required')
        projected=_project_query(b,layout['query_row_ids'],'261 '+newrole)
        _equal(a['raw_predictions'],projected['raw_predictions'],'261 prefix raw predictions');counts['query_prediction_values_compared']+=len(layout['query_row_ids'])
        if ridge:
            for key in ('weights','mean','scale','beta','intercept','gram','rhs'):_equal(a[key],b[key],'261 Ridge '+key)
            counts['ridge_models_compared']+=1
        else:
            from monthly_monotone_tree266_kernel import legacy_unconstrained_tree
            projected_tree=legacy_unconstrained_tree(a['tree'])
            _equal(projected_tree,b['tree'],'261 complete fourteen-field tree')
            _equal(a['manual_predictions'],projected['manual_predictions'],'261 prefix manual predictions')
            counts['query_prediction_values_compared']+=len(layout['query_row_ids']);counts['tree_models_compared']+=1

def _strategy(reference,n,w,total,arm,orders,scores,pre_scores,counts,calibration):
    for key in ('orders','scores','pre_scores'):_keys(reference[key],ARMS,'261 '+key)
    need(type(orders) is list and len(orders)==n,'Current ordered events')
    old=reference['orders'][arm];need(type(old) is list and len(old)==total,'261 ordered event headers')
    _equal(orders,old[:n],'261 exact ordered choices');counts['ordered_events_compared']+=n
    for field,current in (('scores',scores),('pre_scores',pre_scores)):
        a=_array_prefix(current,(n,w),n,'float','Current '+field);b=_array_prefix(reference[field][arm],(total,w),n,'float','261 '+field)
        _array_equal(a,b,'261 '+field);counts['score_numeric_cells_compared']+=a.size
        if calibration:
            inc=_array_prefix(reference[field]['incumbent_local67'],(total,w),n,'float','261 incumbent '+field)
            _array_equal(b,inc,'261 Ridge/incumbent '+field);counts['calibration_incumbent_score_cells_compared']+=b.size
    if calibration:
        inc=reference['orders']['incumbent_local67'];need(type(inc) is list and len(inc)==total,'261 incumbent event headers')
        _equal(old[:n],inc[:n],'261 Ridge/incumbent ordered choices');counts['calibration_incumbent_ordered_events_compared']+=n

def _compare(reference,current,layout,independent_receipt,verification_binding,orders,scores,pre_scores,stage):
    _verify_binding(verification_binding,current,layout,independent_receipt,stage)
    n,w,total,meta,annual,counts=_prepare(reference,current,layout,independent_receipt,stage)
    counts['verification_binding_checks']=1
    calibration=stage=='calibration';newrole='calibration_ridge' if calibration else 'unconstrained_control';oldrole='control' if calibration else 'candidate'
    _forecasts(reference,current,newrole,oldrole,n,w,total,counts)
    if calibration:_labels(reference,current,annual,counts)
    _annual(reference,current,newrole,oldrole,annual,counts)
    _strategy(reference,n,w,total,'matched_pooled_ridge' if calibration else 'weak_pooled_tree',orders,scores,pre_scores,counts,calibration)
    return dict(schema='monthly_monotone_tree266_historical_'+stage+'_reference_v1',passed=True,checked_events=n,width=w,reference_role=oldrole,current_role=newrole,
      source_layout_independently_verified_by_caller=True,reference_manifest_verified_by_caller=True,receipt_authentication_performed=False,
      original_query_domain_used=True,prefix_fit_state_from_verified_source=True,full_annual_fit_not_borrowed=True,exact_bits_and_types=True,
      numeric_tolerance=0.,candidate_models_computed=False,model_fits=0,computes_returns=False,counts=counts)

def compare_calibration_reference(reference,current_bundle,layout,*,independent_receipt,verification_binding,orders,scores,pre_scores):
    return _compare(reference,current_bundle,layout,independent_receipt,verification_binding,orders,scores,pre_scores,'calibration')

def compare_control_reference(reference,current_bundle,layout,*,independent_receipt,verification_binding,orders,scores,pre_scores):
    return _compare(reference,current_bundle,layout,independent_receipt,verification_binding,orders,scores,pre_scores,'control')
