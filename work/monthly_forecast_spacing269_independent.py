"""Scalar forecast-spacing verification; no producer imports or source/return I/O.

The caller authenticates supplied arrays. This verifies all consumed numeric
cells, exact output bytes and every audit field, not historical provenance.
"""
import math
from datetime import date
import numpy as np

ARMS=('incumbent_local67','pooled_affine','mixed_affine')
FLAGS=('source_rank_formula_exact','fixed_slot_weights_exact','active_scope_and_fallback_exact','affine_spacing_and_endpoints_exact','pre_float64_bytes_exact','audit_schema_counts_events_exact','input_arrays_unmodified')
WORK_FIELDS=('transform_calls','event_rows','use_cells','active_rows','consumed_raw_cells','source_rank_checks','affine_evaluations','delta_evaluations','output_cells_compared','audit_event_checks','new_model_fits')
COUNT_FIELDS=('events','symbols_per_event','initial_partial_events','weak_events','ready_events','use_cells','active_events','active_cells','pooled_rank_checks','local_rank_checks','pooled_mapped_cells','local_mapped_cells','pooled_identity_rows','local_identity_rows','pooled_affine_rows','local_affine_rows','pooled_changed_rank_cells','local_changed_rank_cells','pooled_nonzero_delta_cells','local_nonzero_delta_cells','nonzero_delta_cells','changed_pre_cells','transform_calls','new_model_fits','predictive_model_fits')
AUDIT_KEYS=('schema','mode','counts','events','original_inputs_preserved','nonconsumed_pre_preserved','initial_partial_exact','nonweak_exact','not_ready_exact','consumed_source_ranks_checked','local_weight_fixed','no_rerank','no_new_budget','source_files_authenticated','computes_returns')

def need(ok,message):
    if not ok:raise ValueError(message)

def finite(value,label):
    value=float(value);need(math.isfinite(value),'Finite '+label);return value

def samefloat(a,b):return np.float64(a).tobytes()==np.float64(b).tobytes()

def exact(a,b,path='audit'):
    need(type(a) is type(b),'Typed '+path)
    if type(a) is dict:
        need(set(a)==set(b),'Keys '+path)
        for k in a:exact(a[k],b[k],path+'.'+k)
    elif type(a) in (list,tuple):
        need(len(a)==len(b),'Length '+path)
        for i,(x,y) in enumerate(zip(a,b)):exact(x,y,path+'/'+str(i))
    elif type(a) is float:need(samefloat(a,b),'Float bits '+path)
    else:need(a==b,'Value '+path)

def verify_transform(pre,pooled_raw,pooled_rank,local_raw,local_rank,use,ready,weak,dates,mode,actual_pre,actual_audit):
    need(type(mode) is str and mode in ARMS,'Registered mode')
    need(isinstance(pre,np.ndarray) and pre.dtype==np.dtype('float64') and pre.ndim==2,'pre float64 matrix')
    n,w=pre.shape;need(w>0,'Positive width')
    arrays=(pre,pooled_raw,pooled_rank,local_raw,local_rank,use,ready,weak)
    for x in arrays[1:5]:need(isinstance(x,np.ndarray) and x.dtype==np.dtype('float64') and x.shape==(n,w),'Float matrices')
    need(isinstance(use,np.ndarray) and use.dtype==np.dtype(bool) and use.shape==(n,w),'Use bool matrix')
    for x in (ready,weak):need(isinstance(x,np.ndarray) and x.dtype==np.dtype(bool) and x.shape==(n,),'State bool vector')
    need(isinstance(actual_pre,np.ndarray) and actual_pre.dtype==np.dtype('float64') and actual_pre.shape==(n,w),'Output float64 matrix')
    need(type(dates) in (list,tuple) and len(dates)==n,'Date axis')
    previous=None
    for i,d in enumerate(dates):
        need(type(d) is str,'Native date');parsed=date.fromisoformat(d)
        need(parsed.isoformat()==d and (previous is None or d>previous),'Increasing ISO dates')
        need(d!='2016-08-17' or i==0,'Named partial index0');previous=d
    before=[x.tobytes() for x in arrays];out=pre.copy();counts=dict.fromkeys(COUNT_FIELDS,0)
    counts.update(events=n,symbols_per_event=w,transform_calls=1)
    work=dict.fromkeys(WORK_FIELDS,0);work.update(transform_calls=1,event_rows=n,output_cells_compared=n*w,audit_event_checks=n)
    events=[]
    for i,d in enumerate(dates):
        ids=[j for j in range(w) if bool(use[i,j])];partial=d=='2016-08-17';isweak=bool(weak[i]);isready=bool(ready[i]);u=len(ids)
        counts['use_cells']+=u;work['use_cells']+=u
        counts['initial_partial_events']+=int(partial);counts['weak_events']+=int(isweak);counts['ready_events']+=int(isready)
        branch=('initial_partial' if partial else 'incumbent' if mode==ARMS[0] else 'nonweak' if not isweak else 'not_ready' if not isready else 'empty_use' if not ids else 'active')
        event=dict(query_index=i,date=d,initial_partial=partial,weak=isweak,ready=isready,use_indices=ids,branch=branch,pooled=None,local=None,delta=[],changed_pre_indices=[])
        if branch=='active':
            counts['active_events']+=1;counts['active_cells']+=u;work['active_rows']+=1
            maps=[]
            for name,raw,rank,weight in [('pooled',pooled_raw,pooled_rank,.0375)]+([('local',local_raw,local_rank,.0125)] if mode=='mixed_affine' else []):
                xs=[finite(raw[i,j],name+' raw') for j in ids];rs=[finite(rank[i,j],name+' rank') for j in ids]
                work['consumed_raw_cells']+=u;work['source_rank_checks']+=1;work['delta_evaluations']+=u
                counts[name+'_rank_checks']+=1;counts[name+'_mapped_cells']+=u
                expected=[]
                for x in xs:
                    lower=sum(y<x for y in xs);equal=sum(y==x for y in xs)
                    expected.append(((float(2*lower+equal+1)/2.)/float(u))-.5)
                need(all(samefloat(a,b) for a,b in zip(rs,expected)),name+' original centered average pct ranks')
                levels=len(set(xs));lo=min(xs);hi=max(xs);rl=min(rs);rh=max(rs);mapped=[]
                identity=levels<=2;counts[name+('_identity_rows' if identity else '_affine_rows')]+=1
                if not identity:
                    span=finite(hi-lo,name+' raw span');rspan=finite(rh-rl,name+' rank span');need(span>0.,'Positive raw span')
                for x,r in zip(xs,rs):
                    if identity or x==lo or x==hi:a=r
                    else:
                        numerator=finite(x-lo,'numerator');fraction=finite(numerator/span,'fraction');product=finite(fraction*rspan,'product');a=finite(product+rl,'mapped rank');work['affine_evaluations']+=1
                    mapped.append(a)
                delta=[]
                for a,r in zip(mapped,rs):
                    difference=finite(a-r,'rank difference');delta.append(finite(weight*difference,'slot delta'))
                counts[name+'_changed_rank_cells']+=sum(not samefloat(a,b) for a,b in zip(mapped,rs))
                counts[name+'_nonzero_delta_cells']+=sum(x!=0. for x in delta)
                item=dict(raw=xs,old_rank=rs,new_rank=mapped,distinct_levels=levels,branch='old_rank_copy' if identity else 'affine',minimum=lo,maximum=hi,rank_minimum=rl,rank_maximum=rh,delta=delta)
                event[name]=item;maps.append(delta)
            for k,j in enumerate(ids):
                base=finite(pre[i,j],'consumed pre')
                change=maps[0][k] if len(maps)==1 else finite(maps[0][k]+maps[1][k],'total delta')
                event['delta'].append(change)
                if change!=0.:
                    counts['nonzero_delta_cells']+=1;out[i,j]=finite(base+change,'changed pre')
                if not samefloat(out[i,j],pre[i,j]):event['changed_pre_indices'].append(j);counts['changed_pre_cells']+=1
        events.append(event)
    expected=dict(schema='monthly_forecast_spacing269_transform_v1',mode=mode,counts=counts,events=events,original_inputs_preserved=True,nonconsumed_pre_preserved=True,initial_partial_exact=True,nonweak_exact=True,not_ready_exact=True,consumed_source_ranks_checked=True,local_weight_fixed=True,no_rerank=True,no_new_budget=True,source_files_authenticated=False,computes_returns=False)
    need(out.tobytes()==actual_pre.tobytes(),'Every output byte equals independent reconstruction')
    exact(expected,actual_audit)
    need(before==[x.tobytes() for x in arrays],'Inputs unchanged')
    return dict(schema='forecast_spacing269_independent_v1',passed=True,mode=mode,events=n,width=w,flags=dict.fromkeys(FLAGS,True),work=work,numerical_tolerance=0.,source_provenance_verified=False,actual_coverage_claim=False)
