"""Exact contemporaneous core-conditioned auxiliary ordering, no source I/O.

The raw rational residual is orthogonal to the consumed core cross-section.
Its endpoint-normalized ranks and funded final scores need not be orthogonal.
Only caller-provided synthetic or registered cache arrays are consumed.
"""
from copy import deepcopy
from datetime import date
from fractions import Fraction
from hashlib import sha256
import importlib.util
from numbers import Real
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'work/monthly_aux_rank204_kernel.py'
SOURCE_SHA='c44dd6b24ce060513a92890148675d7e091a5e0d91061da9d64637ecd160ac65'
if sha256(SOURCE.read_bytes()).hexdigest()!=SOURCE_SHA:
    raise ValueError('Frozen204 source/funding helper digest')
_spec=importlib.util.spec_from_file_location('core_conditioned243_frozen204',SOURCE)
funding=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(funding)
ARMS=('incumbent_local67','core_conditioned_aux','aux_rank_control')
POLICY=dict(budget=.05,ceiling=5,gap=.02)
COUNT_FIELDS=('events','symbols_per_event','checked_cells','eligible_cells','initial_partial_events',
    'descriptive_ols_projections','constant_core_events','constant_residual_events','nondegenerate_events',
    'ranked_cells','candidate_nonzero_delta_cells','control_nonzero_delta_cells','candidate_changed_holdings',
    'control_changed_holdings','original_episode_calls','new_model_fits','predictive_model_fits')
EVENT_KEYS=('query_index','date','initial_partial','branch','mean_core','mean_support','cxx','cxs','beta',
    'core_weight','episode_budget','cells','incumbent_choices','candidate_choices','control_choices')
CELL_KEYS=('symbol_index','symbol','eligible','active','core','core_rank','support','weight','residual',
    'residual_display','candidate_target','control_target','candidate_delta','control_delta')


def require(ok,message):
    if not ok:raise ValueError(message)


def dependency_files():
    return (SOURCE,)+tuple(ROOT/p for p in (
        'outputs/monthly_local_linear70/study.py','outputs/monthly_path70_research/incumbent_adapter.py',
        'outputs/monthly_daily_episodes/episodes.py','outputs/monthly_return_priority/return_priority.py',
        'outputs/monthly_local_linear70/INCUMBENT_POLICY.json'))


def _snapshot(arrays):
    return {k:(np.asarray(v).dtype.str,np.asarray(v).shape,np.asarray(v).tobytes()) for k,v in arrays.items()}


def _numeric(value,shape,n,label):
    raw=np.asarray(value);require(raw.shape==shape,'Exact shape: '+label)
    require(all(isinstance(v,Real) and not isinstance(v,(bool,np.bool_)) for v in raw[:n].flat),'Real nonboolean: '+label)
    try:out=raw[:n].astype(np.float64,copy=True)
    except (ValueError,TypeError,OverflowError) as exc:raise ValueError('Unrepresentable numeric: '+label) from exc
    require(np.isfinite(out).all(),'Finite: '+label);return out


def _boolean(value,shape,n,label):
    raw=np.asarray(value);require(raw.shape==shape and raw.dtype.kind=='b','Boolean shape: '+label)
    return raw[:n].copy()


def _axes(arrays,meta,n):
    require(isinstance(arrays,dict) and isinstance(meta,dict),'Array and metadata mappings')
    require('baseline' in arrays,'Original baseline required')
    shape=np.asarray(arrays['baseline']).shape
    require(len(shape)==2 and shape[1]>0 and type(n) is int and 0<=n<=shape[0],'Valid prefix with positive width')
    total,width=shape
    for key in ('dates','ends','symbols'):
        require(isinstance(meta.get(key),list) and len(meta[key])==total,'Metadata axis: '+key)
    prior=None
    for i in range(n):
        day,end,names=meta['dates'][i],meta['ends'][i],meta['symbols'][i]
        require(type(day) is str and date.fromisoformat(day).isoformat()==day,'Canonical date')
        require(type(end) is str and date.fromisoformat(end).isoformat()==end and end>day,'Canonical end')
        require(prior is None or day>prior,'Increasing signal dates');prior=day
        require(day!='2016-08-17' or i==0,'Named partial must be first')
        require(isinstance(names,list) and len(names)==width and all(type(s) is str and s and s.strip()==s for s in names)
            and len(set(names))==width,'Unique monthly symbols')
    base=_numeric(arrays['baseline'],shape,n,'baseline');core=_numeric(arrays['core'],shape,n,'weighted core')
    effects=_numeric(arrays['effects'],shape,n,'episode effects')
    good=_boolean(arrays['eligible'],shape,n,'eligible')
    calm=_boolean(arrays['state_calm10'],(total,),n,'calm');weak=_boolean(arrays['state_weak_narrow'],(total,),n,'weak')
    for name in ('extension_available','forecast_available'):_boolean(arrays[name],shape,n,name)
    support,weight,active=funding._sources(arrays,n)
    cw=np.broadcast_to(np.where(calm,.50,.52)[:,None],(n,width)).copy()
    with np.errstate(over='raise',invalid='raise',divide='raise'):x=core/cw
    # Inherited input-validation tolerance only; never used for residual ties.
    require(np.isfinite(x).all() and (np.abs(x)<=.5+1e-12).all(),'Original bounded core rank')
    return shape,base,core,effects,good,calm,weak,cw,x,support,weight,active


def _pair(value):
    return None if value is None else [str(value.numerator),str(value.denominator)]


def _rank(values):
    count=len(values);require(count>1,'Rank needs at least two observations')
    result=[]
    for value in values:
        less=sum(other<value for other in values);equal=sum(other==value for other in values)
        average=(2*less+equal+1)/2.0
        result.append((average-1.0)/(count-1)-.5)
    return np.asarray(result,dtype=np.float64)


def project(core_rank,support):
    """Canonical exact-binary OLS and ranks for one validated finite row."""
    xx=np.asarray(core_rank);ss=np.asarray(support)
    require(xx.ndim==1 and xx.size>0 and ss.shape==xx.shape,'Projection row axes')
    x=_numeric(xx,xx.shape,len(xx),'projection core');s=_numeric(ss,ss.shape,len(ss),'projection support')
    xf=[Fraction(float(v)) for v in x];sf=[Fraction(float(v)) for v in s];width=len(xf)
    mx=sum(xf,Fraction(0))/width;ms=sum(sf,Fraction(0))/width
    dx=[v-mx for v in xf];ds=[v-ms for v in sf]
    cxx=sum((v*v for v in dx),Fraction(0));cxs=sum((a*b for a,b in zip(dx,ds)),Fraction(0))
    beta=None;residual=[None]*width;display=[None]*width;branch='constant_core'
    candidate=np.zeros(width);control=np.zeros(width)
    if cxx:
        beta=cxs/cxx;residual=[b-beta*a for a,b in zip(dx,ds)]
        try:display=[float(v) for v in residual]
        except (ValueError,OverflowError) as exc:raise FloatingPointError('Residual display overflow') from exc
        require(np.isfinite(display).all(),'Finite residual display')
        branch='constant_residual' if len(set(residual))==1 else 'ranked'
        if branch=='ranked':candidate=_rank(residual);control=_rank(sf)
    return dict(branch=branch,mean_core=_pair(mx),mean_support=_pair(ms),cxx=_pair(cxx),cxs=_pair(cxs),
        beta=_pair(beta),residual=[_pair(v) for v in residual],residual_display=display,candidate=candidate,control=control)


def decisions(study,arrays,meta,n):
    shape,base,core,effects,good,calm,weak,cw,x,support,weight,active=_axes(arrays,meta,n)
    before=_snapshot(arrays);before_meta=deepcopy(meta);width=shape[1]
    full_pre=np.asarray(study.modified_pre_score(study.configs()[4],arrays,n))
    pre=_numeric(full_pre,shape,n,'original Local67 pre')
    counts=dict.fromkeys(COUNT_FIELDS,0);counts.update(events=n,symbols_per_event=width,checked_cells=n*width,eligible_cells=int(good.sum()))
    events=[];candidate_delta=np.zeros((n,width));control_delta=np.zeros((n,width))
    for i in range(n):
        row=project(x[i],support[i]);partial=meta['dates'][i]=='2016-08-17';ranked=row['branch']=='ranked'
        if ranked and not partial:
            with np.errstate(over='raise',invalid='raise'):
                candidate_delta[i]=weight[i]*(row['candidate']-support[i])
                control_delta[i]=weight[i]*(row['control']-support[i])
        require(np.isfinite(candidate_delta[i]).all() and np.isfinite(control_delta[i]).all(),'Finite funded deltas')
        cells=[]
        for j,symbol in enumerate(meta['symbols'][i]):
            cell=dict(symbol_index=j,symbol=symbol,eligible=bool(good[i,j]),active=bool(active[i,j]),core=float(core[i,j]),
                core_rank=float(x[i,j]),support=float(support[i,j]),weight=float(weight[i,j]),residual=row['residual'][j],
                residual_display=row['residual_display'][j],candidate_target=float(row['candidate'][j]),control_target=float(row['control'][j]),
                candidate_delta=float(candidate_delta[i,j]),control_delta=float(control_delta[i,j]))
            require(tuple(cell)==CELL_KEYS,'Cell schema');cells.append(cell)
        budget=float(min(.05,max(0.,.10-(.025*calm[i]+.05*weak[i]))))
        event=dict(query_index=i,date=meta['dates'][i],initial_partial=partial,branch=row['branch'],mean_core=row['mean_core'],
            mean_support=row['mean_support'],cxx=row['cxx'],cxs=row['cxs'],beta=row['beta'],core_weight=float(cw[i,0]),
            episode_budget=budget,cells=cells,incumbent_choices=[],candidate_choices=[],control_choices=[])
        require(tuple(event)==EVENT_KEYS,'Event schema');events.append(event)
        counts['initial_partial_events']+=int(partial);counts['descriptive_ols_projections']+=int(row['branch']!='constant_core')
        counts['constant_core_events']+=int(row['branch']=='constant_core');counts['constant_residual_events']+=int(row['branch']=='constant_residual')
        counts['nondegenerate_events']+=int(ranked);counts['ranked_cells']+=width*int(ranked)
    counts['candidate_nonzero_delta_cells']=int(np.count_nonzero(candidate_delta));counts['control_nonzero_delta_cells']=int(np.count_nonzero(control_delta))
    scores={};pres={};orders={};choices={}
    # The adapter consumes only this physical prefix; original full cache stays intact.
    episode_arrays=dict(baseline=base.copy(),core=core.copy(),core_weight=cw.copy(),eligible=good.copy(),
        state_calm10=calm.copy(),state_weak_narrow=weak.copy())
    for arm,delta in zip(ARMS,(np.zeros((n,width)),candidate_delta,control_delta)):
        arm_pre=pre.copy();changed=delta!=0.
        with np.errstate(over='raise',invalid='raise'):arm_pre[changed]+=delta[changed]
        require(np.isfinite(arm_pre).all(),'Finite modified pre');calls=[]
        def boundary(score,eligible,names,source_effects,original,policy):
            require(not calls,'One original episode per arm')
            require(np.array_equal(score,arm_pre) and np.array_equal(eligible,good),'Exact pre and unchanged E')
            require(names==meta['symbols'][:n] and np.array_equal(source_effects,effects),'Original symbols/effects')
            for key,expected in (('baseline',base),('core',core),('core_weight',cw),('state_calm10',calm),('state_weak_narrow',weak)):
                require(np.array_equal(original[key],expected),'Original episode support '+key)
            require(policy==POLICY,'Original episode policy');calls.append(True)
            return study.ep.select_boundary(score,eligible,names,source_effects,original,policy)
        chosen,post=study.adapter.score_and_choose('incumbent66',0.,np.full((n,width),np.nan),episode_arrays,meta,n,
            pre_score=arm_pre,effects=effects,episode_policy=POLICY.copy(),boundary_selector=boundary)
        require(len(calls)==1,'Original episode called once');counts['original_episode_calls']+=len(calls)
        post=_numeric(post,(n,width),n,'postepisode score')
        require(isinstance(chosen,list) and len(chosen)==n,'Choice axis')
        clean=[]
        for i,indices in enumerate(chosen):
            require(all(isinstance(j,(int,np.integer)) and not isinstance(j,(bool,np.bool_)) for j in indices),'Choice indices')
            ids=[int(j) for j in indices]
            require(len(set(ids))==len(ids) and all(0<=j<width and good[i,j] for j in ids),'Unique eligible choices')
            require(len(ids)==min(3,int(good[i].sum())) if good[i].sum()>=2 else len(ids)==0,'EW3/EW2/cash')
            clean.append(ids)
        choices[arm]=clean;scores[arm]=post.copy();pres[arm]=arm_pre.copy();orders[arm]=study.rp.make_orders(clean,post,meta)
    for i,event in enumerate(events):
        for key,arm in zip(('incumbent_choices','candidate_choices','control_choices'),ARMS):event[key]=choices[arm][i]
        counts['candidate_changed_holdings']+=int(set(choices[ARMS[1]][i])!=set(choices[ARMS[0]][i]))
        counts['control_changed_holdings']+=int(set(choices[ARMS[2]][i])!=set(choices[ARMS[0]][i]))
        if event['initial_partial'] or event['branch']!='ranked':
            for arm in ARMS[1:]:require(orders[arm][i]==orders[ARMS[0]][i] and np.array_equal(scores[arm][i],scores[ARMS[0]][i]),'No-op exact incumbent')
    require(_snapshot(arrays)==before and meta==before_meta,'Original inputs mutated')
    return orders,scores,pres,dict(schema='monthly_core_conditioned243_decisions_v1',counts=counts,events=events)
