"""Preserve the frozen episode score while vetoing a lower-core third seat.

Only caller-supplied synthetic or registered cached arrays are consumed. The
original pre-score and episode are called once, before the candidate veto.
No data loading, new factor, model fitting, ranking domain or execution engine.
"""
from copy import deepcopy
from datetime import date
from numbers import Real
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ARMS=('incumbent_local67','core_nonconcession_third_seat','no_episode_control')
POLICY=dict(budget=.05,ceiling=5,gap=.02)
COUNT_FIELDS=('events','symbols_per_event','eligible_cells','initial_partial_events','three_seat_events','boundary_cells','core_compared_cells','vetoed_cells','vetoed_winner_events','candidate_changed_holdings','control_changed_holdings','original_episode_calls','new_model_fits')
EVENT_KEYS=('query_index','date','initial_partial','pre_order','locked_top2','pre_third','boundary_indices','admitted_indices','vetoed_indices','core_anchor','core_weight','episode_budget','original_winner','candidate_winner','control_winner','incumbent_choices','candidate_choices','control_choices')


def require(ok,message):
    if not ok:raise ValueError(message)


def dependency_files():
    return tuple(ROOT/name for name in (
        'outputs/monthly_local_linear70/study.py',
        'outputs/monthly_path70_research/incumbent_adapter.py',
        'outputs/monthly_daily_episodes/episodes.py',
        'outputs/monthly_return_priority/return_priority.py',
        'outputs/monthly_local_linear70/INCUMBENT_POLICY.json'))


def _snapshot(arrays):
    return {k:(np.asarray(v).dtype.str,np.asarray(v).shape,np.asarray(v).tobytes()) for k,v in arrays.items()}


def _numeric(value,shape,n,label):
    raw=np.asarray(value);require(raw.shape==shape,'Exact shape required: '+label)
    consumed=raw[:n]
    require(all(isinstance(x,Real) and not isinstance(x,(bool,np.bool_)) for x in consumed.flat),
            'Real nonboolean values required: '+label)
    result=consumed.astype(np.float64,copy=True)
    require(np.isfinite(result).all(),'Finite values required: '+label)
    return result


def _boolean(value,shape,n,label):
    raw=np.asarray(value);require(raw.shape==shape and raw.dtype.kind=='b','Boolean shape required: '+label)
    return raw[:n].copy()


def _axes(arrays,meta,n):
    require(isinstance(arrays,dict) and isinstance(meta,dict),'Array and metadata mappings required')
    shape=np.asarray(arrays['baseline']).shape
    require(len(shape)==2 and shape[1]>0 and type(n) is int and 0<=n<=shape[0],'Valid nonempty-width prefix required')
    total,width=shape
    for key in ('dates','ends','symbols'):
        require(isinstance(meta.get(key),list) and len(meta[key])==total,'Metadata axis '+key)
    days=[]
    for i in range(n):
        day=meta['dates'][i];end=meta['ends'][i];names=meta['symbols'][i]
        require(type(day) is str and date.fromisoformat(day).isoformat()==day,'Canonical signal date')
        require(type(end) is str and date.fromisoformat(end).isoformat()==end and end>day,'Canonical end date')
        require(not days or day>days[-1],'Increasing signals required')
        require(day!='2016-08-17' or i==0,'Named partial must be first')
        require(isinstance(names,list) and len(names)==width and all(type(s) is str and s and s.strip()==s for s in names)
                and len(set(names))==width,'Unique monthly symbol identities required')
        days.append(day)
    baseline=_numeric(arrays['baseline'],shape,n,'baseline')
    core=_numeric(arrays['core'],shape,n,'already weighted core')
    effects=_numeric(arrays['effects'],shape,n,'original episode effects')
    eligible=_boolean(arrays['eligible'],shape,n,'eligible')
    calm=_boolean(arrays['state_calm10'],(total,),n,'calm')
    weak=_boolean(arrays['state_weak_narrow'],(total,),n,'weak')
    rawcw=np.asarray(arrays['core_weight'])
    if rawcw.shape==(total,):cw=_numeric(rawcw,(total,),n,'core_weight')[:,None]
    else:
        require(rawcw.shape in ((total,1),shape),'Core weight axes')
        cw=_numeric(rawcw,rawcw.shape,n,'core_weight')
    cw=np.broadcast_to(cw,(n,width)).copy()
    expected=np.broadcast_to(np.where(calm,.50,.52)[:,None],(n,width))
    require(np.array_equal(cw,expected),'Core coefficients must be common positive .50/.52 by original calm state')
    support=(baseline-core)/(1.-cw)
    require(np.isfinite(support).all() and (np.abs(support)<=.5+1e-12).all(),'Original bounded support')
    return shape,baseline,core,effects,eligible,calm,weak,cw


def decisions(study,arrays,meta,n):
    shape,baseline,core,effects,eligible,calm,weak,cw=_axes(arrays,meta,n)
    before=_snapshot(arrays);before_meta=deepcopy(meta)
    full_pre=np.asarray(study.modified_pre_score(study.configs()[4],arrays,n))
    pre=_numeric(full_pre,shape,n,'original Local67 pre')
    calls=[]
    def boundary(score,good,names,source_effects,original,policy):
        require(not calls,'Only one original episode call')
        require(np.array_equal(score,pre) and np.array_equal(good,eligible),'Original pre and E required')
        require(names==meta['symbols'][:n] and np.array_equal(source_effects,effects),'Original symbols/effects required')
        for key,value in (('baseline',baseline),('core',core),('core_weight',cw),('state_calm10',calm),('state_weak_narrow',weak)):
            require(np.array_equal(original[key],value),'Original episode support input changed: '+key)
        require(policy==POLICY,'Original consumed episode policy required')
        calls.append(True)
        return study.ep.select_boundary(score,good,names,source_effects,original,policy)
    chosen,full_score=study.adapter.score_and_choose('incumbent66',0.,np.full(shape,np.nan),arrays,meta,n,
        pre_score=full_pre.copy(),effects=arrays['effects'],episode_policy=POLICY.copy(),boundary_selector=boundary)
    require(len(calls)==1,'Original episode must run exactly once')
    require(isinstance(chosen,list) and len(chosen)==n,'Original choice axis')
    post=_numeric(full_score,(n,shape[1]),n,'original full postepisode score')
    scores={ARMS[0]:post.copy(),ARMS[1]:post.copy(),ARMS[2]:pre.copy()}
    pres={arm:pre.copy() for arm in ARMS};choices={arm:[] for arm in ARMS};events=[]
    counts=dict.fromkeys(COUNT_FIELDS,0);counts.update(events=n,symbols_per_event=shape[1],original_episode_calls=1)
    for i in range(n):
        names=meta['symbols'][i];partial=meta['dates'][i]=='2016-08-17'
        rank=sorted(np.flatnonzero(eligible[i]).tolist(),key=lambda j:(-pre[i,j],names[j]))
        inc=[int(j) for j in chosen[i]]
        require(len(set(inc))==len(inc) and all(j in rank for j in inc),'Original eligible unique choices')
        third=rank[2] if len(rank)>=3 else None
        locked=rank[:2] if third is not None else []
        boundary_indices=[j for j in rank[2:5] if pre[i,third]-pre[i,j]<=.02] if third is not None else []
        anchor=float(core[i,third]) if third is not None else None
        if third is None:
            admitted=[];vetoed=[];incwin=candwin=controlwin=None
            require(inc==(rank if len(rank)==2 else []),'Original EW2/cash rule differs')
            candidate=list(inc);control=list(inc)
        else:
            incwin=min(boundary_indices,key=lambda j:(-post[i,j],names[j]))
            require(inc==sorted(locked+[incwin],key=lambda j:(-post[i,j],names[j])),'Original episode boundary choices differ')
            admitted=list(boundary_indices) if partial else [j for j in boundary_indices if core[i,j]>=core[i,third]]
            vetoed=[j for j in boundary_indices if j not in admitted]
            require(third in admitted,'Pre-third must always remain admissible')
            candwin=incwin if partial else min(admitted,key=lambda j:(-post[i,j],names[j]))
            controlwin=incwin if partial else third
            candidate=list(inc) if partial else sorted(locked+[candwin],key=lambda j:(-post[i,j],names[j]))
            control=list(inc) if partial else rank[:3]
            require(len(candidate)==3 and set(locked)<=set(candidate),'Veto cannot reduce seats or replace pre top2')
            if incwin in admitted:require(candidate==inc,'Admissible original winner must remain exact incumbent')
        if partial:scores[ARMS[2]][i]=post[i]
        for arm,row in zip(ARMS,(inc,candidate,control)):choices[arm].append(row)
        budget=float(min(.05,max(0.,.10-(.025*calm[i]+.05*weak[i]))))
        event=dict(query_index=i,date=meta['dates'][i],initial_partial=partial,pre_order=rank,locked_top2=locked,
            pre_third=third,boundary_indices=boundary_indices,admitted_indices=admitted,vetoed_indices=vetoed,
            core_anchor=anchor,core_weight=float(cw[i,0]),episode_budget=budget,original_winner=incwin,
            candidate_winner=candwin,control_winner=controlwin,incumbent_choices=inc,candidate_choices=candidate,control_choices=control)
        require(tuple(event)==EVENT_KEYS,'Event audit schema');events.append(event)
        counts['eligible_cells']+=len(rank);counts['initial_partial_events']+=int(partial)
        counts['three_seat_events']+=int(third is not None);counts['boundary_cells']+=len(boundary_indices)
        counts['core_compared_cells']+=0 if partial else len(boundary_indices);counts['vetoed_cells']+=len(vetoed)
        counts['vetoed_winner_events']+=int(incwin in vetoed)
        counts['candidate_changed_holdings']+=int(set(candidate)!=set(inc));counts['control_changed_holdings']+=int(set(control)!=set(inc))
    orders={arm:study.rp.make_orders(choices[arm],scores[arm],meta) for arm in ARMS}
    require(_snapshot(arrays)==before and meta==before_meta,'Original input mutation')
    audit=dict(schema='monthly_core_veto241_decisions_v1',counts=counts,events=events)
    return orders,scores,pres,audit
