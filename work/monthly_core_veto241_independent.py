"""Independent original Local67 episode and third-seat core-veto oracle.

No production kernel/host imports. The original pre-score is rebuilt through
the pinned independent source bridge; episode arithmetic is scalar below.
"""
import hashlib
import importlib.util
import math
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/'work/monthly_peer_rank240_independent.py'
PRIOR_SHA='b0ba9fbce5ba4c59dd579d42bc08622787c307c6f89f02d5efff2fb2831b5d93'
PRIOR_TEST=ROOT/'work/test_monthly_peer_rank240_independent.py'
PRIOR_TEST_SHA='914d37f357794571e9cf4499591e6516bebc78b47128410c7770169184faa682'
ARMS=('incumbent_local67','core_nonconcession_third_seat','no_episode_control')
COUNT_FIELDS=('events','symbols_per_event','eligible_cells','initial_partial_events',
    'three_seat_events','boundary_cells','core_compared_cells','vetoed_cells','vetoed_winner_events',
    'candidate_changed_holdings','control_changed_holdings','original_episode_calls','new_model_fits')
EVENT_KEYS=('query_index','date','initial_partial','pre_order','locked_top2','pre_third',
    'boundary_indices','admitted_indices','vetoed_indices','core_anchor','core_weight','episode_budget',
    'original_winner','candidate_winner','control_winner','incumbent_choices','candidate_choices','control_choices')
SELECTION_FLAGS=('passed','independent_original_pre_checked','original_pre_unchanged',
    'original_model_rank_scope_preserved','original_eligibility_unchanged','original_support_checked',
    'original_budget_checked','calm_nonweak_extension_budget_checked','original_effects_unchanged',
    'weighted_core_comparison_checked','same_original_boundary_checked','core_ties_admitted',
    'original_top_two_preserved','no_episode_recalculation','candidate_full_scores_equal_incumbent',
    'no_episode_control_checked','initial_partial_unchanged','exact_scores_and_choices_checked',
    'full_audit_checked','descriptive_counts_checked','no_new_model_fits_checked')


def prior():
    if hashlib.sha256(PRIOR.read_bytes()).hexdigest()!=PRIOR_SHA:
        raise ValueError('Frozen independent original-score/accounting bridge changed')
    spec=importlib.util.spec_from_file_location('core_veto241_prior_independent',PRIOR)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


old=prior()
require,exact_array,record_check=old.require,old.exact_array,old.record_check
PATHS,ACCOUNTING_SHA=old.PATHS,old.ACCOUNTING_SHA


def frozen_source_oracle():return old.frozen_source_oracle()
def accounting():return old.accounting()


def dependency_files():
    require(hashlib.sha256(PRIOR_TEST.read_bytes()).hexdigest()==PRIOR_TEST_SHA,'Frozen synthetic fixture changed')
    return tuple(sorted(set(old.dependency_files())|{PRIOR,PRIOR_TEST,Path(__file__).resolve()}))


def axes(original,meta,n):
    _,width,days,names,_=old.old.axes(meta,original,n)
    require(isinstance(meta.get('ends'),list) and len(meta['ends'])==len(meta['dates']),'Full end-date axis')
    for day,end in zip(days,meta['ends'][:n]):require(old.old.iso(end)>day,'Positive holding interval')
    for i,d in enumerate(days):
        if str(d.date())=='2016-08-17':require(i==0,'Named partial only first row')
    return width,days,names


def _selection(original,meta,n=None):
    n=len(meta['dates']) if n is None else n
    require(type(n) is int and 0<=n<=len(meta['dates']),'Valid consumed prefix')
    width,days,names=axes(original,meta,n)
    counts={k:0 for k in COUNT_FIELDS};counts.update(events=n,symbols_per_event=width,original_episode_calls=1)
    audit=dict(schema='monthly_core_veto241_decisions_v1',counts=counts,events=[])
    if n==0:
        scores={k:np.empty((0,width)) for k in ARMS}
        return scores,{k:[] for k in ARMS},{k:a.copy() for k,a in scores.items()},audit,dict(
            minimum_remaining_support=None,maximum_budget_sum_error=0.,calm_nonweak_extension_count=0)
    x=frozen_source_oracle().old.original_helper().original_prefix(original,meta,n)
    shape=(n,width);P=x['local'].copy();F=P.copy();control=P.copy()
    expected_cw=np.broadcast_to(np.where(x['calm'],.5,.52)[:,None],shape)
    exact_array(x['cw'],expected_cw,'Original common monthly core weight')
    ext=np.asarray(original['extension_available'])[:n];forecast=np.asarray(original['forecast_available'])[:n]
    ranklog=np.asarray(original['rank_log_excess'])[:n]
    require(ext.shape==forecast.shape==ranklog.shape==shape and ext.dtype==forecast.dtype==np.dtype(bool),
            'Original extension-mask axes')
    A=(x['calm'] & ~x['weak'])[:,None]&ext&forecast&x['eligible']&np.isfinite(ranklog)
    budget=np.minimum(.05,np.maximum(0.,.10-(.025*x['calm']+.05*x['weak'])))
    chosen={k:[] for k in ARMS};minimum=1.;sum_error=0.
    for i,ss in enumerate(names):
        partial=str(days[i].date())=='2016-08-17'
        rank=sorted([j for j in range(width) if x['eligible'][i,j]],key=lambda j:(-P[i,j],ss[j]))
        third=rank[2] if len(rank)>=3 else None;top2=rank[:2] if third is not None else []
        J=[] if third is None else [j for j in rank[2:5] if float(P[i,third])-float(P[i,j])<=.02]
        # Calculate each original boundary effect exactly once, before the veto.
        for j in J:
            target=min(.5,max(-.5,x['support'][i,j]+x['effects'][i,j]))
            F[i,j]+=budget[i]*(target-x['support'][i,j])
        require(np.isfinite(F[i]).all(),'Finite original episode scores')
        anchor=None if third is None else float(x['core'][i,third])
        admitted=J.copy() if partial else [j for j in J if x['core'][i,j]>=anchor]
        vetoed=[j for j in J if j not in admitted]
        original_winner=None if third is None else min(J,key=lambda j:(-F[i,j],ss[j]))
        candidate_winner=None if third is None else min(admitted,key=lambda j:(-F[i,j],ss[j]))
        control_winner=original_winner if partial else third
        if third is None:
            inc=rank if len(rank)==2 else [];cand=inc.copy();ctl=inc.copy()
        else:
            inc=sorted(top2+[original_winner],key=lambda j:(-F[i,j],ss[j]))
            cand=sorted(top2+[candidate_winner],key=lambda j:(-F[i,j],ss[j]))
            ctl=inc.copy() if partial else rank[:3]
        if partial:control[i]=F[i]
        for arm,row in zip(ARMS,(inc,cand,ctl)):chosen[arm].append(row)
        audit['events'].append(dict(query_index=i,date=meta['dates'][i],initial_partial=partial,pre_order=rank,
            locked_top2=top2,pre_third=third,boundary_indices=J,admitted_indices=admitted,vetoed_indices=vetoed,
            core_anchor=anchor,core_weight=float(x['cw'][i,0]),episode_budget=float(budget[i]),
            original_winner=original_winner,candidate_winner=candidate_winner,control_winner=control_winner,
            incumbent_choices=inc,candidate_choices=cand,control_choices=ctl))
        counts['eligible_cells']+=len(rank);counts['initial_partial_events']+=int(partial)
        counts['three_seat_events']+=int(third is not None);counts['boundary_cells']+=len(J)
        counts['core_compared_cells']+=0 if partial else len(J);counts['vetoed_cells']+=len(vetoed)
        counts['vetoed_winner_events']+=int(original_winner in vetoed)
        counts['candidate_changed_holdings']+=int(set(cand)!=set(inc))
        counts['control_changed_holdings']+=int(set(ctl)!=set(inc))
        for j in range(width):
            # Candidate adds no factor slot. The no-episode control frees the
            # original episode allocation on regular rows only.
            for ep_active in (j in J,partial and j in J):
                weights=[float(x['cw'][i,j]),.025*float(x['calm'][i]),.05*float(x['weak'][i]),
                         .01*float(A[i,j]),float(budget[i])*float(ep_active)]
                remaining=1.-weights[0]
                for value in weights[1:]:remaining-=value
                error=abs(math.fsum(weights+[remaining])-1.)
                require(remaining>=.38-1e-12 and error<=1e-12,'Unchanged total Local67 support budget')
                minimum=min(minimum,remaining);sum_error=max(sum_error,error)
    scores={ARMS[0]:F.copy(),ARMS[1]:F.copy(),ARMS[2]:control}
    pres={k:P.copy() for k in ARMS}
    return scores,chosen,pres,audit,dict(minimum_remaining_support=minimum,maximum_budget_sum_error=sum_error,
                                       calm_nonweak_extension_count=int(A.sum()))


def selection_oracle(original,meta,n=None):
    scores,choices,pres,audit,_=_selection(original,meta,n)
    return scores,choices,pres,audit


def verify_selection(original,meta,orders,*,scores,pre_scores,audit,n=None):
    n=len(meta['dates']) if n is None else n
    expected,choices,pres,expected_audit,budget=_selection(original,meta,n)
    require(isinstance(orders,dict) and set(orders)==set(ARMS),'Three order arms')
    wanted={arm:[dict(date=meta['dates'][i],end=meta['ends'][i],signal_date=meta['dates'][i],
        symbols=[meta['symbols'][i][j] for j in ids],scores=[float(expected[arm][i,j]) for j in ids])
        for i,ids in enumerate(choices[arm])] for arm in ARMS}
    record_check(orders,wanted,'Independent ordered selection')
    for got,ref,label in ((scores,expected,'post-episode scores'),(pre_scores,pres,'original pre-scores')):
        require(isinstance(got,dict) and set(got)==set(ARMS),'Three full '+label)
        for arm in ARMS:exact_array(got[arm],ref[arm],label+'/'+arm)
    record_check(audit,expected_audit,'Independent complete decision audit')
    return dict(**{k:True for k in SELECTION_FLAGS},**budget,counts=expected_audit['counts'],
        checked_events=n,ordered_choices_checked=3*n,full_scores_supplied=True,full_pre_scores_supplied=True,
        full_audit_supplied=True,numerical_tolerance=0.,original_source_parity_tolerance=1e-12,
        producer_modules_imported=False,computes_returns=False,
        holdings_counts={arm:{str(k):sum(len(row)==k for row in choices[arm]) for k in (0,2,3)} for arm in ARMS})


def audit_outcomes(px,opening,orders,exports,results,meta):
    require(set(orders)==set(exports)==set(results)==set(ARMS),'Outcome arms')
    mapping=dict(zip(old.ARMS,ARMS))
    report=prior().audit_outcomes(px,opening,*[{k:data[v] for k,v in mapping.items()}
        for data in (orders,exports,results)],meta)
    report['paths']={mapping[k]:v for k,v in report['paths'].items()}
    return report
