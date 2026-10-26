"""Independent eligibility-only Local67 bridge, with frozen251 endpoints.

No253 producer, host or adapter imports. Source/core and basket endpoint proofs
reuse only the SHA-pinned251 independent oracle; the restricted episode below
is reconstructed from original P and exact M evidence without altering cache E.
"""
import copy
import hashlib
import importlib.util
import math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/'work/monthly_basket_core251_independent.py'
PRIOR_SHA='76bdbe2cc662bf4cabe2e8ac9a090d32e0e007b1b413f34cdf55780f747b67cb'
PRIOR_TEST=ROOT/'work/test_monthly_basket_core251_independent.py'
PRIOR_TEST_SHA='001e90989dbf8daa09298d480a3aac3d6b5e098ce00d90f463a2bc317cae8d0e'
if hashlib.sha256(PRIOR.read_bytes()).hexdigest()!=PRIOR_SHA:raise ValueError('Frozen251 independent changed')
_spec=importlib.util.spec_from_file_location('leader253_prior_independent',PRIOR)
prior=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(prior)
require,exact_array,record_check=prior.require,prior.exact_array,prior.record_check
ARMS=('incumbent_local67','long_leader_local67','basket_core251_control')
EVENT_KEYS=('query_index','date','branch','original_eligible_indices','new_eligible_indices','shortlist_indices',
 'selector_eligible_indices','pre_order','boundary_candidates','episode_budget','original_remaining_support',
 'incumbent_choices','candidate_choices','basket_choices')
COUNT_FIELDS=('events','symbols_per_event','initial_partial_events','formal_events','original_eligible_cells',
 'positive_momentum_cells','shortlist_cells','restricted_eligible_cells','removed_eligible_cells','cash_events',
 'two_name_events','threeplus_name_events','candidate_changed_holdings_events','basket_changed_holdings_events',
 'candidate_vs_basket_changed_holdings_events','original_episode_calls','candidate_episode_calls',
 'diagnostic_episode_calls','reused_feature_events','new_factor_definitions','new_model_fits','predictive_model_fits')
SELECTION_FLAGS=('passed','independent_original_pre_checked','original_core_checked','original_eligibility_preserved_as_input',
 'exact_incumbent_copy_checked','exact_basket251_endpoint_checked','new_annual_momentum_domain_checked',
 'exact_top6_shortlist_checked','restricted_eligibility_only_checked','all49_original_pre_unchanged',
 'original_model_ranks_and_effects_unchanged','original_support_and_budget_checked','single_candidate_episode_checked',
 'restricted_top_two_locked','restricted_boundary_rank3_to5_checked','no_final_global_rerank',
 'named_initial_partial_unchanged','two_names_and_cash_checked','exact_scores_and_choices_checked',
 'full_audit_checked','descriptive_counts_checked','no_predictive_fits_checked','no_future_prefix_consumption')
SOURCE_GATE_FLAGS=prior.SOURCE_GATE_FLAGS
INCUMBENT_GATE_FLAGS=prior.INCUMBENT_GATE_FLAGS
ENDPOINT_FLAGS=('passed','exact_incumbent_copy_checked','exact_basket251_endpoint_checked','full_scores_supplied',
 'full_pre_scores_supplied','full_features_supplied')
PREPARE_FLAGS=SELECTION_FLAGS+tuple(k for k in prior.PREPARE_FLAGS if k not in prior.SELECTION_FLAGS)
C_SOURCE_ATOL,C_SOURCE_RTOL=prior.C_SOURCE_ATOL,prior.C_SOURCE_RTOL
CONTINUOUS_ATOL,CONTINUOUS_RTOL=prior.CONTINUOUS_ATOL,prior.CONTINUOUS_RTOL
PATHS,ACCOUNTING_SHA=prior.PATHS,prior.ACCOUNTING_SHA


def dependency_files():
    require(hashlib.sha256(PRIOR_TEST.read_bytes()).hexdigest()==PRIOR_TEST_SHA,'Frozen fixture dependency changed')
    return tuple(sorted(set(prior.dependency_files())|{PRIOR,PRIOR_TEST,Path(__file__).resolve()}))


def accounting():return prior.accounting()
def frozen_source_oracle():return prior.frozen_source_oracle()
def original_base(original,meta,n):return prior.original_base(original,meta,n)


def source_view(features,n):
    require(type(features) is list and len(features)>=n,'Feature prefix list')
    out=[]
    for event in features[:n]:
        row=copy.deepcopy(event)
        for key in prior.CANDIDATE_KEYS:row[key]=None
        if not row['initial_partial']:row['branch']='source_verified'
        out.append(row)
    return out


def independent_features(panel,meta,calendar,original,review,n=None,*,include_selection=False):
    return prior.independent_features(panel,meta,calendar,original,review,n,include_selection=include_selection)


def verify_source_features(panel,meta,calendar,original,review,features,n=None):
    # The pre-decision gate consumes the source view only, never trusts final S.
    return prior.verify_source_features(panel,meta,calendar,original,review,features,n)


def verify_incumbent(original,meta,orders,scores,pre_scores,n=None):
    return prior.verify_incumbent(original,meta,orders,scores,pre_scores,n)


def _shortlist(event,eligible):
    """Rebuild exact M gate and ordering, independently of saved S/winners."""
    included=[];momentum={}
    for j in np.flatnonzero(eligible).tolist():
        cell=event['cells'][j]
        require(cell['original_eligible'] is True and cell['source_available'] is True,'Required original E evidence')
        prior.validate_measurement(cell['measurement'])
        momentum[j]=prior.exactmath.rational(cell['measurement']['momentum'])
        if momentum[j]>0:included.append(j)
    short=sorted(included,key=lambda j:(-momentum[j],event['symbols'][j]))[:6]
    record_check(event['new_eligible_indices'],included,'Exact M-positive domain')
    record_check(event['shortlist_indices'],short,'Exact annual leaders')
    mask=np.zeros(eligible.shape,dtype=bool)
    for j in short:mask[j]=bool(eligible[j])
    return included,short,mask


def restricted_episode(pre,base,eligible):
    """One scalar reproduction of the original selector with only E replaced."""
    require(eligible.shape==pre.shape and eligible.dtype==np.dtype(bool),'Selector exact bool shape')
    result=pre.copy();choices=[];orders=[];boundaries=[]
    for i,names in enumerate(base['names']):
        ranked=sorted(np.flatnonzero(eligible[i]).tolist(),key=lambda j:(-pre[i,j],names[j]))
        boundary=[] if len(ranked)<3 else [j for j in ranked[2:5] if pre[i,ranked[2]]-pre[i,j]<=.02]
        if len(ranked)<3:chosen=ranked.copy() if len(ranked)==2 else []
        else:
            for j in boundary:
                support=base['support'][i,j]
                target=min(.5,max(-.5,support+base['effects'][i,j]))
                result[i,j]+=base['budget'][i]*(target-support)
            winner=min(boundary,key=lambda j:(-result[i,j],names[j]))
            chosen=sorted(ranked[:2]+[winner],key=lambda j:(-result[i,j],names[j]))
        choices.append(chosen);orders.append(ranked);boundaries.append(boundary)
    require(np.isfinite(result).all(),'Finite restricted episode scores')
    return result,choices,orders,boundaries


def _endpoints(original,meta,features,n):
    scores,choices,pres,audit=prior.selection_oracle(original,meta,features,n)
    return scores,choices,pres,audit


def verify_endpoints(original,meta,features,orders,*,scores,pre_scores,n=None):
    n=len(meta['dates']) if n is None else n
    sc,ch,pr,_=_endpoints(original,meta,features,n)
    names=prior.ARMS[:2];expected_orders=prior.orders_from_choices(sc,ch,meta)
    for supplied,label in ((orders,'Orders'),(scores,'Scores'),(pre_scores,'Pres')):
        require(type(supplied) is dict and set(supplied)==set(names),'Endpoint '+label+' arms')
    for arm in names:
        record_check(orders[arm],expected_orders[arm],'Frozen251 endpoint orders/'+arm)
        exact_array(scores[arm],sc[arm],'Endpoint scores/'+arm)
        exact_array(pre_scores[arm],pr[arm],'Endpoint pre/'+arm)
    return dict(**{k:True for k in ENDPOINT_FLAGS},checked_events=n,ordered_choices_checked=2*n,
      candidate_selection_computed=False,producer_modules_imported=False,computes_returns=False,numerical_tolerance=0.)


def selection_oracle(original,meta,features,n=None):
    n=len(meta['dates']) if n is None else n;base=original_base(original,meta,n)
    # Frozen source-cache algebra and both251 endpoint choices are independently rebuilt.
    old_scores,old_choices,old_pres,_=_endpoints(original,meta,features,n)
    masks=base['E'].copy();included=[];shortlists=[]
    for i,event in enumerate(features[:n]):
        if event['initial_partial']:included.append(None);shortlists.append(None)
        else:
            inc,short,mask=_shortlist(event,base['E'][i]);included.append(inc);shortlists.append(short);masks[i]=mask
    candidate,ch,preorder,boundary=restricted_episode(base['P'],base,masks)
    scores={ARMS[0]:old_scores[prior.ARMS[0]].copy(),ARMS[1]:candidate,ARMS[2]:old_scores[prior.ARMS[1]].copy()}
    pres={ARMS[0]:old_pres[prior.ARMS[0]].copy(),ARMS[1]:base['P'].copy(),ARMS[2]:old_pres[prior.ARMS[1]].copy()}
    choices={ARMS[0]:copy.deepcopy(old_choices[prior.ARMS[0]]),ARMS[1]:ch,ARMS[2]:copy.deepcopy(old_choices[prior.ARMS[1]])}
    counts={key:0 for key in COUNT_FIELDS}
    counts.update(events=n,symbols_per_event=49,original_episode_calls=1,candidate_episode_calls=1,reused_feature_events=n)
    events=[]
    for i,event in enumerate(features[:n]):
        partial=event['initial_partial'];good=np.flatnonzero(base['E'][i]).tolist();restricted=np.flatnonzero(masks[i]).tolist()
        branch='initial_partial' if partial else 'cash' if len(restricted)<2 else 'two_names' if len(restricted)==2 else 'restricted_episode'
        if partial:
            exact_array(candidate[i],scores[ARMS[0]][i],'Partial exact candidate F')
            record_check(ch[i],choices[ARMS[0]][i],'Partial exact candidate choices')
        remaining=(1.-base['cw'][i])-(.025*base['calm'][i]+.05*base['weak'][i]+.01*base['A'][i])-base['budget'][i]
        require(np.isfinite(remaining).all() and (remaining>=.38-1e-12).all(),'Original auxiliary budget')
        counts['initial_partial_events']+=int(partial);counts['formal_events']+=int(not partial)
        counts['original_eligible_cells']+=len(good);counts['positive_momentum_cells']+=len(included[i] or [])
        counts['shortlist_cells']+=len(shortlists[i] or []);counts['restricted_eligible_cells']+=len(restricted)
        counts['removed_eligible_cells']+=len(good)-len(restricted)
        if not partial:counts['cash_events' if len(restricted)<2 else 'two_name_events' if len(restricted)==2 else 'threeplus_name_events']+=1
        for field,left,right in (('candidate_changed_holdings_events',ARMS[1],ARMS[0]),('basket_changed_holdings_events',ARMS[2],ARMS[0]),('candidate_vs_basket_changed_holdings_events',ARMS[1],ARMS[2])):
            counts[field]+=int(set(choices[left][i])!=set(choices[right][i]))
        events.append(dict(query_index=i,date=meta['dates'][i],branch=branch,original_eligible_indices=good,
          new_eligible_indices=included[i],shortlist_indices=shortlists[i],selector_eligible_indices=restricted,
          pre_order=preorder[i],boundary_candidates=boundary[i],episode_budget=float(base['budget'][i]),
          original_remaining_support=remaining.tolist(),incumbent_choices=choices[ARMS[0]][i],
          candidate_choices=choices[ARMS[1]][i],basket_choices=choices[ARMS[2]][i]))
    return scores,choices,pres,dict(schema='monthly_leader_bridge253_decisions_v1',counts=counts,events=events)


def orders_from_choices(scores,choices,meta):
    return {arm:[dict(date=meta['dates'][i],end=meta['ends'][i],signal_date=meta['dates'][i],
      symbols=[meta['symbols'][i][j] for j in row],scores=[float(scores[arm][i,j]) for j in row])
      for i,row in enumerate(choices[arm])] for arm in ARMS}


def verify_selection(original,meta,features,orders,*,scores,pre_scores,audit,n=None):
    n=len(meta['dates']) if n is None else n
    sc,ch,pr,expected=selection_oracle(original,meta,features,n)
    record_check(orders,orders_from_choices(sc,ch,meta),'Complete ordered253 choices')
    for supplied,wanted,label in ((scores,sc,'Full F'),(pre_scores,pr,'Full P')):
        require(type(supplied) is dict and set(supplied)==set(ARMS),'Three '+label+' arms')
        for arm in ARMS:exact_array(supplied[arm],wanted[arm],label+'/'+arm)
    record_check(audit,expected,'Complete253 audit')
    return dict(**{k:True for k in SELECTION_FLAGS},counts=expected['counts'],checked_events=n,ordered_choices_checked=3*n,
      producer_modules_imported=False,computes_returns=False,full_features_supplied=True,full_scores_supplied=True,
      full_pre_scores_supplied=True,full_audit_supplied=True,numerical_tolerance=0.,C_source_atol=C_SOURCE_ATOL,C_source_rtol=C_SOURCE_RTOL)


def verify_prepared(panel,meta,calendar,original,review,features,orders,*,scores,pre_scores,audit,n=None):
    n=len(meta['dates']) if n is None else n
    expected=prior._checked_source_features(panel,meta,calendar,original,review,features,n,include_selection=True)
    receipt=verify_selection(original,meta,expected,orders,scores=scores,pre_scores=pre_scores,audit=audit,n=n)
    receipt.update({k:True for k in PREPARE_FLAGS})
    receipt.update(continuous_atol=CONTINUOUS_ATOL,continuous_rtol=CONTINUOUS_RTOL,rank_tolerance=0.,exact_discrete=True)
    return receipt


def audit_outcomes(px,opening,orders,exports,results,meta):
    require(set(orders)==set(exports)==set(results)==set(ARMS),'Three253 account arms')
    for arm in ARMS:
        for path in ('same_close_zero','next_open_zero'):
            data=exports[arm][path];result=results[arm][path]
            fees=[row['cost'] for row in data['TRADES']]+[row['fee'] for row in data['HOLDINGS']]
            fees+=[result['metrics']['fees_paid'],result['slices']['full122']['fees_paid']]
            require(all(type(v) in (int,float) and math.isfinite(v) and v==0. for v in fees),'Every zero fee exact0')
    mapping=dict(zip(ARMS,prior.ARMS))
    renamed=lambda d:{mapping[k]:v for k,v in d.items()}
    result=prior.audit_outcomes(px,opening,renamed(orders),renamed(exports),renamed(results),meta)
    result=dict(result,paths={arm:result['paths'][mapping[arm]] for arm in ARMS})
    return result
