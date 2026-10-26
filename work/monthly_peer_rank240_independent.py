"""Independent within-sector mid-CDF and funded Local67 selection checks.

Only the pinned prior independent oracle is reused for source admission,
endpoint momentum and legacy median evidence. New CDF arithmetic is rebuilt
here; no new producer, host, strategy engine or real-data entry point is used.
"""
import hashlib
import importlib.util
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT/'work/monthly_peer239_independent.py'
PRIOR_SHA = '412c3f1e5f5c185d4eedea5cc954fcbc81737ae370f24d09cbdb96c166340064'
PRIOR_TEST = ROOT/'work/test_monthly_peer239_independent.py'
PRIOR_TEST_SHA = '938d0de8b29e739502107174b44707333830b7e1c49e6df9734a146f2132ff53'
ARMS = ('incumbent_local67','within_sector_midcdf','global_midcdf_control')


def prior():
    if hashlib.sha256(PRIOR.read_bytes()).hexdigest()!=PRIOR_SHA:
        raise ValueError('Frozen independent source/selection contract changed')
    spec=importlib.util.spec_from_file_location('peer_rank240_prior_independent',PRIOR)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


old=prior()
require,exact_array,record_check=old.require,old.exact_array,old.record_check
VALUE_KEYS,COUNT_FIELDS=old.VALUE_KEYS,old.COUNT_FIELDS
PATHS,ACCOUNTING_SHA=old.PATHS,old.ACCOUNTING_SHA
FEATURE_FLAGS=tuple({'shared_supported_rank_domain_checked':'shared_supported_availability_checked',
    'average_percentile_ties_checked':'midcdf_ties_checked'}.get(k,k) for k in old.FEATURE_FLAGS)+(
    'within_sector_self_included','integer_midcdf_numerator_checked',
    'single_final_division_checked','global_midcdf_matched_support_checked',
    'constant_group_zero_checked','no_global_rerank_of_peer_targets')
SELECTION_FLAGS=old.SELECTION_FLAGS


def frozen_source_oracle():return old.frozen_source_oracle()
def accounting():return old.accounting()


def dependency_files():
    require(hashlib.sha256(PRIOR_TEST.read_bytes()).hexdigest()==PRIOR_TEST_SHA,'Frozen synthetic source fixture changed')
    return tuple(sorted(set(old.dependency_files())|{PRIOR,PRIOR_TEST,Path(__file__).resolve()}))


def midcdf(value,population):
    """Signed pair comparisons, with exact integer accumulation and one divide."""
    require(len(population)>0,'Nonempty CDF population')
    require(math.isfinite(float(value)) and all(math.isfinite(float(x)) for x in population),'Finite CDF values')
    numerator=0
    for other in population:
        if other<value:numerator+=1
        elif other>value:numerator-=1
    denominator=2*len(population)
    return numerator,denominator,float(numerator)/float(denominator)


def independent_features(panel,meta,calendar,arrays,source_report,limit=None):
    original,previous=prior().independent_features(panel,meta,calendar,arrays,source_report,limit)
    values={k:v.copy() for k,v in original.items()}
    values['peer_target'][:]=np.nan;values['absolute_target'][:]=np.nan
    for event in previous['events']:
        i=event['query_index'];global_ids=event['supported_indices']
        global_population=[float(values['momentum'][i,j]) for j in global_ids]
        for cell in event['cells']:
            j=cell['column_index']
            for field in ('peer_numerator','peer_denominator','absolute_numerator','absolute_denominator'):
                cell[field]=None
            cell['peer_target']=cell['absolute_target']=None
            if not values['supported'][i,j]:continue
            # Prior source oracle independently established this complete,
            # same-sector, distinct-issuer peer list without eligibility use.
            cohort=sorted([j]+cell['peer_indices'])
            require(len(cohort)>=3 and len(set(cohort))==len(cohort),'Self plus at least two distinct peers')
            point=float(values['momentum'][i,j])
            pn,pd,q=midcdf(point,[float(values['momentum'][i,k]) for k in cohort])
            an,ad,c=midcdf(point,global_population)
            values['peer_target'][i,j]=q;values['absolute_target'][i,j]=c
            cell.update(peer_target=q,absolute_target=c,peer_numerator=pn,peer_denominator=pd,
                        absolute_numerator=an,absolute_denominator=ad)
    return values,dict(schema='monthly_peer_rank240_features_v1',counts=previous['counts'],events=previous['events'])


def verify_features(panel,meta,calendar,arrays,source_report,values,audit,limit=None):
    expected,ea=independent_features(panel,meta,calendar,arrays,source_report,limit)
    require(isinstance(values,dict) and set(values)==set(VALUE_KEYS),'Exactly six feature arrays')
    for key in VALUE_KEYS:exact_array(values[key],expected[key],'Independent '+key)
    record_check(audit,ea)
    return dict(**{k:True for k in FEATURE_FLAGS},**ea['counts'],counts=ea['counts'],numerical_tolerance=0.,
                producer_modules_imported=False,computes_returns=False)


def selection_evidence(original,meta,values,n):
    _,width,days,names,etfs=old.axes(meta,original,n)
    require(isinstance(values,dict) and set(values)==set(VALUE_KEYS),'Selection feature keys')
    v={k:np.asarray(values[k])[:n] for k in VALUE_KEYS}
    for k in VALUE_KEYS:
        dtype=np.dtype(bool if k in ('mature','supported') else np.float64)
        require(v[k].shape==(n,width) and v[k].dtype==dtype,'Feature shape/dtype '+k)
    require(not (v['supported'] & ~v['mature']).any(),'Supported subset of mature')
    for k in VALUE_KEYS[2:]:
        mask=v['mature'] if k=='momentum' else v['supported']
        require(np.isfinite(v[k][mask]).all() and np.isnan(v[k][~mask]).all(),'Feature availability '+k)
    for k in ('peer_target','absolute_target'):
        require((np.abs(v[k][v['supported']])<.5).all(),'Finite-population mid-CDF range '+k)
    for i,(d,ss) in enumerate(zip(days,names)):
        forbidden=[j for j,s in enumerate(ss) if s in etfs or str(d.date())=='2016-08-17']
        require(not v['mature'][i,forbidden].any(),'ETF/partial no new features')
    # Source cohorts are intentionally not inferred from the values alone.
    # Their exact Q formula is independently established by verify_features.
    return v,width


def _selection(original,meta,values,n=None):
    n=len(meta['dates']) if n is None else n
    require(type(n) is int and 0<=n<=len(meta['dates']),'Selection prefix')
    v,width=selection_evidence(original,meta,values,n)
    if n==0:
        empty={arm:np.empty((0,width)) for arm in ARMS}
        return empty,{arm:[] for arm in ARMS},{arm:x.copy() for arm,x in empty.items()},dict(
            minimum_remaining_support=None,maximum_budget_sum_error=0.,calm_nonweak_extension_count=0,
            maximum_absolute_delta=0.)
    x=frozen_source_oracle().old.original_helper().original_prefix(original,meta,n)
    require(np.array_equal(x['cw'],np.broadcast_to(np.where(x['calm'],.50,.52)[:,None],(n,width))),
            'Original calm-dependent core coefficient')
    extension=np.asarray(original['extension_available'])[:n]
    forecasts=np.asarray(original['forecast_available'])[:n]
    ranklog=np.asarray(original['rank_log_excess'])[:n]
    require(extension.dtype==forecasts.dtype==np.dtype(bool) and extension.shape==forecasts.shape==ranklog.shape==(n,width),
            'Original extension-mask axes')
    active=(x['calm'] & ~x['weak'])[:,None] & extension & forecasts & x['eligible'] & np.isfinite(ranklog)
    budget=np.minimum(.05,np.maximum(0.,.10-(.025*x['calm']+.05*x['weak'])))
    scores={};choices={};pres={};minimum=1.;sum_error=0.;max_delta=0.
    for arm,key in zip(ARMS,(None,'peer_target','absolute_target')):
        pre=x['local'].copy();delta=np.zeros((n,width));hand=np.zeros((n,width))
        if key is not None:
            good=np.isfinite(v[key]);hand[good]=.05
            delta[good]=.05*(v[key][good]-x['support'][good])
            require(np.isfinite(delta).all(),'Finite funded score delta')
            nonzero=delta!=0.;pre[nonzero]+=delta[nonzero]
        max_delta=max(max_delta,float(np.max(np.abs(delta))))
        require(np.isfinite(pre).all(),'Finite modified original pre')
        pres[arm]=pre.copy();score=pre.copy();rows=[];boundary_mask=np.zeros((n,width),bool)
        for i,names in enumerate(x['names']):
            order=sorted([j for j in range(width) if x['eligible'][i,j]],key=lambda j:(-pre[i,j],names[j]))
            if len(order)<3:rows.append(order if len(order)==2 else []);continue
            boundary=[j for j in order[2:5] if float(pre[i,order[2]])-float(pre[i,j])<=.02]
            for j in boundary:
                boundary_mask[i,j]=True
                target=min(.5,max(-.5,x['support'][i,j]+x['effects'][i,j]))
                score[i,j]+=budget[i]*(target-x['support'][i,j])
            winner=min(boundary,key=lambda j:(-score[i,j],names[j]))
            rows.append(sorted(order[:2]+[winner],key=lambda j:(-score[i,j],names[j])))
        require(np.isfinite(score).all(),'Finite final scores')
        for i in range(n):
            for j in range(width):
                weights=[float(x['cw'][i,j]),.025*float(x['calm'][i]),.05*float(x['weak'][i]),
                    .01*float(active[i,j]),float(budget[i])*float(boundary_mask[i,j]),float(hand[i,j])]
                remaining=1.-weights[0]
                for w in weights[1:]:remaining-=w
                error=abs(math.fsum(weights+[remaining])-1.)
                require(remaining>=.33-1e-12 and error<=1e-12,'Complete Local67 budget including calm nonweak extension')
                minimum=min(minimum,remaining);sum_error=max(sum_error,error)
        scores[arm]=score;choices[arm]=rows
    return scores,choices,pres,dict(minimum_remaining_support=minimum,maximum_budget_sum_error=sum_error,
        calm_nonweak_extension_count=int(active.sum()),maximum_absolute_delta=max_delta)


def selection_oracle(original,meta,values,n=None):
    scores,choices,pres,_=_selection(original,meta,values,n)
    return scores,choices,pres


def verify_selection(original,meta,values,orders,*,scores=None,pre_scores=None,n=None):
    n=len(meta['dates']) if n is None else n
    expected,choices,pres,budget=_selection(original,meta,values,n)
    require(isinstance(orders,dict) and set(orders)==set(ARMS),'Three order arms')
    wanted={arm:[dict(date=meta['dates'][i],end=meta['ends'][i],signal_date=meta['dates'][i],
        symbols=[meta['symbols'][i][j] for j in ids],scores=[float(expected[arm][i,j]) for j in ids])
        for i,ids in enumerate(choices[arm])] for arm in ARMS}
    record_check(orders,wanted,'Independent ordered selection')
    for supplied,reference,label in ((scores,expected,'post-episode score'),(pre_scores,pres,'modified pre-score')):
        if supplied is not None:
            require(isinstance(supplied,dict) and set(supplied)==set(ARMS),'Three '+label+' arms')
            for arm in ARMS:exact_array(supplied[arm],reference[arm],label+'/'+arm)
    return dict(**{k:True for k in SELECTION_FLAGS},**budget,checked_events=n,ordered_choices_checked=3*n,
        full_scores_supplied=scores is not None,full_pre_scores_supplied=pre_scores is not None,
        numerical_tolerance=0.,original_source_parity_tolerance=1e-12,
        producer_modules_imported=False,computes_returns=False,
        holdings_counts={arm:{str(k):sum(len(row)==k for row in choices[arm]) for k in (0,2,3)} for arm in ARMS})


def audit_outcomes(px,opening,orders,exports,results,meta):
    """Reuse pinned independent 9-path/45-slice ledger arithmetic, renaming only."""
    require(set(orders)==set(exports)==set(results)==set(ARMS),'Outcome arms')
    mapping=dict(zip(old.ARMS,ARMS))
    report=prior().audit_outcomes(px,opening,*[{k:data[v] for k,v in mapping.items()}
        for data in (orders,exports,results)],meta)
    report['paths']={mapping[k]:v for k,v in report['paths'].items()}
    return report
