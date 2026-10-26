"""Pure draft formation scenarios; no file I/O, strategy imports or real-data run.

The caller must verify no-delete ranks against the bound incumbent core before
using any scenario. All returned prices/factors end at the supplied signal row.
"""
import numpy as np


def centered_rank(values):
    values=np.asarray(values,dtype=float)
    if values.ndim!=1 or np.isinf(values).any():
        raise ValueError('Finite-or-NaN one-dimensional values required')
    answer=np.zeros(values.shape,dtype=float)
    mask=np.isfinite(values)
    if not mask.any():return answer
    good=values[mask]
    # Average rank, matching the frozen pandas average-tie percentile convention.
    answer[mask]=np.asarray([(np.sum(good<x)+0.5*(np.sum(good==x)+1))/len(good)-.5 for x in good])
    return answer


def formation_scenarios(prices, signal_index, *, calm):
    """Return no-delete core ranks and ten/twelve common21-return deletions.

    Missing internal history of an originally measurable core forces whole-month
    fallback. NaN endpoint/divisor yields original missing-core neutrality.
    Future rows are neither converted to float nor validated.
    """
    raw=np.asarray(prices)
    if raw.ndim!=2 or raw.shape[1]<1:
        raise ValueError('Nonempty [observations,securities] prices required')
    if isinstance(signal_index,(bool,np.bool_)) or not isinstance(signal_index,(int,np.integer)):
        raise ValueError('Integer signal index required')
    if not 0<=signal_index<len(raw):raise ValueError('Signal index out of range')
    if not isinstance(calm,(bool,np.bool_)):raise ValueError('Boolean frozen calm state required')
    horizon=252 if calm else 294
    result={'ready':False,'reason':None,'core_weight':.5 if calm else .52,
            'horizon':horizon,'skip':42,'blocks':[],'core_ranks':None,'scenario_core_ranks':None,
            'original_finite_mask':None}
    if signal_index<horizon:
        result['reason']='insufficient_formation_history';return result
    first=signal_index-horizon;last=signal_index-42
    formation=np.asarray(raw[first:last+1],dtype=float)
    if np.isinf(formation).any():raise ValueError('Infinite formation price')
    endpoint_good=np.isfinite(formation[[0,-1]]).all(axis=0)&(formation[[0,-1]]>0).all(axis=0)
    total=np.full(raw.shape[1],np.nan)
    with np.errstate(over='ignore',under='ignore',divide='ignore',invalid='ignore'):
        total[endpoint_good]=formation[-1,endpoint_good]/formation[0,endpoint_good]
    if not (np.isfinite(total[endpoint_good])&(total[endpoint_good]>0)).all():
        raise FloatingPointError('Nonpositive or nonfinite formation gross return')
    denominator=np.ones(raw.shape[1],dtype=float)
    if not calm:
        recent=np.asarray(raw[signal_index-63:signal_index+1],dtype=float)
        if np.isinf(recent).any():raise ValueError('Infinite risk-window price')
        risk_good=np.isfinite(recent).all(axis=0)&(recent>0).all(axis=0)
        denominator[:]=np.nan
        if risk_good.any():
            denominator[risk_good]=np.std(recent[1:,risk_good]/recent[:-1,risk_good]-1,axis=0,ddof=1)
        denominator[denominator==0]=np.nan
    valid=endpoint_good&np.isfinite(denominator)&(denominator>0)
    no_delete=np.full(raw.shape[1],np.nan)
    no_delete[valid]=(total[valid]-1)/denominator[valid]
    if np.isinf(no_delete).any():raise FloatingPointError('Nonfinite no-delete core')
    result['core_ranks']=centered_rank(no_delete)
    result['original_finite_mask']=valid.copy()
    complete=np.isfinite(formation).all(axis=0)&(formation>0).all(axis=0)
    if not np.all(complete[valid]):
        result['reason']='incomplete_originally_measurable_formation';return result
    count=horizon-42
    assert count%21==0 and formation.shape[0]==count+1
    scenarios=[]
    for start in range(0,count,21):
        end=start+21
        values=np.full(raw.shape[1],np.nan)
        with np.errstate(over='ignore',under='ignore',divide='ignore',invalid='ignore'):
            block_ratio=formation[end,valid]/formation[start,valid]
        if not (np.isfinite(block_ratio)&(block_ratio>0)).all():
            raise FloatingPointError('Nonpositive or nonfinite deleted-block gross return')
        with np.errstate(over='ignore',under='ignore',divide='ignore',invalid='ignore'):
            values[valid]=(total[valid]/block_ratio-1)/denominator[valid]
        if not np.array_equal(np.isfinite(values),valid):
            raise FloatingPointError('Scenario changed original finite-core mask')
        scenarios.append(centered_rank(values))
        result['blocks'].append({'start_price_index':first+start,'end_price_index':first+end,'deleted_returns':21})
    result.update(ready=True,reason='ready',scenario_core_ranks=np.stack(scenarios))
    return result
