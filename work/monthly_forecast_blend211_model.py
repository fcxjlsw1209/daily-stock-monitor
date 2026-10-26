"""Pure annual convex combination of immutable, causal round206 forecasts.

No cache loading, label construction, source-model refits, ranks or portfolios.
Only mature exact-identity joins contribute to the single yearly coefficient.
"""
from datetime import date
from pathlib import Path
import math
import numbers

import numpy as np

ARMS = ('incumbent_local67', 'annual_convex_blend', 'fixed_half_control')
TARGETS = ('simple_excess', 'log_excess', 'vol_scaled', 'loss_utility')
MIN_MONTHS = 18
COUNT_FIELDS = ('annual_coefficient_fits', 'cold_start_years', 'zero_denominator_years',
    'clipped_coefficient_years', 'calibration_months', 'calibration_rows',
    'joined_label_rows', 'invalid_label_rows', 'small_month_rows', 'cached_forecast_rows')
INITIAL = '2016-08-17'


def dependency_files():
    return (Path(__file__).resolve(),)


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _day(value):
    _require(isinstance(value, str), 'Dates must be ISO strings')
    try:
        d = date.fromisoformat(value)
    except (ValueError, TypeError) as exc:
        raise ValueError('Invalid ISO day') from exc
    _require(d.isoformat() == value, 'Noncanonical ISO day')
    return value


def _month(value):
    return int(value[:4])*12+int(value[5:7])


def _metadata(meta, width=None, initial_query_exception=False):
    _require(isinstance(meta, dict) and all(k in meta for k in ('dates','ends','symbols')),
             'Explicit dates, ends and security identities required')
    days, ends, names = (meta[k] for k in ('dates','ends','symbols'))
    _require(all(isinstance(x, list) for x in (days,ends,names)) and len(days)==len(ends)==len(names),
             'Monthly metadata axes differ')
    result=[]
    for i,(d,e,ss) in enumerate(zip(days,ends,names)):
        d,e=_day(d),_day(e)
        _require(e>d and (_month(e)==_month(d)+1 or
                 (initial_query_exception and i==0 and d==INITIAL and e=='2016-08-31')),
                 'Labels must end in next natural month, except initial partial')
        _require(i==0 or d>days[i-1], 'Signal dates must be unique and increasing')
        initial_pair=(initial_query_exception and i==1 and days[0]==INITIAL and d=='2016-08-31')
        _require(i==0 or _month(d)>_month(days[i-1]) or initial_pair, 'Duplicate signal month')
        _require(isinstance(ss,list) and (width is None or len(ss)==width)
                 and all(isinstance(s,str) and s for s in ss) and len(set(ss))==len(ss),
                 'Explicit unique security IDs and aligned width required')
        result.append((d,e,ss))
    return result


def _boolean(value, label):
    _require(isinstance(value,(bool,np.bool_)), 'Exact boolean required: '+label)
    return bool(value)


def _number(value, label):
    _require(isinstance(value,numbers.Real) and not isinstance(value,(bool,np.bool_)),
             'Numeric nonboolean value required: '+label)
    return float(value)


def _finite(value, label):
    if not math.isfinite(value):
        raise FloatingPointError('Nonfinite '+label)
    return value


def _sum(values, label):
    try:
        value=math.fsum(values)
    except (OverflowError,ValueError) as exc:
        raise FloatingPointError('Nonfinite '+label) from exc
    return _finite(value,label)


def blend(peer, own, alpha):
    """Declared arithmetic, including exact source-copy identity branches."""
    peer,own,alpha=(_finite(_number(x,'blend'),'blend') for x in (peer,own,alpha))
    _require(0.<=alpha<=1., 'Convex coefficient outside bounds')
    if alpha==0. or own==peer:
        return peer
    if alpha==1.:
        return own
    difference=_finite(own-peer,'blend difference')
    return _finite(peer+_finite(alpha*difference,'weighted blend difference'),'blended forecast')


def coefficient(numerator, denominator, months):
    """One constrained least-squares coefficient; no numerical floor or search."""
    numerator=_finite(_number(numerator,'numerator'),'numerator')
    denominator=_finite(_number(denominator,'denominator'),'denominator')
    _require(type(months) is int and months>=0 and denominator>=0., 'Invalid coefficient support')
    if months<MIN_MONTHS:
        return .5, None, False, 'cold_start'
    if denominator==0.:
        return .5, None, False, 'zero_denominator'
    value=_finite(numerator/denominator,'unclipped coefficient')
    alpha=min(1.,max(0.,value))
    return alpha,value,alpha!=value,'learned'


def _inputs(train, training_meta, query_meta, source, eligible):
    _require(isinstance(train,dict) and isinstance(source,dict),'Caller array dictionaries required')
    required={'offsets','dates','ends','train_plain','common_good'} | {'target_'+k for k in TARGETS}
    _require(required<=set(train),'Original cached training fields missing')
    _require({'raw_candidate','raw_control','raw_source_log','available'}<=set(source),
             'Original cached forecast fields missing')
    eligible=np.asarray(eligible)
    _require(eligible.ndim==2 and eligible.shape[1]>0,'Eligibility must be a matrix with positive width')
    n,width=eligible.shape
    queries=_metadata(query_meta,width,initial_query_exception=True)
    _require(len(queries)==n,'Query metadata differs from eligibility axis')
    history=_metadata(training_meta)
    offsets=[0]
    for _,_,symbols in history:
        offsets.append(offsets[-1]+len(symbols))
    provided=np.asarray(train['offsets'])
    _require(provided.shape==(len(offsets),) and provided.dtype.kind in 'iu'
             and np.array_equal(provided,np.asarray(offsets)), 'Training offsets do not match explicit identities')
    total=offsets[-1]
    rows={}
    td,te=np.asarray(train['dates']),np.asarray(train['ends'])
    _require(td.shape==te.shape==(total,), 'Training date axes differ')
    for i,(d,e,names) in enumerate(history):
        for j,symbol in enumerate(names,offsets[i]):
            _require(str(td[j])==d and str(te[j])==e,'Flattened training date/offset identity differs')
            key=(d,symbol,e)
            _require(key not in rows,'Duplicate training triple identity')
            rows[key]=j
    data={k:np.asarray(train[k]) for k in required if k not in ('offsets','dates','ends')}
    _require(data['train_plain'].shape==(total,10),'Original ten-feature training matrix required')
    _require(all(v.shape==(total,) for k,v in data.items() if k!='train_plain'),'Training vectors differ')
    q=n*width
    available=np.asarray(source['available'])
    _require(available.shape==(q,),'Forecast availability axis differs')
    available=np.array([_boolean(x,'source available') for x in available],dtype=bool)
    eligibility=np.array([_boolean(x,'query eligible') for x in eligible.reshape(-1)],dtype=bool)
    raw={}
    for key in ('raw_candidate','raw_control','raw_source_log'):
        values=np.asarray(source[key])
        _require(values.shape==(q,), 'Forecast vector axis differs: '+key)
        out=np.array([_number(v,key) for v in values],dtype=float)
        _require(np.array_equal(np.isfinite(out),available) and not np.isinf(out).any(),
                 'Forecast finite mask differs: '+key)
        raw[key]=out
    return queries,rows,data,raw,available,eligibility,width


def build_forecasts(train, training_meta, query_meta, source, eligible):
    """Combine original206 forecasts; only exact mature past labels are inspected."""
    queries,index,data,raw,available,eligibility,width=_inputs(train,training_meta,query_meta,source,eligible)
    n=len(queries);q=n*width
    out={key:np.full(q,np.nan) for key in ('candidate','control')}
    annual=[];counts={key:0 for key in COUNT_FIELDS}
    counts['cached_forecast_rows']=int(available.sum())
    peer,own=raw['raw_candidate'],raw['raw_control']
    years=sorted({int(d[:4]) for d,_,_ in queries})
    for year in years:
        cutoff=f'{year-1:04d}-12-31';start=f'{year-6:04d}-12-31'
        current=[i*width+j for i,(d,_,_) in enumerate(queries) if int(d[:4])==year
                 for j in range(width) if available[i*width+j]]
        row=dict(year=year,cutoff=cutoff,lookback_exclusive_start=start,status=None,alpha=None,
            unclipped_alpha=None,clipped=False,prediction_rows=len(current),skipped_initial_rows=0,
            joined_label_rows=0,invalid_label_rows=0,small_month_rows=0,calibration_rows=0,
            calibration_months=0,numerator=0.,denominator=0.,latest_label_end=None,months=[])
        for i,(d,e,names) in enumerate(queries):
            if not (d>start and e<=cutoff):
                continue
            ids=[i*width+j for j in range(width) if eligibility[i*width+j] and available[i*width+j]]
            if d==INITIAL:
                row['skipped_initial_rows']+=len(ids)
                continue
            month=dict(date=d,end=e,source_rows=len(ids),invalid_label_rows=0,source_row_ids=[],
                training_row_ids=[],symbols=[],targets=[],own_predictions=[],peer_predictions=[],
                numerator_terms=[],denominator_terms=[],usable_rows=0,accepted=False,
                numerator_mean=None,denominator_mean=None)
            for source_id in ids:
                symbol=names[source_id%width];key=(d,symbol,e)
                _require(key in index,'Missing mature exact (signal, security, end) label: '+repr(key))
                tid=index[key];row['joined_label_rows']+=1
                if not _boolean(data['common_good'][tid],'common_good'):
                    month['invalid_label_rows']+=1
                    continue
                xx=[_number(x,'train_plain') for x in data['train_plain'][tid]]
                yy={k:_number(data['target_'+k][tid],'target_'+k) for k in TARGETS}
                if not all(math.isfinite(v) for v in xx+list(yy.values())):
                    month['invalid_label_rows']+=1
                    continue
                month['source_row_ids'].append(source_id);month['training_row_ids'].append(tid)
                month['symbols'].append(symbol);month['targets'].append(yy['log_excess'])
                month['own_predictions'].append(float(own[source_id]));month['peer_predictions'].append(float(peer[source_id]))
            count=len(month['source_row_ids']);month['usable_rows']=count
            row['invalid_label_rows']+=month['invalid_label_rows']
            if count>=2:
                for target,o,p in zip(month['targets'],month['own_predictions'],month['peer_predictions']):
                    delta=_finite(o-p,'historical prediction difference')
                    error=_finite(target-p,'historical peer error')
                    month['numerator_terms'].append(_finite(error*delta,'coefficient cross product'))
                    month['denominator_terms'].append(_finite(delta*delta,'coefficient square'))
                month['numerator_mean']=_finite(_sum(month['numerator_terms'],'monthly numerator')/count,'monthly numerator mean')
                month['denominator_mean']=_finite(_sum(month['denominator_terms'],'monthly denominator')/count,'monthly denominator mean')
                month['accepted']=True;row['calibration_months']+=1;row['calibration_rows']+=count
                row['latest_label_end']=e
            else:
                row['small_month_rows']+=count
            row['months'].append(month)
        row['numerator']=_sum([m['numerator_mean'] for m in row['months'] if m['accepted']],'annual numerator')
        row['denominator']=_sum([m['denominator_mean'] for m in row['months'] if m['accepted']],'annual denominator')
        alpha,unclipped,clipped,status=coefficient(row['numerator'],row['denominator'],row['calibration_months'])
        row.update(alpha=alpha,unclipped_alpha=unclipped,clipped=clipped,status=status)
        counts[{'learned':'annual_coefficient_fits','cold_start':'cold_start_years',
                'zero_denominator':'zero_denominator_years'}[status]]+=1
        counts['clipped_coefficient_years']+=int(clipped)
        for key in ('calibration_months','calibration_rows','joined_label_rows','invalid_label_rows','small_month_rows'):
            counts[key]+=row[key]
        for j in current:
            out['candidate'][j]=blend(float(peer[j]),float(own[j]),alpha)
            out['control'][j]=blend(float(peer[j]),float(own[j]),.5)
        annual.append(row)
    return dict(raw=out,available=available.copy(),audit=dict(method='annual_mature_asof_convex_forecast_blend',
        arms=list(ARMS),query_months=n,query_rows=q,symbols_per_event=width,min_months=MIN_MONTHS,
        lookback_calendar_years=5,month_equal_weights=True,source_forecasts_refitted=False,
        labels_recomputed=False,local_available_filtered=False,annual=annual,counts=counts,
        new_model_fits=counts['annual_coefficient_fits'],predictive_model_fits=counts['annual_coefficient_fits'],
        new_ridge_fits=0,new_tree_fits=0))
