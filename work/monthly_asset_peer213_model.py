"""Same-asset peer transfer with the frozen206 all-peer forecast as control.

Pure caller-array model. The only regressions performed here are one pooled
log-target Ridge per predicted annual cohort. Cached source/local forecasts
are copied; own weights and all/same peer corrections are explicitly counted.
"""
from pathlib import Path
import importlib.util

import numpy as np

REFERENCE_PATH = Path(__file__).with_name('monthly_identity_transfer206_model.py').resolve()
_spec=importlib.util.spec_from_file_location('asset_peer213_frozen_helpers',REFERENCE_PATH)
reference=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(reference)
local,numeric=reference.local,reference.numeric
TARGETS=reference.TARGETS
MIN_MONTHS=MIN_OWN_MONTHS=18
ALPHA,HALF_LIFE_MONTHS,BANDWIDTH,INTERCEPT_PENALTY=10.,12.,.5,12.
ARMS=('same_asset_peer_transfer','all_asset_peer_control')
RAW_KEYS=('source_log','local_linear','local_constant','candidate','control')
CACHE_KEYS={'raw_'+key for key in RAW_KEYS}|{'available','local_available'}
COUNT_FIELDS=('annual_pooled_log_refits','cached_forecast_rows','own_weight_evaluations',
    'all_peer_query_evaluations','all_peer_month_evaluations','all_peer_row_observations',
    'same_type_supported_queries','same_type_fallback_queries','same_type_month_evaluations',
    'same_type_row_observations','actual_regression_solves')
CORRECTION_ACCOUNTING=('One pooled log Ridge refit per predicted annual cohort; own constant, all-peer and supported same-type corrections counted separately; '
    'cached four-target/local-linear fits are not repeated; prefix/oracle repeats excluded.')


def dependency_files():
    return tuple(sorted(set(reference.dependency_files())|{REFERENCE_PATH,Path(__file__).resolve()}))


def _cache(cached,q):
    if not isinstance(cached,dict) or set(cached)!=CACHE_KEYS:raise ValueError('Exactly seven original206 cached vectors required')
    available=local.original._bool(cached['available'],q,'cached available').copy()
    own=local.original._bool(cached['local_available'],q,'cached local_available').copy()
    if np.any(own & ~available):raise ValueError('Local support outside cached availability')
    raw={}
    for key in RAW_KEYS:
        value=np.asarray(cached['raw_'+key])
        if value.shape!=(q,) or value.dtype!=np.dtype('float64') or np.isinf(value).any() or not np.array_equal(np.isfinite(value),available):
            raise ValueError('Original finite float64 cache mask required: '+key)
        raw[key]=value.copy()
    np.testing.assert_array_equal(raw['control'],raw['local_constant'])
    return raw,available,own


def _etfs(etfs):
    if not isinstance(etfs,(list,tuple)) or any(not isinstance(x,str) or not x or x.strip()!=x for x in etfs):
        raise ValueError('Explicit ETF identity sequence required')
    if len(set(etfs))!=len(etfs):raise ValueError('ETF identities must be unique')
    return list(etfs),set(etfs)


def _add(source,correction):
    value=float(source)
    if correction!=0.:
        with np.errstate(over='raise',invalid='raise'):value=float(np.float64(value)+correction)
    reference._finite('reconstructed forecast',value)
    return value


def annual_forecasts(dates,ends,symbols,X,targets,common_good,event_dates,event_symbols,event_X,event_good,
                     *,etfs,cached_reference,min_months=MIN_MONTHS,min_own_months=MIN_OWN_MONTHS):
    local._minimum(min_months,'min_months');local._minimum(min_own_months,'min_own_months')
    etf_ids,etf_set=_etfs(etfs)
    args=reference._inputs(dates,ends,symbols,X,targets,common_good,event_dates,event_symbols,event_X,event_good)
    dates,ends,symbols,X,ys,common_good,event_dates,event_symbols,event_X,event_good=args
    cache,available,local_available=_cache(cached_reference,len(event_dates))
    raw={key:cache[key].copy() for key in RAW_KEYS[:3]}
    raw.update(candidate=cache['candidate'].copy(),control=cache['candidate'].copy())
    months=dates.astype('datetime64[M]');years=event_dates.astype('datetime64[Y]').astype(int)+1970
    usable=common_good & np.isfinite(X).all(axis=1)
    for y in ys.values():usable &= np.isfinite(y)
    for month in np.unique(months):
        rows=months==month
        if (usable & rows).sum()<2:usable[rows]=False
    events=event_good & np.isfinite(event_X).all(axis=1)
    rebuilt_available=np.zeros(len(event_dates),bool);rebuilt_local=np.zeros(len(event_dates),bool)
    counts={key:0 for key in COUNT_FIELDS};counts['cached_forecast_rows']=int(available.sum())
    annual=[];details=[]
    with local.threadpool_limits(limits=1):
        for year in np.unique(years):
            cutoff=np.datetime64('%04d-12-31'%(year-1));start=numeric._years_before(cutoff)
            ids=np.flatnonzero(usable & (dates>start) & (ends<=cutoff));nm=numeric._month_count(dates[ids])
            if nm>=min_months:rebuilt_available[(years==year)&events]=True
            np.testing.assert_array_equal(rebuilt_available[years==year],available[years==year])
            predict=np.flatnonzero((years==year)&available)
            record=dict(year=int(year),cutoff=str(cutoff),lookback_exclusive_start=str(start),
                training_rows=len(ids),training_months=nm,prediction_rows=len(predict),training_row_ids=ids.tolist(),
                first_training_signal=str(dates[ids].min()) if len(ids) else None,
                latest_label_end=str(ends[ids].max()) if len(ids) else None,
                status='fit' if nm>=min_months else 'cold_start',weight_sum=None)
            if len(predict):
                identities=list(zip(months[ids].astype(str),symbols[ids]))
                if len(set(identities))!=len(ids):raise ValueError('Duplicate monthly security identity')
                local._ranks('mature training ranks',X[ids]);local._ranks('actual query ranks',event_X[predict])
                weights=numeric.month_balanced_weights(dates[ids],cutoff,HALF_LIFE_MONTHS)
                pooled=numeric._fit_ridge(X[ids],ys['log_excess'][ids],weights,ALPHA)
                counts['annual_pooled_log_refits']+=1
                np.testing.assert_array_equal(numeric._predict(pooled,event_X[predict]),raw['source_log'][predict])
                with np.errstate(over='raise',invalid='raise'):residual=ys['log_excess'][ids]-numeric._predict(pooled,X[ids])
                reference._finite('mature pooled residuals',residual);record['weight_sum']=float(weights.sum())
                for j in predict:
                    own=np.flatnonzero(symbols[ids]==event_symbols[j]);own_ids=ids[own]
                    own_months=numeric._month_count(dates[own_ids]);query_etf=event_symbols[j] in etf_set
                    row=dict(event_row=int(j),date=str(event_dates[j]),symbol=event_symbols[j],asset_type='etf' if query_etf else 'stock',
                        year=int(year),cutoff=str(cutoff),own_rows=len(own),own_months=own_months,
                        own_row_ids=own_ids.tolist(),own_dates=dates[own_ids].astype(str).tolist(),
                        own_label_ends=ends[own_ids].astype(str).tolist(),latest_label_end=str(ends[own_ids].max()) if len(own) else None,
                        status='insufficient_own_history',weight_mass=0.,effective_samples=0.,own_weights=[],own_residuals=[],
                        own_numerator=0.,correction_denominator=None,constant_correction=0.,all_peer_numerator=0.,all_peer_correction=0.,
                        same_peer_numerator=None,same_peer_correction=None,missing_same_type_months=[],peer_months=[])
                    if own_months<min_own_months:
                        np.testing.assert_array_equal(cache['local_constant'][j],raw['source_log'][j])
                        np.testing.assert_array_equal(cache['candidate'][j],raw['source_log'][j])
                        raw['candidate'][j]=raw['control'][j]
                    else:
                        w=local.local_weights(dates[own_ids],cutoff,X[own_ids],event_X[j]);mass=float(w.sum())
                        with np.errstate(over='raise',invalid='raise',divide='raise'):
                            own_num=float(np.einsum('n,n->',w,residual[own],optimize=False));denominator=mass+INTERCEPT_PENALTY
                            c_own=own_num/denominator if own_num!=0. else 0.
                            effective=float(mass**2/np.einsum('n,n->',w,w,optimize=False)) if mass else 0.
                        for name,value in (('own mass',mass),('own numerator',own_num),('own denominator',denominator),
                            ('own constant',c_own),('effective own samples',effective)):reference._finite(name,value)
                        rebuilt_local[j]=mass>0.
                        np.testing.assert_array_equal(_add(raw['source_log'][j],c_own),cache['local_constant'][j])
                        all_means=[];peer_records=[];same_positions=[]
                        for position,own_id,wk in zip(own,own_ids,w):
                            peers=np.flatnonzero((months[ids]==months[own_id])&(symbols[ids]!=event_symbols[j]))
                            if not len(peers):raise ValueError('Own mature month has no other common-cohort security')
                            mean,info=reference.peer_month_average(X[ids[peers]],event_X[j],residual[peers])
                            same=np.array([p for p in peers if (symbols[ids[p]] in etf_set)==query_etf],dtype=int)
                            same_positions.append(same);all_means.append(mean)
                            if not len(same):row['missing_same_type_months'].append(str(months[own_id]))
                            peer_records.append(dict(month=str(months[own_id]),own_row_id=int(own_id),own_weight=float(wk),
                                own_residual=float(residual[position]),peer_row_ids=ids[peers].tolist(),**info,
                                same_peer_row_ids=ids[same].tolist(),same_peer_count=len(same),same_peer_weight_mass=None,
                                same_peer_weighted_residual=None,same_peer_mean=None))
                            counts['all_peer_month_evaluations']+=1;counts['all_peer_row_observations']+=len(peers)
                        with np.errstate(over='raise',invalid='raise',divide='raise'):
                            all_num=float(np.einsum('n,n->',w,np.asarray(all_means),optimize=False))
                            c_all=all_num/denominator if all_num!=0. else 0.
                        reference._finite('all-peer numerator',all_num);reference._finite('all-peer correction',c_all)
                        np.testing.assert_array_equal(_add(raw['source_log'][j],c_all),cache['candidate'][j])
                        counts['own_weight_evaluations']+=1;counts['all_peer_query_evaluations']+=1
                        row.update(weight_mass=mass,effective_samples=effective,own_weights=w.tolist(),own_residuals=residual[own].tolist(),
                            own_numerator=own_num,correction_denominator=denominator,constant_correction=c_own,
                            all_peer_numerator=all_num,all_peer_correction=c_all,peer_months=peer_records)
                        if row['missing_same_type_months']:
                            row['status']='same_type_missing_month';counts['same_type_fallback_queries']+=1
                            raw['candidate'][j]=raw['control'][j]
                        else:
                            same_means=[]
                            for same,entry in zip(same_positions,peer_records):
                                mean,info=reference.peer_month_average(X[ids[same]],event_X[j],residual[same]);same_means.append(mean)
                                for key in ('peer_weight_mass','peer_weighted_residual','peer_mean'):entry['same_'+key]=info[key]
                                counts['same_type_month_evaluations']+=1;counts['same_type_row_observations']+=len(same)
                            with np.errstate(over='raise',invalid='raise',divide='raise'):
                                same_num=float(np.einsum('n,n->',w,np.asarray(same_means),optimize=False))
                                c_same=same_num/denominator if same_num!=0. else 0.
                            reference._finite('same-peer numerator',same_num);reference._finite('same-peer correction',c_same)
                            raw['candidate'][j]=_add(raw['source_log'][j],c_same)
                            row.update(status='same_type_supported',same_peer_numerator=same_num,same_peer_correction=c_same)
                            counts['same_type_supported_queries']+=1
                    details.append(row)
            annual.append(record)
    np.testing.assert_array_equal(rebuilt_available,available);np.testing.assert_array_equal(rebuilt_local,local_available)
    counts['actual_regression_solves']=counts['annual_pooled_log_refits']
    for key,value in raw.items():
        if not np.array_equal(np.isfinite(value),available) or np.isinf(value).any():raise FloatingPointError('Changed availability or nonfinite '+key)
    np.testing.assert_array_equal(raw['control'],cache['candidate'])
    return dict(raw=raw,available=available,local_available=local_available,audit=dict(arms=list(ARMS),raw_keys=list(RAW_KEYS),
        feature_count=10,target='unchanged cached centered log_excess',alpha=ALPHA,half_life_months=HALF_LIFE_MONTHS,
        min_months=int(min_months),min_own_months=int(min_own_months),bandwidth=BANDWIDTH,intercept_penalty=INTERCEPT_PENALTY,
        etfs=etf_ids,counts=counts,annual=annual,local=details,cached_reference_used=True,source_four_target_refits=0,
        local_linear_refits=0,correction_accounting=CORRECTION_ACCOUNTING))
