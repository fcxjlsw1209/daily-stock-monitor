"""Two paired causal annual Ridge targets and the independent original control."""
from pathlib import Path
import importlib.util
import numpy as np
import threadpoolctl
from threadpoolctl import threadpool_limits

O=Path(__file__).resolve().parents[1]
ORIGINAL_PATH=O/'monthly_target_definition70/model.py'
_SPEC=importlib.util.spec_from_file_location('beta_target70_original_model',ORIGINAL_PATH)
original=importlib.util.module_from_spec(_SPEC);_SPEC.loader.exec_module(original)
numeric=original.numeric
SOURCE_TARGETS=original.TARGETS
TARGETS=('log_beta1','log_beta252')
ALPHA=10.
HALF_LIFE_MONTHS=12.
MIN_MONTHS=18


def dependency_files():
    return (Path(__file__).resolve(),ORIGINAL_PATH,Path(numeric.__file__).resolve(),Path(threadpoolctl.__file__).resolve())


def _boolean(value,n,name):
    value=np.asarray(value)
    if value.shape!=(n,) or value.dtype.kind!='b':raise ValueError(name+' must be an exact boolean row vector')
    return value


def annual_forecasts(dates,ends,X,targets,paired_good,event_dates,event_X,event_good,*,
                     source_log_target,source_good,source_targets,min_months=MIN_MONTHS):
    """Raw source_log/log_beta1/log_beta252; paired and source availability masks.

    The original four-target model independently reproduces its original log
    forecast. Only the two new target fits use paired_good. Both use the same
    mature rows, month-balanced decay, training-only scaler and alpha. No beta
    feature or event beta gate is added. Pair cold starts stay NaN for the caller
    to fall back on its entire original monthly ranking population.
    """
    dates,ends=numeric._days(dates,'dates'),numeric._days(ends,'ends')
    event_dates=numeric._days(event_dates,'event_dates');n,q=len(dates),len(event_dates)
    if ends.shape!=(n,) or (ends<=dates).any() or np.any(ends.astype('datetime64[M]')!=dates.astype('datetime64[M]')+1):
        raise ValueError('Labels must end after signals in next calendar month')
    if isinstance(min_months,(bool,np.bool_)) or not isinstance(min_months,(int,np.integer)) or min_months<1:
        raise ValueError('min_months must be a positive integer')
    X,event_X=np.asarray(X,float),np.asarray(event_X,float)
    if X.shape!=(n,10) or event_X.shape!=(q,10):raise ValueError('Original ten-feature matrices required')
    source_good=_boolean(source_good,n,'source_good');paired_good=_boolean(paired_good,n,'paired_good')
    event_good=_boolean(event_good,q,'event_good')
    if (paired_good&~source_good).any():raise ValueError('Paired cohort cannot expand source cohort')
    if set(source_targets)!=set(SOURCE_TARGETS):raise ValueError('All four original targets required')
    sources={key:np.asarray(source_targets[key],float) for key in SOURCE_TARGETS}
    if any(y.shape!=(n,) for y in sources.values()):raise ValueError('Original target lengths differ')
    source_log_target=np.asarray(source_log_target,float)
    if source_log_target.shape!=(n,) or not np.array_equal(source_log_target,sources['log_excess'],equal_nan=True):
        raise ValueError('Source log target must equal unchanged cached log_excess')
    if set(targets)!=set(TARGETS):raise ValueError('Exactly two paired targets required')
    ys={key:np.asarray(targets[key],float) for key in TARGETS}
    if any(y.shape!=(n,) for y in ys.values()):raise ValueError('Paired target lengths differ')
    source_usable=source_good & np.isfinite(X).all(axis=1)
    for y in sources.values():source_usable &= np.isfinite(y)
    months=dates.astype('datetime64[M]')
    for month in np.unique(months):
        use=months==month
        if (source_usable&use).sum()<2:source_usable[use]=False
    usable=paired_good & source_usable
    for month in np.unique(months):
        use=months==month
        if (usable&use).sum()<2:usable[use]=False
    events=event_good & np.isfinite(event_X).all(axis=1)
    with threadpool_limits(limits=1):
        ref=original.annual_forecasts(dates,ends,X,sources,source_good,event_dates,event_X,event_good,min_months=min_months)
        raw=dict(source_log=ref['log_excess'].copy(),**{key:np.full(q,np.nan) for key in TARGETS})
        available=np.zeros(q,bool);years=event_dates.astype('datetime64[Y]').astype(int)+1970;annual=[]
        for year in np.unique(years):
            cutoff=np.datetime64('%04d-12-31' % (year-1),'D');start=numeric._years_before(cutoff)
            train=usable & (dates>start) & (ends<=cutoff);predict=(years==year)&events
            count=numeric._month_count(dates[train])
            source_train=source_usable & (dates>start) & (ends<=cutoff)
            record=dict(year=int(year),cutoff=str(cutoff),lookback_exclusive_start=str(start),
                training_rows=int(train.sum()),training_months=count,source_training_rows=int(source_train.sum()),
                source_training_months=numeric._month_count(dates[source_train]),prediction_rows=int(predict.sum()),
                removed_source_rows=int(source_train.sum()-train.sum()),
                first_training_signal=str(dates[train].min()) if train.any() else None,
                latest_training_signal=str(dates[train].max()) if train.any() else None,
                latest_label_end=str(ends[train].max()) if train.any() else None,
                status='fit' if count>=min_months else 'cold_start')
            if count>=min_months and predict.any():
                # Validate only actual mature fit rows; future unused rows cannot
                # abort or alter an earlier model. No bad used row is discarded.
                if any(not np.isfinite(y[train]).all() for y in ys.values()):
                    raise ValueError('Nonfinite paired target in actual mature training cohort')
                weights=numeric.month_balanced_weights(dates[train],cutoff,HALF_LIFE_MONTHS)
                for key in TARGETS:
                    fit=numeric._fit_ridge(X[train],ys[key][train],weights,ALPHA)
                    raw[key][predict]=numeric._predict(fit,event_X[predict])
                available[predict]=True;record['weight_sum']=float(weights.sum())
            annual.append(record)
    if (available&~ref['available']).any():raise AssertionError('Paired forecasts expanded original availability')
    return dict(raw=raw,available=available,source_available=ref['available'].copy(),audit=dict(
        targets=list(TARGETS),source_control='independent original four-target annual model; log output only',
        alpha=ALPHA,half_life_months=HALF_LIFE_MONTHS,min_months=int(min_months),lookback_years=5,
        feature_count=10,beta_feature_added=False,event_beta_required=False,labels_recentered_in_model=False,
        paired_rows_weights_and_availability_shared=True,standardization='actual training rows with same fit weights',
        weighting='equal_month_total_then_age_decay_normalized_to_row_count',source=ref['audit'],annual=annual))
