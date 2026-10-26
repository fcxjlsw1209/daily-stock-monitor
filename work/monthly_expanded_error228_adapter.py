"""Pure source227-to-frozen226 risk-history adapter; no risk estimation or fit.

The returned `train` dictionary is a compatibility view for226's historical
error join. It MUST NEVER be passed into a Ridge or other source-model fit.
Current pre-scores/eligibility, trading rules and original inputs are untouched.
"""
from pathlib import Path
import numpy as np
import monthly_joint_error226_model as frozen

TARGETS = frozen.TARGETS
RAW_KEYS = ('raw_source_log','raw_local_constant','raw_local_linear')
OLD_SOURCE_KEYS = set(RAW_KEYS) | {'raw_candidate','raw_control','available','local_available'}
EXPANDED_SOURCE_KEYS = set(RAW_KEYS) | {'available','local_available','expansion_allowed'}
LABEL_KEYS = {'target_'+k for k in TARGETS} | {'common_good','added','vol63','dates','ends','offsets'}
COUNT_FIELDS = ('training_rows','query_rows','original_forecast_rows','expanded_forecast_rows',
    'added_forecast_rows','original_label_rows','added_label_rows','extended_label_rows',
    'original_history_rows','expanded_history_rows')


def dependency_files():
    return (Path(__file__).resolve(),Path(frozen.__file__).resolve())


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _exact(left, right, name):
    a,b=np.asarray(left),np.asarray(right)
    _require(a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),
             'Exact cached bytes differ: '+name)


def _bool(value, shape, name):
    value=np.asarray(value)
    _require(value.shape==shape and value.dtype.kind=='b','Boolean axis differs: '+name)
    return value


def _numeric(value, shape, name):
    value=np.asarray(value)
    _require(value.shape==shape and value.dtype.kind in 'fiu','Numeric nonboolean axis differs: '+name)
    return value


def adapt(original_train,training_meta,query_meta,original_source206,expanded_source227,
          expanded_labels227,original_use,*,source_meta227):
    """Validate immutable-cache compatibility and expand only historical support.

    All outputs own their arrays. Source clocks, shared-H admission, statistics
    and current boundary selection remain the frozen226 caller's responsibility.
    """
    _require(isinstance(original_train,dict) and isinstance(original_source206,dict)
             and isinstance(expanded_source227,dict) and isinstance(expanded_labels227,dict),
             'Explicit source dictionaries required')
    use=np.asarray(original_use)
    _require(use.ndim==2 and use.shape[1]>0 and use.dtype.kind=='b','Original U must be boolean NxW')
    n,width=use.shape;q=n*width
    _require(isinstance(source_meta227,dict) and set(source_meta227)>={'monthly','training'},
             'Source227 metadata axes missing')
    _require(source_meta227['monthly']==query_meta and source_meta227['training']==training_meta,
             'Source227 exact query/training metadata differs')
    queries=frozen._metadata(query_meta,width,initial_query_exception=True)
    history=frozen._metadata(training_meta)
    _require(len(queries)==n,'Original U/query count mismatch')
    offsets=[0]
    for _,_,symbols in history:offsets.append(offsets[-1]+len(symbols))
    total=offsets[-1]
    required={'dates','ends','offsets','train_plain','train_eligible','common_good'} | {'target_'+k for k in TARGETS}
    _require(required<=set(original_train),'Original risk-history fields missing')
    old_offsets=np.asarray(original_train['offsets'])
    _require(old_offsets.dtype.kind in 'iu' and np.array_equal(old_offsets,np.asarray(offsets)),
             'Original training offsets differ')
    for key in ('dates','ends'):
        v=np.asarray(original_train[key]);_require(v.shape==(total,),'Original date axis differs')
        expected=[entry[0 if key=='dates' else 1] for entry in history for _ in entry[2]]
        _require([str(x) for x in v]==expected,'Original exact training date identity differs')
    _require(set(expanded_labels227)==LABEL_KEYS,'Exact source227 LABELS schema required')
    for key in ('dates','ends','offsets'):_exact(original_train[key],expanded_labels227[key],key)
    X=_numeric(original_train['train_plain'],(total,10),'train_plain')
    old_good=_bool(original_train['common_good'],(total,),'old common_good')
    train_eligible=_bool(original_train['train_eligible'],(total,),'train_eligible')
    good=_bool(expanded_labels227['common_good'],(total,),'expanded common_good')
    added=_bool(expanded_labels227['added'],(total,),'added labels')
    _numeric(expanded_labels227['vol63'],(total,),'expanded vol63')
    _require(not np.any(added & train_eligible),'Added labels must be previously ineligible')
    _require(not np.any(added & old_good) and np.array_equal(good,old_good|added),
             'Original common-good set changed or added mask inconsistent')
    _require(np.isfinite(X[good]).all(),'Admitted labels require finite original plain10')
    for key in TARGETS:
        name='target_'+key
        old=_numeric(original_train[name],(total,),name)
        new=_numeric(expanded_labels227[name],(total,),'expanded '+name)
        _exact(old[~added],new[~added],name)
        _require(np.array_equal(np.isfinite(old),old_good),'Original label finite support differs')
        _require(np.array_equal(np.isfinite(new),good),'Expanded label finite support differs')
    _require(set(original_source206)==OLD_SOURCE_KEYS,'Exact source206 forecast schema required')
    _require(set(expanded_source227)==EXPANDED_SOURCE_KEYS,'Exact source227 forecast schema required')
    old_available=_bool(original_source206['available'],(q,),'old available')
    available=_bool(expanded_source227['available'],(q,),'expanded available')
    old_local=_bool(original_source206['local_available'],(q,),'old local_available')
    local=_bool(expanded_source227['local_available'],(q,),'expanded local_available')
    allowed=_bool(expanded_source227['expansion_allowed'],(q,),'expansion_allowed')
    _require(not np.any(old_available & ~available),'Original forecast availability lost')
    new_available=available & ~old_available
    _require(not np.any(new_available & ~allowed),'New forecast outside admitted295-price domain')
    _require(not np.any(old_local & ~old_available) and not np.any(local & ~available),
             'Local availability exceeds source availability')
    _exact(old_local[~new_available],local[~new_available],'local availability')
    _require(not np.any(use.reshape(-1)&~old_available),'Original U exceeds original source availability')
    for key in OLD_SOURCE_KEYS:
        _require(np.asarray(original_source206[key]).shape==(q,),'Original source axis differs: '+key)
    for key in RAW_KEYS:
        old=_numeric(original_source206[key],(q,),key)
        new=_numeric(expanded_source227[key],(q,),'expanded '+key)
        _require(np.array_equal(np.isfinite(old),old_available) and not np.isinf(old).any(),
                 'Original forecast finite support differs: '+key)
        _require(np.array_equal(np.isfinite(new),available) and not np.isinf(new).any(),
                 'Expanded forecast finite support differs: '+key)
        _exact(old[~new_available],new[~new_available],key)
    train={key:np.asarray(value).copy() for key,value in original_train.items()}
    for key in ('common_good',)+tuple('target_'+k for k in TARGETS):
        train[key]=np.asarray(expanded_labels227[key]).copy()
    source={'raw_local_linear':np.asarray(expanded_source227['raw_local_linear']).copy(),
            'available':available.copy()}
    historical_use=use.copy() | new_available.reshape(n,width)
    counts=dict(training_rows=total,query_rows=q,original_forecast_rows=int(old_available.sum()),
        expanded_forecast_rows=int(available.sum()),added_forecast_rows=int(new_available.sum()),
        original_label_rows=int(old_good.sum()),added_label_rows=int(added.sum()),
        extended_label_rows=int(good.sum()),original_history_rows=int(use.sum()),
        expanded_history_rows=int(historical_use.sum()))
    audit=dict(method='expanded_same_query_error_history_adapter',
        train_output_role='risk_history_only_never_Ridge_training',
        historical_use_rule='original_use OR newly_available227',original_identity_exact=True,
        old_forecasts_exact=True,old_labels_exact=True,original_common_good_preserved=True,
        added_labels_only_ineligible=True,historical_use_or_expansion=True,outputs_detached=True,
        source_forecasts_refitted=False,labels_recomputed=False,numerical_risk_estimates_computed=False,
        features_recomputed=False,portfolio_returns_computed=False,counts=counts)
    return dict(train=train,source=source,historical_use=historical_use,audit=audit)
