"""Pure round135 mapping of frozen round134 states; no I/O or fitting.

This reinterprets each fitted state's conditional volatility on an absolute
scale. It does not estimate realized volatility, re-fit an HMM, or classify a
state by its relative low/high label alone. Artifact authentication is the
caller's responsibility. Invalid available inputs raise; source-unavailable
signals remain NaN (and null in the JSON-safe audit).
"""
import copy
from datetime import date
import math
import numbers

import numpy as np

ANNUALIZATION = 252
QUIET_VOLATILITY = 0.10
PROBABILITY_SUM_TOLERANCE = 1e-12


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _number(value, name):
    _require(isinstance(value, numbers.Real) and not isinstance(value, (bool, np.bool_)),
             name + ' must be a real number')
    value = float(value)
    _require(math.isfinite(value), name + ' must be finite')
    return value


def _vector(value, name, size):
    result = np.asarray(value)
    _require(result.shape == (size,) and result.dtype.kind in 'iuf',
             name + ' has invalid shape or numeric type')
    result = result.astype(np.float64, copy=False)
    _require(np.isfinite(result).all(), name + ' must be finite')
    return result


def _probability_vector(value, name):
    result = _vector(value, name, 2)
    _require((result >= 0).all() and (result <= 1).all()
             and math.isclose(math.fsum(result), 1.0, rel_tol=0.,
                              abs_tol=PROBABILITY_SUM_TOLERANCE),
             name + ' must be normalized probabilities')
    return result


def classify_model(model):
    """Validate an available frozen model and classify both ordered states.

    Multiplication order is deliberately sd * sqrt(variance) * sqrt(252).
    Equality to the 10% threshold is nonquiet; there is no epsilon band.
    """
    _require(isinstance(model, dict), 'model must be an object')
    _require(model.get('available') is True and model.get('status') == 'available'
             and model.get('reason') is None and model.get('fit_attempted') is True,
             'classification requires an available fitted source model')
    cert = model.get('certificate', {})
    _require(isinstance(cert, dict) and all(cert.get(k) is True for k in
             ('converged', 'all_arithmetic_finite', 'low_variance_state_first')),
             'source model certification is incomplete')
    scaler = model.get('scaler'); parameters = model.get('parameters')
    _require(isinstance(scaler, dict) and isinstance(parameters, dict),
             'source model scaler and parameters are required')
    _number(scaler.get('mean'), 'scaler mean')
    sd = _number(scaler.get('sd'), 'scaler sd')
    _require(sd > 0, 'scaler sd must be positive')
    _vector(parameters.get('means'), 'means', 2)
    variances = _vector(parameters.get('variances'), 'variances', 2)
    _require(0 < variances[0] < variances[1], 'variances must be positive and low-first')
    _probability_vector(parameters.get('initial'), 'initial')
    transition = np.asarray(parameters.get('transition'))
    _require(transition.shape == (2, 2), 'transition must be 2 by 2')
    for row in transition:
        _probability_vector(row, 'transition row')
    _probability_vector(model.get('filtered_posterior'), 'training final posterior')
    sigma = [sd * math.sqrt(float(v)) * math.sqrt(ANNUALIZATION) for v in variances]
    if not all(math.isfinite(v) and v > 0 for v in sigma):
        raise FloatingPointError('annualized volatility overflow or underflow')
    quiet = [v < QUIET_VOLATILITY for v in sigma]
    _require(not quiet[1] or quiet[0], 'absolute state classification violated ordering')
    return dict(annualized_state_volatility=sigma, quiet_state_mask=quiet)


def _input_probabilities(value, name, size):
    result = np.asarray(value)
    _require(result.shape == (size,) and result.dtype.kind in 'iuf',
             name + ' must be a one-dimensional numeric array aligned to signals')
    result = result.astype(np.float64, copy=False)
    _require(not np.isinf(result).any(), name + ' must not contain infinity')
    return result


def _mapped(probability, mask):
    # These exact branches avoid numerical residue and preserve stored p bits.
    if mask == [True, True]:
        return 1.0
    if mask == [False, False]:
        return 0.0
    return probability


def transform(audit, filtered, memoryless):
    """Map frozen STATE_AUDIT and its two aligned low-state probability arrays.

    Returns {'filtered': float64[N], 'memoryless': float64[N], 'audit': dict}.
    Copies the complete source audit, augments each year with volatility/mask,
    and each signal with its original probabilities. Source availability,
    reasons, indices, dates, fitted models, and all other evidence stay intact.
    No caller-owned input is modified. No new model is fit.
    """
    _require(isinstance(audit, dict), 'source audit must be an object')
    years = audit.get('years'); signals = audit.get('signals')
    _require(isinstance(years, list) and isinstance(signals, list),
             'source years and signals must be lists')
    count = audit.get('new_model_fits')
    _require(type(count) is int and count >= 0, 'source fit count must be a nonnegative integer')
    _require('reused_model_fits' not in audit, 'input must be the original round134 audit')
    f = _input_probabilities(filtered, 'filtered', len(signals))
    m = _input_probabilities(memoryless, 'memoryless', len(signals))
    result = copy.deepcopy(audit)
    result['new_model_fits'] = 0
    result['reused_model_fits'] = count
    year_by_id = {}; classification = {}; last_year = None
    for row, out in zip(years, result['years']):
        _require(isinstance(row, dict), 'source year must be an object')
        year = row.get('year')
        _require(type(year) is int and 1 <= year <= 9999
                 and (last_year is None or year > last_year),
                 'source years must be unique and sorted')
        _require(type(row.get('fit_attempted')) is bool, 'source fit_attempted must be boolean')
        _require(row.get('status') in ('available', 'unavailable'), 'invalid source year status')
        if row['status'] == 'available':
            _require(row['fit_attempted'] and row.get('reason') is None,
                     'available source year has inconsistent metadata')
            info = classify_model(row.get('model'))
        else:
            _require(isinstance(row.get('reason'), str) and bool(row['reason']),
                     'unavailable source year must preserve a reason')
            model = row.get('model')
            _require(model is None or (isinstance(model, dict)
                     and model.get('available') is False
                     and model.get('status') == 'unavailable'
                     and model.get('fit_attempted') is row['fit_attempted']),
                     'unavailable source year has inconsistent model')
            _require(model is not None or not row['fit_attempted'],
                     'attempted source fit requires its model')
            info = dict(annualized_state_volatility=None, quiet_state_mask=None)
        out.update(info); year_by_id[year] = row; classification[year] = info
        last_year = year
    _require(count == sum(row['fit_attempted'] for row in years), 'source fit count mismatch')
    out_f = np.full(len(signals), np.nan); out_m = np.full(len(signals), np.nan)
    last_date = None
    for i, (row, out) in enumerate(zip(signals, result['signals'])):
        _require(isinstance(row, dict), 'source signal must be an object')
        _require(type(row.get('signal_index')) is int and row['signal_index'] == i,
                 'source signal index mismatch')
        text = row.get('signal_date')
        _require(isinstance(text, str), 'signal date must be an ISO date')
        day = date.fromisoformat(text)
        _require(str(day) == text and (last_date is None or day > last_date),
                 'signal dates must be canonical, sorted and unique')
        year = row.get('year')
        _require(type(year) is int and year == day.year and year in year_by_id,
                 'signal year is not bound to its source model')
        _require(type(row.get('available')) is bool, 'source availability must be boolean')
        original = []
        for key, values in (('filtered_probability', f), ('memoryless_probability', m)):
            value = row.get(key)
            if row['available']:
                value = _number(value, key)
                _require(0 <= value <= 1 and value == values[i],
                         key + ' is out of range or differs from its frozen array')
            else:
                _require(key in row and value is None and math.isnan(float(values[i])),
                         'unavailable probability must be source null and array NaN')
            original.append(value)
        out['original_filtered_probability'], out['original_memoryless_probability'] = original
        if row['available']:
            _require(year_by_id[year]['status'] == 'available' and row.get('reason') is None,
                     'available signal has unavailable model or nonnull reason')
            mask = classification[year]['quiet_state_mask']
            out_f[i] = _mapped(original[0], mask); out_m[i] = _mapped(original[1], mask)
            out['filtered_probability'] = float(out_f[i]); out['memoryless_probability'] = float(out_m[i])
        else:
            _require(isinstance(row.get('reason'), str) and bool(row['reason']),
                     'unavailable signal must preserve a reason')
        last_date = day
    return dict(filtered=out_f, memoryless=out_m, audit=result)
