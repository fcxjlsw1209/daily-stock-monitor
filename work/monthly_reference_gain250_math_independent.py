"""Independent exact math for the prospective lagged reference-gain contrast.

Pure in-memory source validation, Fraction statistics and exact 49-member ranks.
No strategy, producer, host, filesystem, network or market-data loader is used.
"""
from fractions import Fraction
import math
import numbers
import re
import struct

import numpy as np
import pandas as pd

SCHEMA = 'monthly_reference_gain250_math_v1'
RANK_SCHEMA = 'monthly_reference_gain250_rank49_v1'
MEASURE_KEYS = ('schema','available','reason','price_count','volume_count','historical_count',
    'missing_price_indices','missing_volume_indices','reference_price','historical_price_sum',
    'volume_sum','weighted_price_sum','weighted_reference','equal_reference','raw_volume_gain','raw_equal_gain')
RANK_KEYS = ('schema','width','less','equal','targets')
MEASURE_FLAGS = ('passed','independent_exact_rational_rebuild','historical_reference_window_checked',
    'full_source_windows_checked','hard_invalid_not_hidden_by_missing',
    'exact_weighted_equal_references_checked','reference_day_volume_excluded',
    'canonical_rational_audit_checked','complete_supplied_measure_checked')
RANK_FLAGS = ('passed', 'independent_exact_rational_rank',
    'exact49_domain_checked', 'raw_float_ranking_rejected',
    'average_percentile_ties_checked', 'bounded_float_targets_exact',
    'complete_supplied_rank_checked')


def _vector(values, size, name):
    try:
        array = np.asarray(values, dtype=object)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(name + ' must be a fixed one-dimensional window') from exc
    if array.shape != (size,):
        raise ValueError(name + ' window length/shape')
    return list(array)


def _scan(values, name):
    converted, missing, invalid = [], [], []
    for index, value in enumerate(values):
        if value is None or value is pd.NA:
            converted.append(None)
            missing.append(index)
            continue
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
            converted.append(None)
            invalid.append((name, index, 'nonreal_or_boolean'))
            continue
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            converted.append(None)
            invalid.append((name, index, 'conversion'))
            continue
        if math.isnan(number):
            converted.append(None)
            missing.append(index)
        elif not math.isfinite(number) or number <= 0.0:
            converted.append(None)
            invalid.append((name, index, 'nonfinite_or_nonpositive'))
        else:
            converted.append(number)
    return converted, missing, invalid


def pair(value):
    if not isinstance(value, Fraction):
        raise ValueError('Exact Fraction required')
    return {'numerator': str(value.numerator), 'denominator': str(value.denominator)}


def rational(value):
    if isinstance(value, Fraction):
        return value
    if type(value) is not dict or set(value) != {'numerator', 'denominator'}:
        raise ValueError('Only Fraction or a canonical rational pair may be ranked')
    numerator, denominator = value['numerator'], value['denominator']
    if (type(numerator) is not str or type(denominator) is not str
            or re.fullmatch(r'-?(0|[1-9][0-9]*)', numerator) is None
            or re.fullmatch(r'[1-9][0-9]*', denominator) is None
            or numerator == '-0'):
        raise ValueError('Canonical signed numerator and positive denominator strings required')
    try:
        result = Fraction(int(numerator), int(denominator))
    except (ValueError, OverflowError) as exc:
        raise ValueError('Invalid rational pair') from exc
    if pair(result) != value:
        raise ValueError('Rational pair must be reduced')
    return result


def measure_oracle(prices253, volumes252):
    raw_prices = _vector(prices253, 253, 'price')
    raw_volumes = _vector(volumes252, 252, 'volume')
    prices, missing_prices, bad_prices = _scan(raw_prices, 'price')
    volumes, missing_volumes, bad_volumes = _scan(raw_volumes, 'volume')
    if bad_prices or bad_volumes:
        raise ValueError('Invalid required observations: ' + repr(bad_prices + bad_volumes))
    result = dict(schema=SCHEMA, available=False, reason='missing_source',
        price_count=253, volume_count=252, historical_count=252,
        missing_price_indices=missing_prices, missing_volume_indices=missing_volumes,
        reference_price=None, historical_price_sum=None, volume_sum=None, weighted_price_sum=None,
        weighted_reference=None, equal_reference=None, raw_volume_gain=None, raw_equal_gain=None)
    if missing_prices or missing_volumes:
        return result
    price = [Fraction.from_float(p) for p in prices]
    weights = [Fraction.from_float(v) for v in volumes]
    reference = price[-1]
    historical = price[:-1]
    sum_p = sum(historical, Fraction())
    sum_v = sum(weights, Fraction())
    sum_vp = sum((w*p for w,p in zip(weights,historical)), Fraction())
    # Independent per-observation relative-gain formulation. The producer's
    # historical-reference subtraction is not used to construct either raw gain.
    gains = [(reference-p)/reference for p in historical]
    weighted_gain = sum((w*g for w,g in zip(weights,gains)), Fraction())/sum_v
    equal_gain = sum(gains, Fraction())/252
    result.update(available=True, reason='available', reference_price=pair(reference),
        historical_price_sum=pair(sum_p), volume_sum=pair(sum_v), weighted_price_sum=pair(sum_vp),
        weighted_reference=pair(sum_vp/sum_v), equal_reference=pair(sum_p/252),
        raw_volume_gain=pair(weighted_gain), raw_equal_gain=pair(equal_gain))
    return result


def rank49_oracle(raw_values49):
    values = [rational(v) for v in _vector(raw_values49, 49, 'rank')]
    less, equal, targets = [], [], []
    for value in values:
        below = sum(other < value for other in values)
        above = sum(other > value for other in values)
        tied = 49-below-above
        less.append(below)
        equal.append(tied)
        # Independent below/above form; +1 is the ordinary percentile-rank offset.
        target = Fraction(below-above+1, 98)
        targets.append(float(target))
    return dict(schema=RANK_SCHEMA, width=49, less=less, equal=equal, targets=targets)


def _exact(actual, expected, location='root'):
    if type(actual) is not type(expected):
        raise ValueError('Output type mismatch at ' + location)
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError('Output fields mismatch at ' + location)
        for key in expected:
            _exact(actual[key], expected[key], location + '.' + key)
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError('Output length mismatch at ' + location)
        for index, (a, e) in enumerate(zip(actual, expected)):
            _exact(a, e, location + '[' + str(index) + ']')
    elif isinstance(expected, float):
        if struct.pack('!d', actual) != struct.pack('!d', expected):
            raise ValueError('Output binary64 mismatch at ' + location)
    elif actual != expected:
        raise ValueError('Output value mismatch at ' + location)


def check_measure(prices253, volumes252, actual):
    expected = measure_oracle(prices253, volumes252)
    _exact(actual, expected)
    receipt = {flag: True for flag in MEASURE_FLAGS}
    receipt.update(source_available=expected['available'], checked_prices=253,
        checked_volumes=252, checked_reference_estimates=2 if expected['available'] else 0,
        producer_modules_imported=False, source_io_performed=False,
        computes_strategy_returns=False, new_model_fits=0, numerical_tolerance=0)
    return receipt


def check_rank49(raw_values49, actual):
    _exact(actual, rank49_oracle(raw_values49))
    receipt = {flag: True for flag in RANK_FLAGS}
    receipt.update(width=49, producer_modules_imported=False, source_io_performed=False,
        computes_strategy_returns=False, new_model_fits=0, numerical_tolerance=0)
    return receipt
