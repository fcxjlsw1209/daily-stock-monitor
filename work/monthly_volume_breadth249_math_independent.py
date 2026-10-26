"""Independent exact math for the prospective stored-volume breadth contrast.

Pure in-memory source validation, Fraction statistics and exact 49-member ranks.
No strategy, producer, host, filesystem, network or market-data loader is used.
"""
from fractions import Fraction
from heapq import nsmallest
import math
import numbers
import re
import struct

import numpy as np
import pandas as pd

SCHEMA = 'monthly_volume_breadth249_math_v1'
RANK_SCHEMA = 'monthly_volume_breadth249_rank49_v1'
MEASURE_KEYS = ('schema', 'available', 'reason', 'price_count', 'volume_count',
    'return_count', 'block_count', 'missing_price_indices', 'missing_volume_indices',
    'high_flags', 'blocks', 'raw_breadth', 'raw_mean', 'positive_block_count',
    'zero_block_count', 'negative_block_count')
BLOCK_KEYS = ('block_index', 'first_return_index', 'last_return_index', 'high_count',
    'return_sum', 'high_return_sum', 'covariance', 'sign')
RANK_KEYS = ('schema', 'width', 'less', 'equal', 'targets')
MEASURE_FLAGS = ('passed', 'independent_exact_rational_rebuild',
    'strict_previous63_median_checked', 'full_source_windows_checked',
    'hard_invalid_not_hidden_by_missing', 'twelve_population_covariances_checked',
    'sign_breadth_and_mean_checked', 'canonical_rational_audit_checked',
    'complete_supplied_measure_checked')
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


def measure_oracle(prices253, volumes315):
    raw_prices = _vector(prices253, 253, 'price')
    raw_volumes = _vector(volumes315, 315, 'volume')
    prices, missing_prices, bad_prices = _scan(raw_prices, 'price')
    volumes, missing_volumes, bad_volumes = _scan(raw_volumes, 'volume')
    # Both full windows are checked before any missing-source fallback.
    if bad_prices or bad_volumes:
        raise ValueError('Invalid required observations: ' + repr(bad_prices + bad_volumes))
    result = dict(schema=SCHEMA, available=False, reason='missing_source',
        price_count=253, volume_count=315, return_count=252, block_count=12,
        missing_price_indices=missing_prices, missing_volume_indices=missing_volumes,
        high_flags=[], blocks=[], raw_breadth=None, raw_mean=None,
        positive_block_count=None, zero_block_count=None, negative_block_count=None)
    if missing_prices or missing_volumes:
        return result

    exact_prices = [Fraction.from_float(p) for p in prices]
    returns = [(right-left)/left for left, right in zip(exact_prices, exact_prices[1:])]
    flags = []
    for q in range(252):
        # Independent order statistic, excluding the contemporaneous volume.
        median = nsmallest(32, volumes[q:q+63])[-1]
        flags.append(int(volumes[q+63] > median))
    blocks, covariances = [], []
    for k in range(12):
        first, last = 21*k, 21*k+21
        segment, high = returns[first:last], flags[first:last]
        high_count = sum(high)
        mean_return = sum(segment, Fraction(0))/21
        mean_high = Fraction(high_count, 21)
        # Centered-product expression independent of the producer sumHR formula.
        cov = sum(((Fraction(h)-mean_high)*(r-mean_return)
                   for h, r in zip(high, segment)), Fraction(0))/21
        sign = int(cov > 0)-int(cov < 0)
        blocks.append(dict(block_index=k, first_return_index=first,
            last_return_index=last-1, high_count=high_count,
            return_sum=pair(sum(segment, Fraction(0))),
            high_return_sum=pair(sum((r for h, r in zip(high, segment) if h), Fraction(0))),
            covariance=pair(cov), sign=sign))
        covariances.append(cov)
    positive = sum(c > 0 for c in covariances)
    zero = sum(c == 0 for c in covariances)
    negative = sum(c < 0 for c in covariances)
    # Equal signed-block weighting gives the declared zero-half-credit statistic.
    breadth = Fraction(positive-negative, 24)
    result.update(available=True, reason='available', high_flags=flags, blocks=blocks,
        raw_breadth=pair(breadth), raw_mean=pair(sum(covariances, Fraction(0))/12),
        positive_block_count=positive, zero_block_count=zero, negative_block_count=negative)
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


def check_measure(prices253, volumes315, actual):
    expected = measure_oracle(prices253, volumes315)
    _exact(actual, expected)
    receipt = {flag: True for flag in MEASURE_FLAGS}
    receipt.update(source_available=expected['available'], checked_prices=253,
        checked_volumes=315, checked_block_covariances=12 if expected['available'] else 0,
        producer_modules_imported=False, source_io_performed=False,
        computes_strategy_returns=False, new_model_fits=0, numerical_tolerance=0)
    return receipt


def check_rank49(raw_values49, actual):
    _exact(actual, rank49_oracle(raw_values49))
    receipt = {flag: True for flag in RANK_FLAGS}
    receipt.update(width=49, producer_modules_imported=False, source_io_performed=False,
        computes_strategy_returns=False, new_model_fits=0, numerical_tolerance=0)
    return receipt
