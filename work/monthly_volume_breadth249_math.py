"""Exact 12-by-21 stored-volume covariance aggregation, without source I/O.

Callers own calendar/identity alignment, listing/transition gates and common
49-member availability. This module neither reads sources nor integrates scores.
Accepted sources are first represented as positive finite binary64 numbers.
Every derived return, covariance and raw aggregate stays rational; only the
bounded final rank target is converted to binary64.
"""
from collections.abc import Mapping
from fractions import Fraction
import math
from numbers import Real
from pandas import NA

SCHEMA = 'monthly_volume_breadth249_math_v1'
RANK_SCHEMA = 'monthly_volume_breadth249_rank49_v1'
MEASURE_KEYS = ('schema','available','reason','price_count','volume_count',
 'return_count','block_count','missing_price_indices','missing_volume_indices',
 'high_flags','blocks','raw_breadth','raw_mean','positive_block_count',
 'zero_block_count','negative_block_count')
BLOCK_KEYS = ('block_index','first_return_index','last_return_index','high_count',
 'return_sum','high_return_sum','covariance','sign')
RANK_KEYS = ('schema','width','less','equal','targets')
PRICE_COUNT, VOLUME_COUNT, RETURN_COUNT, BLOCK_COUNT, WIDTH = 253, 315, 252, 12, 49


def _sequence(values, length, label):
    if isinstance(values, (str, bytes, bytearray, Mapping, set, frozenset)):
        raise ValueError(label + ' must be a one-dimensional ordered sequence')
    try:
        if len(values) != length:
            raise ValueError(label + ' must contain exactly ' + str(length) + ' values')
        ndim = getattr(values, 'ndim', 1)
        if ndim != 1:
            raise ValueError(label + ' must be one-dimensional')
        result = list(values)
    except TypeError as exc:
        raise ValueError(label + ' must be a sized sequence') from exc
    if len(result) != length:
        raise ValueError(label + ' length changed while reading')
    return result


def _scan(values, label):
    numbers, missing, errors = [], [], []
    for i, value in enumerate(values):
        if value is None or value is NA:
            numbers.append(None); missing.append(i); continue
        if isinstance(value, bool) or not isinstance(value, Real):
            numbers.append(None); errors.append(label + '[' + str(i) + ']: nonreal_or_boolean'); continue
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            numbers.append(None); errors.append(label + '[' + str(i) + ']: binary64_conversion'); continue
        if math.isnan(number):
            numbers.append(None); missing.append(i); continue
        if not math.isfinite(number):
            numbers.append(None); errors.append(label + '[' + str(i) + ']: nonfinite'); continue
        if number <= 0.0:
            numbers.append(None); errors.append(label + '[' + str(i) + ']: nonpositive'); continue
        numbers.append(number)
    return numbers, missing, errors


def _pair(value):
    return {'numerator': str(value.numerator), 'denominator': str(value.denominator)}


def _rational(value):
    if isinstance(value, Fraction):
        return value
    if not isinstance(value, Mapping) or set(value) != {'numerator', 'denominator'}:
        raise ValueError('Rank raw values must be Fraction or canonical rational pairs')
    strings = (value['numerator'], value['denominator'])
    if any(type(s) is not str for s in strings):
        raise ValueError('Rational numerator and denominator must be decimal strings')
    try:
        numerator, denominator = (int(s) for s in strings)
    except ValueError as exc:
        raise ValueError('Invalid rational integer string') from exc
    if str(numerator) != strings[0] or str(denominator) != strings[1] or denominator <= 0:
        raise ValueError('Rational strings must be canonical with positive denominator')
    result = Fraction(numerator, denominator)
    if result.numerator != numerator or result.denominator != denominator:
        raise ValueError('Rational pair must be reduced')
    return result


def measure(prices253, volumes315):
    """Measure one member; ordinary gaps return unavailable, invalid values fail.

    None, pandas.NA and real numeric NaN are ordinary missing. Both entire source
    windows are scanned before this missing branch, so a gap cannot conceal an
    invalid number elsewhere. Booleans, non-real values, failed binary64 casts,
    infinity and nonpositive observations are hard ValueError violations. Zero
    volume is deliberately excluded by this contract, not diagnosed as vendor
    error. Genuine zero covariance remains available.
    """
    price_input = _sequence(prices253, PRICE_COUNT, 'prices')
    volume_input = _sequence(volumes315, VOLUME_COUNT, 'volumes')
    prices, price_missing, price_errors = _scan(price_input, 'prices')
    volumes, volume_missing, volume_errors = _scan(volume_input, 'volumes')
    if price_errors or volume_errors:
        raise ValueError('; '.join(price_errors + volume_errors))
    result = dict(schema=SCHEMA, available=False, reason='missing_source',
        price_count=PRICE_COUNT, volume_count=VOLUME_COUNT,
        return_count=RETURN_COUNT, block_count=BLOCK_COUNT,
        missing_price_indices=price_missing, missing_volume_indices=volume_missing,
        high_flags=[], blocks=[], raw_breadth=None, raw_mean=None,
        positive_block_count=None, zero_block_count=None, negative_block_count=None)
    if price_missing or volume_missing:
        return result
    p = [Fraction.from_float(value) for value in prices]
    returns = [p[q + 1] / p[q] - 1 for q in range(RETURN_COUNT)]
    high = [int(volumes[q + 63] > sorted(volumes[q:q + 63])[31])
            for q in range(RETURN_COUNT)]
    covariances = []
    blocks = []
    for k in range(BLOCK_COUNT):
        lo, hi = 21 * k, 21 * (k + 1)
        high_count = sum(high[lo:hi])
        return_sum = sum(returns[lo:hi], Fraction(0))
        high_return_sum = sum((returns[q] for q in range(lo, hi) if high[q]), Fraction(0))
        covariance = (21 * high_return_sum - high_count * return_sum) / 441
        sign = int(covariance > 0) - int(covariance < 0)
        covariances.append(covariance)
        blocks.append(dict(block_index=k, first_return_index=lo, last_return_index=hi-1,
            high_count=high_count, return_sum=_pair(return_sum),
            high_return_sum=_pair(high_return_sum), covariance=_pair(covariance), sign=sign))
    positive = sum(x > 0 for x in covariances)
    zero = sum(x == 0 for x in covariances)
    negative = BLOCK_COUNT - positive - zero
    result.update(available=True, reason='available', high_flags=high, blocks=blocks,
        raw_breadth=_pair(Fraction(2 * positive + zero - 12, 24)),
        raw_mean=_pair(sum(covariances, Fraction(0)) / 12),
        positive_block_count=positive, zero_block_count=zero, negative_block_count=negative)
    return result


def rank49(raw_values49):
    """Exact common49 average-percentile ranking; no eligibility filtering.

    Floats and missing raw values are rejected. Exact comparisons precede the
    sole float conversion of (2*less+equal+1-49)/98, including all-tie 1/98.
    Caller must supply the same complete49-member domain for both arms.
    """
    values = [_rational(x) for x in _sequence(raw_values49, WIDTH, 'raw_values')]
    less = [sum(y < x for y in values) for x in values]
    equal = [sum(y == x for y in values) for x in values]
    targets = [float(Fraction(2 * lo + eq + 1 - WIDTH, 2 * WIDTH))
               for lo, eq in zip(less, equal)]
    return dict(schema=RANK_SCHEMA, width=WIDTH, less=less, equal=equal, targets=targets)
