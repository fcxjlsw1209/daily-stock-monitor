"""Unregistered 265 synthetic kernel: daily versus 12-block gain concentration.

The caller supplies exactly 253 adjusted closes in chronological order, intended
to cover t-294 through t-42, and the unchanged original raw score. This module
does not establish that clock or source admission. It has no data loading,
strategy integration, fitting, ranking, orders, replay or registration.

Positive simple gains are binary64 first, then exact rational weights. The
permutation expectation identity is about the exact qualities, before their
separate binary64 conversions; it is not a pathwise ordering of the two arms.
"""
from collections.abc import Mapping
from fractions import Fraction
import math
from numbers import Real


PRICE_COUNT = 253
GAIN_COUNT = 252
BLOCK_SIZE = 21
BLOCK_COUNT = 12
SCHEMA = 'monthly_gain_concentration265_synthetic_statistics_v1'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite(value, label):
    _require(isinstance(value, Real) and not isinstance(value, bool),
             label + ' must be real and nonboolean')
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(label + ' cannot convert to binary64') from exc
    _require(math.isfinite(converted), label + ' must be finite')
    if value != 0 and converted == 0.0:
        raise ArithmeticError(label + ' conversion underflow')
    return converted


def _vector(values, count, label):
    _require(not isinstance(values, (str, bytes, bytearray, Mapping)),
             label + ' must be an ordered numeric vector')
    try:
        size = len(values)
    except TypeError as exc:
        raise ValueError(label + ' must have an explicit length') from exc
    _require(size == count, label + ' must contain exactly ' + str(count) + ' values')
    return [_finite(values[i], label + '[' + str(i) + ']') for i in range(count)]


def exact_qualities_from_gains(gains252):
    """Pure fixed-size arithmetic helper; inputs are already simple gains.

    This helper does not imply that arbitrary supplied gains arose from an
    admissible price path. Negative and zero gains retain their chronological
    slots but receive zero weight. W == 0 has explicit Q == 1, with undefined
    concentration/normalized weights represented by None, not invented zeros.
    """
    gains = _vector(gains252, GAIN_COUNT, 'simple gains')
    positive = [g if g > 0.0 else 0.0 for g in gains]
    exact = tuple(Fraction.from_float(a) for a in positive)
    total = sum(exact, Fraction(0))
    blocks = tuple(sum(exact[start:start + BLOCK_SIZE], Fraction(0))
                   for start in range(0, GAIN_COUNT, BLOCK_SIZE))
    if total == 0:
        weights = block_weights = daily_h = block_h = None
        daily_q = block_q = Fraction(1)
    else:
        weights = tuple(a / total for a in exact)
        block_weights = tuple(a / total for a in blocks)
        daily_h = sum((a * a for a in weights), Fraction(0))
        block_h = sum((a * a for a in block_weights), Fraction(0))
        _require(Fraction(1, GAIN_COUNT) <= daily_h <= 1,
                 'Exact daily concentration bounds')
        _require(Fraction(1, BLOCK_COUNT) <= block_h <= 1,
                 'Exact block concentration bounds')
        _require(block_h >= daily_h, 'Exact aggregation concentration bound')
        daily_q = (1 - daily_h) / (1 - Fraction(1, GAIN_COUNT))
        block_q = (1 - block_h) / (1 - Fraction(1, BLOCK_COUNT))
    _require(0 <= daily_q <= 1 and 0 <= block_q <= 1,
             'Exact quality unit range before conversion')
    return dict(daily_gains=tuple(gains), positive_parts=tuple(positive),
                positive_gain_indices=tuple(i + 1 for i, a in enumerate(exact) if a > 0),
                gain_sum=total, block_gain_sums=blocks, daily_weights=weights,
                block_weights=block_weights, daily_concentration=daily_h,
                block_concentration=block_h, daily_quality=daily_q,
                block_quality=block_q, no_positive_gains=(total == 0))


def _quality_float(exact_quality):
    _require(isinstance(exact_quality, Fraction) and 0 <= exact_quality <= 1,
             'Exact quality range required before binary64 conversion')
    value = float(exact_quality)  # Exactly one conversion of this exact Q.
    if exact_quality != 0 and value == 0.0:
        raise ArithmeticError('Positive exact quality conversion underflow')
    _require(math.isfinite(value) and 0.0 <= value <= 1.0,
             'Converted quality unit range')
    return value


def _transform(raw, quality):
    if raw <= 0.0:
        return raw  # Includes negative zero; do not multiply it by quality.
    value = raw * quality
    if not math.isfinite(value):
        raise ArithmeticError('Transformed raw overflow')
    if quality != 0.0 and value == 0.0:
        raise ArithmeticError('Positive transformed raw underflow')
    return value  # Q == 0 legitimately produces +0.


def _rational_record(value):
    return None if value is None else dict(numerator=value.numerator,
                                          denominator=value.denominator)


def measure(prices253, original_raw):
    """Measure both qualities and transform a supplied original score only."""
    prices = _vector(prices253, PRICE_COUNT, 'required closes')
    _require(all(value > 0.0 for value in prices), 'Required closes must be positive')
    raw = _finite(original_raw, 'original raw')
    gains = []
    for previous, current in zip(prices, prices[1:]):
        gross = current / previous
        if not math.isfinite(gross):
            raise ArithmeticError('Daily gross return overflow')
        if gross == 0.0:
            raise ArithmeticError('Daily gross return underflow')
        gain = gross - 1.0
        if not math.isfinite(gain):
            raise ArithmeticError('Daily simple return overflow')
        gains.append(gain)
    evidence = exact_qualities_from_gains(gains)
    daily_q = _quality_float(evidence['daily_quality'])
    block_q = _quality_float(evidence['block_quality'])
    daily_raw = _transform(raw, daily_q)
    block_raw = _transform(raw, block_q)
    rational_fields = ('gain_sum', 'daily_concentration', 'block_concentration',
                       'daily_quality', 'block_quality')
    return dict(schema=SCHEMA, price_count=PRICE_COUNT, gain_count=GAIN_COUNT,
                block_size=BLOCK_SIZE, block_count=BLOCK_COUNT,
                daily_gains=list(evidence['daily_gains']),
                positive_parts=list(evidence['positive_parts']),
                positive_gain_indices=list(evidence['positive_gain_indices']),
                exact={name: _rational_record(evidence[name]) for name in rational_fields},
                exact_block_gain_sums=[_rational_record(a) for a in evidence['block_gain_sums']],
                daily_quality=daily_q, block_quality=block_q,
                original_raw=raw, daily_raw=daily_raw, block_raw=block_raw,
                nonpositive_original_preserved=(raw <= 0.0),
                no_positive_gains=evidence['no_positive_gains'])
