"""Pure exact historical reference gains; no files, prices loaders or scoring.

Required inputs:253 positive binary64 adjusted prices and252 historical stored
volumes. The last price is reference only. Reference-day volume is not an input.
Kernel owns calendar, listing, zero-volume and whole49 availability gates.
"""
from collections.abc import Mapping
from fractions import Fraction
import math
from numbers import Real
from pandas import NA

SCHEMA='monthly_reference_gain250_math_v1'
RANK_SCHEMA='monthly_reference_gain250_rank49_v1'
MEASURE_KEYS=('schema','available','reason','price_count','volume_count','historical_count',
 'missing_price_indices','missing_volume_indices','reference_price','historical_price_sum',
 'volume_sum','weighted_price_sum','weighted_reference','equal_reference','raw_volume_gain','raw_equal_gain')
RANK_KEYS=('schema','width','less','equal','targets')
PRICE_COUNT,VOLUME_COUNT,HISTORICAL_COUNT,WIDTH=253,252,252,49

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


def measure(prices253,volumes252):
    """Scan both entire inputs before missing fallback; retain exact raw gains.

    Native zero volume is unavailable by kernel policy and is not passed here.
    This mathematical positive-input primitive rejects all nonpositive values.
    """
    pin=_sequence(prices253,PRICE_COUNT,'prices')
    vin=_sequence(volumes252,VOLUME_COUNT,'volumes')
    prices,pm,pe=_scan(pin,'prices');volumes,vm,ve=_scan(vin,'volumes')
    if pe or ve:raise ValueError('; '.join(pe+ve))
    result=dict(schema=SCHEMA,available=False,reason='missing_source',price_count=253,
      volume_count=252,historical_count=252,missing_price_indices=pm,missing_volume_indices=vm,
      reference_price=None,historical_price_sum=None,volume_sum=None,weighted_price_sum=None,
      weighted_reference=None,equal_reference=None,raw_volume_gain=None,raw_equal_gain=None)
    if pm or vm:return result
    p=[Fraction.from_float(x) for x in prices];v=[Fraction.from_float(x) for x in volumes]
    ref=p[-1];sp=sum(p[:-1],Fraction(0));sv=sum(v,Fraction(0))
    spv=sum((x*y for x,y in zip(p[:-1],v)),Fraction(0))
    weighted=spv/sv;equal=sp/HISTORICAL_COUNT
    result.update(available=True,reason='available',reference_price=_pair(ref),
      historical_price_sum=_pair(sp),volume_sum=_pair(sv),weighted_price_sum=_pair(spv),
      weighted_reference=_pair(weighted),equal_reference=_pair(equal),
      raw_volume_gain=_pair((ref-weighted)/ref),raw_equal_gain=_pair((ref-equal)/ref))
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
