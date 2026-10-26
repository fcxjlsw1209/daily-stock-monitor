"""Pure proposed ROA-stability arithmetic; no source admission, dates or I/O.

Caller must select an admitted causal three-report chain before using this.
Reported amounts, ratios, mean and variance stay rational. Sample standard
deviation is sqrt(round64(exact variance)); that binary64 result is converted
back to Fraction for the penalty. Each exact rational h is finally rounded to
binary64. Any nonzero-to-zero conversion or nonfinite conversion fails.
This is a dispersion penalty, not a statistically calibrated confidence bound.
"""
from fractions import Fraction
import math


def require(ok, message):
    if not ok:
        raise ValueError(message)


def bounded(value):
    require(type(value) is Fraction, 'Exact rational required')
    return finite_float(value / (1 + abs(value)))


def finite_float(value):
    try:
        out = float(value)
    except OverflowError as exc:
        raise FloatingPointError('Rational magnitude outside binary64') from exc
    if not math.isfinite(out) or (value != 0 and out == 0.):
        raise FloatingPointError('Nonfinite or underflowed nonzero rational')
    return out


def measure(amounts):
    """Three own-year tuples (NI, opening assets, closing assets), all Fraction.

    No cross-report denominator stitching or annualization. Negative NI is valid.
    Missing/held observations are the caller's shared-mask concern; malformed
    source amounts here fail rather than become neutral numerical observations.
    """
    require(type(amounts) in (list, tuple) and len(amounts) == 3,
            'Exactly three annual observations required')
    ratios = []
    for row in amounts:
        require(type(row) in (list, tuple) and len(row) == 3,
                'Own-year NI and two asset endpoints required')
        require(all(type(x) is Fraction for x in row), 'Exact original amounts required')
        ni, opening, closing = row
        require(opening > 0 and closing > 0, 'Positive asset endpoints required')
        ratios.append(ni / ((opening + closing) / 2))
    mean = sum(ratios, Fraction(0)) / 3
    variance = sum(((x - mean) ** 2 for x in ratios), Fraction(0)) / 2
    # The power above is exact Fraction arithmetic, not binary64 pow.
    sigma = math.sqrt(finite_float(variance)) if variance != 0 else 0.
    penalty = mean - Fraction.from_float(sigma)
    control = bounded(mean)
    candidate = control if variance == 0 else bounded(penalty)
    require(candidate <= control, 'Dispersion penalty increased score')
    return dict(roa=tuple(ratios), mean=mean, sample_variance=variance,
                sample_std=sigma, penalized_mean=penalty,
                candidate=candidate, control=control,
                equal_arms=(candidate == control), zero_dispersion=(variance == 0))
