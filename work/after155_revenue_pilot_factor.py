"""Pure fixed revenue-persistence hypothesis; synthetic inputs before registration.

The source gate and historical cutoff must be independently admitted by the
caller. This module has no source I/O, model fitting, portfolio or return code.
"""
from decimal import Decimal, localcontext
from fractions import Fraction
import math

NAMES = ('latest_revenue_growth', 'annualized_total_revenue_growth',
         'weakest_annual_revenue_growth')


def _level(value):
    if not isinstance(value, str):
        raise ValueError('Original exact decimal strings required')
    d = Decimal(value)
    if not d.is_finite() or d <= 0:
        raise ValueError('Positive finite annual revenue required')
    return Fraction(d)


def _bounded_fraction(value):
    return float(value / (1 + abs(value)))


def measure_levels(levels):
    """Four consecutive same-scale own-year levels, oldest first.

    Annual growth and its bounded transform use exact fractions. The three-year
    geometric rate uses 50-digit Decimal precision, then one bounded float.
    Nothing is ranked across issuers; the future caller adds exactly .01*h to
    its original pre-score under an identical source mask for all three arms.
    """
    if not isinstance(levels, (list, tuple)) or len(levels) != 4:
        raise ValueError('Exactly four consecutive levels required')
    values = [_level(value) for value in levels]
    growth = [b/a-1 for a, b in zip(values, values[1:])]
    ratio = values[3]/values[0]
    with localcontext() as ctx:
        ctx.prec = 50
        total = Decimal(ratio.numerator)/Decimal(ratio.denominator)
        annualized = total ** (Decimal(1)/Decimal(3)) - 1
        total_bounded = float(annualized/(1+abs(annualized)))
    result = dict(zip(NAMES, (_bounded_fraction(growth[-1]), total_bounded,
                             _bounded_fraction(min(growth)))))
    if not all(math.isfinite(value) and abs(value) <= 1 for value in result.values()):
        raise ValueError('Invalid bounded output')
    return result
