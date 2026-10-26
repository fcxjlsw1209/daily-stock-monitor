"""Independent pure arithmetic oracle for the unregistered 265 proposal.

No producer imports, files, prices from a dataset, fitting, strategy integration,
ranking or replay. The 253-price clock and original score provenance are the
caller's responsibility. Concentrations use unnormalized squared masses divided
by exact total mass squared; normalized weights are returned only as evidence.
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


def binary64_number(source):
    """Validate the Real domain and preserve signed zero on conversion."""
    if isinstance(source, bool) or not isinstance(source, Real):
        raise ValueError('Expected a nonboolean real number')
    try:
        number = float(source)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError('Input cannot be represented as finite binary64') from exc
    if not math.isfinite(number):
        raise ValueError('Input must be finite')
    if number == 0.0 and source != 0:
        raise ArithmeticError('Nonzero input rounded to zero')
    return number


def ordered_numbers(source, length):
    if isinstance(source, (Mapping, str, bytes, bytearray)):
        raise ValueError('Expected an indexed numeric sequence')
    try:
        if len(source) != length:
            raise ValueError('Wrong fixed sequence length')
    except TypeError as exc:
        raise ValueError('Expected an explicit sequence length') from exc
    result = []
    for index in range(length):
        try:
            item = source[index]
        except (TypeError, IndexError, KeyError) as exc:
            raise ValueError('Sequence must support ordered integer indexing') from exc
        result.append(binary64_number(item))
    return result


def exact_qualities_from_gains(gains252):
    """Audit arbitrary finite simple-gain inputs, without a reachability claim."""
    gains = ordered_numbers(gains252, GAIN_COUNT)
    positives = []
    locations = []
    total = Fraction(0, 1)
    square_sum = Fraction(0, 1)
    block_totals = [Fraction(0, 1) for unused in range(BLOCK_COUNT)]
    masses = []
    for offset, gain in enumerate(gains):
        if gain > 0:
            locations.append(offset + 1)
            positive = gain
            mass = Fraction.from_float(gain)
        else:
            positive = 0.0
            mass = Fraction(0, 1)
        positives.append(positive)
        masses.append(mass)
        total += mass
        square_sum += mass * mass
        block_totals[offset // BLOCK_SIZE] += mass

    if total:
        total_squared = total * total
        block_square_sum = Fraction(0, 1)
        for block_mass in block_totals:
            block_square_sum += block_mass * block_mass
        hd = square_sum / total_squared
        hb = block_square_sum / total_squared
        if not (Fraction(1, 252) <= hd <= hb <= 1 and hb >= Fraction(1, 12)):
            raise ArithmeticError('Exact concentration invariant failed')
        qd = Fraction(252, 251) * (1 - hd)
        qb = Fraction(12, 11) * (1 - hb)
        if not (0 <= qd <= 1 and 0 <= qb <= 1):
            raise ArithmeticError('Exact quality invariant failed')
        # These divisions are evidence only, never the concentration algorithm.
        daily_weights = tuple(mass / total for mass in masses)
        block_weights = tuple(mass / total for mass in block_totals)
    else:
        hd = hb = None
        daily_weights = block_weights = None
        qd = qb = Fraction(1, 1)

    return {
        'daily_gains': tuple(gains),
        'positive_parts': tuple(positives),
        'positive_gain_indices': tuple(locations),
        'gain_sum': total,
        'block_gain_sums': tuple(block_totals),
        'daily_weights': daily_weights,
        'block_weights': block_weights,
        'daily_concentration': hd,
        'block_concentration': hb,
        'daily_quality': qd,
        'block_quality': qb,
        'no_positive_gains': not bool(total),
    }


def quality_to_binary64(quality):
    if not isinstance(quality, Fraction) or quality < 0 or quality > 1:
        raise ValueError('Quality must be an exact bounded Fraction')
    converted = float(quality)
    if quality > 0 and converted == 0.0:
        raise ArithmeticError('Positive quality conversion rounded to zero')
    if not math.isfinite(converted) or not 0 <= converted <= 1:
        raise ArithmeticError('Converted quality invariant failed')
    return converted


def apply_quality(original, quality):
    if original <= 0:
        return original
    transformed = original * quality
    if not math.isfinite(transformed):
        raise ArithmeticError('Raw score product overflow')
    if quality > 0 and transformed == 0:
        raise ArithmeticError('Nonzero score product rounded to zero')
    return transformed


def fraction_record(number):
    if number is None:
        return None
    return {'numerator': number.numerator, 'denominator': number.denominator}


def measure(prices253, original_raw):
    prices = ordered_numbers(prices253, PRICE_COUNT)
    if any(price <= 0 for price in prices):
        raise ValueError('Every required price must be strictly positive')
    original = binary64_number(original_raw)
    differences = []
    for index in range(1, PRICE_COUNT):
        quotient = prices[index] / prices[index - 1]
        if not math.isfinite(quotient):
            raise ArithmeticError('Price quotient overflow')
        if quotient == 0:
            raise ArithmeticError('Positive price quotient rounded to zero')
        change = quotient - 1.0
        if not math.isfinite(change):
            raise ArithmeticError('Simple gain overflow')
        differences.append(change)

    stats = exact_qualities_from_gains(differences)
    # Even for a nonpositive original score, verify both source-derived qualities.
    daily = quality_to_binary64(stats['daily_quality'])
    block = quality_to_binary64(stats['block_quality'])
    exact = {}
    for field in ('gain_sum', 'daily_concentration', 'block_concentration',
                  'daily_quality', 'block_quality'):
        exact[field] = fraction_record(stats[field])
    return {
        'schema': SCHEMA,
        'price_count': PRICE_COUNT,
        'gain_count': GAIN_COUNT,
        'block_size': BLOCK_SIZE,
        'block_count': BLOCK_COUNT,
        'daily_gains': list(stats['daily_gains']),
        'positive_parts': list(stats['positive_parts']),
        'positive_gain_indices': list(stats['positive_gain_indices']),
        'exact': exact,
        'exact_block_gain_sums': [fraction_record(mass) for mass in stats['block_gain_sums']],
        'daily_quality': daily,
        'block_quality': block,
        'original_raw': original,
        'daily_raw': apply_quality(original, daily),
        'block_raw': apply_quality(original, block),
        'nonpositive_original_preserved': original <= 0,
        'no_positive_gains': stats['no_positive_gains'],
    }
