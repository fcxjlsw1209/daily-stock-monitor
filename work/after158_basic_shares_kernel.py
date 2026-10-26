"""Unregistered, source-independent basic-share feature kernel.

No I/O or real-data execution. Source admission, original-universe alignment,
historical availability construction and experiment registration belong to
the caller. The two transforms share the same valid-source population.
"""
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import math


def require(ok, message):
    if not ok:
        raise ValueError(message)


def stamp(value):
    require(isinstance(value, str), 'Timestamp string required')
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(result.tzinfo is not None, 'Timezone required')
    return result


def day(value):
    require(isinstance(value, str), 'Date string required')
    result = date.fromisoformat(value)
    require(result.isoformat() == value, 'Strict ISO date required')
    return result


def count(value):
    """Consume exact normalized decimal strings; never infer a missing count."""
    require(isinstance(value, str) and value.strip() == value, 'Decimal string required')
    try:
        result = Decimal(value)
    except InvalidOperation as error:
        raise ValueError('Invalid share count') from error
    require(result.is_finite() and result > 0, 'Positive finite share count required')
    return result


def latest_known(events, symbol, cutoff):
    """Choose latest known annual period BEFORE testing its field quality.

    ``known_at`` belongs to the release. ``usable_at`` includes any later
    corroboration. An already known broken/new report must block an older one.
    Future field values and future quality flags are never examined here.
    """
    cutoff = stamp(cutoff)
    known = [e for e in events if e['symbol'] == symbol and stamp(e['known_at']) <= cutoff]
    if not known:
        return None, 'no_known_original'
    ends = [day(e['period_end']) for e in known]
    require(all(end <= cutoff.date() for end in ends), 'Known report ends in future')
    latest_end = max(ends)
    chosen = [e for e, end in zip(known, ends) if end == latest_end]
    require(len(chosen) == 1, 'Ambiguous latest original; source review required')
    return chosen[0], None


def measure(event, cutoff, max_age_days=550):
    """Return negative same-report annual denominator growth, or a reason."""
    cutoff = stamp(cutoff)
    require(type(max_age_days) is int and max_age_days >= 0, 'Nonnegative age limit required')
    known, usable = stamp(event['known_at']), stamp(event['usable_at'])
    require(known <= usable, 'Usable clock precedes knowledge')
    if usable > cutoff:
        return None, 'latest_not_yet_usable'
    if event.get('conditional_source_ready') is not True:
        return None, 'latest_source_not_conditionally_ready'
    if event.get('same_report_annual_pair') is not True:
        return None, 'same_report_annual_pair_required'
    if event.get('same_share_basis') is not True:
        return None, 'same_share_basis_unverified'
    if event.get('unresolved_reasons') != []:
        return None, 'unresolved_source_reasons'
    if (cutoff.date() - day(event['period_end'])).days > max_age_days:
        return None, 'latest_annual_period_stale'
    require(day(event['prior_period_end']) < day(event['period_end']), 'Annual period order')
    try:
        current, previous = count(event['current_shares']), count(event['prior_shares'])
    except (ValueError, KeyError):
        return None, 'invalid_share_denominator'
    # Unit normalization occurs at extraction; no split inference or adjustment.
    negative_growth = Fraction(1) - Fraction(current) / Fraction(previous)
    return negative_growth, None


def transform(values):
    """Matched availability; both arms neutral if fewer than two valid stocks.

    h = x/(.05+abs(x)); magnitude = (h-mean(h))/2.
    Rank is the centered average rank scaled to [-1,1]. The .05 scale
    is a fixed five-percent annual share change, never estimated from data.
    None is unavailable, not an observed zero. Returned deltas contain no dose.
    """
    require(isinstance(values, (list, tuple)), 'Ordered values required')
    available = [i for i, x in enumerate(values) if x is not None]
    require(all(type(values[i]) in (float, int, Fraction) and
                (not isinstance(values[i], float) or math.isfinite(values[i]))
                for i in available), 'Finite numeric values required')
    magnitudes, ranks = [0.] * len(values), [0.] * len(values)
    if len(available) < 2:
        return magnitudes, ranks
    # Source decimal counts produce exact rational changes. Rank ties must
    # not depend on an early float conversion or Decimal's division context.
    exact = {i: (Fraction(str(values[i])) if isinstance(values[i], float)
                 else Fraction(values[i])) for i in available}
    population = [exact[i] for i in available]
    scale = Fraction(1,20)
    bounded = [x/(scale+abs(x)) for x in population]
    mean = sum(bounded)/len(bounded)
    if len(set(population)) == 1:
        return magnitudes, ranks
    for i in available:
        x = exact[i]
        magnitudes[i] = float((x/(scale+abs(x))-mean)/2)
        ranks[i] = 2*((sum(y < x for y in population) +
                     .5*(sum(y == x for y in population)-1)) /
                    (len(population)-1) - .5)
    return magnitudes, ranks


def evaluate_month(events, symbols, eligible, cutoff, initial_partial=False):
    """Consume caller-reviewed events only; no prices, ranking eligibility or I/O."""
    require(len(symbols) == len(set(symbols)), 'Duplicate original candidate')
    require(len(eligible) == len(symbols) and all(type(x) is bool for x in eligible),
            'Original boolean eligibility required')
    values, audit = [], []
    for symbol, allowed in zip(symbols, eligible):
        if initial_partial:
            event, reason = None, 'initial_partial_exact_fallback'
        elif not allowed:
            event, reason = None, 'original_ineligible'
        else:
            event, reason = latest_known(events, symbol, cutoff)
        value = None
        if event is not None:
            value, reason = measure(event, cutoff)
        values.append(value)
        audit.append(dict(symbol=symbol, event_id=event['event_id'] if event else None,
                          reason=reason, available=value is not None))
    magnitude, rank = transform(values)
    return dict(negative_growth=values, magnitude=magnitude, rank=rank, audit=audit)
