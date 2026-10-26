"""Pure filing-drift calculations on supplied arrays and admitted integer anchors.

No file, network, timestamp parsing, registration, scoring or portfolio API is
provided. Source admission is an explicit caller obligation; passing True is
not evidence that admission has actually happened. The current real-data clock
gate remains pending. See the separate frozen after156_filing_drift_design.json.
"""
from dataclasses import dataclass
import math
from numbers import Real


NAMES = ('whole_formation_beta_residual_momentum', 'filing_window_density',
         'filing_vs_nonfiling_beta_residual_contrast')
ORIGINAL = frozenset(('10-Q', '10-K'))
AMENDMENT = frozenset(('10-Q/A', '10-K/A'))
DOSE = .01  # Informational; this module never applies a score correction.


@dataclass(frozen=True)
class Admission:
    clock_admitted: bool
    chronology_complete: bool
    identity_admitted: bool
    first_anchor: int
    last_anchor: int


@dataclass(frozen=True)
class Event:
    accession: str
    anchor: int
    form: str


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _admitted(signal_index, admission):
    _require(type(signal_index) is int and signal_index >= 251,
             'A nonnegative full formation window is required')
    _require(isinstance(admission, Admission), 'Explicit Admission is required')
    for name in ('clock_admitted', 'chronology_complete', 'identity_admitted'):
        _require(getattr(admission, name) is True, 'Source precondition not admitted: ' + name)
    _require(type(admission.first_anchor) is int and type(admission.last_anchor) is int
             and admission.first_anchor <= signal_index - 271
             and admission.last_anchor >= signal_index - 42,
             'Complete chronology must cover every relevant anchor')


def partition(signal_index, events, admission):
    """Return a complete metadata-only E/B/N partition on canonical session indices.

    The caller, not this kernel, proves timestamp-to-anchor mapping and original
    report identity. Exact accession duplicates coalesce; conflicting ones fail.
    Events outside the consumed anchor interval do not enter the returned audit.
    """
    _admitted(signal_index, admission)
    left, right = signal_index - 251, signal_index - 42
    first_anchor = signal_index - 271
    unique = {}
    for event in events:
        _require(isinstance(event, Event), 'Typed Event is required')
        _require(type(event.accession) is str and bool(event.accession.strip()),
                 'Nonempty accession is required')
        _require(type(event.anchor) is int and event.anchor >= 0,
                 'Already-admitted integer session anchor required')
        _require(event.form in ORIGINAL | AMENDMENT, 'Unexpected filing form')
        prior = unique.setdefault(event.accession, event)
        _require(prior == event, 'Conflicting duplicate accession')
    grouped = {}
    for event in unique.values():
        if first_anchor <= event.anchor <= right:
            grouped.setdefault(event.anchor, []).append(event)
    bundles = []
    for anchor, group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda event: event.accession)
        bundles.append(dict(anchor=anchor, accessions=[event.accession for event in ordered],
            forms=[event.form for event in ordered],
            amendment=any(event.form in AMENDMENT for event in ordered),
            treatment_left_clipped=anchor + 2 < left,
            treatment_right_clipped=anchor + 20 > right))
    days, position = [], -1
    for day in range(left, right + 1):
        while position + 1 < len(bundles) and bundles[position + 1]['anchor'] <= day:
            position += 1
        bundle = bundles[position] if position >= 0 else None
        active = bundle is not None and day - bundle['anchor'] <= 20
        if not active:
            label, owner = 'N', None
        else:
            owner = bundle['anchor']
            label = 'B' if bundle['amendment'] or day - owner <= 1 else 'E'
        days.append(dict(session=day, label=label, owner_anchor=owner,
                         accessions=list(bundle['accessions']) if active else []))
    full = []
    for bundle in bundles:
        anchor = bundle['anchor']
        owned = [day['session'] for day in days
                 if day['label'] == 'E' and day['owner_anchor'] == anchor]
        bundle['owned_event_days'] = owned
        bundle['full_event_window'] = not bundle['amendment'] and owned == list(range(anchor + 2, anchor + 21))
        if bundle['full_event_window']:
            full.append(anchor)
    counts = {label: sum(day['label'] == label for day in days) for label in ('E', 'B', 'N')}
    return dict(formation_first=left, formation_last=right, required_first_anchor=first_anchor,
                required_last_anchor=right, days=days, bundles=bundles, counts=counts,
                full_original_anchors=full,
                minimum_passed=len(full) >= 2 and counts['E'] >= 38 and counts['N'] >= 80)


def _unavailable(reason, parts, attempted=0):
    return dict(available=False, reason=reason, hands=dict.fromkeys(NAMES, float('nan')),
                raw=dict.fromkeys(NAMES, None), beta=None, partition=parts,
                nuisance_beta_estimates_attempted=attempted,
                nuisance_beta_estimates_completed=0, new_predictive_model_fits=0)


def _positive_price(value):
    if value is None:
        return False
    _require(isinstance(value, Real) and not isinstance(value, bool),
             'Price arrays require real scalar values, not booleans or strings')
    try:
        return math.isfinite(value) and value > 0
    except OverflowError:
        return False


def _log_change(previous, current):
    ratio = current / previous
    value = math.log(ratio) if 0 < ratio < math.inf else math.log(current) - math.log(previous)
    if not math.isfinite(value):
        raise ArithmeticError('Nonfinite log-return arithmetic')
    return value


def _sum(values):
    try:
        value = math.fsum(values)
    except (OverflowError, ValueError) as exc:
        raise ArithmeticError('Unresolved finite-sum arithmetic') from exc
    if not math.isfinite(value):
        raise ArithmeticError('Nonfinite sum')
    return value


def _bounded(value):
    if not math.isfinite(value):
        raise ArithmeticError('Nonfinite unbounded hand')
    return value / (1 + abs(value))


def evaluate_cell(signal_index, stock_prices, spy_prices, events, admission,
                  *, price_start_index=0):
    """Compute one cell, without reading prices beyond t-42 or shortening windows.

    The two equally sized arrays share canonical session indices, beginning at
    price_start_index. They may end at t-42; values after it are never accessed.
    Missing required prices/warmup or a zero market variance makes ALL hands
    unavailable. Malformed axes/admission and unexpected arithmetic are errors.
    No price-based trading eligibility is accepted or altered here.
    """
    parts = partition(signal_index, events, admission)
    _require(type(price_start_index) is int and price_start_index >= 0,
             'Canonical nonnegative price start index required')
    _require(len(stock_prices) == len(spy_prices), 'Stock/SPY price axes differ')
    if not parts['minimum_passed']:
        return _unavailable('insufficient_event_or_nonevent_days', parts)
    first, split, last = signal_index - 504, signal_index - 252, signal_index - 42
    if first < price_start_index:
        return _unavailable('insufficient_preformation_warmup', parts)
    if last >= price_start_index + len(stock_prices):
        return _unavailable('incomplete_formation_price_window', parts)
    start, stop = first - price_start_index, last - price_start_index + 1
    stock = list(stock_prices[start:stop])
    market = list(spy_prices[start:stop])
    # Check only consumed cells. A future NaN or discontinuity is immaterial.
    for offset, (value, benchmark) in enumerate(zip(stock, market)):
        if not _positive_price(value) or not _positive_price(benchmark):
            name = 'beta' if first + offset <= split else 'formation'
            return _unavailable('invalid_required_' + name + '_price', parts)
    stock_r = [_log_change(a, b) for a, b in zip(stock, stock[1:])]
    market_r = [_log_change(a, b) for a, b in zip(market, market[1:])]
    _require(len(stock_r) == len(market_r) == 462, 'Fixed return windows changed')
    y, x = stock_r[:252], market_r[:252]
    mean_y, mean_x = _sum(y) / 252, _sum(x) / 252
    centered_x = [value - mean_x for value in x]
    denominator = _sum(value * value for value in centered_x)
    if denominator == 0:
        return _unavailable('zero_preformation_market_variance', parts, attempted=1)
    numerator = _sum(a * (b - mean_y) for a, b in zip(centered_x, y))
    beta = numerator / denominator
    if not math.isfinite(beta):
        raise ArithmeticError('Nonfinite beta estimate')
    residuals = [a - beta * b for a, b in zip(stock_r[252:], market_r[252:])]
    if not all(math.isfinite(value) for value in residuals):
        raise ArithmeticError('Nonfinite formation residual')
    event_values = [value for value, day in zip(residuals, parts['days']) if day['label'] == 'E']
    ordinary_values = [value for value, day in zip(residuals, parts['days']) if day['label'] == 'N']
    whole = 21 * (_sum(residuals) / 210)
    contrast = 21 * (_sum(event_values) / len(event_values) - _sum(ordinary_values) / len(ordinary_values))
    density = len(event_values) / (len(event_values) + len(ordinary_values))
    hands = dict(zip(NAMES, (_bounded(whole / .05), density, _bounded(contrast / .05))))
    if not all(math.isfinite(value) and abs(value) <= 1 for value in hands.values()):
        raise ArithmeticError('Invalid bounded hands')
    return dict(available=True, reason=None, hands=hands,
                raw=dict(zip(NAMES, (whole, density, contrast))), beta=beta, partition=parts,
                nuisance_beta_estimates_attempted=1, nuisance_beta_estimates_completed=1,
                new_predictive_model_fits=0)
