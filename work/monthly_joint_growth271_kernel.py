"""Prospective 271 pure, caller-supplied monthly basket selection.

The 251 domain is retained: every original-E name has 253 positive binary64
prices checked, positive exact annual momentum is required, and its strongest
six names form the trio universe. C is mean constituent block log growth;
J is mean log growth of a blockwise equal-weight basket. Exact rational products
determine winners. Log displays never determine order or ties.

This module performs no file I/O, source authentication, predictive fit, strategy
return calculation, or registration. A date and a 253-value array do NOT prove
the t-294..t-42 clock, official month end, or original eligibility/incumbent.
The caller must authenticate these before use. Historical blockwise equal
weights describe an objective; deployment remains monthly equal weights and
fixed units, without an intramonth rebalance. The long-momentum Top6 gate is
not the old Local67 numerical 52% core. Global novelty remains UNKNOWN: the
prospective audit was bounded to 140 available PLANs, with 125 unavailable.
"""
from collections.abc import Mapping
from datetime import date as Date
from fractions import Fraction
from itertools import combinations
import math
from numbers import Real
import re

ARMS = ('incumbent_local67', 'member_log_growth', 'joint_log_growth')
INITIAL_PARTIAL = '2016-08-17'
MEASURE_SCHEMA = 'monthly_joint_growth271_measure_v1'
BASKET_SCHEMA = 'monthly_joint_growth271_basket_v1'
SCHEMA = 'monthly_joint_growth271_selection_v1'
INPUT_KEYS = ('date', 'symbols', 'eligible', 'prices253', 'incumbent_choices')
COUNT_FIELDS = ('events', 'symbols_per_event', 'incumbent_copy_events',
                'initial_partial_events', 'formal_events', 'required_cells',
                'price_observations', 'momentum_evaluations',
                'block_gross_evaluations', 'positive_momentum_cells',
                'excluded_momentum_cells', 'source_rankings', 'shortlist_cells',
                'trio_evaluations', 'objective_comparisons',
                'exact_amgm_block_checks', 'exact_amgm_total_checks',
                'cash_events', 'two_name_events', 'optimized_events',
                'selection_calls', 'new_model_fits', 'predictive_model_fits')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _sequence(value, size, label):
    require(not isinstance(value, (str, bytes, Mapping, set, frozenset)), label + ' ordered sequence')
    try:
        out = list(value)
    except TypeError as exc:
        raise ValueError(label + ' sequence') from exc
    require(len(out) == size and getattr(value, 'ndim', 1) == 1, label + ' exact length')
    return out


def _integer_text(value):
    # Valid extreme binary64 inputs can produce >4300 decimal-digit products.
    # Chunking avoids a process-wide change to Python's integer conversion cap.
    negative = value < 0
    value = abs(value)
    chunks = []
    while value >= 10**9:
        value, remainder = divmod(value, 10**9)
        chunks.append(remainder)
    return ('-' if negative else '') + str(value) + ''.join(f'{v:09d}' for v in reversed(chunks))


def _integer_value(value):
    negative = value.startswith('-')
    digits = value[1:] if negative else value
    out = 0
    for i in range(0, len(digits), 9):
        part = digits[i:i + 9]
        out = out * 10**len(part) + int(part)
    return -out if negative else out


def rational(value):
    """Read a reduced canonical rational (or an already exact Fraction)."""
    if type(value) is Fraction:
        return value
    require(type(value) is dict and set(value) == {'numerator', 'denominator'}, 'Canonical rational fields')
    a, b = value['numerator'], value['denominator']
    require(type(a) is str and type(b) is str and
            re.fullmatch(r'0|-?[1-9][0-9]*', a) is not None and
            re.fullmatch(r'[1-9][0-9]*', b) is not None, 'Canonical integer strings')
    out = Fraction(_integer_value(a), _integer_value(b))
    require(pair(out) == value, 'Reduced rational')
    return out


def pair(value):
    value = Fraction(value)
    return dict(numerator=_integer_text(value.numerator), denominator=_integer_text(value.denominator))


def _product(values):
    out = Fraction(1)
    for value in values:
        out *= value
    return out


def _log(value):
    """Finite display only; never round a ratio before the exact comparison."""
    require(value > 0, 'Positive log argument')
    if value == 1:
        return 0.0
    if value < 1:
        return -_log(1 / value)
    if value < 2:
        # Very small positive differences may display as zero; exact evidence
        # remains available. No clipping or epsilon influences decisions.
        out = math.log1p(float(value - 1))
    else:
        out = math.log(value.numerator) - math.log(value.denominator)
    require(math.isfinite(out) and out >= 0, 'Finite log display')
    return out


def slice_history(price_prefix_to_signal):
    """Take exactly t-294..t-42 from a caller's prefix ending at signal t.

    Only those 253 positions are accessed. No date/source authentication occurs.
    Latest42 values and any earlier history are not inspected.
    """
    require(not isinstance(price_prefix_to_signal, (str, bytes, Mapping, set, frozenset)), 'Ordered history')
    try:
        size = len(price_prefix_to_signal)
    except TypeError as exc:
        raise ValueError('Sized history') from exc
    require(size >= 295 and getattr(price_prefix_to_signal, 'ndim', 1) == 1, 'Complete history clock')
    return [price_prefix_to_signal[i] for i in range(size - 295, size - 42)]


def measure(prices253):
    """Validate ALL253 prices, then compute exact annual and block gross ratios."""
    values = _sequence(prices253, 253, 'Prices')
    prices, errors = [], []
    for i, value in enumerate(values):
        try:
            require(isinstance(value, Real) and not isinstance(value, bool) and
                    type(value).__name__ != 'bool_', f'Price {i} real nonboolean')
            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f'Price {i} binary64') from exc
            require(math.isfinite(number) and number > 0, f'Price {i} finite positive')
            prices.append(Fraction.from_float(number))
        except ValueError as exc:
            errors.append(str(exc))
            prices.append(None)
    require(not errors, '; '.join(errors))
    gross = [prices[21 * (b + 1)] / prices[21 * b] for b in range(12)]
    annual = prices[-1] / prices[0]
    require(_product(gross) == annual, 'Exact annual/block telescoping')
    return dict(schema=MEASURE_SCHEMA, price_count=253, block_count=12,
                annual_gross=pair(annual), momentum=pair(annual - 1),
                block_gross=[pair(value) for value in gross])


def basket(gross3x12):
    """All exact evidence for C=sum(log g)/36 and J=sum(log(mean g))/12."""
    rows = [[rational(x) for x in _sequence(row, 12, 'Blocks')]
            for row in _sequence(gross3x12, 3, 'Basket')]
    require(all(x > 0 for row in rows for x in row), 'Positive block gross')
    annual = [_product(row) for row in rows]
    require(all(x > 1 for x in annual), 'Positive annual momentum for every constituent')
    means = [sum((row[b] for row in rows), Fraction(0)) / 3 for b in range(12)]
    constituent_products = [_product(row[b] for row in rows) for b in range(12)]
    block_gaps = [means[b]**3 - constituent_products[b] for b in range(12)]
    require(all(x >= 0 for x in block_gaps), 'Exact block AM/GM invariant')
    control, joint = _product(annual), _product(means)
    total_gap = joint**3 - control
    require(total_gap >= 0, 'Exact cumulative AM/GM invariant')
    c = math.fsum(_log(x) for row in rows for x in row) / 36
    j = math.fsum(_log(x) for x in means) / 12
    gap = math.fsum(_log(means[b]**3 / constituent_products[b]) for b in range(12)) / 36
    require(all(math.isfinite(x) for x in (c, j, gap)) and gap >= 0, 'Finite log evidence')
    return dict(schema=BASKET_SCHEMA, block_count=12,
                constituent_annual_gross=[pair(x) for x in annual],
                mean_block_gross=[pair(x) for x in means],
                control_product=pair(control), joint_product=pair(joint),
                amgm_block_gaps=[pair(x) for x in block_gaps],
                amgm_total_gap=pair(total_gap), control_log=c, joint_log=j,
                direct_log_gap=j - c, log_gap_from_exact_ratios=gap,
                log_displays_used_for_selection=False)


def _metadata(event, previous, width):
    require(type(event) is dict and set(event) == set(INPUT_KEYS), 'Event exact fields')
    day = event['date']
    require(type(day) is str, 'Date string')
    try:
        require(Date.fromisoformat(day).isoformat() == day, 'Canonical date')
    except (TypeError, ValueError) as exc:
        raise ValueError('Canonical date') from exc
    require(previous is None or previous < day, 'Increasing unique event dates')
    symbols = event['symbols']
    require(not isinstance(symbols, (str, bytes, Mapping, set, frozenset)), 'Ordered symbols')
    try:
        size = len(symbols)
    except TypeError as exc:
        raise ValueError('Sized symbols') from exc
    require(size > 0 and (width is None or size == width), 'Stable nonempty symbol width')
    symbols = _sequence(symbols, size, 'Symbols')
    require(all(type(x) is str and x and x.strip() == x for x in symbols) and len(set(symbols)) == size, 'Unique ticker strings')
    eligible = _sequence(event['eligible'], size, 'Eligibility')
    require(all(type(x) is bool for x in eligible), 'Boolean original eligibility')
    original = event['incumbent_choices']
    require(type(original) in (list, tuple), 'Ordered incumbent choices')
    chosen = list(original)
    require(all(type(x) is int and 0 <= x < size and eligible[x] for x in chosen) and
            len(set(chosen)) == len(chosen), 'Eligible unique incumbent indices')
    expected = min(sum(eligible), 3) if sum(eligible) >= 2 else 0
    require(len(chosen) == expected, 'Incumbent EW3/halves/cash cardinality')
    return day, symbols, eligible, chosen


def select(events, mode, *, n=None):
    """Return ordered index choices and a fully recomputable pure audit.

    The caller supplies each event's authenticated original E and incumbent
    choices. Only events[:n] are touched. Initial partial and incumbent mode do
    not access prices253. All formal original-E prices are checked before the
    positive-momentum filter; Efalse values are not accessed.
    """
    require(mode in ARMS and type(mode) is str, 'Registered arm name')
    require(not isinstance(events, (str, bytes, Mapping, set, frozenset)), 'Ordered events')
    try:
        available = len(events)
    except TypeError as exc:
        raise ValueError('Sized events') from exc
    if n is None:
        n = available
    require(type(n) is int and 0 <= n <= available, 'Event prefix')
    counts = dict.fromkeys(COUNT_FIELDS, 0)
    counts.update(events=n, selection_calls=1)
    choices, audit_events = [], []
    previous, width = None, None
    for i in range(n):
        event = events[i]
        day, symbols, eligible, incumbent = _metadata(event, previous, width)
        previous, width = day, len(symbols)
        partial = day == INITIAL_PARTIAL
        required = [k for k, good in enumerate(eligible) if good]
        out = dict(query_index=i, date=day, symbols=symbols,
                   initial_partial=partial, original_eligible_indices=required,
                   incumbent_choices=incumbent, branch=None, measurements=[],
                   new_eligible_indices=[], excluded_momentum_indices=[],
                   shortlist_indices=[], trios=[], winning_trio_index=None,
                   tied_trio_indices=[], selected_indices=None, selected_symbols=None,
                   equal_weights=None, control_top3_equivalence_checked=False)
        counts['initial_partial_events'] += int(partial)
        counts['formal_events'] += int(not partial)
        if partial or mode == ARMS[0]:
            selected = incumbent.copy()
            out['branch'] = 'initial_partial' if partial else 'incumbent_copy'
            counts['incumbent_copy_events'] += 1
        else:
            paths = event['prices253']
            require(not isinstance(paths, (str, bytes, Mapping, set, frozenset)), 'Price columns')
            try:
                require(len(paths) == width, 'Price column width')
            except TypeError as exc:
                raise ValueError('Sized price columns') from exc
            measures, errors = {}, []
            for k in required:
                try:
                    measures[k] = measure(paths[k])
                except ValueError as exc:
                    errors.append(f'Column {k}: {exc}')
            require(not errors, '; '.join(errors))
            counts['required_cells'] += len(required)
            counts['price_observations'] += 253 * len(required)
            counts['momentum_evaluations'] += len(required)
            counts['block_gross_evaluations'] += 12 * len(required)
            out['measurements'] = [dict(column_index=k, symbol=symbols[k], measurement=measures[k]) for k in required]
            momentum = {k: rational(measures[k]['momentum']) for k in required}
            positive = [k for k in required if momentum[k] > 0]
            excluded = [k for k in required if momentum[k] <= 0]
            short = sorted(positive, key=lambda k: (-momentum[k], symbols[k]))[:6]
            out.update(new_eligible_indices=positive, excluded_momentum_indices=excluded, shortlist_indices=short)
            counts['positive_momentum_cells'] += len(positive)
            counts['excluded_momentum_cells'] += len(excluded)
            counts['shortlist_cells'] += len(short)
            counts['source_rankings'] += 1
            if len(positive) < 2:
                selected, branch = [], 'cash'
                counts['cash_events'] += 1
            elif len(positive) == 2:
                selected, branch = short.copy(), 'two_names'
                counts['two_name_events'] += 1
            else:
                branch = 'optimized'
                counts['optimized_events'] += 1
                best_value, best, ties = None, None, []
                for group in combinations(sorted(short, key=lambda k: symbols[k]), 3):
                    statistics = basket([measures[k]['block_gross'] for k in group])
                    trio = dict(indices=list(group), symbols=[symbols[k] for k in group], statistics=statistics)
                    index = len(out['trios'])
                    out['trios'].append(trio)
                    key = 'control_product' if mode == ARMS[1] else 'joint_product'
                    value = rational(statistics[key])
                    if best_value is not None:
                        counts['objective_comparisons'] += 1
                    if best_value is None or value > best_value:
                        best_value, best, ties = value, index, [index]
                    elif value == best_value:
                        ties.append(index)
                counts['trio_evaluations'] += len(out['trios'])
                counts['exact_amgm_block_checks'] += 12 * len(out['trios'])
                counts['exact_amgm_total_checks'] += len(out['trios'])
                selected = sorted(out['trios'][best]['indices'], key=lambda k: (-momentum[k], symbols[k]))
                out.update(winning_trio_index=best, tied_trio_indices=ties)
                if mode == ARMS[1]:
                    require(selected == short[:3], 'Exact C/top3 annual momentum equivalence')
                    out['control_top3_equivalence_checked'] = True
            out['branch'] = branch
        out.update(selected_indices=selected, selected_symbols=[symbols[k] for k in selected],
                   equal_weights=[1.0 / len(selected)] * len(selected) if selected else [])
        choices.append(selected.copy())
        audit_events.append(out)
    counts['symbols_per_event'] = width or 0
    return choices, dict(schema=SCHEMA, mode=mode, counts=counts, events=audit_events,
                         source_files_authenticated=False, original_incumbent_authenticated=False,
                         official_clock_authenticated=False, log_displays_used_for_selection=False,
                         computes_strategy_returns=False, computes_predictive_labels=False)
