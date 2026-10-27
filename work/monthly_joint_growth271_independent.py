"""Independent, in-memory verifier for prospective 271 selections.

The caller supplies events and producer outputs. This module imports neither
the producer nor any research source/clock module and does no I/O. It authenticates
no source, date, eligibility, incumbent, or deployment return. All 253 required
observations are checked before the new positive-momentum domain is constructed.

The arithmetic uses reduced integer pairs, endpoint telescoping for C, and
three-bit subsets for J. Selection and ties are exact. Decimal logarithms only
check descriptive displays; their tolerance never affects any holding or gate.
Historical block equal weights imply resets between score blocks, not actual
intramonth portfolio rebalancing.
"""
from collections.abc import Mapping
from datetime import date as CalendarDate
from decimal import Decimal, localcontext
from functools import cmp_to_key
from math import gcd, isfinite, isclose
from numbers import Real


ARMS = ('incumbent_local67', 'member_log_growth', 'joint_log_growth')
PARTIAL_DATE = '2016-08-17'
_ONE = (1, 1)
_COUNT_NAMES = (
    'events', 'symbols_per_event', 'incumbent_copy_events',
    'initial_partial_events', 'formal_events', 'required_cells',
    'price_observations', 'momentum_evaluations', 'block_gross_evaluations',
    'positive_momentum_cells', 'excluded_momentum_cells', 'source_rankings',
    'shortlist_cells', 'trio_evaluations', 'objective_comparisons',
    'exact_amgm_block_checks', 'exact_amgm_total_checks', 'cash_events',
    'two_name_events', 'optimized_events', 'selection_calls', 'new_model_fits',
    'predictive_model_fits')
_LOG_FIELDS = frozenset(('control_log', 'joint_log', 'direct_log_gap',
                         'log_gap_from_exact_ratios'))
# Display agreement only: binary64 log subtraction can lose a few ulps.
LOG_ABSOLUTE_TOLERANCE = 3e-12
LOG_RELATIVE_TOLERANCE = 3e-13


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _q(a, b=1):
    _need(b != 0, 'zero denominator')
    if b < 0:
        a, b = -a, -b
    g = gcd(a, b)
    return a // g, b // g


def _times(a, b):
    # Cancel cross factors before multiplication, independently of Fraction.
    left, right = gcd(a[0], b[1]), gcd(b[0], a[1])
    return _q((a[0] // left) * (b[0] // right),
              (a[1] // right) * (b[1] // left))


def _over(a, b):
    return _times(a, _q(b[1], b[0]))


def _plus(a, b):
    return _q(a[0] * b[1] + b[0] * a[1], a[1] * b[1])


def _minus(a, b):
    return _plus(a, (-b[0], b[1]))


def _cube(a):
    return a[0] ** 3, a[1] ** 3


def _product(values):
    out = _ONE
    for value in values:
        out = _times(out, value)
    return out


def _compare(a, b):
    difference = a[0] * b[1] - b[0] * a[1]
    return (difference > 0) - (difference < 0)


def _wire(a):
    # Decimal(int) is exact and avoids interpreter integer-to-string caps.
    return {'numerator': str(Decimal(a[0])), 'denominator': str(Decimal(a[1]))}


def _decimal_log(a):
    with localcontext() as ctx:
        ctx.prec = 100
        return Decimal(a[0]).ln() - Decimal(a[1]).ln()


def _ordered(value, size, label):
    _need(not isinstance(value, (str, bytes, Mapping, set, frozenset)),
          label + ': ordered sequence required')
    _need(getattr(value, 'ndim', 1) == 1, label + ': one dimension required')
    try:
        out = list(value)
    except TypeError as exc:
        raise ValueError(label + ': sequence required') from exc
    _need(size is None or len(out) == size, label + ': length mismatch')
    return out


def _metadata(event, index, width, previous):
    _need(type(event) is dict and set(event) == {
        'date', 'symbols', 'eligible', 'prices253', 'incumbent_choices'},
        'event exact five fields')
    day = event['date']
    _need(type(day) is str, 'event date string')
    try:
        canonical = CalendarDate.fromisoformat(day).isoformat()
    except ValueError as exc:
        raise ValueError('canonical ISO date required') from exc
    _need(canonical == day and (previous is None or day > previous),
          'canonical dates must strictly increase')
    try:
        symbol_count = len(event['symbols'])
    except TypeError as exc:
        raise ValueError('sized symbols required') from exc
    _need(width is None or symbol_count == width, 'stable symbol width')
    names = _ordered(event['symbols'], symbol_count, 'symbols')
    _need(len(names) > 0 and all(type(s) is str and s and s.strip() == s
                               for s in names), 'nonempty trimmed symbols')
    _need(len(set(names)) == len(names), 'distinct symbols')
    flags = _ordered(event['eligible'], len(names), 'eligible')
    _need(all(type(f) is bool for f in flags), 'Python bool eligibility required')
    old = event['incumbent_choices']
    _need(type(old) in (list, tuple), 'incumbent list or tuple required')
    _need(all(type(k) is int and 0 <= k < len(names) and flags[k] for k in old),
          'incumbent requires eligible Python integer columns')
    _need(len(set(old)) == len(old), 'duplicate incumbent column')
    _need(len(old) == (0 if sum(flags) < 2 else min(3, sum(flags))),
          'incumbent count mismatch')
    original = [i for i, flag in enumerate(flags) if flag]
    return names, original, list(old), day


def _prices(event, original, width):
    columns = _ordered(event['prices253'], width, 'prices253 columns')
    read, errors = {}, []
    # Complete validation precedes momentum, including names later outside Top6.
    for column in original:
        try:
            values = _ordered(columns[column], 253, 'price column ' + str(column))
        except ValueError as exc:
            errors.append(str(exc))
            continue
        row = []
        for offset, value in enumerate(values):
            label = 'price[%d][%d]' % (column, offset)
            if not isinstance(value, Real) or isinstance(value, bool) or type(value).__name__ == 'bool_':
                errors.append(label + ': real nonboolean value required')
                continue
            try:
                binary64 = float(value)
            except (TypeError, ValueError, OverflowError):
                errors.append(label + ': binary64 conversion failed')
                continue
            if not isfinite(binary64) or binary64 <= 0:
                errors.append(label + ': finite positive value required')
                continue
            row.append(binary64.as_integer_ratio())
        if len(row) == 253:
            read[column] = row
    _need(not errors, 'required price validation: ' + '; '.join(errors))
    return read


def _statistics(group, gross, blocks):
    # Independent C construction: three annual endpoint ratios, not 36 factors.
    control = _product(gross[i] for i in group)
    means, gaps, block_products = [], [], []
    for block in range(12):
        triple = [blocks[i][block] for i in group]
        mean = _over(_plus(_plus(triple[0], triple[1]), triple[2]), (3, 1))
        product = _product(triple)
        gap = _minus(_cube(mean), product)
        _need(gap[0] >= 0, 'independent block AMGM failure')
        means.append(mean)
        gaps.append(gap)
        block_products.append(product)
    _need(_product(block_products) == control, 'independent telescoping failure')
    joint = _product(means)
    gap = _minus(_cube(joint), control)
    _need(gap[0] >= 0, 'independent aggregate AMGM failure')
    # Whole-product Decimal logs are independent of producer per-block float logs.
    with localcontext() as ctx:
        ctx.prec = 100
        c = _decimal_log(control) / 36
        j = _decimal_log(joint) / 12
        log_gap = _decimal_log(_over(_cube(joint), control)) / 36
    return {
        'schema': 'monthly_joint_growth271_basket_v1', 'block_count': 12,
        'constituent_annual_gross': [_wire(gross[i]) for i in group],
        'mean_block_gross': [_wire(m) for m in means],
        'control_product': _wire(control), 'joint_product': _wire(joint),
        'amgm_block_gaps': [_wire(g) for g in gaps], 'amgm_total_gap': _wire(gap),
        'control_log': float(c), 'joint_log': float(j),
        'direct_log_gap': float(j - c), 'log_gap_from_exact_ratios': float(log_gap),
        'log_displays_used_for_selection': False}, control, joint


def _reference(events, mode, n):
    _need(type(mode) is str and mode in ARMS, 'unknown arm')
    _need(not isinstance(events, (str, bytes, Mapping, set, frozenset)),
          'ordered indexable events required')
    try:
        count = len(events)
    except TypeError as exc:
        raise ValueError('events length required') from exc
    if n is None:
        n = count
    _need(type(n) is int and 0 <= n <= count, 'valid Python integer n required')
    counts = dict.fromkeys(_COUNT_NAMES, 0)
    counts.update(events=n, selection_calls=1)
    expected, choices, width, previous = [], [], None, None
    for position in range(n):
        event = events[position]
        names, original, old, day = _metadata(event, position, width, previous)
        width, previous = len(names), day
        partial = day == PARTIAL_DATE
        counts['initial_partial_events' if partial else 'formal_events'] += 1
        row = {
            'query_index': position, 'date': day, 'symbols': names,
            'initial_partial': partial, 'original_eligible_indices': original,
            'incumbent_choices': old, 'branch': None, 'measurements': [],
            'new_eligible_indices': [], 'excluded_momentum_indices': [],
            'shortlist_indices': [], 'trios': [], 'winning_trio_index': None,
            'tied_trio_indices': [], 'selected_indices': [], 'selected_symbols': [],
            'equal_weights': [], 'control_top3_equivalence_checked': False}
        if partial or mode == ARMS[0]:
            selected = old[:]
            row['branch'] = 'initial_partial' if partial else 'incumbent_copy'
            counts['incumbent_copy_events'] += 1
        else:
            prices = _prices(event, original, width)
            annual, blocks = {}, {}
            for i in original:
                annual[i] = _over(prices[i][-1], prices[i][0])
                boundaries = prices[i][::21]
                blocks[i] = [_over(boundaries[k + 1], boundaries[k]) for k in range(12)]
                _need(_product(blocks[i]) == annual[i], 'telescoping annual gross')
                row['measurements'].append({
                    'column_index': i, 'symbol': names[i], 'measurement': {
                        'schema': 'monthly_joint_growth271_measure_v1',
                        'price_count': 253, 'block_count': 12,
                        'annual_gross': _wire(annual[i]),
                        'momentum': _wire(_minus(annual[i], _ONE)),
                        'block_gross': [_wire(x) for x in blocks[i]]}})
            positive = [i for i in original if _compare(annual[i], _ONE) > 0]
            excluded = [i for i in original if _compare(annual[i], _ONE) <= 0]

            def strength(a, b):
                rank = _compare(annual[b], annual[a])
                return rank or ((names[a] > names[b]) - (names[a] < names[b]))

            short = sorted(positive, key=cmp_to_key(strength))[:6]
            row.update(new_eligible_indices=positive, excluded_momentum_indices=excluded,
                       shortlist_indices=short)
            counts['required_cells'] += len(original)
            counts['price_observations'] += 253 * len(original)
            counts['momentum_evaluations'] += len(original)
            counts['block_gross_evaluations'] += 12 * len(original)
            counts['positive_momentum_cells'] += len(positive)
            counts['excluded_momentum_cells'] += len(excluded)
            counts['shortlist_cells'] += len(short)
            counts['source_rankings'] += 1
            if len(short) < 2:
                selected = []
                row['branch'] = 'cash'
                counts['cash_events'] += 1
            elif len(short) == 2:
                selected = short[:]
                row['branch'] = 'two_names'
                counts['two_name_events'] += 1
            else:
                row['branch'] = 'optimized'
                counts['optimized_events'] += 1
                # Enumerate subsets as bit masks; sort by the specified symbol tuple.
                groups = [tuple(sorted((short[k] for k in range(len(short)) if mask & (1 << k)),
                                       key=lambda i: names[i]))
                          for mask in range(1 << len(short)) if bin(mask).count('1') == 3]
                groups.sort(key=lambda group: tuple(names[i] for i in group))
                objectives = []
                for group in groups:
                    stats, control, joint = _statistics(group, annual, blocks)
                    row['trios'].append({'indices': list(group),
                                         'symbols': [names[i] for i in group],
                                         'statistics': stats})
                    objectives.append(control if mode == ARMS[1] else joint)
                # Find all undominated objective values, then the first lexical tuple.
                best_value = sorted(objectives, key=cmp_to_key(_compare))[-1]
                tied = [i for i, objective in enumerate(objectives) if objective == best_value]
                winner = tied[0]
                selected = sorted(groups[winner], key=cmp_to_key(strength))
                if mode == ARMS[1]:
                    _need(selected == short[:3], 'C must equal Top3 even at exact ties')
                    row['control_top3_equivalence_checked'] = True
                row.update(winning_trio_index=winner, tied_trio_indices=tied)
                evaluated = len(groups)
                counts['trio_evaluations'] += evaluated
                counts['objective_comparisons'] += evaluated - 1
                counts['exact_amgm_block_checks'] += 12 * evaluated
                counts['exact_amgm_total_checks'] += evaluated
        row.update(selected_indices=selected, selected_symbols=[names[i] for i in selected],
                   equal_weights=[1.0 / len(selected)] * len(selected) if selected else [])
        expected.append(row)
        choices.append(selected[:])
    counts['symbols_per_event'] = width or 0
    return choices, {
        'schema': 'monthly_joint_growth271_selection_v1', 'mode': mode,
        'counts': counts, 'events': expected, 'source_files_authenticated': False,
        'original_incumbent_authenticated': False, 'official_clock_authenticated': False,
        'log_displays_used_for_selection': False, 'computes_strategy_returns': False,
        'computes_predictive_labels': False}


def verify_selection(events, mode, choices, audit, n=None):
    """Raise ValueError on malformed inputs/mismatched outputs; return checks.

    All exact rational strings, domains, exhaustive trios, ties, selected order,
    weights, bypass branches, and counts are independently reconstructed. Object
    key ordering is immaterial; sequence ordering and scalar types are checked.
    The four explicitly descriptive log fields allow only the documented display
    tolerance. No source or clock claim can become true through this verifier.
    """
    reference_choices, reference_audit = _reference(events, mode, n)
    compared = {'exact_scalar_checks': 0, 'log_display_checks': 0}

    def compare(actual, expected, path, display=False):
        if type(expected) is dict:
            _need(type(actual) is dict and set(actual) == set(expected), path + ': exact fields')
            for key, value in expected.items():
                compare(actual[key], value, path + '.' + key, key in _LOG_FIELDS)
        elif type(expected) is list:
            _need(type(actual) is list and len(actual) == len(expected), path + ': list length/type')
            for index, value in enumerate(expected):
                compare(actual[index], value, path + '[' + str(index) + ']')
        elif display:
            _need(type(actual) is float and isfinite(actual), path + ': finite float display')
            _need(isclose(actual, expected, rel_tol=LOG_RELATIVE_TOLERANCE,
                          abs_tol=LOG_ABSOLUTE_TOLERANCE), path + ': display mismatch')
            compared['log_display_checks'] += 1
        else:
            _need(type(actual) is type(expected) and actual == expected, path + ': exact mismatch')
            compared['exact_scalar_checks'] += 1

    compare(choices, reference_choices, 'choices')
    compare(audit, reference_audit, 'audit')
    return {
        'schema': 'monthly_joint_growth271_independent_v1', 'passed': True,
        'mode': mode, 'checked_events': len(reference_choices),
        'checked_required_cells': reference_audit['counts']['required_cells'],
        'checked_trios': reference_audit['counts']['trio_evaluations'],
        'exact_scalar_checks': compared['exact_scalar_checks'],
        'log_display_checks': compared['log_display_checks'],
        'display_absolute_tolerance': LOG_ABSOLUTE_TOLERANCE,
        'display_relative_tolerance': LOG_RELATIVE_TOLERANCE,
        'source_files_authenticated': False, 'original_incumbent_authenticated': False,
        'official_clock_authenticated': False, 'log_displays_used_for_selection': False,
        'computes_strategy_returns': False, 'computes_predictive_labels': False}
