"""Pure binary64 fixed-unit trio statistics; no source I/O or strategy calls.

Input rows are three distinct securities identified by the caller; equal price
paths are valid. The caller also owns official-date alignment and lookback length.
Diagnostic tolerances never change statistics, availability, ranks, or ties.
"""
from collections.abc import Mapping
import math
from numbers import Real

SCHEMA = 'monthly_portfolio_path247_math_v1'
DIAGNOSTIC_ATOL = 2e-10
DIAGNOSTIC_RTOL = 2e-10
NAV_ATOL = 2e-12
NAV_RTOL = 2e-12
PAIR_INDICES = ((0, 1), (0, 2), (1, 2))


def _finite(value, label):
    if not math.isfinite(value):
        raise ArithmeticError('Nonfinite derived ' + label)
    return value


def _sum(values, label):
    try:
        result = math.fsum(values)
    except (OverflowError, ValueError) as exc:
        raise ArithmeticError('Invalid derived sum: ' + label) from exc
    return _finite(result, label)


def _divide(numerator, denominator, label):
    try:
        result = numerator / denominator
    except (OverflowError, ZeroDivisionError) as exc:
        raise ArithmeticError('Invalid derived division: ' + label) from exc
    _finite(result, label)
    if numerator != 0.0 and result == 0.0:
        raise ArithmeticError('Underflow in derived division: ' + label)
    return result


def _square(value, label):
    result = _finite(value * value, label)
    if value != 0.0 and result == 0.0:
        raise ArithmeticError('Underflow in centered square: ' + label)
    return result


def _product(left, right, label):
    result = _finite(left * right, label)
    if left != 0.0 and right != 0.0 and result == 0.0:
        raise ArithmeticError('Underflow in derived product: ' + label)
    return result


def _difference(left, right, label):
    return _finite(left - right, label)


def _tolerance(left, right, absolute, relative):
    return _finite(absolute + relative * max(abs(left), abs(right)), 'diagnostic tolerance')


def _close(left, right, absolute, relative, label):
    error = abs(_difference(left, right, label + ' difference'))
    tolerance = _tolerance(left, right, absolute, relative)
    if error > tolerance:
        raise ArithmeticError('Diagnostic failed: ' + label)
    return error


def _prices(paths):
    if isinstance(paths, (str, bytes, Mapping)):
        raise ValueError('Exactly three price series required')
    try:
        if len(paths) != 3:
            raise ValueError('Exactly three price series required')
        rows = list(paths)
        lengths = [len(row) for row in rows]
    except TypeError as exc:
        raise ValueError('Three sized price series required') from exc
    if lengths[0] < 3 or any(n != lengths[0] for n in lengths):
        raise ValueError('Common price length must be at least three')
    result = []
    for row in rows:
        if isinstance(row, (str, bytes, Mapping)):
            raise ValueError('Each price series must be a numeric sequence')
        converted = []
        for value in row:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError('Prices must be real nonboolean numbers')
            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError('Price cannot be represented as finite binary64') from exc
            if not math.isfinite(number) or number <= 0.0:
                raise ValueError('Prices must be positive finite binary64 values')
            converted.append(number)
        result.append(converted)
    return result


def measure(paths):
    """Return plain auditable statistics for one complete 3-by-L price array.

    Source shape/type/value errors raise ValueError. Derived overflow, false-zero
    underflow and failed algebra diagnostics raise ArithmeticError. Genuine zero
    variance is common unavailable support with both ratios None, not an error.

    Canonical order: row-major normalization; three prior normalized values in
    row order; per-row normalized differences; fsum contributions; two-pass
    centered sample sums. Diagonal is sum(column square sums)/(T-1), not the sum
    of independently rounded column variances. Pair diagnostics use i<j order.
    """
    prices = _prices(paths)
    length = len(prices[0])
    periods = length - 1
    normalized = [[_divide(p, row[0], 'normalized price') for p in row] for row in prices]
    if any(p <= 0.0 for row in normalized for p in row):
        raise ArithmeticError('Nonpositive normalized price')
    totals = [_sum((normalized[i][k] for i in range(3)), 'normalized total')
              for k in range(length)]
    wealth = [_divide(total, 3.0, 'fixed-unit portfolio value') for total in totals]
    contributions = [[], [], []]
    returns = []
    direct_nav = []
    nav_errors = []
    for k in range(1, length):
        denominator = totals[k - 1]
        for i in range(3):
            change = _difference(normalized[i][k], normalized[i][k - 1], 'normalized increment')
            contributions[i].append(_divide(change, denominator, 'daily contribution'))
        ret = _sum((contributions[i][-1] for i in range(3)), 'portfolio return')
        direct = _difference(_divide(wealth[k], wealth[k - 1], 'direct NAV ratio'), 1.0,
                             'direct NAV return')
        nav_errors.append(_close(ret, direct, NAV_ATOL, NAV_RTOL, 'direct NAV identity'))
        returns.append(ret)
        direct_nav.append(direct)
    total_return_sum = _sum(returns, 'portfolio return sum')
    mean = _divide(total_return_sum, float(periods), 'portfolio return mean')
    centered = [_difference(r, mean, 'centered portfolio return') for r in returns]
    joint_ss = _sum((_square(d, 'portfolio centered square') for d in centered), 'joint square sum')
    contribution_means = [_divide(_sum(row, 'contribution sum'), float(periods),
                                  'contribution mean') for row in contributions]
    centered_columns = [[_difference(c, contribution_means[i], 'centered contribution') for c in row]
                        for i, row in enumerate(contributions)]
    column_ss = [_sum((_square(d, 'contribution centered square') for d in row),
                      'contribution square sum') for row in centered_columns]
    diagonal_ss = _sum(column_ss, 'diagonal square sum')
    divisor = float(periods - 1)
    joint_variance = _divide(joint_ss, divisor, 'joint sample variance')
    diagonal_variance = _divide(diagonal_ss, divisor, 'diagonal sample variance')
    if joint_variance < 0.0 or diagonal_variance < 0.0:
        raise ArithmeticError('Negative sample variance')
    pair_ss = [_sum((_product(a, b, 'centered pair product')
                     for a, b in zip(centered_columns[i], centered_columns[j])),
                    'centered pair sum') for i, j in PAIR_INDICES]
    covariances = [_divide(ss, divisor, 'pair sample covariance') for ss in pair_ss]
    cross = _divide(_product(2.0, _sum(pair_ss, 'pair square sums'), 'twice pair sum'),
                    divisor, 'cross covariance')
    observed_cross = _difference(joint_variance, diagonal_variance, 'joint minus diagonal')
    cross_error = _close(observed_cross, cross, DIAGNOSTIC_ATOL, DIAGNOSTIC_RTOL,
                         'covariance decomposition')
    upper = _product(3.0, diagonal_variance, 'three times diagonal variance')
    upper_tolerance = _tolerance(joint_variance, upper, DIAGNOSTIC_ATOL, DIAGNOSTIC_RTOL)
    if _difference(joint_variance, upper, 'variance upper bound difference') > upper_tolerance:
        raise ArithmeticError('Diagnostic failed: joint variance exceeds three times diagonal')
    joint_sd = _finite(math.sqrt(joint_variance), 'joint standard deviation')
    diagonal_sd = _finite(math.sqrt(diagonal_variance), 'diagonal standard deviation')
    available = joint_variance > 0.0 and diagonal_variance > 0.0
    joint_ratio = _divide(mean, joint_sd, 'joint ratio') if available else None
    diagonal_ratio = _divide(mean, diagonal_sd, 'diagonal ratio') if available else None
    return dict(schema=SCHEMA, price_count=length, return_count=periods,
                available=available, reason='available' if available else 'zero_variance',
                normalized_prices=normalized, portfolio_values=wealth,
                prior_denominators=totals[:-1], contributions=contributions,
                portfolio_returns=returns, direct_nav_returns=direct_nav,
                mean_return=mean, joint_square_sum=joint_ss,
                contribution_means=contribution_means, contribution_square_sums=column_ss,
                diagonal_square_sum=diagonal_ss, joint_variance=joint_variance,
                diagonal_variance=diagonal_variance, joint_sd=joint_sd, diagonal_sd=diagonal_sd,
                joint_ratio=joint_ratio, diagonal_ratio=diagonal_ratio,
                pair_covariances=[dict(indices=list(pair), square_sum=ss, covariance=cov)
                                  for pair, ss, cov in zip(PAIR_INDICES, pair_ss, covariances)],
                cross_covariance=cross,
                diagnostics=dict(nav_identity_atol=NAV_ATOL, nav_identity_rtol=NAV_RTOL,
                                 covariance_atol=DIAGNOSTIC_ATOL, covariance_rtol=DIAGNOSTIC_RTOL,
                                 nav_max_absolute_error=max(nav_errors),
                                 covariance_absolute_error=cross_error,
                                 joint_minus_diagonal=observed_cross,
                                 three_times_diagonal=upper,
                                 nav_identity_passed=True, covariance_identity_passed=True,
                                 variance_upper_bound_passed=True))
