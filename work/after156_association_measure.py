"""Pure old-window return association, with no source-admission claims.

The caller must pass its independently reviewed research-eligibility gate
before reading actual prices. This module accepts a previously constructed
partition; it does not pretend an F-only label is an exact acceptance clock.
"""
import math
from numbers import Real

NAMES = ('whole_formation_beta_residual_momentum', 'filing_window_density',
         'filing_vs_nonfiling_beta_residual_contrast')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def total(values):
    result = math.fsum(values)
    if not math.isfinite(result):
        raise ArithmeticError('Nonfinite sum')
    return result


def missing(reason, attempted=0):
    return dict(available=False, reason=reason, beta=None,
        hands=dict.fromkeys(NAMES, float('nan')), raw=dict.fromkeys(NAMES, None),
        nuisance_beta_estimates_attempted=attempted,
        nuisance_beta_estimates_completed=0, new_predictive_model_fits=0)


def log_return(a, b):
    ratio = b / a
    value = math.log(ratio) if 0 < ratio < math.inf else math.log(b) - math.log(a)
    if not math.isfinite(value):
        raise ArithmeticError('Nonfinite log return')
    return value


def measure(stock_prices, market_prices, labels):
    """Exactly463 prices P[t-504:t-42] and210 formation labels.

    A prefix of252 log returns estimates one centered OLS slope. The following
    210 returns share that slope and partition. No label, dose or price window
    is selected here. Partial price histories fail, never compress the dates.
    """
    require(len(stock_prices) == len(market_prices) == 463,
            'Exactly463 aligned prices required')
    require(len(labels) == 210 and set(labels) <= set('EBN'), 'Fixed partition required')
    ne, nn = labels.count('E'), labels.count('N')
    require(ne >= 38 and nn >= 80, 'Partition day minima required')
    # The separate metadata gate additionally proves two full original bundles.
    for i, (a, b) in enumerate(zip(stock_prices, market_prices)):
        require(all(isinstance(v, Real) and not isinstance(v, bool) for v in (a, b)),
                'Prices must be real scalars')
        if not all(math.isfinite(v) and v > 0 for v in (a, b)):
            return missing('invalid_required_'+('beta' if i <= 252 else 'formation')+'_price')
    y = [log_return(a, b) for a, b in zip(stock_prices, stock_prices[1:])]
    x = [log_return(a, b) for a, b in zip(market_prices, market_prices[1:])]
    mx, my = total(x[:252])/252, total(y[:252])/252
    xc = [v-mx for v in x[:252]]
    denominator = total(v*v for v in xc)
    if denominator == 0:
        return missing('zero_preformation_market_variance', 1)
    beta = total(a*(b-my) for a, b in zip(xc, y[:252])) / denominator
    if not math.isfinite(beta):
        raise ArithmeticError('Nonfinite beta')
    residual = [a-beta*b for a, b in zip(y[252:], x[252:])]
    if not all(math.isfinite(v) for v in residual):
        raise ArithmeticError('Nonfinite residual')
    whole = 21*(total(residual)/210)
    contrast = 21*(total(v for v, lab in zip(residual, labels) if lab == 'E')/ne
                   - total(v for v, lab in zip(residual, labels) if lab == 'N')/nn)
    density = ne/(ne+nn)
    def bounded(x):
        if not math.isfinite(x):
            raise ArithmeticError('Nonfinite raw signal')
        return x/(1+abs(x))
    hands = dict(zip(NAMES, (bounded(whole/.05), density, bounded(contrast/.05))))
    require(all(math.isfinite(v) and abs(v) <= 1 for v in hands.values()), 'Bounded signal')
    return dict(available=True, reason=None, beta=beta,
        hands=hands, raw=dict(zip(NAMES, (whole, density, contrast))),
        nuisance_beta_estimates_attempted=1, nuisance_beta_estimates_completed=1,
        new_predictive_model_fits=0)
