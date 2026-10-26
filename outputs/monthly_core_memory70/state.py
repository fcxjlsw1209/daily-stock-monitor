"""Fixed monthly volatility hysteresis; inspect only the requested prefix.

True denotes the calm core flavor. This helper does not select securities,
adjust weights, construct market data, or change any other strategy state.
"""
from numbers import Real

import numpy as np

INITIAL_THRESHOLD = .10
ENTER_CALM_THRESHOLD = .09
EXIT_CALM_THRESHOLD = .11


def hysteresis(vol, n=None):
    """Return ``(states, audit)`` for the first n chronological signal values.

    The first finite observation initializes calm iff volatility is strictly
    below .10. A missing first value initializes False. Subsequently <=.09
    enters calm, >=.11 exits calm, and values strictly inside the band or NaN
    preserve the previous state. Missingness is counted separately from band
    holds. An initialization is not a transition. These thresholds are fixed;
    no tuning argument exists. No conversion or value inspection reaches past
    ``vol[:n]``. The caller is responsible for chronological signal order.
    """
    if isinstance(vol, (str, bytes, dict, set, frozenset)):
        raise ValueError('vol must be a one-dimensional sliceable numeric sequence')
    try:
        length = len(vol)
    except (TypeError, AttributeError) as exc:
        raise ValueError('vol must be a one-dimensional sliceable numeric sequence') from exc
    if n is None:
        count = length
    elif (isinstance(n, (bool, np.bool_)) or
          not isinstance(n, (int, np.integer)) or n < 0 or n > length):
        raise ValueError('n must be a nonboolean integer between zero and len(vol)')
    else:
        count = int(n)
    # Slicing must precede conversion. An object in the unqueried suffix may be
    # deliberately invalid or raise when converted; it is not today's input.
    try:
        prefix = np.asarray(vol[:count], dtype=object)
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError('vol must be a one-dimensional sliceable numeric sequence') from exc
    if prefix.shape != (count,):
        raise ValueError('vol prefix must be one-dimensional and align with n')
    if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, Real) for x in prefix):
        raise ValueError('vol prefix must contain real numeric values or NaN')
    values = prefix.astype(float)
    if np.isinf(values).any() or (values < 0.).any():
        raise ValueError('vol prefix cannot contain infinity or negative volatility')
    states = np.zeros(count, dtype=bool)
    audit = dict(
        count=count, observed=int(np.isfinite(values).sum()), missing=int(np.isnan(values).sum()),
        transitions=0, band_holds=0, missing_holds=0,
        enter_threshold_observations=0, exit_threshold_observations=0,
        initial_missing=bool(count and np.isnan(values[0])),
        initial_state=None, final_state=None,
        initial_threshold=INITIAL_THRESHOLD, enter_calm_threshold=ENTER_CALM_THRESHOLD,
        exit_calm_threshold=EXIT_CALM_THRESHOLD, thresholds_fixed=True,
        observations='signal-order prefix only',
        initialization='first value <0.10; missing first value=False',
        later_rule='<=0.09 True; >=0.11 False; interior band or NaN retains prior state',
        count_semantics='observed/missing include initial; transitions and holds exclude initial')
    if count == 0:
        return states, audit
    states[0] = bool(values[0] < INITIAL_THRESHOLD)
    audit['initial_state'] = bool(states[0])
    for i in range(1, count):
        previous = bool(states[i-1])
        value = values[i]
        if np.isnan(value):
            current = previous
            audit['missing_holds'] += 1
        elif value <= ENTER_CALM_THRESHOLD:
            current = True
            audit['enter_threshold_observations'] += 1
        elif value >= EXIT_CALM_THRESHOLD:
            current = False
            audit['exit_threshold_observations'] += 1
        else:
            current = previous
            audit['band_holds'] += 1
        states[i] = current
        audit['transitions'] += int(current != previous)
    audit['final_state'] = bool(states[-1])
    return states, audit
