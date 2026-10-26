"""Pure prospective common-scenario selector; no market data or model imports.

Both candidate arms replace the old locked-top2 episode selector with one common
portfolio objective. All old pre-score components stay frozen except the stated
weighted-core replacement. Callers must NOT apply the old episode selector again.
This module does not register an experiment or establish historical-data novelty.
"""
from itertools import combinations
import math
from numbers import Integral

import numpy as np


TOLERANCE = 1e-12
ARMS = ("mean_scenario", "worst_scenario")


def _numeric(value, name, ndim):
    raw = np.asarray(value)
    if raw.ndim != ndim or raw.dtype.kind not in "iuf":
        raise ValueError(name + " must be a numeric array of dimension " + str(ndim))
    arr = np.array(raw, dtype=float, copy=True)
    if np.isinf(arr).any():
        raise ValueError(name + " contains infinity")
    return arr


def _boolean(value, name, n, allow_scalar=False):
    raw = np.asarray(value)
    if allow_scalar and raw.ndim == 0 and raw.dtype.kind == "b":
        return np.full(n, bool(raw), dtype=bool)
    if raw.ndim != 1 or raw.shape != (n,) or raw.dtype.kind != "b":
        raise ValueError(name + " must be a boolean vector" + (" or boolean scalar" if allow_scalar else ""))
    return np.array(raw, dtype=bool, copy=True)


def select_portfolio(pre_score, original_selected, original_weighted_core,
                     core_weight, scenario_core_ranks, eligible, symbol_keys,
                     *, arm, history_available=True):
    """Select three assets jointly under a frozen common scenario matrix.

    Inputs use the same candidate column order. ``scenario_core_ranks`` has shape
    (scenario_count, candidate_count), with centered ranks in [-.5, .5]. Missing
    scenarios (None or empty) and NaN scenario values cause explicit fallback;
    infinity and malformed schemas raise ValueError. ``history_available`` is a
    bool or a candidate-length bool vector; only frozen candidate-set members
    require history. Candidate set = top eight eligible by pre_score/key, union
    original_selected. No current or future return information is accepted.

    Objective is mean or minimum, across common scenarios, of the three-asset
    average score. Scores replace, not supplement, the weighted core:
        pre_score + core_weight * scenario_core_rank - original_weighted_core.
    Objective ties within 1e-12 prefer the original set, then the highest sum of
    original pre_scores, then lexicographic sorted symbol keys. Active outputs
    are ordered by original pre_score descending, then stable symbol key.
    Fallback preserves the original input ordering exactly. Inputs are copied.
    """
    if arm not in ARMS:
        raise ValueError("arm must be mean_scenario or worst_scenario")
    if isinstance(core_weight, (bool, np.bool_)) or not isinstance(core_weight, (int, float, np.integer, np.floating)):
        raise ValueError("core_weight must be .50 or .52")
    weight = float(core_weight)
    if not math.isfinite(weight) or weight not in (.50, .52):
        raise ValueError("core_weight must be .50 or .52")
    scores = _numeric(pre_score, "pre_score", 1)
    n = len(scores)
    core = _numeric(original_weighted_core, "original_weighted_core", 1)
    if core.shape != (n,):
        raise ValueError("original_weighted_core length mismatch")
    if np.isnan(scores).any() or np.isnan(core).any():
        raise ValueError("original score and weighted core must be finite")
    mask = _boolean(eligible, "eligible", n)
    history = _boolean(history_available, "history_available", n, allow_scalar=True)
    if isinstance(symbol_keys, (str, bytes)):
        raise ValueError("symbol_keys must be a sequence of unique nonempty strings")
    try:
        keys = tuple(symbol_keys)
    except TypeError as exc:
        raise ValueError("symbol_keys must be a sequence") from exc
    if len(keys) != n or any(not isinstance(k, str) or not k for k in keys) or len(set(keys)) != n:
        raise ValueError("symbol_keys must be candidate-length unique nonempty strings")
    if isinstance(original_selected, (str, bytes)):
        raise ValueError("original_selected must contain integer indices")
    try:
        selected = tuple(original_selected)
    except TypeError as exc:
        raise ValueError("original_selected must be a sequence") from exc
    if len(selected) > 3 or any(isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral) for i in selected):
        raise ValueError("original_selected must contain at most three integer indices")
    selected = tuple(int(i) for i in selected)
    if len(set(selected)) != len(selected) or any(i < 0 or i >= n for i in selected):
        raise ValueError("original_selected indices must be unique and in range")
    if any(not mask[i] for i in selected):
        raise ValueError("original_selected must be eligible")

    order_key = lambda i: (-float(scores[i]), keys[i])
    top = sorted((i for i in range(n) if mask[i]), key=order_key)[:8]
    candidates = tuple(sorted(set(top).union(selected), key=order_key))
    if scenario_core_ranks is None:
        scenarios = None
    else:
        raw = np.asarray(scenario_core_ranks)
        # [] is a convenient spelling of zero rows; nonempty vectors are errors.
        if raw.ndim == 1 and raw.size == 0:
            raw = np.empty((0, n), dtype=float)
        scenarios = _numeric(raw, "scenario_core_ranks", 2)
        if scenarios.shape[1] != n:
            raise ValueError("scenario_core_ranks candidate length mismatch")
        finite = scenarios[np.isfinite(scenarios)]
        if np.any(finite < -.5) or np.any(finite > .5):
            raise ValueError("scenario_core_ranks must be centered ranks in [-.5, .5]")

    scenario_count = 0 if scenarios is None else int(scenarios.shape[0])

    def result(chosen, objective, fallback, reason):
        return {"arm": arm, "selected_indices": list(chosen),
                "selected_keys": [keys[i] for i in chosen],
                "candidate_indices": list(candidates),
                "candidate_keys": [keys[i] for i in candidates],
                "scenario_count": scenario_count,
                "objective": objective, "fallback": fallback, "reason": reason}

    if len(selected) < 3:
        return result(selected, None, True, "original_fewer_than_three")
    if scenarios is None:
        return result(selected, None, True, "missing_scenarios")
    if scenario_count == 0:
        return result(selected, None, True, "empty_scenarios")
    if np.isnan(scenarios).any():
        return result(selected, None, True, "nan_scenario_values")
    if not all(history[i] for i in candidates):
        return result(selected, None, True, "candidate_history_unavailable")
    weighted = weight * scenarios
    if np.all(np.abs(weighted - core[None, :]) <= TOLERANCE):
        return result(selected, None, True, "unchanged_weighted_core")
    adjusted = scores[None, :] + weighted - core[None, :]
    if not np.isfinite(adjusted).all():
        raise ValueError("adjusted scenario scores overflowed")
    entries = []
    for combo in combinations(candidates, 3):
        averages = [math.fsum(float(row[i]) for i in combo) / 3. for row in adjusted]
        objective = (math.fsum(averages) / scenario_count
                     if arm == "mean_scenario" else min(averages))
        original_sum = math.fsum(float(scores[i]) for i in combo)
        entries.append((objective, original_sum, tuple(sorted(keys[i] for i in combo)), combo))
    best = max(e[0] for e in entries)
    tied = [e for e in entries if best - e[0] <= TOLERANCE]
    original_set = frozenset(selected)
    original_tie = [e for e in tied if frozenset(e[3]) == original_set]
    chosen = original_tie[0] if original_tie else min(tied, key=lambda e: (-e[1], e[2]))
    return result(tuple(sorted(chosen[3], key=order_key)), float(chosen[0]), False,
                  "selected_by_" + arm)
