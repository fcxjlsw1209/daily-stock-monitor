"""Pure hand-factor support replacement before the frozen episode selector.

No imports from strategy drivers, file I/O, fitting or return calculation.
The caller injects the incumbent's real pre-boundary scores, episode effects,
frozen policy and original select_boundary function. Its existing selectors and
budgets are reused, while new hand-factor scores may change all three seats.
"""
import numpy as np


BASES = ("original65", "incumbent66")
HAND_LIMIT = .10
OLD_LIMIT = .10
EPISODE_POLICY = dict(budget=.05, ceiling=5, gap=.02)


def _budget(value):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError("hand budget must be a finite scalar between zero and .10")
    value = float(value)
    if not np.isfinite(value) or not 0. <= value <= HAND_LIMIT:
        raise ValueError("hand budget must be a finite scalar between zero and .10")
    return value


def _matrix(values, full_shape, n, name, boolean=False):
    value = np.asarray(values)
    if value.shape != full_shape or (boolean and value.dtype.kind != "b"):
        raise ValueError(name + " must exactly match the full [months, candidates] shape"
                         + (" and contain booleans" if boolean else ""))
    return value[:n].astype(bool if boolean else float, copy=False)


def _core_weight(value, full_shape, n):
    value = np.asarray(value, float)
    if value.shape == (full_shape[0],):
        value = value[:, None]
    if value.shape not in ((full_shape[0], 1), full_shape):
        raise ValueError("core_weight must be [months], [months, 1], or [months, candidates]")
    result = np.broadcast_to(value[:n], (n, full_shape[1]))
    if not np.isfinite(result).all() or not np.isin(result, [.50, .52]).all():
        raise ValueError("original core weights must remain exactly .50 or .52")
    return result


def _inputs(base, budget, factor, a, n):
    if base not in BASES:
        raise ValueError("base must be original65 or incumbent66")
    budget = _budget(budget)
    baseline = np.asarray(a["baseline"])
    if (baseline.ndim != 2 or baseline.shape[1] == 0
            or isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer))
            or not 0 <= n <= len(baseline)):
        raise ValueError("n must be a valid monthly prefix of a nonempty-width baseline")
    shape = baseline.shape
    baseline = _matrix(baseline, shape, n, "baseline")
    core = _matrix(a["core"], shape, n, "core")
    cw = _core_weight(a["core_weight"], shape, n)
    factor = _matrix(factor, shape, n, "factor")
    if not np.isfinite(baseline).all() or not np.isfinite(core).all():
        raise ValueError("original baseline and weighted core must be finite")
    support = (baseline-core)/(1-cw)
    if (np.abs(support) > .5+1e-12).any():
        raise ValueError("original auxiliary support must be a bounded rank in [-.5, .5]")
    valid = np.isfinite(factor)
    if (np.abs(factor[valid]) > .5).any():
        raise ValueError("finite hand factors must be bounded ranks in [-.5, .5]")
    return budget, shape, baseline, core, cw, factor, valid, support


def _state(a, name, full_rows, n):
    state = np.asarray(a[name])
    if state.shape != (full_rows,) or state.dtype.kind != "b":
        raise ValueError(name + " must be an exact [months] boolean vector")
    return state[:n]


def budget_weights(base, budget, factor, a, n, *, episode_mask=None):
    """Return coefficient matrices for a convex decomposition of each score.

    Keys: core, original_support, prior_residual, prior_ml, episode, hand.
    ``core`` is the original core coefficient, not its already weighted score.
    For incumbent66, pass the actual [n, candidates] episode candidate mask for
    exact accounting. If omitted, conservatively reserve the episode budget on
    every candidate. The mask describes the boundary from MODIFIED pre-scores.

    The old residual+ML+episode slots total <= .10; the separately declared
    non-ML hand slot totals <= .10. Combined replacement may reach .20. Missing
    hand factors receive zero hand weight. Remaining original support is >= .28
    for the original .50/.52 core. This helper does not choose a portfolio.
    """
    budget, full_shape, _, _, cw, _, valid, _ = _inputs(base, budget, factor, a, n)
    shape = cw.shape
    residual = np.zeros(shape)
    ml = np.zeros(shape)
    episode = np.zeros(shape)
    if base == "incumbent66":
        calm = _state(a, "state_calm10", full_shape[0], n)
        weak = _state(a, "state_weak_narrow", full_shape[0], n)
        residual = np.broadcast_to(.025*calm[:, None], shape)
        ml = np.broadcast_to(.05*weak[:, None], shape)
        mask = np.ones(shape, bool) if episode_mask is None else np.asarray(episode_mask)
        if mask.shape != shape or mask.dtype.kind != "b":
            raise ValueError("episode_mask must be a boolean [n, candidates] matrix")
        episode = np.minimum(.05, np.maximum(0., OLD_LIMIT-residual-ml))*mask
    elif episode_mask is not None:
        raise ValueError("original65 has no episode budget")
    hand = budget*valid
    old_total = residual+ml+episode
    remaining = 1-cw-old_total-hand
    if ((old_total > OLD_LIMIT+1e-12).any()
            or (old_total+hand > OLD_LIMIT+HAND_LIMIT+1e-12).any()
            or (remaining < .28-1e-12).any()):
        raise ValueError("auxiliary budget exceeds its registered limits")
    return dict(core=cw.copy(), original_support=remaining,
                prior_residual=residual.copy(), prior_ml=ml.copy(),
                episode=episode, hand=hand)


def score_and_choose(base, budget, factor, a, m, n, *, pre_score=None,
                     effects=None, episode_policy=None, boundary_selector=None):
    """Return (choices, scores) after a hand-factor support replacement.

    ``a`` supplies baseline, core (already weighted), core_weight and eligible.
    All candidate matrices must match baseline's full shape; only prefix values
    are used. Core weights support [months], [months,1] and exact full shape.
    ``m['symbols']`` supplies corresponding per-month candidate symbols.

    original65: score = baseline + budget*(factor-original_support).
    incumbent66: apply that same delta to the REAL pre-boundary score, then call
    the injected frozen boundary_selector with the ORIGINAL episode effects,
    ORIGINAL support inputs and fixed policy. Its candidate anchors therefore
    use modified pre-scores. Returned post-boundary scores MUST NOT be globally
    reranked. No updated score is fed back as original support.

    Missing/nonfinite factors apply no hand correction. A factor of zero is a
    valid neutral rank and replaces support; it is not a missing value. A zero
    budget, or wholly unavailable hand factors, exactly reproduces the selected
    base. Eligibility and three/two/cash rules remain unchanged.
    """
    budget, shape, baseline, core, cw, factor, valid, support = _inputs(base, budget, factor, a, n)
    good = _matrix(a["eligible"], shape, n, "eligible", boolean=True)
    symbols = m["symbols"][:n]
    if len(symbols) != n or any(len(ss) != shape[1] for ss in symbols):
        raise ValueError("monthly symbols must align exactly with candidate matrices")
    if any(any(not isinstance(s, str) or not s for s in ss) or len(set(ss)) != len(ss) for ss in symbols):
        raise ValueError("monthly symbols must be unique nonempty strings")
    # Reserve the maximum possible episode exposure before knowing its candidates.
    # This checks the stronger all-candidate bound; actual boundary exposure is no larger.
    if base == "incumbent66":
        if not callable(boundary_selector) or episode_policy is None or pre_score is None or effects is None:
            raise ValueError("incumbent66 requires its real pre-score, effects, frozen policy and selector")
        if any(episode_policy.get(k) != v for k, v in EPISODE_POLICY.items()):
            raise ValueError("incumbent episode policy must retain budget .05, ceiling 5 and gap .02")
        score = _matrix(pre_score, shape, n, "pre_score").copy()
        episode_effect = _matrix(effects, shape, n, "effects")
        if not np.isfinite(score).all() or not np.isfinite(episode_effect).all():
            raise ValueError("incumbent pre-scores and episode effects must be finite")
        calm = _state(a, "state_calm10", shape[0], n)
        weak = _state(a, "state_weak_narrow", shape[0], n)
        old_budget = .025*calm[:, None]+.05*weak[:, None]
        episode_budget = np.minimum(.05, np.maximum(0., OLD_LIMIT-old_budget))
        if (1-cw-old_budget-episode_budget-budget*valid < .28-1e-12).any():
            raise ValueError("hand budget cannot be funded from remaining original support")
    else:
        score = baseline.copy()
    changed = valid & (budget != 0.)
    delta = np.zeros(score.shape)
    delta[changed] = budget*(factor[changed]-support[changed])
    nonzero = delta != 0.
    score[nonzero] += delta[nonzero]
    if base == "original65":
        ranks = [sorted(np.flatnonzero(g).tolist(), key=lambda j: (-row[j], ss[j]))
                 for row, g, ss in zip(score, good, symbols)]
        return [rank[:3] if len(rank) >= 2 else [] for rank in ranks], score

    # Canonicalizing core_weight avoids NumPy's wrong-axis broadcast for [months].
    # These are copies of the original support/state data, never modified pre-score.
    original = dict(baseline=baseline.copy(), core=core.copy(), core_weight=cw.copy(),
                    state_calm10=calm.copy(), state_weak_narrow=weak.copy())
    return boundary_selector(score, good.copy(), [list(ss) for ss in symbols],
                             episode_effect.copy(), original, dict(episode_policy))
