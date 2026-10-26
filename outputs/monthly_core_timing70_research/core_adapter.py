"""Replace only the weighted long core, then reuse the frozen path selector.

No research-driver imports, data loading, fitting, or return calculations. The
dependency below is the fixed, pure path adapter; callers hash this source path
alongside this module. Core ranks have already been formed and normalized by
the caller. Blending happens between those ranks, never before normalization.
"""
import copy
import importlib.util
from pathlib import Path

import numpy as np


PATH_ADAPTER_PATH = Path(__file__).resolve().parent.parent / "monthly_path70_research" / "incumbent_adapter.py"
_SPEC = importlib.util.spec_from_file_location("core_timing70_frozen_path_adapter", PATH_ADAPTER_PATH)
path_adapter = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(path_adapter)


def _blend(value):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError("core blend must be exactly 0, .5, or 1")
    value = float(value)
    if value not in (0., .5, 1.):
        raise ValueError("core blend must be exactly 0, .5, or 1")
    return value


def _copy_with_prefix(values, prefix):
    """Copy the full shape without converting or validating future values."""
    result = np.array(values, copy=True)
    # Integer inputs must not truncate a fractional core delta. Object storage
    # also leaves arbitrary unused suffix values uninterpreted.
    if result.dtype.kind != "f":
        result = result.astype(object)
    result[:len(prefix)] = prefix
    return result


def _canonical_weights(a, n):
    """Keep the frozen helper's full-array float conversion within the prefix."""
    baseline = np.asarray(a["baseline"])
    if (baseline.ndim != 2 or baseline.shape[1] == 0
            or isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer))
            or not 0 <= n <= len(baseline)):
        raise ValueError("n must be a valid monthly prefix of a nonempty-width baseline")
    weights = np.asarray(a["core_weight"])
    if weights.shape == (len(baseline),):
        weights = weights[:, None]
    if weights.shape not in ((len(baseline), 1), baseline.shape):
        raise ValueError("core_weight must be [months], [months, 1], or [months, candidates]")
    canonical = np.full(baseline.shape, .5)
    canonical[:n] = weights[:n]
    return canonical


def score_and_choose(base, blend, new_core_rank, a, m, n, *, pre_score=None,
                     effects=None, episode_policy=None, boundary_selector=None):
    """Return (choices, scores) using a convex replacement of the long core.

    ``a`` has the frozen path adapter's baseline, already-weighted core,
    core_weight, eligible and incumbent states. All candidate matrices have
    baseline's FULL [months, candidates] shape. Core weights may also be
    [months] or [months, 1]. Only values in the prefix ``[:n]`` are examined.

    On finite new ranks, delta = blend * (core_weight * new_rank - old_core).
    This same delta enters copies of baseline, weighted core and real
    pre-boundary score. Core weights and every auxiliary component stay fixed.
    Ranks are bounded by [-.5, .5]; a finite zero is a valid neutral rank.
    Nonfinite ranks have zero delta. A zero blend or wholly missing ranks
    reproduce the original scores and choices exactly. A missing cell's
    pre-score stays unchanged; its eventual boundary membership can still
    change when other candidates' cores change.

    Reuse the path adapter with zero hand budget and entirely missing hand
    factors. The real episode selector receives ORIGINAL support inputs to
    avoid floating-point cancellation from recomputing (baseline-core) after
    adding delta to both. It still anchors its candidates to the MODIFIED
    pre-score. Its returned choices must never be globally re-ranked.
    """
    blend = _blend(blend)
    canonical = _canonical_weights(a, n)
    source = dict(a, core_weight=canonical)
    _, shape, baseline, core, cw, rank, finite, _ = path_adapter._inputs(
        base, 0., new_core_rank, source, n)
    delta = np.zeros((n, shape[1]), dtype=float)
    changed = finite & (blend != 0.)
    delta[changed] = blend * (cw[changed] * rank[changed] - core[changed])
    nonzero = delta != 0.
    adjusted_baseline = baseline.copy()
    adjusted_core = core.copy()
    adjusted_baseline[nonzero] += delta[nonzero]
    adjusted_core[nonzero] += delta[nonzero]

    updated = copy.deepcopy(a)
    updated["baseline"] = _copy_with_prefix(a["baseline"], adjusted_baseline)
    updated["core"] = _copy_with_prefix(a["core"], adjusted_core)
    updated["core_weight"] = canonical
    adjusted_pre = None
    if pre_score is not None:
        prefix = path_adapter._matrix(pre_score, shape, n, "pre_score").copy()
        prefix[nonzero] += delta[nonzero]
        adjusted_pre = _copy_with_prefix(pre_score, prefix)
    if effects is not None:
        path_adapter._matrix(effects, shape, n, "effects")

    selector = boundary_selector
    if base == "incumbent66" and callable(boundary_selector):
        def selector(score, eligible, symbols, episode_effects, original, policy):
            original = dict(original)
            original.update(baseline=baseline.copy(), core=core.copy(),
                            core_weight=cw.copy())
            return boundary_selector(score, eligible, symbols, episode_effects,
                                     original, policy)

    return path_adapter.score_and_choose(
        base, 0., np.full(shape, np.nan), updated, m, n,
        pre_score=adjusted_pre, effects=effects, episode_policy=episode_policy,
        boundary_selector=selector)
