"""Fixed bounded support composite within the original local third-seat cohort.

Pure prefix scoring: no feature computation, fitting, prices or file outputs.
The caller supplies the frozen original episode selector and its policy.
"""
import importlib.util
from pathlib import Path

import numpy as np


SOURCE_PLACEMENT_PATH = Path(__file__).resolve().parent.parent / 'monthly_ordinal_complexity70' / 'placement.py'
_SPEC = importlib.util.spec_from_file_location('preserve70_frozen_placement', SOURCE_PLACEMENT_PATH)
source_placement = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(source_placement)
path_adapter = source_placement.path_adapter
HAND_BUDGET = .025


def score_and_choose(pre, hand_rank, a, meta, n, *, policy, boundary_selector):
    """Return choices/scores with a fixed .025 bounded composite slot.

    Lock eligible top2 and rank3..5/gap .02 using the caller's ACTUAL local
    pre-score. Run the original episode selector exactly once with original
    support inputs, then change only nonzero finite hand ranks in that cohort:

        S = (original_baseline - original_weighted_core)/(1-core_weight)
        H = clip(S+F, -.5, .5)
        delta = .025*(H-S)

    A zero or missing F explicitly bypasses clipping, preserving original
    output bytes even when S has tolerated boundary roundoff. No second rank
    normalization, episode call or global selection is allowed. A hand-free
    month (including only zero deltas) returns the original choices exactly.
    S and H have the same bounded auxiliary-rank units. The composite replaces
    .025 of original support, preserving the .50/.52 core; after reserving the
    conservative .10 total existing model/scope/episode weight, support left
    is at least .355. All unchanged models and episode effects retain funding.
    """
    source = source_placement._prefix_source(a, n)
    _, shape, _, _, cw, factor, finite, support = path_adapter._inputs(
        'incumbent66', HAND_BUDGET, hand_rank, source, n)
    # A stronger all-cell reserve than actual boundary-only exposure. Local
    # reallocation divides existing model slots and creates no extra weight.
    remaining_lower = 1.-cw-path_adapter.OLD_LIMIT-HAND_BUDGET*finite
    if (remaining_lower < .355-1e-12).any():
        raise ValueError('Bounded composite cannot be funded from original support')
    original_pre = path_adapter._matrix(pre, shape, n, 'pre_score').copy()
    good = path_adapter._matrix(a['eligible'], shape, n, 'eligible', boolean=True).copy()
    choices, score = path_adapter.score_and_choose('incumbent66', 0., hand_rank,
        source, meta, n, pre_score=pre, effects=a['effects'], episode_policy=policy,
        boundary_selector=boundary_selector)
    choices = [list(row) for row in choices]
    score = score.copy()
    for i, names in enumerate(meta['symbols'][:n]):
        rank = sorted(np.flatnonzero(good[i]).tolist(), key=lambda j: (-original_pre[i, j], names[j]))
        if len(rank) < 3:
            continue
        candidates = [j for j in rank[2:policy['ceiling']]
                      if original_pre[i, rank[2]]-original_pre[i, j] <= policy['gap']]
        changed = False
        for j in candidates:
            if not finite[i, j] or factor[i, j] == 0.:
                continue
            target = np.clip(support[i, j]+factor[i, j], -.5, .5)
            delta = HAND_BUDGET*(target-support[i, j])
            if not np.isfinite(delta) or abs(delta) > .0125+1e-12:
                raise FloatingPointError('Composite shift exceeds its registered bound')
            if delta != 0.:
                score[i, j] += delta
                if not np.isfinite(score[i, j]):
                    raise FloatingPointError('Nonfinite composite score')
                changed = True
        if changed:
            winner = min(candidates, key=lambda j: (-score[i, j], names[j]))
            choices[i] = sorted(rank[:2]+[winner], key=lambda j: (-score[i, j], names[j]))
    return choices, score
