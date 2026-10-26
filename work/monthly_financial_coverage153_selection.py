"""Pure six-arm coverage selection; no I/O, model fit or actual replay.

All five earnings hands use the frozen149 correction once at the same dose.
The caller authenticates each prior cache and the new source admission.
"""
import monthly_stock_comp149_runner as old

NAMES = ("incumbent_local67", "original_earnings", "batch1_earnings",
         "batch2_earnings", "batch3_earnings", "expanded_earnings")
HANDS = ("earnings_original", "earnings_batch1", "earnings_batch2",
         "earnings_batch3", "earnings_expanded")
DOSE = old.DOSE
corrected = old.corrected


def selection_contract():
    result = old.selection_contract().copy()
    result.update(
        same_close_cagr="strict_greater_than_incumbent_original_batch1_batch2_and_batch3_earnings",
        next_open_cost_cagr="noninferior_incumbent_strict_greater_than_original_batch1_batch2_and_batch3_earnings",
        financial_formula="NI_current/((assets_current+assets_prior)/2)",
        bounded="h(x)=x/(1+abs(x))", dose=DOSE,
        coverage_masks="distinct_original_batch1_batch2_batch3_and_expanded_masks",
        unavailable="exact_incumbent_fallback_for_that_hand",
        old_coverage="frozen149_earnings_control_including_SBC_common_mask",
        batch1_coverage="frozen150_expanded_earnings_and_mask",
        batch2_coverage="frozen151_expanded_earnings_and_mask",
        batch3_coverage="frozen152_expanded_earnings_and_mask",
        episode="one_original_episode_selector_after_same_original_pre_score",
    )
    return result


def choose(name, arrays, meta, n, backend, policy):
    """Use precisely the frozen correction placement, separately per mask."""
    old.require(name in NAMES, "Undeclared coverage arm")
    if name == NAMES[0]:
        return backend.baseline(arrays, meta, n, policy)
    hand = arrays[HANDS[NAMES.index(name)-1]]
    return corrected(backend.pre(arrays, n), hand, arrays, meta, n, backend, policy)


def select(rows, opened):
    """Only first88 same-close and next-open5bp summaries select primary."""
    for group in (rows, opened):
        old.require([row["name"] for row in group] == list(NAMES),
                    "Six exact coverage arms in declared order required")
    close = {NAMES[i]: old.previous.guard(rows[5], rows[i], strict=True)
             for i in range(5)}
    op = {NAMES[i]: old.previous.guard(opened[5], opened[i], strict=i != 0)
          for i in range(5)}
    passed = all(close.values()) and all(op.values())
    return dict(primary=NAMES[5] if passed else NAMES[0], admitted=passed,
                guards_against=close, next_open_guards_against=op,
                mechanism_confirmed=all(op.values()))
