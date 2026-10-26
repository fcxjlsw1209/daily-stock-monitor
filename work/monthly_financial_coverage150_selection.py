"""Pure coverage-only selection adapter; no I/O, models or actual replay.

The two earnings hands intentionally have different availability masks. Their
formula and dose are identical. The caller supplies the original149 earnings
hand and preserves its source gate, and binds this module plus the complete
frozen149 runner dependency closure before any actual experiment.
"""
import monthly_stock_comp149_runner as old

NAMES = ("incumbent_local67", "original_earnings", "expanded_earnings")
HANDS = ("earnings_original", "earnings_expanded")
DOSE = old.DOSE
corrected = old.corrected


def selection_contract():
    result = old.selection_contract().copy()
    result.update(
        same_close_cagr="strict_greater_than_incumbent_and_original_earnings",
        next_open_cost_cagr="noninferior_incumbent_strict_greater_than_original_earnings",
        financial_formula="NI_current/((assets_current+assets_prior)/2)",
        bounded="h(x)=x/(1+abs(x))", dose=DOSE,
        coverage_masks="distinct_original_and_expanded_masks",
        unavailable="exact_incumbent_fallback_for_that_hand",
        old_coverage="frozen149_earnings_control_including_SBC_common_mask",
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
    """Only first88 same-close and next-open5bp summaries select the primary."""
    for group in (rows, opened):
        old.require([row["name"] for row in group] == list(NAMES),
                    "Three exact coverage arms in declared order required")
    close = {NAMES[i]: old.previous.guard(rows[2], rows[i], strict=True)
             for i in range(2)}
    op = {NAMES[i]: old.previous.guard(opened[2], opened[i], strict=i != 0)
          for i in range(2)}
    passed = all(close.values()) and all(op.values())
    return dict(primary=NAMES[2] if passed else NAMES[0], admitted=passed,
                guards_against=close, next_open_guards_against=op,
                mechanism_confirmed=all(op.values()))
