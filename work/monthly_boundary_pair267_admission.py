"""267 role adapter for the unchanged, frozen266 first88 admission rules.

Caller-supplied metrics/orders only. No new gate, future-period score or state.
"""
from copy import deepcopy
import monthly_monotone_tree266_admission as original

ARMS = ('incumbent_local67', 'top5_all_pairs_control', 'original_p34_pair_candidate')
ORIGINAL_ARMS = ('incumbent_local67', 'unconstrained_weak_tree261_control', 'monotone_raw_weak_tree_candidate')


def select_first88(metrics, orders):
    if set(metrics) != set(ARMS) or set(orders) != set(ARMS):
        raise ValueError('Three fixed267 roles required')
    if original.ARMS != ORIGINAL_ARMS:
        raise ValueError('Frozen admission roles changed')
    translated_metrics = {old: metrics[new] for old, new in zip(ORIGINAL_ARMS, ARMS)}
    translated_orders = {old: orders[new] for old, new in zip(ORIGINAL_ARMS, ARMS)}
    result = deepcopy(original.select_first88(translated_metrics, translated_orders))
    names = dict(zip(ORIGINAL_ARMS, ARMS))
    result['primary'] = names[result['primary']]
    for key in ('guards', 'changed_months_by_reference', 'undefined_sharpe'):
        result[key] = {names[k]: v for k, v in result[key].items()}
    return result
