"""Fixed first88 selection between two prospective selector-scope candidates.

The frozen266 validation/risk guards are reused without changing their roles
or globals. Each candidate is compared with Local67; later returns never enter.
"""
from monthly_monotone_tree266_admission import guard, required_metrics

ARMS = ('incumbent_local67', 'pooled_affine', 'mixed_affine')


def select_first88(metrics, orders):
    if set(metrics) != set(ARMS) or set(orders) != set(ARMS):
        raise ValueError('Three fixed269 roles required')
    invalid = {a: required_metrics(metrics[a], a == ARMS[0]) for a in ARMS}
    if any(len(orders[a]) < 88 for a in ARMS):
        raise ValueError('First88 orders required')
    sets = {}
    for a in ARMS:
        sets[a] = []
        for i in range(88):
            names = orders[a][i]['symbols']
            if (not isinstance(names, list) or len(names) not in (0, 2, 3)
                    or any(not isinstance(s, str) or not s for s in names)
                    or len(set(names)) != len(names)):
                raise ValueError('Valid unique EW3/two/cash decision required')
            sets[a].append(set(names))
    guards, changes, accepted = {}, {}, []
    for a in ARMS[1:]:
        changes[a] = [i for i in range(88) if sets[a][i] != sets[ARMS[0]][i]]
        close = not invalid[a] and guard(metrics[a]['same_close_zero'], metrics[ARMS[0]]['same_close_zero'], .005)
        opening = not invalid[a] and guard(metrics[a]['next_open_zero'], metrics[ARMS[0]]['next_open_zero'], 0.)
        support = (len(changes[a]) >= 3 and any(i < 53 for i in changes[a])
                   and any(i >= 53 for i in changes[a]))
        guards[a] = dict(close=bool(close), open=bool(opening), support=bool(support))
        if all(guards[a].values()):
            accepted.append(a)
    selected = ARMS[0]
    if accepted:
        best = max(metrics[a]['same_close_zero']['training']['cagr'] for a in accepted)
        selected = next(a for a in ARMS[1:] if a in accepted
                        and best - metrics[a]['same_close_zero']['training']['cagr'] <= 1e-12)
    return dict(primary=selected, admitted=bool(accepted), qualified=accepted,
                guards=guards, changed_months_by_candidate=changes,
                undefined_sharpe=invalid, first88_only=True, later34_used=False,
                cost_used=False, tie_rule='fixed_arm_order_with_1e-12_cagr_tolerance')
