"""Causal monthend policy research with endogenous holdings, costs and peaks.

Monthly decisions only. Author holdings are labels for research, never trading state.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import itertools
import json
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize

P = Path(__file__).resolve().parent
O = P.parent
sys.path.insert(0, str(O / 'monthly_strict_relative'))
import strict_study as strict
e = strict.e


def read(path):
    return json.loads(Path(path).read_text())


def dump(name, value):
    (P / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
                                   default=lambda z: z.item() if isinstance(z, np.generic) else z) + '\n')


def init():
    assert not (P / 'PLAN.json').exists()
    dump('PLAN.json', dict(
        created_at=datetime.now(timezone.utc).isoformat(),
        objective='Compare observable-regime fresh ranking against monthend retention/replacement and conditional exits, using supplied ordered holdings.',
        hard_rules='No intramonth trading. Long momentum and stock/own-sector-ETF relative momentum still skip latest42 sessions/two calendar months. No new recent-return ranking factor, ticker/date intercept, or author holdings injected into trading state.',
        source='Unchanged reconstructed Top32 + assumed theme17 from strict-relative study; no certified original-author pool claim.',
        seeds='Six distinct core families from prior264 trials, chosen only by first48 calibration similarity; keep their original normalization and coefficients.',
        states='SPY aboveMA200, SPYvol63 above trailing252 median, and their four-way cross. Each group needs8 calibration months, otherwise uses global coefficients. Every state retains original core/industry/correction budgets.',
        state_grid='6 seeds x (global + 3 partitions x2 coefficient-turnover budgets .10/.20) =42 scoring rules. Fit deltas only on first48 labels, using ordered likelihood and .10 shrinkage toward global coefficients.',
        retention_grid='(rank buffer, replacement gap): (3,0),(4,0),(5,0),(7,0),(5,.03),(5,.06). Gap0 means retain all qualifying incumbents within the buffer; positive gap permits replacing even within-buffer incumbents when an outsider exceeds their score by that gap. Fill remaining slots with best eligible outsiders.',
        exit_grid='none; fixed loss15%/profit50%; sigma-normalized loss2.5/profit5/trail3; adaptive version; adaptive no-profit variant; adaptive tight(1.5,3,2); adaptive wide(3.5,8,5). 42x6x7=1764 policies.',
        threshold_units='sigma=known daily63-session std *sqrt21. Adaptive multiplier: weak SPYtrend .8, highmarketvol .9, strong core percentile>=.75 1.25, weak core percentile<.5 .8. Loss clipped5%-50%,profit10%-100%,trail8%-60%. Hypotheses, not author parameters.',
        holding_state='Own recursively chosen positions, weighted-average adjusted cost updated only on additions, cost unchanged on trims; own episode closing-price peak. Conditions checked only at current monthend close. Exit names blocked from immediate same-rebalance repurchase.',
        selection='Freeze leaders for fresh, retention-only, explicit exits and overall by calibration matching, never later returns. Evaluate first48/later21/all with CAGR, daily Sharpe, drawdown, ordered/set/slot match and transition fit. Later21 previously seen, not pristine OOS.',
        execution='Zero cost, same-close idealized reproduction primary; next-open sensitivity regenerates decisions with its own executed costs/peaks, never blindly shifts stateful primary orders.',
        protected='Read-only source prices and every existing observation account; no auto-promotion or live trades.'))
    old = read(O / 'monthly_strict_relative/PROTECTED_BEFORE.json')
    dump('PROTECTED_BEFORE.json', {f: e.sha(f) for f in old})


def calibration_key(row):
    m = row['calibration']
    return m['selection_score'], m['order_exact'], m['set_exact'], -row['id']


def load():
    root = O / 'monthly_strict_relative'
    meta = read(root / 'INPUTS.json')
    packs = {n: dict(np.load(root / (n + '.npz')), **meta['packs'][n]) for n in ['percentile', 'magnitude']}
    px, raw, members, periods, sources, pools = e.load_prices()
    opening = raw['opening'] * px / raw['close']
    dates = pd.to_datetime([p['start'] for p in periods])
    st = e.market_states(px).loc[dates].astype(int)
    symbols = sorted({s for d in packs['percentile']['symbols'] for s in d})
    ids = {s: j for j, s in enumerate(symbols)}
    eligible = np.zeros((len(periods), len(symbols)), bool)
    for i, ss in enumerate(packs['percentile']['symbols']):
        eligible[i, [ids[s] for s in ss]] = packs['percentile']['eligible'][i, :len(ss)]
    sigma = px.pct_change(fill_method=None).rolling(63).std() * np.sqrt(21)
    peaks = []
    for i, date in enumerate(dates):
        start = date if not i else px.index[px.index.get_loc(dates[i - 1]) + 1]
        peaks.append(px.loc[start:date, symbols].max().to_numpy())
    exec_dates = [px.index[px.index.get_loc(d) + 1] for d in dates]
    context = dict(symbols=symbols, ids=ids, dates=[str(d.date()) for d in dates],
                   eligible=eligible, close=px.loc[dates, symbols].to_numpy(),
                   opening=opening.loc[exec_dates, symbols].to_numpy(),
                   opening_dates=[str(d.date()) for d in exec_dates],
                   peak=np.stack(peaks), sigma=sigma.loc[dates, symbols].to_numpy(),
                   trend=st.spy_trend.to_numpy(), highvol=st.high_market_vol.to_numpy(),
                   periods=periods)
    return meta, packs, px, opening, context, sources


def fit_delta(X, d, mask, spec, base, budget, temp):
    if mask.sum() < 8:
        return base.copy(), dict(months=int(mask.sum()), fallback='fewer_than8_months')
    E, allowed, selected, pw = e.events(X, d, mask)
    A, b, prior = e.limits(spec)
    n = len(base)
    B = np.column_stack([np.eye(n), -np.eye(n)])

    def weights(z):
        return base + z[:n] - z[n:]

    def objective(z):
        value, grad = e.likelihood(weights(z), E, allowed, selected, pw, temp, .10, base)
        return value + 1e-9 * z.sum(), np.r_[grad, -grad] + 1e-9

    constraints = [
        dict(type='eq', fun=lambda z: weights(z).sum() - 1, jac=lambda z: np.r_[np.ones(n), -np.ones(n)]),
        dict(type='ineq', fun=lambda z: 2 * budget - z.sum(), jac=lambda z: -np.ones(2 * n)),
        dict(type='ineq', fun=lambda z: weights(z), jac=lambda z: B),
        dict(type='ineq', fun=lambda z: 1 - weights(z), jac=lambda z: -B),
        dict(type='ineq', fun=lambda z: b - np.einsum('ij,j->i', A, weights(z), optimize=False),
             jac=lambda z: -np.einsum('ij,jk->ik', A, B, optimize=False)),
    ]
    opt = minimize(objective, np.zeros(2 * n), jac=True, method='SLSQP', constraints=constraints,
                   bounds=[(0, 2 * budget)] * (2 * n), options=dict(maxiter=500, ftol=1e-10))
    assert opt.success, opt.message
    w = weights(opt.x)
    assert abs(w.sum() - 1) < 1e-7 and w.min() > -1e-7
    assert np.abs(w - base).sum() <= 2 * budget + 1e-7
    assert (np.einsum('ij,j->i', A, w, optimize=False) <= b + 1e-7).all()
    return w, dict(months=int(mask.sum()), l1_delta=float(np.abs(w - base).sum()), loss=float(opt.fun))


def scores_from_rule(rule, packs, context):
    d = packs[rule['seed']['norm']]
    X, spec = strict.design(d, rule['seed']['core'])
    groups = (np.zeros(len(X), int) if rule['partition'] == 'global' else
              context['trend'] if rule['partition'] == 'trend' else
              context['highvol'] if rule['partition'] == 'volatility' else 2 * context['trend'] + context['highvol'])
    local = np.zeros(X.shape[:2])
    for g in np.unique(groups):
        w = rule['weights'][str(g)]
        local[groups == g] = np.einsum('ijk,k->ij', X[groups == g], w, optimize=False)
    score = np.full(context['eligible'].shape, -np.inf)
    strength = np.zeros_like(score)
    core_i = [d['names'].index(k) for k in rule['seed']['core']]
    for i, ss in enumerate(d['symbols']):
        js = [context['ids'][s] for s in ss]
        score[i, js] = local[i, :len(ss)]
        core_value = d['X'][i, :len(ss)][:, core_i].mean(axis=1)
        good = d['eligible'][i, :len(ss)]
        strength[i, js] = pd.Series(core_value).where(good).rank(pct=True).fillna(.5).to_numpy()
    return score, strength


def score_train():
    meta, packs, px, opening, c, sources = load()
    dev = np.array(meta['development'])
    old = sorted(read(O / 'monthly_strict_relative/TRIALS.json'), key=calibration_key, reverse=True)
    seeds, seen = [], set()
    for r in old:
        key = tuple(r['core'])
        if key not in seen:
            seeds.append(r)
            seen.add(key)
        if len(seeds) == 6:
            break
    dump('SEEDS.json', seeds)
    rules = []
    for seed in seeds:
        d = packs[seed['norm']]
        X, spec = strict.design(d, seed['core'])
        base = np.array(seed['weights'])
        rules.append(dict(id=len(rules), seed=seed, partition='global', budget=0, weights={'0': base.tolist()}, audit={}))
        for partition, budget in itertools.product(['trend', 'volatility', 'joint'], [.10, .20]):
            groups = c['trend'] if partition == 'trend' else c['highvol'] if partition == 'volatility' else 2 * c['trend'] + c['highvol']
            weights, audit = {}, {}
            for g in np.unique(groups):
                w, a = fit_delta(X, d, dev & (groups == g), spec, base, budget, seed['temp'])
                weights[str(g)], audit[str(g)] = w.tolist(), a
            rules.append(dict(id=len(rules), seed=seed, partition=partition, budget=budget, weights=weights, audit=audit))
        print('Scoring families finished', len(rules), 'seed', seed['core'], flush=True)
    arrays, strengths = [], []
    for rule in rules:
        score, strength = scores_from_rule(rule, packs, c)
        arrays.append(score)
        strengths.append(strength)
    np.savez_compressed(P / 'SCORES.npz', scores=np.stack(arrays), strengths=np.stack(strengths))
    dump('SCORING_RULES.json', rules)
    for f in [O / 'monthly_strict_relative' / n for n in ['INPUTS.json', 'TRIALS.json', 'percentile.npz', 'magnitude.npz']]:
        sources[str(f)] = e.sha(f)
    dump('SOURCE_HASHES.json', sources)


EXIT_POLICIES = [
    dict(name='none'),
    dict(name='fixed', loss=.15, profit=.50),
    dict(name='sigma', loss_k=2.5, profit_k=5., trail_k=3.),
    dict(name='adaptive', loss_k=2.5, profit_k=5., trail_k=3., adaptive=True),
    dict(name='adaptive_no_profit', loss_k=2.5, profit_k=None, trail_k=3., adaptive=True),
    dict(name='adaptive_tight', loss_k=1.5, profit_k=3., trail_k=2., adaptive=True),
    dict(name='adaptive_wide', loss_k=3.5, profit_k=8., trail_k=5., adaptive=True),
]
RETENTIONS = [(3, 0.), (4, 0.), (5, 0.), (7, 0.), (5, .03), (5, .06)]


def exit_thresholds(policy, sigma, strength, trend, highvol):
    if policy['name'] == 'none':
        return None
    if policy['name'] == 'fixed':
        return dict(loss=policy['loss'], profit=policy['profit'], trail=None)
    assert np.isfinite(sigma) and sigma > 0
    multiplier = 1.
    if policy.get('adaptive'):
        multiplier *= (1. if trend else .8) * (.9 if highvol else 1.)
        multiplier *= 1.25 if strength >= .75 else .8 if strength < .5 else 1.
    v = sigma * multiplier
    return dict(loss=float(np.clip(policy['loss_k'] * v, .05, .5)),
                profit=float(np.clip(policy['profit_k'] * v, .1, 1.)) if policy.get('profit_k') is not None else None,
                trail=float(np.clip(policy['trail_k'] * v, .08, .6)) if policy.get('trail_k') is not None else None)


def select_month(score, eligible, state, close, peak, sigma, strength, trend, highvol, buffer, gap, policy):
    ranked = sorted(np.flatnonzero(eligible), key=lambda j: (-score[j], j))
    rank = {j: r + 1 for r, j in enumerate(ranked)}
    banned, diagnostics = set(), {}
    for j, hold in state.items():
        if j not in rank:
            banned.add(j)
            diagnostics[j] = dict(reason='pool_or_momentum_ineligible')
            continue
        pnl = close[j] / hold['cost'] - 1
        drawdown = close[j] / max(hold['peak'], peak[j]) - 1
        threshold = exit_thresholds(policy, sigma[j], strength[j], trend, highvol)
        reason = None
        if threshold:
            if pnl <= -threshold['loss']:
                reason = 'monthend_stop_loss'
            elif threshold['profit'] is not None and pnl >= threshold['profit']:
                reason = 'monthend_take_profit'
            elif threshold['trail'] is not None and drawdown <= -threshold['trail']:
                reason = 'monthend_trailing_exit'
        diagnostics[j] = dict(pnl=float(pnl), drawdown=float(drawdown), threshold=threshold, reason=reason)
        if reason:
            banned.add(j)
    candidates = [j for j in ranked if j not in banned]
    kept = [j for j in candidates if j in state and rank[j] <= buffer][:3]
    chosen = kept + [j for j in candidates if j not in kept][:3 - len(kept)]
    # An incumbent may survive an outsider's small ranking lead. If the lead
    # exceeds the gap, replace the weakest chosen name; ties favor continuity.
    for j in candidates if gap > 0 else []:
        if j in chosen or not chosen:
            continue
        worst = min(chosen, key=lambda k: (score[k], -k))
        if score[j] > score[worst] + gap + 1e-12:
            chosen.remove(worst)
            chosen.append(j)
    chosen = sorted(chosen, key=lambda j: (-score[j], j)) if len(chosen) >= 2 else []
    for j in state:
        if j not in chosen and diagnostics[j]['reason'] is None:
            diagnostics[j]['reason'] = 'ranking_or_replacement'
        elif j in chosen:
            diagnostics[j]['reason'] = 'retained'
    return chosen, diagnostics


def rebalance_state(state, chosen, prices, nav, date, known_peak):
    new = {}
    for j in chosen:
        price = float(prices[j])
        assert np.isfinite(price) and price > 0
        quantity = nav / len(chosen) / price
        if j not in state:
            new[j] = dict(units=quantity, cost=price, peak=price, entry_date=date)
        else:
            h = state[j]
            addition = max(quantity - h['units'], 0.)
            cost = ((h['cost'] * h['units'] + addition * price) / quantity) if addition > 0 else h['cost']
            new[j] = dict(units=quantity, cost=cost, peak=max(h['peak'], float(known_peak[j]), price), entry_date=h['entry_date'])
    return new


def simulate(c, scores, strengths, config, count=None, execution='close', detailed=False):
    count = len(c['dates']) if count is None else count
    state, cash, choices, events, holdings = {}, 100000., [], [], []
    for i in range(count):
        close, peak = c['close'][i], c['peak'][i]
        for j, h in state.items():
            assert np.isfinite(close[j]) and close[j] > 0 and np.isfinite(peak[j]), (i, j, 'Missing held price')
            h['peak'] = max(h['peak'], float(peak[j]))
        chosen, diagnostics = select_month(scores[i], c['eligible'][i], state, close, peak, c['sigma'][i], strengths[i],
                                            c['trend'][i], c['highvol'][i], config['buffer'], config['gap'], config['exit'])
        if detailed:
            events.append(dict(date=c['dates'][i], before={e.display(c['symbols'][j]): dict(h) for j, h in state.items()},
                               decisions={e.display(c['symbols'][j]): d for j, d in diagnostics.items()},
                               chosen=[e.display(c['symbols'][j]) for j in chosen]))
        fill = c['close'][i] if execution == 'close' else c['opening'][i]
        fill_date = c['dates'][i] if execution == 'close' else c['opening_dates'][i]
        assert all(np.isfinite(fill[j]) and fill[j] > 0 for j in set(chosen) | set(state))
        nav = cash + sum(h['units'] * fill[j] for j, h in state.items())
        assert nav > 0
        state = rebalance_state(state, chosen, fill, nav, fill_date, peak)
        cash = nav if not state else 0.
        choices.append(chosen)
        if detailed:
            holdings.append(dict(signal_date=c['dates'][i], execution_date=fill_date,
                                 positions={e.display(c['symbols'][j]): dict(h) for j, h in state.items()}, cash=float(cash)))
    return dict(choices=choices, events=events, holdings=holdings, state=state, cash=cash)


def matching(choices, scores, c, mask):
    m = dict(months=0, set_exact=0, order_exact=0, shared_slots=0, pairs_correct=0, pairs_total=0,
             unavailable_author_slots=0, exit_true_positive=0, exit_false_positive=0, exit_false_negative=0,
             retain_true_positive=0, retain_false_positive=0, retain_false_negative=0)
    for i in np.flatnonzero(mask):
        names = [e.display(c['symbols'][j]) for j in choices[i]]
        author = c['periods'][i]['holdings']
        m['months'] += 1
        m['set_exact'] += set(names) == set(author)
        m['order_exact'] += names == author
        m['shared_slots'] += len(set(names) & set(author))
        ranked = sorted(np.flatnonzero(c['eligible'][i]), key=lambda j: (-scores[i, j], j))
        ranks = {e.display(c['symbols'][j]): r for r, j in enumerate(ranked)}
        m['unavailable_author_slots'] += sum(s not in ranks for s in author)
        for a, b in itertools.combinations(author, 2):
            if a in ranks and b in ranks:
                m['pairs_total'] += 1
                m['pairs_correct'] += ranks[a] < ranks[b]
        if i:
            prev = {e.display(c['symbols'][j]) for j in choices[i - 1]}
            aprev = set(c['periods'][i - 1]['holdings'])
            for k, mine, target in [('exit', prev - set(names), aprev - set(author)),
                                    ('retain', prev & set(names), aprev & set(author))]:
                m[k + '_true_positive'] += len(mine & target)
                m[k + '_false_positive'] += len(mine - target)
                m[k + '_false_negative'] += len(target - mine)
    m['selection_score'] = .5 * m['shared_slots'] / (3 * m['months']) + .5 * m['pairs_correct'] / max(1, m['pairs_total'])
    for key in ['exit', 'retain']:
        tp, fp, fn = [m[key + suffix] for suffix in ['_true_positive', '_false_positive', '_false_negative']]
        m[key + '_f1'] = 2 * tp / max(1, 2 * tp + fp + fn)
    return m


def policy_train():
    meta, packs, px, opening, c, sources = load()
    a = np.load(P / 'SCORES.npz')
    scores, strengths = a['scores'], a['strengths']
    dev = np.array(meta['development'])
    assert dev.sum() == 48 and dev[:48].all()
    rows = []
    for score_id in range(len(scores)):
        for (buffer, gap), policy in itertools.product(RETENTIONS, EXIT_POLICIES):
            config = dict(buffer=buffer, gap=gap, exit=policy)
            sim = simulate(c, scores[score_id], strengths[score_id], config, count=48)
            rows.append(dict(id=len(rows), score_id=score_id, config=config,
                             calibration=matching(sim['choices'], scores[score_id], c, dev)))
        if (score_id + 1) % 7 == 0:
            print('Monthly policy trials', len(rows), flush=True)
    assert len(rows) == 1764
    categories = dict(
        fresh=[r for r in rows if r['config']['buffer'] == 3 and r['config']['gap'] == 0 and r['config']['exit']['name'] == 'none'],
        retention_only=[r for r in rows if r['config']['buffer'] > 3 and r['config']['exit']['name'] == 'none'],
        explicit_exits=[r for r in rows if r['config']['exit']['name'] != 'none'],
        overall=rows)
    leaders = {name: max(group, key=calibration_key) for name, group in categories.items()}
    dump('POLICY_TRIALS.json', rows)
    dump('FROZEN.json', dict(leaders=leaders, trial_count=len(rows), created_at=datetime.now(timezone.utc).isoformat(),
                             only_first48_used=True, performance_not_used_to_select=True))
    print('FROZEN', json.dumps(leaders, default=lambda z: z.item(), ensure_ascii=False), flush=True)


def make_orders(sim, scores, c):
    return [dict(date=c['dates'][i], signal_date=c['dates'][i], end=c['periods'][i]['end'],
                 symbols=[c['symbols'][j] for j in chosen], scores=[float(scores[i, j]) for j in chosen])
            for i, chosen in enumerate(sim['choices'])]


def evaluate():
    frozen_hash = e.sha(P / 'FROZEN.json')
    source_hashes = read(P / 'SOURCE_HASHES.json')
    assert all(e.sha(f) == h for f, h in source_hashes.items())
    meta, packs, px, opening, c, sources = load()
    a = np.load(P / 'SCORES.npz')
    frozen = read(P / 'FROZEN.json')
    rules = read(P / 'SCORING_RULES.json')
    dev = np.array(meta['development'])
    start, end = pd.Timestamp(c['dates'][0]), pd.Timestamp(c['periods'][-1]['end'])
    results = {}
    for name, row in frozen['leaders'].items():
        score, strength = a['scores'][row['score_id']], a['strengths'][row['score_id']]
        sim = simulate(c, score, strength, row['config'], detailed=True)
        orders = make_orders(sim, score, c)
        metrics, equity, trades, holds = e.engine.replay(px, opening, orders, cost=0, execution='close', start=start, end=end)
        np.testing.assert_allclose(equity, e.previous.independent(px, orders), rtol=1e-11, atol=1e-6)
        own_final = sim['cash'] + sum(h['units'] * px.at[end, c['symbols'][j]] for j, h in sim['state'].items())
        np.testing.assert_allclose(own_final, equity.iloc[-1], rtol=1e-11)
        os = simulate(c, score, strength, row['config'], execution='open', detailed=True)
        om, oe, ot, oh = e.engine.replay(px, opening, make_orders(os, score, c), cost=0, execution='open', start=start, end=end)
        ofinal = os['cash'] + sum(h['units'] * px.at[end, c['symbols'][j]] for j, h in os['state'].items())
        np.testing.assert_allclose(ofinal, oe.iloc[-1], rtol=1e-11)
        mat = {k: matching(sim['choices'], score, c, mask) for k, mask in [('all', np.ones(len(dev), bool)), ('calibration48', dev), ('later21', ~dev)]}
        slices = {}
        for k, begin, finish in [('calibration48', '2020-12-31', '2024-12-31'), ('later21', '2024-12-31', '2026-09-24')]:
            z = equity.loc[begin:finish]
            slices[k] = e.engine.stats(z / z.iloc[0] * 100000.)
        reasons = {}
        for event in sim['events']:
            for d in event['decisions'].values():
                reasons[d['reason']] = reasons.get(d['reason'], 0) + 1
        months = [dict(date=c['dates'][i], end=c['periods'][i]['end'], author=c['periods'][i]['holdings'],
                       ours=[e.display(c['symbols'][j]) for j in choice]) for i, choice in enumerate(sim['choices'])]
        results[name] = dict(metrics=metrics, next_open=om, matching=mat, slices=slices, policy=row,
                             scoring_rule=rules[row['score_id']], monthend_decision_counts=reasons, months=months)
        dump(name + '_TRADES.json', trades)
        dump(name + '_DECISIONS.json', sim['events'])
        dump(name + '_HOLDING_STATES.json', sim['holdings'])
        dump(name + '_DAILY.json', [dict(date=str(d.date()), equity=float(v)) for d, v in equity.items()])
        dump(name + '_NEXT_OPEN_DECISIONS.json', os['events'])
        dump(name + '_NEXT_OPEN_TRADES.json', ot)
        print(name, json.dumps(dict(cagr=metrics['cagr'], sharpe=metrics['sharpe'], matching=mat['all'], reasons=reasons)), flush=True)
    dump('RESULTS.json', results)
    assert e.sha(P / 'FROZEN.json') == frozen_hash
    changed = [f for f, h in read(P / 'PROTECTED_BEFORE.json').items() if e.sha(f) != h]
    assert not changed, changed
    dump('VALIDATION.json', dict(protected_changes=changed, independent_close_replay=True,
                                 stateful_next_open_accounting_parity=True, source_hashes_verified=True,
                                 frozen_choice_unchanged=True, no_intramonth_decisions=True,
                                 causal_own_holdings_recursion=True, subsequent_period_not_pristine_oos=True))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=['init', 'score_train', 'policy_train', 'evaluate'])
    globals()[ap.parse_args().phase]()
