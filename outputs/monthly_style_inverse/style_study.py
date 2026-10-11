"""Monthly author-choice reconstruction, chronological 35/34 split.

Fits are explanatory research, never a claim to have recovered the author's code.
Per-month oracle diagnostics are written only after the trading rule is frozen.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import itertools
import json
import sys
import numpy as np
import pandas as pd
from scipy.optimize import linprog

P = Path(__file__).resolve().parent
O = P.parent
sys.path.insert(0, str(O / 'monthly_conditional_retention'))
import policy_study as policy
strict, e = policy.strict, policy.e
read, sha = e.read, e.sha
TRAIN, INNER = 35, 23
PARTITIONS = ['global', 'growth', 'smallcap', 'defensive', 'bonds', 'growth_vol']
RETENTION = [(3, 0.), (5, 0.), (5, .03), (5, .06)]


def dump(name, value):
    (P / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
                                    default=lambda v: v.item() if isinstance(v, np.generic) else v) + '\n')


def masks(n):
    t = np.arange(n)
    return dict(inner_fit=t < INNER, inner_validation=(t >= INNER) & (t < TRAIN),
                training=t < TRAIN, validation=t >= TRAIN, all=np.ones(n, bool))


def market_styles(px, dates):
    r = px.pct_change(63, fill_method=None)
    growth = (1 + r.QQQ) / (1 + r.SPY) - 1
    small = (1 + r.IWM) / (1 + r.SPY) - 1
    defense = r[['XLP', 'XLV', 'XLU']].mean(axis=1) - r[['XLK', 'XLY', 'XLI']].mean(axis=1)
    vol = px.SPY.pct_change(fill_method=None).rolling(63).std()
    out = pd.DataFrame(dict(growth_relative63=growth, smallcap_relative63=small,
                            defensive_minus_cyclical63=defense, tlt_return63=r.TLT,
                            spy_above200=px.SPY > px.SPY.rolling(200).mean(),
                            highvol=vol > vol.rolling(252).median()))
    out['growth'] = (growth > 0).astype(int)
    out['smallcap'] = (small > 0).astype(int)
    out['defensive'] = (defense > 0).astype(int)
    out['bonds'] = (r.TLT > 0).astype(int)
    out['growth_vol'] = 2 * out.growth + out.highvol.astype(int)
    out['global'] = 0
    ans = out.loc[pd.to_datetime(dates)]
    assert ans.notna().all().all(), 'Missing market-style inputs'
    return ans


def load():
    meta, packs, px, opening, c, sources = policy.load()
    styles = market_styles(px, c['dates'])
    return meta, packs, px, opening, c, sources, styles


def initialize():
    assert not (P / 'PLAN.json').exists()
    meta, packs, px, opening, c, sources, styles = load()
    dump('PLAN.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
        task='Explain every author monthly holding change using long momentum, volatility, own-sector relative strength, price-volume features and observable market styles.',
        split=dict(train_months=35, validation_months=34, training_signal_end=c['dates'][34],
                   training_return_end=c['periods'][34]['end'], validation_signal_start=c['dates'][35],
                   validation_end=c['periods'][-1]['end'], inner_fit=23, inner_validation=12),
        warning='Previously explored history; newly enforced split does not make the latter34 untouched OOS.',
        core='All22 previously defined momentum families, both normalizations; no seed or coefficient selected using the old48-month fit. All return numerators and own-sector relative returns skip42sessions/two calendar months.',
        styles='63-session QQQ/SPY, IWM/SPY, defensive(XLP/XLV/XLU) minus cyclical(XLK/XLY/XLI), TLT return, growth x market-volatility. These select shared factor weights, never directly reward a stock for sector absolute return.',
        fitting='Fit on first23 only. Global constrained ordered likelihood temp.05, regularization.10. State deltas L1<=.30; state requires8 fit-months or uses global. 44base families x6 partitions x4 retention policies=1056 comparisons, selected only by next12 months.',
        inner_selection='0.60 slot recall +0.25 exact set fraction +0.15 exact order fraction; ties favor fewer regimes, fresh ranking, simpler core, earlier candidate. Never select by return.',
        freeze='Choose global and style leaders in inner validation, refit selected specifications on first35, freeze before outer34 evaluation. No label-based updates in outer34.',
        constraints='Core>=50%; volatility<=30%; industry<=20%; combined shape/acceleration<=20%; acceleration<=10%. Author list treated as descending rank by user hypothesis. No ticker/date coefficients, no intramonth trade, no copied author holdings in model state.',
        diagnostics='After freezing, report all69 author additions/retentions/exits and all candidate scores. Solve single-month constrained linear feasibility and minimum arbitrary residual score correction; these use author labels and are explanatory oracles, never executable models.',
        execution='Zero costs, last-trading-day close signal and same-close idealization; fixed adjusted total-return units within each month. Next-open sensitivity separate. Original pool and assumed17ETF list unchanged.',
        protection='Isolated research outputs; no original account, source, plan or fingerprint changes.'))
    old = read(O / 'monthly_conditional_retention/PROTECTED_BEFORE.json')
    dump('PROTECTED_BEFORE.json', {f: sha(f) for f in old})
    for f in [O / 'monthly_strict_relative' / n for n in ['INPUTS.json', 'percentile.npz', 'magnitude.npz']]:
        sources[str(f)] = sha(f)
    for module in [Path(__file__), Path(policy.__file__), Path(strict.__file__), Path(e.__file__)]:
        sources[str(module)] = sha(module)
    dump('SOURCE_HASHES.json', sources)
    dump('MARKET_STYLES.json', [dict(date=str(d.date()), **r.to_dict()) for d, r in styles.iterrows()])


def fit_rule(d, core, groups, mask, base=None):
    X, spec = strict.design(d, core)
    if base is None:
        base, loss = e.fit(X, d, mask, spec, .05, .10)
    weights, audit = {}, {}
    for g in sorted(set(groups)):
        if len(set(groups)) == 1:
            w, a = base.copy(), dict(months=int(mask.sum()), l1_delta=0.)
        else:
            w, a = policy.fit_delta(X, d, mask & (groups == g), spec, base, .15, .05)
        weights[str(g)], audit[str(g)] = w.tolist(), a
    return dict(core=core, spec=spec, base=base.tolist(), weights=weights, audit=audit)


def rule_scores(rule, d, c, groups):
    X, spec = strict.design(d, rule['core'])
    W = np.array([rule['weights'][str(g)] for g in groups])
    local = np.einsum('ijk,ik->ij', X, W, optimize=False)
    scores = np.full(c['eligible'].shape, -np.inf)
    strength = np.zeros_like(scores)
    for i, ss in enumerate(d['symbols']):
        js = [c['ids'][s] for s in ss]
        scores[i, js] = local[i, :len(ss)]
        ci = [k for k, (_, _, group) in enumerate(spec) if group == 'core']
        q = X[i, :len(ss)][:, ci].mean(axis=1)
        strength[i, js] = pd.Series(q).where(d['eligible'][i, :len(ss)]).rank(pct=True).fillna(.5)
    return scores, strength, local, X, W


def selection_key(row):
    m = row['inner_validation']
    metric = .60 * m['shared_slots'] / (3 * m['months']) + .25 * m['set_exact'] / m['months'] + .15 * m['order_exact'] / m['months']
    return metric, -(1 if row['partition'] == 'global' else 4 if row['partition'] == 'growth_vol' else 2), -row['config']['buffer'], -len(row['core']), -row['id']


def train():
    meta, packs, px, opening, c, sources, styles = load()
    ms = masks(len(c['dates']))
    rows = []
    for norm, core in itertools.product(['percentile', 'magnitude'], meta['families']):
        d = packs[norm]
        X, spec = strict.design(d, core)
        base, _ = e.fit(X, d, ms['inner_fit'], spec, .05, .10)
        for partition in PARTITIONS:
            groups = styles[partition].to_numpy()
            rule = fit_rule(d, core, groups, ms['inner_fit'], base)
            scores, strength, local, X, W = rule_scores(rule, d, c, groups)
            for buffer, gap in RETENTION:
                config = dict(buffer=buffer, gap=gap, exit=dict(name='none'))
                sim = policy.simulate(c, scores, strength, config, count=TRAIN)
                rows.append(dict(id=len(rows), norm=norm, core=core, partition=partition, config=config,
                                 inner_validation=policy.matching(sim['choices'], scores, c, ms['inner_validation'])))
        print('Completed model comparisons', len(rows), norm, core, flush=True)
    assert len(rows) == 1056
    selected = {name: max(group, key=selection_key) for name, group in [
        ('global', [r for r in rows if r['partition'] == 'global']),
        ('style', [r for r in rows if r['partition'] != 'global']), ('overall', rows)]}
    dump('INNER_TRIALS.json', rows)
    frozen = {}
    for name, row in selected.items():
        rule = fit_rule(packs[row['norm']], row['core'], styles[row['partition']].to_numpy(), ms['training'])
        frozen[name] = dict(**row, rule=rule)
    dump('FROZEN.json', dict(created_at=datetime.now(timezone.utc).isoformat(), selection='first23fit/next12selection; refitfirst35', leaders=frozen))
    print('FROZEN', {k: {t: v[t] for t in ['id', 'norm', 'core', 'partition', 'config', 'inner_validation']} for k, v in frozen.items()}, flush=True)


def constraints_for_month(X, eligible, order, ordered=True):
    chosen = [int(j) for j in order if j >= 0]
    other = [int(j) for j in np.flatnonzero(eligible) if j not in chosen]
    pairs = [(a, b) for a in chosen for b in other]
    if ordered:
        pairs += list(itertools.combinations(chosen, 2))
    return pairs, np.array([X[a] - X[b] for a, b in pairs])


def feasible_margin(X, eligible, order, spec, ordered=True):
    if min(order) < 0:
        return dict(status='author_not_eligible_or_not_in_pool', margin=None)
    pairs, D = constraints_for_month(X, eligible, order, ordered)
    A, b, prior = e.limits(spec)
    n = X.shape[1]
    res = linprog(np.r_[np.zeros(n), -1.], A_ub=np.vstack([np.column_stack([A, np.zeros(len(A))]),
                  np.column_stack([-D, np.ones(len(D))])]), b_ub=np.r_[b, np.zeros(len(D))],
                  A_eq=np.r_[np.ones(n), 0.][None, :], b_eq=[1.], bounds=[(0, 1)] * n + [(None, None)], method='highs')
    assert res.success, res.message
    return dict(status='strictly_feasible' if res.x[-1] > 1e-8 else 'tie_only' if res.x[-1] >= -1e-8 else 'infeasible',
                margin=float(res.x[-1]), weights=res.x[:-1].tolist())


def residual_needed(scores, eligible, order):
    if min(order) < 0:
        return None
    # Best conceivable per-name correction, not a learned ML forecast.
    pairs, _ = constraints_for_month(np.eye(len(scores)), eligible, order)
    n = len(scores)
    rows, rhs = [], []
    for a, b in pairs:
        r = np.zeros(n + 1)
        r[b], r[a] = 1., -1.
        rows.append(r)
        rhs.append(scores[a] - scores[b] - 1e-6)
    for j in range(n):
        for sign in [-1, 1]:
            r = np.zeros(n + 1)
            r[j], r[-1] = sign, -1
            rows.append(r)
            rhs.append(0.)
    ans = linprog(np.r_[np.zeros(n), 1.], A_ub=rows, b_ub=rhs,
                  bounds=[(None, None)] * n + [(0, None)], method='highs')
    assert ans.success, ans.message
    return float(ans.x[-1])


def performance(px, opening, c, sim, scores, execution):
    orders = policy.make_orders(sim, scores, c)
    met, eq, trades, holds = e.engine.replay(px, opening, orders, cost=0, execution=execution,
            start=pd.Timestamp(c['dates'][0]), end=pd.Timestamp(c['periods'][-1]['end']))
    if execution == 'close':
        np.testing.assert_allclose(eq, e.previous.independent(px, orders), rtol=1e-11, atol=1e-6)
    slices = {}
    for name, start, end in [('training35', c['dates'][0], c['periods'][34]['end']),
                            ('validation34', c['dates'][35], c['periods'][-1]['end'])]:
        part = eq.loc[start:end]
        slices[name] = e.engine.stats(part / part.iloc[0] * 100000.)
    return met, slices, eq, trades


def explain(row, d, c, styles, scores, local, X, W, sim):
    result, text = [], []
    text += ['逐月选股与换仓归因（研究假说，不是作者自述）',
             '前35个月训练，后34个月验证；按月末收盘等权，月内不换仓，零交易成本。',
             '排名、因子贡献来自冻结模型。按月另拟权重和任意残差只诊断可解释性，不用于交易。', '']
    spec = row['rule']['spec']
    for i, p in enumerate(c['periods']):
        ss = d['symbols'][i]
        names = list(map(e.display, ss))
        author = p['holdings']
        prior = c['periods'][i - 1]['holdings'] if i else []
        picks = [e.display(c['symbols'][j]) for j in sim['choices'][i]]
        ranks = {names[j]: k + 1 for k, j in enumerate(sorted(np.flatnonzero(d['eligible'][i]), key=lambda j: (-local[i, j], j)))}
        allrows = []
        union = sorted(set(author) | set(prior) | set(picks))
        for s in union:
            action = 'retain' if s in author and s in prior else 'enter' if s in author else 'exit' if s in prior else 'model_only'
            rec = dict(ticker=s, author_action=action, author_order=author.index(s) + 1 if s in author else None,
                       frozen_rank=ranks.get(s), model_selected=s in picks)
            if s in names:
                j = names.index(s)
                rec['score'] = float(local[i, j])
                rec['components'] = {g: float(sum(X[i, j, k] * W[i, k] for k, t in enumerate(spec) if t[2] == g)) for g in ['core', 'vol', 'industry', 'shape', 'accel']}
                rec['normalized_factors'] = {k: float(d['X'][i, j, d['names'].index(k)]) for k in list(dict.fromkeys(t[0] for t in spec))}
                if i and s in list(map(e.display, d['symbols'][i - 1])):
                    prevj = list(map(e.display, d['symbols'][i - 1])).index(s)
                    rec['score_change_decomposition'] = dict(
                        feature_change=float(np.sum((X[i, j] - X[i - 1, prevj]) * W[i - 1])),
                        market_style_weight_change=float(np.sum(X[i, j] * (W[i] - W[i - 1]))),
                        total=float(local[i, j] - local[i - 1, prevj]))
                rec['explanation'] = ('ranking_supports_choice' if s in author and ranks.get(s, 999) <= 3 else
                    'retention_supports_choice' if s in author and s in picks else
                    'ranking_supports_exit' if action == 'exit' and ranks.get(s, 999) > 3 else
                    'unexplained_by_frozen_rule')
                if s not in ranks:
                    rec['explanation'] = 'in_pool_but_momentum_or_warmup_ineligible'
            else:
                rec['explanation'] = 'outside_reconstructed_pool'
            allrows.append(rec)
        differences = []
        missed = [s for s in author if s not in picks and s in ranks]
        outsiders = [s for s in picks if s not in author]
        for a, b in itertools.product(missed, outsiders):
            ja, jb = names.index(a), names.index(b)
            dif = {g: float(sum((X[i, ja, k] - X[i, jb, k]) * W[i, k] for k, t in enumerate(spec) if t[2] == g)) for g in ['core', 'vol', 'industry', 'shape', 'accel']}
            differences.append(dict(author_pick=a, model_pick=b, author_minus_model_score=float(local[i, ja] - local[i, jb]), contributions=dif))
        order = d['order'][i]
        ordered = feasible_margin(X[i], d['eligible'][i], order, spec)
        set_only = feasible_margin(X[i], d['eligible'][i], order, spec, ordered=False)
        residual = residual_needed(local[i], d['eligible'][i], order)
        m = dict(signal_date=p['start'], holding_end=p['end'], split='training35' if i < TRAIN else 'validation34',
                 author=author, model=picks, entered=[s for s in author if s not in prior],
                 retained=[s for s in author if s in prior], exited=[s for s in prior if s not in author],
                 market_style=styles.iloc[i].to_dict(), active_group=int(styles.iloc[i][row['partition']]),
                 active_weights=W[i].tolist(), securities=allrows, missed_vs_model=differences,
                 oracle_same_family_ordered=ordered, oracle_same_family_set=set_only,
                 minimum_arbitrary_per_name_score_residual=residual,
                 candidate_ranking=[dict(ticker=s, rank=ranks[s], score=float(local[i, names.index(s)])) for s in sorted(ranks, key=ranks.get)])
        result.append(m)
        style = m['market_style']
        text.append(f"{p['start']} → {p['end']} [{m['split']}] 成长相对SPY {style['growth_relative63']:+.1%}；小盘相对SPY {style['smallcap_relative63']:+.1%}；防御-周期 {style['defensive_minus_cyclical63']:+.1%}")
        text.append('作者：' + ' / '.join(author) + '；模型：' + ' / '.join(picks))
        text.append('新增：' + ','.join(m['entered']) + '；保留：' + ','.join(m['retained']) + '；退出：' + ','.join(m['exited']))
        for z in allrows:
            comp = z.get('components', {})
            desc = dict(ranking_supports_choice='排名支持入选', retention_supports_choice='保留规则支持入选',
                        ranking_supports_exit='排名支持退出', unexplained_by_frozen_rule='冻结规则尚不能解释',
                        in_pool_but_momentum_or_warmup_ineligible='在池但未满足动量/预热资格', outside_reconstructed_pool='不在重建池')
            action = dict(retain='保留', enter='新增', exit='退出', model_only='仅模型选中')[z['author_action']]
            text.append(f"  {z['ticker']} {action}：模型排名 {z['frozen_rank']}；" + desc[z['explanation']] +
                        ('；分数组成 ' + ', '.join(f'{g}={v:+.4f}' for g, v in comp.items()) if comp else ''))
        for dif in differences:
            text.append(f"  未选作者 {dif['author_pick']} vs 模型 {dif['model_pick']}：分差 {dif['author_minus_model_score']:+.4f}；" + ', '.join(f'{g}={v:+.4f}' for g, v in dif['contributions'].items()))
        text.append(f"  同因子体系逐月自由权重可解释集合/顺序：{set_only['status']}/{ordered['status']}；冻结评分所需最小任意修正：{residual}")
        text.append('')
    return result, '\n'.join(text) + '\n'


def evaluate():
    fingerprint = sha(P / 'FROZEN.json')
    sources = read(P / 'SOURCE_HASHES.json')
    assert all(sha(f) == h for f, h in sources.items())
    meta, packs, px, opening, c, sources, styles = load()
    ms = masks(len(c['dates']))
    frozen = read(P / 'FROZEN.json')
    results = {}
    for name, row in frozen['leaders'].items():
        d = packs[row['norm']]
        scores, strength, local, X, W = rule_scores(row['rule'], d, c, styles[row['partition']].to_numpy())
        sim = policy.simulate(c, scores, strength, row['config'], detailed=True)
        met, slices, eq, trades = performance(px, opening, c, sim, scores, 'close')
        os = policy.simulate(c, scores, strength, row['config'], execution='open')
        op, opslices, _, _ = performance(px, opening, c, os, scores, 'open')
        match = {k: policy.matching(sim['choices'], scores, c, ms[k]) for k in ['training', 'validation', 'all']}
        results[name] = dict(rule=row, metrics=met, slices=slices, next_open=op, next_open_slices=opslices, matching=match)
        dump(name + '_TRADES.json', trades)
        dump(name + '_DECISIONS.json', sim['events'])
        dump(name + '_DAILY.json', [dict(date=str(t.date()), equity=float(v)) for t, v in eq.items()])
        if name == 'overall':
            months, txt = explain(row, d, c, styles, scores, local, X, W, sim)
            dump('MONTHLY_EXPLANATIONS.json', months)
            (P / 'MONTHLY_EXPLANATIONS.txt').write_text(txt)
            counts = {}
            for k, mask in [('training', ms['training']), ('validation', ms['validation']), ('all', ms['all'])]:
                selected = [m for j, m in enumerate(months) if mask[j]]
                counts[k] = dict(months=len(selected),
                    ordered_feasible=sum(m['oracle_same_family_ordered']['status'] == 'strictly_feasible' for m in selected),
                    set_feasible=sum(m['oracle_same_family_set']['status'] == 'strictly_feasible' for m in selected),
                    unavailable=sum(m['oracle_same_family_set']['status'] == 'author_not_eligible_or_not_in_pool' for m in selected),
                    residual_at_most_005=sum(m['minimum_arbitrary_per_name_score_residual'] is not None and m['minimum_arbitrary_per_name_score_residual'] <= .05 for m in selected))
            results[name]['explanatory_oracle_counts'] = counts
        print(name, 'RESULT', json.dumps(dict(metrics=met, slices=slices, matching=match), ensure_ascii=False), flush=True)
    author_choices = [[c['ids'][e.actual(s)] for s in p['holdings']] for p in c['periods']]
    dummy = np.zeros_like(c['eligible'], dtype=float)
    am, ass, ae, _ = performance(px, opening, c, dict(choices=author_choices), dummy, 'close')
    results['author'] = dict(metrics=am, slices=ass)
    qeq = px.QQQ.loc[c['dates'][0]:c['periods'][-1]['end']]
    results['qqq'] = dict(metrics=e.engine.stats(qeq / qeq.iloc[0] * 100000.), slices={})
    for key, a, b in [('training35', c['dates'][0], c['periods'][34]['end']), ('validation34', c['dates'][35], c['periods'][-1]['end'])]:
        q = qeq.loc[a:b]
        results['qqq']['slices'][key] = e.engine.stats(q / q.iloc[0] * 100000.)
    dump('RESULTS.json', results)
    assert sha(P / 'FROZEN.json') == fingerprint
    changed = [f for f, h in read(P / 'PROTECTED_BEFORE.json').items() if sha(f) != h]
    assert not changed, changed
    dump('VALIDATION.json', dict(frozen_hash=fingerprint, protected_changes=changed, inputs_verified=True,
        independent_daily_share_replay=True, training35_validation34=True, inner23_selection12=True,
        no_outer_labels_or_returns_used_for_selection=True, author_diagnostics_after_freeze=True,
        no_intramonth_decisions=True, zero_transaction_costs=True, reused_history_not_pristine_oos=True))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=['initialize', 'train', 'evaluate'])
    globals()[ap.parse_args().phase]()
