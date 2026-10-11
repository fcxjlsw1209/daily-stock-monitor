"""Post-freeze diagnostics. Oracle results never enter the strategy or selection."""
import numpy as np
import pandas as pd
import style_study as s


def finite(x):
    return float(x) if np.isfinite(x) else None


def main():
    frozen_hash = s.sha(s.P / 'FROZEN.json')
    results_hash = s.sha(s.P / 'RESULTS.json')
    meta, packs, px, opening, c, sources, styles = s.load()
    frozen = s.read(s.P / 'FROZEN.json')
    row = frozen['leaders']['style']
    d = packs[row['norm']]
    scores, strength, local, X, W = s.rule_scores(row['rule'], d, c, styles[row['partition']].to_numpy())
    sim = s.policy.simulate(c, scores, strength, row['config'], detailed=True)
    months, txt = s.explain(row, d, c, styles, scores, local, X, W, sim)
    s.dump('STYLE_MONTHLY_EXPLANATIONS.json', months)
    (s.P / 'STYLE_MONTHLY_EXPLANATIONS.txt').write_text(txt)
    # Every family is pre-existing and descriptive only; no return-based choice.
    feasible = [[] for _ in c['dates']]
    for norm in ['percentile', 'magnitude']:
        d = packs[norm]
        for core in meta['families']:
            X, spec = s.strict.design(d, core)
            for i in range(len(c['dates'])):
                ordered = s.feasible_margin(X[i], d['eligible'][i], d['order'][i], spec)
                sets = s.feasible_margin(X[i], d['eligible'][i], d['order'][i], spec, ordered=False)
                feasible[i].append(dict(norm=norm, core=core, set_status=sets['status'], order_status=ordered['status'],
                    set_margin=sets['margin'], order_margin=ordered['margin'], set_weights=sets.get('weights'), order_weights=ordered.get('weights'), spec=spec))
        print('Post-freeze family feasibility finished', norm, flush=True)
    # Raw numerical evidence for each author holding, exit, and model competitor.
    pp, raw, members, periods, input_sources, pools = s.e.load_prices()
    feats, _, _ = s.e.previous.panels(pp, raw)
    sf = s.O / 'index_union/alias_resolved_inputs/sectors.csv.gz'
    sectors = pd.read_csv(sf, index_col=0, parse_dates=True, low_memory=False)
    sources[str(sf)] = s.sha(sf)
    rv = pp.pct_change(fill_method=None).rolling(126).std() * np.sqrt(252)
    rm = pp.shift(42) / pp.shift(252) - 1
    low = pp.rolling(21).min()
    high = pp.rolling(21).max()
    records = []
    for i, month in enumerate(months):
        date = c['dates'][i]
        syms = [s.e.actual(r['ticker']) for r in month['securities']]
        bm = s.strict.benchmark_map(date, syms, sectors, members['theme17'][1])
        rel = s.strict.relative_row(rm, date, syms, bm)
        proof = []
        for rec in month['securities']:
            symbol = s.e.actual(rec['ticker'])
            z = dict(ticker=rec['ticker'], author_action=rec['author_action'], model_rank=rec['frozen_rank'],
                     sector_etf=bm[symbol], long_momentum_skip42=finite(rm.at[date, symbol]),
                     calendar14_skip2_return=finite(feats['calendar14_skip2_raw'].at[date, symbol]),
                     annual_volatility126=finite(rv.at[date, symbol]), stock_vs_own_sector_skip42=finite(rel[symbol]),
                     cmf63=finite(feats['cmf63'].at[date, symbol]),
                     recent_close_position21=finite((pp.at[date, symbol] - low.at[date, symbol]) / (high.at[date, symbol] - low.at[date, symbol])) if high.at[date, symbol] > low.at[date, symbol] else None,
                     recent_position_use='descriptive evidence only, not a fitted score or entry gate')
            proof.append(z)
        options = feasible[i]
        available = options[0]['set_status'] != 'author_not_eligible_or_not_in_pool'
        sets = [z for z in options if z['set_status'] == 'strictly_feasible']
        orders = [z for z in options if z['order_status'] == 'strictly_feasible']
        rec = dict(signal_date=date, author=month['author'], model=month['model'],
                   raw_stock_evidence=proof, pool_and_eligibility_available=available,
                   feasible_set_families=len(sets), feasible_order_families=len(orders),
                   best_set=max(sets, key=lambda z: z['set_margin']) if sets else None,
                   best_order=max(orders, key=lambda z: z['order_margin']) if orders else None,
                   limitation='Month-specific oracle uses the supplied choices. It is not the frozen model and cannot be reported as validated trading performance.')
        records.append(rec)
    s.dump('ALL_FAMILY_ORACLE.json', [dict(date=d, families=f) for d, f in zip(c['dates'], feasible)])
    s.dump('MONTHLY_RAW_EVIDENCE.json', records)
    summary = {}
    for key, mask in [('training35', np.arange(69) < 35), ('validation34', np.arange(69) >= 35), ('all69', np.ones(69, bool))]:
        rows = [r for i, r in enumerate(records) if mask[i]]
        summary[key] = dict(months=len(rows), set_possible=sum(r['feasible_set_families'] > 0 for r in rows),
            ordered_possible=sum(r['feasible_order_families'] > 0 for r in rows),
            unavailable=[dict(date=r['signal_date'], author=r['author']) for r in rows if not r['pool_and_eligibility_available']],
            available_but_set_impossible=[dict(date=r['signal_date'], author=r['author']) for r in rows if r['pool_and_eligibility_available'] and not r['feasible_set_families']],
            available_but_order_impossible=[dict(date=r['signal_date'], author=r['author']) for r in rows if r['pool_and_eligibility_available'] and not r['feasible_order_families']])
    s.dump('DIAGNOSTIC_SUMMARY.json', summary)
    s.dump('DIAGNOSTIC_SOURCES.json', sources)
    assert s.sha(s.P / 'FROZEN.json') == frozen_hash and s.sha(s.P / 'RESULTS.json') == results_hash
    print('SUMMARY', summary, flush=True)


if __name__ == '__main__':
    main()
