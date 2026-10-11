"""Explain frozen candidates' return differences without choosing new trades."""
from pathlib import Path
import json
import sys
import numpy as np

P = Path(__file__).resolve().parent
sys.path.insert(0, str(P.parent / 'monthly_guidance_reset'))
from study import load_prices, actual


def main():
    read = lambda path: json.loads(path.read_text())
    px, raw, members, periods, sources, pools = load_prices()
    new = read(P / 'RESULTS.json')['balanced']
    old = read(P.parent / 'monthly_decision_audit/MONTHS.json')
    old_stats = read(P.parent / 'monthly_ordered_inverse/RESULTS.json')['previous_candidate']
    rows = []
    for period, n, b in zip(periods, new['months'], old):
        assert period['start'] == n['date']

        def ret(symbols):
            ss = [actual(s) for s in symbols]
            return float((px.loc[period['end'], ss] / px.loc[period['start'], ss]).mean() - 1)

        ar, nr, br = ret(period['holdings']), ret(n['ours']), ret(b['ours'])
        rows.append(dict(start=period['start'], end=period['end'], author=period['holdings'],
                         new=n['ours'], old=b['ours'], author_return=ar, new_return=nr, old_return=br,
                         new_hits=len(set(n['ours']) & set(period['holdings'])),
                         old_hits=len(set(b['ours']) & set(period['holdings'])),
                         log_old_minus_new=float(np.log1p(br) - np.log1p(nr)),
                         log_author_minus_new=float(np.log1p(ar) - np.log1p(nr))))
    for key, metrics in [('new', new['metrics']), ('old', old_stats['metrics'])]:
        wealth = np.prod([1 + row[key + '_return'] for row in rows])
        np.testing.assert_allclose(wealth * 100000., metrics['final_equity'], rtol=1e-11)
    log_gap = sum(r['log_old_minus_new'] for r in rows)
    np.testing.assert_allclose(log_gap, np.log(old_stats['metrics']['final_equity'] / new['metrics']['final_equity']), rtol=1e-11)
    out = dict(
        purpose='Post-hoc attribution of two already-frozen candidates. Not a new signal, objective, or per-month switching rule.',
        new_metrics=new['metrics'], old_metrics=old_stats['metrics'],
        new_matching=new['matching']['all'], old_matching=old_stats['matching']['all'],
        counts=dict(more_hits=sum(r['new_hits'] > r['old_hits'] for r in rows),
                    fewer_hits=sum(r['new_hits'] < r['old_hits'] for r in rows),
                    same_hits=sum(r['new_hits'] == r['old_hits'] for r in rows),
                    new_outperforms=sum(r['new_return'] > r['old_return'] + 1e-10 for r in rows),
                    new_underperforms=sum(r['new_return'] < r['old_return'] - 1e-10 for r in rows)),
        additive_log_wealth_gap=log_gap,
        largest_old_advantage=sorted(rows, key=lambda r: r['log_old_minus_new'], reverse=True)[:7],
        largest_author_misses=sorted(rows, key=lambda r: r['log_author_minus_new'], reverse=True)[:7],
        months=rows,
        validation=dict(monthly_returns_match_daily_replay=True, additive_log_attribution_verified=True,
                        no_reselection=True, fixed_monthly_units=True))
    (P / 'RETURN_ATTRIBUTION.json').write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(counts=out['counts'], validation=out['validation']), ensure_ascii=False))


if __name__ == '__main__':
    main()
