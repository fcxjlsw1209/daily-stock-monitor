"""Test whether the missing first 10 return observations explain metric gaps.

This is a permissive aggregate feasibility bound, not a strategy search. Do not
export any of the enumerated holdings as potential author transactions. All old
monthly choices after Aug31 remain fixed. No score coefficients are fitted.
"""
from itertools import combinations
import json
import numpy as np
import pandas as pd
from audit import O, P, R, I, read, sha, dump

START = pd.Timestamp('2016-08-17')
JOIN = pd.Timestamp('2016-08-31')
END = pd.Timestamp('2026-09-24')
JULY = pd.Timestamp('2016-07-29')


def metrics_from_segments(prefixes, suffix):
    """Each row starts at one; suffix starts at the prefix's last date."""
    a = prefixes[:, 1:] / prefixes[:, :-1] - 1
    b = suffix[1:] / suffix[:-1] - 1
    n = a.shape[1] + len(b)
    sr = a.sum(axis=1) + b.sum()
    sr2 = (a * a).sum(axis=1) + (b * b).sum()
    sd = np.sqrt(np.maximum(0, (sr2 - sr * sr / n) / (n - 1)))
    wealth = prefixes[:, -1] * suffix[-1]
    peak = np.maximum.accumulate(prefixes, axis=1)
    dd = (prefixes / peak - 1).min(axis=1)
    relative_previous_peak = peak[:, -1] / prefixes[:, -1]
    suffix_peak = np.maximum.accumulate(suffix)
    for i in range(0, len(prefixes), 256):
        prior = relative_previous_peak[i:i + 256, None]
        tail = (suffix[None, :] / np.maximum(prior, suffix_peak[None, :]) - 1).min(axis=1)
        dd[i:i + 256] = np.minimum(dd[i:i + 256], tail)
    return dict(wealth=wealth, cagr=wealth ** (252 / n) - 1,
                sharpe=sr / n / sd * np.sqrt(252), max_dd=dd)


def run():
    protected = read(P / 'INPUT_HASHES.json')
    assert all(sha(f) == h for f, h in protected.items())
    original = read(O / 'four_factor_rebound/ETF_SOURCES.json')
    additional = read(R / 'ETF_MANIFEST.json')
    px = pd.read_csv(O / 'index_union/alias_resolved_inputs/adj.csv.gz', index_col=0, parse_dates=True)
    for row in original + additional:
        assert sha(row['file']) == row['sha256']
        protected[row['file']] = row['sha256']
        px[row['symbol']] = pd.read_csv(row['file'], index_col=0, parse_dates=True)['Adj Close'].reindex(px.index)
    member = pd.read_csv(O / 'daily_rebuild_research/price_scaled_membership.csv.gz', index_col=0, parse_dates=True)
    member = member.reindex(index=px.index, columns=px.columns, fill_value=False).fillna(False).astype(bool)
    etfs = {row['symbol'] for row in original + additional} | {'TQQQ', 'QLD'}
    member.loc[:, list(etfs & set(member.columns))] = False
    core = [row['symbol'] for row in original]
    theme = [s for s in core if s not in {'DIA', 'XLB', 'XLC', 'XLRE', 'XLU', 'XLP'}] + ['SMH', 'SOXX', 'IGV', 'VGT', 'MTUM', 'ARKK']
    pools = {'core17': core, 'theme17': theme}
    q = px.loc[START:END, 'QQQ'].dropna()
    r = q.pct_change().dropna()
    control = dict(start=str(q.index[0].date()), end=str(q.index[-1].date()),
        observations=len(r), wealth=float(q.iloc[-1] / q.iloc[0]),
        cagr=float((q.iloc[-1] / q.iloc[0]) ** (252 / len(r)) - 1),
        sharpe=float(r.mean() / r.std(ddof=1) * np.sqrt(252)),
        max_dd=float((q / q.cummax() - 1).min()))
    expected = dict(wealth=6.78, cagr=.2092, sharpe=.958, max_dd=-.3512)
    tolerance = dict(wealth=.005, cagr=.00005, sharpe=.0005, max_dd=.00005)
    control['matches_all_four_printed_qqq_values'] = all(abs(control[k] - expected[k]) <= tolerance[k] for k in expected)
    assert control['matches_all_four_printed_qqq_values']
    control['convention'] = 'Adjusted close, daily simple returns, 252-day annualization, ddof=1, zero risk-free rate. Matches printed QQQ; does not uniquely prove author implementation.'
    target = dict(wealth=387.72, cagr=.8064, sharpe=1.867, max_dd=-.3446)
    results = {}
    for pool, etf in pools.items():
        mask = member.copy()
        mask[etf] = px[etf].notna()
        for convention, signal in [('initialize_equal_at_plot_start', START), ('carry_july_month_end_fixed_shares', JULY)]:
            i = px.index.get_loc(signal)
            # Superset: no positive-momentum gate; permit any two/three or cash.
            # A no-match result therefore also holds for stricter momentum gates
            # inside THIS known proxy universe, but not for an unknown author pool.
            available = mask.loc[signal] & px.iloc[i-252:i+1].notna().all() & px.loc[signal].gt(0)
            names = sorted(px.columns[available])
            frame = px.loc[signal:JOIN, names]
            if not frame.notna().all().all():
                raise ValueError('Incomplete initial-window price paths')
            ratios = (frame / frame.iloc[0]).to_numpy()
            triples = np.array(list(combinations(range(len(names)), 3)))
            pairs = np.array(list(combinations(range(len(names)), 2)))
            paths = np.concatenate([ratios[:, triples].mean(axis=2).T,
                                    ratios[:, pairs].mean(axis=2).T,
                                    np.ones((1, len(frame)))])
            first = frame.index.get_loc(START)
            paths = paths[:, first:] / paths[:, first, None]
            assert paths.shape[1] == len(px.loc[START:JOIN])
            for file in sorted(R.glob(f'INVERSE_{pool}_*_NAV.csv')):
                suffix = pd.read_csv(file, index_col=0, parse_dates=True).iloc[:, 0]
                assert suffix.index[0] == JOIN and suffix.index[-1] == END and suffix.iloc[0] == 1
                m = metrics_from_segments(paths, suffix.to_numpy())
                matches = {k: np.abs(m[k] - target[k]) <= tolerance[k] for k in target}
                ident = file.stem + '__' + convention
                results[ident] = dict(pool=pool, boundary_convention=convention,
                    initial_information_date=str(signal.date()), eligible_securities=len(names),
                    initial_path_count=len(paths), initial_return_observations=paths.shape[1]-1,
                    full_return_observations=paths.shape[1]-1+len(suffix)-1,
                    ranges={k: [float(v.min()), float(v.max())] for k, v in m.items()},
                    within_printed_rounding_count={k: int(v.sum()) for k, v in matches.items()},
                    all_four_match_count=int(np.logical_and.reduce(list(matches.values())).sum()),
                    holdings_exported=False)
    dump('BOUNDARY_AUDIT.json', dict(qqq_control=control, target=target,
        hypothesis='Only initial Aug17-Aug31 holdings vary. Every existing monthly choice from Aug31 onward remains fixed.',
        scope='Known historical index-union Top32 plus the two declared 17-ETF proxy lists; no claim covering unknown author securities.',
        permissive_bound='All eligible pairs/triples and cash, regardless of momentum rank. This is intentionally a superset for exclusion, not an entry rule or an export of potential transactions.',
        results=results, every_case_rejected_by_sharpe=all(v['within_printed_rounding_count']['sharpe'] == 0 for v in results.values()),
        every_case_rejected_by_max_dd=all(v['within_printed_rounding_count']['max_dd'] == 0 for v in results.values())))
    assert all(sha(f) == h for f, h in protected.items())
    dump('BOUNDARY_INPUT_HASHES.json', protected)
    dump('CODE_HASHES.json', {str(f): sha(f) for f in P.glob('*.py')})
    text = ['首段日期和Sharpe口径核对', '',
        f"QQQ：年化{control['cagr']:.5%}，Sharpe {control['sharpe']:.6f}，最大回撤{control['max_dd']:.5%}，倍数{control['wealth']:.6f}。四项按原图位数取整均吻合。",
        '这支持使用日收益、252日、样本标准差、零无风险利率作为复现对照口径，但不能唯一证明作者的内部代码。', '',
        '冻结已有接近80%的三条事后路径，只枚举缺少的最初10个日收益：8月17日等权初始化，或7月末等权持股延续至8月17日。',
        '只输出可行指标范围，不输出未经完整图像/作者思路准入的潜在交易。为避免过早排除，允许代理池所有有足够历史的两只、三只组合及现金；这是排错上界，不是新增策略。', '',
        '路径/起点假设 | 组合数 | 全期Sharpe范围 | 全期最大回撤范围 | 同时匹配四项数']
    for key, v in results.items():
        s = v['ranges']['sharpe']; d = v['ranges']['max_dd']
        text.append(f"{key} | {v['initial_path_count']} | {s[0]:.3f}—{s[1]:.3f} | {d[0]:.2%}—{d[1]:.2%} | {v['all_four_match_count']}")
    text += ['', '若范围仍不包含原图Sharpe 1.867和回撤−34.46%，这些路径的偏差就不能归因于缺失的前两周。需要重建后续月份的完整路径，不得仅改初始金额或年化口径凑数。',
        '结论仅针对当前已保存的三条路径和代理股票池，不证明作者策略不存在，也不证明其他组合无法匹配。']
    (P / 'BOUNDARY_REPORT.txt').write_text('\n'.join(text) + '\n')
    print(json.dumps(dict(qqq_control=control, cases=len(results), all_four_matches=sum(v['all_four_match_count'] for v in results.values())), indent=2))


if __name__ == '__main__':
    run()
