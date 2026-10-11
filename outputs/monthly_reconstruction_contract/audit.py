"""Reclassify saved experiments without changing a price, rule, or ledger.

Image comparison uses ONE uncertain initial scale for suffix diagnostics.
Column envelopes contain all underlying daily line vertices; monthly peaks/NAV
are never reset. A screen pass is NOT sufficient to export candidate trades.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from gate import contract, assess

P = Path(__file__).resolve().parent
O = P.parent
J = O / 'monthly_inverse_joint'
I = O / 'monthly_curve_reverse'
R = O / 'monthly_reconstruction_v2'


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(name, value):
    (P / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def line_envelope(x, y, centers, radius):
    """Exact min/max of piecewise-linear path over each visible x interval."""
    lows, highs = [], []
    for center in centers:
        lo, hi = max(x[0], center - radius), min(x[-1], center + radius)
        if lo > hi:
            raise ValueError('Observation outside computed path')
        vertices = y[(x >= lo) & (x <= hi)]
        vals = np.r_[np.interp([lo, hi], x, y), vertices]
        lows.append(float(vals.min()))
        highs.append(float(vals.max()))
    return np.array(lows), np.array(highs)


def panel_error(x, ylo, yhi, obs, radius, tolerance):
    z = obs[(obs.x >= x[0]) & (obs.x <= x[-1])].copy()
    if not len(z):
        raise ValueError('No visible image observations')
    a, _ = line_envelope(x, ylo, z.x.to_numpy(), radius)
    _, b = line_envelope(x, yhi, z.x.to_numpy(), radius)
    gap = np.maximum(np.maximum(a - z.ymax.to_numpy() - tolerance,
                                z.ymin.to_numpy() - tolerance - b), 0)
    midpoint = np.interp(z.x, x, (ylo + yhi) / 2)
    z['gap_px'] = gap
    result = dict(visible_columns=len(z), outside_fraction=float(np.mean(gap > 1e-9)),
                  outside_gap_rmse_px=float(np.sqrt(np.mean(gap ** 2))),
                  outside_gap_p95_px=float(np.quantile(gap, .95)),
                  center_rmse_px=float(np.sqrt(np.mean((midpoint - z.y) ** 2))),
                  screen_rejected=bool(np.mean(gap > 1e-9) > .10))
    return result, z[['x', 'gap_px']]


def initial_anchor(x0, observations, cal, cfg):
    """A boundary interval from the relative panel, not a fitted return boost."""
    z = observations['relative']
    before = z[z.x <= x0].tail(1)
    after = z[z.x >= x0].head(1)
    if before.empty or after.empty:
        raise ValueError('Missing observations bracketing the partial-window boundary')
    z = pd.concat([before, after]).drop_duplicates('x')
    # log_wealth already adds independently calibrated gold QQQ log wealth.
    lo = z.log_wealth - (z.ymax - z.y + cfg['wealth_relative_extra_y_pixels']) / cal['relative']['log_scale']
    hi = z.log_wealth + (z.y - z.ymin + cfg['wealth_relative_extra_y_pixels']) / cal['relative']['log_scale']
    return float(lo.min()), float(hi.max())


def image_diagnostics(nav, qqq, observations, cal, cfg):
    start, end = pd.to_datetime(cal['dates'])
    axis = cal['wealth']
    x = axis['x0'] + axis['width'] * (nav.index - start).days.to_numpy() / (end - start).days
    if nav.index[0] == start:
        alo, ahi = 0., 0.
    else:
        alo, ahi = initial_anchor(x[0], observations, cal, cfg)
    logs = np.log(nav / nav.iloc[0]).to_numpy()
    qlog = np.log(qqq.reindex(nav.index).to_numpy() / qqq.loc[start])
    if not np.isfinite(qlog).all():
        raise ValueError('Missing QQQ comparison prices')
    result, columns = {}, {}
    for panel in ['wealth', 'relative']:
        sc = cal[panel]
        panel_log = logs - (qlog if panel == 'relative' else 0.)
        ylo = sc['y0'] - sc['log_scale'] * (panel_log + ahi)
        yhi = sc['y0'] - sc['log_scale'] * (panel_log + alo)
        result[panel], columns[panel] = panel_error(x, ylo, yhi, observations[panel],
            cfg['x_pixel_radius'], cfg['wealth_relative_extra_y_pixels'])
    # A suffix drawdown depends on an unknown earlier peak. After its running
    # log peak exceeds every upper-bound early wealth observation, it is exact.
    early = pd.concat([observations[p][observations[p].x <= x[0]] for p in ['wealth', 'relative']])
    upper = [0., ahi]
    for row in early.itertuples():
        upper.append(row.log_wealth + (row.y - row.ymin + cfg['wealth_relative_extra_y_pixels']) / cal[row.panel]['log_scale'])
    peak_upper = max(upper)
    running = np.maximum.accumulate(logs)
    if nav.index[0] == start:
        dd_start = 0
    else:
        surpassed = np.flatnonzero(running + alo > peak_upper)
        if not len(surpassed):
            raise ValueError('Cannot resolve initial drawdown peak for suffix diagnostic')
        dd_start = int(surpassed[0])
    dd = np.exp(logs - running) - 1
    yd = cal['dd']['y0'] - cal['dd']['scale'] * dd
    result['drawdown'], columns['drawdown'] = panel_error(x[dd_start:], yd[dd_start:], yd[dd_start:],
        observations['dd'], cfg['x_pixel_radius'], cfg['drawdown_extra_y_pixels'])
    result['partial_window'] = dict(start=str(nav.index[0].date()),
        initial_nav_interval=[float(np.exp(alo)), float(np.exp(ahi))],
        drawdown_comparable_from=str(nav.index[dd_start].date()),
        no_monthly_anchor_or_peak_reset=True,
        terminal_wealth_interval=[float(nav.iloc[-1] / nav.iloc[0] * np.exp(alo)),
                                 float(nav.iloc[-1] / nav.iloc[0] * np.exp(ahi))])
    months = pd.period_range(nav.index[0].to_period('M'), nav.index[-1].to_period('M'), freq='M')
    monthly = []
    for month in months:
        row = dict(month=str(month))
        for panel, z in columns.items():
            date = start + pd.to_timedelta((z.x - axis['x0']) / axis['width'] * (end - start).days, unit='D')
            zz = z[pd.DatetimeIndex(date).to_period('M') == month]
            row[panel] = None if not len(zz) else dict(visible_columns=len(zz),
                outside_fraction=float(np.mean(zz.gap_px > 1e-9)),
                gap_rmse_px=float(np.sqrt(np.mean(zz.gap_px ** 2))))
        monthly.append(row)
    return result, monthly


def run():
    c = contract()
    source = [J / f for f in ['ALL_RESULTS.json', 'ALL_ORDERS.json', 'ALL_CURVES.csv.gz', 'VALIDATION.json']]
    source += [I / f for f in ['SOURCE.jpg', 'CALIBRATION.json', 'DIGITIZED_WEALTH.csv', 'purple_dd_pixels.csv']]
    inverse_files = sorted(R.glob('INVERSE_*_NAV.csv'))
    source += inverse_files + [f.with_name(f.name.replace('_NAV.csv', '.json')) for f in inverse_files]
    price_file = O / 'index_union/alias_resolved_inputs/adj.csv.gz'
    etf_sources = O / 'four_factor_rebound/ETF_SOURCES.json'
    source += [price_file, etf_sources]
    protection = {str(f): sha(f) for f in source}
    for name in ['PROTECTED_BEFORE.json', 'SOURCE_HASHES.json', 'PREVIOUS_STUDY_HASHES.json']:
        for f, digest in read(J / name).items():
            if sha(f) != digest:
                raise AssertionError(f'Protected source changed: {f}')
            protection[f] = digest
    dump('INPUT_HASHES.json', protection)
    assert sha(I / 'SOURCE.jpg') == c['author_target']['image_sha256']
    cal = read(I / 'CALIBRATION.json')
    z = pd.read_csv(I / 'DIGITIZED_WEALTH.csv')
    obs = {panel: z[z.panel == panel].copy() for panel in ['wealth', 'relative']}
    obs['dd'] = pd.read_csv(I / 'purple_dd_pixels.csv')
    # Read-only loader: do not call old base.load(), which refreshes an old manifest.
    qrow = next(v for v in read(etf_sources) if v['symbol'] == 'QQQ')
    assert sha(qrow['file']) == qrow['sha256']
    qqq = pd.read_csv(qrow['file'], index_col=0, parse_dates=True)['Adj Close']
    result = read(J / 'ALL_RESULTS.json')
    curves = pd.read_csv(J / 'ALL_CURVES.csv.gz', index_col=0, parse_dates=True)
    for f in inverse_files:
        name = 'HINDSIGHT_' + f.name.replace('INVERSE_', '').replace('_NAV.csv', '')
        curves[name] = pd.read_csv(f, index_col=0, parse_dates=True).iloc[:, 0]
        result[name] = read(f.with_name(f.name.replace('_NAV.csv', '.json')))
    boundary = read(P / 'BOUNDARY_AUDIT.json') if (P / 'BOUNDARY_AUDIT.json').exists() else None
    out, monthly = {}, {}
    for key, data in result.items():
        nav = curves[key].dropna()
        diag, month = image_diagnostics(nav, qqq, obs, cal, c['image_audit'])
        hindsight = key.startswith('HINDSIGHT_')
        portfolio = '__portfolio__' in key
        match_clock = hindsight or (data['signal_lag'] == 0 and data['cost_bps_roundtrip'] == 0)
        evidence = {k: 'unresolved' for k in c['admission_requirements']}
        evidence.update(author_scope='failed' if portfolio else 'unresolved',
            execution_and_accounting='passed' if match_clock else 'failed',
            exact_window='passed' if str(nav.index[0].date()) == c['author_target']['start'] else 'unresolved',
            continuous_wealth_and_relative_path='failed' if any(diag[p]['screen_rejected'] for p in ['wealth', 'relative']) else 'unresolved',
            continuous_drawdown_path='failed' if diag['drawdown']['screen_rejected'] else 'unresolved',
            no_monthly_rescaling_or_peak_reset='passed')
        low, high = diag['partial_window']['terminal_wealth_interval']
        if not low <= c['author_target']['terminal_wealth'] <= high:
            evidence['cagr_terminal_wealth_maxdd'] = 'failed'
        boundary_cases = []
        if hindsight and boundary:
            stem = 'INVERSE_' + key.replace('HINDSIGHT_', '') + '_NAV__'
            boundary_cases = [v for k, v in boundary['results'].items() if k.startswith(stem)]
            if len(boundary_cases) == 2:
                if all(v['within_printed_rounding_count']['max_dd'] == 0 for v in boundary_cases):
                    evidence['cagr_terminal_wealth_maxdd'] = 'failed'
                if all(v['within_printed_rounding_count']['sharpe'] == 0 for v in boundary_cases):
                    evidence['sharpe_definition_and_value'] = 'failed'
        if not hindsight:
            evidence.update(uniform_rule='passed', decision_information_timing='unresolved')
        kind = 'global_attribution_hypothesis' if hindsight else 'executable_rule_hypothesis'
        admitted = assess(evidence, kind)
        out[key] = dict(classification='off_main_scope' if portfolio else ('inverse_attribution_only' if hindsight else 'rule_replay_only'),
            old_orders_are_author_transactions=False, comparable_execution=match_clock,
            observed_metrics={k: data[k] for k in ['start', 'end', 'wealth', 'cagr_252', 'sharpe', 'max_dd']},
            metric_differences_diagnostic_only=dict(cagr=data['cagr_252'] - c['author_target']['cagr'],
                sharpe=data['sharpe'] - c['author_target']['sharpe'],
                max_dd=data['max_dd'] - c['author_target']['max_drawdown']),
            image=diag, evidence=evidence, admission=admitted,
            initial_period_feasibility_cases=boundary_cases,
            sharpe_audit_scope='Under the daily 252-day zero-risk-free convention independently matching all four printed QQQ values; not proof of unique author implementation.' if boundary_cases else None)
        monthly[key] = month
    dump('AUDIT.json', out)
    dump('MONTHLY_PATH_ERRORS.json', monthly)
    dump('ADMITTED_POTENTIAL_TRANSACTIONS.json', dict(contract=c['version'],
        admitted_hypotheses=[], transactions=[], verified_author_transactions=False,
        reason='No saved candidate satisfies all full-trajectory and author-scope requirements. Existing orders remain archived research replays.'))
    assert not any(v['admission']['admitted'] for v in out.values())
    for f, digest in protection.items():
        assert sha(f) == digest, f
    summary = dict(audited_paths=len(out), off_main_scope=sum(v['classification'] == 'off_main_scope' for v in out.values()),
        curve_screen_rejected=sum(v['evidence']['continuous_wealth_and_relative_path'] == 'failed' for v in out.values()),
        drawdown_screen_rejected=sum(v['evidence']['continuous_drawdown_path'] == 'failed' for v in out.values()),
        admitted_potential_transaction_paths=0, protected_files_unchanged=len(protection),
        original_accounts_unchanged=True, author_reproduced=False,
        future_work='Every new hypothesis must cite the author concept and use gate.py before any candidate transaction export.')
    dump('SUMMARY.json', summary)
    lines = ['月策略复现：作者思路与完整曲线准入', '2026-10-09', '',
        '用户要求：所有后续探索符合作者中心思想；潜在交易必须共同解释给定收益路径和Sharpe。',
        '执行：只做多/空仓；跳过最近两个月的中期动量为主，结合波动、相对行业、量价及小幅模型修正；月底等权3只（2只各半、不足2只空仓），月内固定股数。',
        '组合相关性优化移出主复现分支，旧实验保留。原六账户、计划、指纹不改。', '',
        f"审计{summary['audited_paths']}条已有完整路径，其中{summary['off_main_scope']}条含未获作者描述支持的组合优化，移出主分支。合格潜在交易路径：0。",
        '旧ORDERS等文件是研究回放，不是已找出的作者交易。局部拟合、年化接近80%、单项Sharpe接近均不构成准入。', '',
        '对照（均从2016-08-31开始；原图从08-17开始，因此指标差仅作诊断）：',
        '版本 | 年化 | Sharpe | 最大回撤 | 净值图不相交列占比 | 相对QQQ图不相交列占比 | 回撤图不相交列占比']
    for key in ['theme17__all_retrospective__stock_only__lag0_cost0', 'HINDSIGHT_core17_q0', 'HINDSIGHT_theme17_q0']:
        a = out[key]; m = a['observed_metrics']; d = a['image']
        lines.append(f"{key} | {m['cagr_252']:.2%} | {m['sharpe']:.3f} | {m['max_dd']:.2%} | {d['wealth']['outside_fraction']:.1%} | {d['relative']['outside_fraction']:.1%} | {d['drawdown']['outside_fraction']:.1%}")
    lines += ['', '图像检查口径：使用独立QQQ标定坐标；保留可见线条上下沿，额外放宽横向±1像素、净值纵向±2像素、回撤±4像素。',
        '这些是有依据但尚未统计标定的宽松排错范围。超过10%可见列仍不相交被筛除；未被筛除也不能自动准入。不会为了接纳某个结果放宽误差。',
        '原图给的是收益、相对收益、回撤路径和整体Sharpe，没有逐日Sharpe曲线。需同时核对全程、分月形状和总体指标，不能由截图恢复不可见的每一个日收益。',
        '现有路径缺2016-08-17至08-31首段；仅给首段末净值一个来自相对QQQ图的统一误差区间，绝不每个月重置净值或回撤峰值。首段未恢复前不能声称全时段指标吻合。',
        '接近80%的三条路径使用未来价格作反向归因，尚无统一可事前执行的选股规则；即使像素吻合，也必须进一步匹配Sharpe和回撤，不能视为已复现。',
        '作者Sharpe的频率、无风险利率和年化/标准差约定尚未独立确定。当前按日收益、252日年化、样本标准差、零无风险利率计算，只作明确口径的对照。',
        '原作者17只ETF、全市场Top32历史口径也未确认。允许有标注的代理池研究，不将其宣称为作者原池。', '',
        '首段补充核对见BOUNDARY_REPORT.txt：QQQ四项指标按原图精度全部吻合。固定后三条近80%事后路径后，枚举代理池初期两只/三只/现金的两种持仓初始化；没有一条能用最初10个日收益消除Sharpe和最大回撤差异。',
        '后续顺序：以整个持仓路径为单位，同时约束净值、Sharpe和回撤，再检查同一套中期动量排序规则能否选出这些组合；不能继续把偏差解释成仅缺开头两周。',
        '不得直接把单月最佳组合连起来，当成已恢复的真实交易。图像一致的假设仍不等于作者真实交易，也不等于独立样本外有效。', '',
        f"验证：{len(protection)}个保护/输入文件校验未变；所有输出只写入本目录。机器规则CONTRACT.json；逐路径AUDIT.json；逐月MONTHLY_PATH_ERRORS.json；准入出口gate.py。"]
    (P / 'REPORT.txt').write_text('\n'.join(lines) + '\n')
    dump('CODE_HASHES.json', {str(f): sha(f) for f in P.glob('*.py')})
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    run()
