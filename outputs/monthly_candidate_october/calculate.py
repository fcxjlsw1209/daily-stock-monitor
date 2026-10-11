"""Read-only retrospective September 30 selection of the frozen 66.10% candidate.

This is deliberately a single-date audit, not an observer or a generalized live
implementation. Fail if either added gate is active; do not silently substitute
the baseline for the research candidate in another market state.
"""
from pathlib import Path
from datetime import datetime, timezone
import importlib.util
import json
import sys

import numpy as np
import pandas as pd

P = Path(__file__).resolve().parent
O = P.parent
DAY = pd.Timestamp('2026-09-30')
sys.dont_write_bytecode = True
sys.path.insert(0, str(O / 'monthly_state_rank_research'))
import state_rank as research

sys.path.insert(0, str(O / 'monthly_stability_monitor'))
spec = importlib.util.spec_from_file_location('october_frozen_stability', O / 'monthly_stability_monitor/strategy.py')
strategy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(strategy)
assert Path(sys.modules['base_strategy'].__file__).parent == O / 'monthly_stability_monitor'


def read(path):
    return json.loads(path.read_text())


def internal(symbol):
    return {'META': 'research_equity_FB', 'BRK-B': 'BRK.B'}.get(symbol, symbol)


def protected():
    folders = ['daily_stock_monitor', 'pressure_shadow_monitor', 'resilience_shadow_monitor',
               'accel_shadow_monitor', 'high52_shadow_monitor', 'accel40_shadow_monitor',
               'monthly_buffer5_monitor', 'monthly_regime_monitor', 'monthly_industry_monitor',
               'monthly_stability_monitor', 'monthly_state_rank_research', 'monthly_signal_20260930']
    return {str(f): research.sha(f) for folder in folders for f in (O / folder).rglob('*')
            if f.is_file() and f.suffix != '.pyc' and f.name != 'run.lock'}


def evaluate(panel, etfs, stocks):
    symbols = sorted(set(stocks) | set(etfs))
    states = research.state_row(panel['adj'], DAY, symbols, set(etfs))
    spy_vol = float(panel['adj'].SPY.pct_change(fill_method=None).rolling(63).std().at[DAY] * np.sqrt(252))
    gates = {'calm10': spy_vol < .10, 'weak_narrow': states['weak_narrow']}
    assert all(not gates[z['state']] for z in FROZEN_SPEC), 'An active correction requires full candidate scoring'
    baseline = strategy.selection(panel, DAY, etfs)
    assert baseline['calm'] == gates['calm10']
    # Both budgets are exactly zero in research.score; its additive corrections
    # therefore leave the frozen 65.34 baseline ranking exactly unchanged.
    return {'selection': baseline, 'gates': gates, 'spy_63_session_annual_volatility': spy_vol,
            'states': states, 'added_correction_budget': 0.0}


FROZEN_SPEC = read(O / 'monthly_state_rank_research/FROZEN.json')['selected']['primary']['spec']
assert FROZEN_SPEC == [
    {'family': 'residual', 'key': 'resid_126_210_sum', 'amount': .025, 'state': 'calm10'},
    {'key': 'ml_4', 'state': 'weak_narrow', 'amount': .05}]


def main():
    before = protected()
    etfs = read(O / 'monthly_stability_monitor/config.json')['etfs']
    panel, info = research.panel_load(etfs)
    sources = dict(info['source_hashes'])
    pool_path = O / 'monthly_signal_20260930/POOL.json'
    pool = read(pool_path)
    assert pool['date'] == str(DAY.date())
    stocks = sorted(internal(z['symbol']) for z in pool['top32'])
    assert len(set(stocks)) == 32
    sources[str(pool_path)] = research.sha(pool_path)
    for f in (O / 'monthly_signal_20260930/prices').glob('*.csv'):
        meta = read(f.with_suffix('.source.json'))
        assert research.sha(f) == meta['sha256']
        prices = pd.read_csv(f, index_col=0, parse_dates=True)
        assert prices.index.is_unique and prices.index.is_monotonic_increasing
        for key, col in [('adj', 'Adj Close'), ('open', 'Open'), ('close', 'Close'),
                         ('high', 'High'), ('low', 'Low'), ('volume', 'Volume')]:
            panel[key][internal(meta['symbol'])] = prices[col].reindex(panel['adj'].index)
        sources[str(f)] = research.sha(f)
        sources[str(f.with_suffix('.source.json'))] = research.sha(f.with_suffix('.source.json'))
    symbols = sorted(set(stocks) | set(etfs))
    for key in ['adj', 'close', 'high', 'low', 'volume']:
        assert panel[key].loc[DAY, symbols].notna().all(), key
    member = pd.DataFrame(False, index=[DAY], columns=panel['adj'].columns)
    member.loc[DAY, stocks] = True
    panel['member'] = member
    result = evaluate(panel, etfs, stocks)
    prefix = {key: frame.loc[:DAY].copy() for key, frame in panel.items()}
    assert result == evaluate(prefix, etfs, stocks), 'Future-row truncation changed the selection'
    rows = []
    for s in result['selection']['symbols']:
        assert prefix['adj'][s].tail(295).notna().all()
        assert all(prefix[k][s].tail(126).notna().all() for k in ['close', 'high', 'low', 'volume'])
        rows.append({'ticker': s, 'score': result['selection']['scores'][s],
                     'weight': result['selection']['target_weights'][s],
                     'signal_close_raw': float(panel['close'].at[DAY, s]),
                     'next_session_open_raw': float(panel['open'].at[pd.Timestamp('2026-10-01'), s])})
    after = protected()
    assert before == after, 'Protected observer/research files changed'
    output = dict(created_utc=datetime.now(timezone.utc).isoformat(), signal_date=str(DAY.date()),
                  hypothetical_holding_month='2026-10', candidate='monthly_state_rank_research primary id105',
                  spec=FROZEN_SPEC, selected=rows, calculation=result,
                  research_candidate_registered=False, actual_paper_trades=False,
                  execution='Retrospective signal only. Close/open quotes are references, not ledger fills.',
                  limitations=[pool['limit'], 'Same-close signal and fill is an idealized research convention.',
                               'Truncation proves no future price rows are used; it does not certify source publication vintages.'],
                  validation=dict(future_price_rows_removed_same_selection=True,
                                  protected_files_unchanged=len(before),
                                  selected_history_complete=True), source_hashes=sources)
    (P / 'RESULTS.json').write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    report = '\n'.join([
        '66.10% 月策略研究候选：2026年10月回溯选股',
        '信号日期：2026-09-30。候选尚未注册观察账户，本次未生成计划或交易。',
        '该日期低波动条件和弱市条件都不成立，新增2.5%残差、5%机器学习修正均为零。',
        '因此与已登记65.34%版本的当期选股完全一致。',
        '股票 | 评分 | 目标权重',
        *[f"{z['ticker']} | {z['score']:.8f} | {z['weight']:.6%}" for z in rows],
        f"SPY63交易日年化波动率：{result['spy_63_session_annual_volatility']:.4%}；低于10%才启用低波动修正。",
        f"股票高于200日均线比例：{result['states']['breadth']:.2%}；SPY高于200日均线：{result['states']['trend']}。",
        '股票池：已缓存9月30日供应商历史市值Top32＋既有17ETF；历史版本及上市/边界证据仍有研究限制。',
        f'验证：移除9月30日之后全部价格行，结果相同；{len(before)}个受保护文件哈希未变。',
        '现有65.34%观察账户首个信号日期10月30日，计划11月2日开盘首次执行；没有10月已成交仓位。',
        '后续优先研究：日级独立回撤事件的5/10/21日收益与不利路径；无可靠增量证据时严格零修正；再研究可验证时间戳的财报/事件背景。',
        '这些方向尚未验证，不承诺提高年化。',
    ])
    (P / 'REPORT.txt').write_text(report + '\n')
    print(json.dumps(dict(selected=rows, gates=result['gates'], validation=output['validation']), indent=2))


if __name__ == '__main__':
    main()
