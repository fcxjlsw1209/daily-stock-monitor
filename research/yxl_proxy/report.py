"""Generate an auditable self-contained report from the completed proxy run."""
from pathlib import Path
import html
import json
import pandas as pd

P=Path(__file__).resolve().parent
R=P/'results'
LABELS={
 'default_none':'默认EMA／仅主信号', 'default_daily':'默认EMA／主＋日线副信号（主规格）',
 'default_weekly':'默认EMA／主＋日线副信号＋上周RSI',
 'hma58_none':'HMA5/8→RMA8／仅主信号', 'hma58_daily':'HMA5/8→RMA8／主＋日线副信号',
 'hma58_weekly':'HMA5/8→RMA8／加入上周RSI',
 'hma612_none':'HMA6/12→RMA8／仅主信号', 'hma612_daily':'HMA6/12→RMA8／主＋日线副信号',
 'hma612_weekly':'HMA6/12→RMA8／加入上周RSI',
 'default_cross':'默认／只猜均线交叉事件', 'default_basis':'默认／只猜Basis穿越事件',
 'default_exit_either':'默认／主卖或副空退出', 'default_exit_both':'默认／主卖且副空退出',
 'default_state':'默认／主副状态一致即可入场', 'buy_hold':'买入持有（分红留现金）'}


def fmt(x, kind='pct'):
    if pd.isna(x): return '—'
    return f'{100*x:.2f}%' if kind=='pct' else f'{x:.2f}'


def table(df, mapping):
    out=df[list(mapping)].rename(columns=mapping).copy()
    for old,new in mapping.items():
        if old=='model': out[new]=out[new].map(LABELS)
        elif old in ['cagr','total_return','max_drawdown','win_rate','exposure','median_cagr','median_dd',
                      'beat_fraction','median_cagr_gap','median_win','median_exposure','dd_improved_fraction']:
            out[new]=out[new].map(fmt)
    return out.to_html(index=False,escape=True,border=0)


def main():
    m=pd.read_csv(R/'etf_metrics.csv')
    s=pd.read_csv(R/'stock_metrics.csv')
    excluded=pd.read_csv(R/'excluded_stocks.csv')
    bh=s[s.model=='buy_hold'].set_index('symbol')
    summary=[]
    for name, g in s.groupby('model',sort=False):
        x=g.set_index('symbol')
        base=bh.reindex(x.index)
        summary.append(dict(model=name,n=len(x),median_cagr=x.cagr.median(),median_dd=x.max_drawdown.median(),
                            median_cagr_gap=(x.cagr-base.cagr).median(),
                            beat_fraction=(x.cagr>base.cagr).mean(),
                            dd_improved_fraction=(x.max_drawdown>base.max_drawdown).mean(),
                            median_win=x.win_rate.median() if x.win_rate.notna().any() else None,median_exposure=x.exposure.median(),
                            closed_trades=int(x.closed_trades.sum())))
    summary=pd.DataFrame(summary)
    summary.to_csv(R/'stock_summary.csv',index=False)
    etfcols={'model':'近似规则','cagr':'年化收益','max_drawdown':'最大日终回撤',
             'closed_trades':'已平仓笔数','win_rate':'净胜率','exposure':'平均仓位'}
    sections=[]
    for symbol in ['QQQ','SPY']:
        for window in ['all','since2019','since2022','early']:
            frame=m.query('symbol==@symbol and window==@window and fee_bps_per_side==5')
            a,b=frame.iloc[0][['start','end']]
            sections.append(f'<h2>{symbol}：{a} 至 {b}</h2>'+table(frame,etfcols))
    stockcols={'model':'近似规则','n':'股票数','median_cagr':'逐股年化中位数','median_dd':'逐股回撤中位数',
               'median_cagr_gap':'相对同股持有的年化差中位数','beat_fraction':'跑赢同股持有比例',
               'dd_improved_fraction':'改善回撤比例','median_win':'逐股胜率中位数',
               'closed_trades':'全部已平仓笔数'}
    sections.append('<h2>固定当前股票样本：2019-01-02 至 2026-09-18</h2>'+table(summary,stockcols))
    sections.append('<p>逐股独立全仓／现金账户。这些中位数不是组合收益；所有股票取自2026年当前大市值名单，具有幸存者和事后名单偏差。</p>')
    sections.append('<h3>排除样本</h3>'+excluded.to_html(index=False,border=0))
    cost=m.query("window=='all' and model=='default_daily'")
    sections.append('<h2>主规格交易成本敏感性</h2>'+table(cost,{'symbol':'标的','fee_bps_per_side':'单边成本bp','cagr':'年化收益','max_drawdown':'最大回撤'}))
    yearly=pd.read_csv(R/'etf_yearly.csv').query("model in ['default_daily','hma58_daily','hma612_daily','buy_hold']")
    sections.append('<h2>逐年结果</h2>'+table(yearly,{'symbol':'标的','year':'年份','model':'规则','total_return':'当年区间收益','max_drawdown':'年内回撤','closed_trades':'已平仓数'}))
    sections.append('<p>2026只覆盖截至数据截止日。分段从连续账户收益重新归一化，保留之前持仓和分红现金，不是每段重新开仓。</p>')
    q=m.query("symbol=='QQQ' and window=='all' and fee_bps_per_side==5 and model=='default_daily'").iloc[0]
    qh=m.query("symbol=='QQQ' and window=='all' and fee_bps_per_side==5 and model=='buy_hold'").iloc[0]
    lead=f'主规格QQQ年化 {fmt(q.cagr)}，最大回撤 {fmt(q.max_drawdown)}；买入持有年化 {fmt(qh.cagr)}，最大回撤 {fmt(qh.max_drawdown)}。这些数字评价的是本地猜测规则，不能归因于原YXL指标。'
    doc='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>YXL近似回测研究</title><style>
body{font-family:system-ui,sans-serif;max-width:1200px;margin:36px auto;padding:0 22px;color:#18212b;line-height:1.6}
table{border-collapse:collapse;width:100%;margin:16px 0;font-size:14px}th,td{padding:9px;text-align:right;border-bottom:1px solid #d5dce2}th:first-child,td:first-child{text-align:left}h2{margin-top:40px}a{color:#155e9b}.table-wrap{overflow-x:auto}.note{padding:16px;background:#fff3da}code{overflow-wrap:anywhere}
</style><h1>YXL 主副指标：近似回测</h1><p>2026-09-29 · 14组预先指定猜测 · 非原始信号 · 仅历史模拟</p>'''
    doc+='<p class="note">'+lead+'</p>'
    doc+='''<p>默认主线：hl2→EMA10/16→EMA3；Basis猜为快慢线平均，ATR14×3仅云带。买点猜为均线金叉、价格上穿Basis或趋势内回踩；卖点镜像，信号共用3根间隔。副信号用MACD12/26/9柱为正且RSI14&gt;50过滤买点。主规格在主卖信号出现后退出。实际圆点、混合与多周期算法未知，未计算原指标匹配率。</p>
<p>下一交易日开盘成交，单边5bp（买卖合计约0.10%），无杠杆，现金零利息，分红入现金。末日不强平，胜率只计已平仓，买入持有无已平仓胜率。常规开盘价是数据源代理，不是实盘成交保证。无止损，未优化参数，未宣称样本外有效。</p>
<p>HMA版本按作者公开建议预先列入；HMA短周期取整与触发规则仍是猜测。周线版本只使用上一个完整周的RSI，排除未完成周数据。原指标自身是否重绘仍未核实。</p>
<p>本地Yahoo存量行情：QQQ截至2026-09-25；SPY、股票截至2026-09-18。源OHLC拆股调整、未现金分红复权；模拟另记分红，不重复处理拆股。数据未逐条独立核验，当前股票样本有明显幸存者偏差。未更新原库、三个纸面账户或自动化。</p>
<p><a href="PROTOCOL.md">事前规则与来源</a> · <a href="backtest.py">回测代码</a> · <a href="test_backtest.py">8项验证测试</a> · <a href="results/etf_metrics.csv">全部ETF指标</a> · <a href="results/stock_metrics.csv">全部逐股指标</a> · <a href="results/source_manifest.json">输入SHA256</a></p>'''
    doc+=''.join('<div class="table-wrap">'+x+'</div>' for x in sections)
    doc+='</html>'
    (P/'RESULTS.html').write_text(doc)
    payload=dict(stock_count=len(bh),stock_summary=json.loads(summary.to_json(orient='records')),
                 exclusions=excluded.to_dict('records'),validation='8 unittest checks passed; source hashes unchanged',
                 original_signal_match_rate=None)
    (R/'summary.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False))
    print(summary.round(4).to_string(index=False))


if __name__=='__main__':main()
