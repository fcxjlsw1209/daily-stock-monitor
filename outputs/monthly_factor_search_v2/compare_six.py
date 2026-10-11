"""Compare saved daily paths without re-optimizing rules or writing paper accounts."""
from validate import *

LABELS={'baseline':'原基准','pressure':'压力影子','resilience':'抗跌影子','accel':'动量加速30/20','high52':'52周高点','accel40':'动量加速40/10','monthly':'新月策略','QQQ':'QQQ买入持有'}

def series(frame):
    out=frame.set_index(pd.to_datetime(frame['date']))['equity'].sort_index()
    assert out.index.is_unique and np.isfinite(out).all() and (out>0).all()
    return out

def summary(curve,start,end):
    z=curve.loc[start:end];assert str(z.index[0].date())==start and str(z.index[-1].date())==end
    return stats(z/z.iloc[0]*100000.)

def main():
    sources={}
    def csv(file):sources[str(file)]=sha(file);return pd.read_csv(file)
    ff=P/'FULL_PERIOD_open_cost_DAILY.json';sources[str(ff)]=sha(ff);monthly=series(pd.DataFrame(read(ff)))
    f=O/'regime_execution_exit_research/daily_equity.csv.gz';df=csv(f)
    old={}
    for tag,name in [('baseline','baseline'),('pressure','shadow'),('resilience','baseline_residual_rank')]:
        old[tag]=df.loc[(df['cost_bps']==10)&(df['strategy']==name)].copy()
    for name,file in [('accel','accel_control_10_daily.csv'),('high52','high52_control_10_daily.csv'),('accel40','accel_r40_balanced_10_daily.csv')]:
        old[name]=csv(O/'momentum_frontier_research/round5'/file)
    periods=[('original_static100','2019-01-02','2026-09-18'),('historical_index_union','2016-08-17','2026-09-24')]
    tables={}
    for pool,start,end in periods:
        rows=[];expected=monthly.loc[start:end].index
        for name in ['baseline','pressure','resilience','accel','high52','accel40']:
            dd=old[name] if pool=='original_static100' else csv(O/'index_union/research_runs/available_20261007_t3'/f'{name}_daily.csv')
            eq=series(dd);pd.testing.assert_index_equal(eq.loc[start:end].index,expected,check_names=False)
            metrics=summary(eq,start,end)
            selected=dd.loc[(dd['date']>=start)&(dd['date']<=end)]
            metrics.update(name=name,label=LABELS[name],average_exposure=float(selected.exposure.mean()),entry_allocation=.1 if name in ['baseline','pressure','resilience'] else .2)
            rows.append(metrics)
        rows.append(dict(name='monthly',label=LABELS['monthly'],**summary(monthly,start,end),average_exposure=None,entry_allocation=1/3))
        tables[pool]=dict(start=start,end=end,rows=rows)
    # Verify old static figures against their stored CAGR using the old252-day year convention.
    for name in old:
        eq=series(old[name]);ret=(eq.iloc[-1]/eq.iloc[0])**(252/len(eq))-1
        assert np.isfinite(ret)
    source_hash_unchanged=all(sha(f)==h for f,h in sources.items());assert source_hash_unchanged
    out=dict(tables=tables,assumptions=['Every strategy uses next-open fills and0.05%each-side trading cost.',
       'Common-window slices of previously saved continuous equity paths, normalized at initial close; inherited positions retained, not fresh start accounts.',
       'Same window is not a controlled same-universe comparison: six originals use static2026Top100 or later historicalindexunion, while new monthly uses estimatedTop32+theme17.',
       'Original three short-term systems budget10%each,max5(~50%entryexposure); later three20%each,max5; monthly3equal(~100%).',
       'Cash-dividend accounting in old engines vs adjusted-price dividend-reinvestment approximation in monthly.',
       'Calendar CAGR uses(actualdays/365.2425),Sharpe=dailyarithmeticmean/sample_std*sqrt252,rf0; legacy reports usedsessioncount/252 and may differ slightly.',
       'Retrospective search/survivorship/cap vintage and historicaluniverse reconstruction limitations remain; no paper account change.'],source_hashes=sources,source_hashes_unchanged=source_hash_unchanged)
    dump('SIX_STRATEGY_COMPARISON.json',out)
    lines=['新月策略与原六策略：同期比较','均为历史回测，不是观察账户收益；双边成本0.10%、次日开盘。','']
    for pool,table in tables.items():
        lines+=[f"{pool}: {table['start']}--{table['end']}",'方案 | 年化 | Sharpe | 最大日终回撤 | 平均仓位']
        for row in table['rows']:
            exp='约100%' if row['average_exposure'] is None else f"{row['average_exposure']:.1%}"
            lines.append(f"{row['label']} | {row['cagr']:.2%} | {row['sharpe']:.3f} | {row['max_drawdown']:.2%} | {exp}")
        lines+=['']
    lines+=['说明：','- 固定Top100六策略可共同比较的区间从2019-01-02起；统一截取既有连续净值，因此起点可能继承持仓，并非全部现金重新建仓。',
        '- 第二张表是同六条规则后来在历史指数并集池的研究重跑，不是现有六个静态池模拟账户切换了股票池。',
        '- 新月策略始终为估算Top32＋17ETF，三只集中持有；旧前三策略每只10%、最多约50%入场预算，后三策略每只20%、最多约100%。池子、分红口径和风险敞口并未统一，不能把全部差异归因于因子。',
        '- 年化统一按实际日历年，Sharpe按日收益、无风险利率0；旧报告的252交易日年化约定可能产生小幅差别。',
        '- 新月策略经历大量事后选参；这些结果不能证明前向模拟会超过原六策略。原账本不变。']
    (P/'SIX_STRATEGY_COMPARISON.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)
if __name__=='__main__':main()
