"""Continuous-period return diagnostics for already-selected reconstruction rules.
Never use these metrics to retune the selection search; all artifacts stay local.
"""
from datetime import datetime,timezone
from pathlib import Path
import json, sys, hashlib
import numpy as np
import pandas as pd
import research as r
import validate as engine


def make_orders(rows):
    return [dict(date=z['date'],end=z['end'],signal_date=z['date'],symbols=[r.actual(s) for s in z['selected']],scores=[]) for z in rows]


def independently_compound(px,orders,start,end):
    nav=100000.;pieces=[]
    for i,o in enumerate(orders):
        local=px.loc[o['date']:o['end'],o['symbols']]
        assert local.index[0]==pd.Timestamp(o['date']) and local.index[-1]==pd.Timestamp(o['end'])
        assert np.isfinite(local).all().all() and local.gt(0).all().all()
        curve=local.div(local.iloc[0]).mean(axis=1)*nav if len(o['symbols']) else pd.Series(nav,index=local.index)
        pieces.append(curve if i==0 else curve.iloc[1:]);nav=float(curve.iloc[-1])
    result=pd.concat(pieces);assert result.index.equals(px.loc[start:end].index)
    return result


def main():
    chosen_hash=r.sha(r.P/'LEADERS.json');px,raw,members,periods,sources=r.load_inputs();opening=engine.load_open(px,raw)
    start=pd.Timestamp(periods[0]['start']);end=pd.Timestamp(periods[-1]['end']);assert all(a['end']==b['start'] for a,b in zip(periods,periods[1:]))
    leaders=r.read(r.P/'LEADERS.json');base=r.read(r.P/'ABLATIONS.json')[0]
    author=[dict(date=x['start'],end=x['end'],selected=x['holdings']) for x in periods]
    order_sets=dict(author_supplied_holdings=make_orders(author),previous_compact_replica=make_orders(base['details']),current_selection_replica=make_orders(leaders['A']['details']))
    # The previously registered return-first model is a reference, never a newly tuned choice.
    old=r.read(r.O/'monthly_factor_search_v2/return_leader_open_cost_HOLDINGS.json')
    order_sets['previous_return_first_candidate']=[dict(date=z['signal_date'],end=periods[i]['end'],signal_date=z['signal_date'],symbols=[r.actual(s) for s in z['symbols']],scores=[]) for i,z in enumerate(old)]
    assert len(old)==len(periods) and [z['date'] for z in order_sets['previous_return_first_candidate']]==[z['start'] for z in periods]
    summary={};equities={};checks={};rounding=[]
    for name,orders in order_sets.items():
        summary[name]={}
        for mode,cost in [('close',0.),('open',.0005)]:
            key='same_close_zero_cost' if mode=='close' else 'next_open_5bp_per_side'
            stats,eq,trades,holds=engine.replay(px,opening,orders,cost,mode,start=start,end=end)
            if mode=='close':
                independent=independently_compound(px,orders,start,end)
                np.testing.assert_allclose(eq,independent,rtol=1e-11,atol=1e-6)
                checks[name+'_independent_daily_replay']=True
            summary[name][key]=stats;equities[name+'__'+key]=eq
            r.dump('PERFORMANCE_'+name+'__'+key+'_TRADES.json',trades)
        if name=='author_supplied_holdings':
            for z,p in zip(orders,periods):
                gain=float((px.loc[z['end'],z['symbols']]/px.loc[z['date'],z['symbols']]).mean()-1)
                err=(gain-p['reported_return'])*10000
                rounding.append(dict(start=z['date'],replay_return=gain,reported_return=p['reported_return'],error_bp=err,within_reported_rounding=abs(err)<=.5+1e-8))
    aeq=equities['author_supplied_holdings__same_close_zero_cost'];oldnav=r.read(r.O/'monthly_supplied_ledger_audit/DAILY_REPLAY.json')
    oldeq=pd.Series({pd.Timestamp(z['date']):z['nav'] for z in oldnav});np.testing.assert_allclose(aeq,oldeq,rtol=1e-10,atol=1e-6)
    checks['author_matches_prior_verified_daily_replay']=True;checks['author_reported_months_rounding_matches']=sum(x['within_reported_rounding'] for x in rounding)
    assert checks['author_reported_months_rounding_matches']==69
    qqq=px.loc[start:end,'QQQ']/px.at[start,'QQQ']*100000.;equities['QQQ_buy_hold__same_close_zero_cost']=qqq
    summary['QQQ_buy_hold']=dict(same_close_zero_cost=engine.stats(qqq))
    checks['model_selection_unchanged']=r.sha(r.P/'LEADERS.json')==chosen_hash;assert checks['model_selection_unchanged']
    checks['input_hashes_unchanged']=all(r.sha(f)==h for f,h in sources.items());assert checks['input_hashes_unchanged']
    checks['observer_hashes_unchanged']=all(r.sha(f)==h for f,h in r.read(r.P/'PROTECTED_BEFORE.json').items());assert checks['observer_hashes_unchanged']
    data=dict(generated_utc=datetime.now(timezone.utc).isoformat(),start=str(start.date()),end=str(end.date()),period_count=len(periods),daily_equity_points=len(aeq),selection_sha256=chosen_hash,
      annualization='CAGR: calendar days/365.2425; Sharpe: arithmetic daily mean / sample daily std * sqrt252, risk-free0; drawdown from daily closing NAV.',
      primary='Identical month-end close, zero-cost, equal-weight at rebalance / fixed adjusted units within month. Matches source descriptive replay; not evidence of executable close fills.',
      execution_control='Unchanged signals, next trading-day open, 0.05% each buy/sell notional on net rebalance deltas. Hypothetical delayed replay also shown for author known holdings.',
      classification='Historical reconstruction diagnostics. Author holdings known retrospectively; candidate combinations selected on this history. Not live or independent OOS.',
      coverage='2020-12-31--2026-09-24 supplied ledger only; last holding month partial. No statistics stitched from A54/B39. No inference about full 2016-start source curve.',
      future_cadence='Recompute after each audited reconstruction experiment. Preserve matching as selection objective; report performance independently, with timestamp and candidate hash. No new timed automation.',
      runs=summary,checks=checks,author_monthly_checks=rounding,selection_matches=leaders['A']['metrics'])
    r.dump('PERFORMANCE.json',data)
    r.dump('PERFORMANCE_DAILY.json',[dict(date=str(d.date()),**{k:float(eq.at[d]) for k,eq in equities.items()}) for d in aeq.index])
    sourcefiles={**sources,str(r.P/'LEADERS.json'):chosen_hash,str(r.O/'monthly_factor_search_v2/validate.py'):r.sha(r.O/'monthly_factor_search_v2/validate.py'),str(r.O/'monthly_supplied_ledger_audit/DAILY_REPLAY.json'):r.sha(r.O/'monthly_supplied_ledger_audit/DAILY_REPLAY.json')}
    r.dump('PERFORMANCE_SOURCES.json',sourcefiles)
    fingerprint=hashlib.sha256(json.dumps(dict(selection_sha256=chosen_hash,sources=sourcefiles,replay_code_sha256=r.sha(Path(__file__)),start=data['start'],end=data['end']),sort_keys=True).encode()).hexdigest()
    data['replay_fingerprint']=fingerprint;r.dump('PERFORMANCE.json',data)
    history=r.read(r.P/'PERFORMANCE_HISTORY.json') if (r.P/'PERFORMANCE_HISTORY.json').exists() else []
    history=[x for x in history if 'replay_fingerprint' in x]
    if not any(x['replay_fingerprint']==fingerprint for x in history):history.append({k:data[k] for k in ['generated_utc','start','end','selection_sha256','replay_fingerprint','runs','selection_matches']})
    r.dump('PERFORMANCE_HISTORY.json',history)
    labels={'author_supplied_holdings':'作者提供持仓的回放','previous_compact_replica':'上轮简洁复现规则','current_selection_replica':'本轮选股复现候选','previous_return_first_candidate':'此前收益优先候选（已在观察列表）','QQQ_buy_hold':'QQQ买入持有'}
    lines=['同区间收益与Sharpe定期核对','',f'连续时间段：{start.date()}收盘至{end.date()}，69期（末期未满月）。初始资金10万美元。','主比较统一：月底收盘调仓、零交易成本、三只等权、月内固定复权份额。此处是描述性回放；不能据此认定收盘后得到的信号可按同一收盘价成交。','', '|方案|年化收益|每日Sharpe|最大日终回撤|累计收益|','|---|---:|---:|---:|---:|']
    for name in summary:
        z=summary[name]['same_close_zero_cost'];lines.append(f"|{labels[name]}|{z['cagr']:.2%}|{z['sharpe']:.3f}|{z['max_drawdown']:.2%}|{z['total_return']:.2%}|")
    lines+=['','执行敏感性：相同选股延至下一交易日开盘，买/卖各0.05%成本，按差额调仓收费。作者这一行也是给定持仓的假设性延迟回放，不能称其真实策略执行结果。','', '|方案|年化收益|每日Sharpe|最大日终回撤|','|---|---:|---:|---:|']
    for name in order_sets:
        z=summary[name]['next_open_5bp_per_side'];lines.append(f"|{labels[name]}|{z['cagr']:.2%}|{z['sharpe']:.3f}|{z['max_drawdown']:.2%}|")
    lines+=['','- 作者69期月收益全部符合原文舍入精度；日净值与之前已核验回放一致。Sharpe由逐日持仓回放计算，不是从69个月收益直接推算。','- 年化按实际日历天数；Sharpe为每日算术收益均值÷样本标准差×√252，无风险利率0；没有从不连续的54/39期拼出年化。','- 本轮候选先按持仓匹配选定，再计算本表；本表没有用于重新选择参数。收益相近也不等于交易规则复现。','- 作者原图从2016年开始，本表从2020-12-31开始，不能混用两个区间的年化与Sharpe。','- 候选和作者规则均在这些历史上研究过，表内均非独立样本外成绩。','- 以后每轮完成audit_results.py，自动重算这份对照并保留PERFORMANCE_HISTORY.json；没有新增定时任务或观察账户。','- 所有逐日净值见PERFORMANCE_DAILY.json；各执行口径交易记录单独保存；旧账本和受保护文件未修改。']
    (r.P/'PERFORMANCE.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps({n:{k:{q:z[q] for q in ['cagr','sharpe','max_drawdown']} for k,z in modes.items()} for n,modes in summary.items()},indent=2))

if __name__=='__main__':main()
